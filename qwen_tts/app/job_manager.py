from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional

from .metadata import MetadataStore
from .model_manager import ModelManager
from .runtime import JobLockRecord, RuntimeBaseline


JobStatus = Literal["pending", "running", "succeeded", "failed"]

# Type alias for the subprocess runner (injectable for testing)
SubprocessRunner = Callable[[List[str], Path], int]


class JobAlreadyRunningError(RuntimeError):
    """Raised when a training job cannot start because the GPU is already occupied."""


class JobNotFoundError(LookupError):
    """Raised when a job_id does not correspond to an existing job directory."""


# ── Internal helpers ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_subprocess_runner(cmd: List[str], log_file: Path) -> int:
    """Run a subprocess and redirect stdout+stderr to log_file. Returns exit code."""
    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.wait()


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Process exists but we lack permission to signal it — treat as alive.
        return True
    except OSError:
        # Covers ProcessLookupError (dead PID) and Windows-specific errors for invalid PIDs.
        return False


# ── JobManager ──────────────────────────────────────────────────────────────


class JobManager:
    """
    Manages training job lifecycle:

    - submit_training_job() — validates input, writes metadata.json, acquires disk GPU lock,
      unloads any loaded model, then spawns a daemon thread to run prepare_data.py + sft_12hz.py.
    - get_job() / get_job_log_tail() — read job state and log tail for GET /api/v1/jobs/{job_id}.
    - recover_stale_lock() — called on service startup to clean up orphaned gpu.lock files.
    """

    def __init__(
        self,
        baseline: RuntimeBaseline,
        metadata_store: MetadataStore,
        model_manager: ModelManager,
        *,
        subprocess_runner: SubprocessRunner | None = None,
    ) -> None:
        self._baseline = baseline
        self._store = metadata_store
        self._manager = model_manager
        self._subprocess_runner: SubprocessRunner = subprocess_runner or _default_subprocess_runner
        # Prevent two concurrent /train requests from both passing the is_gpu_busy check
        self._submit_lock = threading.Lock()

    # ── Startup recovery ────────────────────────────────────────────────────

    def recover_stale_lock(self) -> Optional[str]:
        """
        Check the disk GPU lock on service startup.

        Returns a log message string if a stale lock was cleaned up, or None if
        everything looks healthy.
        """
        lock_path = self._baseline.paths.gpu_lock_path
        if not lock_path.exists():
            return None

        try:
            record_dict = json.loads(lock_path.read_text(encoding="utf-8"))
            record = JobLockRecord(**record_dict)
        except Exception as exc:
            lock_path.unlink(missing_ok=True)
            return f"Removed malformed gpu.lock on startup: {exc}"

        if record.type != "training":
            # Inference locks should not survive a restart; clean them silently.
            lock_path.unlink(missing_ok=True)
            return f"Removed stale inference gpu.lock on startup (job_id={record.job_id})."

        if _is_pid_alive(record.pid):
            return None  # Training process is still running; leave the lock in place.

        # PID is dead → clean up lock and mark job as failed.
        lock_path.unlink(missing_ok=True)
        self._mark_job_failed_on_disk(
            record.job_id,
            error=f"Process PID={record.pid} not found on service startup; job marked failed.",
        )
        return (
            f"Cleaned stale training gpu.lock for job {record.job_id} "
            f"(PID {record.pid} no longer alive)."
        )

    # ── Job submission ───────────────────────────────────────────────────────

    def submit_training_job(
        self,
        *,
        base_model_id: int,
        speaker_name: str,
        input_jsonl: str | Path,
        num_epochs: int = 3,
        batch_size: int = 2,
        lr: float = 2e-5,
        tokenizer_model_path: str | Path | None = None,
    ) -> str:
        """
        Validate the training request, acquire the disk GPU lock, unload any loaded model,
        start a background daemon thread, and return a new job_id immediately.

        Raises:
            FileNotFoundError — input_jsonl does not exist, or base_model_id is unknown.
            ValueError — model type is not 'base', or sample count < min_training_samples.
            JobAlreadyRunningError — GPU is already busy (training or inference).
        """
        input_jsonl_path = Path(input_jsonl).expanduser().resolve()
        if not input_jsonl_path.is_file():
            raise FileNotFoundError(f"input_jsonl not found: {input_jsonl_path}")

        # Count samples (non-blank lines).
        raw_lines = input_jsonl_path.read_text(encoding="utf-8").splitlines()
        samples = [l for l in raw_lines if l.strip()]
        min_required = self._baseline.min_training_samples
        if len(samples) < min_required:
            raise ValueError(
                f"Training requires at least {min_required} samples, "
                f"found {len(samples)} in {input_jsonl_path}."
            )

        # Preflight: clamp batch_size to available samples.
        effective_batch = min(batch_size, len(samples))

        # Resolve base model record.
        model_record = self._store.get_model(base_model_id)
        if model_record is None:
            raise FileNotFoundError(f"Model not found in database: id={base_model_id}")
        if model_record.type != "base":
            raise ValueError(
                f"Training requires a model of type 'base', but model id={base_model_id} "
                f"has type '{model_record.type}'."
            )

        base_model_path = str(model_record.path)
        tok_path = str(tokenizer_model_path) if tokenizer_model_path else base_model_path

        with self._submit_lock:
            if self._manager.is_gpu_busy():
                raise JobAlreadyRunningError(
                    "GPU is currently busy (training or inference in progress). "
                    "Cannot start a new training job."
                )

            job_id = str(uuid.uuid4())
            job_dir = self._baseline.paths.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            output_model_dir = job_dir / "output"
            prepared_jsonl = job_dir / "train_prepared.jsonl"

            meta: dict[str, Any] = {
                "job_id": job_id,
                "status": "pending",
                "created_at": _now_iso(),
                "started_at": None,
                "finished_at": None,
                "base_model_id": base_model_id,
                "base_model_path": base_model_path,
                "speaker_name": speaker_name,
                "output_model_path": str(output_model_dir),
                "output_model_id": None,
                "error": None,
                "num_epochs": num_epochs,
                "batch_size": effective_batch,
                "lr": lr,
                "input_jsonl": str(input_jsonl_path),
                "prepared_jsonl": str(prepared_jsonl),
            }
            (job_dir / "metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Acquire disk lock before starting the thread.
            lock_record = JobLockRecord.create(job_id, os.getpid(), "training")
            self._baseline.paths.gpu_lock_path.write_text(
                json.dumps(lock_record.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Unload any currently loaded model so the training process has full VRAM.
            self._manager.unload_model()

        # Start daemon thread (outside submit_lock to avoid long hold).
        thread = threading.Thread(
            target=self._run_training_background,
            kwargs=dict(
                job_id=job_id,
                job_dir=job_dir,
                base_model_path=base_model_path,
                tok_model_path=tok_path,
                input_jsonl=input_jsonl_path,
                prepared_jsonl=prepared_jsonl,
                output_model_dir=output_model_dir,
                speaker_name=speaker_name,
                num_epochs=num_epochs,
                batch_size=effective_batch,
                lr=lr,
            ),
            daemon=True,
            name=f"train-{job_id[:8]}",
        )
        thread.start()
        return job_id

    # ── Job query ────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Return the job metadata dict, or None if the job_id does not exist."""
        meta_path = self._baseline.paths.jobs_dir / job_id / "metadata.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def get_job_log_tail(self, job_id: str, lines: int = 50) -> Optional[str]:
        """Return the last `lines` lines of train.log, or None if job not found."""
        job_dir = self._baseline.paths.jobs_dir / job_id
        if not job_dir.is_dir():
            return None
        log_path = job_dir / "train.log"
        if not log_path.exists():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = text.splitlines()[-lines:]
        return "\n".join(tail)

    # ── Background training thread ───────────────────────────────────────────

    def _run_training_background(
        self,
        *,
        job_id: str,
        job_dir: Path,
        base_model_path: str,
        tok_model_path: str,
        input_jsonl: Path,
        prepared_jsonl: Path,
        output_model_dir: Path,
        speaker_name: str,
        num_epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        meta_path = job_dir / "metadata.json"

        def _update_meta(**kwargs: Any) -> None:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta.update(kwargs)
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                pass  # Best-effort; don't let metadata errors crash the thread.

        def _release_lock() -> None:
            self._baseline.paths.gpu_lock_path.unlink(missing_ok=True)

        _update_meta(status="running", started_at=_now_iso())

        try:
            # ── Step 1: tokenize training data ───────────────────────────────
            prepare_cmd = self._build_prepare_cmd(tok_model_path, input_jsonl, prepared_jsonl)
            prepare_log = job_dir / "prepare.log"
            ret = self._subprocess_runner(prepare_cmd, prepare_log)
            if ret != 0:
                raise RuntimeError(
                    f"prepare_data.py failed with exit code {ret}. "
                    f"See {prepare_log} for details."
                )

            # ── Step 2: fine-tune ────────────────────────────────────────────
            output_model_dir.mkdir(parents=True, exist_ok=True)
            train_cmd = self._build_train_cmd(
                base_model_path,
                output_model_dir,
                prepared_jsonl,
                speaker_name,
                num_epochs,
                batch_size,
                lr,
            )
            train_log = job_dir / "train.log"
            ret = self._subprocess_runner(train_cmd, train_log)
            if ret != 0:
                raise RuntimeError(
                    f"sft_12hz.py failed with exit code {ret}. "
                    f"See {train_log} for details."
                )

            # ── Step 3: register final checkpoint ────────────────────────────
            final_ckpt = output_model_dir / f"checkpoint-epoch-{num_epochs - 1}"
            output_model_id: Optional[int] = None
            if final_ckpt.is_dir():
                ckpt_record = self._store.register_model(
                    name=f"{speaker_name}_finetuned_ep{num_epochs - 1}",
                    model_type="custom_voice",
                    path=final_ckpt,
                    speaker=speaker_name,
                )
                output_model_id = ckpt_record.id

            _release_lock()
            _update_meta(
                status="succeeded",
                finished_at=_now_iso(),
                output_model_id=output_model_id,
            )

        except Exception as exc:
            _release_lock()
            _update_meta(
                status="failed",
                finished_at=_now_iso(),
                error=str(exc),
            )

    # ── Command builders ─────────────────────────────────────────────────────

    def _build_prepare_cmd(
        self,
        tok_path: str,
        input_jsonl: Path,
        output_jsonl: Path,
    ) -> List[str]:
        finetuning_dir = Path(__file__).resolve().parents[2] / "finetuning"
        return [
            sys.executable,
            str(finetuning_dir / "prepare_data.py"),
            "--device", self._baseline.device_map,
            "--tokenizer_model_path", tok_path,
            "--input_jsonl", str(input_jsonl),
            "--output_jsonl", str(output_jsonl),
        ]

    def _build_train_cmd(
        self,
        base_model_path: str,
        output_model_dir: Path,
        train_jsonl: Path,
        speaker_name: str,
        num_epochs: int,
        batch_size: int,
        lr: float,
    ) -> List[str]:
        finetuning_dir = Path(__file__).resolve().parents[2] / "finetuning"
        return [
            sys.executable,
            str(finetuning_dir / "sft_12hz.py"),
            "--init_model_path", base_model_path,
            "--output_model_path", str(output_model_dir),
            "--train_jsonl", str(train_jsonl),
            "--speaker_name", speaker_name,
            "--num_epochs", str(num_epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
        ]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _mark_job_failed_on_disk(self, job_id: str, *, error: str) -> None:
        meta_path = self._baseline.paths.jobs_dir / job_id / "metadata.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") in ("pending", "running"):
                meta["status"] = "failed"
                meta["error"] = error
                meta["finished_at"] = _now_iso()
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        except Exception:
            pass


def build_job_manager(
    baseline: RuntimeBaseline,
    metadata_store: MetadataStore,
    model_manager: ModelManager,
) -> JobManager:
    return JobManager(
        baseline=baseline,
        metadata_store=metadata_store,
        model_manager=model_manager,
    )
