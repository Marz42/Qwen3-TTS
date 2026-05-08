# coding=utf-8
"""
Phase 9 – End-to-end acceptance and regression checklist (scripted).

This script maps the MVP Phase 9 checklist into executable assertions.
It uses fake model loaders and fake subprocess runners, so no real GPU model
weights or flash-attn training environment is required.

Run with:
    python examples/test_phase9_e2e_checklist.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_tts.app.api.main import create_app
from qwen_tts.app.job_manager import JobManager
from qwen_tts.app.metadata import build_metadata_store
from qwen_tts.app.model_manager import ModelManager
from qwen_tts.app.runtime import build_runtime_baseline, ensure_phase0_layout
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem


def _write_wav_placeholder(path: Path) -> None:
    path.write_bytes(b"RIFF0000WAVEfmt ")


def _create_model_dir(root: Path, name: str, model_type: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"tts_model_type": model_type}), encoding="utf-8")
    return d


def _fake_model_loader(path: str, **kwargs: Any):
    model_dir = Path(path)
    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tts_model_type = cfg.get("tts_model_type", "custom_voice")
    else:
        tts_model_type = "custom_voice"

    class FakeModel:
        class _Config:
            pass

        def __init__(self):
            self.config = self._Config()
            self.config.tts_model_type = tts_model_type
            self.model = self

        def generate_custom_voice(self, *args, **kwargs):
            return [np.zeros(12000, dtype=np.float32)], 24000

        def generate_voice_design(self, *args, **kwargs):
            return [np.zeros(11000, dtype=np.float32)], 24000

        def generate_voice_clone(self, *args, **kwargs):
            return [np.zeros(10000, dtype=np.float32)], 24000

        def create_voice_clone_prompt(self, *args, **kwargs):
            item = VoiceClonePromptItem(
                ref_code=torch.zeros((1, 8), dtype=torch.long),
                ref_spk_embedding=torch.zeros((1, 256), dtype=torch.float32),
                x_vector_only_mode=bool(kwargs.get("x_vector_only_mode", False)),
                icl_mode=False,
                ref_text=kwargs.get("ref_text"),
            )
            return [item]

    return FakeModel()


def _success_runner(delay_s: float = 0.05) -> Callable[[List[str], Path], int]:
    def runner(cmd: List[str], log_file: Path) -> int:
        log_file.write_text(f"runner ok: {cmd}\n", encoding="utf-8")
        time.sleep(delay_s)
        if "--output_model_path" in cmd:
            out_dir = Path(cmd[cmd.index("--output_model_path") + 1])
            epochs = int(cmd[cmd.index("--num_epochs") + 1]) if "--num_epochs" in cmd else 1
            ckpt = out_dir / f"checkpoint-epoch-{epochs - 1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "config.json").write_text('{"tts_model_type":"custom_voice"}', encoding="utf-8")
        return 0

    return runner


def _fail_runner() -> Callable[[List[str], Path], int]:
    def runner(cmd: List[str], log_file: Path) -> int:
        log_file.write_text("runner fail\n", encoding="utf-8")
        return 1

    return runner


def _build_test_stack(
    root: Path,
    *,
    runner: Callable[[List[str], Path], int],
    output_cleanup_max_files: int = 200,
    output_cleanup_delete_batch: int = 20,
) -> tuple[TestClient, ModelManager, Dict[str, int], Any]:
    baseline = build_runtime_baseline(repo_root=root)
    baseline = replace(
        baseline,
        output_cleanup_max_files=output_cleanup_max_files,
        output_cleanup_delete_batch=output_cleanup_delete_batch,
    )
    ensure_phase0_layout(baseline)
    (root / "static" / "outputs").mkdir(parents=True, exist_ok=True)

    store = build_metadata_store(baseline)
    manager = ModelManager(baseline=baseline, loader=_fake_model_loader)

    base_dir = _create_model_dir(root, "m_base", "base")
    custom_dir = _create_model_dir(root, "m_custom", "custom_voice")
    vd_dir = _create_model_dir(root, "m_vd", "voice_design")

    base_rec = store.register_model(name="base-m", model_type="base", path=base_dir)
    custom_rec = store.register_model(
        name="custom-m",
        model_type="custom_voice",
        path=custom_dir,
        speaker="speakerA",
    )
    vd_rec = store.register_model(name="vd-m", model_type="voice_design", path=vd_dir)

    job_manager = JobManager(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        subprocess_runner=runner,
    )

    app = create_app(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        job_manager=job_manager,
    )

    ids = {
        "base": base_rec.id,
        "custom": custom_rec.id,
        "voice_design": vd_rec.id,
    }
    client = TestClient(app, raise_server_exceptions=False)
    return client, manager, ids, baseline


def _poll_job(client: TestClient, job_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        if resp.status_code == 200:
            last = resp.json()
            if last.get("status") in ("succeeded", "failed"):
                return last
        time.sleep(0.2)
    raise AssertionError(f"Job polling timed out, last={last}")


def _assert_output_exists(root: Path, output_urls: list[str]) -> None:
    assert output_urls, "output_urls is empty"
    first = output_urls[0]
    assert first.startswith("/static/outputs/"), first
    local = root / first.lstrip("/")
    assert local.is_file(), f"Output file missing: {local}"


def _check_no_dangling_records(client: TestClient) -> None:
    models = client.get("/api/v1/models/list").json()
    for item in models:
        p = Path(item["path"])
        assert p.exists(), f"Dangling model record path: {p}"

    prompts = client.get("/api/v1/voices/list").json()
    for item in prompts:
        p = Path(item["prompt_file"])
        assert p.exists(), f"Dangling prompt record path: {p}"


def main() -> int:
    checks: dict[str, bool] = {}

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)

        # Main stack for end-to-end success path and regression checks.
        client, manager, ids, baseline = _build_test_stack(
            root,
            runner=_success_runner(),
            output_cleanup_max_files=3,
            output_cleanup_delete_batch=2,
        )

        # Phase 9 Path A-1: DB has custom + base models.
        models = client.get("/api/v1/models/list")
        assert models.status_code == 200, models.text
        model_types = {m["type"] for m in models.json()}
        assert "base" in model_types and "custom_voice" in model_types, model_types
        checks["A1_base_custom_exist"] = True

        # Phase 9 Path A-2: custom_voice generate.
        r_custom = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["custom"],
                "text": "hello custom",
                "speaker": "speakerA",
            },
        )
        assert r_custom.status_code == 200, r_custom.text
        _assert_output_exists(root, r_custom.json().get("output_urls", []))
        checks["A2_custom_generate"] = True

        # Phase 9 Path A-3: base generate with ref_audio/ref_text.
        ref_audio = root / "ref_a.wav"
        _write_wav_placeholder(ref_audio)
        r_base_ref = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["base"],
                "text": "hello base",
                "ref_audio": str(ref_audio),
                "ref_text": "ref text",
            },
        )
        assert r_base_ref.status_code == 200, r_base_ref.text
        _assert_output_exists(root, r_base_ref.json().get("output_urls", []))
        checks["A3_base_ref_generate"] = True

        # Phase 9 Path A-4: extract prompt from base.
        r_extract = client.post(
            "/api/v1/voices/extract_prompt",
            json={
                "model_id": ids["base"],
                "ref_audio": str(ref_audio),
                "ref_text": "prompt ref",
                "x_vector_only_mode": False,
            },
        )
        assert r_extract.status_code == 200, r_extract.text
        prompt_id = int(r_extract.json()["prompt_id"])
        prompt_file = Path(r_extract.json()["prompt_file"])
        assert prompt_file.is_file(), prompt_file
        checks["A4_extract_prompt"] = True

        # Phase 9 Path A-5: generate with prompt_id.
        r_base_prompt = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["base"],
                "text": "hello by prompt",
                "prompt_id": prompt_id,
            },
        )
        assert r_base_prompt.status_code == 200, r_base_prompt.text
        _assert_output_exists(root, r_base_prompt.json().get("output_urls", []))
        checks["A5_base_prompt_generate"] = True

        # Phase 9 Path B-1..2: collect samples + build train_raw.jsonl.
        sample_files: list[Path] = []
        for i in range(5):
            p = root / f"train_{i}.wav"
            _write_wav_placeholder(p)
            sample_files.append(p)

        r_collect = client.post(
            "/api/v1/data/collect_samples",
            json={
                "audio_files": [str(p) for p in sample_files],
                "archives": [],
                "use_asr_placeholder": True,
            },
        )
        assert r_collect.status_code == 200, r_collect.text
        assert r_collect.json()["sample_count"] == 5, r_collect.json()

        r_build = client.post(
            "/api/v1/data/build_train_jsonl",
            json={
                "output_name": "phase9_train_raw",
                "samples": r_collect.json()["samples"],
            },
        )
        assert r_build.status_code == 200, r_build.text
        train_jsonl = Path(r_build.json()["output_jsonl"])
        assert train_jsonl.is_file(), train_jsonl
        checks["B1_B2_collect_and_build"] = True

        # Phase 9 Path B-3..5: submit train -> succeeded -> model registered.
        r_train = client.post(
            "/api/v1/models/train",
            json={
                "base_model_id": ids["base"],
                "speaker_name": "phase9_spk",
                "input_jsonl": str(train_jsonl),
                "num_epochs": 1,
                "batch_size": 2,
                "lr": 2e-5,
            },
        )
        assert r_train.status_code == 202, r_train.text
        job_id = r_train.json()["job_id"]

        final = _poll_job(client, job_id)
        assert final["status"] == "succeeded", final
        assert final.get("output_model_id") is not None, final
        checks["B3_B5_training_success_and_register"] = True

        # Phase 9 Path B-6: new model can generate custom_voice.
        new_model_id = int(final["output_model_id"])
        r_new = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": new_model_id,
                "text": "hello new model",
                "speaker": "phase9_spk",
            },
        )
        assert r_new.status_code == 200, r_new.text
        _assert_output_exists(root, r_new.json().get("output_urls", []))
        checks["B6_new_model_generate"] = True

        # Regression-1: training lock released after success.
        assert not baseline.paths.gpu_lock_path.exists(), "gpu.lock should be removed after successful train"
        checks["R1_training_lock_released"] = True

        # Regression-2: old/new model switching still works.
        r_old_again = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["custom"],
                "text": "switch back",
                "speaker": "speakerA",
            },
        )
        assert r_old_again.status_code == 200, r_old_again.text
        checks["R2_model_switch_ok"] = True

        # Regression-3 and Regression-4: paths traceable and no dangling records.
        _check_no_dangling_records(client)
        checks["R3_R4_paths_traceable_no_dangling"] = True

        # Regression-5: concurrent inference gets controlled rejection.
        got_lock = manager.inference_lock.acquire(blocking=False)
        assert got_lock, "failed to acquire inference_lock for test setup"
        try:
            r_busy = client.post(
                "/api/v1/tts/generate",
                json={
                    "model_id": ids["custom"],
                    "text": "concurrent test",
                    "speaker": "speakerA",
                },
            )
        finally:
            if got_lock:
                manager.inference_lock.release()

        assert r_busy.status_code == 503, r_busy.text
        checks["R5_concurrent_inference_rejected"] = True

        # Regression-7: output cleanup strategy effective.
        for i in range(6):
            resp = client.post(
                "/api/v1/tts/generate",
                json={
                    "model_id": ids["custom"],
                    "text": f"cleanup {i}",
                    "speaker": "speakerA",
                },
            )
            assert resp.status_code == 200, resp.text
        wavs = list((root / "static" / "outputs").glob("*.wav"))
        assert len(wavs) <= baseline.output_cleanup_max_files, len(wavs)
        checks["R7_output_cleanup_effective"] = True

        # Regression-log tracking: jobs endpoint returns log tail.
        r_job_detail = client.get(f"/api/v1/jobs/{job_id}")
        assert r_job_detail.status_code == 200, r_job_detail.text
        assert "runner ok" in (r_job_detail.json().get("log_tail") or ""), r_job_detail.json()
        checks["R_log_tracking"] = True

        # Failure path-1: missing prompt file.
        r_extract2 = client.post(
            "/api/v1/voices/extract_prompt",
            json={
                "model_id": ids["base"],
                "ref_audio": str(ref_audio),
                "ref_text": "for delete",
            },
        )
        assert r_extract2.status_code == 200, r_extract2.text
        prompt_file2 = Path(r_extract2.json()["prompt_file"])
        prompt_id2 = int(r_extract2.json()["prompt_id"])
        prompt_file2.unlink(missing_ok=True)
        r_missing_prompt = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["base"],
                "text": "should fail missing prompt",
                "prompt_id": prompt_id2,
            },
        )
        assert r_missing_prompt.status_code == 404, r_missing_prompt.text
        checks["F_prompt_missing"] = True

        # Failure path-2: invalid payload (custom_voice without speaker).
        r_invalid = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["custom"],
                "text": "missing speaker",
            },
        )
        assert r_invalid.status_code == 400, r_invalid.text
        checks["F_payload_invalid"] = True

        # Regression-6: restart recovery for stale lock.
        stale_job_id = "stale-phase9-job"
        stale_job_dir = baseline.paths.jobs_dir / stale_job_id
        stale_job_dir.mkdir(parents=True, exist_ok=True)
        (stale_job_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "job_id": stale_job_id,
                    "status": "running",
                    "created_at": "2026-05-08T00:00:00+00:00",
                    "started_at": "2026-05-08T00:00:00+00:00",
                    "finished_at": None,
                    "base_model_id": ids["base"],
                    "base_model_path": str(root / "m_base"),
                    "speaker_name": "stale",
                    "output_model_path": str(stale_job_dir / "output"),
                    "output_model_id": None,
                    "error": None,
                    "num_epochs": 1,
                    "batch_size": 1,
                    "lr": 2e-5,
                    "input_jsonl": str(train_jsonl),
                    "prepared_jsonl": str(stale_job_dir / "prepared.jsonl"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        baseline.paths.gpu_lock_path.write_text(
            json.dumps(
                {
                    "job_id": stale_job_id,
                    "pid": 99999999,
                    "created_at": "2026-05-08T00:00:00+00:00",
                    "type": "training",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Recreate app to trigger startup lock recovery.
        store2 = build_metadata_store(baseline)
        manager2 = ModelManager(baseline=baseline, loader=_fake_model_loader)
        jm2 = JobManager(
            baseline=baseline,
            metadata_store=store2,
            model_manager=manager2,
            subprocess_runner=_success_runner(),
        )
        _ = create_app(
            baseline=baseline,
            metadata_store=store2,
            model_manager=manager2,
            job_manager=jm2,
        )

        assert not baseline.paths.gpu_lock_path.exists(), "stale gpu.lock should be cleaned on startup"
        recovered_meta = json.loads((stale_job_dir / "metadata.json").read_text(encoding="utf-8"))
        assert recovered_meta["status"] == "failed", recovered_meta
        checks["R6_restart_recovery"] = True

    # Failure path-3: training failure should set failed and release lock.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td2:
        root2 = Path(td2)
        client2, _manager2, ids2, baseline2 = _build_test_stack(root2, runner=_fail_runner())

        wavs = []
        for i in range(5):
            p = root2 / f"f_{i}.wav"
            _write_wav_placeholder(p)
            wavs.append(p)

        r_build2 = client2.post(
            "/api/v1/data/build_train_jsonl",
            json={
                "output_name": "phase9_fail",
                "samples": [{"audio": str(p), "text": f"t{i}"} for i, p in enumerate(wavs)],
            },
        )
        assert r_build2.status_code == 200, r_build2.text

        r_train2 = client2.post(
            "/api/v1/models/train",
            json={
                "base_model_id": ids2["base"],
                "speaker_name": "fail_case",
                "input_jsonl": r_build2.json()["output_jsonl"],
                "num_epochs": 1,
                "batch_size": 1,
            },
        )
        assert r_train2.status_code == 202, r_train2.text

        final2 = _poll_job(client2, r_train2.json()["job_id"])
        assert final2["status"] == "failed", final2
        assert not baseline2.paths.gpu_lock_path.exists(), "gpu.lock should be released on failed train"
        checks["F_training_failed_release_lock"] = True

    expected = {
        "A1_base_custom_exist",
        "A2_custom_generate",
        "A3_base_ref_generate",
        "A4_extract_prompt",
        "A5_base_prompt_generate",
        "B1_B2_collect_and_build",
        "B3_B5_training_success_and_register",
        "B6_new_model_generate",
        "R1_training_lock_released",
        "R2_model_switch_ok",
        "R3_R4_paths_traceable_no_dangling",
        "R5_concurrent_inference_rejected",
        "R6_restart_recovery",
        "R7_output_cleanup_effective",
        "R_log_tracking",
        "F_prompt_missing",
        "F_payload_invalid",
        "F_training_failed_release_lock",
    }

    missing = sorted(expected - set(checks.keys()))
    assert not missing, f"Checklist items not executed: {missing}"

    print("Phase 9 checklist passed")
    for key in sorted(checks.keys()):
        print(f"  [PASS] {key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
