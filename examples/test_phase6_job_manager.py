# coding=utf-8
"""
Phase 6 – TestClient narrow validation.

Validates the job state machine and REST API endpoints using fake subprocess
runners and fake model loaders.  Does NOT require real model weights or a GPU.

Run with:
    python examples/test_phase6_job_manager.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, List

# ── Make sure qwen_tts package is importable ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from qwen_tts.app.api.main import create_app
from qwen_tts.app.job_manager import JobManager, build_job_manager
from qwen_tts.app.metadata import build_metadata_store
from qwen_tts.app.model_manager import ModelManager
from qwen_tts.app.runtime import build_runtime_baseline, ensure_phase0_layout


# ── Helpers ────────────────────────────────────────────────────────────────

def fake_model_loader(path: str, **kwargs: Any):
    """No-op loader that pretends to load a CustomVoice model."""
    class FakeModel:
        class _Config:
            tts_model_type = "base"
        config = _Config()
        def generate_voice_clone(self, *a, **kw):
            import numpy as np
            return [np.zeros(16000, dtype=np.float32)]
    return FakeModel()


def make_success_subprocess_runner(job_dir_ref: dict):
    """
    Returns a SubprocessRunner that simulates a successful training run.
    Creates the expected checkpoint directory so the DB registration path fires.
    """
    def runner(cmd: List[str], log_file: Path) -> int:
        log_file.write_text(f"fake runner: {cmd}\nDone.\n", encoding="utf-8")
        # Determine if this is sft_12hz (--output_model_path in cmd)
        if "--output_model_path" in cmd:
            idx = cmd.index("--output_model_path")
            output_dir = Path(cmd[idx + 1])
            num_epochs_idx = cmd.index("--num_epochs") if "--num_epochs" in cmd else None
            num_epochs = int(cmd[num_epochs_idx + 1]) if num_epochs_idx is not None else 1
            ckpt = output_dir / f"checkpoint-epoch-{num_epochs - 1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            # Write minimal config.json so register_model (which checks is_dir) passes
            (ckpt / "config.json").write_text('{"tts_model_type":"custom_voice"}', encoding="utf-8")
        return 0
    return runner


def make_fail_subprocess_runner():
    """Returns a SubprocessRunner that immediately fails."""
    def runner(cmd: List[str], log_file: Path) -> int:
        log_file.write_text("fake runner: failing intentionally\n", encoding="utf-8")
        return 1
    return runner


def write_fake_jsonl(path: Path, n: int = 5) -> None:
    """Write n minimal JSONL lines (raw input format for prepare_data.py)."""
    lines = [json.dumps({"audio": "fake.wav", "text": f"sample {i}"}) for i in range(n)]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Test setup ─────────────────────────────────────────────────────────────

def build_test_app(tmp_dir: Path, subprocess_runner=None):
    baseline = build_runtime_baseline(repo_root=tmp_dir)
    ensure_phase0_layout(baseline)

    # Create the static/outputs dir expected by StaticFiles mount
    (tmp_dir / "static" / "outputs").mkdir(parents=True, exist_ok=True)

    store = build_metadata_store(baseline)
    manager = ModelManager(baseline=baseline, loader=fake_model_loader)

    # Register a fake base model directory
    base_model_dir = tmp_dir / "fake_base_model"
    base_model_dir.mkdir(parents=True, exist_ok=True)
    (base_model_dir / "config.json").write_text('{"tts_model_type":"base"}', encoding="utf-8")
    base_record = store.register_model(
        name="fake-base",
        model_type="base",
        path=base_model_dir,
    )

    job_mgr = JobManager(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        subprocess_runner=subprocess_runner,
    )

    app = create_app(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        job_manager=job_mgr,
    )
    return app, base_record.id


# ── Tests ──────────────────────────────────────────────────────────────────

def test_submit_and_job_enters_running(tmp_path: Path) -> None:
    """Test 1: a submitted job should reach 'running' (then 'succeeded')."""
    app, base_id = build_test_app(tmp_path, subprocess_runner=make_success_subprocess_runner({}))
    client = TestClient(app, raise_server_exceptions=True)

    input_jsonl = tmp_path / "train_raw.jsonl"
    write_fake_jsonl(input_jsonl, n=6)

    resp = client.post("/api/v1/models/train", json={
        "base_model_id": base_id,
        "speaker_name": "tester",
        "input_jsonl": str(input_jsonl),
        "num_epochs": 1,
        "batch_size": 2,
    })
    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
    body = resp.json()
    job_id = body["job_id"]
    assert body["status"] in ("pending", "running"), f"Unexpected status: {body['status']}"
    print(f"  [PASS] POST /api/v1/models/train → job_id={job_id}, status={body['status']}")

    # Poll until succeeded or timeout
    for _ in range(30):
        time.sleep(0.3)
        s = client.get(f"/api/v1/jobs/{job_id}")
        assert s.status_code == 200
        if s.json()["status"] in ("succeeded", "failed"):
            break

    final = client.get(f"/api/v1/jobs/{job_id}").json()
    assert final["status"] == "succeeded", f"Expected succeeded, got: {final}"
    print(f"  [PASS] Job reached succeeded: output_model_id={final.get('output_model_id')}")


def test_tts_returns_503_during_training(tmp_path: Path) -> None:
    """Test 2: TTS generate returns 503 while disk GPU lock exists."""
    app, base_id = build_test_app(tmp_path, subprocess_runner=make_success_subprocess_runner({}))
    client = TestClient(app, raise_server_exceptions=False)

    # Manually create a disk GPU lock to simulate training in progress
    lock_path = tmp_path / "data" / "jobs" / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_data = {"job_id": "fake-job-001", "pid": 99999, "created_at": "2025-01-01T00:00:00+00:00", "type": "training"}
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")

    # Register a custom_voice model to use for TTS
    from qwen_tts.app.metadata import build_metadata_store as bms
    store = app.state.metadata_store
    cv_dir = tmp_path / "fake_cv"
    cv_dir.mkdir(exist_ok=True)
    (cv_dir / "config.json").write_text('{}', encoding="utf-8")
    cv_rec = store.register_model(name="fake-cv", model_type="custom_voice", path=cv_dir, speaker="spk1")

    resp = client.post("/api/v1/tts/generate", json={
        "model_id": cv_rec.id,
        "text": "hello",
        "speaker": "spk1",
    })
    # Clean up lock before assertions
    lock_path.unlink(missing_ok=True)

    assert resp.status_code == 503, f"Expected 503 during training lock, got {resp.status_code}: {resp.text}"
    print(f"  [PASS] TTS returns 503 during training lock: {resp.json()['detail'][:60]}")


def test_failed_training_releases_lock(tmp_path: Path) -> None:
    """Test 3: a failed training run releases the disk lock and marks job failed."""
    app, base_id = build_test_app(tmp_path, subprocess_runner=make_fail_subprocess_runner())
    client = TestClient(app, raise_server_exceptions=True)

    input_jsonl = tmp_path / "train_raw.jsonl"
    write_fake_jsonl(input_jsonl, n=5)

    resp = client.post("/api/v1/models/train", json={
        "base_model_id": base_id,
        "speaker_name": "tester",
        "input_jsonl": str(input_jsonl),
        "num_epochs": 1,
        "batch_size": 1,
    })
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Wait for thread to finish
    for _ in range(30):
        time.sleep(0.3)
        s = client.get(f"/api/v1/jobs/{job_id}")
        if s.json()["status"] in ("succeeded", "failed"):
            break

    final = client.get(f"/api/v1/jobs/{job_id}").json()
    assert final["status"] == "failed", f"Expected failed, got: {final}"
    assert final.get("error"), "Error message should be populated"

    # Lock file must be gone
    lock_path = tmp_path / "data" / "jobs" / "gpu.lock"
    assert not lock_path.exists(), "gpu.lock should have been released after failure"
    print(f"  [PASS] Failed job: status=failed, lock released. error={final['error'][:60]}")


def test_preflight_rejects_too_few_samples(tmp_path: Path) -> None:
    """Test 4 (preflight): fewer than 5 samples should be rejected with 400."""
    app, base_id = build_test_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    input_jsonl = tmp_path / "small.jsonl"
    write_fake_jsonl(input_jsonl, n=3)

    resp = client.post("/api/v1/models/train", json={
        "base_model_id": base_id,
        "speaker_name": "tester",
        "input_jsonl": str(input_jsonl),
    })
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    print(f"  [PASS] Rejected too-few samples: {resp.json()['detail'][:60]}")


def test_preflight_rejects_non_base_model(tmp_path: Path) -> None:
    """Test 5 (preflight): training on a non-base model should be rejected with 400."""
    app, base_id = build_test_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    # Register a custom_voice model
    cv_dir = tmp_path / "cv_model"
    cv_dir.mkdir(exist_ok=True)
    (cv_dir / "config.json").write_text('{}', encoding="utf-8")
    cv_rec = app.state.metadata_store.register_model(
        name="cv", model_type="custom_voice", path=cv_dir, speaker="spk"
    )

    input_jsonl = tmp_path / "train.jsonl"
    write_fake_jsonl(input_jsonl, n=5)

    resp = client.post("/api/v1/models/train", json={
        "base_model_id": cv_rec.id,
        "speaker_name": "tester",
        "input_jsonl": str(input_jsonl),
    })
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    print(f"  [PASS] Rejected non-base model: {resp.json()['detail'][:60]}")


def test_stale_lock_recovery(tmp_path: Path) -> None:
    """Test 6: startup recovers a stale training lock whose PID is dead."""
    baseline_pre = build_runtime_baseline(repo_root=tmp_path)
    ensure_phase0_layout(baseline_pre)

    # Write a stale lock with an impossible PID
    lock_path = baseline_pre.paths.gpu_lock_path
    stale_job_id = "stale-job-abc"
    lock_data = {
        "job_id": stale_job_id,
        "pid": 9999999,  # almost certainly dead
        "created_at": "2020-01-01T00:00:00+00:00",
        "type": "training",
    }
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")

    # Create a matching job directory in running state
    job_dir = baseline_pre.paths.jobs_dir / stale_job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "job_id": stale_job_id,
        "status": "running",
        "created_at": "2020-01-01T00:00:00+00:00",
        "started_at": "2020-01-01T00:01:00+00:00",
        "finished_at": None,
        "base_model_id": 1,
        "base_model_path": "/fake",
        "speaker_name": "spk",
        "output_model_path": "/fake/out",
        "output_model_id": None,
        "error": None,
        "num_epochs": 1,
        "batch_size": 1,
        "lr": 2e-5,
        "input_jsonl": "/fake/train.jsonl",
        "prepared_jsonl": "/fake/prepared.jsonl",
    }
    (job_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Now create the app — recovery should fire during create_app()
    (tmp_path / "static" / "outputs").mkdir(parents=True, exist_ok=True)
    store = build_metadata_store(baseline_pre)
    manager = ModelManager(baseline=baseline_pre, loader=fake_model_loader)
    app = create_app(baseline=baseline_pre, metadata_store=store, model_manager=manager)
    client = TestClient(app, raise_server_exceptions=True)

    # Lock should be gone
    assert not lock_path.exists(), "Stale lock should have been removed by recovery"

    # Job should be marked failed
    recovered_meta = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
    assert recovered_meta["status"] == "failed", f"Stale job should be failed: {recovered_meta}"
    print(f"  [PASS] Stale lock recovered: job status={recovered_meta['status']}")


def test_get_job_log_tail(tmp_path: Path) -> None:
    """Test 7: GET /api/v1/jobs/{job_id} includes log_tail after training completes."""
    app, base_id = build_test_app(tmp_path, subprocess_runner=make_success_subprocess_runner({}))
    client = TestClient(app, raise_server_exceptions=True)

    input_jsonl = tmp_path / "train_raw.jsonl"
    write_fake_jsonl(input_jsonl, n=5)

    resp = client.post("/api/v1/models/train", json={
        "base_model_id": base_id,
        "speaker_name": "tester",
        "input_jsonl": str(input_jsonl),
        "num_epochs": 1,
        "batch_size": 1,
    })
    job_id = resp.json()["job_id"]

    for _ in range(30):
        time.sleep(0.3)
        s = client.get(f"/api/v1/jobs/{job_id}")
        if s.json()["status"] in ("succeeded", "failed"):
            break

    s = client.get(f"/api/v1/jobs/{job_id}")
    body = s.json()
    assert body["log_tail"] is not None, "log_tail should be present"
    print(f"  [PASS] log_tail returned ({len(body['log_tail'])} chars): {body['log_tail'][:60]!r}")


def test_concurrent_train_rejected(tmp_path: Path) -> None:
    """Test 8: A second training submission while one is running returns 409."""

    # Use a slow subprocess runner (sleeps briefly so the job stays in running)
    slow_done = threading.Event()

    def slow_runner(cmd: List[str], log_file: Path) -> int:
        log_file.write_text("slow fake runner\n", encoding="utf-8")
        slow_done.wait(timeout=5)
        return 0

    app, base_id = build_test_app(tmp_path, subprocess_runner=slow_runner)
    client = TestClient(app, raise_server_exceptions=False)

    input_jsonl = tmp_path / "train_raw.jsonl"
    write_fake_jsonl(input_jsonl, n=5)

    train_payload = {
        "base_model_id": base_id,
        "speaker_name": "tester",
        "input_jsonl": str(input_jsonl),
        "num_epochs": 1,
        "batch_size": 1,
    }

    # First submission should succeed
    resp1 = client.post("/api/v1/models/train", json=train_payload)
    assert resp1.status_code == 202, f"First submit should be 202, got {resp1.status_code}: {resp1.text}"

    # Wait briefly so the thread acquires the lock
    time.sleep(0.3)

    # Second submission should be rejected
    resp2 = client.post("/api/v1/models/train", json=train_payload)
    assert resp2.status_code == 409, f"Second submit should be 409, got {resp2.status_code}: {resp2.text}"
    print(f"  [PASS] Concurrent training rejected with 409: {resp2.json()['detail'][:60]}")

    # Allow the slow runner to finish
    slow_done.set()
    time.sleep(0.5)


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        ("submit_and_job_enters_running", test_submit_and_job_enters_running),
        ("tts_returns_503_during_training", test_tts_returns_503_during_training),
        ("failed_training_releases_lock", test_failed_training_releases_lock),
        ("preflight_rejects_too_few_samples", test_preflight_rejects_too_few_samples),
        ("preflight_rejects_non_base_model", test_preflight_rejects_non_base_model),
        ("stale_lock_recovery", test_stale_lock_recovery),
        ("get_job_log_tail", test_get_job_log_tail),
        ("concurrent_train_rejected", test_concurrent_train_rejected),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            print(f"\n[TEST] {name}")
            try:
                fn(Path(td))
                passed += 1
            except Exception:
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"Phase 6 TestClient validation: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
