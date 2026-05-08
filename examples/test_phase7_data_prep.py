# coding=utf-8
"""
Phase 7 – Data Prep API + Train submit narrow validation.

Run with:
    python examples/test_phase7_data_prep.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from qwen_tts.app.api.main import create_app
from qwen_tts.app.job_manager import JobManager
from qwen_tts.app.metadata import build_metadata_store
from qwen_tts.app.model_manager import ModelManager
from qwen_tts.app.runtime import build_runtime_baseline, ensure_phase0_layout


def fake_model_loader(path: str, **kwargs: Any):
    class FakeModel:
        class _Config:
            tts_model_type = "base"
        config = _Config()
    return FakeModel()


def success_runner(cmd: List[str], log_file: Path) -> int:
    log_file.write_text("ok\n", encoding="utf-8")
    if "--output_model_path" in cmd:
        out_dir = Path(cmd[cmd.index("--output_model_path") + 1])
        epochs = int(cmd[cmd.index("--num_epochs") + 1]) if "--num_epochs" in cmd else 1
        ckpt = out_dir / f"checkpoint-epoch-{epochs - 1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "config.json").write_text('{"tts_model_type":"custom_voice"}', encoding="utf-8")
    return 0


def _write_wav_placeholder(path: Path) -> None:
    path.write_bytes(b"RIFF0000WAVEfmt ")


def _build_app(tmp: Path) -> tuple[TestClient, int]:
    baseline = build_runtime_baseline(repo_root=tmp)
    ensure_phase0_layout(baseline)
    (tmp / "static" / "outputs").mkdir(parents=True, exist_ok=True)

    store = build_metadata_store(baseline)
    manager = ModelManager(baseline=baseline, loader=fake_model_loader)

    base_model = tmp / "models" / "base"
    base_model.mkdir(parents=True, exist_ok=True)
    (base_model / "config.json").write_text('{"tts_model_type":"base"}', encoding="utf-8")
    rec = store.register_model(name="base", model_type="base", path=base_model)

    job_manager = JobManager(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        subprocess_runner=success_runner,
    )

    app = create_app(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        job_manager=job_manager,
    )
    return TestClient(app, raise_server_exceptions=True), rec.id


def main() -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        client, base_id = _build_app(root)

        # Prepare 5 fake audio files and samples
        audio_files: list[Path] = []
        for i in range(5):
            p = root / f"a{i}.wav"
            _write_wav_placeholder(p)
            audio_files.append(p)

        # Build a zip archive for 2 files, and keep 3 as direct audio paths.
        zip_path = root / "batch.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(audio_files[0], arcname="nested/a0.wav")
            zf.write(audio_files[1], arcname="nested/a1.wav")

        r0 = client.post(
            "/api/v1/data/collect_samples",
            json={
                "audio_files": [str(audio_files[2]), str(audio_files[3]), str(audio_files[4])],
                "archives": [str(zip_path)],
                "use_asr_placeholder": True,
            },
        )
        assert r0.status_code == 200, r0.text
        body0 = r0.json()
        assert body0["sample_count"] == 5, body0
        assert body0.get("imported_dir"), body0
        samples = body0["samples"]
        assert all(s.get("asr_text") for s in samples), samples
        assert all(Path(s["audio"]).is_file() for s in samples), samples
        print("[PASS] collect_samples (audio + zip)")

        payload = {
            "output_name": "phase7_train_raw",
            "samples": samples,
        }

        r1 = client.post("/api/v1/data/build_train_jsonl", json=payload)
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        jsonl_path = Path(body1["output_jsonl"])
        assert jsonl_path.is_file(), f"jsonl missing: {jsonl_path}"
        assert body1["sample_count"] == 5, body1
        print("[PASS] build_train_jsonl")

        # Guard rail: less than 5 samples should be rejected by /models/train preflight.
        payload_small = {
            "output_name": "phase7_train_raw_small",
            "samples": samples[:4],
        }
        r_small_jsonl = client.post("/api/v1/data/build_train_jsonl", json=payload_small)
        assert r_small_jsonl.status_code == 200, r_small_jsonl.text
        small_jsonl = r_small_jsonl.json()["output_jsonl"]

        r_small_train = client.post(
            "/api/v1/models/train",
            json={
                "base_model_id": base_id,
                "speaker_name": "phase7_small",
                "input_jsonl": small_jsonl,
                "num_epochs": 1,
                "batch_size": 2,
                "lr": 2e-5,
            },
        )
        assert r_small_train.status_code == 400, r_small_train.text
        print("[PASS] train preflight rejects <5 samples")

        r2 = client.post(
            "/api/v1/models/train",
            json={
                "base_model_id": base_id,
                "speaker_name": "phase7_spk",
                "input_jsonl": str(jsonl_path),
                "num_epochs": 1,
                "batch_size": 2,
                "lr": 2e-5,
            },
        )
        assert r2.status_code == 202, r2.text
        job_id = r2.json()["job_id"]
        print("[PASS] submit train")

        for _ in range(20):
            time.sleep(0.2)
            r3 = client.get(f"/api/v1/jobs/{job_id}")
            assert r3.status_code == 200, r3.text
            status = r3.json()["status"]
            if status in ("succeeded", "failed"):
                break

        final = client.get(f"/api/v1/jobs/{job_id}").json()
        assert final["status"] == "succeeded", final
        assert final.get("output_model_id") is not None, final
        print("[PASS] query job status + model register")

    print("Phase 7 narrow validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
