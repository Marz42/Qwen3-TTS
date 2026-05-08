# coding=utf-8
"""
Phase 8 – TTS GUI backend narrow validation.

Validates API paths used by Phase 8 GUI:
- model list
- voice prompt list
- tts generate for custom_voice / voice_design / base
- 503 error while training lock is present

Run with:
    python examples/test_phase8_tts_gui.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_tts.app.api.main import create_app
from qwen_tts.app.job_manager import JobManager
from qwen_tts.app.metadata import build_metadata_store
from qwen_tts.app.model_manager import ModelManager
from qwen_tts.app.runtime import build_runtime_baseline, ensure_phase0_layout


def fake_model_loader(path: str, **kwargs: Any):
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
            # ModelManager expects Qwen3TTSModel-like objects with .model.config.tts_model_type
            self.model = self

        def generate_custom_voice(self, *args, **kwargs):
            return [np.zeros(8000, dtype=np.float32)], 24000

        def generate_voice_design(self, *args, **kwargs):
            return [np.zeros(9000, dtype=np.float32)], 24000

        def generate_voice_clone(self, *args, **kwargs):
            return [np.zeros(10000, dtype=np.float32)], 24000

    return FakeModel()


def _write_wav_placeholder(path: Path) -> None:
    path.write_bytes(b"RIFF0000WAVEfmt ")


def _create_model_dir(root: Path, name: str, model_type: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"tts_model_type": model_type}), encoding="utf-8")
    return d


def _build_app(tmp: Path) -> tuple[TestClient, dict[str, int]]:
    baseline = build_runtime_baseline(repo_root=tmp)
    ensure_phase0_layout(baseline)
    (tmp / "static" / "outputs").mkdir(parents=True, exist_ok=True)

    store = build_metadata_store(baseline)
    manager = ModelManager(baseline=baseline, loader=fake_model_loader)

    base_dir = _create_model_dir(tmp, "m_base", "base")
    cv_dir = _create_model_dir(tmp, "m_cv", "custom_voice")
    vd_dir = _create_model_dir(tmp, "m_vd", "voice_design")

    base_rec = store.register_model(name="base-m", model_type="base", path=base_dir)
    cv_rec = store.register_model(name="cv-m", model_type="custom_voice", path=cv_dir, speaker="spkA")
    vd_rec = store.register_model(name="vd-m", model_type="voice_design", path=vd_dir)

    prompt_file = tmp / "prompt.pt"
    prompt_file.write_text("dummy", encoding="utf-8")
    prompt_rec = store.register_voice_prompt(name="p1", prompt_file=prompt_file, ref_text="hello")

    job_manager = JobManager(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        subprocess_runner=lambda cmd, log: 0,
    )

    app = create_app(
        baseline=baseline,
        metadata_store=store,
        model_manager=manager,
        job_manager=job_manager,
    )

    ids = {
        "base": base_rec.id,
        "custom": cv_rec.id,
        "voice_design": vd_rec.id,
        "prompt": prompt_rec.id,
    }
    return TestClient(app, raise_server_exceptions=False), ids


def main() -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        client, ids = _build_app(root)

        r1 = client.get("/api/v1/models/list")
        assert r1.status_code == 200, r1.text
        models = r1.json()
        assert len(models) == 3, models
        print("[PASS] models list")

        r2 = client.get("/api/v1/voices/list")
        assert r2.status_code == 200, r2.text
        prompts = r2.json()
        assert len(prompts) >= 1, prompts
        print("[PASS] voices list")

        r3 = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["custom"],
                "text": "hello custom",
                "speaker": "spkA",
            },
        )
        assert r3.status_code == 200, r3.text
        assert r3.json().get("output_urls"), r3.json()
        print("[PASS] custom_voice generate")

        r4 = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["voice_design"],
                "text": "hello vd",
                "instruct": "energetic female voice",
            },
        )
        assert r4.status_code == 200, r4.text
        assert r4.json().get("output_urls"), r4.json()
        print("[PASS] voice_design generate")

        ref_wav = root / "ref.wav"
        _write_wav_placeholder(ref_wav)
        r5 = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["base"],
                "text": "hello base",
                "ref_audio": str(ref_wav),
                "ref_text": "reference text",
            },
        )
        assert r5.status_code == 200, r5.text
        assert r5.json().get("output_urls"), r5.json()
        print("[PASS] base generate")

        lock_path = root / "data" / "jobs" / "gpu.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(
                {
                    "job_id": "train-job-1",
                    "pid": 999999,
                    "created_at": "2026-05-08T00:00:00+00:00",
                    "type": "training",
                }
            ),
            encoding="utf-8",
        )
        r6 = client.post(
            "/api/v1/tts/generate",
            json={
                "model_id": ids["custom"],
                "text": "should fail",
                "speaker": "spkA",
            },
        )
        lock_path.unlink(missing_ok=True)

        assert r6.status_code == 503, r6.text
        print("[PASS] tts returns 503 while training lock exists")

    print("Phase 8 backend narrow validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
