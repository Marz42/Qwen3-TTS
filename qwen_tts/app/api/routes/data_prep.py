from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request

from ..schemas import (
    BuildTrainJsonlRequest,
    BuildTrainJsonlResponse,
    CollectSamplesRequest,
    CollectSamplesResponse,
    DataPrepSample,
)

router = APIRouter(prefix="/api/v1/data", tags=["data"])

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}


def _sanitize_output_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "train_raw"


def _is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTS


def _make_asr_placeholder(path: Path) -> str:
    return re.sub(r"[_\-]+", " ", path.stem).strip()


def _collect_audio_candidates(
    *,
    baseline,
    audio_files: list[str],
    archives: list[str],
) -> tuple[list[Path], Path | None]:
    candidates: list[Path] = []

    for audio in audio_files:
        audio_path = Path((audio or "").strip()).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not _is_audio_file(audio_path):
            raise ValueError(f"Unsupported audio extension: {audio_path}")
        candidates.append(audio_path)

    if not archives:
        return sorted(set(candidates)), None

    imports_root = baseline.paths.data_dir / "datasets" / "imports"
    imports_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    imported_dir = imports_root / f"import_{stamp}"
    imported_dir.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        archive_path = Path((archive or "").strip()).expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        if archive_path.suffix.lower() != ".zip":
            raise ValueError(f"Only .zip archives are supported in MVP: {archive_path}")

        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                member_path = Path(member.filename)
                if not _is_audio_file(member_path):
                    continue

                safe_name = member_path.name
                if not safe_name:
                    continue

                target = imported_dir / safe_name
                dedupe_idx = 1
                while target.exists():
                    target = imported_dir / f"{Path(safe_name).stem}_{dedupe_idx}{Path(safe_name).suffix}"
                    dedupe_idx += 1

                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    dst.write(src.read())

                candidates.append(target.resolve())

    return sorted(set(candidates)), imported_dir


@router.post("/collect_samples", response_model=CollectSamplesResponse)
def collect_samples(body: CollectSamplesRequest, request: Request) -> CollectSamplesResponse:
    if not body.audio_files and not body.archives:
        raise ValueError("audio_files and archives cannot both be empty.")

    baseline = request.app.state.baseline
    paths, imported_dir = _collect_audio_candidates(
        baseline=baseline,
        audio_files=body.audio_files,
        archives=body.archives,
    )

    if not paths:
        raise ValueError("No audio files were found from provided inputs.")

    samples: list[DataPrepSample] = []
    for audio_path in paths:
        asr_text = _make_asr_placeholder(audio_path) if body.use_asr_placeholder else None
        text = asr_text or ""
        samples.append(
            DataPrepSample(
                audio=str(audio_path),
                text=text,
                asr_text=asr_text,
            )
        )

    return CollectSamplesResponse(
        samples=samples,
        sample_count=len(samples),
        imported_dir=str(imported_dir) if imported_dir else None,
    )


@router.post("/build_train_jsonl", response_model=BuildTrainJsonlResponse)
def build_train_jsonl(body: BuildTrainJsonlRequest, request: Request) -> BuildTrainJsonlResponse:
    if not body.samples:
        raise ValueError("samples cannot be empty.")

    records: list[dict] = []
    for idx, sample in enumerate(body.samples):
        audio = (sample.audio or "").strip()
        text = (sample.text or "").strip()
        if not audio:
            raise ValueError(f"samples[{idx}].audio cannot be empty.")
        if not text:
            raise ValueError(f"samples[{idx}].text cannot be empty.")

        audio_path = Path(audio).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        record = {
            "audio": str(audio_path),
            "text": text,
        }
        if sample.asr_text:
            record["asr_text"] = sample.asr_text
        records.append(record)

    baseline = request.app.state.baseline
    out_dir = baseline.paths.data_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = _sanitize_output_name(body.output_name) if body.output_name else "train_raw"
    out_path = out_dir / f"{base_name}_{timestamp}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return BuildTrainJsonlResponse(
        output_jsonl=str(out_path),
        sample_count=len(records),
    )
