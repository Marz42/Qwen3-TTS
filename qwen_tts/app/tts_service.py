from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch

from .metadata import MetadataStore, ModelRecord
from .model_manager import ModelManager
from .runtime import RuntimeBaseline


@dataclass(frozen=True)
class TTSGenerationResult:
    request_id: str
    model_id: int
    model_type: str
    sample_rate: int
    output_files: list[Path]


def generate_tts(
    *,
    payload: dict[str, Any],
    metadata_store: MetadataStore,
    model_manager: ModelManager,
    baseline: RuntimeBaseline,
) -> TTSGenerationResult:
    model_id = int(payload["model_id"])
    record = metadata_store.get_model(model_id)
    if record is None:
        raise FileNotFoundError(f"Model id not found: {model_id}")

    prompt_object = _resolve_prompt_if_needed(payload=payload, metadata_store=metadata_store)
    _validate_payload(record=record, payload=payload, prompt_object=prompt_object)
    generation_kwargs = _collect_generation_kwargs(payload)

    _cleanup_outputs_dir(baseline)

    def run_generation(model):
        if record.type == "custom_voice":
            return model.generate_custom_voice(
                text=payload["text"],
                language=payload.get("language"),
                speaker=payload["speaker"],
                instruct=payload.get("instruct"),
                **generation_kwargs,
            )
        if record.type == "voice_design":
            return model.generate_voice_design(
                text=payload["text"],
                language=payload.get("language"),
                instruct=payload["instruct"],
                **generation_kwargs,
            )
        if record.type == "base":
            base_kwargs: dict[str, Any] = {
                "text": payload["text"],
                "language": payload.get("language"),
                "x_vector_only_mode": payload.get("x_vector_only_mode", False),
                **generation_kwargs,
            }
            if prompt_object is not None:
                base_kwargs["voice_clone_prompt"] = prompt_object
            else:
                base_kwargs["ref_audio"] = payload.get("ref_audio")
                base_kwargs["ref_text"] = payload.get("ref_text")
            return model.generate_voice_clone(**base_kwargs)
        raise ValueError(f"Unsupported model type: {record.type}")

    wavs, sample_rate = model_manager.run_inference(
        record.path,
        run_generation,
        expected_model_type=record.type,
        blocking=False,
    )

    request_id = str(uuid.uuid4())
    output_files = _save_wavs(
        wavs=wavs,
        sample_rate=int(sample_rate),
        request_id=request_id,
        outputs_dir=baseline.paths.outputs_dir,
    )
    return TTSGenerationResult(
        request_id=request_id,
        model_id=record.id,
        model_type=record.type,
        sample_rate=int(sample_rate),
        output_files=output_files,
    )


def _resolve_prompt_if_needed(*, payload: dict[str, Any], metadata_store: MetadataStore) -> Optional[Any]:
    prompt_id = payload.get("prompt_id")
    if prompt_id is None:
        return None
    prompt_record = metadata_store.get_voice_prompt(int(prompt_id))
    if prompt_record is None:
        raise FileNotFoundError(f"Voice prompt id not found: {prompt_id}")
    prompt_path = Path(prompt_record.prompt_file)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return torch.load(prompt_path, map_location="cpu")


def _validate_payload(*, record: ModelRecord, payload: dict[str, Any], prompt_object: Optional[Any]) -> None:
    if not payload.get("text"):
        raise ValueError("text is required")

    if record.type == "custom_voice":
        if payload.get("speaker") in (None, ""):
            raise ValueError("custom_voice requires speaker")
        return

    if record.type == "voice_design":
        if payload.get("instruct") in (None, ""):
            raise ValueError("voice_design requires instruct")
        return

    if record.type == "base":
        has_ref_audio = payload.get("ref_audio") not in (None, "")
        has_prompt = prompt_object is not None
        xvec_only = payload.get("x_vector_only_mode", False)
        if not has_ref_audio and not has_prompt:
            raise ValueError("base requires ref_audio or prompt_id")
        if not xvec_only and not has_prompt and payload.get("ref_text") in (None, ""):
            raise ValueError("base requires ref_text when x_vector_only_mode is False and prompt_id is not provided")
        return

    raise ValueError(f"Unsupported model type: {record.type}")


def _collect_generation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "max_new_tokens",
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "subtalker_dosample",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_temperature",
        "non_streaming_mode",
    )
    kwargs = {}
    for field in fields:
        value = payload.get(field)
        if value is not None:
            kwargs[field] = value
    return kwargs


def _cleanup_outputs_dir(baseline: RuntimeBaseline) -> None:
    outputs_dir = baseline.paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    files = [path for path in outputs_dir.iterdir() if path.is_file()]
    if not files:
        return

    total_bytes = sum(path.stat().st_size for path in files)
    over_files = len(files) >= baseline.output_cleanup_max_files
    over_bytes = total_bytes >= baseline.output_cleanup_max_total_bytes
    if not (over_files or over_bytes):
        return

    files_sorted = sorted(files, key=lambda path: path.stat().st_mtime)
    to_delete = min(len(files_sorted), max(1, baseline.output_cleanup_delete_batch))
    for path in files_sorted[:to_delete]:
        path.unlink(missing_ok=True)


def _save_wavs(*, wavs: list[Any], sample_rate: int, request_id: str, outputs_dir: Path) -> list[Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    for index, wav in enumerate(wavs):
        wav_array = _ensure_numpy_array(wav)
        output_path = outputs_dir / f"{request_id}_{index}.wav"
        sf.write(output_path, wav_array, sample_rate)
        output_files.append(output_path)
    return output_files


def _ensure_numpy_array(wav: Any) -> np.ndarray:
    if isinstance(wav, np.ndarray):
        return wav
    if isinstance(wav, torch.Tensor):
        return wav.detach().cpu().numpy()
    return np.asarray(wav)