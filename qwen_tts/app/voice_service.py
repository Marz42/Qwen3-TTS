from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..inference.qwen3_tts_model import VoiceClonePromptItem
from .metadata import MetadataStore
from .model_manager import ModelManager
from .runtime import RuntimeBaseline


@dataclass(frozen=True)
class VoicePromptExtractionResult:
    prompt_id: int
    model_id: int
    model_type: str
    prompt_name: str
    prompt_file: Path
    ref_text: str | None


def extract_voice_prompt(
    *,
    payload: dict[str, Any],
    metadata_store: MetadataStore,
    model_manager: ModelManager,
    baseline: RuntimeBaseline,
) -> VoicePromptExtractionResult:
    model_id = int(payload["model_id"])
    record = metadata_store.get_model(model_id)
    if record is None:
        raise FileNotFoundError(f"Model id not found: {model_id}")
    if record.type != "base":
        raise ValueError("extract_prompt only supports base model")

    ref_audio = payload.get("ref_audio")
    if ref_audio in (None, ""):
        raise ValueError("extract_prompt requires ref_audio")

    xvec_only = bool(payload.get("x_vector_only_mode", False))
    ref_text = payload.get("ref_text")
    if not xvec_only and ref_text in (None, ""):
        raise ValueError("extract_prompt requires ref_text when x_vector_only_mode is False")

    def run_extract(model):
        return model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=xvec_only,
        )

    prompt_object = model_manager.run_inference(
        record.path,
        run_extract,
        expected_model_type="base",
        blocking=False,
    )

    serializable = _to_serializable(prompt_object)
    prompt_name = str(payload.get("name") or f"prompt_{uuid.uuid4()}")

    baseline.paths.prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = baseline.paths.prompts_dir / f"{uuid.uuid4()}.pt"
    torch.save(serializable, prompt_file)

    saved = metadata_store.register_voice_prompt(
        name=prompt_name,
        ref_text=str(ref_text) if ref_text is not None else None,
        prompt_file=prompt_file,
    )

    return VoicePromptExtractionResult(
        prompt_id=saved.id,
        model_id=record.id,
        model_type=record.type,
        prompt_name=saved.name,
        prompt_file=Path(saved.prompt_file),
        ref_text=saved.ref_text,
    )


def _to_serializable(prompt_object: Any) -> Any:
    """Convert List[VoiceClonePromptItem] to a list of plain dicts safe for torch.save.

    PyTorch >= 2.4 defaults weights_only=True on torch.load, which cannot unpickle
    custom dataclasses.  We save as a list of plain dicts (tensors + Python builtins)
    and reconstruct VoiceClonePromptItem after loading.
    """
    if isinstance(prompt_object, list) and prompt_object and isinstance(prompt_object[0], VoiceClonePromptItem):
        return [
            {
                "ref_code": item.ref_code.detach().cpu() if item.ref_code is not None else None,
                "ref_spk_embedding": item.ref_spk_embedding.detach().cpu(),
                "x_vector_only_mode": item.x_vector_only_mode,
                "icl_mode": item.icl_mode,
                "ref_text": item.ref_text,
            }
            for item in prompt_object
        ]
    # Fallback: already a plain dict or unknown format; move tensors to CPU
    return _move_tensors_to_cpu(prompt_object)


def _move_tensors_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_move_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _move_tensors_to_cpu(item) for key, item in value.items()}
    return value