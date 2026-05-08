from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    db_path: str
    outputs_dir: str
    gpu_busy: bool
    current_model_path: Optional[str]
    model_record_count: int
    voice_prompt_count: int


class ModelSummary(BaseModel):
    id: int
    name: str
    type: str
    path: str
    speaker: Optional[str] = None


class VoicePromptSummary(BaseModel):
    id: int
    name: str
    ref_text: Optional[str] = None
    prompt_file: str


class VoicePromptExtractRequest(BaseModel):
    model_id: int
    ref_audio: str
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False
    name: Optional[str] = None


class VoicePromptExtractResponse(BaseModel):
    prompt_id: int
    model_id: int
    model_type: str
    prompt_name: str
    prompt_file: str
    ref_text: Optional[str] = None


class TTSGenerateRequest(BaseModel):
    model_id: int
    text: str | list[str]
    language: Optional[str | list[str]] = "Auto"
    speaker: Optional[str | list[str]] = None
    instruct: Optional[str | list[str]] = None
    prompt_id: Optional[int] = None
    ref_audio: Optional[str | list[str]] = None
    ref_text: Optional[str | list[str]] = None
    x_vector_only_mode: bool | list[bool] = False

    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    subtalker_dosample: Optional[bool] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None
    non_streaming_mode: Optional[bool] = None


class TTSGenerateResponse(BaseModel):
    request_id: str
    model_id: int
    model_type: str
    sample_rate: int
    output_urls: list[str]