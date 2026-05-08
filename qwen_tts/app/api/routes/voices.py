from __future__ import annotations

from fastapi import APIRouter, Request

from ...voice_service import extract_voice_prompt
from ..schemas import VoicePromptSummary
from ..schemas import VoicePromptExtractRequest, VoicePromptExtractResponse

router = APIRouter(prefix="/api/v1/voices", tags=["voices"])


@router.get("/list", response_model=list[VoicePromptSummary])
def list_voices(request: Request) -> list[VoicePromptSummary]:
    store = request.app.state.metadata_store
    return [VoicePromptSummary(**record.__dict__) for record in store.list_voice_prompts()]


@router.post("/extract_prompt", response_model=VoicePromptExtractResponse)
def extract_prompt(request: Request, payload: VoicePromptExtractRequest) -> VoicePromptExtractResponse:
    baseline = request.app.state.baseline
    metadata_store = request.app.state.metadata_store
    model_manager = request.app.state.model_manager

    result = extract_voice_prompt(
        payload=payload.model_dump(),
        metadata_store=metadata_store,
        model_manager=model_manager,
        baseline=baseline,
    )
    return VoicePromptExtractResponse(
        prompt_id=result.prompt_id,
        model_id=result.model_id,
        model_type=result.model_type,
        prompt_name=result.prompt_name,
        prompt_file=str(result.prompt_file),
        ref_text=result.ref_text,
    )