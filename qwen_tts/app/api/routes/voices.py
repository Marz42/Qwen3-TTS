from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import VoicePromptSummary

router = APIRouter(prefix="/api/v1/voices", tags=["voices"])


@router.get("/list", response_model=list[VoicePromptSummary])
def list_voices(request: Request) -> list[VoicePromptSummary]:
    store = request.app.state.metadata_store
    return [VoicePromptSummary(**record.__dict__) for record in store.list_voice_prompts()]