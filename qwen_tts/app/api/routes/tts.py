from __future__ import annotations

from fastapi import APIRouter, Request

from ...tts_service import generate_tts
from ..schemas import TTSGenerateRequest, TTSGenerateResponse

router = APIRouter(prefix="/api/v1/tts", tags=["tts"])


@router.post("/generate", response_model=TTSGenerateResponse)
def generate(request: Request, payload: TTSGenerateRequest) -> TTSGenerateResponse:
    baseline = request.app.state.baseline
    metadata_store = request.app.state.metadata_store
    model_manager = request.app.state.model_manager

    result = generate_tts(
        payload=payload.model_dump(),
        metadata_store=metadata_store,
        model_manager=model_manager,
        baseline=baseline,
    )

    output_urls = [f"/static/outputs/{path.name}" for path in result.output_files]
    return TTSGenerateResponse(
        request_id=result.request_id,
        model_id=result.model_id,
        model_type=result.model_type,
        sample_rate=result.sample_rate,
        output_urls=output_urls,
    )