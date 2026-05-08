from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import ModelSummary
from ..schemas import ModelSummary, TrainRequest, TrainResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/list", response_model=list[ModelSummary])
def list_models(request: Request) -> list[ModelSummary]:
    store = request.app.state.metadata_store
    return [ModelSummary(**record.__dict__) for record in store.list_models()]


@router.post("/train", response_model=TrainResponse, status_code=202)
def train_model(body: TrainRequest, request: Request) -> TrainResponse:
    job_manager = request.app.state.job_manager
    job_id = job_manager.submit_training_job(
        base_model_id=body.base_model_id,
        speaker_name=body.speaker_name,
        input_jsonl=body.input_jsonl,
        num_epochs=body.num_epochs,
        batch_size=body.batch_size,
        lr=body.lr,
        tokenizer_model_path=body.tokenizer_model_path,
    )
    meta = job_manager.get_job(job_id)
    return TrainResponse(
        job_id=job_id,
        status=meta["status"],
        created_at=meta["created_at"],
    )