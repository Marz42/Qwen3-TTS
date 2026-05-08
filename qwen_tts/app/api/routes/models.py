from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import ModelSummary

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/list", response_model=list[ModelSummary])
def list_models(request: Request) -> list[ModelSummary]:
    store = request.app.state.metadata_store
    return [ModelSummary(**record.__dict__) for record in store.list_models()]