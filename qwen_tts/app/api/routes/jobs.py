from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import JobStatusResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    job_manager = request.app.state.job_manager
    meta = job_manager.get_job(job_id)
    if meta is None:
        raise FileNotFoundError(f"Job not found: {job_id}")
    log_tail = job_manager.get_job_log_tail(job_id, lines=50)
    return JobStatusResponse(**meta, log_tail=log_tail)
