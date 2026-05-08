from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..metadata import MetadataStore, build_metadata_store
from ..model_manager import (
    GPUResourceBusyError,
    InferenceInProgressError,
    ModelManager,
    ModelManagerError,
    ModelTypeMismatchError,
    build_model_manager,
)
from ..runtime import RuntimeBaseline, build_runtime_baseline, ensure_phase0_layout
from .routes import models_router, tts_router, voices_router
from .schemas import ErrorResponse, HealthResponse


def create_app(
    baseline: RuntimeBaseline | None = None,
    *,
    metadata_store: MetadataStore | None = None,
    model_manager: ModelManager | None = None,
) -> FastAPI:
    active_baseline = baseline or build_runtime_baseline()
    ensure_phase0_layout(active_baseline)
    active_store = metadata_store or build_metadata_store(active_baseline)
    active_manager = model_manager or build_model_manager(active_baseline)

    app = FastAPI(title="Qwen3-TTS MVP API", version="0.1.0")
    app.state.baseline = active_baseline
    app.state.metadata_store = active_store
    app.state.model_manager = active_manager

    static_root = active_baseline.paths.repo_root / "static"
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")

    register_exception_handlers(app)
    app.include_router(models_router)
    app.include_router(voices_router)
    app.include_router(tts_router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        manager_status = active_manager.get_status()
        models = active_store.list_models()
        voices = active_store.list_voice_prompts()
        return HealthResponse(
            status="ok",
            db_path=str(active_baseline.paths.app_data_db),
            outputs_dir=str(active_baseline.paths.outputs_dir),
            gpu_busy=bool(manager_status["gpu_lock"] or manager_status["disk_gpu_lock"]),
            current_model_path=manager_status["current_model_path"],
            model_record_count=len(models),
            voice_prompt_count=len(voices),
        )

    return app


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GPUResourceBusyError)
    def handle_gpu_busy(_: Request, exc: GPUResourceBusyError) -> JSONResponse:
        return JSONResponse(status_code=503, content=ErrorResponse(detail=str(exc)).model_dump())

    @app.exception_handler(InferenceInProgressError)
    def handle_inference_busy(_: Request, exc: InferenceInProgressError) -> JSONResponse:
        return JSONResponse(status_code=503, content=ErrorResponse(detail=str(exc)).model_dump())

    @app.exception_handler(ModelTypeMismatchError)
    def handle_model_type_conflict(_: Request, exc: ModelTypeMismatchError) -> JSONResponse:
        return JSONResponse(status_code=409, content=ErrorResponse(detail=str(exc)).model_dump())

    @app.exception_handler(ModelManagerError)
    def handle_model_manager_error(_: Request, exc: ModelManagerError) -> JSONResponse:
        return JSONResponse(status_code=503, content=ErrorResponse(detail=str(exc)).model_dump())

    @app.exception_handler(ValueError)
    def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())

    @app.exception_handler(FileNotFoundError)
    def handle_not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=ErrorResponse(detail=str(exc)).model_dump())


app = create_app()