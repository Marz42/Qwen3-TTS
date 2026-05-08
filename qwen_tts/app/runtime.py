from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch


def dtype_from_name(name: str) -> torch.dtype:
    normalized = (name or "").strip().lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}. Use bfloat16/float16/float32.")


@dataclass(frozen=True)
class Phase0Paths:
    repo_root: Path
    data_dir: Path
    pretrained_models_dir: Path
    prompts_dir: Path
    jobs_dir: Path
    outputs_dir: Path
    app_data_db: Path

    @property
    def gpu_lock_path(self) -> Path:
        return self.jobs_dir / "gpu.lock"

    def as_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class RuntimeBaseline:
    paths: Phase0Paths
    device_map: str = "cuda:0"
    dtype_name: str = "float16"
    attn_implementation: Optional[str] = None
    min_training_samples: int = 5
    single_gpu_single_flight: bool = True
    max_concurrent_gpu_tasks: int = 1
    output_cleanup_max_files: int = 200
    output_cleanup_max_total_mb: int = 1024
    output_cleanup_delete_batch: int = 20
    default_job_type: Literal["inference", "training"] = "inference"

    @property
    def dtype(self) -> torch.dtype:
        return dtype_from_name(self.dtype_name)

    def require_local_model_path(self, model_path: str | Path) -> Path:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.is_absolute():
            raise ValueError(f"Model path must be absolute: {model_path}")
        return resolved

    def resolve_pretrained_model(self, model_name: str) -> Path:
        return self.require_local_model_path(self.paths.pretrained_models_dir / model_name)

    def validate_concurrency_limit(self, requested_limit: int) -> int:
        if requested_limit < 1:
            raise ValueError("Concurrency limit must be >= 1.")
        if self.single_gpu_single_flight and requested_limit != self.max_concurrent_gpu_tasks:
            raise ValueError(
                "Phase 0 baseline only supports single-flight GPU execution; use concurrency=1."
            )
        return requested_limit

    def model_load_kwargs(self) -> Dict[str, Any]:
        return {
            "device_map": self.device_map,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
        }

    @property
    def output_cleanup_max_total_bytes(self) -> int:
        return int(self.output_cleanup_max_total_mb * 1024 * 1024)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "paths": self.paths.as_dict(),
            "device_map": self.device_map,
            "dtype_name": self.dtype_name,
            "attn_implementation": self.attn_implementation,
            "model_load_kwargs": {
                "device_map": self.device_map,
                "dtype": str(self.dtype),
                "attn_implementation": self.attn_implementation,
            },
            "min_training_samples": self.min_training_samples,
            "single_gpu_single_flight": self.single_gpu_single_flight,
            "max_concurrent_gpu_tasks": self.max_concurrent_gpu_tasks,
            "output_cleanup_max_files": self.output_cleanup_max_files,
            "output_cleanup_max_total_mb": self.output_cleanup_max_total_mb,
            "output_cleanup_delete_batch": self.output_cleanup_delete_batch,
            "default_job_type": self.default_job_type,
        }


@dataclass(frozen=True)
class JobLockRecord:
    job_id: str
    pid: int
    created_at: str
    type: Literal["inference", "training"]

    @classmethod
    def create(cls, job_id: str, pid: int, job_type: Literal["inference", "training"]) -> "JobLockRecord":
        return cls(
            job_id=job_id,
            pid=pid,
            created_at=datetime.now(timezone.utc).isoformat(),
            type=job_type,
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_runtime_baseline(repo_root: str | Path | None = None) -> RuntimeBaseline:
    root = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    paths = Phase0Paths(
        repo_root=root,
        data_dir=root / "data",
        pretrained_models_dir=root / "data" / "pretrained_models",
        prompts_dir=root / "data" / "prompts",
        jobs_dir=root / "data" / "jobs",
        outputs_dir=root / "static" / "outputs",
        app_data_db=root / "data" / "app_data.db",
    )
    return RuntimeBaseline(paths=paths)


def ensure_phase0_layout(baseline: RuntimeBaseline) -> Dict[str, Path]:
    created_paths = {}
    for path in (
        baseline.paths.data_dir,
        baseline.paths.pretrained_models_dir,
        baseline.paths.prompts_dir,
        baseline.paths.jobs_dir,
        baseline.paths.outputs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
        created_paths[path.name] = path
    return created_paths