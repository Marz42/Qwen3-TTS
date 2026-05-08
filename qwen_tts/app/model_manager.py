from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

import torch

from .. import Qwen3TTSModel
from .metadata import ModelRecord
from .runtime import RuntimeBaseline, build_runtime_baseline, ensure_phase0_layout

ManagedModelType = Literal["base", "custom_voice", "voice_design"]
InferenceResult = TypeVar("InferenceResult")
ModelLoader = Callable[..., Qwen3TTSModel]


class ModelManagerError(RuntimeError):
    pass


class GPUResourceBusyError(ModelManagerError):
    pass


class InferenceInProgressError(ModelManagerError):
    pass


class ModelTypeMismatchError(ModelManagerError):
    pass


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        baseline: RuntimeBaseline | None = None,
        *,
        loader: ModelLoader | None = None,
    ):
        self.baseline = baseline or build_runtime_baseline()
        ensure_phase0_layout(self.baseline)

        self._loader = loader or Qwen3TTSModel.from_pretrained
        self._state_lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self._inference_owner_thread_id: Optional[int] = None

        self.current_model_path: Optional[str] = None
        self.current_model_type: Optional[ManagedModelType] = None
        self.model: Optional[Qwen3TTSModel] = None
        self.gpu_lock = False

    @classmethod
    def get_instance(
        cls,
        baseline: RuntimeBaseline | None = None,
        *,
        loader: ModelLoader | None = None,
        force_new: bool = False,
    ) -> "ModelManager":
        with cls._instance_lock:
            if force_new and cls._instance is not None:
                cls._instance.unload_model()
                cls._instance = None
            if cls._instance is None:
                cls._instance = cls(baseline=baseline, loader=loader)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.unload_model()
            cls._instance = None

    def set_gpu_lock(self, active: bool) -> None:
        with self._state_lock:
            self.gpu_lock = active

    def has_disk_gpu_lock(self) -> bool:
        return self.baseline.paths.gpu_lock_path.exists()

    def is_gpu_busy(self) -> bool:
        with self._state_lock:
            return self.gpu_lock or self.has_disk_gpu_lock()

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "current_model_path": self.current_model_path,
                "current_model_type": self.current_model_type,
                "has_model": self.model is not None,
                "gpu_lock": self.gpu_lock,
                "disk_gpu_lock": self.has_disk_gpu_lock(),
                "inference_active": self.inference_lock.locked(),
            }

    def load_registered_model(self, record: ModelRecord) -> Qwen3TTSModel:
        return self.load_model(record.path, expected_model_type=record.type)

    def load_model(
        self,
        model_path: str | Path,
        *,
        expected_model_type: ManagedModelType | None = None,
    ) -> Qwen3TTSModel:
        resolved_path = self.baseline.require_local_model_path(model_path)
        if not resolved_path.is_dir():
            raise ValueError(f"Model path must be an existing directory: {resolved_path}")

        with self._state_lock:
            self._raise_if_gpu_busy()
            self._raise_if_inference_owned_by_other_thread()

            if self.current_model_path == str(resolved_path) and self.model is not None:
                self._validate_loaded_model_type(self.current_model_type, expected_model_type)
                return self.model

            self._unload_model_locked()
            loaded_model = self._loader(str(resolved_path), **self.baseline.model_load_kwargs())
            loaded_model_type = self._extract_model_type(loaded_model)
            self._validate_loaded_model_type(loaded_model_type, expected_model_type)

            self.model = loaded_model
            self.current_model_path = str(resolved_path)
            self.current_model_type = loaded_model_type or expected_model_type
            return loaded_model

    def unload_model(self) -> None:
        with self._state_lock:
            self._raise_if_inference_owned_by_other_thread()
            self._unload_model_locked()

    def run_inference(
        self,
        model_path: str | Path,
        inference_fn: Callable[[Qwen3TTSModel], InferenceResult],
        *,
        expected_model_type: ManagedModelType | None = None,
        blocking: bool = False,
    ) -> InferenceResult:
        if not self._acquire_inference_lock(blocking=blocking):
            raise InferenceInProgressError(
                "Another inference request is already using the GPU; Phase 2 only allows one active inference."
            )

        try:
            model = self.load_model(model_path, expected_model_type=expected_model_type)
            return self._materialize_cpu_outputs(inference_fn(model))
        finally:
            self._release_inference_lock()

    def _acquire_inference_lock(self, *, blocking: bool) -> bool:
        acquired = self.inference_lock.acquire(blocking=blocking)
        if acquired:
            self._inference_owner_thread_id = threading.get_ident()
        return acquired

    def _release_inference_lock(self) -> None:
        self._inference_owner_thread_id = None
        self.inference_lock.release()

    def _raise_if_gpu_busy(self) -> None:
        if self.gpu_lock:
            raise GPUResourceBusyError("GPU access is locked by the local manager state.")
        if self.has_disk_gpu_lock():
            raise GPUResourceBusyError(
                f"GPU access is locked by an on-disk job lock: {self.baseline.paths.gpu_lock_path}"
            )

    def _raise_if_inference_owned_by_other_thread(self) -> None:
        if self.inference_lock.locked() and self._inference_owner_thread_id != threading.get_ident():
            raise InferenceInProgressError(
                "Cannot change models while another thread is inside the inference section."
            )

    def _unload_model_locked(self) -> None:
        if self.model is None:
            self.current_model_path = None
            self.current_model_type = None
            return

        model_to_free = self.model
        self.model = None
        self.current_model_path = None
        self.current_model_type = None
        del model_to_free
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _extract_model_type(model: Qwen3TTSModel) -> Optional[ManagedModelType]:
        candidate = getattr(model.model, "tts_model_type", None)
        if candidate is None:
            candidate = getattr(getattr(model.model, "config", None), "tts_model_type", None)
        return candidate

    @staticmethod
    def _validate_loaded_model_type(
        loaded_model_type: ManagedModelType | None,
        expected_model_type: ManagedModelType | None,
    ) -> None:
        if expected_model_type is None or loaded_model_type is None:
            return
        if loaded_model_type != expected_model_type:
            raise ModelTypeMismatchError(
                f"Loaded model type {loaded_model_type} does not match expected type {expected_model_type}."
            )

    @classmethod
    def _materialize_cpu_outputs(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, tuple):
            return tuple(cls._materialize_cpu_outputs(item) for item in value)
        if isinstance(value, list):
            return [cls._materialize_cpu_outputs(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._materialize_cpu_outputs(item) for key, item in value.items()}
        return value


def build_model_manager(
    baseline: RuntimeBaseline | None = None,
    *,
    loader: ModelLoader | None = None,
    force_new: bool = False,
) -> ModelManager:
    return ModelManager.get_instance(baseline=baseline, loader=loader, force_new=force_new)