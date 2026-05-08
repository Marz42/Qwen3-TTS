from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from .runtime import RuntimeBaseline, build_runtime_baseline, ensure_phase0_layout

ModelType = Literal["base", "custom_voice", "voice_design"]


@dataclass(frozen=True)
class ModelRecord:
    id: int
    name: str
    type: ModelType
    path: str
    speaker: Optional[str]


@dataclass(frozen=True)
class VoicePromptRecord:
    id: int
    name: str
    ref_text: Optional[str]
    prompt_file: str


class MetadataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL CHECK(type IN ('base', 'custom_voice', 'voice_design')),
                    path TEXT NOT NULL UNIQUE,
                    speaker TEXT
                );

                CREATE TABLE IF NOT EXISTS voice_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    ref_text TEXT,
                    prompt_file TEXT NOT NULL UNIQUE
                );
                """
            )

    def list_models(self) -> list[ModelRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, type, path, speaker FROM models ORDER BY id ASC"
            ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def get_model(self, model_id: int) -> Optional[ModelRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, type, path, speaker FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
        return self._row_to_model(row) if row is not None else None

    def register_model(
        self,
        *,
        name: str,
        model_type: ModelType,
        path: str | Path,
        speaker: Optional[str] = None,
    ) -> ModelRecord:
        normalized_path = self._require_existing_path(path, expected_kind="dir")
        self._validate_model_type(model_type)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, name, type, path, speaker FROM models WHERE path = ?",
                (str(normalized_path),),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    "INSERT INTO models(name, type, path, speaker) VALUES (?, ?, ?, ?)",
                    (name, model_type, str(normalized_path), speaker),
                )
                model_id = int(cursor.lastrowid)
            else:
                model_id = int(existing["id"])
                connection.execute(
                    "UPDATE models SET name = ?, type = ?, speaker = ? WHERE id = ?",
                    (name, model_type, speaker, model_id),
                )
            row = connection.execute(
                "SELECT id, name, type, path, speaker FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
        return self._row_to_model(row)

    def list_voice_prompts(self) -> list[VoicePromptRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, ref_text, prompt_file FROM voice_prompts ORDER BY id ASC"
            ).fetchall()
        return [self._row_to_voice_prompt(row) for row in rows]

    def get_voice_prompt(self, prompt_id: int) -> Optional[VoicePromptRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, ref_text, prompt_file FROM voice_prompts WHERE id = ?",
                (prompt_id,),
            ).fetchone()
        return self._row_to_voice_prompt(row) if row is not None else None

    def register_voice_prompt(
        self,
        *,
        name: str,
        prompt_file: str | Path,
        ref_text: Optional[str] = None,
    ) -> VoicePromptRecord:
        normalized_path = self._require_existing_path(prompt_file, expected_kind="file")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, name, ref_text, prompt_file FROM voice_prompts WHERE prompt_file = ?",
                (str(normalized_path),),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    "INSERT INTO voice_prompts(name, ref_text, prompt_file) VALUES (?, ?, ?)",
                    (name, ref_text, str(normalized_path)),
                )
                prompt_id = int(cursor.lastrowid)
            else:
                prompt_id = int(existing["id"])
                connection.execute(
                    "UPDATE voice_prompts SET name = ?, ref_text = ? WHERE id = ?",
                    (name, ref_text, prompt_id),
                )
            row = connection.execute(
                "SELECT id, name, ref_text, prompt_file FROM voice_prompts WHERE id = ?",
                (prompt_id,),
            ).fetchone()
        return self._row_to_voice_prompt(row)

    def list_table_columns(self, table_name: Literal["models", "voice_prompts"]) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [str(row["name"]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _validate_model_type(model_type: str) -> None:
        if model_type not in ("base", "custom_voice", "voice_design"):
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _require_existing_path(path: str | Path, *, expected_kind: Literal["dir", "file"]) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_absolute():
            raise ValueError(f"Path must be absolute: {path}")
        if expected_kind == "dir" and not resolved.is_dir():
            raise ValueError(f"Model path must be an existing directory: {resolved}")
        if expected_kind == "file" and not resolved.is_file():
            raise ValueError(f"Prompt path must be an existing file: {resolved}")
        return resolved

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            type=str(row["type"]),
            path=str(row["path"]),
            speaker=row["speaker"],
        )

    @staticmethod
    def _row_to_voice_prompt(row: sqlite3.Row) -> VoicePromptRecord:
        return VoicePromptRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            ref_text=row["ref_text"],
            prompt_file=str(row["prompt_file"]),
        )


def build_metadata_store(baseline: RuntimeBaseline | None = None) -> MetadataStore:
    active_baseline = baseline or build_runtime_baseline()
    ensure_phase0_layout(active_baseline)
    store = MetadataStore(active_baseline.paths.app_data_db)
    store.initialize()
    return store


def model_records_to_paths(records: Iterable[ModelRecord]) -> list[str]:
    return [record.path for record in records]