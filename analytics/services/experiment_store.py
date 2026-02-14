from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from analytics.utils.config import get_model_artifact_dir


class ExperimentStore:
    def __init__(self) -> None:
        base_dir = Path(get_model_artifact_dir())
        self.runs_dir = base_dir / "experiments"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.runs_dir / "runs_index.jsonl"

    def record_run(
        self,
        *,
        task: str,
        metrics: dict[str, Any],
        data_summary: dict[str, Any] | None = None,
        statistical_summary: dict[str, Any] | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        run_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "run_id": run_id,
            "task": task,
            "created_at": created_at,
            "model_version": model_version,
            "metrics": metrics,
            "data_summary": data_summary or {},
            "statistical_summary": statistical_summary or {},
        }

        run_file = self.runs_dir / f"{created_at.replace(':', '-')}_{task}_{run_id}.json"
        run_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        index_entry = {
            "run_id": run_id,
            "task": task,
            "created_at": created_at,
            "model_version": model_version,
            "run_file": run_file.name,
        }
        with self.index_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(index_entry) + "\n")

        return index_entry
