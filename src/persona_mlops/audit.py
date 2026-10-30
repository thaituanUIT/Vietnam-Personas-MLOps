"""Persist success and failure metadata without changing stage error semantics."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@contextmanager
def record_run(command: str, output_dir: Path, **parameters):
    run_id = uuid4().hex
    record = {
        "run_id": run_id, "command": command, "parameters": parameters,
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
    }
    try:
        yield record
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    else:
        record["status"] = "succeeded"
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        directory = output_dir / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{run_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
