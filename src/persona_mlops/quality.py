"""Schema and data-quality checks for source and silver records."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from persona_mlops.constants import REQUIRED_COLUMNS, TARGET_COLUMN, TEXT_COLUMN

LOGGER = logging.getLogger(__name__)


def validate(frame: pd.DataFrame, report_path: Path) -> dict[str, Any]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"status": "failed", "missing_columns": missing_columns}), encoding="utf-8")
        raise ValueError(f"Required columns are missing: {missing_columns}")
    age = pd.to_numeric(frame["age"], errors="coerce")
    quality = {
        "rows": int(len(frame)),
        "duplicate_uuid_count": int(frame["uuid"].duplicated().sum()),
        "null_counts": {
            column: int(count)
            for column, count in frame[sorted(REQUIRED_COLUMNS)].isna().sum().items()
        },
        "blank_counts": {
            column: int(frame[column].map(lambda value: isinstance(value, str) and not value.strip()).sum())
            for column in sorted(REQUIRED_COLUMNS - {"age"})
        },
        "non_string_counts": {
            column: int(frame[column].map(lambda value: not isinstance(value, str)).sum())
            for column in sorted(REQUIRED_COLUMNS - {"age"})
        },
        "invalid_age_count": int((age.isna() | ~age.between(18, 90) | age.mod(1).ne(0)).sum()),
        "empty_text_count": int(frame[TEXT_COLUMN].map(
            lambda value: isinstance(value, str) and not value.strip()
        ).sum()),
        "target_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts(dropna=False).items()
        },
    }
    failed = (
        frame.empty or any(quality["null_counts"].values())
        or any(quality["blank_counts"].values()) or any(quality["non_string_counts"].values())
        or quality["duplicate_uuid_count"] or quality["invalid_age_count"]
    )
    quality["status"] = "failed" if failed else "passed"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Quality report written to %s", report_path)
    if failed:
        raise ValueError(f"Data quality checks failed: {quality}")
    return quality
