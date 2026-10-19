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
        raise ValueError(f"Required columns are missing: {missing_columns}")
    quality = {
        "rows": int(len(frame)),
        "duplicate_uuid_count": int(frame["uuid"].duplicated().sum()),
        "null_counts": {
            column: int(count)
            for column, count in frame[list(REQUIRED_COLUMNS)].isna().sum().items()
        },
        "empty_text_count": int(frame[TEXT_COLUMN].fillna("").str.strip().eq("").sum()),
        "target_distribution": {
            str(label): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts(dropna=False).items()
        },
    }
    if any(quality["null_counts"].values()) or quality["duplicate_uuid_count"]:
        raise ValueError(f"Data quality checks failed: {quality}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Quality report written to %s", report_path)
    return quality
