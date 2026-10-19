"""Extract a bounded sample from Hugging Face and load it as bronze Parquet."""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from persona_mlops.constants import DATASET_ID

LOGGER = logging.getLogger(__name__)


def ingest(sample_size: int, output: Path) -> pd.DataFrame:
    """Stream a deterministic prefix of the dataset and persist it as Parquet."""
    if sample_size < 100:
        raise ValueError("sample-size must be at least 100")
    LOGGER.info("Streaming up to %s records from %s", sample_size, DATASET_ID)
    rows = []
    stream = load_dataset(DATASET_ID, split="train", streaming=True)
    for row in stream:
        rows.append(row)
        if len(rows) >= sample_size:
            break
    if not rows:
        raise RuntimeError("The dataset stream returned no records")
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    LOGGER.info("Wrote %s rows to %s", len(frame), output)
    return frame
