from pathlib import Path

import pandas as pd
import pytest

from persona_mlops import ingestion


def test_ingest_streams_only_requested_sample(monkeypatch, tmp_path: Path, persona_frame):
    calls = {}
    records = persona_frame.to_dict(orient="records") * 2

    def fake_load_dataset(dataset_id, *, split, streaming):
        calls.update(dataset_id=dataset_id, split=split, streaming=streaming)
        return iter(records)

    monkeypatch.setattr(ingestion, "load_dataset", fake_load_dataset)
    output = tmp_path / "bronze" / "sample.parquet"

    result = ingestion.ingest(100, output)

    assert len(result) == 100
    assert output.is_file()
    assert len(pd.read_parquet(output)) == 100
    assert calls == {
        "dataset_id": "nvidia/Nemotron-Personas-Vietnam",
        "split": "train",
        "streaming": True,
    }


def test_ingest_rejects_too_small_sample(tmp_path: Path):
    with pytest.raises(ValueError, match="at least 100"):
        ingestion.ingest(99, tmp_path / "bronze.parquet")
