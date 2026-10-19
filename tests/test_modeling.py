from pathlib import Path

import duckdb

from persona_mlops.modeling import train
from persona_mlops.warehouse import transform_to_warehouse


def test_train_logs_models_and_persists_predictions(tmp_path: Path, persona_frame):
    warehouse_path = tmp_path / "warehouse.duckdb"
    transform_to_warehouse(persona_frame, tmp_path / "silver.parquet", warehouse_path)
    from persona_mlops.warehouse import load_training_data

    training_frame = load_training_data(warehouse_path)
    result = train(
        training_frame,
        tmp_path / "gold",
        tmp_path / "mlruns",
        warehouse_path,
        clusters=3,
    )

    assert result["rows_used"] == len(persona_frame)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["cluster_count"] == 3
    assert result["mlflow_run_id"]
    assert (tmp_path / "mlruns" / "mlflow.db").is_file()
    assert (tmp_path / "gold" / "metrics.json").is_file()
    assert (tmp_path / "gold" / "scored_personas.parquet").is_file()
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        count = connection.execute(
            "SELECT count(*) FROM gold.fact_model_predictions"
        ).fetchone()[0]
    assert count == len(persona_frame)


def test_train_rejects_too_few_rows(tmp_path: Path):
    import pandas as pd
    import pytest

    frame = pd.DataFrame({
        "uuid": ["a", "b"],
        "professional_persona": ["text alpha", "text beta"],
        "region": ["North", "South"],
    })

    with pytest.raises(ValueError, match="at least 20"):
        train(frame, tmp_path / "gold", tmp_path / "mlruns", tmp_path / "warehouse.duckdb", 2)
