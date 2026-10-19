from pathlib import Path

import duckdb
import pandas as pd

from persona_mlops.warehouse import (
    load_training_data,
    save_predictions,
    transform_to_warehouse,
)


def test_transform_builds_silver_and_gold_star_schema(
    tmp_path: Path, persona_frame
):
    silver_path = tmp_path / "silver" / "personas.parquet"
    warehouse_path = tmp_path / "warehouse.duckdb"
    source = persona_frame.copy()
    source.loc[0, "professional_persona"] = "  padded source text  "
    source["age"] = source["age"].astype(object)
    source.loc[0, "age"] = "20"

    transform_to_warehouse(source, silver_path, warehouse_path)

    silver = pd.read_parquet(silver_path)
    assert silver.loc[0, "professional_persona"] == "padded source text"
    assert str(silver["age"].dtype) == "Int64"
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'gold'"
            ).fetchall()
        }
        assert {
            "dim_geography", "dim_demographics", "dim_occupation", "fact_persona"
        }.issubset(tables)
        row_count = connection.execute("SELECT count(*) FROM gold.fact_persona").fetchone()[0]
        distinct_ids = connection.execute(
            "SELECT count(DISTINCT persona_id) FROM gold.fact_persona"
        ).fetchone()[0]
        assert row_count == len(source)
        assert distinct_ids == len(source)

    features = load_training_data(warehouse_path)
    assert len(features) == len(source)
    assert set(features.columns) == {"uuid", "professional_persona", "region"}


def test_prediction_fact_persists_as_gold_table(tmp_path: Path, persona_frame):
    warehouse_path = tmp_path / "warehouse.duckdb"
    transform_to_warehouse(persona_frame, tmp_path / "silver.parquet", warehouse_path)
    save_predictions(warehouse_path, pd.DataFrame({
        "persona_id": ["persona-0000"],
        "predicted_region": ["North"],
        "cluster": [0],
    }))
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        row = connection.execute(
            "SELECT persona_id, predicted_region, cluster FROM gold.fact_model_predictions"
        ).fetchone()

    assert row == ("persona-0000", "North", 0)
