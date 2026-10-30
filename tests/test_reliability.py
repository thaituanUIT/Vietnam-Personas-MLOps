"""Regression tests for warehouse load safety and operational visibility."""

import argparse
import json

import duckdb
import pytest

from persona_mlops.audit import record_run
from persona_mlops.cli import execute
from persona_mlops.quality import validate
from persona_mlops.warehouse import transform_to_warehouse


@pytest.mark.parametrize("age", [17, 91, 25.5, "unknown"])
def test_bad_age_reports_failure(persona_frame, tmp_path, age):
    persona_frame["age"] = persona_frame["age"].astype(object)
    persona_frame.loc[0, "age"] = age
    path = tmp_path / "quality.json"
    with pytest.raises(ValueError, match="Data quality"):
        validate(persona_frame, path)
    assert json.loads(path.read_text())["invalid_age_count"] == 1


def test_blank_required_field_fails(persona_frame, tmp_path):
    persona_frame.loc[0, "region"] = "   "
    with pytest.raises(ValueError, match="Data quality"):
        validate(persona_frame, tmp_path / "quality.json")


def test_repeat_load_preserves_keys_and_counts(persona_frame, tmp_path):
    warehouse = tmp_path / "warehouse.duckdb"
    silver = tmp_path / "silver/personas.parquet"
    transform_to_warehouse(persona_frame, silver, warehouse)
    with duckdb.connect(str(warehouse)) as db:
        before = db.sql("SELECT * FROM gold.dim_geography ORDER BY geography_key").fetchall()
    transform_to_warehouse(persona_frame.iloc[::-1], silver, warehouse)
    with duckdb.connect(str(warehouse)) as db:
        assert db.sql("SELECT * FROM gold.dim_geography ORDER BY geography_key").fetchall() == before
        assert db.sql("SELECT count(*) FROM gold.fact_persona").fetchone()[0] == len(persona_frame)
        assert db.sql("SELECT sum(persona_count) FROM gold.mart_audience_summary").fetchone()[0] == len(persona_frame)


def test_sql_failure_rolls_back_snapshot(persona_frame, tmp_path, monkeypatch):
    from persona_mlops import warehouse as module

    warehouse = tmp_path / "warehouse.duckdb"
    silver = tmp_path / "silver/personas.parquet"
    transform_to_warehouse(persona_frame, silver, warehouse)

    class BrokenModel:
        def joinpath(self, path):
            return self

        def read_text(self, **kwargs):
            return "DELETE FROM gold.fact_persona; SELECT * FROM nonexistent_table;"

    monkeypatch.setattr(module, "files", lambda package: BrokenModel())
    with pytest.raises(duckdb.Error):
        transform_to_warehouse(persona_frame.iloc[:10], silver, warehouse)
    with duckdb.connect(str(warehouse)) as db:
        assert db.sql("SELECT count(*) FROM gold.fact_persona").fetchone()[0] == len(persona_frame)
        assert db.sql("SELECT count(*) FROM silver.personas").fetchone()[0] == len(persona_frame)


def test_failure_is_audited_and_reraised(tmp_path):
    with pytest.raises(ValueError, match="bad input"):
        with record_run("transform", tmp_path):
            raise ValueError("bad input")
    records = list((tmp_path / "runs").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["status"] == "failed"


def test_offline_demo_exports_report(tmp_path):
    args = argparse.Namespace(
        command="demo", output_dir=tmp_path,
        data=tmp_path / "bronze/personas.parquet", warehouse=tmp_path / "warehouse.duckdb",
    )
    record = {}
    execute(args, record)
    assert record["rows"] == 120
    assert len(record["bronze_sha256"]) == 64
    assert (tmp_path / "reports/audience_summary.csv").exists()


def test_demo_cli_writes_success_audit(tmp_path, monkeypatch):
    from persona_mlops.cli import main

    monkeypatch.setattr("sys.argv", ["persona-pipeline", "demo", "--output-dir", str(tmp_path)])
    main()
    record = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert record["status"] == "succeeded"
    assert record["rows"] == 120


def test_foreign_keys_reject_orphan_fact(persona_frame, tmp_path):
    warehouse = tmp_path / "warehouse.duckdb"
    transform_to_warehouse(persona_frame, tmp_path / "silver/personas.parquet", warehouse)
    with duckdb.connect(str(warehouse)) as db:
        with pytest.raises(duckdb.ConstraintException):
            db.execute("""
                UPDATE gold.fact_persona SET geography_key = 'missing-dimension'
                WHERE persona_id = 'persona-0000'
            """)
