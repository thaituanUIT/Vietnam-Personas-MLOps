"""Silver transformations and transactional gold warehouse loads."""

import logging
from importlib.resources import files
from pathlib import Path

import duckdb
import pandas as pd

from persona_mlops.quality import validate

LOGGER = logging.getLogger(__name__)


def transform_to_warehouse(
    frame: pd.DataFrame, silver_path: Path, warehouse_path: Path
) -> None:
    """Validate a complete snapshot, then replace warehouse tables atomically."""
    silver = frame.copy()
    for column in silver.select_dtypes(include=["object", "string"]).columns:
        silver[column] = silver[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    report_path = silver_path.parent.parent / "reports/silver_quality_report.json"
    validate(silver, report_path)
    silver["age"] = pd.to_numeric(silver["age"]).astype("Int64")
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    sql = files("persona_mlops").joinpath("sql/warehouse.sql").read_text(encoding="utf-8")
    with duckdb.connect(str(warehouse_path)) as connection:
        connection.begin()
        try:
            connection.execute("CREATE SCHEMA IF NOT EXISTS silver")
            connection.execute("CREATE SCHEMA IF NOT EXISTS gold")
            connection.register("cleaned_snapshot", silver)
            connection.execute("CREATE OR REPLACE TABLE silver.personas AS SELECT * FROM cleaned_snapshot")
            connection.execute(sql)
            count = connection.execute("SELECT count(*) FROM gold.fact_persona").fetchone()[0]
            if count != len(silver):
                raise ValueError(f"Warehouse row conservation failed: {count} != {len(silver)}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(silver_path, index=False)
    LOGGER.info("Loaded %s personas into %s", len(silver), warehouse_path)


def load_training_data(warehouse_path: Path) -> pd.DataFrame:
    """Read the ML feature and target view from the gold warehouse."""
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        return connection.execute("""
            SELECT f.persona_id AS uuid, f.professional_persona, g.region
            FROM gold.fact_persona f
            JOIN gold.dim_geography g USING (geography_key)
            ORDER BY f.persona_id
        """).df()


def save_predictions(warehouse_path: Path, predictions: pd.DataFrame) -> None:
    """Replace the latest predictions for the current snapshot."""
    with duckdb.connect(str(warehouse_path)) as connection:
        connection.register("scored_frame", predictions)
        connection.execute("""
            CREATE OR REPLACE TABLE gold.fact_model_predictions AS
            SELECT persona_id, predicted_region, cluster,
                   current_timestamp AS scored_at
            FROM scored_frame
        """)


def export_report(warehouse_path: Path, output: Path) -> pd.DataFrame:
    """Export the audience analytics mart for spreadsheet/BI consumption."""
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        report = connection.execute("""
            SELECT * FROM gold.mart_audience_summary
            ORDER BY persona_count DESC, region, zone, occupation
        """).df()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    return report
