"""Silver transformations and gold dimensional warehouse models."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

LOGGER = logging.getLogger(__name__)


def transform_to_warehouse(
    frame: pd.DataFrame, silver_path: Path, warehouse_path: Path
) -> None:
    """Clean bronze rows, persist silver Parquet, and build a gold star schema."""
    silver = frame.copy()
    for column in silver.select_dtypes(include=["object", "string"]).columns:
        silver[column] = silver[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    silver["age"] = pd.to_numeric(silver["age"], errors="coerce").astype("Int64")
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(silver_path, index=False)

    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse_path)) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS silver")
        connection.execute("CREATE SCHEMA IF NOT EXISTS gold")
        connection.execute(
            "CREATE OR REPLACE TABLE silver.personas AS SELECT * FROM read_parquet(?)",
            [str(silver_path)],
        )
        connection.execute("""
            CREATE OR REPLACE TABLE gold.dim_geography AS
            SELECT row_number() OVER (ORDER BY country, region, zone) AS geography_key,
                   country, region, zone
            FROM (SELECT DISTINCT country, region, zone FROM silver.personas)
        """)
        connection.execute("""
            CREATE OR REPLACE TABLE gold.dim_demographics AS
            SELECT row_number() OVER (
                       ORDER BY sex, age, marital_status, education_level
                   ) AS demographics_key,
                   sex, age, marital_status, education_level
            FROM (
                SELECT DISTINCT sex, age, marital_status, education_level
                FROM silver.personas
            )
        """)
        connection.execute("""
            CREATE OR REPLACE TABLE gold.dim_occupation AS
            SELECT row_number() OVER (ORDER BY occupation) AS occupation_key, occupation
            FROM (SELECT DISTINCT occupation FROM silver.personas)
        """)
        connection.execute("""
            CREATE OR REPLACE TABLE gold.fact_persona AS
            SELECT p.uuid AS persona_id, g.geography_key, d.demographics_key,
                   o.occupation_key, p.professional_persona, p.sports_persona,
                   p.arts_persona, p.travel_persona, p.culinary_persona, p.persona,
                   p.cultural_background, p.skills_and_expertise,
                   p.hobbies_and_interests, p.career_goals_and_ambitions
            FROM silver.personas p
            JOIN gold.dim_geography g USING (country, region, zone)
            JOIN gold.dim_demographics d USING (sex, age, marital_status, education_level)
            JOIN gold.dim_occupation o USING (occupation)
        """)
    LOGGER.info("Built silver Parquet and gold star schema in %s", warehouse_path)


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
    """Replace the gold prediction fact with the latest local model output."""
    with duckdb.connect(str(warehouse_path)) as connection:
        connection.register("scored_frame", predictions)
        connection.execute("""
            CREATE OR REPLACE TABLE gold.fact_model_predictions AS
            SELECT persona_id, predicted_region, cluster,
                   current_timestamp AS scored_at
            FROM scored_frame
        """)
