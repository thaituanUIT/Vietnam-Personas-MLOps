# Vietnamese Persona Data + MLOps

This portfolio project focuses on Python ETL, SQL dimensional modeling, data contracts, repeatable snapshot loads, and automated checks. ML classification and clustering are downstream consumers of the warehouse.

### Try the ETL offline

```bash
uv sync --extra dev
uv run persona-pipeline demo --output-dir data/demo
uv run pytest -q
```

The demo creates 120 invented records locally and produces a DuckDB warehouse and `data/demo/reports/audience_summary.csv`. Demo records are separate from the NVIDIA dataset and are only for demonstrating mechanics. They are not a model benchmark.

For a real source sample, run ETL without model training:

```bash
uv run persona-pipeline etl --sample-size 5000
uv run persona-pipeline report
```

See [the data model and design decisions](docs/data-model.md) and [the operations and interview guide](docs/operations.md).

A local-first data engineering and ML project built around NVIDIA's synthetic [Nemotron-Personas-Vietnam](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam) dataset. It demonstrates ETL, a Parquet data lake, DuckDB warehouse modeling, classification, clustering, and local MLflow tracking without a cloud account or paid service.

## Architecture

```text
Hugging Face streaming source
        ↓ extract/load
Bronze: data/bronze/personas.parquet (source-shaped Parquet)
        ↓ clean, normalize, validate
Silver: data/silver/personas.parquet
        ↓ dimensional modeling in DuckDB
Gold: data/warehouse.duckdb
  ├── dim_geography
  ├── dim_demographics
  ├── dim_occupation
  ├── fact_persona
  ├── mart_audience_summary (SQL view)
  └── fact_model_predictions
        ↓ analytical feature/label query
TF-IDF + LogisticRegression classifier and TF-IDF + MiniBatchKMeans clustering
        ↓
Local MLflow runs/models + scored Parquet
```

The Parquet folders are the local data lake; the DuckDB file is the embedded analytical warehouse. `fact_persona` is the central fact table, keyed to geography, demographic, and occupation dimensions. The classifier predicts the synthetic persona's `region` from `professional_persona`; clustering groups that text without labels. A held-out accuracy and macro F1 are logged for classification, and a cosine silhouette score is logged as a clustering diagnostic.

## Run locally

Requires Python 3.10+ and `uv`.

```bash
uv sync
uv run persona-pipeline run --sample-size 5000 --clusters 8
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Install the development extra and run the offline test suite with:

```bash
uv sync --extra dev
uv run pytest
```

Tests use generated fixtures and a temporary DuckDB/MLflow directory. They do not fetch the Hugging Face dataset.

The `run` command streams a deterministic prefix of the dataset, writes bronze Parquet, validates and cleans it into silver, builds the gold warehouse, trains both algorithms, and writes local MLflow and prediction artifacts. Open MLflow at `http://127.0.0.1:5000`.

Stages can also be run independently:

```bash
uv run persona-pipeline ingest --sample-size 5000
uv run persona-pipeline transform
uv run persona-pipeline train --clusters 8
```

`train` reads directly from an existing warehouse; it does not require the bronze file. Options `--data`, `--output-dir`, `--warehouse`, and `--tracking-dir` change local paths. Bronze and warehouse defaults follow `--output-dir`. Keep the sample size modest on a laptop; TF-IDF runs in memory.

Each command writes a JSON record under `<output-dir>/runs/`, including status, UTC timestamps, parameters, and errors. Commands that read or create bronze also record the row count and a SHA-256 fingerprint. Validation writes failure reports before raising an error. Gold SQL models live in `src/persona_mlops/sql/warehouse.sql` and run inside a DuckDB transaction. A successful snapshot reload invalidates previous model predictions; rerun `train` to generate predictions for that snapshot.

## Outputs

- `data/bronze/personas.parquet`: source-shaped sampled records
- `data/silver/personas.parquet`: trimmed text and normalized age field
- `data/warehouse.duckdb`: silver table and gold dimensions/facts
- `data/reports/quality_report.json`: schema, null, duplicate-ID, and target checks
- `data/reports/silver_quality_report.json`: validation after trimming and before warehouse load
- `data/reports/audience_summary.csv`: counts and average ages by region, zone, occupation
- `data/runs/*.json`: command audit records, including failed runs
- `data/gold/scored_personas.parquet`: predicted region and cluster per persona
- `data/gold/metrics.json`: classification and clustering metrics
- `mlruns/`: local MLflow parameters, metrics, models, and reports

The stream takes the first N rows, so it is deterministic but not statistically random. For stronger evaluation, create a fixed-seed random sample and split by UUID. Cluster IDs have no inherent meaning; inspect representative personas before naming clusters. Since the source is synthetic and uses assumptions for some demographic relationships, describe model behavior as results on this dataset, not as conclusions about real Vietnamese people.

## Dataset license

The Hugging Face card lists CC BY 4.0. Attribute NVIDIA and the dataset when redistributing derived material, and review the dataset card for full terms and limitations.
