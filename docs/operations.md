# Running and presenting the project

## Local workflow

```bash
uv sync --extra dev
# Generated input, no dataset download
uv run persona-pipeline demo --output-dir data/demo
# Inspect the CSV in a spreadsheet
# Public source ETL (network required)
uv run persona-pipeline etl --sample-size 5000
# Optional ML stage
uv run persona-pipeline train --clusters 8
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

The Hugging Face input is a prefix, not a representative random sample. Its default branch can change. Preserve bronze files plus audit fingerprints to reproduce an existing sample; a future enhancement is pinning a dataset commit at ingestion. Credentials and paid APIs are not required for this public source. Dependencies are captured in `uv.lock`; `uv sync --frozen --extra dev` installs that resolution.

## Failure recovery

| Symptom | Evidence | Recovery |
| --- | --- | --- |
| Source unavailable | Failed run JSON and command error | Retry `ingest`/`etl`, or run `transform` on existing bronze |
| Contract violation | Failed quality JSON with invalid counts | Inspect bronze and fix the extraction/contract; retain the source for diagnosis |
| Warehouse SQL fails | Failed run JSON; transaction rolled back | Correct the SQL, rerun `transform` using the same bronze |
| CSV export fails | Failed run JSON; warehouse may already be committed | Correct the output path/permissions and rerun `report` |
| ML stage fails | Command error, run audit, MLflow run status when created | Fix the issue and rerun `train`; ETL data remains available |

Run commands sequentially. There is no orchestrator service or automatic retry policy. The CLI has explicit dependent stages; it can be invoked by a local scheduler if needed. Scheduled downloads should not be enabled by default for a static source.

## Automated checks

`uv run --extra dev pytest -q` covers mocked streaming, contract failures, dimensional relationships, load repetition, SQL rollback, audience count reconciliation, command audits, and downstream ML artifacts. The GitHub Actions workflow runs tests and the offline demo with Hugging Face offline flags. The workflow only runs after the repository is pushed to GitHub; it has not been executed remotely here. Check account usage limits before enabling hosted workflows; local pytest requires no hosted service.

## Interview demonstration (about five minutes)

1. State the scope: an analytical warehouse for sampled synthetic Vietnamese personas, with ML as a downstream consumer.
2. Run the offline demo and open the CSV; show the audience segmentation questions the mart answers.
3. Show the SQL fact grain, dimension keys, primary/foreign keys, joins, and window-function example.
4. Explain reruns, validation, transactions, and why a failed load should retain the last valid warehouse.
5. Run the tests and inspect a run JSON; discuss how failures can be diagnosed.
6. Explain limitations: sampled synthetic data, full snapshots, in-memory processing, no population-level accuracy claim.

## CV wording

Use a statement you can demonstrate, for example:

> Built a local Python and DuckDB ETL pipeline for synthetic Vietnamese persona data, with Parquet lake layers, a SQL star schema, transactional snapshot loads, automated data-quality checks, and CI tests; integrated scikit-learn models with MLflow tracking.

After processing real data, add measured record counts, run time, or peak memory together with the machine and sample size. Do not claim production deployment, incremental processing, business impact, or model accuracy unless you have evidence. The generated offline demo is intentionally simple and produces no meaningful benchmark accuracy.
