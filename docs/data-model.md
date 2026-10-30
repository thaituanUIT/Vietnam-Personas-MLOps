# Data model and decisions

## Business questions

An analyst can use the warehouse to ask how the sampled synthetic personas are distributed by region, occupation, education, and urban/rural zone. The summary mart provides a CSV for spreadsheet analysis. It describes the loaded sample, not actual customer demand or Vietnam's population.

## Grain and relationships

| Model | Grain / key | Purpose |
| --- | --- | --- |
| `silver.personas` | One validated source record per UUID | Source-shaped, cleaned analytical input |
| `gold.dim_geography` | One country + region + zone combination | Geographic slicing |
| `gold.dim_demographics` | One sex + age + marital status + education combination | Demographic slicing |
| `gold.dim_occupation` | One occupation | Occupation slicing |
| `gold.fact_persona` | One persona UUID in the current snapshot | Factless snapshot fact linking all dimensions and narrative attributes |
| `gold.mart_audience_summary` | One region + zone + occupation combination | Persona count and average age |
| `gold.fact_model_predictions` | One usable persona from the latest training run | Region prediction and cluster ID |

The central fact has a UUID primary key and non-null foreign keys enforced by DuckDB. Narrative attributes remain in the fact for convenient text analysis; a separate text table would be appropriate if multiple narratives or revisions per persona were introduced. The demographic dimension is a small-project convenience; at larger scale age could move into a separate band dimension to limit dimension growth.

Dimension keys use MD5 over JSON-encoded natural-key fields. JSON preserves field boundaries; keys are stable across row order and snapshot reruns. These hashes are identifiers, not security features. The geography available as structured columns is region/zone/country; province is not inferred from free text.

## Load contract

1. Extract at most N rows from the public training stream into source-shaped bronze Parquet.
2. Record a bronze SHA-256 fingerprint and row count in the command audit log.
3. Validate required columns, strings, nulls, blanks, unique UUIDs, and integer ages 18–90. This is the source-specific age range and must be reviewed if the dataset changes.
4. Trim text fields and coerce age to a nullable integer; validate again after trimming to detect whitespace-induced duplicate keys.
5. Replace silver and gold tables in one DuckDB transaction; check that fact row counts match silver before committing.
6. Persist silver Parquet and export the summary mart as CSV.

The load uses complete snapshot replacement. Repeating the same input preserves keys and row counts without appending duplicate facts. It is not an incremental or change-data-capture pipeline, and it does not maintain slowly changing dimension history. The warehouse transaction does not include filesystem exports: if an export fails after commit, the warehouse is current but that export may be stale; rerun `transform` or `report` as appropriate.

A successful reload drops previous prediction results because they may refer to a different snapshot. `train` rebuilds them. Failed warehouse SQL rolls back to the last valid warehouse state. Only one process should write to the warehouse at a time.

## Example analytical SQL

```sql
-- Sample coverage by geographic segment
SELECT region, zone, sum(persona_count) AS personas
FROM gold.mart_audience_summary
GROUP BY region, zone
ORDER BY personas DESC;

-- Rank occupations within each region (demonstrates a window function)
WITH counts AS (
    SELECT region, occupation, sum(persona_count) AS personas
    FROM gold.mart_audience_summary
    GROUP BY region, occupation
)
SELECT *, dense_rank() OVER (
    PARTITION BY region ORDER BY personas DESC
) AS occupation_rank
FROM counts;

-- Educational distribution in the loaded snapshot
SELECT d.education_level, count(*) AS personas
FROM gold.fact_persona f
JOIN gold.dim_demographics d USING (demographics_key)
GROUP BY d.education_level
ORDER BY personas DESC;
```

## Why this stack

Parquet provides portable columnar files and DuckDB runs analytical SQL without managing a database server. Python coordinates extraction, validation, and loading. Classical ML consumes warehouse data on CPU. This keeps local operation simple. The entire sampled DataFrame is held in memory, so larger-scale processing would need chunked ingestion and SQL transformations directly over partitioned Parquet.
