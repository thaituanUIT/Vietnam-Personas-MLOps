"""Classification, clustering, MLflow tracking, and prediction artifacts."""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from persona_mlops.constants import DATASET_ID, TARGET_COLUMN, TEXT_COLUMN
from persona_mlops.warehouse import save_predictions

LOGGER = logging.getLogger(__name__)


def train(
    frame: pd.DataFrame, artifact_dir: Path, tracking_dir: Path,
    warehouse_path: Path, clusters: int,
) -> dict[str, Any]:
    clean = frame[["uuid", TEXT_COLUMN, TARGET_COLUMN]].dropna()
    clean = clean[clean[TEXT_COLUMN].str.strip().ne("")]
    counts = clean[TARGET_COLUMN].value_counts()
    clean = clean[clean[TARGET_COLUMN].isin(counts[counts >= 2].index)]
    if clean[TARGET_COLUMN].nunique() < 2 or len(clean) < 20:
        raise ValueError("Need at least 20 usable records and two target classes with 2+ rows each")

    x_train, x_test, y_train, y_test = train_test_split(
        clean[TEXT_COLUMN], clean[TARGET_COLUMN], test_size=0.2, random_state=42,
        stratify=clean[TARGET_COLUMN],
    )
    classifier = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))

    cluster_count = min(max(2, clusters), max(2, len(clean) - 1))
    cluster_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10_000, min_df=2, max_df=0.95)),
        ("model", MiniBatchKMeans(
            n_clusters=cluster_count, random_state=42, n_init=3, batch_size=512
        )),
    ])
    cluster_pipeline.fit(clean[TEXT_COLUMN])
    matrix = cluster_pipeline.named_steps["tfidf"].transform(clean[TEXT_COLUMN])
    labels = cluster_pipeline.named_steps["model"].labels_
    silhouette_sample = min(2_000, len(clean))
    silhouette = float(silhouette_score(
        matrix, labels, metric="cosine", sample_size=silhouette_sample, random_state=42
    )) if len(set(labels)) > 1 and silhouette_sample > cluster_count else None

    artifact_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("vietnam-persona-classification-clustering")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "dataset": DATASET_ID,
            "sample_rows": len(frame),
            "classifier": "TF-IDF + balanced LogisticRegression",
            "target": TARGET_COLUMN,
            "cluster_algorithm": "MiniBatchKMeans",
            "n_clusters": cluster_count,
            "random_state": 42,
        })
        mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1})
        if silhouette is not None:
            mlflow.log_metric("cluster_silhouette_cosine", silhouette)
        mlflow.sklearn.log_model(classifier, name="region-classifier")
        mlflow.sklearn.log_model(cluster_pipeline, name="persona-clusterer")
        mlflow.log_text(
            classification_report(y_test, predictions, zero_division=0),
            "classification_report.txt",
        )
        mlflow.log_dict(
            {str(key): int(count) for key, count in pd.Series(labels).value_counts().items()},
            "cluster_sizes.json",
        )
        mlflow.log_dict({"silhouette_cosine": silhouette}, "cluster_metrics.json")
        run_id = run.info.run_id

    scored = clean[["uuid"]].rename(columns={"uuid": "persona_id"}).copy()
    scored["cluster"] = labels
    scored["predicted_region"] = classifier.predict(clean[TEXT_COLUMN])
    scored.to_parquet(artifact_dir / "scored_personas.parquet", index=False)
    save_predictions(warehouse_path, scored)
    summary = {
        "rows_used": int(len(clean)), "accuracy": accuracy, "macro_f1": macro_f1,
        "cluster_count": cluster_count, "cluster_silhouette_cosine": silhouette,
        "mlflow_run_id": run_id,
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote model results to %s", artifact_dir)
    return summary
