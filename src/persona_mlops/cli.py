"""Command-line orchestration for the local ETL and ML pipeline."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from persona_mlops.ingestion import ingest
from persona_mlops.modeling import train
from persona_mlops.quality import validate
from persona_mlops.warehouse import load_training_data, transform_to_warehouse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Vietnamese persona ML pipeline")
    parser.add_argument("command", choices=["run", "ingest", "transform", "train"])
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--data", type=Path, default=Path("data/bronze/personas.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse.duckdb"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("mlruns"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command in {"run", "ingest"}:
        frame = ingest(args.sample_size, args.data)
    else:
        if not args.data.exists():
            parser.error(f"Input Parquet not found: {args.data}; run ingest first")
        frame = pd.read_parquet(args.data)
    if args.command == "ingest":
        return

    validate(frame, args.output_dir / "reports/quality_report.json")
    if args.command in {"run", "transform"}:
        transform_to_warehouse(
            frame, args.output_dir / "silver/personas.parquet", args.warehouse
        )
    if args.command in {"run", "train"}:
        if not args.warehouse.exists():
            parser.error(f"Warehouse not found: {args.warehouse}; run transform first")
        model_frame = load_training_data(args.warehouse)
        result = train(
            model_frame, args.output_dir / "gold", args.tracking_dir,
            args.warehouse, args.clusters,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
