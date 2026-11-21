"""Resume um CSV por imagem em metricas macro e micro reproduziveis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.partition not in {"train", "validation"}:
        raise ValueError("A agregacao de desenvolvimento aceita somente train ou validation")
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))

    from src.tcc_segmentation.metric_aggregation import aggregate_global_pixel_metrics

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = aggregate_global_pixel_metrics(rows)
    summary.update(
        {
            "status": "completed",
            "partition": args.partition,
            "run_id": args.run_id,
            "source_csv": input_path.name,
            "protected_partitions_accessed": False,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
