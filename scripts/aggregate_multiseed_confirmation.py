"""Resume DR-IFT menos DR-IF entre seeds sem pseudorreplicar imagens.

Cada `--pair` recebe SEED CSV_DRIF CSV_DRIFT. O resumo usa a seed como unidade
da variabilidade de treinamento; as imagens continuam pareadas apenas dentro de
cada seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ("dice", "iou", "precision", "recall", "balanced_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("SEED", "DRIF_CSV", "DRIFT_CSV"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis", choices=("selected_thresholds", "fixed_threshold_0_5"), required=True
    )
    return parser.parse_args()


def load(path: Path) -> tuple[list[str], dict[str, dict[str, float]], float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 294:
        raise ValueError(f"{path}: esperado 294 registros, obtido {len(raw)}")
    ids = [row["canonical_id"] for row in raw]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: IDs duplicados")
    thresholds = {float(row["threshold"]) for row in raw}
    if len(thresholds) != 1:
        raise ValueError(f"{path}: threshold inconsistente")
    return (
        ids,
        {row["canonical_id"]: {metric: float(row[metric]) for metric in METRICS} for row in raw},
        thresholds.pop(),
    )


def main() -> int:
    args = parse_args()
    if len(args.pair) < 2:
        raise ValueError("a confirmação multiseed exige pelo menos duas seeds")
    seeds = [int(item[0]) for item in args.pair]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds duplicadas")

    rows = []
    sources = {}
    for raw_seed, raw_drif, raw_drift in args.pair:
        seed = int(raw_seed)
        drif_path, drift_path = Path(raw_drif).resolve(), Path(raw_drift).resolve()
        drif_ids, drif, drif_threshold = load(drif_path)
        drift_ids, drift, drift_threshold = load(drift_path)
        if drif_ids != drift_ids:
            raise ValueError(f"seed {seed}: identidade/ordem das imagens difere")
        row: dict[str, int | float] = {
            "seed": seed,
            "drif_threshold": drif_threshold,
            "drift_threshold": drift_threshold,
        }
        for metric in METRICS:
            drif_mean = statistics.fmean(drif[item][metric] for item in drif_ids)
            drift_mean = statistics.fmean(drift[item][metric] for item in drif_ids)
            row[f"drif_{metric}"] = drif_mean
            row[f"drift_{metric}"] = drift_mean
            row[f"difference_{metric}"] = drift_mean - drif_mean
        rows.append(row)
        sources[str(seed)] = {"DR-IF": str(drif_path), "DR-IFT": str(drift_path)}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_metrics = {}
    for metric in METRICS:
        drif_values = [float(row[f"drif_{metric}"]) for row in rows]
        drift_values = [float(row[f"drift_{metric}"]) for row in rows]
        differences = [float(row[f"difference_{metric}"]) for row in rows]
        summary_metrics[metric] = {
            "DR-IF_mean_across_seeds": statistics.fmean(drif_values),
            "DR-IF_sample_sd_across_seeds": statistics.stdev(drif_values),
            "DR-IFT_mean_across_seeds": statistics.fmean(drift_values),
            "DR-IFT_sample_sd_across_seeds": statistics.stdev(drift_values),
            "paired_difference_mean_across_seeds": statistics.fmean(differences),
            "paired_difference_sample_sd_across_seeds": statistics.stdev(differences),
            "seeds_favoring_DR-IFT": sum(value > 0 for value in differences),
            "seeds_equal": sum(value == 0 for value in differences),
            "seeds_favoring_DR-IF": sum(value < 0 for value in differences),
        }

    summary = {
        "status": "completed",
        "partition": "validation",
        "analysis": args.analysis,
        "training_variability_unit": "seed",
        "image_pairing": "within_seed_only",
        "seeds": sorted(seeds),
        "seed_count": len(seeds),
        "inference_note": (
            "Mean and sample SD across seeds are descriptive; with three seeds, "
            "no asymptotic significance claim across training runs is made."
        ),
        "metrics": summary_metrics,
        "sources": sources,
        "protected_partitions_accessed": False,
    }
    (output_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
