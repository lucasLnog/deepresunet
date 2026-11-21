"""Audita as seis execuções DR-IF/DR-IFT usadas na confirmação multiseed.

O script lê somente artefatos de treino/validação já exportados. Não acessa dados,
manifestos de teste ou FUSeg.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


ARMS = {"DR-IF": "fixed", "DR-IFT": "progressive"}
EXPECTED_SEEDS = (42, 1337, 2026)
REQUIRED = (
    "config_used.yaml",
    "preflight.json",
    "training_summary.json",
    "augmentation_safety.json",
    "history.csv",
    "threshold_curve.csv",
    "validation_per_image_at_0_5.csv",
    "validation_per_image_selected.csv",
    "validation_metrics_at_0_5.json",
    "validation_metrics_selected.json",
    "validation_by_area_bin.csv",
    "threshold_selection.json",
    "final_summary.json",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", nargs=3, action="append", metavar=("SEED", "ARM", "DIR"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    errors, warnings, runs = [], [], {}
    for raw_seed, arm, raw_dir in args.run:
        seed, directory = int(raw_seed), Path(raw_dir).resolve()
        key = (seed, arm)
        if key in runs:
            errors.append(f"execução duplicada: {key}")
            continue
        if arm not in ARMS:
            errors.append(f"braço inesperado: {arm}")
            continue
        missing = [name for name in REQUIRED if not (directory / name).is_file()]
        if missing:
            errors.append(f"{key}: arquivos ausentes: {missing}")
            continue
        missing_checkpoints = [
            name
            for name in ("best_checkpoint.pt", "last_checkpoint.pt")
            if not (directory / name).is_file()
        ]
        if missing_checkpoints:
            if seed == 42:
                warnings.append(f"{key}: cópia histórica sem checkpoints: {missing_checkpoints}")
            else:
                errors.append(f"{key}: checkpoints ausentes: {missing_checkpoints}")
                continue

        config = yaml.safe_load((directory / "config_used.yaml").read_text(encoding="utf-8"))
        final = json.loads((directory / "final_summary.json").read_text(encoding="utf-8"))
        training = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
        safety = json.loads((directory / "augmentation_safety.json").read_text(encoding="utf-8"))
        threshold = json.loads((directory / "threshold_selection.json").read_text(encoding="utf-8"))
        history = read_csv(directory / "history.csv")
        selected = read_csv(directory / "validation_per_image_selected.csv")
        fixed = read_csv(directory / "validation_per_image_at_0_5.csv")

        checks = {
            "status_completed": final.get("status") == "completed",
            "identity": final.get("experiment_id") == arm and int(final.get("seed")) == seed,
            "strategy": final.get("fine_tuning_strategy") == ARMS[arm],
            "architecture": final.get("model_architecture") == "deep_res_unet_resnet34",
            "imagenet": final.get("encoder_weights") == "imagenet",
            "p0_a1": final.get("color_preprocessing") == "none"
            and final.get("augmentation") == "A1",
            "batch_four": int(training.get("selected_batch_size")) == 4,
            "stopped_early": bool(training.get("stopped_early")),
            "protected_false": not any(
                [
                    final.get("protected_partitions_accessed"),
                    training.get("protected_partitions_accessed"),
                    safety.get("protected_partitions_accessed"),
                    threshold.get("protected_partitions_accessed"),
                ]
            ),
            "partitions_train_validation_only": set(final.get("partitions_accessed", []))
            == {"train", "validation"},
            "validation_294": len(selected) == len(fixed) == 294,
            "unique_ids": len({r["canonical_id"] for r in selected}) == 294,
            "same_fixed_selected_ids": [r["canonical_id"] for r in selected]
            == [r["canonical_id"] for r in fixed],
            "history_matches_training": len(history) == int(training.get("epochs_executed")),
            "no_identity_fallback": int(safety.get("identity_fallbacks", -1)) == 0,
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(f"{key}: falhou {name}")

        selected_thresholds = {float(r["threshold"]) for r in selected}
        if len(selected_thresholds) != 1:
            errors.append(f"{key}: threshold selecionado não é constante")
        selected_threshold = (
            next(iter(selected_thresholds)) if len(selected_thresholds) == 1 else None
        )
        declared_threshold = float(final["selection"]["threshold"])
        if selected_threshold is not None and abs(selected_threshold - declared_threshold) > 1e-12:
            errors.append(f"{key}: threshold do CSV difere do resumo")

        runs[key] = {
            "directory": str(directory),
            "run_id": final["run_id"],
            "seed": seed,
            "arm": arm,
            "epochs": int(training["epochs_executed"]),
            "best_epoch": int(training["best_epoch"]),
            "threshold": declared_threshold,
            "dice_at_0_5": float(final["selection"]["metrics_at_0_5"]["macro"]["dice"]["mean"]),
            "dice_selected": float(final["selection"]["metrics_selected"]["macro"]["dice"]["mean"]),
            "empty_mask_rejections": int(safety["empty_mask_rejections"]),
            "samples_seen": int(safety["samples_seen"]),
            "selected_csv_sha256": sha256(directory / "validation_per_image_selected.csv"),
            "fixed_csv_sha256": sha256(directory / "validation_per_image_at_0_5.csv"),
            "ids": [r["canonical_id"] for r in selected],
            "history": history,
            "config": config,
        }

    expected = {(seed, arm) for seed in EXPECTED_SEEDS for arm in ARMS}
    if set(runs) != expected:
        errors.append(f"matriz incompleta: esperado {sorted(expected)}, recebido {sorted(runs)}")

    if runs:
        reference_ids = next(iter(runs.values()))["ids"]
        for key, run in runs.items():
            if run["ids"] != reference_ids:
                errors.append(f"{key}: ordem/identidade das imagens difere")

    pair_checks = {}
    history_fields = ("train_loss", "validation_loss", "validation_dice_at_0_5")
    for seed in EXPECTED_SEEDS:
        if (seed, "DR-IF") not in runs or (seed, "DR-IFT") not in runs:
            continue
        fixed_run, tuned_run = runs[(seed, "DR-IF")], runs[(seed, "DR-IFT")]
        first15_equal = len(fixed_run["history"]) >= 15 and len(tuned_run["history"]) >= 15
        if first15_equal:
            for index in range(15):
                for field in history_fields:
                    first15_equal &= (
                        fixed_run["history"][index][field] == tuned_run["history"][index][field]
                    )
        if not first15_equal:
            errors.append(f"seed {seed}: históricos não são idênticos nas épocas 1–15")
        cfg_a, cfg_b = fixed_run["config"], tuned_run["config"]
        randomness_equal = cfg_a.get("augmentation_randomness") == cfg_b.get(
            "augmentation_randomness"
        )
        if not randomness_equal:
            errors.append(f"seed {seed}: streams de augmentation diferem")
        pair_checks[str(seed)] = {
            "first_15_epochs_identical": first15_equal,
            "augmentation_randomness_equal": randomness_equal,
            "selected_dice_difference_DRIFT_minus_DRIF": tuned_run["dice_selected"]
            - fixed_run["dice_selected"],
            "fixed_0_5_dice_difference_DRIFT_minus_DRIF": tuned_run["dice_at_0_5"]
            - fixed_run["dice_at_0_5"],
        }

    serializable_runs = {
        f"{seed}_{arm}": {k: v for k, v in run.items() if k not in {"ids", "history", "config"}}
        for (seed, arm), run in sorted(runs.items())
    }
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "expected_seeds": list(EXPECTED_SEEDS),
        "records_per_validation_run": 294,
        "runs": serializable_runs,
        "paired_controls": pair_checks,
        "protected_partitions_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
