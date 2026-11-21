"""Audita a execução dos braços DR-IF e DR-IFT sem acessar dados protegidos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MATCHED_FIELDS = (
    "train_loss",
    "train_bce",
    "train_dice_loss",
    "validation_loss",
    "validation_bce",
    "validation_dice_loss",
    "validation_dice_at_0_5",
    "validation_iou_at_0_5",
    "augmentation_epoch_seed",
    "augmentation_samples_seen",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drif-dir", type=Path, required=True)
    parser.add_argument("--drift-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    drif_dir, drift_dir = args.drif_dir.resolve(), args.drift_dir.resolve()
    drif_summary = read_json(drif_dir / "final_summary.json")
    drift_summary = read_json(drift_dir / "final_summary.json")
    drif_history = read_history(drif_dir / "history.csv")
    drift_history = read_history(drift_dir / "history.csv")

    first_phase_matches = all(
        drif_history[index][field] == drift_history[index][field]
        for index in range(15)
        for field in MATCHED_FIELDS
    )
    expected_transitions = {
        1: ("decoder_warmup", "", 3_348_669),
        16: ("layer4_released", "layer4", 16_463_037),
        31: ("layer3_layer4_released", "layer3|layer4", 23_285_437),
    }
    observed_transitions = {}
    transitions_match = True
    for epoch, expected in expected_transitions.items():
        row = drift_history[epoch - 1]
        observed = (
            row["fine_tuning_stage"],
            row["trainable_encoder_stages"],
            int(row["trainable_parameters"]),
        )
        observed_transitions[str(epoch)] = {
            "stage": observed[0],
            "trainable_encoder_stages": observed[1],
            "trainable_parameters": observed[2],
        }
        transitions_match &= observed == expected

    protected_access = bool(
        drif_summary.get("protected_partitions_accessed")
        or drift_summary.get("protected_partitions_accessed")
    )
    report = {
        "status": "ok"
        if (
            drif_summary.get("status") == "completed"
            and drift_summary.get("status") == "completed"
            and first_phase_matches
            and transitions_match
            and not protected_access
        )
        else "failed",
        "runs_completed": {
            "DR-IF": drif_summary.get("status") == "completed",
            "DR-IFT": drift_summary.get("status") == "completed",
        },
        "first_15_epochs_numerically_identical": first_phase_matches,
        "matched_history_fields": list(MATCHED_FIELDS),
        "progressive_phase_transitions_match_protocol": transitions_match,
        "observed_transitions": observed_transitions,
        "best_checkpoint_after_all_phases": int(drift_summary["training"]["best_epoch"]) >= 31,
        "best_checkpoint_epoch": int(drift_summary["training"]["best_epoch"]),
        "protected_partitions_accessed": protected_access,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
