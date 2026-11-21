#!/usr/bin/env python3
"""Sela os artefatos da avaliação final sem executar inferência."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc_segmentation.final_protocol import audit_runtime_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/final_internal_test_protocol_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/reproduction/internal_dfuc2022_preflight.json"),
    )
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else args.project_root / args.protocol
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    report = audit_runtime_artifacts(args.project_root, protocol, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nPreflight concluído sem inferência: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
