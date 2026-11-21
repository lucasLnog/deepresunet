#!/usr/bin/env python3
"""Sela a avaliacao externa FUSeg sem executar inferencia."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc_segmentation.external_protocol import audit_external_runtime_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/final_external_fuseg_protocol_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/reproduction/external_fuseg_preflight.json"),
    )
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else args.project_root / args.protocol
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    report = audit_external_runtime_artifacts(args.project_root, args.data_root, protocol, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nPreflight externo concluido sem inferencia: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
