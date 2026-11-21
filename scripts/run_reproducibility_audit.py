"""Confere a identidade dos artefatos finais sem executar inferência."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from tcc_segmentation.reproducibility_audit import (
    grouped_unique_counts,
    hash_matches,
    load_json,
    read_csv,
    sha256,
    write_csv,
)

EXPERIMENT_CONFIG_REVISION = "4007b873018045a1223385ae956803011fb53708"


def historical_config_hash(
    project: Path, relative: str, snapshot: Path | None
) -> tuple[str, str]:
    """Identifica e calcula o hash do arquivo original usado na avaliação."""
    if snapshot is not None:
        path = snapshot / relative
        return str(path), sha256(path)

    object_name = f"{EXPERIMENT_CONFIG_REVISION}:{relative}"
    try:
        contents = subprocess.run(
            ["git", "-C", str(project), "show", object_name],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Configuração histórica indisponível: {object_name}. "
            "Use um clone com essa revisão ou indique a cópia extraída com "
            "--historical-project-root."
        ) from error
    return f"git:{object_name}", hashlib.sha256(contents).hexdigest()


def git_revision(project: Path) -> str | None:
    """Retorna a revisão registrada na raiz do projeto, quando disponível."""
    for git_directory in (project / ".git", project / ".tmp/git_metadata"):
        if not git_directory.exists():
            continue
        try:
            result = subprocess.run(
                ["git", f"--git-dir={git_directory}", "rev-parse", "--verify", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        return result.stdout.strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("output/reproducibility_audit"))
    parser.add_argument(
        "--historical-project-root",
        type=Path,
        help="Raiz do projeto extraído da revisão usada para conferir os hashes originais.",
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    internal_dir = project / "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z"
    external_dir = project / "results/final_external_fuseg/FINAL_EXTERNAL_FUSEG_20260930T151950Z"
    day16_dir = project / "results/day16_analysis/DAY16_FINAL_20260930"
    day17_dir = project / "output/day17_final_assets"
    internal_identity = load_json(internal_dir / "execution_identity.json")
    internal_report = load_json(internal_dir / "final_test_report.json")
    external_identity = load_json(external_dir / "execution_identity.json")
    external_report = load_json(external_dir / "final_external_report.json")
    external_lock = load_json(project / "results/final_external_fuseg/preflight_lock.json")

    checks: list[dict[str, object]] = []

    def add(check_id: str, category: str, status: str, evidence: str, detail: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "category": category,
                "status": status,
                "evidence": evidence,
                "detail": detail,
            }
        )

    required_files = [
        "configs/study_protocol_v1.yaml",
        "configs/final_internal_test_protocol_v1.yaml",
        "configs/final_external_fuseg_protocol_v1.yaml",
        "configs/fine_tuning_progressive.yaml",
        "configs/sam_validation.yaml",
        "manifests/splits/dfuc_train.csv",
        "manifests/splits/dfuc_validation.csv",
        "manifests/splits/dfuc_test.csv",
        "manifests/canonical/fuseg_external_canonical.csv",
        "requirements-colab.txt",
        "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z/test_per_image_all_arms.csv",
        "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z/test_summary_all_arms.csv",
        "results/final_external_fuseg/FINAL_EXTERNAL_FUSEG_20260930T151950Z/external_per_unit.csv",
        "results/final_external_fuseg/FINAL_EXTERNAL_FUSEG_20260930T151950Z/external_summary.csv",
        "results/day16_analysis/DAY16_FINAL_20260930/dataset_summary_macro_micro.csv",
        "output/day17_final_assets/day17_report.json",
    ]
    for relative in required_files:
        path = project / relative
        add(
            f"file:{relative}",
            "completeness",
            "PASS" if path.is_file() else "FAIL",
            relative,
            "arquivo obrigatório presente" if path.is_file() else "arquivo obrigatório ausente",
        )

    historical_configs = [
        (
            "internal_protocol",
            "configs/final_internal_test_protocol_v1.yaml",
            internal_identity["protocol_sha256"],
        ),
        (
            "external_protocol",
            "configs/final_external_fuseg_protocol_v1.yaml",
            external_identity["protocol_sha256"],
        ),
        (
            "drift_config",
            "configs/day10_dr_ift.yaml",
            external_lock["primary_model"]["config_sha256"],
        ),
        (
            "sam_config",
            "configs/day13_sam_validation.yaml",
            external_lock["secondary_model"]["config_sha256"],
        ),
    ]
    hashes = []
    for name, relative, expected in historical_configs:
        evidence, actual = historical_config_hash(project, relative, args.historical_project_root)
        hashes.append((name, evidence, expected, actual))
    for name, relative, expected in (
        ("internal_manifest", "manifests/splits/dfuc_test.csv", internal_identity["manifest_sha256"]),
        (
            "external_manifest",
            "manifests/canonical/fuseg_external_canonical.csv",
            external_identity["manifest_sha256"],
        ),
    ):
        path = project / relative
        hashes.append((name, relative, expected, sha256(path) if path.is_file() else ""))

    artifact_hashes: list[dict[str, object]] = []
    for name, evidence, expected, actual in hashes:
        ok = actual.lower() == expected.lower()
        add(
            f"hash:{name}",
            "identity",
            "PASS" if ok else "FAIL",
            evidence,
            f"esperado={expected}; observado={actual}",
        )
        artifact_hashes.append(
            {
                "artifact": name,
                "path": evidence,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": ok,
            }
        )

    internal_rows = read_csv(internal_dir / "test_per_image_all_arms.csv")
    internal_groups = grouped_unique_counts(
        internal_rows, ("arm_id", "threshold_policy"), "canonical_id"
    )
    internal_ok = (
        len(internal_rows) == 5586
        and len(internal_groups) == 19
        and set(internal_groups.values()) == {294}
    )
    add(
        "rows:internal",
        "results",
        "PASS" if internal_ok else "FAIL",
        str((internal_dir / "test_per_image_all_arms.csv").relative_to(project)),
        f"linhas={len(internal_rows)}; grupos={len(internal_groups)}; unidades_por_grupo={sorted(set(internal_groups.values()))}",
    )

    external_rows = read_csv(external_dir / "external_per_unit.csv")
    external_groups = grouped_unique_counts(
        external_rows, ("arm_id", "threshold_policy"), "canonical_id"
    )
    external_ok = (
        len(external_rows) == 2886
        and len(external_groups) == 3
        and set(external_groups.values()) == {962}
    )
    add(
        "rows:external",
        "results",
        "PASS" if external_ok else "FAIL",
        str((external_dir / "external_per_unit.csv").relative_to(project)),
        f"linhas={len(external_rows)}; grupos={len(external_groups)}; unidades_por_grupo={sorted(set(external_groups.values()))}",
    )

    identity_ok = (
        internal_report["run_id"] == internal_identity["run_id"]
        and external_report["run_id"] == external_identity["run_id"]
        and internal_report["threshold_optimization_performed"] is False
        and internal_report["model_selection_performed"] is False
        and external_report["threshold_optimization_performed"] is False
        and external_report["fine_tuning_performed"] is False
    )
    add(
        "identity:final_runs",
        "identity",
        "PASS" if identity_ok else "FAIL",
        "execution_identity.json; final reports",
        "run IDs e travas científicas coerentes",
    )

    day17_manifest = read_csv(day17_dir / "manifesto_artefatos.csv")
    day17_bad = []
    for row in day17_manifest:
        path = project / row["path"]
        if not hash_matches(path, row["sha256"]):
            day17_bad.append(row["path"])
    add(
        "hash:day17_package",
        "reporting",
        "PASS" if not day17_bad else "FAIL",
        "output/day17_final_assets/manifesto_artefatos.csv",
        f"artefatos={len(day17_manifest)}; divergências={day17_bad}",
    )

    notebook_path = project / "notebooks/26_reproducibility_audit.ipynb"
    notebook_ok = False
    notebook_detail = "arquivo ausente"
    if notebook_path.exists():
        notebook = load_json(notebook_path)
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        try:
            for cell in code_cells:
                source = "".join(cell.get("source", []))
                source = "\n".join(
                    line for line in source.splitlines() if not line.lstrip().startswith("%")
                )
                compile(source, str(notebook_path), "exec")
            notebook_ok = len(code_cells) == 4
            notebook_detail = f"células_de_código={len(code_cells)}; compilação Python válida"
        except SyntaxError as exc:
            notebook_detail = f"erro de sintaxe: {exc}"
    add(
        "notebook:day18",
        "reproduction",
        "PASS" if notebook_ok else "FAIL",
        str(notebook_path.relative_to(project)),
        notebook_detail,
    )

    external_dependencies = [
        (
            "DR-IFT seed 42",
            external_lock["primary_model"]["checkpoint"],
            external_lock["primary_model"]["checkpoint_sha256"],
        ),
        (
            "SAM ViT-B",
            external_lock["secondary_model"]["sam_checkpoint"],
            external_lock["secondary_model"]["sam_checkpoint_sha256"],
        ),
    ]
    for name, path, expected in external_dependencies:
        add(
            f"external:{name}",
            "external_dependency",
            "EXTERNAL_REQUIRED",
            path,
            f"arquivo grande mantido no Drive; SHA-256 congelado={expected}",
        )

    revision = git_revision(project)
    add(
        "versioning:git",
        "identity",
        "PASS" if revision else "DECLARED_LIMITATION",
        ".git",
        f"revisão do código auditado={revision}" if revision else "revisão Git indisponível",
    )
    add(
        "independence:patient",
        "limitation",
        "DECLARED_LIMITATION",
        "configs/final_internal_test_protocol_v1.yaml",
        "IDs de paciente indisponíveis; independência no nível do paciente não é reivindicada",
    )

    failed = [row for row in checks if row["status"] == "FAIL"]
    status = "passed_with_declared_external_dependencies" if not failed else "failed"
    write_csv(
        output / "reproducibility_checklist.csv",
        checks,
        ["check_id", "category", "status", "evidence", "detail"],
    )
    write_csv(
        output / "artifact_hashes.csv",
        artifact_hashes,
        ["artifact", "path", "expected_sha256", "actual_sha256", "match"],
    )

    environment = {
        "audit_environment": {"python": sys.version, "platform": platform.platform()},
        "observed_final_colab_environment": {
            "python": internal_report["python"],
            "platform": internal_report["platform"],
            "torch": internal_report["torch"],
            "gpu": internal_report["gpu"],
        },
        "requirements_file": "requirements-colab.txt",
        "environment_config": "configs/environment.yaml",
        "note": "PyTorch and torchvision are supplied by the Colab runtime; the exact observed torch build is recorded in both final reports.",
    }
    (output / "environment_snapshot.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "status": status,
        "checks": len(checks),
        "passed": sum(row["status"] == "PASS" for row in checks),
        "failed": len(failed),
        "external_required": sum(row["status"] == "EXTERNAL_REQUIRED" for row in checks),
        "declared_limitations": sum(row["status"] == "DECLARED_LIMITATION" for row in checks),
        "internal_run_id": internal_identity["run_id"],
        "external_run_id": external_identity["run_id"],
        "day16_directory": str(day16_dir.relative_to(project)),
        "day17_directory": str(day17_dir.relative_to(project)),
        "scientific_results_recomputed": False,
        "inference_executed": False,
        "audited_code_revision": revision,
    }
    (output / "day18_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_paths = [
        output / "reproducibility_checklist.csv",
        output / "artifact_hashes.csv",
        output / "environment_snapshot.json",
        output / "day18_audit_report.json",
        output / "README.md",
        output / "REPRODUCAO_COLAB.md",
        output / "METODOS_RESULTADOS_RESUMO.md",
    ]
    manifest_rows = [
        {
            "path": str(path.relative_to(project)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in manifest_paths
        if path.exists()
    ]
    write_csv(output / "day18_manifest.csv", manifest_rows, ["path", "bytes", "sha256"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
