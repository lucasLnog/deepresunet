"""Compara DR-IT com DR-RT por imagem apenas na validação congelada."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, statistics
from pathlib import Path

METRICS = ("dice", "iou", "precision", "recall", "balanced_accuracy")


def sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 294:
        raise ValueError(f"{path}: esperado 294, obtido {len(rows)}")
    ids = [r["canonical_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs duplicados")
    return (
        ids,
        {r["canonical_id"]: {m: float(r[m]) for m in METRICS} for r in rows},
        float(rows[0]["threshold"]),
    )


def percentile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = math.floor(x)
    hi = math.ceil(x)
    return values[lo] if lo == hi else values[lo] * (hi - x) + values[hi] * (x - lo)


def inference(diff, seed, bootstrap=10000, permutations=50000):
    rng = random.Random(seed)
    n = len(diff)
    means = [sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(bootstrap)]
    observed = abs(statistics.fmean(diff))
    extreme = 0
    for _ in range(permutations):
        value = abs(sum(x if rng.getrandbits(1) else -x for x in diff) / n)
        extreme += value >= observed - 1e-15
    return {
        "mean_difference": statistics.fmean(diff),
        "median_difference": statistics.median(diff),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "improved": sum(x > 0 for x in diff),
        "equal": sum(x == 0 for x in diff),
        "worsened": sum(x < 0 for x in diff),
        "permutation_p_two_sided": (extreme + 1) / (permutations + 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drrt", type=Path, required=True)
    p.add_argument("--drit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260901)
    a = p.parse_args()
    rt_ids, rt, rt_t = load(a.drrt)
    it_ids, it, it_t = load(a.drit)
    if rt_ids != it_ids:
        raise ValueError("ordem/identidade das imagens difere")
    results = {
        m: inference([it[i][m] - rt[i][m] for i in rt_ids], a.seed + j)
        for j, m in enumerate(METRICS)
    }
    out = {
        "status": "completed",
        "partition": "validation",
        "statistical_unit": "canonical_image",
        "records": 294,
        "contrast": "DR-IT_minus_DR-RT",
        "thresholds": {"DR-RT": rt_t, "DR-IT": it_t},
        "primary_metric": "dice",
        "bootstrap": {"method": "paired_percentile", "samples": 10000, "confidence": 0.95},
        "permutation": {"method": "paired_sign_flip_two_sided", "samples": 50000},
        "results": results,
        "sources": {
            "DR-RT": {"path": str(a.drrt.resolve()), "sha256": sha(a.drrt)},
            "DR-IT": {"path": str(a.drit.resolve()), "sha256": sha(a.drit)},
        },
        "protected_partitions_accessed": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
