"""AHENK kimlik kontrollerini şirket-blok bootstrap ile karşılaştırır."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-12
COMPARISONS = (
    ("original_text", "ticker_masked_text"),
    ("original_text", "previous_same_company_text"),
    ("original_text", "ticker_only_text"),
)


def qlike(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    predicted = np.maximum(np.asarray(predicted, dtype=float), EPSILON)
    ratio = actual / predicted
    return ratio - np.log(ratio) - 1.0


def compare(
    pivot: pd.DataFrame,
    candidate: str,
    reference: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict:
    paired = pivot[["company_id", "actual", candidate, reference]].dropna()
    candidate_loss = qlike(paired["actual"], paired[candidate])
    reference_loss = qlike(paired["actual"], paired[reference])
    paired = paired.assign(delta=candidate_loss - reference_loss)
    grouped = paired.groupby("company_id")["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    draws = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        draws[start : start + size] = sums[indices].sum(1) / counts[indices].sum(1)

    candidate_mean = float(candidate_loss.mean())
    reference_mean = float(reference_loss.mean())
    delta = candidate_mean - reference_mean
    interval = np.quantile(draws, [0.025, 0.975])
    return {
        "candidate": candidate,
        "reference": reference,
        "calls": len(paired),
        "companies": len(grouped),
        "candidate_qlike": candidate_mean,
        "reference_qlike": reference_mean,
        "delta_candidate_minus_reference": delta,
        "relative_improvement_percent": float(
            -100.0 * delta / max(reference_mean, EPSILON)
        ),
        "company_block_bootstrap_95_percent_ci": [
            float(interval[0]),
            float(interval[1]),
        ],
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "conclusion": (
            "SUPPORTED"
            if interval[1] < 0
            else "NOT_SUPPORTED" if interval[0] >= 0 else "INCONCLUSIVE"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    predictions = pd.read_csv(
        root / "artifacts/predictions/identity_control_predictions.csv"
    )
    metadata = pd.read_parquet(
        root / "data/processed/ahenk_tabular_features.parquet",
        columns=["call_id", "company_id"],
    )
    predictions = predictions.merge(
        metadata, on="call_id", how="left", validate="many_to_one"
    )

    rng = np.random.default_rng(20260927)
    regimes = {}
    for regime, rows in predictions.groupby("evaluation_regime"):
        pivot = rows.pivot_table(
            index=["call_id", "company_id", "actual"],
            columns="model",
            values="predicted",
            aggfunc="first",
        ).reset_index()
        regimes[regime] = [
            compare(pivot, candidate, reference, args.iterations, rng)
            for candidate, reference in COMPARISONS
        ]

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": args.iterations,
        "resampling_unit": "company",
        "paired_by": "call_id",
        "regimes": regimes,
        "human_or_expert_labels_used": False,
        "status": "PASS",
    }
    output = root / "reports/controls/identity_significance.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")


if __name__ == "__main__":
    main()