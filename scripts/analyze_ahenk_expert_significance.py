"""AHENK v2 modellerini şirket-blok eşleştirilmiş bootstrap ile karşılaştırır."""

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
    ("ahenk_expert_gate_25", "fixed_expert_dropout_10"),
    ("ahenk_expert_gate_10", "fixed_expert_dropout_10"),
    ("ahenk_expert_gate_25", "price_expert"),
    ("ahenk_expert_gate_10", "price_expert"),
)
SCENARIOS = ("full", "random_25", "random_50", "robust_composite")


def qlike(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    predicted = np.maximum(np.asarray(predicted, dtype=float), EPSILON)
    ratio = actual / predicted
    return ratio - np.log(ratio) - 1.0


def bootstrap_compare(
    paired: pd.DataFrame,
    candidate: str,
    reference: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict:
    values = paired[["company_id", candidate, reference]].dropna()
    values = values.assign(delta=values[candidate] - values[reference])
    grouped = values.groupby("company_id")["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    draws = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        draws[start : start + size] = sums[indices].sum(1) / counts[indices].sum(1)

    candidate_mean = float(values[candidate].mean())
    reference_mean = float(values[reference].mean())
    delta = candidate_mean - reference_mean
    interval = np.quantile(draws, [0.025, 0.975])
    return {
        "candidate": candidate,
        "reference": reference,
        "paired_seed_call_observations": len(values),
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


def scenario_loss_table(rows: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "robust_composite":
        selected = rows[rows["scenario"].isin(["full", "random_25", "random_50"])].copy()
        selected["loss"] = qlike(selected["actual"], selected["predicted"])
        losses = (
            selected.groupby(
                ["call_id", "company_id", "seed", "model"], as_index=False
            )["loss"]
            .mean()
        )
    else:
        selected = rows[rows["scenario"] == scenario].copy()
        selected["loss"] = qlike(selected["actual"], selected["predicted"])
        losses = selected[["call_id", "company_id", "seed", "model", "loss"]]
    return losses.pivot_table(
        index=["call_id", "company_id", "seed"],
        columns="model",
        values="loss",
        aggfunc="first",
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    predictions = pd.read_csv(
        root / "artifacts/predictions/ahenk_expert_predictions.csv"
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
        regimes[regime] = {}
        for scenario in SCENARIOS:
            pivot = scenario_loss_table(rows, scenario)
            available = set(pivot.columns)
            regimes[regime][scenario] = [
                bootstrap_compare(
                    pivot, candidate, reference, args.iterations, rng
                )
                for candidate, reference in COMPARISONS
                if candidate in available and reference in available
            ]

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": args.iterations,
        "resampling_unit": "company",
        "paired_by": ["call_id", "seed"],
        "robust_composite": "mean per-example QLIKE across full, random_25, random_50",
        "regimes": regimes,
        "human_or_expert_labels_used": False,
        "status": "PASS",
    }
    output = root / "reports/models/ahenk_expert_significance.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")


if __name__ == "__main__":
    main()