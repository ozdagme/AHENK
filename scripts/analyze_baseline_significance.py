"""AHENK taban ablasyonları için şirket-blok eşleştirilmiş bootstrap analizi.

Aynı çağrıdaki iki model kaybını eşleştirir; tekrar eden şirket çağrılarının
bağımsız olduğu varsayımını yapmamak için şirketleri blok olarak örnekler.
GPU veya insan etiketi kullanmaz.
"""

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
    ("hist_gradient_boosting_combined", "hist_gradient_boosting_price"),
    ("hist_gradient_boosting_price_acoustic", "hist_gradient_boosting_price"),
    ("hist_gradient_boosting_price_text_structure", "hist_gradient_boosting_price"),
    ("hist_gradient_boosting_combined", "hist_gradient_boosting_price_acoustic"),
)


def qlike_loss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    predicted = np.maximum(np.asarray(predicted, dtype=float), EPSILON)
    ratio = actual / predicted
    return ratio - np.log(ratio) - 1.0


def block_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict:
    required = ["call_id", "company_id", "actual", candidate, reference]
    paired = frame[required].dropna()
    candidate_loss = qlike_loss(paired["actual"], paired[candidate])
    reference_loss = qlike_loss(paired["actual"], paired[reference])
    paired = paired.assign(loss_delta=candidate_loss - reference_loss)

    grouped = paired.groupby("company_id")["loss_delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    company_count = len(grouped)

    bootstrap_delta = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        indices = rng.integers(0, company_count, size=(size, company_count))
        bootstrap_delta[start : start + size] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )

    observed_candidate = float(candidate_loss.mean())
    observed_reference = float(reference_loss.mean())
    observed_delta = observed_candidate - observed_reference
    interval = np.quantile(bootstrap_delta, [0.025, 0.975])
    return {
        "candidate": candidate,
        "reference": reference,
        "calls": len(paired),
        "companies": company_count,
        "candidate_qlike": observed_candidate,
        "reference_qlike": observed_reference,
        "delta_candidate_minus_reference": observed_delta,
        "relative_improvement_percent": float(
            -100.0 * observed_delta / max(observed_reference, EPSILON)
        ),
        "company_block_bootstrap_95_percent_ci": [
            float(interval[0]),
            float(interval[1]),
        ],
        "bootstrap_probability_candidate_better": float(
            np.mean(bootstrap_delta < 0)
        ),
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
    predictions_path = root / "artifacts/predictions/cpu_baseline_predictions.csv"
    features_path = root / "data/processed/ahenk_tabular_features.parquet"
    predictions = pd.read_csv(predictions_path)
    metadata = pd.read_parquet(features_path, columns=["call_id", "company_id"])
    predictions = predictions.merge(metadata, on="call_id", how="left", validate="many_to_one")

    rng = np.random.default_rng(20260927)
    regimes: dict[str, list[dict]] = {}
    for regime_name, regime_rows in predictions.groupby("evaluation_regime"):
        pivot = regime_rows.pivot(
            index=["call_id", "company_id", "actual"],
            columns="model",
            values="predicted",
        ).reset_index()
        available = set(pivot.columns)
        results = []
        for candidate, reference in COMPARISONS:
            if candidate in available and reference in available:
                results.append(
                    block_bootstrap(
                        pivot,
                        candidate,
                        reference,
                        args.iterations,
                        rng,
                    )
                )
        regimes[regime_name] = results

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": args.iterations,
        "resampling_unit": "company",
        "paired_by": "call_id",
        "primary_metric": "QLIKE",
        "negative_delta_means_candidate_is_better": True,
        "regimes": regimes,
        "human_or_expert_labels_used": False,
        "status": "PASS",
    }
    output = root / "reports/baselines/cpu_baseline_significance.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")


if __name__ == "__main__":
    main()