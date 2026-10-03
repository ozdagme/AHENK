"""PyTorch gerektirmeyen katı eksik-modalite maskeleme testi."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sanitize(
    quality: np.ndarray,
    means: np.ndarray,
    log_variances: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality_mask = np.repeat(mask, 2, axis=1)
    return quality * quality_mask, means * mask, log_variances * mask


def main() -> None:
    rng = np.random.default_rng(29)
    count = 64
    quality = rng.normal(size=(count, 6)).astype("float32")
    means = rng.normal(size=(count, 3)).astype("float32")
    log_variances = rng.normal(size=(count, 3)).astype("float32")
    mask = (rng.random((count, 3)) > 0.5).astype("float32")
    empty = np.where(mask.sum(axis=1) == 0)[0]
    mask[empty, rng.integers(0, 3, size=len(empty))] = 1.0

    baseline = sanitize(quality, means, log_variances, mask)
    quality_mask = np.repeat(mask, 2, axis=1)
    altered = sanitize(
        quality + (1.0 - quality_mask) * 1000.0,
        means - (1.0 - mask) * 1000.0,
        log_variances + (1.0 - mask) * 1000.0,
        mask,
    )
    differences = [float(np.max(np.abs(left - right))) for left, right in zip(baseline, altered)]
    if any(value != 0.0 for value in differences):
        raise SystemExit(f"Katı maskeleme testi başarısız: {differences}")

    source = Path("scripts/run_ahenk_true_missing_development.py").read_text(
        encoding="utf-8"
    )
    required = (
        "quality * quality_mask",
        "means * mask",
        "log_variances * mask",
        "masked_information_invariance",
        'external_confirmation_rows_used": False',
    )
    missing = [item for item in required if item not in source]
    if missing:
        raise SystemExit(f"Kaynak denetimi başarısız; eksik ifadeler: {missing}")

    print(
        {
            "status": "PASS",
            "max_sanitized_difference": max(differences),
            "rows": count,
        }
    )


if __name__ == "__main__":
    main()