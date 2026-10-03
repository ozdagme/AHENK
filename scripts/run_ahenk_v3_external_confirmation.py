"""Dondurulmuş AHENK V3'ün tek seferlik Tiingo dış doğrulaması.

Mimari, seed'ler, ön işleme, senaryolar ve karar kuralı V3 geliştirme
sonucundan değiştirilmez. Önceki temporal test satırları kullanılmaz. Başarılı
bir rapor oluştuğunda betik ikinci bir doğrulama çalıştırmasını reddeder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ahenk_pilot import EPSILON, TARGET, fit_dense_text, fit_numeric, metric_set, quality_features
from run_ahenk_v3_development import (
    EVAL_SCENARIOS,
    MODEL_NAMES,
    MODALITIES,
    aggregate,
    build_oof_predictions,
    conformal_quantile,
    fuse_predictions,
    interval_metrics,
    predict_experts,
    train_experts,
    train_gate,
)
from run_extended_validation import stress_scenarios


FROZEN_SEEDS = (3407, 2026)
BOOTSTRAP_REPETITIONS = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qlike_per_call(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    prediction = np.maximum(np.asarray(prediction, dtype=float), EPSILON)
    ratio = actual / prediction
    return ratio - np.log(ratio) - 1.0


def company_block_interval(
    differences: np.ndarray,
    companies: np.ndarray,
    seed: int = 20260928,
) -> dict[str, float | int]:
    unique = np.unique(companies.astype(str))
    blocks = [differences[companies.astype(str) == company] for company in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[repetition] = np.concatenate([blocks[index] for index in chosen]).mean()
    return {
        "mean_fixed_minus_gate": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "company_blocks": int(len(unique)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA istendi fakat kullanılamıyor.")
    torch.set_num_threads(4)
    device = torch.device(args.device)

    root = Path(args.project_root).resolve()
    output_path = root / "reports/confirmation/ahenk_v3_external_confirmation.json"
    if output_path.exists():
        raise SystemExit(
            "Dış doğrulama raporu zaten var. Tek-sefer protokolü nedeniyle yeniden çalıştırılmadı."
        )

    freeze_path = root / "reports/confirmation/tiingo_confirmation_freeze.json"
    external_path = root / "data/confirmation/tiingo_confirmation_features.parquet"
    index_path = root / "data/processed/ahenk_dataset_index.csv"
    tabular_path = root / "data/processed/ahenk_tabular_features.parquet"
    for path in (freeze_path, external_path, index_path, tabular_path):
        if not path.is_file():
            raise SystemExit(f"Gerekli dosya bulunamadı: {path}")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_EXTERNAL_CONFIRMATION":
        raise SystemExit("Dış doğrulama kümesi dondurulmuş durumda değil.")
    if freeze.get("features_sha256") != sha256(external_path):
        raise SystemExit("Dış özellik dosyası dondurma hash'iyle uyuşmuyor.")
    if freeze.get("overlap_with_original_model_call_ids") != 0:
        raise SystemExit("Dış kümede model verisi çakışması raporlanmış.")

    index = pd.read_csv(index_path)
    tabular = pd.read_parquet(tabular_path)
    original = tabular.merge(
        index[["call_id", "text_path"]], on="call_id", how="left", validate="one_to_one"
    ).sort_values(["call_date", "call_id"])
    train = original[original["temporal_split"] == "train"].reset_index(drop=True)
    dev = original[original["temporal_split"] == "dev"].reset_index(drop=True)
    calibration = dev.iloc[: len(dev) // 2].reset_index(drop=True)
    external = pd.read_parquet(external_path).sort_values(["call_date", "call_id"]).reset_index(drop=True)

    if len(train) != 1725 or len(calibration) != 181 or len(external) != 662:
        raise SystemExit(
            f"Dondurulmuş boyutlar değişmiş: train={len(train)}, "
            f"calibration={len(calibration)}, external={len(external)}"
        )
    if set(external["call_id"]).intersection(index["call_id"]):
        raise SystemExit("Dış çağrı kimlikleri özgün model indeksine sızmış.")

    acoustic_columns = [
        column for column in train
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    price_columns = [column for column in train if column.startswith("price_")]
    missing_external = [
        column for column in acoustic_columns + price_columns if column not in external
    ]
    if missing_external:
        raise SystemExit(f"Dış kümede eksik özellik sütunları: {missing_external[:10]}")

    text_train, text_calibration, text_external, text_info = fit_dense_text(
        root, train, calibration, external, dimensions=128
    )
    acoustic_train, acoustic_calibration, acoustic_external = fit_numeric(
        train, calibration, external, acoustic_columns
    )
    price_train, price_calibration, price_external = fit_numeric(
        train, calibration, external, price_columns
    )
    quality_train, quality_calibration, quality_external = quality_features(
        train, (train, calibration, external)
    )
    train_arrays = (text_train, acoustic_train, price_train)
    calibration_arrays = (text_calibration, acoustic_calibration, price_calibration)
    external_arrays = (text_external, acoustic_external, price_external)
    train_target = train[TARGET].to_numpy(float)
    calibration_target = calibration[TARGET].to_numpy(float)
    external_target = external[TARGET].to_numpy(float)

    model_seed_results = {model: [] for model in MODEL_NAMES}
    per_call_losses: dict[str, dict[str, list[np.ndarray]]] = {
        model: {scenario: [] for scenario in EVAL_SCENARIOS}
        for model in MODEL_NAMES
    }

    for seed in FROZEN_SEEDS:
        print(f"Seed {seed}: dondurulmuş OOF uzman eğitimi")
        oof_means, oof_log_variances, valid = build_oof_predictions(
            train_arrays, train_target, device, seed
        )
        gate = train_gate(
            quality_train[valid],
            oof_means[valid],
            oof_log_variances[valid],
            train_target[valid],
            device,
            seed,
        )
        internal_split = int(len(train) * 0.90)
        experts = train_experts(
            tuple(value[:internal_split] for value in train_arrays),
            train_target[:internal_split],
            tuple(value[internal_split:] for value in train_arrays),
            train_target[internal_split:],
            device,
            seed + 100,
            epochs=90,
            patience=12,
        )
        calibration_means, calibration_log_variances = predict_experts(
            experts, calibration_arrays, device
        )
        external_stresses = stress_scenarios(external_arrays, quality_external, seed)

        for model_name in MODEL_NAMES:
            calibration_mask = np.ones((len(calibration), 3), dtype="float32")
            calibration_log_prediction, _ = fuse_predictions(
                model_name,
                gate,
                quality_calibration,
                calibration_means,
                calibration_log_variances,
                calibration_mask,
                device,
            )
            radius = conformal_quantile(
                calibration_target, calibration_log_prediction, level=0.90
            )
            scenarios = {}
            for scenario in EVAL_SCENARIOS:
                scenario_arrays, scenario_quality, scenario_mask = external_stresses[scenario]
                means, log_variances = predict_experts(experts, scenario_arrays, device)
                log_prediction, weights = fuse_predictions(
                    model_name,
                    gate,
                    scenario_quality,
                    means,
                    log_variances,
                    scenario_mask,
                    device,
                )
                prediction = np.exp(log_prediction)
                per_call_losses[model_name][scenario].append(
                    qlike_per_call(external_target, prediction)
                )
                scenarios[scenario] = {
                    "point": metric_set(external_target, prediction),
                    "interval": interval_metrics(external_target, log_prediction, radius),
                    "mean_weights": {
                        modality: float(weights[:, index].mean())
                        for index, modality in enumerate(MODALITIES)
                    },
                }
            model_seed_results[model_name].append(
                {"seed": seed, "conformal_radius": radius, "scenarios": scenarios}
            )

    models = {
        model: {"seed_results": values, "aggregate": aggregate(values)}
        for model, values in model_seed_results.items()
    }
    fixed_robust = models["fixed_mean"]["aggregate"]["robust_qlike_mean"]
    gate_robust = models["oof_risk_gate"]["aggregate"]["robust_qlike_mean"]
    relative_improvement = 100.0 * (fixed_robust - gate_robust) / fixed_robust

    paired = {}
    robust_difference_parts = []
    for scenario in EVAL_SCENARIOS:
        fixed_loss = np.mean(per_call_losses["fixed_mean"][scenario], axis=0)
        gate_loss = np.mean(per_call_losses["oof_risk_gate"][scenario], axis=0)
        difference = fixed_loss - gate_loss
        robust_difference_parts.append(difference)
        paired[scenario] = company_block_interval(
            difference, external["company_id"].to_numpy(), seed=20260928
        )
    robust_difference = np.mean(robust_difference_parts, axis=0)
    paired["robust_six_scenario_mean"] = company_block_interval(
        robust_difference, external["company_id"].to_numpy(), seed=20260929
    )
    robust_supported = paired["robust_six_scenario_mean"]["ci95_low"] > 0
    decision = (
        "CONFIRMED"
        if relative_improvement >= 3.0 and robust_supported
        else "NOT_CONFIRMED"
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "single_locked_external_confirmation",
        "status": decision,
        "device": str(device),
        "frozen_seeds": list(FROZEN_SEEDS),
        "train_calls": len(train),
        "calibration_calls": len(calibration),
        "external_calls": len(external),
        "external_companies": int(external["company_id"].nunique()),
        "external_companies_seen_in_training": int(
            external["company_id"].isin(train["company_id"]).sum()
        ),
        "previous_temporal_test_rows_used": False,
        "development_evaluation_rows_used": False,
        "model_or_hyperparameter_selection_on_external_set": False,
        "text_preprocessing": text_info,
        "architecture": "frozen heteroscedastic modality experts + OOF risk gate",
        "primary_comparison": {
            "fixed_mean_robust_qlike": fixed_robust,
            "oof_risk_gate_robust_qlike": gate_robust,
            "relative_improvement_percent": relative_improvement,
            "minimum_required_improvement_percent": 3.0,
            "company_block_ci": paired["robust_six_scenario_mean"],
            "statistically_supported": bool(robust_supported),
        },
        "paired_company_block_results": paired,
        "models": models,
        "external_freeze_sha256": sha256(freeze_path),
        "external_features_sha256": sha256(external_path),
        "human_or_expert_labels_used": False,
        "rerun_for_model_selection_allowed": False,
        "decision_rule": "CONFIRMED iff robust relative QLIKE improvement >=3% and 95% company-block CI lower bound >0",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": decision,
                "fixed_mean_robust_qlike": fixed_robust,
                "oof_risk_gate_robust_qlike": gate_robust,
                "relative_improvement_percent": relative_improvement,
                "company_block_ci": paired["robust_six_scenario_mean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Tek seferlik dış doğrulama raporu: {output_path}")


if __name__ == "__main__":
    main()