"""Dondurulmuş dış sonuç için kimlik-audit duyarlılık analizi.

Bu analiz yeni bir doğrulama veya model seçimi değildir. Dondurulmuş model,
seed'ler ve stres senaryoları değiştirilmeden yeniden çalıştırılır; sonuçlar
önceden tanımlı kimlik-audit alt kümelerinde betimsel olarak yeniden hesaplanır.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ahenk_pilot import EPSILON, TARGET, fit_dense_text, fit_numeric, quality_features
from run_ahenk_v3_development import (
    EVAL_SCENARIOS,
    build_oof_predictions,
    fuse_predictions,
    predict_experts,
    train_experts,
    train_gate,
)
from run_ahenk_v3_external_confirmation import FROZEN_SEEDS, sha256
from run_extended_validation import stress_scenarios


BOOTSTRAP_REPETITIONS = 5000
MODEL_NAMES = ("fixed_mean", "oof_risk_gate")
SUPPORTED_STATUSES = {
    "consistent_provisional_text_support",
    "majority_provisional_text_support",
    "limited_text_support_review",
}
MAJORITY_STATUSES = {
    "consistent_provisional_text_support",
    "majority_provisional_text_support",
}


def qlike_per_call(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    prediction = np.maximum(np.asarray(prediction, dtype=float), EPSILON)
    ratio = actual / prediction
    return ratio - np.log(ratio) - 1.0


def company_block_interval(
    differences: np.ndarray,
    companies: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    companies = companies.astype(str)
    unique = np.unique(companies)
    blocks = [differences[companies == company] for company in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[repetition] = np.concatenate(
            [blocks[index] for index in chosen]
        ).mean()
    return {
        "mean_fixed_minus_gate": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "company_blocks": int(len(unique)),
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA istendi fakat kullanılamıyor.")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    root = Path(args.project_root).resolve()

    output_path = root / "reports/confirmation/external_identity_sensitivity.json"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Çıktı zaten var: {output_path}. Yenilemek için --overwrite kullanın."
        )

    paths = {
        "freeze": root / "reports/confirmation/tiingo_confirmation_freeze.json",
        "original_report": root / "reports/confirmation/ahenk_v3_external_confirmation.json",
        "external": root / "data/confirmation/tiingo_confirmation_features.parquet",
        "index": root / "data/processed/ahenk_dataset_index.csv",
        "tabular": root / "data/processed/ahenk_tabular_features.parquet",
        "identity_queue": (
            root
            / "reports/confirmation/security_identity_audit/identity_review_queue_v2.csv"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Gerekli dosyalar eksik:\n" + "\n".join(missing))

    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    original_report = json.loads(
        paths["original_report"].read_text(encoding="utf-8")
    )
    if freeze.get("status") != "FROZEN_EXTERNAL_CONFIRMATION":
        raise SystemExit("Dış veri dondurma durumu geçerli değil.")
    if freeze.get("features_sha256") != sha256(paths["external"]):
        raise SystemExit("Dış özellik dosyası dondurulmuş hash ile uyuşmuyor.")
    if tuple(original_report.get("frozen_seeds", [])) != tuple(FROZEN_SEEDS):
        raise SystemExit("Özgün raporun seed'leri kodla uyuşmuyor.")

    index = pd.read_csv(paths["index"])
    tabular = pd.read_parquet(paths["tabular"])
    original = tabular.merge(
        index[["call_id", "text_path"]],
        on="call_id",
        how="left",
        validate="one_to_one",
    ).sort_values(["call_date", "call_id"])
    train = original[original["temporal_split"] == "train"].reset_index(drop=True)
    dev = original[original["temporal_split"] == "dev"].reset_index(drop=True)
    calibration = dev.iloc[: len(dev) // 2].reset_index(drop=True)
    external = (
        pd.read_parquet(paths["external"])
        .sort_values(["call_date", "call_id"])
        .reset_index(drop=True)
    )
    identity = pd.read_csv(paths["identity_queue"]).fillna("")
    identity = identity[identity["audit_group"] == "previous_external"].copy()

    if len(train) != 1725 or len(calibration) != 181 or len(external) != 662:
        raise SystemExit(
            f"Dondurulmuş boyutlar değişmiş: train={len(train)}, "
            f"calibration={len(calibration)}, external={len(external)}"
        )
    if identity["source_ticker"].nunique() != 248:
        raise SystemExit(
            "Kimlik kuyruğunda beklenen 248 dış şirket bulunamadı: "
            f"{identity['source_ticker'].nunique()}"
        )

    external_companies = set(external["company_id"].astype(str))
    identity_companies = set(identity["source_ticker"].astype(str))
    if external_companies != identity_companies:
        raise SystemExit(
            "Dış veri ile kimlik kuyruğundaki şirketler bire bir uyuşmuyor. "
            f"Yalnız dışta={sorted(external_companies - identity_companies)[:10]}, "
            f"yalnız kuyrukta={sorted(identity_companies - external_companies)[:10]}"
        )

    status_by_company = dict(
        zip(identity["source_ticker"].astype(str), identity["company_review_status"])
    )
    metadata_available = set(
        identity.loc[
            identity["company_review_status"] != "provider_metadata_missing_review",
            "source_ticker",
        ].astype(str)
    )
    any_text_support = set(
        identity.loc[
            identity["company_review_status"].isin(SUPPORTED_STATUSES),
            "source_ticker",
        ].astype(str)
    )
    majority_text_support = set(
        identity.loc[
            identity["company_review_status"].isin(MAJORITY_STATUSES),
            "source_ticker",
        ].astype(str)
    )
    consistent_text_support = set(
        identity.loc[
            identity["company_review_status"]
            == "consistent_provisional_text_support",
            "source_ticker",
        ].astype(str)
    )
    reuse_pattern = re.compile(r"\b(?:ETF|ETN|2X|3X|DAILY|SHORT|LONG)\b", re.I)
    ticker_reuse_candidates = set(
        identity.loc[
            identity["provider_names"].astype(str).map(
                lambda value: bool(reuse_pattern.search(value))
            ),
            "source_ticker",
        ].astype(str)
    )

    subsets = {
        "all_frozen_external": external_companies,
        "exclude_current_ticker_reuse_candidates": (
            external_companies - ticker_reuse_candidates
        ),
        "provider_metadata_available": metadata_available,
        "any_provisional_text_support": any_text_support,
        "majority_or_consistent_text_support": majority_text_support,
        "consistent_text_support_only": consistent_text_support,
    }

    acoustic_columns = [
        column
        for column in train
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    price_columns = [column for column in train if column.startswith("price_")]
    missing_external_columns = [
        column
        for column in acoustic_columns + price_columns
        if column not in external
    ]
    if missing_external_columns:
        raise SystemExit(
            f"Dış özellik sütunları eksik: {missing_external_columns[:10]}"
        )

    text_train, text_calibration, text_external, text_info = fit_dense_text(
        root, train, calibration, external, dimensions=128
    )
    acoustic_train, acoustic_calibration, acoustic_external = fit_numeric(
        train, calibration, external, acoustic_columns
    )
    price_train, price_calibration, price_external = fit_numeric(
        train, calibration, external, price_columns
    )
    quality_train, _quality_calibration, quality_external = quality_features(
        train, (train, calibration, external)
    )
    train_arrays = (text_train, acoustic_train, price_train)
    external_arrays = (text_external, acoustic_external, price_external)
    train_target = train[TARGET].to_numpy(float)
    external_target = external[TARGET].to_numpy(float)

    losses: dict[str, dict[str, list[np.ndarray]]] = {
        model: {scenario: [] for scenario in EVAL_SCENARIOS}
        for model in MODEL_NAMES
    }

    for seed in FROZEN_SEEDS:
        print(f"Seed {seed}: dondurulmuş uzmanlar ve risk kapısı yeniden kuruluyor")
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
        stressed = stress_scenarios(external_arrays, quality_external, seed)

        for scenario in EVAL_SCENARIOS:
            scenario_arrays, scenario_quality, scenario_mask = stressed[scenario]
            means, log_variances = predict_experts(
                experts, scenario_arrays, device
            )
            for model in MODEL_NAMES:
                log_prediction, _weights = fuse_predictions(
                    model,
                    gate,
                    scenario_quality,
                    means,
                    log_variances,
                    scenario_mask,
                    device,
                )
                losses[model][scenario].append(
                    qlike_per_call(external_target, np.exp(log_prediction))
                )

    mean_losses = {
        model: {
            scenario: np.mean(seed_losses, axis=0)
            for scenario, seed_losses in scenario_losses.items()
        }
        for model, scenario_losses in losses.items()
    }

    company_values = external["company_id"].astype(str).to_numpy()
    results: dict[str, object] = {}
    for subset_index, (subset_name, companies) in enumerate(subsets.items()):
        mask = np.isin(company_values, sorted(companies))
        if not mask.any():
            raise SystemExit(f"Boş duyarlılık alt kümesi: {subset_name}")

        scenario_results: dict[str, object] = {}
        robust_differences = []
        fixed_scenario_means = []
        gate_scenario_means = []
        for scenario_index, scenario in enumerate(EVAL_SCENARIOS):
            fixed = mean_losses["fixed_mean"][scenario][mask]
            gate = mean_losses["oof_risk_gate"][scenario][mask]
            difference = fixed - gate
            robust_differences.append(difference)
            fixed_scenario_means.append(float(fixed.mean()))
            gate_scenario_means.append(float(gate.mean()))
            scenario_results[scenario] = {
                "fixed_mean_qlike": float(fixed.mean()),
                "oof_risk_gate_qlike": float(gate.mean()),
                "relative_improvement_percent": float(
                    100.0 * (fixed.mean() - gate.mean()) / fixed.mean()
                ),
                "company_block_interval": company_block_interval(
                    difference,
                    company_values[mask],
                    seed=20260928 + subset_index * 100 + scenario_index,
                ),
            }

        fixed_robust = float(np.mean(fixed_scenario_means))
        gate_robust = float(np.mean(gate_scenario_means))
        robust_difference = np.mean(robust_differences, axis=0)
        results[subset_name] = {
            "calls": int(mask.sum()),
            "companies": int(np.unique(company_values[mask]).size),
            "fixed_mean_robust_qlike": fixed_robust,
            "oof_risk_gate_robust_qlike": gate_robust,
            "relative_improvement_percent": float(
                100.0 * (fixed_robust - gate_robust) / fixed_robust
            ),
            "robust_company_block_interval": company_block_interval(
                robust_difference,
                company_values[mask],
                seed=20260929 + subset_index,
            ),
            "scenarios": scenario_results,
        }

    all_result = results["all_frozen_external"]
    original_primary = original_report["primary_comparison"]
    reproduction = {
        "fixed_mean_absolute_difference": abs(
            all_result["fixed_mean_robust_qlike"]
            - original_primary["fixed_mean_robust_qlike"]
        ),
        "oof_risk_gate_absolute_difference": abs(
            all_result["oof_risk_gate_robust_qlike"]
            - original_primary["oof_risk_gate_robust_qlike"]
        ),
    }
    reproduction["within_tolerance"] = bool(
        reproduction["fixed_mean_absolute_difference"] <= 1e-6
        and reproduction["oof_risk_gate_absolute_difference"] <= 1e-6
    )
    if not reproduction["within_tolerance"]:
        raise SystemExit(
            "Dondurulmuş tam-küme sonuçları 1e-6 toleransla yeniden üretilemedi: "
            + json.dumps(reproduction, ensure_ascii=False)
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_hoc_external_identity_sensitivity",
        "confirmatory_status": "EXPLORATORY_SENSITIVITY_ONLY",
        "model_or_hyperparameter_selection_performed": False,
        "original_external_set_already_observed": True,
        "device": str(device),
        "frozen_seeds": list(FROZEN_SEEDS),
        "evaluation_scenarios": list(EVAL_SCENARIOS),
        "text_preprocessing": text_info,
        "identity_queue_sha256": sha256(paths["identity_queue"]),
        "external_features_sha256": sha256(paths["external"]),
        "ticker_reuse_candidates_excluded_in_named_subset": sorted(
            ticker_reuse_candidates
        ),
        "company_status_counts": dict(
            (
                str(status),
                int(count),
            )
            for status, count in pd.Series(
                list(status_by_company.values())
            ).value_counts().items()
        ),
        "full_set_reproduction": reproduction,
        "subsets": results,
        "interpretation_constraint": (
            "Alt kümeler kimlik taraması görüldükten sonra tanımlandığı için "
            "sonuçlar yalnız duyarlılık analizi olarak yorumlanmalıdır."
        ),
    }
    atomic_json(output_path, report)

    concise = {
        name: {
            "calls": value["calls"],
            "companies": value["companies"],
            "relative_improvement_percent": value[
                "relative_improvement_percent"
            ],
            "ci95_low": value["robust_company_block_interval"]["ci95_low"],
            "ci95_high": value["robust_company_block_interval"]["ci95_high"],
        }
        for name, value in results.items()
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"\nDuyarlılık raporu: {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()