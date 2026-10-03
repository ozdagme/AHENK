#!/usr/bin/env python3
"""Run the one-time locked Earnings25 strict-missing forward confirmation.

The primary endpoint is read from the target-blind forward-cohort freeze:
strict OOF-risk AHENK versus available-modality equal weighting under 25% and
50% independently missing modalities.  Five frozen training seeds and twenty
paired mask draws per seed/scenario are evaluated.  Inference uses the frozen
10,000-repetition hierarchical bootstrap over CIK blocks, training seeds, and
paired missingness draws.

No architecture, hyperparameter, preprocessing, cohort, or success-rule
selection is performed on Earnings25.  A completed report prevents rerunning
the locked test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_ahenk_v3_development as base
from run_ahenk_pilot import (
    EPSILON,
    TARGET,
    fit_dense_text,
    fit_numeric,
    metric_set,
    quality_features,
    random_mask,
)
from run_ahenk_true_missing_development import (
    masked_information_invariance,
    self_test as strict_gate_self_test,
    train_strict_gate,
)
from run_extended_validation import stress_scenarios


FORWARD_FREEZE = Path(
    "reports/confirmation/earnings25_forward_confirmation_freeze.json"
)
FEATURE_FREEZE = Path(
    "reports/confirmation/earnings25_forward_confirmation_feature_freeze.json"
)
EXTERNAL_FEATURES = Path(
    "data/confirmation/earnings25_forward_confirmation_features.parquet"
)
MODEL_INDEX = Path("data/processed/ahenk_dataset_index.csv")
MODEL_TABULAR = Path("data/processed/ahenk_tabular_features.parquet")
PROTOCOL_OUTPUT = Path(
    "reports/confirmation/earnings25_forward_locked_execution_protocol.json"
)
REPORT_OUTPUT = Path(
    "reports/confirmation/earnings25_forward_locked_confirmation.json"
)
PER_CALL_OUTPUT = Path(
    "reports/confirmation/earnings25_forward_locked_per_call.csv"
)

PRIMARY_SCENARIOS = {
    "random_25_strict_missing": 0.25,
    "random_50_strict_missing": 0.50,
}
PRIMARY_MODELS = ("fixed_mean", "oof_risk_gate")
SECONDARY_MODELS = ("price_expert", "fixed_mean", "oof_risk_gate")
SECONDARY_SCENARIOS = (
    "full",
    "text_noise_05",
    "acoustic_noise_10",
    "price_noise_05",
)
EXPECTED_TRAIN_CALLS = 1725
EXPECTED_CALIBRATION_CALLS = 181
EXPECTED_EXTERNAL_CALLS = 114
EXPECTED_EXTERNAL_BLOCKS = 114
MASK_RNG_NAMESPACE = 20261003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def qlike_per_call(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    prediction = np.maximum(np.asarray(prediction, dtype=float), EPSILON)
    ratio = actual / prediction
    return ratio - np.log(ratio) - 1.0


def coverage(actual: np.ndarray, log_prediction: np.ndarray, radius: float) -> float:
    lower = np.exp(log_prediction - radius)
    upper = np.exp(log_prediction + radius)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def strict_mask(
    count: int,
    probability: float,
    training_seed: int,
    repetition: int,
) -> np.ndarray:
    probability_code = int(round(probability * 100))
    seed = np.random.SeedSequence(
        [MASK_RNG_NAMESPACE, int(training_seed), probability_code, int(repetition)]
    )
    return random_mask(count, probability, np.random.default_rng(seed))


def apply_strict_missing(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    quality: np.ndarray,
    mask: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    masked_arrays = tuple(
        value * mask[:, modality : modality + 1]
        for modality, value in enumerate(arrays)
    )
    masked_quality = quality * mask.repeat(2, axis=1)
    return masked_arrays, masked_quality


@torch.no_grad()
def raw_modality_invariance(
    experts: base.ExpertSet,
    gate: torch.nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    quality: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> dict[str, float | bool]:
    masked_arrays, masked_quality = apply_strict_missing(arrays, quality, mask)
    baseline_means, baseline_log_variances = base.predict_experts(
        experts, masked_arrays, device
    )
    baseline_prediction, baseline_weights = base.fuse_predictions(
        "oof_risk_gate",
        gate,
        masked_quality,
        baseline_means,
        baseline_log_variances,
        mask,
        device,
    )
    altered = [value.copy() for value in masked_arrays]
    for modality, value in enumerate(altered):
        unavailable = mask[:, modality] == 0
        value[unavailable] = 1000.0 + modality
    altered_means, altered_log_variances = base.predict_experts(
        experts, tuple(altered), device
    )
    altered_quality = masked_quality + (1.0 - mask.repeat(2, axis=1)) * 1000.0
    altered_prediction, altered_weights = base.fuse_predictions(
        "oof_risk_gate",
        gate,
        altered_quality,
        altered_means,
        altered_log_variances,
        mask,
        device,
    )
    prediction_difference = float(
        np.max(np.abs(baseline_prediction - altered_prediction))
    )
    weight_difference = float(np.max(np.abs(baseline_weights - altered_weights)))
    tolerance = 1e-7
    return {
        "max_prediction_difference": prediction_difference,
        "max_weight_difference": weight_difference,
        "tolerance": tolerance,
        "passed": prediction_difference <= tolerance and weight_difference <= tolerance,
    }


def hierarchical_bootstrap(
    differences: np.ndarray,
    companies: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap [training_seed, scenario, mask_draw, call] differences."""

    if differences.ndim != 4:
        raise ValueError("differences must have shape [seed, scenario, draw, call]")
    seed_count, scenario_count, draw_count, call_count = differences.shape
    companies = companies.astype(str)
    if len(companies) != call_count:
        raise ValueError("company labels do not match the difference tensor")
    unique = np.unique(companies)
    blocks = [np.flatnonzero(companies == company) for company in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for bootstrap in range(repetitions):
        chosen_blocks = rng.integers(0, len(blocks), size=len(blocks))
        call_indices = np.concatenate([blocks[index] for index in chosen_blocks])
        chosen_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_values: list[float] = []
        for chosen_seed in chosen_seeds:
            scenario_values: list[float] = []
            for scenario in range(scenario_count):
                chosen_draws = rng.integers(0, draw_count, size=draw_count)
                selected = differences[
                    chosen_seed,
                    scenario,
                    chosen_draws,
                ][:, call_indices]
                scenario_values.append(float(selected.mean()))
            seed_values.append(float(np.mean(scenario_values)))
        samples[bootstrap] = float(np.mean(seed_values))
    return {
        "mean_equal_weight_minus_gate": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_repetitions": int(repetitions),
        "cik_blocks": int(len(unique)),
        "training_seeds": int(seed_count),
        "scenarios": int(scenario_count),
        "paired_mask_draws_per_seed_scenario": int(draw_count),
    }


def aggregate_secondary(results: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model, scenarios in results.items():
        output[model] = {}
        for scenario, values in scenarios.items():
            output[model][scenario] = {
                "seed_results": values,
                "qlike_mean": float(np.mean([item["point"]["qlike"] for item in values])),
                "qlike_std": float(np.std([item["point"]["qlike"] for item in values])),
                "conformal_coverage_mean": float(
                    np.mean([item["conformal_coverage"] for item in values])
                ),
                "conformal_coverage_std": float(
                    np.std([item["conformal_coverage"] for item in values])
                ),
            }
    return output


def write_per_call_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_self_test() -> dict[str, Any]:
    strict = strict_gate_self_test()
    mask_a = strict_mask(64, 0.25, 3407, 0)
    mask_b = strict_mask(64, 0.25, 3407, 0)
    mask_c = strict_mask(64, 0.25, 3407, 1)
    synthetic = np.ones((2, 2, 3, 8), dtype=float) * 0.1
    bootstrap = hierarchical_bootstrap(
        synthetic,
        np.asarray([str(index) for index in range(8)]),
        repetitions=200,
        seed=29,
    )
    passed = bool(
        strict["passed"]
        and np.array_equal(mask_a, mask_b)
        and not np.array_equal(mask_a, mask_c)
        and abs(float(bootstrap["mean_equal_weight_minus_gate"]) - 0.1) < 1e-12
        and abs(float(bootstrap["ci95_low"]) - 0.1) < 1e-12
        and abs(float(bootstrap["ci95_high"]) - 0.1) < 1e-12
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "strict_gate_invariance": strict,
        "mask_reproducible": bool(np.array_equal(mask_a, mask_b)),
        "distinct_repetitions_differ": bool(not np.array_equal(mask_a, mask_c)),
        "bootstrap_constant_signal": bootstrap,
    }


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "PASS":
            raise SystemExit("Locked forward-confirmation self-test failed.")
        return

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Activate the "ahenk" conda environment first.')
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; rerun with --device cpu.")
    torch.set_num_threads(4)
    torch.manual_seed(MASK_RNG_NAMESPACE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MASK_RNG_NAMESPACE)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    root = Path(args.project_root).resolve()
    paths = {
        "forward_freeze": root / FORWARD_FREEZE,
        "feature_freeze": root / FEATURE_FREEZE,
        "external_features": root / EXTERNAL_FEATURES,
        "model_index": root / MODEL_INDEX,
        "model_tabular": root / MODEL_TABULAR,
        "protocol_output": root / PROTOCOL_OUTPUT,
        "report_output": root / REPORT_OUTPUT,
        "per_call_output": root / PER_CALL_OUTPUT,
        "script": Path(__file__).resolve(),
    }
    missing = [
        str(path.relative_to(root))
        for key, path in paths.items()
        if key not in {"protocol_output", "report_output", "per_call_output"}
        and not path.is_file()
    ]
    if missing:
        raise SystemExit("Missing required frozen inputs: " + ", ".join(missing))
    if paths["report_output"].exists() or paths["per_call_output"].exists():
        raise SystemExit("Locked Earnings25 confirmation output already exists; refusing a rerun.")

    forward = read_json(paths["forward_freeze"])
    feature_freeze = read_json(paths["feature_freeze"])
    if forward.get("status") != "FROZEN_BEFORE_ANY_PRICE_OR_TARGET_ACCESS":
        raise SystemExit("Unexpected forward-cohort freeze status.")
    if feature_freeze.get("status") != "FROZEN_EARNINGS25_FORWARD_CONFIRMATION_FEATURES":
        raise SystemExit("Model-ready Earnings25 features are not frozen.")
    feature_info = feature_freeze.get("outputs", {}).get("features", {})
    if feature_info.get("sha256") != sha256(paths["external_features"]):
        raise SystemExit("External feature hash differs from its freeze report.")
    if feature_freeze.get("disjointness", {}).get("overlap_with_original_call_ids") != 0:
        raise SystemExit("External call IDs overlap original modeling data.")
    if feature_freeze.get("disjointness", {}).get("overlap_with_original_ticker_strings") != 0:
        raise SystemExit("External ticker strings overlap original modeling identifiers.")

    analysis = forward["prespecified_primary_analysis"]
    expected_scenarios = list(PRIMARY_SCENARIOS)
    if analysis.get("scenarios") != expected_scenarios:
        raise SystemExit("Frozen primary scenarios changed.")
    mask_repetitions = int(analysis["mask_draw_repetitions_per_seed_and_scenario"])
    bootstrap_repetitions = int(analysis["bootstrap_repetitions"])
    if mask_repetitions != 20 or bootstrap_repetitions != 10_000:
        raise SystemExit("Frozen repetition counts changed.")
    seeds = tuple(int(value) for value in forward["frozen_model_recipe"]["seeds"])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise SystemExit("Expected five unique frozen training seeds.")
    development_path = root / forward["frozen_model_recipe"]["development_report"]
    if sha256(development_path) != forward["frozen_model_recipe"]["development_report_sha256"]:
        raise SystemExit("Frozen strict-missing development report changed.")
    development = read_json(development_path)
    if development.get("status") != "PASS":
        raise SystemExit("Strict-missing development report did not pass.")
    if tuple(int(value) for value in development.get("seeds", [])) != seeds:
        raise SystemExit("Development seeds differ from the forward freeze.")
    if development.get("best_primary_model_by_development_robust_qlike") != "oof_risk_gate":
        raise SystemExit("The frozen development-selected model is not oof_risk_gate.")

    immutable_protocol = {
        "stage": "ONE_TIME_LOCKED_EARNINGS25_FORWARD_CONFIRMATION",
        "forward_freeze_sha256": sha256(paths["forward_freeze"]),
        "feature_freeze_sha256": sha256(paths["feature_freeze"]),
        "external_features_sha256": sha256(paths["external_features"]),
        "development_report_sha256": sha256(development_path),
        "model_index_sha256": sha256(paths["model_index"]),
        "model_tabular_sha256": sha256(paths["model_tabular"]),
        "evaluation_script_sha256": sha256(paths["script"]),
        "training_seeds": list(seeds),
        "primary_scenarios": expected_scenarios,
        "mask_draw_repetitions_per_seed_scenario": mask_repetitions,
        "mask_rng": (
            "numpy SeedSequence([20261003, training_seed, missing_percent, repetition])"
        ),
        "pairing": "identical availability mask for equal weighting and AHENK",
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_levels": ["CIK block", "training seed", "paired missingness draw"],
        "cohort_or_recipe_changes_allowed": False,
        "model_or_hyperparameter_selection_on_earnings25_allowed": False,
    }
    if paths["protocol_output"].exists():
        protocol = read_json(paths["protocol_output"])
        for key, expected in immutable_protocol.items():
            if protocol.get(key) != expected:
                raise SystemExit(f"Locked execution protocol changed: {key}")
    else:
        write_json_atomic(
            paths["protocol_output"],
            {"created_at_utc": now_utc(), **immutable_protocol},
        )

    index = pd.read_csv(paths["model_index"])
    tabular = pd.read_parquet(paths["model_tabular"])
    frame = tabular.merge(
        index[["call_id", "text_path"]],
        on="call_id",
        how="left",
        validate="one_to_one",
    ).sort_values(["call_date", "call_id"])
    train = frame[frame["temporal_split"] == "train"].reset_index(drop=True)
    dev = frame[frame["temporal_split"] == "dev"].reset_index(drop=True)
    calibration = dev.iloc[: len(dev) // 2].reset_index(drop=True)
    external = pd.read_parquet(paths["external_features"]).sort_values(
        ["call_date", "call_id"]
    ).reset_index(drop=True)
    if len(train) != EXPECTED_TRAIN_CALLS or len(calibration) != EXPECTED_CALIBRATION_CALLS:
        raise SystemExit(
            f"Frozen modeling sizes changed: train={len(train)}, calibration={len(calibration)}"
        )
    if len(external) != EXPECTED_EXTERNAL_CALLS or external["company_id"].nunique() != EXPECTED_EXTERNAL_BLOCKS:
        raise SystemExit(
            f"Frozen external size changed: calls={len(external)}, "
            f"CIK blocks={external['company_id'].nunique()}"
        )
    if not np.isfinite(external[TARGET].to_numpy(float)).all() or not (external[TARGET] > 0).all():
        raise SystemExit("External targets are not positive and finite.")

    acoustic_columns = [
        column
        for column in train
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    price_columns = [column for column in train if column.startswith("price_")]
    missing_external = [
        column for column in acoustic_columns + price_columns if column not in external
    ]
    if missing_external:
        raise SystemExit(f"External features lack frozen columns: {missing_external[:10]}")

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
    companies = external["company_id"].astype(str).to_numpy()

    seed_count = len(seeds)
    scenario_names = list(PRIMARY_SCENARIOS)
    scenario_count = len(scenario_names)
    call_count = len(external)
    primary_losses = {
        model: np.empty(
            (seed_count, scenario_count, mask_repetitions, call_count),
            dtype="float64",
        )
        for model in PRIMARY_MODELS
    }
    primary_coverages: dict[str, dict[str, list[float]]] = {
        model: {scenario: [] for scenario in scenario_names}
        for model in PRIMARY_MODELS
    }
    primary_weight_sums: dict[str, dict[str, list[list[float]]]] = {
        model: {scenario: [] for scenario in scenario_names}
        for model in PRIMARY_MODELS
    }
    matched_price_results: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in scenario_names
    }
    secondary_results: dict[str, dict[str, list[dict[str, Any]]]] = {
        model: {scenario: [] for scenario in SECONDARY_SCENARIOS}
        for model in SECONDARY_MODELS
    }
    invariance_results: list[dict[str, Any]] = []
    per_call_rows: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        print(f"Seed {seed_index + 1}/{len(seeds)} ({seed}): frozen OOF experts and strict gate")
        oof_means, oof_log_variances, valid = base.build_oof_predictions(
            train_arrays, train_target, device, seed
        )
        gate = train_strict_gate(
            quality_train[valid],
            oof_means[valid],
            oof_log_variances[valid],
            train_target[valid],
            device,
            seed,
        )
        internal_split = int(len(train) * 0.90)
        experts = base.train_experts(
            tuple(value[:internal_split] for value in train_arrays),
            train_target[:internal_split],
            tuple(value[internal_split:] for value in train_arrays),
            train_target[internal_split:],
            device,
            seed + 100,
            epochs=90,
            patience=12,
        )

        calibration_means, calibration_log_variances = base.predict_experts(
            experts, calibration_arrays, device
        )
        calibration_mask = np.ones((len(calibration), 3), dtype="float32")
        radii: dict[str, float] = {}
        for model in SECONDARY_MODELS:
            log_prediction, _ = base.fuse_predictions(
                model,
                gate,
                quality_calibration,
                calibration_means,
                calibration_log_variances,
                calibration_mask,
                device,
            )
            radii[model] = base.conformal_quantile(
                calibration_target, log_prediction, level=0.90
            )

        stressed = stress_scenarios(external_arrays, quality_external, seed)
        for scenario in SECONDARY_SCENARIOS:
            scenario_arrays, scenario_quality, scenario_mask = stressed[scenario]
            means, log_variances = base.predict_experts(
                experts, scenario_arrays, device
            )
            for model in SECONDARY_MODELS:
                log_prediction, weights = base.fuse_predictions(
                    model,
                    gate,
                    scenario_quality,
                    means,
                    log_variances,
                    scenario_mask,
                    device,
                )
                prediction = np.exp(log_prediction)
                secondary_results[model][scenario].append(
                    {
                        "seed": seed,
                        "point": metric_set(external_target, prediction),
                        "conformal_radius_from_clean_calibration": radii[model],
                        "conformal_coverage": coverage(
                            external_target, log_prediction, radii[model]
                        ),
                        "mean_weights": {
                            modality: float(weights[:, modality_index].mean())
                            for modality_index, modality in enumerate(base.MODALITIES)
                        },
                    }
                )

        for scenario_index, (scenario, probability) in enumerate(
            PRIMARY_SCENARIOS.items()
        ):
            for repetition in range(mask_repetitions):
                mask = strict_mask(call_count, probability, seed, repetition)
                masked_arrays, masked_quality = apply_strict_missing(
                    external_arrays, quality_external, mask
                )
                means, log_variances = base.predict_experts(
                    experts, masked_arrays, device
                )
                predictions: dict[str, np.ndarray] = {}
                log_predictions: dict[str, np.ndarray] = {}
                weights_by_model: dict[str, np.ndarray] = {}
                for model in PRIMARY_MODELS:
                    log_prediction, weights = base.fuse_predictions(
                        model,
                        gate,
                        masked_quality,
                        means,
                        log_variances,
                        mask,
                        device,
                    )
                    prediction = np.exp(log_prediction)
                    log_predictions[model] = log_prediction
                    predictions[model] = prediction
                    weights_by_model[model] = weights
                    primary_losses[model][seed_index, scenario_index, repetition] = qlike_per_call(
                        external_target, prediction
                    )
                    primary_coverages[model][scenario].append(
                        coverage(external_target, log_prediction, radii[model])
                    )
                    primary_weight_sums[model][scenario].append(
                        [float(weights[:, index].mean()) for index in range(3)]
                    )

                price_available = mask[:, 2].astype(bool)
                price_log_prediction = means[:, 2]
                price_prediction = np.exp(price_log_prediction)
                matched_price_results[scenario].append(
                    {
                        "seed": seed,
                        "repetition": repetition,
                        "available_calls": int(price_available.sum()),
                        "price_only_qlike": float(
                            qlike_per_call(
                                external_target[price_available],
                                price_prediction[price_available],
                            ).mean()
                        ),
                        "equal_weight_qlike": float(
                            primary_losses["fixed_mean"][
                                seed_index, scenario_index, repetition, price_available
                            ].mean()
                        ),
                        "gate_qlike": float(
                            primary_losses["oof_risk_gate"][
                                seed_index, scenario_index, repetition, price_available
                            ].mean()
                        ),
                        "price_only_conformal_coverage": coverage(
                            external_target[price_available],
                            price_log_prediction[price_available],
                            radii["price_expert"],
                        ),
                    }
                )

                if scenario == "random_50_strict_missing" and repetition == 0:
                    derived = masked_information_invariance(
                        gate,
                        masked_quality,
                        means,
                        log_variances,
                        mask,
                        device,
                    )
                    raw = raw_modality_invariance(
                        experts,
                        gate,
                        external_arrays,
                        quality_external,
                        mask,
                        device,
                    )
                    invariance = {
                        "seed": seed,
                        "derived_information": derived,
                        "raw_modality": raw,
                        "passed": bool(derived["passed"] and raw["passed"]),
                    }
                    invariance_results.append(invariance)
                    if not invariance["passed"]:
                        raise RuntimeError("Strict-missing invariance check failed.")

                fixed_losses = primary_losses["fixed_mean"][
                    seed_index, scenario_index, repetition
                ]
                gate_losses = primary_losses["oof_risk_gate"][
                    seed_index, scenario_index, repetition
                ]
                for call_index, source in external.iterrows():
                    per_call_rows.append(
                        {
                            "call_id": source["call_id"],
                            "company_id": source["company_id"],
                            "call_date": source["call_date"],
                            "training_seed": seed,
                            "scenario": scenario,
                            "mask_repetition": repetition,
                            "text_available": int(mask[call_index, 0]),
                            "acoustic_available": int(mask[call_index, 1]),
                            "price_available": int(mask[call_index, 2]),
                            "actual": format(float(external_target[call_index]), ".12g"),
                            "equal_weight_prediction": format(
                                float(predictions["fixed_mean"][call_index]), ".12g"
                            ),
                            "gate_prediction": format(
                                float(predictions["oof_risk_gate"][call_index]), ".12g"
                            ),
                            "equal_weight_qlike": format(
                                float(fixed_losses[call_index]), ".12g"
                            ),
                            "gate_qlike": format(float(gate_losses[call_index]), ".12g"),
                            "equal_weight_minus_gate_qlike": format(
                                float(fixed_losses[call_index] - gate_losses[call_index]),
                                ".12g",
                            ),
                        }
                    )

    differences = primary_losses["fixed_mean"] - primary_losses["oof_risk_gate"]
    primary_interval = hierarchical_bootstrap(
        differences,
        companies,
        repetitions=bootstrap_repetitions,
        seed=20261003,
    )
    scenario_intervals: dict[str, Any] = {}
    scenario_directions: dict[str, bool] = {}
    primary_models: dict[str, Any] = {}
    for model in PRIMARY_MODELS:
        primary_models[model] = {}
        for scenario_index, scenario in enumerate(scenario_names):
            losses = primary_losses[model][:, scenario_index : scenario_index + 1]
            weights = np.asarray(primary_weight_sums[model][scenario], dtype=float)
            primary_models[model][scenario] = {
                "qlike_mean": float(losses.mean()),
                "qlike_seed_draw_std": float(
                    primary_losses[model][:, scenario_index].mean(axis=2).std()
                ),
                "conformal_coverage_mean": float(
                    np.mean(primary_coverages[model][scenario])
                ),
                "conformal_coverage_std": float(
                    np.std(primary_coverages[model][scenario])
                ),
                "mean_weights": {
                    modality: float(weights[:, index].mean())
                    for index, modality in enumerate(base.MODALITIES)
                },
            }
        primary_models[model]["two_scenario_mean_qlike"] = float(
            primary_losses[model].mean()
        )
    for scenario_index, scenario in enumerate(scenario_names):
        interval = hierarchical_bootstrap(
            differences[:, scenario_index : scenario_index + 1],
            companies,
            repetitions=bootstrap_repetitions,
            seed=20261004 + scenario_index,
        )
        scenario_intervals[scenario] = interval
        scenario_directions[scenario] = bool(
            primary_models["oof_risk_gate"][scenario]["qlike_mean"]
            < primary_models["fixed_mean"][scenario]["qlike_mean"]
        )

    fixed_primary = float(primary_losses["fixed_mean"].mean())
    gate_primary = float(primary_losses["oof_risk_gate"].mean())
    relative_reduction = 100.0 * (fixed_primary - gate_primary) / max(
        fixed_primary, EPSILON
    )
    interval_excludes_zero = bool(
        primary_interval["ci95_low"] > 0 or primary_interval["ci95_high"] < 0
    )
    success = bool(
        primary_interval["mean_equal_weight_minus_gate"] > 0
        and primary_interval["ci95_low"] > 0
        and all(scenario_directions.values())
    )

    matched_price_summary = {
        scenario: {
            "draws": len(values),
            "mean_available_calls": float(
                np.mean([item["available_calls"] for item in values])
            ),
            "price_only_qlike": float(
                np.mean([item["price_only_qlike"] for item in values])
            ),
            "equal_weight_qlike_on_price_available_rows": float(
                np.mean([item["equal_weight_qlike"] for item in values])
            ),
            "gate_qlike_on_price_available_rows": float(
                np.mean([item["gate_qlike"] for item in values])
            ),
            "price_only_conformal_coverage": float(
                np.mean([item["price_only_conformal_coverage"] for item in values])
            ),
        }
        for scenario, values in matched_price_results.items()
    }

    write_per_call_atomic(paths["per_call_output"], per_call_rows)
    report = {
        "created_at_utc": now_utc(),
        "execution_status": "COMPLETE",
        "status": "CONFIRMED" if success else "NOT_CONFIRMED",
        "stage": "ONE_TIME_LOCKED_EARNINGS25_FORWARD_CONFIRMATION",
        "device": str(device),
        "deterministic_torch_algorithms_requested": True,
        "training_seeds": list(seeds),
        "train_calls": len(train),
        "calibration_calls": len(calibration),
        "external_calls": len(external),
        "external_cik_blocks": int(external["company_id"].nunique()),
        "external_date_min": str(external["call_date"].min()),
        "external_date_max": str(external["call_date"].max()),
        "text_preprocessing": text_info,
        "primary_analysis": {
            "comparison": analysis["comparison"],
            "scenarios": scenario_names,
            "scenario_aggregation": analysis["scenario_aggregation"],
            "mask_draw_repetitions_per_seed_and_scenario": mask_repetitions,
            "equal_weight_two_scenario_mean_qlike": fixed_primary,
            "gate_two_scenario_mean_qlike": gate_primary,
            "relative_qlike_reduction_percent": relative_reduction,
            "hierarchical_bootstrap": primary_interval,
            "two_sided_interval_excludes_zero": interval_excludes_zero,
            "direction_gate_better_by_scenario": scenario_directions,
            "success_rule_passed": success,
            "frozen_success_rule": analysis["success_rule"],
        },
        "primary_models": primary_models,
        "scenario_hierarchical_bootstrap": scenario_intervals,
        "strict_missing_implementation": {
            "masked_raw_modality_zeroed_before_expert_prediction": True,
            "masked_quality_zeroed": True,
            "masked_expert_mean_zeroed_inside_gate": True,
            "masked_expert_log_variance_zeroed_inside_gate": True,
            "masked_final_weight_zeroed": True,
            "availability_mask_supplied_to_gate": True,
            "paired_masks_for_compared_methods": True,
            "invariance_checks": invariance_results,
        },
        "secondary_analyses": {
            "full_and_prespecified_corruptions": aggregate_secondary(
                secondary_results
            ),
            "price_only_under_matched_availability": matched_price_summary,
            "conformal_rule": (
                "unchanged 90% radii estimated on the original clean calibration split"
            ),
            "larger_forward_verified_cohort": {
                "status": "NOT_RUN_IN_PRIMARY_LOCKED_TEST",
                "reason": (
                    "requires its own separately frozen text/acoustic/market feature set; "
                    "it is not part of the primary success decision"
                ),
            },
        },
        "interpretation_constraints": {
            "historical_security_master_claimed": False,
            "intraday_call_timing_claimed": False,
            "model_or_hyperparameter_selection_on_earnings25": False,
            "rerun_for_model_selection_allowed": False,
            "primary_decision_must_be_reported_regardless_of_direction": True,
        },
        "inputs": {
            "forward_freeze": {
                "path": FORWARD_FREEZE.as_posix(),
                "sha256": sha256(paths["forward_freeze"]),
            },
            "feature_freeze": {
                "path": FEATURE_FREEZE.as_posix(),
                "sha256": sha256(paths["feature_freeze"]),
            },
            "external_features": {
                "path": EXTERNAL_FEATURES.as_posix(),
                "sha256": sha256(paths["external_features"]),
            },
            "development_report": {
                "path": development_path.relative_to(root).as_posix(),
                "sha256": sha256(development_path),
            },
            "model_index": {
                "path": MODEL_INDEX.as_posix(),
                "sha256": sha256(paths["model_index"]),
            },
            "model_tabular": {
                "path": MODEL_TABULAR.as_posix(),
                "sha256": sha256(paths["model_tabular"]),
            },
            "execution_protocol": {
                "path": PROTOCOL_OUTPUT.as_posix(),
                "sha256": sha256(paths["protocol_output"]),
            },
        },
        "outputs": {
            "report": REPORT_OUTPUT.as_posix(),
            "per_call": {
                "path": PER_CALL_OUTPUT.as_posix(),
                "sha256": sha256(paths["per_call_output"]),
                "rows": len(per_call_rows),
            },
        },
    }
    write_json_atomic(paths["report_output"], report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "status": report["status"],
                "primary_analysis": report["primary_analysis"],
                "report": REPORT_OUTPUT.as_posix(),
                "per_call": report["outputs"]["per_call"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()