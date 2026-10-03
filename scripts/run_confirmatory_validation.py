"""P02-AHENK birleşik doğrulayıcı deney boru hattı.

Tek çalıştırmada üç ileri-zaman katı ve üç seed üzerinde:
- fiyat uzmanı,
- gürültü artırımlı sabit uzman füzyonu,
- yalnız eksiklik artırımlı AHENK,
- eksiklik + kaliteyle eşlenmiş gürültü artırımlı AHENK
karşılaştırılır. Tam veri, kip kaybı ve sürekli gürültü stresleri ölçülür;
şirket-blok bootstrap ve otomatik doğrulayıcı karar tek JSON'a yazılır.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_ahenk_expert_pilot import ExpertFusion, predict, torch_qlike
from run_ahenk_pilot import (
    EPSILON,
    TARGET,
    fit_dense_text,
    fit_numeric,
    metric_set,
    quality_features,
    random_mask,
    tensors,
)
from run_extended_validation import make_folds, stress_scenarios


MODEL_SPECS = {
    "price_expert": {
        "mode": "price",
        "dropout": 0.0,
        "missing_weight": 0.0,
        "noise_weight": 0.0,
    },
    "fixed_noise_aug_25": {
        "mode": "fixed",
        "dropout": 0.25,
        "missing_weight": 0.5,
        "noise_weight": 0.4,
    },
    "ahenk_missing_only_25": {
        "mode": "gate",
        "dropout": 0.25,
        "missing_weight": 0.5,
        "noise_weight": 0.0,
    },
    "ahenk_noise_aug_25": {
        "mode": "gate",
        "dropout": 0.25,
        "missing_weight": 0.5,
        "noise_weight": 0.4,
    },
}

CONFIRMATORY_SCENARIOS = (
    "full",
    "random_25_missing",
    "random_50_missing",
    "text_noise_05",
    "acoustic_noise_10",
    "price_noise_05",
)
BOOTSTRAP_COMPARISONS = (
    ("ahenk_noise_aug_25", "fixed_noise_aug_25"),
    ("ahenk_noise_aug_25", "ahenk_missing_only_25"),
    ("ahenk_noise_aug_25", "price_expert"),
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("En az bir seed gereklidir.")
    return seeds


def augment_batch_noise(
    arrays: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    quality: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Her aktif örnekte bir kipi bozar ve ilgili kalite değerini düşürür."""
    corrupted = [value.clone() for value in arrays]
    corrupted_quality = quality.clone()
    count = len(quality)
    active = torch.rand(count, device=quality.device) < 0.70
    modality = torch.randint(0, 3, (count,), device=quality.device)
    strength = 0.25 + 0.75 * torch.rand(count, device=quality.device)

    for index in range(3):
        selected = active & (modality == index)
        if not torch.any(selected):
            continue
        sigma = strength[selected].unsqueeze(1)
        corrupted[index][selected] += torch.randn_like(corrupted[index][selected]) * sigma
        factor = (1.0 / (1.0 + 2.0 * strength[selected])).unsqueeze(1)
        corrupted_quality[selected, 2 * index : 2 * index + 2] *= factor
    return tuple(corrupted), corrupted_quality  # type: ignore[return-value]


def train_confirmatory(
    dimensions: tuple[int, int, int],
    specification: dict,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    dev_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_quality: np.ndarray,
    dev_quality: np.ndarray,
    train_target: np.ndarray,
    dev_target: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[ExpertFusion, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ExpertFusion(dimensions, specification["mode"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(
        tensors(train_arrays, train_quality, train_target),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    lower = float(max(np.quantile(train_target, 0.001), EPSILON))
    upper = float(max(np.quantile(train_target, 0.999), EPSILON * 10))
    dev_scenarios = stress_scenarios(dev_arrays, dev_quality, seed)
    selection_scenarios = (
        "full",
        "random_25_missing",
        "random_50_missing",
        "acoustic_noise_10",
        "price_noise_05",
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for text, acoustic, price, quality, base_mask, log_target in loader:
            arrays = (text.to(device), acoustic.to(device), price.to(device))
            quality = quality.to(device)
            log_target = log_target.to(device)
            full_mask = base_mask.to(device)
            optimizer.zero_grad(set_to_none=True)

            full_log, expert_logs, _ = model(*arrays, quality, full_mask)
            loss = torch_qlike(log_target, full_log)

            if specification["missing_weight"] > 0:
                missing_mask = torch.from_numpy(
                    random_mask(len(log_target), specification["dropout"], rng)
                ).to(device)
                missing_log, _, _ = model(*arrays, quality, missing_mask)
                loss = loss + specification["missing_weight"] * torch_qlike(
                    log_target, missing_log
                )

            if specification["noise_weight"] > 0:
                noisy_arrays, noisy_quality = augment_batch_noise(arrays, quality)
                noisy_log, _, _ = model(
                    *noisy_arrays, noisy_quality, full_mask
                )
                loss = loss + specification["noise_weight"] * torch_qlike(
                    log_target, noisy_log
                )

            expert_loss = torch.mean(
                torch.stack(
                    [torch_qlike(log_target, expert_logs[:, index]) for index in range(3)]
                )
            )
            loss = loss + 0.15 * expert_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = []
        for scenario in selection_scenarios:
            arrays_eval, quality_eval, mask_eval = dev_scenarios[scenario]
            prediction, _, _ = predict(
                model,
                arrays_eval,
                quality_eval,
                mask_eval,
                device,
                lower,
                upper,
            )
            actual = np.maximum(dev_target, EPSILON)
            ratio = actual / np.maximum(prediction, EPSILON)
            scores.append(float(np.mean(ratio - np.log(ratio) - 1.0)))
        score = float(np.mean(scores))
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_dev_confirmatory_qlike": best_score,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def aggregate_seed_results(seed_results: list[dict]) -> dict:
    output = {}
    for scenario in CONFIRMATORY_SCENARIOS:
        output[scenario] = {}
        for metric in seed_results[0]["scenarios"][scenario]:
            values = [item["scenarios"][scenario][metric] for item in seed_results]
            output[scenario][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    output["confirmatory_robust_qlike"] = float(
        np.mean([output[name]["qlike"]["mean"] for name in CONFIRMATORY_SCENARIOS])
    )
    return output


def qlike_loss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    predicted = np.maximum(np.asarray(predicted, dtype=float), EPSILON)
    ratio = actual / predicted
    return ratio - np.log(ratio) - 1.0


def loss_pivot(predictions: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "confirmatory_composite":
        selected = predictions[predictions["scenario"].isin(CONFIRMATORY_SCENARIOS)].copy()
        selected["loss"] = qlike_loss(selected["actual"], selected["predicted"])
        losses = (
            selected.groupby(
                ["call_id", "company_id", "seed", "model"], as_index=False
            )["loss"]
            .mean()
        )
    else:
        selected = predictions[predictions["scenario"] == scenario].copy()
        selected["loss"] = qlike_loss(selected["actual"], selected["predicted"])
        losses = selected[["call_id", "company_id", "seed", "model", "loss"]]
    return losses.pivot_table(
        index=["call_id", "company_id", "seed"],
        columns="model",
        values="loss",
        aggfunc="first",
    ).reset_index()


def bootstrap_compare(
    pivot: pd.DataFrame,
    candidate: str,
    reference: str,
    iterations: int,
    rng: np.random.Generator,
) -> dict:
    paired = pivot[["company_id", candidate, reference]].dropna()
    paired = paired.assign(delta=paired[candidate] - paired[reference])
    grouped = paired.groupby("company_id")["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    draws = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        draws[start : start + size] = sums[indices].sum(1) / counts[indices].sum(1)
    candidate_mean = float(paired[candidate].mean())
    reference_mean = float(paired[reference].mean())
    delta = candidate_mean - reference_mean
    interval = np.quantile(draws, [0.025, 0.975])
    return {
        "candidate": candidate,
        "reference": reference,
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


def build_bootstrap_report(
    predictions: pd.DataFrame, iterations: int
) -> dict:
    rng = np.random.default_rng(20260927)
    scenarios = (*CONFIRMATORY_SCENARIOS, "confirmatory_composite")
    report = {}
    for scenario in scenarios:
        pivot = loss_pivot(predictions, scenario)
        report[scenario] = [
            bootstrap_compare(pivot, candidate, reference, iterations, rng)
            for candidate, reference in BOOTSTRAP_COMPARISONS
        ]
    return report


def decision_from_bootstrap(bootstrap: dict) -> dict:
    composite = bootstrap["confirmatory_composite"]
    gate_vs_fixed = next(
        item
        for item in composite
        if item["candidate"] == "ahenk_noise_aug_25"
        and item["reference"] == "fixed_noise_aug_25"
    )
    acoustic = next(
        item
        for item in bootstrap["acoustic_noise_10"]
        if item["candidate"] == "ahenk_noise_aug_25"
        and item["reference"] == "fixed_noise_aug_25"
    )
    random50 = next(
        item
        for item in bootstrap["random_50_missing"]
        if item["candidate"] == "ahenk_noise_aug_25"
        and item["reference"] == "fixed_noise_aug_25"
    )
    if (
        gate_vs_fixed["conclusion"] == "SUPPORTED"
        and acoustic["delta_candidate_minus_reference"] < 0
        and random50["delta_candidate_minus_reference"] < 0
    ):
        decision = "PASS_CONFIRMATORY"
    elif gate_vs_fixed["delta_candidate_minus_reference"] < 0:
        decision = "PARTIAL_CONFIRMATORY"
    else:
        decision = "REDESIGN_REQUIRED"
    return {
        "decision": decision,
        "required_primary_comparison": "noise-augmented AHENK vs noise-augmented fixed fusion",
        "composite_conclusion": gate_vs_fixed["conclusion"],
        "acoustic_noise_direction": "better" if acoustic["delta_candidate_minus_reference"] < 0 else "worse",
        "random50_direction": "better" if random50["delta_candidate_minus_reference"] < 0 else "worse",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=parse_seeds, default=(3407, 2026, 2709))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA istendi fakat kullanılamıyor.")
    torch.set_num_threads(4)
    device = torch.device(args.device)

    root = Path(args.project_root).resolve()
    index = pd.read_csv(root / "data/processed/ahenk_dataset_index.csv")
    tabular = pd.read_parquet(root / "data/processed/ahenk_tabular_features.parquet")
    frame = tabular.merge(
        index[["call_id", "text_path"]], on="call_id", how="left", validate="one_to_one"
    ).sort_values(["call_date", "call_id"])
    acoustic_columns = [
        column
        for column in frame
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    price_columns = [column for column in frame if column.startswith("price_")]

    folds_report = {}
    predictions = []
    for fold in make_folds(frame):
        name = fold["name"]
        train, dev, test = fold["train"], fold["dev"], fold["test"]
        print(f"{name}: train={len(train)}, dev={len(dev)}, test={len(test)}")
        text_train, text_dev, text_test, text_info = fit_dense_text(
            root, train, dev, test, dimensions=128
        )
        acoustic_train, acoustic_dev, acoustic_test = fit_numeric(
            train, dev, test, acoustic_columns
        )
        price_train, price_dev, price_test = fit_numeric(
            train, dev, test, price_columns
        )
        train_quality, dev_quality, test_quality = quality_features(
            train, (train, dev, test)
        )
        train_arrays = (text_train, acoustic_train, price_train)
        dev_arrays = (text_dev, acoustic_dev, price_dev)
        test_arrays = (text_test, acoustic_test, price_test)
        dimensions = tuple(array.shape[1] for array in train_arrays)
        train_target = train[TARGET].to_numpy(float)
        dev_target = dev[TARGET].to_numpy(float)
        test_target = test[TARGET].to_numpy(float)
        lower = float(max(np.quantile(train_target, 0.001), EPSILON))
        upper = float(max(np.quantile(train_target, 0.999), EPSILON * 10))

        model_reports = {}
        for model_name, specification in MODEL_SPECS.items():
            print(f"  {model_name}")
            seed_results = []
            for seed in args.seeds:
                model, training = train_confirmatory(
                    dimensions,
                    specification,
                    train_arrays,
                    dev_arrays,
                    train_quality,
                    dev_quality,
                    train_target,
                    dev_target,
                    device,
                    seed,
                    args.epochs,
                    args.patience,
                )
                scenario_results = {}
                for scenario, (
                    scenario_arrays,
                    scenario_quality,
                    scenario_mask,
                ) in stress_scenarios(test_arrays, test_quality, seed).items():
                    if scenario not in CONFIRMATORY_SCENARIOS:
                        continue
                    prediction, _, _ = predict(
                        model,
                        scenario_arrays,
                        scenario_quality,
                        scenario_mask,
                        device,
                        lower,
                        upper,
                    )
                    scenario_results[scenario] = metric_set(test_target, prediction)
                    for call_id, company_id, actual, predicted in zip(
                        test["call_id"],
                        test["company_id"],
                        test_target,
                        prediction,
                        strict=True,
                    ):
                        predictions.append(
                            {
                                "fold": name,
                                "model": model_name,
                                "seed": seed,
                                "scenario": scenario,
                                "call_id": call_id,
                                "company_id": company_id,
                                "actual": actual,
                                "predicted": predicted,
                            }
                        )
                seed_results.append(
                    {"seed": seed, "training": training, "scenarios": scenario_results}
                )
            model_reports[model_name] = {
                "specification": specification,
                "seed_results": seed_results,
                "aggregate": aggregate_seed_results(seed_results),
            }

        folds_report[name] = {
            "train_end": fold["train_end"],
            "dev_end": fold["dev_end"],
            "test_end": fold["test_end"],
            "train_calls": len(train),
            "dev_calls": len(dev),
            "test_calls": len(test),
            "text_preprocessing": text_info,
            "models": model_reports,
        }

    prediction_frame = pd.DataFrame(predictions)
    bootstrap = build_bootstrap_report(
        prediction_frame, args.bootstrap_iterations
    )
    decision = decision_from_bootstrap(bootstrap)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_architecture": "learned gate with positive monotonic log-quality prior",
        "device": str(device),
        "seeds": list(args.seeds),
        "fold_design": "three expanding forward-time folds with non-overlapping tests",
        "training_noise": "one randomly selected modality corrupted in 70% of training examples; quality reduced consistently",
        "confirmatory_scenarios": list(CONFIRMATORY_SCENARIOS),
        "human_or_expert_labels_used": False,
        "test_used_for_model_selection": False,
        "folds": folds_report,
        "company_block_bootstrap": bootstrap,
        "confirmatory_decision": decision,
        "status": "PASS",
    }

    output_dir = root / "reports/confirmatory"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "confirmatory_validation.json"
    prediction_path = root / "artifacts/predictions/confirmatory_predictions.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_frame.to_csv(prediction_path, index=False)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"Birleşik doğrulama raporu: {report_path}")
    print(f"Tahminler: {prediction_path}")


if __name__ == "__main__":
    main()