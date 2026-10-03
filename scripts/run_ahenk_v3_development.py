"""AHENK V3: insan etiketsiz belirsizlik ve out-of-fold risk kapısı.

Bu betik önceki test bölümüne dokunmaz. Yalnız temporal train/dev kullanır:
- her kip ortalama log-varyans ve tahmin belirsizliği üretir,
- sentetik gürültüde belirsizliğin artması otomatik olarak öğretilir,
- kapı out-of-fold uzman hatalarından risk tahmini öğrenir,
- More-vs-Fewer kaybı tam kip sonucunun alt kümelerden kötüleşmesini cezalandırır,
- dev'in ilk yarısıyla %90 conformal aralık kalibre edilir,
- dev'in ikinci yarısında tam/eksik/gürültülü senaryolar raporlanır.

Sonuç geliştirme kanıtıdır; yeni dış test yerine geçmez.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_ahenk_pilot import (
    EPSILON,
    TARGET,
    fit_dense_text,
    fit_numeric,
    metric_set,
    quality_features,
    random_mask,
)
from run_extended_validation import stress_scenarios


MODALITIES = ("text", "acoustic", "price")
MODEL_NAMES = ("price_expert", "fixed_mean", "precision_fusion", "oof_risk_gate")
EVAL_SCENARIOS = (
    "full",
    "random_25_missing",
    "random_50_missing",
    "text_noise_05",
    "acoustic_noise_10",
    "price_noise_05",
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("En az bir seed gereklidir.")
    return seeds


def torch_qlike_per(log_target: torch.Tensor, log_prediction: torch.Tensor) -> torch.Tensor:
    difference = torch.clamp(log_target - log_prediction, -20.0, 20.0)
    return torch.exp(difference) - difference - 1.0


class ExpertSet(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int]) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, 64),
                    nn.GELU(),
                    nn.LayerNorm(64),
                    nn.Linear(64, 32),
                    nn.GELU(),
                    nn.Linear(32, 2),
                )
                for dimension in dimensions
            ]
        )

    def forward(
        self, text: torch.Tensor, acoustic: torch.Tensor, price: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.stack(
            [expert(value) for expert, value in zip(self.experts, (text, acoustic, price), strict=True)],
            dim=1,
        )
        means = outputs[:, :, 0]
        log_variances = torch.clamp(outputs[:, :, 1], -6.0, 4.0)
        return means, log_variances


class RiskGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(15, 48),
            nn.GELU(),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 3),
            nn.Softplus(),
        )

    def risks(
        self,
        quality: torch.Tensor,
        means: torch.Tensor,
        log_variances: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([quality, means, log_variances, mask], dim=1))

    def forward(
        self,
        quality: torch.Tensor,
        means: torch.Tensor,
        log_variances: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        risks = self.risks(quality, means, log_variances, mask)
        logits = -risks
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        fused = (weights * means).sum(1)
        return fused, weights, risks


def array_dataset(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray], target: np.ndarray
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays[0]),
        torch.from_numpy(arrays[1]),
        torch.from_numpy(arrays[2]),
        torch.from_numpy(np.log(target).astype("float32")),
    )


def expert_loss(
    log_target: torch.Tensor,
    means: torch.Tensor,
    log_variances: torch.Tensor,
) -> torch.Tensor:
    error = log_target.unsqueeze(1) - means
    gaussian = 0.5 * torch.exp(-log_variances) * error.square() + 0.5 * log_variances
    qlike = torch_qlike_per(log_target.unsqueeze(1), means)
    return gaussian.mean() + 0.25 * qlike.mean()


def noisy_modality(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    strength = 0.25 + 0.75 * torch.rand(len(value), device=value.device)
    noisy = value + torch.randn_like(value) * strength.unsqueeze(1)
    return noisy, strength


@torch.no_grad()
def predict_experts(
    model: ExpertSet,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    means, log_variances = model(
        torch.from_numpy(arrays[0]).to(device),
        torch.from_numpy(arrays[1]).to(device),
        torch.from_numpy(arrays[2]).to(device),
    )
    return means.cpu().numpy(), log_variances.cpu().numpy()


def train_experts(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_target: np.ndarray,
    validation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_target: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 80,
    patience: int = 10,
) -> ExpertSet:
    torch.manual_seed(seed)
    dimensions = tuple(array.shape[1] for array in train_arrays)
    model = ExpertSet(dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    loader = DataLoader(
        array_dataset(train_arrays, train_target),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    stale = 0

    validation_tensors = tuple(torch.from_numpy(value).to(device) for value in validation_arrays)
    validation_log_target = torch.from_numpy(
        np.log(validation_target).astype("float32")
    ).to(device)

    for _epoch in range(epochs):
        model.train()
        for text, acoustic, price, log_target in loader:
            values = [text.to(device), acoustic.to(device), price.to(device)]
            log_target = log_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            means, log_variances = model(*values)
            loss = expert_loss(log_target, means, log_variances)

            uncertainty_ranking = 0.0
            noisy_supervision = 0.0
            for modality in range(3):
                noisy_values = [value.clone() for value in values]
                noisy_values[modality], strength = noisy_modality(noisy_values[modality])
                noisy_means, noisy_log_variances = model(*noisy_values)
                uncertainty_ranking = uncertainty_ranking + torch.relu(
                    0.20 * strength
                    - (noisy_log_variances[:, modality] - log_variances[:, modality].detach())
                ).mean()
                noisy_supervision = noisy_supervision + torch_qlike_per(
                    log_target, noisy_means[:, modality]
                ).mean()
            loss = loss + 0.10 * uncertainty_ranking + 0.05 * noisy_supervision
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            means, log_variances = model(*validation_tensors)
            score = float(expert_loss(validation_log_target, means, log_variances).cpu())
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model


def oof_ranges(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    ranges = []
    for train_fraction, validation_fraction in (
        (0.40, 0.55),
        (0.55, 0.70),
        (0.70, 0.85),
        (0.85, 1.00),
    ):
        train_end = max(50, int(count * train_fraction))
        validation_end = min(count, int(count * validation_fraction))
        ranges.append(
            (
                np.arange(0, train_end),
                np.arange(train_end, validation_end),
            )
        )
    return ranges


def build_oof_predictions(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    target: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(target)
    means = np.full((count, 3), np.nan, dtype="float32")
    log_variances = np.full((count, 3), np.nan, dtype="float32")
    valid = np.zeros(count, dtype=bool)
    for fold, (train_indices, validation_indices) in enumerate(oof_ranges(count), start=1):
        print(f"    OOF uzman katı {fold}/4")
        fold_train_arrays = tuple(value[train_indices] for value in arrays)
        fold_validation_arrays = tuple(value[validation_indices] for value in arrays)
        model = train_experts(
            fold_train_arrays,
            target[train_indices],
            fold_validation_arrays,
            target[validation_indices],
            device,
            seed + fold,
            epochs=60,
            patience=8,
        )
        fold_means, fold_log_variances = predict_experts(
            model, fold_validation_arrays, device
        )
        means[validation_indices] = fold_means
        log_variances[validation_indices] = fold_log_variances
        valid[validation_indices] = True
    return means, log_variances, valid


def gate_dataset(
    quality: np.ndarray,
    means: np.ndarray,
    log_variances: np.ndarray,
    target: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(quality.astype("float32")),
        torch.from_numpy(means.astype("float32")),
        torch.from_numpy(log_variances.astype("float32")),
        torch.from_numpy(np.log(target).astype("float32")),
    )


def train_gate(
    quality: np.ndarray,
    means: np.ndarray,
    log_variances: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 150,
    patience: int = 15,
) -> RiskGate:
    torch.manual_seed(seed)
    split = max(100, int(len(target) * 0.80))
    train_indices = np.arange(0, split)
    validation_indices = np.arange(split, len(target))
    loader = DataLoader(
        gate_dataset(
            quality[train_indices],
            means[train_indices],
            log_variances[train_indices],
            target[train_indices],
        ),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = RiskGate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    stale = 0

    validation_tensors = (
        torch.from_numpy(quality[validation_indices].astype("float32")).to(device),
        torch.from_numpy(means[validation_indices].astype("float32")).to(device),
        torch.from_numpy(log_variances[validation_indices].astype("float32")).to(device),
        torch.from_numpy(np.log(target[validation_indices]).astype("float32")).to(device),
    )

    for _epoch in range(epochs):
        model.train()
        for batch_quality, batch_means, batch_log_variances, log_target in loader:
            batch_quality = batch_quality.to(device)
            batch_means = batch_means.to(device)
            batch_log_variances = batch_log_variances.to(device)
            log_target = log_target.to(device)
            full_mask = torch.ones((len(log_target), 3), device=device)
            subset_mask = torch.from_numpy(
                random_mask(len(log_target), 0.35, np.random.default_rng(seed + _epoch))
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            full_prediction, _weights, predicted_risks = model(
                batch_quality, batch_means, batch_log_variances, full_mask
            )
            subset_prediction, _subset_weights, _subset_risks = model(
                batch_quality, batch_means, batch_log_variances, subset_mask
            )
            expert_risks = torch_qlike_per(
                log_target.unsqueeze(1), batch_means
            ).detach()
            risk_loss = torch.mean(
                (torch.log1p(predicted_risks) - torch.log1p(expert_risks)) ** 2
            )
            full_losses = torch_qlike_per(log_target, full_prediction)
            subset_losses = torch_qlike_per(log_target, subset_prediction)
            more_vs_fewer = torch.relu(full_losses - subset_losses).mean()
            loss = (
                risk_loss
                + full_losses.mean()
                + 0.35 * subset_losses.mean()
                + 0.25 * more_vs_fewer
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            quality_v, means_v, log_variances_v, log_target_v = validation_tensors
            full_mask = torch.ones((len(log_target_v), 3), device=device)
            prediction, _, _ = model(
                quality_v, means_v, log_variances_v, full_mask
            )
            score = float(torch_qlike_per(log_target_v, prediction).mean().cpu())
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def fuse_predictions(
    model_name: str,
    gate: RiskGate,
    quality: np.ndarray,
    means: np.ndarray,
    log_variances: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mask_t = torch.from_numpy(mask.astype("float32")).to(device)
    means_t = torch.from_numpy(means.astype("float32")).to(device)
    log_variances_t = torch.from_numpy(log_variances.astype("float32")).to(device)
    quality_t = torch.from_numpy(quality.astype("float32")).to(device)
    if model_name == "price_expert":
        weights = torch.zeros_like(mask_t)
        weights[:, 2] = mask_t[:, 2]
        log_prediction = means_t[:, 2] * mask_t[:, 2]
    elif model_name == "fixed_mean":
        weights = mask_t / mask_t.sum(1, keepdim=True).clamp_min(1.0)
        log_prediction = (weights * means_t).sum(1)
    elif model_name == "precision_fusion":
        logits = -log_variances_t
        logits = logits.masked_fill(mask_t == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        log_prediction = (weights * means_t).sum(1)
    elif model_name == "oof_risk_gate":
        log_prediction, weights, _ = gate(
            quality_t, means_t, log_variances_t, mask_t
        )
    else:
        raise ValueError(model_name)
    return log_prediction.cpu().numpy(), weights.cpu().numpy()


def conformal_quantile(
    actual: np.ndarray, log_prediction: np.ndarray, level: float = 0.90
) -> float:
    scores = np.abs(np.log(actual) - log_prediction)
    rank = min(len(scores), math.ceil((len(scores) + 1) * level))
    return float(np.partition(scores, rank - 1)[rank - 1])


def interval_metrics(
    actual: np.ndarray, log_prediction: np.ndarray, radius: float
) -> dict[str, float]:
    lower = np.exp(log_prediction - radius)
    upper = np.exp(log_prediction + radius)
    return {
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }


def aggregate(seed_results: list[dict]) -> dict:
    output = {}
    for scenario in EVAL_SCENARIOS:
        output[scenario] = {}
        for metric in seed_results[0]["scenarios"][scenario]["point"]:
            values = [item["scenarios"][scenario]["point"][metric] for item in seed_results]
            output[scenario][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        for metric in ("coverage", "mean_width"):
            values = [
                item["scenarios"][scenario]["interval"][metric]
                for item in seed_results
            ]
            output[scenario][f"conformal_{metric}"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    output["robust_qlike_mean"] = float(
        np.mean([output[scenario]["qlike"]["mean"] for scenario in EVAL_SCENARIOS])
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=parse_seeds, default=(3407, 2026))
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
    train = frame[frame["temporal_split"] == "train"].reset_index(drop=True)
    dev = frame[frame["temporal_split"] == "dev"].reset_index(drop=True)
    split = len(dev) // 2
    calibration = dev.iloc[:split].reset_index(drop=True)
    evaluation = dev.iloc[split:].reset_index(drop=True)
    acoustic_columns = [
        column
        for column in frame
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    price_columns = [column for column in frame if column.startswith("price_")]

    text_train, text_calibration, text_evaluation, text_info = fit_dense_text(
        root, train, calibration, evaluation, dimensions=128
    )
    acoustic_train, acoustic_calibration, acoustic_evaluation = fit_numeric(
        train, calibration, evaluation, acoustic_columns
    )
    price_train, price_calibration, price_evaluation = fit_numeric(
        train, calibration, evaluation, price_columns
    )
    quality_train, quality_calibration, quality_evaluation = quality_features(
        train, (train, calibration, evaluation)
    )
    train_arrays = (text_train, acoustic_train, price_train)
    calibration_arrays = (text_calibration, acoustic_calibration, price_calibration)
    evaluation_arrays = (text_evaluation, acoustic_evaluation, price_evaluation)
    train_target = train[TARGET].to_numpy(float)
    calibration_target = calibration[TARGET].to_numpy(float)
    evaluation_target = evaluation[TARGET].to_numpy(float)

    model_seed_results = {model: [] for model in MODEL_NAMES}
    uncertainty_diagnostics = []
    for seed in args.seeds:
        print(f"Seed {seed}: OOF uzman tahminleri")
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
        final_experts = train_experts(
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
            final_experts, calibration_arrays, device
        )

        evaluation_stresses = stress_scenarios(
            evaluation_arrays, quality_evaluation, seed
        )
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
                scenario_arrays, scenario_quality, scenario_mask = evaluation_stresses[
                    scenario
                ]
                means, log_variances = predict_experts(
                    final_experts, scenario_arrays, device
                )
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
                scenarios[scenario] = {
                    "point": metric_set(evaluation_target, prediction),
                    "interval": interval_metrics(
                        evaluation_target, log_prediction, radius
                    ),
                    "mean_weights": {
                        modality: float(weights[:, index].mean())
                        for index, modality in enumerate(MODALITIES)
                    },
                }
            model_seed_results[model_name].append(
                {"seed": seed, "conformal_radius": radius, "scenarios": scenarios}
            )

        full_means, full_log_variances = predict_experts(
            final_experts, evaluation_arrays, device
        )
        noisy_arrays, noisy_quality, _mask = evaluation_stresses["acoustic_noise_10"]
        _noisy_means, noisy_log_variances = predict_experts(
            final_experts, noisy_arrays, device
        )
        uncertainty_diagnostics.append(
            {
                "seed": seed,
                "clean_acoustic_log_variance_mean": float(
                    full_log_variances[:, 1].mean()
                ),
                "noisy_acoustic_log_variance_mean": float(
                    noisy_log_variances[:, 1].mean()
                ),
                "uncertainty_increased": bool(
                    noisy_log_variances[:, 1].mean()
                    > full_log_variances[:, 1].mean()
                ),
                "mean_noisy_acoustic_quality": float(
                    noisy_quality[:, 2:4].mean()
                ),
            }
        )

    models_report = {
        model: {
            "seed_results": results,
            "aggregate": aggregate(results),
        }
        for model, results in model_seed_results.items()
    }
    best_model = min(
        MODEL_NAMES,
        key=lambda model: models_report[model]["aggregate"]["robust_qlike_mean"],
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development_only_no_previous_test_access",
        "device": str(device),
        "seeds": list(args.seeds),
        "train_calls": len(train),
        "calibration_calls": len(calibration),
        "development_evaluation_calls": len(evaluation),
        "text_preprocessing": text_info,
        "architecture": "heteroscedastic modality experts + out-of-fold risk gate",
        "automatic_supervision": [
            "market-derived variance target",
            "synthetic corruption severity",
            "out-of-fold expert QLIKE risk",
            "More-vs-Fewer ranking",
        ],
        "human_or_expert_labels_used": False,
        "previous_test_rows_used": False,
        "uncertainty_diagnostics": uncertainty_diagnostics,
        "models": models_report,
        "best_development_model_by_robust_qlike": best_model,
        "next_required_evidence": "new locked external or newly acquired confirmation set",
        "status": "PASS",
    }
    output = root / "reports/v3/ahenk_v3_development.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "best_model": best_model,
        "robust_qlike": {
            model: models_report[model]["aggregate"]["robust_qlike_mean"]
            for model in MODEL_NAMES
        },
        "uncertainty_diagnostics": uncertainty_diagnostics,
    }, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")


if __name__ == "__main__":
    main()