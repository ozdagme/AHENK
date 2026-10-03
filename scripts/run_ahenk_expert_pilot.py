"""AHENK v2: kalite kapılı, derin-denetimli uzmanlar karışımı pilotu.

Her kip ayrı log-varyans tahmini üretir. Kapı, kalite ve eksiklik maskesine göre
uzman tahminlerini birleştirir. Tam-veri QLIKE, eksik-kip QLIKE ve tekil uzman
kayıpları birlikte optimize edilir. Varsayılan CPU'dur.
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
from torch import nn
from torch.utils.data import DataLoader

from run_ahenk_pilot import (
    EPSILON,
    MODALITIES,
    TARGET,
    fit_dense_text,
    fit_numeric,
    metric_set,
    quality_features,
    random_mask,
    scenario_masks,
    tensors,
)


MODEL_SPECS = {
    "price_expert": {"mode": "price", "dropout": 0.0},
    "fixed_expert_dropout_10": {"mode": "fixed", "dropout": 0.10},
    "ahenk_expert_gate_10": {"mode": "gate", "dropout": 0.10},
    "ahenk_expert_gate_25": {"mode": "gate", "dropout": 0.25},
}


def torch_qlike(log_target: torch.Tensor, log_prediction: torch.Tensor) -> torch.Tensor:
    difference = torch.clamp(log_target - log_prediction, -20.0, 20.0)
    return torch.mean(torch.exp(difference) - difference - 1.0)


class ExpertFusion(nn.Module):
    def __init__(self, dimensions: tuple[int, int, int], mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, 64),
                    nn.GELU(),
                    nn.LayerNorm(64),
                    nn.Linear(64, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                )
                for dimension in dimensions
            ]
        )
        self.gate = (
            nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, 3))
            if mode == "gate"
            else None
        )
        self.raw_quality_strength = (
            nn.Parameter(torch.full((3,), 0.5)) if mode == "gate" else None
        )

    def forward(
        self,
        text: torch.Tensor,
        acoustic: torch.Tensor,
        price: torch.Tensor,
        quality: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_logs = torch.cat(
            [expert(value) for expert, value in zip(self.experts, (text, acoustic, price), strict=True)],
            dim=1,
        )
        if self.mode == "price":
            weights = torch.zeros_like(mask)
            weights[:, 2] = mask[:, 2]
        elif self.mode == "fixed":
            weights = mask / mask.sum(1, keepdim=True).clamp_min(1.0)
        elif self.mode == "gate":
            masked_quality = quality * mask.repeat_interleave(2, dim=1)
            logits = self.gate(torch.cat([masked_quality, mask], dim=1))
            modality_quality = quality.reshape(-1, 3, 2).mean(dim=2).clamp_min(1e-3)
            quality_strength = torch.nn.functional.softplus(
                self.raw_quality_strength
            ).unsqueeze(0)
            # Monoton kalite önceliği: kalite düştükçe ilgili kip logiti
            # öğrenilen pozitif katsayıyla zorunlu olarak azalır.
            logits = logits + quality_strength * torch.log(modality_quality)
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=1)
        else:
            raise ValueError(self.mode)
        fused_log = (weights * expert_logs).sum(1)
        return fused_log, expert_logs, weights


@torch.no_grad()
def predict(
    model: ExpertFusion,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    quality: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    log_prediction, expert_logs, weights = model(
        torch.from_numpy(arrays[0]).to(device),
        torch.from_numpy(arrays[1]).to(device),
        torch.from_numpy(arrays[2]).to(device),
        torch.from_numpy(quality).to(device),
        torch.from_numpy(mask).to(device),
    )
    prediction = np.clip(np.exp(log_prediction.cpu().numpy()), lower, upper)
    expert_predictions = np.clip(np.exp(expert_logs.cpu().numpy()), lower, upper)
    return prediction, weights.cpu().numpy(), expert_predictions


def train_model(
    dimensions: tuple[int, int, int],
    mode: str,
    dropout_probability: float,
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
    model = ExpertFusion(dimensions, mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(
        tensors(train_arrays, train_quality, train_target),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    lower = float(max(np.quantile(train_target, 0.001), EPSILON))
    upper = float(max(np.quantile(train_target, 0.999), EPSILON * 10))
    dev_masks = {
        "full": np.ones((len(dev_target), 3), dtype="float32"),
        "random_25": random_mask(len(dev_target), 0.25, np.random.default_rng(seed + 25)),
        "random_50": random_mask(len(dev_target), 0.50, np.random.default_rng(seed + 50)),
    }

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for text, acoustic, price, quality, base_mask, log_target in loader:
            text = text.to(device)
            acoustic = acoustic.to(device)
            price = price.to(device)
            quality = quality.to(device)
            log_target = log_target.to(device)
            full_mask = base_mask.to(device)
            dropout_mask = torch.from_numpy(
                random_mask(len(log_target), dropout_probability, rng)
                if dropout_probability > 0
                else base_mask.numpy()
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            full_log, expert_logs, _weights = model(
                text, acoustic, price, quality, full_mask
            )
            masked_log, _masked_experts, _masked_weights = model(
                text, acoustic, price, quality, dropout_mask
            )
            full_loss = torch_qlike(log_target, full_log)
            missing_loss = torch_qlike(log_target, masked_log)
            expert_loss = torch.mean(
                torch.stack(
                    [torch_qlike(log_target, expert_logs[:, index]) for index in range(3)]
                )
            )
            loss = full_loss + 0.50 * missing_loss + 0.15 * expert_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = []
        for mask in dev_masks.values():
            prediction, _weights, _experts = predict(
                model, dev_arrays, dev_quality, mask, device, lower, upper
            )
            actual = np.maximum(dev_target, EPSILON)
            ratio = actual / np.maximum(prediction, EPSILON)
            scores.append(float(np.mean(ratio - np.log(ratio) - 1.0)))
        robust_score = float(np.mean(scores))
        if robust_score < best_score - 1e-5:
            best_score = robust_score
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
        "best_dev_robust_qlike": best_score,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def aggregate(seed_results: list[dict]) -> dict:
    scenario_names = seed_results[0]["scenarios"].keys()
    scenarios = {}
    weights = {}
    for scenario in scenario_names:
        metric_names = seed_results[0]["scenarios"][scenario].keys()
        scenarios[scenario] = {
            metric: {
                "mean": float(np.mean([item["scenarios"][scenario][metric] for item in seed_results])),
                "std": float(np.std([item["scenarios"][scenario][metric] for item in seed_results])),
            }
            for metric in metric_names
        }
        weights[scenario] = {
            modality: {
                "mean": float(np.mean([item["weights"][scenario][index] for item in seed_results])),
                "std": float(np.std([item["weights"][scenario][index] for item in seed_results])),
            }
            for index, modality in enumerate(MODALITIES)
        }
    full_qlike = scenarios["full"]["qlike"]["mean"]
    degradation = {
        scenario: float(
            100.0 * (values["qlike"]["mean"] - full_qlike) / max(full_qlike, EPSILON)
        )
        for scenario, values in scenarios.items()
    }
    robust_qlike_mean = float(
        np.mean(
            [
                scenarios[scenario]["qlike"]["mean"]
                for scenario in ("full", "random_25", "random_50")
            ]
        )
    )
    expert_full = {
        modality: {
            metric: {
                "mean": float(
                    np.mean(
                        [
                            item["expert_full_metrics"][modality][metric]
                            for item in seed_results
                        ]
                    )
                ),
                "std": float(
                    np.std(
                        [
                            item["expert_full_metrics"][modality][metric]
                            for item in seed_results
                        ]
                    )
                ),
            }
            for metric in seed_results[0]["expert_full_metrics"][modality]
        }
        for modality in MODALITIES
    }
    return {
        "scenarios": scenarios,
        "mean_gate_weights": weights,
        "qlike_relative_degradation_percent": degradation,
        "robust_qlike_mean_full_random25_random50": robust_qlike_mean,
        "individual_expert_full_metrics": expert_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
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
        column for column in frame
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    price_columns = [column for column in frame if column.startswith("price_")]

    regimes = {}
    prediction_records = []
    for split_column in ("temporal_split", "company_holdout_split"):
        print(f"Hazırlanıyor: {split_column}")
        train = frame[frame[split_column] == "train"].reset_index(drop=True)
        dev = frame[frame[split_column] == "dev"].reset_index(drop=True)
        test = frame[frame[split_column] == "test"].reset_index(drop=True)
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

        model_results = {}
        for model_name, specification in MODEL_SPECS.items():
            print(f"  Model: {model_name}")
            seed_results = []
            for seed in (3407, 2026, 2709):
                model, training = train_model(
                    dimensions,
                    specification["mode"],
                    specification["dropout"],
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
                scenarios = {}
                weights = {}
                expert_full_metrics = None
                for scenario, mask in scenario_masks(len(test), seed).items():
                    prediction, gate_weights, expert_predictions = predict(
                        model,
                        test_arrays,
                        test_quality,
                        mask,
                        device,
                        lower,
                        upper,
                    )
                    scenarios[scenario] = metric_set(test_target, prediction)
                    weights[scenario] = gate_weights.mean(axis=0).tolist()
                    if scenario == "full":
                        expert_full_metrics = {
                            modality: metric_set(test_target, expert_predictions[:, index])
                            for index, modality in enumerate(MODALITIES)
                        }
                    for call_id, actual, predicted in zip(
                        test["call_id"], test_target, prediction, strict=True
                    ):
                        prediction_records.append(
                            {
                                "evaluation_regime": split_column,
                                "model": model_name,
                                "seed": seed,
                                "scenario": scenario,
                                "call_id": call_id,
                                "actual": actual,
                                "predicted": predicted,
                            }
                        )
                seed_results.append(
                    {
                        "seed": seed,
                        "training": training,
                        "scenarios": scenarios,
                        "weights": weights,
                        "expert_full_metrics": expert_full_metrics,
                    }
                )
            model_results[model_name] = {
                "mode": specification["mode"],
                "training_modality_dropout": specification["dropout"],
                "seed_results": seed_results,
                "aggregate": aggregate(seed_results),
            }

        regimes[split_column] = {
            "train_calls": len(train),
            "dev_calls": len(dev),
            "test_calls": len(test),
            "modality_dimensions": dict(zip(MODALITIES, dimensions, strict=True)),
            "text_preprocessing": text_info,
            "models": model_results,
        }

    prediction_output = root / "artifacts/predictions/ahenk_expert_predictions.csv"
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_records).to_csv(prediction_output, index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "deeply supervised prediction-space mixture of experts",
        "training_loss": "full QLIKE + 0.5 missing-mask QLIKE + 0.15 expert QLIKE",
        "device": str(device),
        "synthetic_missingness_only": True,
        "test_used_for_model_selection": False,
        "future_prices_used_as_inputs": False,
        "human_or_expert_labels_used": False,
        "regimes": regimes,
        "status": "PASS",
    }
    output = root / "reports/models/ahenk_expert_pilot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")
    print(f"Tahminler: {prediction_output}")


if __name__ == "__main__":
    main()