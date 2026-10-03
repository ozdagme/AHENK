"""AHENK katı eksik-modalite geliştirme deneyi.

Bu betik yalnız zamansal train/dev verisini kullanır. Mevcut dış doğrulama
kümesini okumaz. Bir modalite maskelendiğinde aşağıdaki alanlar kapıya
girmeden birlikte sıfırlanır:

- modaliteye ait uzman ortalaması,
- modaliteye ait uzman log-varyansı,
- modaliteye ait iki kalite göstergesi.

Böylece maskelenmiş modalitenin kalan modalite ağırlıklarını dolaylı
olarak etkilemesi engellenir. Sonuçlar geliştirme kanıtıdır; yeni ve
önceden ayrılmış bir dış doğrulamanın yerine geçmez.
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

import run_ahenk_v3_development as base
from run_ahenk_pilot import (
    TARGET,
    fit_dense_text,
    fit_numeric,
    metric_set,
    quality_features,
    random_mask,
)
from run_extended_validation import stress_scenarios


PRIMARY_MODELS = ("fixed_mean", "oof_risk_gate")


class StrictMissingRiskGate(base.RiskGate):
    """Kapıdan maskeli modaliteye ait bütün türetilmiş bilgiyi kaldırır."""

    @staticmethod
    def sanitize_inputs(
        quality: torch.Tensor,
        means: torch.Tensor,
        log_variances: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quality_mask = mask.repeat_interleave(2, dim=1)
        return (
            quality * quality_mask,
            means * mask,
            log_variances * mask,
        )

    def risks(
        self,
        quality: torch.Tensor,
        means: torch.Tensor,
        log_variances: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        quality, means, log_variances = self.sanitize_inputs(
            quality, means, log_variances, mask
        )
        return self.network(torch.cat([quality, means, log_variances, mask], dim=1))


def train_strict_gate(
    quality: np.ndarray,
    means: np.ndarray,
    log_variances: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 150,
    patience: int = 15,
) -> StrictMissingRiskGate:
    """OOF risk kapısını gerçek eksiklik maskeleriyle eğitir."""

    torch.manual_seed(seed)
    split = max(100, int(len(target) * 0.80))
    train_indices = np.arange(0, split)
    validation_indices = np.arange(split, len(target))
    loader = DataLoader(
        base.gate_dataset(
            quality[train_indices],
            means[train_indices],
            log_variances[train_indices],
            target[train_indices],
        ),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = StrictMissingRiskGate().to(device)
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

    for epoch in range(epochs):
        model.train()
        for batch_quality, batch_means, batch_log_variances, log_target in loader:
            batch_quality = batch_quality.to(device)
            batch_means = batch_means.to(device)
            batch_log_variances = batch_log_variances.to(device)
            log_target = log_target.to(device)
            full_mask = torch.ones((len(log_target), 3), device=device)
            subset_mask = torch.from_numpy(
                random_mask(len(log_target), 0.35, np.random.default_rng(seed + epoch))
            ).to(device)

            optimizer.zero_grad(set_to_none=True)
            full_prediction, _weights, predicted_risks = model(
                batch_quality, batch_means, batch_log_variances, full_mask
            )
            subset_prediction, _subset_weights, _subset_risks = model(
                batch_quality, batch_means, batch_log_variances, subset_mask
            )
            expert_risks = base.torch_qlike_per(
                log_target.unsqueeze(1), batch_means
            ).detach()
            risk_loss = torch.mean(
                (torch.log1p(predicted_risks) - torch.log1p(expert_risks)) ** 2
            )
            full_losses = base.torch_qlike_per(log_target, full_prediction)
            subset_losses = base.torch_qlike_per(log_target, subset_prediction)
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
            score = float(base.torch_qlike_per(log_target_v, prediction).mean().cpu())
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
def masked_information_invariance(
    gate: StrictMissingRiskGate,
    quality: np.ndarray,
    means: np.ndarray,
    log_variances: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> dict[str, float | bool]:
    """Maskeli alanlar değiştirildiğinde çıktının sabit kaldığını sınar."""

    gate.eval()
    tensors = (
        torch.from_numpy(quality.astype("float32")).to(device),
        torch.from_numpy(means.astype("float32")).to(device),
        torch.from_numpy(log_variances.astype("float32")).to(device),
        torch.from_numpy(mask.astype("float32")).to(device),
    )
    baseline_prediction, baseline_weights, _ = gate(*tensors)

    altered_quality = tensors[0].clone()
    altered_means = tensors[1].clone()
    altered_log_variances = tensors[2].clone()
    quality_mask = tensors[3].repeat_interleave(2, dim=1)
    altered_quality = altered_quality + (1.0 - quality_mask) * 1000.0
    altered_means = altered_means - (1.0 - tensors[3]) * 1000.0
    altered_log_variances = altered_log_variances + (1.0 - tensors[3]) * 1000.0
    altered_prediction, altered_weights, _ = gate(
        altered_quality, altered_means, altered_log_variances, tensors[3]
    )

    prediction_difference = float(
        torch.max(torch.abs(baseline_prediction - altered_prediction)).cpu()
    )
    weight_difference = float(
        torch.max(torch.abs(baseline_weights - altered_weights)).cpu()
    )
    tolerance = 1e-7
    return {
        "max_prediction_difference": prediction_difference,
        "max_weight_difference": weight_difference,
        "tolerance": tolerance,
        "passed": prediction_difference <= tolerance and weight_difference <= tolerance,
    }


def self_test() -> dict[str, float | bool]:
    torch.manual_seed(29)
    rng = np.random.default_rng(29)
    gate = StrictMissingRiskGate()
    count = 32
    quality = rng.normal(size=(count, 6)).astype("float32")
    means = rng.normal(size=(count, 3)).astype("float32")
    log_variances = rng.normal(size=(count, 3)).astype("float32")
    mask = random_mask(count, 0.50, rng)
    return masked_information_invariance(
        gate, quality, means, log_variances, mask, torch.device("cpu")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=base.parse_seeds, default=(3407, 2026))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit("Katı maskeleme invariance testi başarısız.")
        return

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

    model_seed_results = {model: [] for model in base.MODEL_NAMES}
    invariance_results = []
    for seed in args.seeds:
        print(f"Başlangıç {seed}: zamansal örnek-dışı uzman tahminleri")
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
        evaluation_stresses = stress_scenarios(
            evaluation_arrays, quality_evaluation, seed
        )

        strict_arrays, strict_quality, strict_mask = evaluation_stresses[
            "random_50_missing"
        ]
        strict_means, strict_log_variances = base.predict_experts(
            experts, strict_arrays, device
        )
        invariance = masked_information_invariance(
            gate,
            strict_quality,
            strict_means,
            strict_log_variances,
            strict_mask,
            device,
        )
        invariance["seed"] = seed
        invariance_results.append(invariance)
        if not invariance["passed"]:
            raise RuntimeError("Maskeli modalite bilgisi kapı çıktısını etkiliyor.")

        for model_name in base.MODEL_NAMES:
            calibration_mask = np.ones((len(calibration), 3), dtype="float32")
            calibration_log_prediction, _ = base.fuse_predictions(
                model_name,
                gate,
                quality_calibration,
                calibration_means,
                calibration_log_variances,
                calibration_mask,
                device,
            )
            radius = base.conformal_quantile(
                calibration_target, calibration_log_prediction, level=0.90
            )
            scenarios = {}
            for scenario in base.EVAL_SCENARIOS:
                scenario_arrays, scenario_quality, scenario_mask = evaluation_stresses[
                    scenario
                ]
                means, log_variances = base.predict_experts(
                    experts, scenario_arrays, device
                )
                log_prediction, weights = base.fuse_predictions(
                    model_name,
                    gate,
                    scenario_quality,
                    means,
                    log_variances,
                    scenario_mask,
                    device,
                )

                evaluation_rows = np.ones(len(evaluation_target), dtype=bool)
                comparable_to_multimodal = True
                if model_name == "price_expert" and scenario.startswith("random_"):
                    evaluation_rows = scenario_mask[:, 2].astype(bool)
                    comparable_to_multimodal = False
                prediction = np.exp(log_prediction[evaluation_rows])
                actual = evaluation_target[evaluation_rows]
                scenarios[scenario] = {
                    "point": metric_set(actual, prediction),
                    "interval": base.interval_metrics(
                        actual, log_prediction[evaluation_rows], radius
                    ),
                    "evaluation_calls": int(evaluation_rows.sum()),
                    "same_rows_as_multimodal_methods": comparable_to_multimodal,
                    "mean_weights": {
                        modality: float(weights[evaluation_rows, index].mean())
                        for index, modality in enumerate(base.MODALITIES)
                    },
                }
            model_seed_results[model_name].append(
                {"seed": seed, "conformal_radius": radius, "scenarios": scenarios}
            )

    models = {
        model: {
            "seed_results": results,
            "aggregate": base.aggregate(results),
        }
        for model, results in model_seed_results.items()
    }
    best_primary = min(
        PRIMARY_MODELS,
        key=lambda model: models[model]["aggregate"]["robust_qlike_mean"],
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development_only_strict_missing_modality",
        "device": str(device),
        "seeds": list(args.seeds),
        "train_calls": len(train),
        "calibration_calls": len(calibration),
        "development_evaluation_calls": len(evaluation),
        "text_preprocessing": text_info,
        "strict_masking": {
            "masked_quality_zeroed": True,
            "masked_expert_mean_zeroed": True,
            "masked_expert_log_variance_zeroed": True,
            "mask_still_supplied_to_gate": True,
            "invariance_checks": invariance_results,
        },
        "external_confirmation_rows_used": False,
        "human_or_expert_labels_used": False,
        "price_only_missingness_metrics_use_price_available_subset": True,
        "models": models,
        "best_primary_model_by_development_robust_qlike": best_primary,
        "next_required_evidence": (
            "new prespecified company-disjoint external confirmation set"
        ),
        "status": "PASS",
    }
    output = root / "reports/development/ahenk_true_missing_development.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_primary_model": best_primary,
                "invariance_checks": invariance_results,
                "report": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()