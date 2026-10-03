"""P02-AHENK birleşik genişletilmiş doğrulama boru hattı.

Tek çalıştırmada:
1) üç genişleyen ileri-zaman katı kurar,
2) fiyat uzmanı, sabit uzman füzyonu ve AHENK kapısını eğitir,
3) tam veri, %25/%50 kip kaybı ve kip-gürültü streslerini ölçer,
4) katlar arası tutarlılık ve otomatik devam kararını tek raporda toplar.

Varsayılan CPU ve iki seed, hızlı tarama içindir. Nihai tabloda üçüncü seed
ayrıca çalıştırılabilir.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ahenk_expert_pilot import MODEL_SPECS, predict, train_model
from run_ahenk_pilot import (
    EPSILON,
    TARGET,
    fit_dense_text,
    fit_numeric,
    metric_set,
    quality_features,
    random_mask,
)


SELECTED_MODELS = (
    "price_expert",
    "fixed_expert_dropout_10",
    "ahenk_expert_gate_25",
)
FOLD_FRACTIONS = (
    (0.50, 0.60, 0.70),
    (0.60, 0.70, 0.80),
    (0.70, 0.80, 1.00),
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("En az bir seed gereklidir.")
    return seeds


def date_at_fraction(unique_dates: list[str], fraction: float) -> str:
    index = min(len(unique_dates) - 1, max(0, int(round(fraction * (len(unique_dates) - 1)))))
    return unique_dates[index]


def make_folds(frame: pd.DataFrame) -> list[dict]:
    unique_dates = sorted(frame["call_date"].astype(str).unique())
    folds = []
    for number, (train_fraction, dev_fraction, test_fraction) in enumerate(
        FOLD_FRACTIONS, start=1
    ):
        train_end = date_at_fraction(unique_dates, train_fraction)
        dev_end = date_at_fraction(unique_dates, dev_fraction)
        test_end = date_at_fraction(unique_dates, test_fraction)
        train = frame[frame["call_date"] <= train_end].copy()
        dev = frame[
            (frame["call_date"] > train_end) & (frame["call_date"] <= dev_end)
        ].copy()
        test = frame[
            (frame["call_date"] > dev_end) & (frame["call_date"] <= test_end)
        ].copy()
        if min(len(train), len(dev), len(test)) == 0:
            raise RuntimeError(f"Boş kat oluştu: fold_{number}")
        folds.append(
            {
                "name": f"fold_{number}",
                "train_end": train_end,
                "dev_end": dev_end,
                "test_end": test_end,
                "train": train.reset_index(drop=True),
                "dev": dev.reset_index(drop=True),
                "test": test.reset_index(drop=True),
            }
        )
    return folds


def copy_arrays(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(array.copy() for array in arrays)  # type: ignore[return-value]


def stress_scenarios(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    quality: np.ndarray,
    seed: int,
) -> dict[str, tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]]:
    count = len(quality)
    full_mask = np.ones((count, 3), dtype="float32")
    scenarios = {
        "full": (arrays, quality.copy(), full_mask),
        "random_25_missing": (
            arrays,
            quality.copy(),
            random_mask(count, 0.25, np.random.default_rng(seed + 25)),
        ),
        "random_50_missing": (
            arrays,
            quality.copy(),
            random_mask(count, 0.50, np.random.default_rng(seed + 50)),
        ),
    }

    corruption_specs = (
        ("text_noise_05", 0, 0.50, slice(0, 2), 0.50),
        ("acoustic_noise_05", 1, 0.50, slice(2, 4), 0.50),
        ("acoustic_noise_10", 1, 1.00, slice(2, 4), 0.20),
        ("price_noise_05", 2, 0.50, slice(4, 6), 0.50),
    )
    for offset, (name, modality, sigma, quality_slice, quality_factor) in enumerate(
        corruption_specs, start=1
    ):
        rng = np.random.default_rng(seed + 100 * offset)
        corrupted = list(copy_arrays(arrays))
        corrupted[modality] += rng.normal(
            0.0, sigma, size=corrupted[modality].shape
        ).astype("float32")
        corrupted_quality = quality.copy()
        corrupted_quality[:, quality_slice] *= quality_factor
        scenarios[name] = (
            tuple(corrupted),
            corrupted_quality,
            full_mask.copy(),
        )
    return scenarios


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    scenarios = seed_results[0]["scenarios"].keys()
    result = {}
    for scenario in scenarios:
        metrics = seed_results[0]["scenarios"][scenario].keys()
        result[scenario] = {
            metric: {
                "mean": float(
                    np.mean([item["scenarios"][scenario][metric] for item in seed_results])
                ),
                "std": float(
                    np.std([item["scenarios"][scenario][metric] for item in seed_results])
                ),
            }
            for metric in metrics
        }
    result["robust_qlike_mean"] = float(
        np.mean(
            [
                result[scenario]["qlike"]["mean"]
                for scenario in (
                    "full",
                    "random_25_missing",
                    "random_50_missing",
                )
            ]
        )
    )
    return result


def cross_fold_summary(folds: dict) -> dict:
    scenarios = next(iter(next(iter(folds.values()))["models"].values()))[
        "aggregate"
    ]
    scenario_names = [key for key in scenarios if key != "robust_qlike_mean"]
    summary = {}
    for model in SELECTED_MODELS:
        summary[model] = {}
        for scenario in scenario_names:
            values = [
                fold["models"][model]["aggregate"][scenario]["qlike"]["mean"]
                for fold in folds.values()
            ]
            summary[model][scenario] = {
                "fold_mean": float(np.mean(values)),
                "fold_std": float(np.std(values)),
                "fold_values": values,
            }
        robust_values = [
            fold["models"][model]["aggregate"]["robust_qlike_mean"]
            for fold in folds.values()
        ]
        summary[model]["robust_qlike_mean"] = {
            "fold_mean": float(np.mean(robust_values)),
            "fold_std": float(np.std(robust_values)),
            "fold_values": robust_values,
        }

    price_robust = summary["price_expert"]["robust_qlike_mean"]["fold_mean"]
    fixed_robust = summary["fixed_expert_dropout_10"]["robust_qlike_mean"]["fold_mean"]
    ahenk_robust = summary["ahenk_expert_gate_25"]["robust_qlike_mean"]["fold_mean"]
    ahenk_random50 = summary["ahenk_expert_gate_25"]["random_50_missing"]["fold_mean"]
    fixed_random50 = summary["fixed_expert_dropout_10"]["random_50_missing"]["fold_mean"]

    robust_vs_price = 100.0 * (price_robust - ahenk_robust) / max(price_robust, EPSILON)
    robust_vs_fixed = 100.0 * (fixed_robust - ahenk_robust) / max(fixed_robust, EPSILON)
    random50_vs_fixed = 100.0 * (fixed_random50 - ahenk_random50) / max(
        fixed_random50, EPSILON
    )
    consistent_robust_vs_fixed = sum(
        fold["models"]["ahenk_expert_gate_25"]["aggregate"]["robust_qlike_mean"]
        < fold["models"]["fixed_expert_dropout_10"]["aggregate"]["robust_qlike_mean"]
        for fold in folds.values()
    )

    criteria = {
        "robust_improvement_vs_price_percent": robust_vs_price,
        "robust_improvement_vs_fixed_percent": robust_vs_fixed,
        "random50_improvement_vs_fixed_percent": random50_vs_fixed,
        "folds_where_ahenk_robust_beats_fixed": consistent_robust_vs_fixed,
        "total_folds": len(folds),
    }
    if robust_vs_price >= 3.0 and random50_vs_fixed > 0 and consistent_robust_vs_fixed >= 2:
        decision = "PASS_TO_CONFIRMATORY_STAGE"
    elif robust_vs_price > 0 or robust_vs_fixed > 0:
        decision = "PARTIAL_CONTINUE_METHOD_REFINEMENT"
    else:
        decision = "STOP_OR_REDESIGN"
    return {"models": summary, "criteria": criteria, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=parse_seeds, default=(3407, 2026))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
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

    fold_reports = {}
    prediction_records = []
    for fold in make_folds(frame):
        name = fold["name"]
        train, dev, test = fold["train"], fold["dev"], fold["test"]
        print(
            f"{name}: train={len(train)}, dev={len(dev)}, test={len(test)} "
            f"({fold['train_end']} / {fold['dev_end']} / {fold['test_end']})"
        )
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
        for model_name in SELECTED_MODELS:
            specification = MODEL_SPECS[model_name]
            print(f"  {model_name}")
            seed_results = []
            for seed in args.seeds:
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
                scenario_results = {}
                for scenario, (
                    scenario_arrays,
                    scenario_quality,
                    scenario_mask,
                ) in stress_scenarios(test_arrays, test_quality, seed).items():
                    prediction, weights, _experts = predict(
                        model,
                        scenario_arrays,
                        scenario_quality,
                        scenario_mask,
                        device,
                        lower,
                        upper,
                    )
                    scenario_results[scenario] = metric_set(test_target, prediction)
                    for call_id, actual, predicted in zip(
                        test["call_id"], test_target, prediction, strict=True
                    ):
                        prediction_records.append(
                            {
                                "fold": name,
                                "model": model_name,
                                "seed": seed,
                                "scenario": scenario,
                                "call_id": call_id,
                                "actual": actual,
                                "predicted": predicted,
                            }
                        )
                seed_results.append(
                    {"seed": seed, "training": training, "scenarios": scenario_results}
                )
            model_reports[model_name] = {
                "mode": specification["mode"],
                "training_modality_dropout": specification["dropout"],
                "seed_results": seed_results,
                "aggregate": aggregate_seed_metrics(seed_results),
            }

        fold_reports[name] = {
            "train_end": fold["train_end"],
            "dev_end": fold["dev_end"],
            "test_end": fold["test_end"],
            "train_calls": len(train),
            "dev_calls": len(dev),
            "test_calls": len(test),
            "text_preprocessing": text_info,
            "models": model_reports,
        }

    summary = cross_fold_summary(fold_reports)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "seeds": list(args.seeds),
        "fold_design": "expanding train; non-overlapping forward test windows",
        "synthetic_missingness_and_noise": True,
        "test_used_for_model_selection": False,
        "human_or_expert_labels_used": False,
        "folds": fold_reports,
        "cross_fold_summary": summary,
        "status": "PASS",
    }

    output_dir = root / "reports/extended"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "extended_validation.json"
    prediction_path = root / "artifacts/predictions/extended_validation_predictions.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(prediction_records).to_csv(prediction_path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Birleşik rapor: {report_path}")
    print(f"Tahminler: {prediction_path}")


if __name__ == "__main__":
    main()