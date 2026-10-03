"""P02-AHENK için sızıntısız CPU taban çizgilerini çalıştırır.

Zaman ve görülmemiş şirket ayrımlarını ayrı değerlendirir. Eksik değer doldurma,
ölçekleme ve hiperparametre seçimi yalnız train/dev üzerinde yapılır. Test,
seçim tamamlandıktan sonra bir kez ölçülür. GPU ve insan etiketi kullanmaz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


EPSILON = 1e-12
TARGET = "target_realized_variance_5d"
METADATA = {
    "call_id",
    "company_id",
    "ticker",
    "call_date",
    "temporal_split",
    "company_holdout_split",
    TARGET,
}


def positive(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), max(lower, EPSILON), upper)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.maximum(np.asarray(y_true, dtype=float), EPSILON)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), EPSILON)
    ratio = y_true / y_pred
    return {
        "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "log_rmse": float(
            np.sqrt(np.mean(np.square(np.log(y_true) - np.log(y_pred))))
        ),
    }


def bounds(y: np.ndarray) -> tuple[float, float]:
    return (
        float(max(np.quantile(y, 0.001), EPSILON)),
        float(max(np.quantile(y, 0.999), EPSILON * 10)),
    )


def ridge_model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def tune_ridge(
    frame: pd.DataFrame,
    columns: list[str],
    split_column: str,
) -> tuple[Pipeline, float, dict[str, float]]:
    train = frame[frame[split_column] == "train"]
    dev = frame[frame[split_column] == "dev"]
    train_y = train[TARGET].to_numpy(float)
    dev_y = dev[TARGET].to_numpy(float)
    lower, upper = bounds(train_y)

    scores: dict[str, float] = {}
    best_alpha = 1.0
    best_score = float("inf")
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        model = ridge_model(alpha)
        model.fit(train[columns], np.log(train_y))
        prediction = positive(np.exp(model.predict(dev[columns])), lower, upper)
        score = metrics(dev_y, prediction)["qlike"]
        scores[str(alpha)] = score
        if score < best_score:
            best_score = score
            best_alpha = alpha

    fit = frame[frame[split_column].isin(["train", "dev"])]
    final_model = ridge_model(best_alpha)
    final_model.fit(fit[columns], np.log(fit[TARGET].to_numpy(float)))
    return final_model, best_alpha, scores


def tune_hist_gradient_boosting(
    frame: pd.DataFrame,
    columns: list[str],
    split_column: str,
) -> tuple[HistGradientBoostingRegressor, int, dict[str, float]]:
    train = frame[frame[split_column] == "train"]
    dev = frame[frame[split_column] == "dev"]
    train_y = train[TARGET].to_numpy(float)
    dev_y = dev[TARGET].to_numpy(float)
    lower, upper = bounds(train_y)

    scores: dict[str, float] = {}
    best_leaves = 15
    best_score = float("inf")
    for leaves in (7, 15, 31):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=leaves,
            l2_regularization=1.0,
            random_state=20260927,
        )
        model.fit(train[columns], np.log(train_y))
        prediction = positive(np.exp(model.predict(dev[columns])), lower, upper)
        score = metrics(dev_y, prediction)["qlike"]
        scores[str(leaves)] = score
        if score < best_score:
            best_score = score
            best_leaves = leaves

    fit = frame[frame[split_column].isin(["train", "dev"])]
    final_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=best_leaves,
        l2_regularization=1.0,
        random_state=20260927,
    )
    final_model.fit(fit[columns], np.log(fit[TARGET].to_numpy(float)))
    return final_model, best_leaves, scores


def evaluate_regime(
    frame: pd.DataFrame,
    split_column: str,
    feature_groups: dict[str, list[str]],
) -> tuple[dict, list[dict]]:
    train = frame[frame[split_column] == "train"]
    fit = frame[frame[split_column].isin(["train", "dev"])]
    test = frame[frame[split_column] == "test"].copy()
    fit_y = fit[TARGET].to_numpy(float)
    test_y = test[TARGET].to_numpy(float)
    lower, upper = bounds(fit_y)

    results: dict[str, dict] = {}
    predictions: list[dict] = []

    def record(model_name: str, prediction: np.ndarray, tuning: dict | None = None) -> None:
        prediction = positive(prediction, lower, upper)
        results[model_name] = {
            "test": metrics(test_y, prediction),
            "tuning": tuning or {},
        }
        for call_id, actual, predicted in zip(
            test["call_id"], test_y, prediction, strict=True
        ):
            predictions.append(
                {
                    "evaluation_regime": split_column,
                    "model": model_name,
                    "call_id": call_id,
                    "actual": actual,
                    "predicted": predicted,
                }
            )

    record("constant_median", np.full(len(test), np.median(fit_y)))

    company_means = fit.groupby("company_id")[TARGET].mean()
    global_mean = float(fit_y.mean())
    company_prediction = np.array(
        [company_means.get(company, global_mean) for company in test["company_id"]]
    )
    record("company_identity_mean", company_prediction)

    for window in (5, 20, 60):
        column = f"price_{window}d_realized_variance"
        fill_value = float(train[column].median())
        horizon_scale = 5.0 / window
        direct = test[column].fillna(fill_value).to_numpy(float) * horizon_scale
        record(
            f"historical_variance_{window}d_scaled",
            direct,
            {"forecast_horizon_days": 5, "history_window_days": window,
             "fixed_scale": horizon_scale},
        )

    for group_name, columns in feature_groups.items():
        model, alpha, dev_scores = tune_ridge(frame, columns, split_column)
        prediction = np.exp(model.predict(test[columns]))
        record(
            f"ridge_{group_name}",
            prediction,
            {"selected_alpha": alpha, "dev_qlike_by_alpha": dev_scores},
        )

    for group_name in (
        "price",
        "price_text_structure",
        "price_acoustic",
        "combined",
    ):
        columns = feature_groups[group_name]
        hgb, leaves, dev_scores = tune_hist_gradient_boosting(
            frame, columns, split_column
        )
        prediction = np.exp(hgb.predict(test[columns]))
        record(
            f"hist_gradient_boosting_{group_name}",
            prediction,
            {"selected_max_leaf_nodes": leaves, "dev_qlike_by_leaves": dev_scores},
        )

    ordered = sorted(
        results.items(), key=lambda item: item[1]["test"]["qlike"]
    )
    return {
        "split_column": split_column,
        "train_calls": int((frame[split_column] == "train").sum()),
        "dev_calls": int((frame[split_column] == "dev").sum()),
        "test_calls": int((frame[split_column] == "test").sum()),
        "models": dict(ordered),
        "best_test_model_by_qlike_for_reporting_only": ordered[0][0],
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    feature_path = root / "data/processed/ahenk_tabular_features.parquet"
    frame = pd.read_parquet(feature_path).sort_values(["call_date", "call_id"])

    all_features = [column for column in frame.columns if column not in METADATA]
    acoustic = [
        column
        for column in all_features
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    text = [column for column in all_features if column.startswith("text_")]
    price = [column for column in all_features if column.startswith("price_")]
    feature_groups = {
        "price": price,
        "text_structure": text,
        "acoustic": acoustic,
        "price_text_structure": price + text,
        "price_acoustic": price + acoustic,
        "combined": all_features,
    }

    regimes: dict[str, dict] = {}
    prediction_rows: list[dict] = []
    with threadpool_limits(limits=4):
        for split_column in ("temporal_split", "company_holdout_split"):
            print(f"Çalıştırılıyor: {split_column}")
            regime, predictions = evaluate_regime(
                frame, split_column, feature_groups
            )
            regimes[split_column] = regime
            prediction_rows.extend(predictions)

    predictions_path = root / "artifacts/predictions/cpu_baseline_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_rows": len(frame),
        "feature_groups": {key: len(value) for key, value in feature_groups.items()},
        "target_transform": "natural_log",
        "primary_metric": "QLIKE",
        "test_used_for_hyperparameter_selection": False,
        "future_prices_used_as_inputs": False,
        "human_or_expert_labels_used": False,
        "regimes": regimes,
        "status": "PASS",
    }
    report_path = root / "reports/baselines/cpu_baselines.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {report_path}")
    print(f"Tahminler: {predictions_path}")


if __name__ == "__main__":
    main()