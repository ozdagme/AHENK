"""AHENK için metin içeriğini kullanan sızıntısız TF-IDF tabanları.

Kelime dağarcığı ve bütün sayısal dönüşümler yalnız train üzerinde öğrenilir;
hiperparametre dev üzerinde seçilir. Seçimden sonra train+dev ile yeniden
eğitilip test bir kez ölçülür. GPU veya insan etiketi kullanmaz.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


EPSILON = 1e-12
TARGET = "target_realized_variance_5d"


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.maximum(np.asarray(actual, dtype=float), EPSILON)
    predicted = np.maximum(np.asarray(predicted, dtype=float), EPSILON)
    ratio = actual / predicted
    return {
        "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "log_rmse": float(
            np.sqrt(np.mean(np.square(np.log(actual) - np.log(predicted))))
        ),
    }


def bounds(values: np.ndarray) -> tuple[float, float]:
    return (
        float(max(np.quantile(values, 0.001), EPSILON)),
        float(max(np.quantile(values, 0.999), EPSILON * 10)),
    )


def read_texts(root: Path, index: pd.DataFrame) -> dict[str, str]:
    texts = {}
    for row in index.itertuples(index=False):
        texts[row.call_id] = (root / row.text_path).read_text(
            encoding="utf-8", errors="replace"
        )
    return texts


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )


def numeric_matrix(
    fit: pd.DataFrame,
    other: pd.DataFrame,
    columns: list[str],
) -> tuple[csr_matrix, csr_matrix]:
    if not columns:
        return csr_matrix((len(fit), 0)), csr_matrix((len(other), 0))
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    fit_values = imputer.fit_transform(fit[columns])
    other_values = imputer.transform(other[columns])
    scaler = StandardScaler()
    fit_values = scaler.fit_transform(fit_values)
    other_values = scaler.transform(other_values)
    return csr_matrix(fit_values), csr_matrix(other_values)


def build_matrices(
    fit: pd.DataFrame,
    other: pd.DataFrame,
    texts: dict[str, str],
    numeric_columns: list[str],
) -> tuple[csr_matrix, csr_matrix, int]:
    tfidf = vectorizer()
    fit_text = [texts[call_id] for call_id in fit["call_id"]]
    other_text = [texts[call_id] for call_id in other["call_id"]]
    fit_tfidf = tfidf.fit_transform(fit_text)
    other_tfidf = tfidf.transform(other_text)
    fit_numeric, other_numeric = numeric_matrix(
        fit, other, numeric_columns
    )
    return (
        hstack([fit_tfidf, fit_numeric], format="csr"),
        hstack([other_tfidf, other_numeric], format="csr"),
        len(tfidf.vocabulary_),
    )


def evaluate_group(
    frame: pd.DataFrame,
    texts: dict[str, str],
    split_column: str,
    numeric_columns: list[str],
) -> tuple[dict, pd.DataFrame]:
    train = frame[frame[split_column] == "train"]
    dev = frame[frame[split_column] == "dev"]
    test = frame[frame[split_column] == "test"]
    train_y = train[TARGET].to_numpy(float)
    dev_y = dev[TARGET].to_numpy(float)

    train_x, dev_x, train_vocabulary = build_matrices(
        train, dev, texts, numeric_columns
    )
    lower, upper = bounds(train_y)
    scores: dict[str, float] = {}
    best_alpha = 10.0
    best_score = float("inf")
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train_x, np.log(train_y))
        prediction = np.clip(np.exp(model.predict(dev_x)), lower, upper)
        score = metrics(dev_y, prediction)["qlike"]
        scores[str(alpha)] = score
        if score < best_score:
            best_alpha = alpha
            best_score = score

    fit = frame[frame[split_column].isin(["train", "dev"])]
    fit_y = fit[TARGET].to_numpy(float)
    fit_x, test_x, final_vocabulary = build_matrices(
        fit, test, texts, numeric_columns
    )
    lower, upper = bounds(fit_y)
    model = Ridge(alpha=best_alpha, solver="lsqr")
    model.fit(fit_x, np.log(fit_y))
    prediction = np.clip(np.exp(model.predict(test_x)), lower, upper)

    prediction_frame = pd.DataFrame(
        {
            "evaluation_regime": split_column,
            "call_id": test["call_id"].to_numpy(),
            "actual": test[TARGET].to_numpy(float),
            "predicted": prediction,
        }
    )
    return {
        "selected_alpha": best_alpha,
        "dev_qlike_by_alpha": scores,
        "train_vocabulary_size": train_vocabulary,
        "final_vocabulary_size": final_vocabulary,
        "numeric_feature_count": len(numeric_columns),
        "test": metrics(test[TARGET].to_numpy(float), prediction),
    }, prediction_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    index = pd.read_csv(root / "data/processed/ahenk_dataset_index.csv")
    tabular = pd.read_parquet(root / "data/processed/ahenk_tabular_features.parquet")
    frame = tabular.merge(
        index[["call_id", "text_path"]], on="call_id", how="left", validate="one_to_one"
    ).sort_values(["call_date", "call_id"])
    texts = read_texts(root, frame[["call_id", "text_path"]])

    price_columns = [column for column in frame if column.startswith("price_")]
    acoustic_columns = [
        column
        for column in frame
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    groups = {
        "tfidf_text": [],
        "price_tfidf": price_columns,
        "price_acoustic_tfidf": price_columns + acoustic_columns,
    }

    regimes: dict[str, dict] = {}
    predictions = []
    with threadpool_limits(limits=4):
        for split_column in ("temporal_split", "company_holdout_split"):
            regimes[split_column] = {}
            for group_name, numeric_columns in groups.items():
                print(f"Çalıştırılıyor: {split_column} / {group_name}")
                result, prediction_frame = evaluate_group(
                    frame, texts, split_column, numeric_columns
                )
                regimes[split_column][group_name] = result
                prediction_frame["model"] = group_name
                predictions.append(prediction_frame)

    prediction_output = root / "artifacts/predictions/tfidf_baseline_predictions.csv"
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(prediction_output, index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_representation": "train-only TF-IDF word uni/bi-grams",
        "test_used_for_hyperparameter_selection": False,
        "future_prices_used_as_inputs": False,
        "human_or_expert_labels_used": False,
        "regimes": regimes,
        "status": "PASS",
    }
    output = root / "reports/baselines/tfidf_baselines.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")
    print(f"Tahminler: {prediction_output}")


if __name__ == "__main__":
    main()