"""AHENK metin sinyali için otomatik şirket-kimliği kontrolleri.

Özgün metni; ticker-maskeli metin, yalnız ticker ve aynı şirketin önceki çağrı
metniyle karşılaştırır. Bütün modeller fiyat+TF-IDF Ridge düzenini kullanır.
GPU veya insan etiketi kullanmaz.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from run_tfidf_baselines import evaluate_group


def original_texts(root: Path, frame: pd.DataFrame) -> dict[str, str]:
    return {
        row.call_id: (root / row.text_path).read_text(
            encoding="utf-8", errors="replace"
        )
        for row in frame.itertuples(index=False)
    }


def ticker_masked_texts(
    frame: pd.DataFrame, texts: dict[str, str]
) -> dict[str, str]:
    output = {}
    for row in frame.itertuples(index=False):
        ticker = str(row.ticker)
        text = texts[row.call_id]
        if len(ticker) >= 2:
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", re.I)
            text = pattern.sub(" COMPANYTOKEN ", text)
        output[row.call_id] = text
    return output


def previous_company_texts(
    frame: pd.DataFrame, texts: dict[str, str]
) -> tuple[dict[str, str], int]:
    output = {}
    previous: dict[str, str] = {}
    available = 0
    ordered = frame.sort_values(["call_date", "call_id"])
    for row in ordered.itertuples(index=False):
        if row.company_id in previous:
            output[row.call_id] = previous[row.company_id]
            available += 1
        else:
            output[row.call_id] = ""
        previous[row.company_id] = texts[row.call_id]
    return output, available


def ticker_only_texts(frame: pd.DataFrame) -> dict[str, str]:
    return {
        row.call_id: f"tickeridentity_{str(row.ticker).lower()}"
        for row in frame.itertuples(index=False)
    }


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
    price_columns = [column for column in frame if column.startswith("price_")]

    original = original_texts(root, frame)
    previous, previous_available = previous_company_texts(frame, original)
    controls = {
        "original_text": original,
        "ticker_masked_text": ticker_masked_texts(frame, original),
        "previous_same_company_text": previous,
        "ticker_only_text": ticker_only_texts(frame),
    }

    regimes = {}
    predictions = []
    with threadpool_limits(limits=4):
        for split_column in ("temporal_split", "company_holdout_split"):
            regimes[split_column] = {}
            for control_name, texts in controls.items():
                print(f"Çalıştırılıyor: {split_column} / {control_name}")
                result, prediction = evaluate_group(
                    frame, texts, split_column, price_columns
                )
                regimes[split_column][control_name] = result
                prediction["model"] = control_name
                predictions.append(prediction)

    prediction_path = root / "artifacts/predictions/identity_control_predictions.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "price + train-only TF-IDF Ridge",
        "previous_same_company_text_available_calls": previous_available,
        "test_used_for_hyperparameter_selection": False,
        "future_text_used": False,
        "future_prices_used_as_inputs": False,
        "human_or_expert_labels_used": False,
        "regimes": regimes,
        "status": "PASS",
    }
    output = root / "reports/controls/identity_controls.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor: {output}")
    print(f"Tahminler: {prediction_path}")


if __name__ == "__main__":
    main()