"""AHENK çağrı düzeyi akustik, metin-yapı ve geçmiş fiyat özelliklerini üretir.

Yalnız çağrı tarihinden önceki fiyatlar kullanılır. Sayısal olmayan akustik
değerler NaN olarak korunur; doldurma yalnız sonraki modelleme aşamasında ve
eğitim ayrımı üzerinde yapılacaktır. GPU ve insan etiketi kullanmaz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def safe_column(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value or "unnamed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acoustic_features(path: Path, token_counts: Counter[str]) -> dict[str, float]:
    raw = pd.read_csv(path)
    numeric = raw.apply(pd.to_numeric, errors="coerce")

    for column in raw.columns:
        invalid = raw[column].notna() & numeric[column].isna()
        token_counts.update(str(value) for value in raw.loc[invalid, column])

    output: dict[str, float] = {"acoustic_segment_count": float(len(raw))}
    for column in raw.columns:
        name = safe_column(str(column))
        series = numeric[column]
        output[f"ac_{name}_mean"] = float(series.mean())
        output[f"ac_{name}_std"] = float(series.std(ddof=0))
        output[f"ac_{name}_min"] = float(series.min())
        output[f"ac_{name}_max"] = float(series.max())
        output[f"ac_{name}_median"] = float(series.median())
        output[f"ac_{name}_missing_rate"] = float(series.isna().mean())
    return output


def text_structure_features(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokens = text.split()
    characters = len(text)
    digits = sum(character.isdigit() for character in text)
    uppercase = sum(character.isupper() for character in text)
    return {
        "text_character_count": float(characters),
        "text_token_count": float(len(tokens)),
        "text_nonempty_line_count": float(len(lines)),
        "text_average_line_characters": float(
            sum(len(line) for line in lines) / max(len(lines), 1)
        ),
        "text_digit_fraction": float(digits / max(characters, 1)),
        "text_uppercase_fraction": float(uppercase / max(characters, 1)),
    }


def load_price_history(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    column = "Adj Close" if "Adj Close" in frame.columns else "Close"
    prices = pd.to_numeric(frame[column], errors="coerce").dropna()
    prices = prices[prices > 0]
    return prices[~prices.index.duplicated(keep="last")].sort_index()


def price_features(prices: pd.Series, call_date: str) -> dict[str, float]:
    past = prices[prices.index < pd.Timestamp(call_date)]
    output: dict[str, float] = {"price_history_days": float(len(past))}
    returns = np.log(past / past.shift(1)).dropna()

    for window in (5, 20, 60):
        sample = returns.iloc[-window:]
        prefix = f"price_{window}d"
        output[f"{prefix}_available_returns"] = float(len(sample))
        output[f"{prefix}_log_return"] = float(sample.sum()) if len(sample) else float("nan")
        output[f"{prefix}_realized_variance"] = (
            float(np.square(sample).sum()) if len(sample) else float("nan")
        )
        output[f"{prefix}_volatility"] = (
            float(sample.std(ddof=0)) if len(sample) else float("nan")
        )
        output[f"{prefix}_mean_absolute_return"] = (
            float(sample.abs().mean()) if len(sample) else float("nan")
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    index_path = root / "data/processed/ahenk_dataset_index.csv"
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))

    price_cache: dict[str, pd.Series] = {}
    nonnumeric_tokens: Counter[str] = Counter()
    output_rows: list[dict[str, str | float]] = []
    errors: list[dict[str, str]] = []

    for number, row in enumerate(index_rows, start=1):
        try:
            text_path = root / row["text_path"]
            acoustic_path = root / row["audio_feature_path"]
            symbol = row["yahoo_symbol"]
            price_path = root / "data/raw/market/yfinance" / f"{safe_name(symbol)}.csv"
            if symbol not in price_cache:
                price_cache[symbol] = load_price_history(price_path)

            features: dict[str, str | float] = {
                "call_id": row["call_id"],
                "company_id": row["company_id"],
                "ticker": row["ticker"],
                "call_date": row["call_date"],
                "temporal_split": row["temporal_split"],
                "company_holdout_split": row["company_holdout_split"],
                "target_realized_variance_5d": float(
                    row["target_realized_variance_5d"]
                ),
            }
            features.update(acoustic_features(acoustic_path, nonnumeric_tokens))
            features.update(text_structure_features(text_path))
            features.update(price_features(price_cache[symbol], row["call_date"]))
            output_rows.append(features)
        except Exception as exc:
            errors.append(
                {"call_id": row.get("call_id", ""), "error": f"{type(exc).__name__}: {exc}"}
            )

        if number % 250 == 0:
            print(f"{number}/{len(index_rows)} çağrı için özellik üretildi.")

    if not output_rows:
        raise SystemExit("Hiç özellik satırı üretilemedi.")

    frame = pd.DataFrame(output_rows)
    output_path = root / "data/processed/ahenk_tabular_features.parquet"
    frame.to_parquet(output_path, index=False)

    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "call_id",
            "company_id",
            "ticker",
            "call_date",
            "temporal_split",
            "company_holdout_split",
            "target_realized_variance_5d",
        }
    ]
    numeric_frame = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    missing_by_column = numeric_frame.isna().mean().sort_values(ascending=False)
    labels = frame["target_realized_variance_5d"].astype(float)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calls_in_index": len(index_rows),
        "feature_rows_created": len(frame),
        "error_count": len(errors),
        "errors_first_50": errors[:50],
        "feature_count": len(feature_columns),
        "acoustic_feature_count": sum(column.startswith("ac_") for column in feature_columns),
        "text_structure_feature_count": sum(
            column.startswith("text_") for column in feature_columns
        ),
        "price_feature_count": sum(column.startswith("price_") for column in feature_columns),
        "columns_with_missing_values": int((missing_by_column > 0).sum()),
        "maximum_feature_missing_fraction": float(missing_by_column.max()),
        "top_missing_columns": {
            column: float(value) for column, value in missing_by_column.head(20).items()
        },
        "top_nonnumeric_acoustic_tokens": dict(nonnumeric_tokens.most_common(20)),
        "target_positive_and_finite": bool(
            np.isfinite(labels).all() and bool((labels > 0).all())
        ),
        "future_prices_used_as_inputs": False,
        "human_or_expert_labels_used": False,
        "input_index_sha256": sha256(index_path),
        "output_parquet_sha256": sha256(output_path),
        "status": "PASS"
        if len(frame) == len(index_rows) and not errors
        else "REVIEW_REQUIRED",
    }

    report_path = root / "reports/data_quality/tabular_feature_build.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Özellik dosyası: {output_path}")
    print(f"Rapor: {report_path}")


if __name__ == "__main__":
    main()