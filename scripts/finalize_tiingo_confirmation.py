"""Tiingo adaylarını dondurur ve AHENK dış doğrulama özelliklerini üretir.

Bu betik yalnız Tiingo toplaması tamamen tükendiğinde çalışır. Dış kümenin
hedeflerine göre seçim yapmaz; mevcut model indeksinde yer alan çağrıları
reddeder. Çıktı daha sonra tek seferlik AHENK V3 doğrulamasında kullanılacaktır.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_tabular_features import acoustic_features, text_structure_features


TARGET = "target_realized_variance_5d"
PRICE_COLUMNS = [
    "price_history_days",
    *[
        f"price_{window}d_{suffix}"
        for window in (5, 20, 60)
        for suffix in (
            "available_returns",
            "log_return",
            "realized_variance",
            "volatility",
            "mean_absolute_return",
        )
    ],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    protocol_path = root / "reports/confirmation/tiingo_confirmation_protocol.json"
    progress_path = root / "reports/confirmation/tiingo_confirmation_progress.json"
    candidates_path = root / "data/confirmation/tiingo_confirmation_candidates.csv"
    model_index_path = root / "data/processed/ahenk_dataset_index.csv"
    required = (protocol_path, progress_path, candidates_path, model_index_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Eksik dosyalar: " + ", ".join(missing))

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("retryable_or_unattempted_symbols") != 0:
        raise SystemExit("Tiingo toplaması bitmedi; dondurma yapılmadı.")
    if progress.get("status") not in {"COMPLETE", "EXHAUSTED_WITH_GAPS"}:
        raise SystemExit(f"Beklenmeyen toplama durumu: {progress.get('status')}")
    if progress.get("protocol_sha256") != sha256(protocol_path):
        raise SystemExit("İlerleme raporu ile kilitli protokol hash'i uyuşmuyor.")

    candidates = read_csv(candidates_path)
    model_index = read_csv(model_index_path)
    if len(candidates) != int(progress.get("recovered_calls", -1)):
        raise SystemExit("Aday satır sayısı ilerleme raporuyla uyuşmuyor.")
    candidate_ids = [row.get("call_id", "") for row in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise SystemExit("Aday call_id alanları boş veya yinelenmiş.")
    used_ids = {row["call_id"] for row in model_index}
    overlap = used_ids.intersection(candidate_ids)
    if overlap:
        raise SystemExit(f"Dış küme model verisiyle çakışıyor: {len(overlap)} çağrı")

    targets = pd.to_numeric(
        pd.Series([row.get(TARGET, "") for row in candidates]), errors="coerce"
    )
    if targets.isna().any() or not np.isfinite(targets).all() or (targets <= 0).any():
        raise SystemExit("Dış kümede geçersiz finansal hedef var.")

    candidates.sort(key=lambda row: (row["call_date"], row["call_id"]))
    frozen_index_path = root / "data/confirmation/tiingo_confirmation_index.csv"
    frozen_index_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_fields = list(candidates[0].keys())
    if "confirmation_role" not in frozen_fields:
        frozen_fields.append("confirmation_role")
    with frozen_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=frozen_fields)
        writer.writeheader()
        for row in candidates:
            writer.writerow({**row, "confirmation_role": "external_test_locked"})

    nonnumeric_tokens: Counter[str] = Counter()
    feature_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for number, row in enumerate(candidates, start=1):
        try:
            text_path = root / row["text_path"]
            acoustic_path = root / row["audio_feature_path"]
            if not text_path.is_file() or not acoustic_path.is_file():
                raise FileNotFoundError("metin veya akustik özellik dosyası bulunamadı")
            output: dict[str, object] = {
                "call_id": row["call_id"],
                "company_id": row["company_id"],
                "ticker": row["ticker"],
                "call_date": row["call_date"],
                "text_path": row["text_path"],
                TARGET: float(row[TARGET]),
            }
            output.update(acoustic_features(acoustic_path, nonnumeric_tokens))
            output.update(text_structure_features(text_path))
            for column in PRICE_COLUMNS:
                value = row.get(column, "")
                output[column] = float(value) if value not in {"", None} else float("nan")
            feature_rows.append(output)
        except Exception as exc:
            errors.append(
                {"call_id": row.get("call_id", ""), "error": f"{type(exc).__name__}: {exc}"}
            )
        if number % 100 == 0:
            print(f"{number}/{len(candidates)} dış çağrı hazırlandı.")

    if errors or len(feature_rows) != len(candidates):
        report_path = root / "reports/confirmation/tiingo_confirmation_freeze.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "FEATURE_BUILD_FAILED",
                    "candidate_calls": len(candidates),
                    "feature_rows_created": len(feature_rows),
                    "errors_first_50": errors[:50],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"{len(errors)} özellik üretim hatası; raporu inceleyin.")

    features = pd.DataFrame(feature_rows)
    features_path = root / "data/confirmation/tiingo_confirmation_features.parquet"
    features.to_parquet(features_path, index=False)

    original_companies = {row["company_id"] for row in model_index}
    external_companies = set(features["company_id"].astype(str))
    calls_by_year = Counter(str(value)[:4] for value in features["call_date"])
    numeric_features = features.drop(
        columns=["call_id", "company_id", "ticker", "call_date", "text_path", TARGET]
    ).apply(pd.to_numeric, errors="coerce")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_EXTERNAL_CONFIRMATION",
        "collection_status": progress.get("status"),
        "eligible_yahoo_failed_calls": progress.get("eligible_calls"),
        "recovered_calls": len(features),
        "unavailable_calls": int(progress.get("eligible_calls", 0)) - len(features),
        "recovery_fraction": len(features) / int(progress.get("eligible_calls", 1)),
        "external_companies": len(external_companies),
        "companies_unseen_in_original_model_data": len(external_companies - original_companies),
        "calls_from_unseen_companies": int(
            features["company_id"].astype(str).isin(external_companies - original_companies).sum()
        ),
        "date_min": str(features["call_date"].min()),
        "date_max": str(features["call_date"].max()),
        "calls_by_year": dict(sorted(calls_by_year.items())),
        "target_quantiles": {
            str(q): float(features[TARGET].quantile(q))
            for q in (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0)
        },
        "feature_count": int(len(numeric_features.columns)),
        "columns_with_missing_values": int(numeric_features.isna().any().sum()),
        "maximum_feature_missing_fraction": float(numeric_features.isna().mean().max()),
        "top_nonnumeric_acoustic_tokens": dict(nonnumeric_tokens.most_common(20)),
        "protocol_sha256": sha256(protocol_path),
        "progress_sha256": sha256(progress_path),
        "candidate_manifest_sha256": sha256(candidates_path),
        "frozen_index_sha256": sha256(frozen_index_path),
        "features_sha256": sha256(features_path),
        "overlap_with_original_model_call_ids": 0,
        "raw_prices_persisted": False,
        "api_key_persisted": False,
        "human_or_expert_labels_used": False,
        "model_selection_on_external_set_allowed": False,
        "next_step": "single confirmatory run of frozen V3 and frozen fixed-mean baseline",
    }
    report_path = root / "reports/confirmation/tiingo_confirmation_freeze.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Dondurulmuş indeks: {frozen_index_path}")
    print(f"Dış özellikler: {features_path}")
    print(f"Dondurma raporu: {report_path}")


if __name__ == "__main__":
    main()