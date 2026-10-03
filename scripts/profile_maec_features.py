"""Dondurulmuş AHENK alt kümesinde metin ve akustik özellik kalitesini profiller.

GPU ve insan etiketi kullanmaz. Ham metin veya özellikleri rapora kopyalamaz;
yalnız şema, boyut, eksiklik ve hizalama istatistiklerini üretir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def text_stats(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "characters": len(text),
        "nonempty_lines": len(lines),
        "replacement_characters": text.count("\ufffd"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="0 bütün dondurulmuş çağrıları işler.",
    )
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    index_path = root / "data/processed/ahenk_dataset_index.csv"
    if not index_path.is_file():
        raise SystemExit(f"Dondurulmuş veri indeksi bulunamadı: {index_path}")

    with index_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_calls > 0:
        rows = rows[: args.max_calls]

    schema_counts: Counter[str] = Counter()
    column_counts: Counter[int] = Counter()
    feature_row_counts: list[int] = []
    text_line_counts: list[int] = []
    exact_alignment = 0
    missing_paths = 0
    parse_errors: list[dict[str, str]] = []
    total_cells = 0
    missing_cells = 0
    infinite_cells = 0
    nonnumeric_cells = 0
    total_replacement_characters = 0
    empty_texts = 0
    empty_feature_files = 0

    for number, row in enumerate(rows, start=1):
        text_path = root / row["text_path"]
        feature_path = root / row["audio_feature_path"]
        if not text_path.is_file() or not feature_path.is_file():
            missing_paths += 1
            parse_errors.append(
                {"call_id": row["call_id"], "error": "missing_text_or_feature_path"}
            )
            continue

        try:
            tstats = text_stats(text_path)
            total_replacement_characters += tstats["replacement_characters"]
            text_line_counts.append(tstats["nonempty_lines"])
            if tstats["characters"] == 0:
                empty_texts += 1

            frame = pd.read_csv(feature_path)
            feature_row_counts.append(len(frame))
            if frame.empty:
                empty_feature_files += 1

            schema = "|".join(str(column) for column in frame.columns)
            schema_counts[schema] += 1
            column_counts[len(frame.columns)] += 1

            if len(frame) == tstats["nonempty_lines"]:
                exact_alignment += 1

            numeric = frame.apply(pd.to_numeric, errors="coerce")
            total_cells += int(frame.size)
            missing_cells += int(numeric.isna().sum().sum())
            original_nonmissing = int(frame.notna().sum().sum())
            numeric_nonmissing = int(numeric.notna().sum().sum())
            nonnumeric_cells += max(0, original_nonmissing - numeric_nonmissing)

            values = numeric.to_numpy(dtype="float64", na_value=float("nan"))
            infinite_cells += sum(
                1 for value in values.ravel() if math.isinf(float(value))
            )
        except Exception as exc:
            parse_errors.append(
                {"call_id": row["call_id"], "error": f"{type(exc).__name__}: {exc}"}
            )

        if number % 250 == 0:
            print(f"{number}/{len(rows)} çağrı profillendi.")

    processed = len(rows) - len(parse_errors)
    dominant_schema_count = schema_counts.most_common(1)[0][1] if schema_counts else 0
    dominant_schema_fraction = dominant_schema_count / processed if processed else 0.0
    alignment_fraction = exact_alignment / processed if processed else 0.0
    missing_fraction = missing_cells / total_cells if total_cells else 1.0

    status = (
        "PASS"
        if processed >= 1000
        and len(parse_errors) == 0
        and dominant_schema_fraction >= 0.95
        and missing_fraction <= 0.01
        else "REVIEW_REQUIRED"
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calls_requested": len(rows),
        "calls_processed": processed,
        "missing_paths": missing_paths,
        "parse_error_count": len(parse_errors),
        "parse_errors_first_50": parse_errors[:50],
        "schema_count": len(schema_counts),
        "dominant_schema_fraction": round(dominant_schema_fraction, 6),
        "column_count_distribution": {
            str(key): value for key, value in sorted(column_counts.items())
        },
        "top_schema_counts": [
            {"columns": schema.split("|"), "calls": count}
            for schema, count in schema_counts.most_common(5)
        ],
        "feature_rows": {
            "min": min(feature_row_counts) if feature_row_counts else None,
            "median": float(pd.Series(feature_row_counts).median())
            if feature_row_counts
            else None,
            "max": max(feature_row_counts) if feature_row_counts else None,
        },
        "text_nonempty_lines": {
            "min": min(text_line_counts) if text_line_counts else None,
            "median": float(pd.Series(text_line_counts).median())
            if text_line_counts
            else None,
            "max": max(text_line_counts) if text_line_counts else None,
        },
        "exact_text_feature_row_alignment_fraction": round(alignment_fraction, 6),
        "total_feature_cells": total_cells,
        "missing_or_unparseable_cells": missing_cells,
        "missing_or_unparseable_fraction": round(missing_fraction, 8),
        "explicit_nonnumeric_cells": nonnumeric_cells,
        "infinite_cells": infinite_cells,
        "empty_texts": empty_texts,
        "empty_feature_files": empty_feature_files,
        "text_replacement_characters": total_replacement_characters,
        "human_or_expert_labels_used": False,
        "status": status,
    }

    output = root / "reports/data_quality/maec_feature_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor yazıldı: {output}")


if __name__ == "__main__":
    main()