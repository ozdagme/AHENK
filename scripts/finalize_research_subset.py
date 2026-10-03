"""P02-AHENK yfinance-kapsamlı araştırma alt kümesini dondurur.

Özgün %70 piyasa kapsama kapısı geçilmediği için bunu PASS olarak değiştirmez.
Kullanıcının yalnız yfinance ile devam kararı üzerine, fiyatı erişilebilen çağrıları
ayrı ve açıkça sınırlı bir araştırma evreni olarak kaydeder. GPU kullanmaz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def company_split(company_id: str) -> str:
    bucket = int(hashlib.sha256(company_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def temporal_cutoffs(rows: list[dict[str, str]]) -> tuple[str, str]:
    dates = sorted(row["call_date"] for row in rows)
    train_index = min(len(dates) - 1, max(0, math.ceil(len(dates) * 0.70) - 1))
    dev_index = min(len(dates) - 1, max(0, math.ceil(len(dates) * 0.85) - 1))
    return dates[train_index], dates[dev_index]


def temporal_split(call_date: str, train_end: str, dev_end: str) -> str:
    if call_date <= train_end:
        return "train"
    if call_date <= dev_end:
        return "dev"
    return "test"


def summarize_split(rows: list[dict[str, str]], field: str) -> dict:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {
        name: {
            "calls": len(group),
            "companies": len({row["company_id"] for row in group}),
            "date_min": min(row["call_date"] for row in group),
            "date_max": max(row["call_date"] for row in group),
        }
        for name, group in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    source = root / "data/manifests/market_full_calls.csv"
    if not source.is_file():
        raise SystemExit(f"Tam piyasa manifesti bulunamadı: {source}")

    with source.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    eligible: list[dict[str, str]] = []
    exclusion_reasons: Counter[str] = Counter()
    seen_call_ids: set[str] = set()
    for row in all_rows:
        reason = None
        if row.get("call_id") in seen_call_ids:
            reason = "duplicate_call_id"
        elif row.get("market_status") != "PASS":
            reason = row.get("market_error") or "market_status_not_pass"
        elif not row.get("text_path") or not row.get("audio_feature_path"):
            reason = "missing_text_or_audio_path"
        else:
            try:
                target = float(row["target_realized_variance_5d"])
                if not math.isfinite(target) or target <= 0:
                    reason = "invalid_target"
            except (TypeError, ValueError):
                reason = "invalid_target"

        if reason:
            exclusion_reasons[reason] += 1
            continue
        seen_call_ids.add(row["call_id"])
        eligible.append(row)

    if len(eligible) < 1000:
        raise SystemExit(
            f"Yalnız {len(eligible)} geçerli çağrı bulundu; modelleme için kapsam yetersiz."
        )

    eligible.sort(key=lambda row: (row["call_date"], row["call_id"]))
    train_end, dev_end = temporal_cutoffs(eligible)
    for row in eligible:
        row["temporal_split"] = temporal_split(row["call_date"], train_end, dev_end)
        row["company_holdout_split"] = company_split(row["company_id"])
        row["research_scope"] = "yahoo_historical_price_available"

    output = root / "data/processed/ahenk_dataset_index.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(eligible[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eligible)

    targets = pd.Series(
        [float(row["target_realized_variance_5d"]) for row in eligible],
        dtype="float64",
    )
    target_quantiles = {
        str(quantile): float(targets.quantile(quantile))
        for quantile in (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0)
    }
    calls_by_year = Counter(row["call_date"][:4] for row in eligible)

    observed_coverage = len(eligible) / len(all_rows)
    original_gate_status = "PASS" if observed_coverage >= 0.70 else "FAIL"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_market_gate": {
            "required_coverage": 0.70,
            "observed_coverage": round(observed_coverage, 6),
            "status": original_gate_status,
        },
        "scope_decision": {
            "status": "PASS_WITH_SCOPE_LIMITATION",
            "scope": "MAEC calls with Yahoo Finance historical prices available on retrieval date",
            "reason": "No institutional price-data subscription; yfinance-only continuation approved.",
            "raw_market_data_redistribution": False,
        },
        "total_maec_calls": len(all_rows),
        "eligible_calls": len(eligible),
        "excluded_calls": len(all_rows) - len(eligible),
        "exclusion_reasons": dict(exclusion_reasons),
        "eligible_companies": len({row["company_id"] for row in eligible}),
        "calls_by_year": dict(sorted(calls_by_year.items())),
        "temporal_cutoffs": {"train_end": train_end, "dev_end": dev_end},
        "temporal_split": summarize_split(eligible, "temporal_split"),
        "company_holdout_split": summarize_split(eligible, "company_holdout_split"),
        "target_quantiles": target_quantiles,
        "human_or_expert_labels_used": False,
        "input_manifest_sha256": sha256(source),
        "dataset_index_sha256": sha256(output),
        "dataset_status": "FROZEN_FOR_DATA_QUALITY_AUDIT",
    }

    report_path = root / "reports/data_quality/dataset_scope.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Dondurulmuş indeks: {output}")
    print(f"Kapsam raporu: {report_path}")


if __name__ == "__main__":
    main()