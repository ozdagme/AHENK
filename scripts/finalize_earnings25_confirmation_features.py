#!/usr/bin/env python3
"""Build and freeze the model-ready Earnings25 forward confirmation table.

The target-blind text/acoustic artifacts and the one-time market/target
retrieval must already be frozen.  This script creates the same call-level
acoustic, text-structure, and price interface used by AHENK, checks it against
the original training schema, and writes immutable Parquet/CSV outputs.  It
does not fit, select, predict, or evaluate any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_tabular_features import acoustic_features, text_structure_features
from retrieve_true_missing_frozen_market_targets import PRICE_FIELDS, TARGET


FEATURE_FREEZE = Path(
    "reports/confirmation/earnings25_primary_feature_freeze.json"
)
MARKET_SUMMARY = Path(
    "reports/confirmation/earnings25_frozen_market/summary.json"
)
MODEL_INDEX = Path("data/processed/ahenk_dataset_index.csv")
MODEL_TABULAR = Path("data/processed/ahenk_tabular_features.parquet")
FEATURES_OUTPUT = Path(
    "data/confirmation/earnings25_forward_confirmation_features.parquet"
)
INDEX_OUTPUT = Path(
    "data/confirmation/earnings25_forward_confirmation_index.csv"
)
REPORT_OUTPUT = Path(
    "reports/confirmation/earnings25_forward_confirmation_feature_freeze.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_identifier(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def ensure_inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"{label} escapes project root: {relative}") from error
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {relative}")
    return path


def raw_response_set_sha256(
    root: Path, rows: list[dict[str, str]]
) -> tuple[str, int]:
    records: list[dict[str, str]] = []
    for row in rows:
        path = ensure_inside(root, row["market_raw_response_path"], "raw market response")
        observed = sha256(path)
        if observed != row["market_raw_response_sha256"]:
            raise SystemExit(f"{row['call_id']}: raw market-response hash changed.")
        records.append(
            {
                "call_id": row["call_id"],
                "path": row["market_raw_response_path"],
                "sha256": observed,
            }
        )
    encoded = json.dumps(
        sorted(records, key=lambda item: item["call_id"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(records)


def validate_target_dates(row: dict[str, str]) -> None:
    anchor = date.fromisoformat(row["call_date"])
    trading_dates = [
        date.fromisoformat(value)
        for value in row["target_trading_dates"].split("|")
    ]
    if (
        len(trading_dates) != 5
        or trading_dates != sorted(set(trading_dates))
        or any(day <= anchor for day in trading_dates)
    ):
        raise SystemExit(f"{row['call_id']}: target dates violate the frozen rule.")
    reference = date.fromisoformat(row["target_first_previous_close_date"])
    if reference > anchor:
        raise SystemExit(f"{row['call_id']}: target reference close is post-call.")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    paths = {
        "feature_freeze": root / FEATURE_FREEZE,
        "market_summary": root / MARKET_SUMMARY,
        "model_index": root / MODEL_INDEX,
        "model_tabular": root / MODEL_TABULAR,
        "features_output": root / FEATURES_OUTPUT,
        "index_output": root / INDEX_OUTPUT,
        "report_output": root / REPORT_OUTPUT,
    }
    missing = [
        str(path.relative_to(root))
        for key, path in paths.items()
        if not key.endswith("_output") and not path.is_file()
    ]
    if missing:
        raise SystemExit("Missing required inputs: " + ", ".join(missing))
    for key in ("features_output", "index_output", "report_output"):
        if paths[key].exists():
            raise SystemExit(
                f"Frozen output already exists; refusing overwrite: {paths[key].relative_to(root)}"
            )

    feature_freeze = read_json(paths["feature_freeze"])
    market_summary = read_json(paths["market_summary"])
    if feature_freeze.get("status") != "FROZEN_PRIMARY_FEATURES_BEFORE_MARKET_ACCESS":
        raise SystemExit("Primary text/acoustic features are not frozen.")
    if market_summary.get("status") != "PASS_FROZEN_MARKET_AND_TARGET_RETRIEVAL":
        raise SystemExit("Market/target retrieval has not passed.")
    if market_summary.get("error_companies") != 0 or market_summary.get("recovered_calls") != feature_freeze["cohort"]["calls"]:
        raise SystemExit("Market/target retrieval is incomplete.")
    if market_summary.get("data_use_guards", {}).get("cohort_membership_changed") is not False:
        raise SystemExit("Market retrieval reports a cohort-membership change.")
    if market_summary.get("data_use_guards", {}).get(
        "model_training_selection_prediction_or_evaluation_performed"
    ) is not False:
        raise SystemExit("Market retrieval reports premature model access.")

    market_info = market_summary["outputs"]["market_rows"]
    market_path = root / market_info["path"]
    if sha256(market_path) != market_info["sha256"]:
        raise SystemExit("Frozen market-row hash changed.")
    protocol_info = market_summary["outputs"]["protocol"]
    protocol_path = root / protocol_info["path"]
    if sha256(protocol_path) != protocol_info["sha256"]:
        raise SystemExit("Frozen market-retrieval protocol hash changed.")
    protocol = read_json(protocol_path)
    if protocol.get("feature_freeze_sha256") != sha256(paths["feature_freeze"]):
        raise SystemExit("Market protocol references a different feature freeze.")

    market_rows = read_csv(market_path)
    if len(market_rows) != int(feature_freeze["cohort"]["calls"]):
        raise SystemExit("Market rows differ from the frozen cohort size.")
    if len({row["call_id"] for row in market_rows}) != len(market_rows):
        raise SystemExit("Market rows contain duplicate call IDs.")
    expected_ids_hash = hashlib.sha256(
        "\n".join(sorted(row["call_id"] for row in market_rows)).encode("utf-8")
    ).hexdigest()
    if expected_ids_hash != feature_freeze["cohort"]["call_ids_sha256"]:
        raise SystemExit("Market-row membership differs from the frozen cohort.")

    targets = pd.to_numeric(
        pd.Series([row.get(TARGET, "") for row in market_rows]), errors="coerce"
    )
    if targets.isna().any() or not np.isfinite(targets).all() or (targets <= 0).any():
        raise SystemExit("Market rows contain invalid targets.")
    for row in market_rows:
        validate_target_dates(row)
        if int(float(row["price_60d_available_returns"])) != 60:
            raise SystemExit(f"{row['call_id']}: fewer than 60 frozen pre-call returns.")
        if any(column not in row for column in PRICE_FIELDS):
            raise SystemExit(f"{row['call_id']}: frozen price schema is incomplete.")
    raw_set_hash, raw_count = raw_response_set_sha256(root, market_rows)
    if raw_count != len(market_rows):
        raise SystemExit("Raw market-response coverage is incomplete.")

    original_index = pd.read_csv(paths["model_index"])
    original_tabular = pd.read_parquet(paths["model_tabular"])
    original_call_ids = set(original_index["call_id"].astype(str))
    call_overlap = original_call_ids.intersection(row["call_id"] for row in market_rows)
    if call_overlap:
        raise SystemExit(f"Forward confirmation contains {len(call_overlap)} original call IDs.")
    prior_identifier_columns = [
        column
        for column in ("ticker", "company_id", "yahoo_symbol")
        if column in original_index.columns
    ]
    prior_identifiers = {
        canonical_identifier(value)
        for column in prior_identifier_columns
        for value in original_index[column].dropna().astype(str)
        if canonical_identifier(value)
    }
    forward_tickers = {
        canonical_identifier(row["ticker"]) for row in market_rows
    }
    ticker_overlap = sorted(prior_identifiers.intersection(forward_tickers))
    if ticker_overlap:
        raise SystemExit(
            "Forward confirmation ticker strings overlap original model identifiers: "
            + ", ".join(ticker_overlap[:10])
        )

    nonnumeric_tokens: Counter[str] = Counter()
    feature_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for row in sorted(market_rows, key=lambda item: (item["call_date"], item["call_id"])):
        text_path = ensure_inside(root, row["text_path"], "frozen text")
        acoustic_path = ensure_inside(root, row["audio_feature_path"], "frozen acoustic features")
        output: dict[str, Any] = {
            "call_id": row["call_id"],
            "company_id": row["company_id"],
            "ticker": row["ticker"],
            "call_date": row["call_date"],
            "text_path": row["text_path"],
            TARGET: float(row[TARGET]),
        }
        output.update(acoustic_features(acoustic_path, nonnumeric_tokens))
        output.update(text_structure_features(text_path))
        for column in PRICE_FIELDS:
            value = float(row[column])
            if not math.isfinite(value):
                raise SystemExit(f"{row['call_id']}: nonfinite price feature {column}.")
            output[column] = value
        feature_rows.append(output)
        index_rows.append(
            {
                "call_id": row["call_id"],
                "company_id": row["company_id"],
                "ticker": row["ticker"],
                "exchange": row.get("exchange", ""),
                "call_date": row["call_date"],
                "text_path": row["text_path"],
                "audio_feature_path": row["audio_feature_path"],
                TARGET: row[TARGET],
                "confirmation_role": "earnings25_forward_current_SEC_mapping_disjoint_locked",
            }
        )

    features = pd.DataFrame(feature_rows).sort_values(
        ["call_date", "call_id"]
    ).reset_index(drop=True)
    metadata = {"call_id", "company_id", "ticker", "call_date", "text_path", TARGET}
    acoustic_columns = [
        column
        for column in features
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    text_columns = [
        column
        for column in features
        if column.startswith("text_") and column != "text_path"
    ]
    price_columns = [column for column in features if column.startswith("price_")]
    original_acoustic = [
        column
        for column in original_tabular
        if column.startswith("ac_") or column == "acoustic_segment_count"
    ]
    original_text = [column for column in original_tabular if column.startswith("text_")]
    original_price = [column for column in original_tabular if column.startswith("price_")]
    comparisons = {
        "acoustic": (set(acoustic_columns), set(original_acoustic)),
        "text_structure": (set(text_columns), set(original_text)),
        "price": (set(price_columns), set(original_price)),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            raise SystemExit(
                f"{label} schema differs from training data; "
                f"missing={sorted(expected - observed)[:10]}, "
                f"extra={sorted(observed - expected)[:10]}"
            )
    if len(acoustic_columns) != 175 or len(price_columns) != 16:
        raise SystemExit(
            f"Unexpected model interface: acoustic={len(acoustic_columns)}, price={len(price_columns)}"
        )

    numeric = features[[column for column in features if column not in metadata]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.shape[1] != len(acoustic_columns) + len(text_columns) + len(price_columns):
        raise SystemExit("Unexpected nonnumeric feature columns in the confirmation table.")
    maximum_missing = float(numeric.isna().mean().max())

    paths["features_output"].parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(paths["features_output"], index=False)
    write_csv(paths["index_output"], index_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_EARNINGS25_FORWARD_CONFIRMATION_FEATURES",
        "calls": len(features),
        "cik_blocks": int(features["company_id"].nunique()),
        "date_min": str(features["call_date"].min()),
        "date_max": str(features["call_date"].max()),
        "target_positive_and_finite": True,
        "feature_interface": {
            "acoustic_features": len(acoustic_columns),
            "text_structure_features": len(text_columns),
            "price_features": len(price_columns),
            "dense_text_representation": "fitted later on original training text only",
            "matches_original_training_schema": True,
        },
        "missingness": {
            "columns_with_missing_values": int((numeric.isna().mean() > 0).sum()),
            "maximum_feature_missing_fraction": maximum_missing,
            "imputation_performed": False,
            "imputation_rule": "fit later on original training rows only",
        },
        "disjointness": {
            "overlap_with_original_call_ids": len(call_overlap),
            "overlap_with_original_ticker_strings": len(ticker_overlap),
            "current_SEC_mapping_disjointness_inherited_from_frozen_cohort": True,
            "historical_security_master_claimed": False,
        },
        "market_audit": {
            "raw_response_files": raw_count,
            "raw_response_set_sha256": raw_set_hash,
            "minimum_available_60d_returns": min(
                int(float(row["price_60d_available_returns"])) for row in market_rows
            ),
            "cohort_membership_changed": False,
        },
        "inputs": {
            "primary_feature_freeze": {
                "path": FEATURE_FREEZE.as_posix(),
                "sha256": sha256(paths["feature_freeze"]),
            },
            "market_summary": {
                "path": MARKET_SUMMARY.as_posix(),
                "sha256": sha256(paths["market_summary"]),
            },
            "market_rows": {
                "path": market_info["path"],
                "sha256": sha256(market_path),
            },
            "market_protocol": {
                "path": protocol_info["path"],
                "sha256": sha256(protocol_path),
            },
            "original_model_index": {
                "path": MODEL_INDEX.as_posix(),
                "sha256": sha256(paths["model_index"]),
            },
            "original_model_tabular": {
                "path": MODEL_TABULAR.as_posix(),
                "sha256": sha256(paths["model_tabular"]),
            },
        },
        "outputs": {
            "features": {
                "path": FEATURES_OUTPUT.as_posix(),
                "sha256": sha256(paths["features_output"]),
            },
            "index": {
                "path": INDEX_OUTPUT.as_posix(),
                "sha256": sha256(paths["index_output"]),
            },
        },
        "top_nonnumeric_acoustic_tokens": dict(nonnumeric_tokens.most_common(20)),
        "data_use_guards": {
            "future_prices_used_as_inputs": False,
            "model_training_selection_prediction_or_evaluation_performed": False,
            "human_or_expert_labels_used": False,
        },
        "next_step": "Run the prespecified strict-missing forward confirmation exactly once.",
    }
    paths["report_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_output"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()