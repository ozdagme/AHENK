#!/usr/bin/env python3
"""Retrieve frozen Earnings25 market inputs and five-day targets once.

The primary feature set must already be frozen.  Daily Twelve Data responses
are retained verbatim, while pre-call features and the post-call target are
derived from the same response.  The script is resumable, stops cleanly on a
rate limit, and never fits, selects, predicts, or evaluates a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import run_true_missing_twelve_data_price_probe as twelve_data
from retrieve_true_missing_frozen_market_targets import (
    PRICE_FIELDS,
    TARGET,
    derive_call_values,
    parse_prices,
)


FEATURE_FREEZE = Path(
    "reports/confirmation/earnings25_primary_feature_freeze.json"
)
DEFAULT_OUTPUT = Path(
    "reports/confirmation/earnings25_frozen_market"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--max-symbols",
        type=int,
        help="Process at most N pending companies in this run; omit for all.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry companies whose preceding non-rate-limit request failed.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def canonical_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def artifact_set_sha256(root: Path, rows: list[dict[str, str]]) -> str:
    records: list[dict[str, str]] = []
    for row in rows:
        features_path = (root / row["audio_feature_path"]).resolve()
        text_path = (root / row["text_path"]).resolve()
        try:
            features_path.relative_to(root)
            text_path.relative_to(root)
        except ValueError as error:
            raise SystemExit(f"{row['call_id']}: feature artifact escapes project root.") from error
        completion_path = features_path.parent / "completion.json"
        if not all(path.is_file() for path in (features_path, text_path, completion_path)):
            raise SystemExit(f"{row['call_id']}: a frozen feature artifact is missing.")
        completion = read_json(completion_path)
        if (
            completion.get("features_sha256") != sha256(features_path)
            or completion.get("text_sha256") != sha256(text_path)
            or completion.get("source_audio_sha256") != row.get("source_audio_sha256")
        ):
            raise SystemExit(f"{row['call_id']}: frozen feature artifact changed.")
        records.append(
            {
                "call_id": row["call_id"],
                "completion_sha256": sha256(completion_path),
                "features_sha256": sha256(features_path),
                "text_sha256": sha256(text_path),
                "source_audio_sha256": str(completion["source_audio_sha256"]),
            }
        )
    encoded = json.dumps(
        sorted(records, key=lambda row: row["call_id"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_parameters(row: dict[str, str], recipe: dict[str, Any]) -> dict[str, Any]:
    call_date = date.fromisoformat(row["call_date"])
    return {
        "symbol": row["ticker"],
        "interval": recipe["interval"],
        "start_date": (
            call_date - timedelta(days=int(recipe["pre_call_calendar_days"]))
        ).isoformat(),
        "end_date": (
            call_date + timedelta(days=int(recipe["post_call_calendar_days"]))
        ).isoformat(),
        "adjust": recipe["adjust"],
        "order": recipe["order"],
    }


def formatted(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: format(value, ".12g") if isinstance(value, float) else value
        for key, value in values.items()
    }


def validate_derived(row: dict[str, Any], call_date: date) -> None:
    target = float(row[TARGET])
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target is not positive and finite")
    dates = [date.fromisoformat(value) for value in str(row["target_trading_dates"]).split("|")]
    if len(dates) != 5 or dates != sorted(set(dates)) or any(day <= call_date for day in dates):
        raise ValueError("target trading dates violate the frozen date-only rule")
    previous = date.fromisoformat(str(row["target_first_previous_close_date"]))
    if previous > call_date:
        raise ValueError("first target return uses a post-call reference close")


def main() -> None:
    args = parse_args()
    if args.sleep < 0 or args.timeout <= 0:
        raise SystemExit("--sleep must be nonnegative and --timeout must be positive.")
    if args.max_symbols is not None and args.max_symbols < 0:
        raise SystemExit("--max-symbols cannot be negative.")
    root = Path(args.project_root).resolve()
    freeze_path = root / FEATURE_FREEZE
    if not freeze_path.is_file():
        raise SystemExit(f"Missing feature freeze: {FEATURE_FREEZE}")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN_PRIMARY_FEATURES_BEFORE_MARKET_ACCESS":
        raise SystemExit("Unexpected primary-feature freeze status.")
    if freeze.get("authorization", {}).get("market_and_target_retrieval_for_frozen_primary_calls") is not True:
        raise SystemExit("Feature freeze does not authorize market/target retrieval.")

    index_path = root / freeze["features"]["index_path"]
    if sha256(index_path) != freeze["features"]["index_sha256"]:
        raise SystemExit("Frozen feature-index hash changed.")
    rows, index_fields = read_csv(index_path)
    if len(rows) != int(freeze["cohort"]["calls"]):
        raise SystemExit("Frozen feature-index row count changed.")
    if len({row["call_id"] for row in rows}) != len(rows):
        raise SystemExit("Feature index contains duplicate call IDs.")
    if artifact_set_sha256(root, rows) != freeze["features"]["artifact_set_sha256"]:
        raise SystemExit("Frozen per-call feature artifact set changed.")

    api_key = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "TWELVE_DATA_API_KEY is required. Export it in the shell without writing it to a file."
        )

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "frozen_retrieval_protocol.json"
    progress_path = output_dir / "progress.json"
    market_path = output_dir / "earnings25_primary_market_rows.csv"
    summary_path = output_dir / "summary.json"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.is_file() and read_json(summary_path).get("status") == "PASS_FROZEN_MARKET_AND_TARGET_RETRIEVAL":
        raise SystemExit("Frozen market/target retrieval already passed; refusing another provider pass.")

    recipe = freeze["frozen_market_and_target_recipe"]
    immutable_protocol = {
        "stage": "LOCKED_EARNINGS25_MARKET_AND_TARGET_RETRIEVAL",
        "feature_freeze_path": FEATURE_FREEZE.as_posix(),
        "feature_freeze_sha256": sha256(freeze_path),
        "feature_index_sha256": sha256(index_path),
        "calls": len(rows),
        "companies": len({row["company_id"] for row in rows}),
        "recipe": recipe,
        "cohort_membership_changes_allowed": False,
        "model_training_selection_prediction_or_evaluation_allowed": False,
    }
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        for key, expected in immutable_protocol.items():
            if key != "created_at_utc" and protocol.get(key) != expected:
                raise SystemExit(f"Frozen retrieval protocol changed: {key}")
    else:
        write_json(
            protocol_path,
            {"created_at_utc": now_utc(), **immutable_protocol},
        )

    progress = read_json(progress_path) if progress_path.exists() else {"companies": {}}
    states: dict[str, Any] = progress.setdefault("companies", {})
    existing, _ = read_csv(market_path) if market_path.exists() else ([], [])
    market_rows = {row["call_id"]: row for row in existing}
    by_company: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_company.setdefault(row["company_id"], []).append(row)
    if any(len(items) != 1 for items in by_company.values()):
        raise SystemExit("The frozen primary cohort no longer has one call per CIK block.")

    pending: list[str] = []
    for company_id in sorted(by_company):
        state = states.get(company_id)
        if state is None or (args.retry_errors and state.get("status") != "complete"):
            pending.append(company_id)
    if args.max_symbols is not None:
        pending = pending[: args.max_symbols]

    fields = [
        *index_fields,
        TARGET,
        "target_trading_dates",
        "target_first_previous_close_date",
        *PRICE_FIELDS,
        "market_provider",
        "market_returned_symbol",
        "market_returned_exchange",
        "market_raw_response_path",
        "market_raw_response_sha256",
        "market_collection_locked",
        "used_for_model_selection",
    ]
    stopped_error = ""
    processed_this_run = 0
    for position, company_id in enumerate(pending, start=1):
        source = by_company[company_id][0]
        params = request_parameters(source, recipe)
        raw_path = raw_dir / f"{safe_component(company_id)}_{safe_component(source['ticker'])}.json"
        prior_state = states.get(company_id)
        if raw_path.is_file() and prior_state is None:
            payload, error = read_json(raw_path), ""
            retrieval_source = "retained_raw_response"
        else:
            payload, error = twelve_data.api_get(params, api_key, args.timeout)
            retrieval_source = "provider_request"
        if error.startswith("rate_limited"):
            stopped_error = error
            break

        raw_hash = ""
        if payload is not None:
            write_json(raw_path, payload)
            raw_hash = sha256(raw_path)
        prices, parse_error = parse_prices(payload)
        failure = error or parse_error
        meta = (payload or {}).get("meta", {})
        returned_symbol = str(meta.get("symbol", ""))
        if not failure and returned_symbol and canonical_symbol(returned_symbol) != canonical_symbol(source["ticker"]):
            failure = "returned_meta_symbol_mismatch"

        if not failure:
            try:
                values, target_error = derive_call_values(
                    prices, date.fromisoformat(source["call_date"])
                )
                if target_error:
                    failure = target_error
                else:
                    assert values is not None
                    validate_derived(values, date.fromisoformat(source["call_date"]))
                    market_rows[source["call_id"]] = {
                        **source,
                        **formatted(values),
                        "market_provider": "Twelve Data daily close adjusted for splits",
                        "market_returned_symbol": returned_symbol,
                        "market_returned_exchange": str(meta.get("exchange", "")),
                        "market_raw_response_path": raw_path.relative_to(root).as_posix(),
                        "market_raw_response_sha256": raw_hash,
                        "market_collection_locked": True,
                        "used_for_model_selection": False,
                    }
            except (AssertionError, KeyError, TypeError, ValueError) as exc:
                failure = f"derived_value_error:{type(exc).__name__}:{exc}"

        states[company_id] = {
            "status": "error" if failure else "complete",
            "call_id": source["call_id"],
            "ticker": source["ticker"],
            "expected_exchange": source.get("exchange", ""),
            "request_start_date": params["start_date"],
            "request_end_date": params["end_date"],
            "raw_response_path": raw_path.relative_to(root).as_posix() if raw_hash else "",
            "raw_response_sha256": raw_hash,
            "retrieval_source": retrieval_source,
            "failure": failure,
            "attempted_at_utc": now_utc(),
        }
        processed_this_run += 1
        if market_rows:
            write_csv(
                market_path,
                sorted(market_rows.values(), key=lambda row: (row["call_date"], row["call_id"])),
                fields,
            )
        write_json(
            progress_path,
            {
                "updated_at_utc": now_utc(),
                "protocol_sha256": sha256(protocol_path),
                "companies": states,
                "stopped_error": stopped_error,
            },
        )
        print(
            f"[{position}/{len(pending)}] {source['ticker']} ({source['call_id']}): "
            f"{'error: ' + failure if failure else 'complete'}"
        )
        if position < len(pending):
            time.sleep(args.sleep)

    complete = sum(state.get("status") == "complete" for state in states.values())
    errors = [state for state in states.values() if state.get("status") == "error"]
    unattempted = len(by_company) - len(states)
    if complete == len(by_company) and len(market_rows) == len(rows):
        status = "PASS_FROZEN_MARKET_AND_TARGET_RETRIEVAL"
    elif stopped_error:
        status = "PAUSED_RATE_LIMIT"
    elif errors and unattempted == 0:
        status = "RETRY_REQUIRED"
    else:
        status = "IN_PROGRESS"
    failures: Counter[str] = Counter(state.get("failure", "") for state in errors)
    history_counts = [
        int(row["price_history_days"])
        for row in market_rows.values()
        if row.get("price_history_days", "") != ""
    ]
    available_60 = [
        int(row["price_60d_available_returns"])
        for row in market_rows.values()
        if row.get("price_60d_available_returns", "") != ""
    ]
    summary = {
        "created_at_utc": now_utc(),
        "status": status,
        "frozen_calls": len(rows),
        "frozen_cik_blocks": len(by_company),
        "attempted_companies": len(states),
        "complete_companies": complete,
        "error_companies": len(errors),
        "unattempted_companies": unattempted,
        "recovered_calls": len(market_rows),
        "processed_this_run": processed_this_run,
        "failure_counts": dict(failures),
        "stopped_error": stopped_error,
        "minimum_pre_call_price_history_days": min(history_counts) if history_counts else None,
        "minimum_available_60d_returns": min(available_60) if available_60 else None,
        "targets_positive_and_finite": bool(market_rows) and all(
            math.isfinite(float(row[TARGET])) and float(row[TARGET]) > 0
            for row in market_rows.values()
        ),
        "outputs": {
            "market_rows": {
                "path": market_path.relative_to(root).as_posix(),
                "sha256": sha256(market_path) if market_path.exists() else "",
            },
            "raw_response_directory": raw_dir.relative_to(root).as_posix(),
            "protocol": {
                "path": protocol_path.relative_to(root).as_posix(),
                "sha256": sha256(protocol_path),
            },
        },
        "data_use_guards": {
            "cohort_membership_changed": False,
            "model_training_selection_prediction_or_evaluation_performed": False,
        },
        "next_step": (
            "Freeze the complete market rows and construct the model-ready confirmation table."
            if status == "PASS_FROZEN_MARKET_AND_TARGET_RETRIEVAL"
            else "Rerun the same command; add --retry-errors when all remaining rows are recorded errors."
        ),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()