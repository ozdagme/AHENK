#!/usr/bin/env python3
"""Freeze the completed target-blind Earnings25 primary feature artifacts.

This is the final gate before any pre-call market features or post-call
targets are retrieved.  It rechecks the frozen cohort, every per-call
completion record, text/acoustic alignment, and all recorded hashes.  The
resulting manifest also fixes the market-data and target-construction recipe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORWARD_FREEZE = Path(
    "reports/confirmation/earnings25_forward_confirmation_freeze.json"
)
ACOUSTIC_FREEZE = Path(
    "reports/confirmation/earnings25_acoustic_extraction_freeze.json"
)
PILOT_V3 = Path("reports/confirmation/earnings25_acoustic_pilot_v3.json")
EXTRACTION_REPORT = Path(
    "reports/confirmation/earnings25_primary_feature_extraction.json"
)
FEATURE_INDEX = Path(
    "data/forward_confirmation/earnings25_primary_feature_index.csv"
)
OUTPUT = Path(
    "reports/confirmation/earnings25_primary_feature_freeze.json"
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


def safe_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"Artifact escapes project root: {relative}") from error
    if not path.is_file():
        raise SystemExit(f"Missing frozen feature artifact: {relative}")
    return path


def canonical_artifact_digest(records: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        sorted(records, key=lambda row: row["call_id"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    paths = {
        "forward": root / FORWARD_FREEZE,
        "acoustic": root / ACOUSTIC_FREEZE,
        "pilot": root / PILOT_V3,
        "report": root / EXTRACTION_REPORT,
        "index": root / FEATURE_INDEX,
        "output": root / OUTPUT,
    }
    missing = [
        str(path.relative_to(root))
        for key, path in paths.items()
        if key != "output" and not path.is_file()
    ]
    if missing:
        raise SystemExit("Missing required inputs: " + ", ".join(missing))
    if paths["output"].exists():
        raise SystemExit("Primary feature freeze already exists; refusing to overwrite it.")

    forward = read_json(paths["forward"])
    acoustic = read_json(paths["acoustic"])
    pilot = read_json(paths["pilot"])
    report = read_json(paths["report"])
    if forward.get("status") != "FROZEN_BEFORE_ANY_PRICE_OR_TARGET_ACCESS":
        raise SystemExit("Unexpected forward-cohort freeze status.")
    if acoustic.get("status") != "FROZEN_ACOUSTIC_EXTRACTION_BEFORE_BATCH_AND_TARGETS":
        raise SystemExit("Unexpected acoustic-extraction freeze status.")
    if report.get("status") != "PASS_FROZEN_FEATURE_EXTRACTION":
        raise SystemExit("Primary feature extraction has not passed.")
    if report.get("cohort") != "primary":
        raise SystemExit("Feature extraction report is not for the primary cohort.")
    if pilot.get("run_label") != "v3":
        raise SystemExit("The accepted v3 extraction pilot is missing.")

    source_hashes = report.get("source_and_protocol_hashes", {})
    expected_hashes = {
        "forward_confirmation_freeze": sha256(paths["forward"]),
        "acoustic_extraction_freeze": sha256(paths["acoustic"]),
        "accepted_v3_pilot": sha256(paths["pilot"]),
    }
    if source_hashes != expected_hashes:
        raise SystemExit("Feature extraction references different frozen inputs.")
    if report.get("feature_index", {}).get("sha256") != sha256(paths["index"]):
        raise SystemExit("Feature-index hash differs from the extraction report.")
    for key, value in report.get("data_use_guards", {}).items():
        if key in {"prices_retrieved", "targets_constructed", "model_fitted_or_evaluated"} and value is not False:
            raise SystemExit(f"Feature-extraction data-use guard failed: {key}")

    cohort_info = forward["primary_cohort"]
    cohort_path = root / cohort_info["path"]
    if sha256(cohort_path) != cohort_info["sha256"]:
        raise SystemExit("Frozen primary-cohort hash changed.")
    cohort = read_csv(cohort_path)
    index = read_csv(paths["index"])
    if len(cohort) != int(cohort_info["calls"]):
        raise SystemExit("Frozen primary-cohort row count changed.")
    if len(index) != len(cohort) or len(index) != int(report["calls"]):
        raise SystemExit("Feature index does not cover every frozen primary call exactly once.")
    if len({row["call_id"] for row in index}) != len(index):
        raise SystemExit("Duplicate call_id values found in the feature index.")

    cohort_by_id = {row["call_id"]: row for row in cohort}
    if set(cohort_by_id) != {row["call_id"] for row in index}:
        raise SystemExit("Feature-index membership differs from the frozen primary cohort.")

    expected_schema = acoustic.get("feature_schema") or []
    if len(expected_schema) != 29:
        raise SystemExit("Frozen acoustic schema does not contain 29 columns.")
    acoustic_hash = sha256(paths["acoustic"])
    artifacts: list[dict[str, str]] = []
    total_rows = 0
    total_missing = 0
    total_cells = 0
    calls_with_errors = 0
    for row in sorted(index, key=lambda item: item["call_id"]):
        call_id = row["call_id"]
        source = cohort_by_id[call_id]
        for field in ("company_id", "ticker", "call_date"):
            if row.get(field) != source.get(field):
                raise SystemExit(f"{call_id}: feature-index {field} changed.")
        features_path = safe_artifact(root, row["audio_feature_path"])
        text_path = safe_artifact(root, row["text_path"])
        completion_path = features_path.parent / "completion.json"
        if not completion_path.is_file():
            raise SystemExit(f"{call_id}: completion.json is missing.")
        completion = read_json(completion_path)
        if completion.get("status") != "PASS" or completion.get("call_id") != call_id:
            raise SystemExit(f"{call_id}: invalid completion record.")
        if completion.get("acoustic_freeze_sha256") != acoustic_hash:
            raise SystemExit(f"{call_id}: completion uses a different acoustic freeze.")
        if completion.get("features_sha256") != sha256(features_path):
            raise SystemExit(f"{call_id}: acoustic feature hash mismatch.")
        if completion.get("text_sha256") != sha256(text_path):
            raise SystemExit(f"{call_id}: text hash mismatch.")
        with features_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_schema:
                raise SystemExit(f"{call_id}: acoustic schema changed.")
            observed_rows = sum(1 for _ in reader)
        observed_text_lines = len(text_path.read_text(encoding="utf-8").splitlines())
        feature_rows = int(completion["feature_rows"])
        if not (
            observed_rows
            == observed_text_lines
            == feature_rows
            == int(row["acoustic_segment_count"])
            == int(completion["text_lines"])
        ):
            raise SystemExit(f"{call_id}: text/acoustic row alignment changed.")
        total_rows += feature_rows
        total_missing += int(completion["missing_cells"])
        total_cells += int(completion["total_feature_cells"])
        calls_with_errors += bool(completion.get("extraction_error_counts"))
        artifacts.append(
            {
                "call_id": call_id,
                "completion_sha256": sha256(completion_path),
                "features_sha256": sha256(features_path),
                "text_sha256": sha256(text_path),
                "source_audio_sha256": str(completion["source_audio_sha256"]),
            }
        )

    if total_rows != int(report["feature_rows"]):
        raise SystemExit("Feature-row total differs from the extraction report.")
    if total_missing != int(report["missing_cells"]) or total_cells != int(report["total_feature_cells"]):
        raise SystemExit("Missing-cell totals differ from the extraction report.")
    if calls_with_errors != int(report["calls_with_extraction_errors"]):
        raise SystemExit("Extraction-error call count differs from the report.")
    missing_fraction = total_missing / total_cells
    maximum_missing = float(acoustic["batch_acceptance_rule"]["maximum_global_missing_cell_fraction"])
    if missing_fraction > maximum_missing:
        raise SystemExit("Global missing-cell fraction exceeds the frozen acceptance rule.")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_PRIMARY_FEATURES_BEFORE_MARKET_ACCESS",
        "cohort": {
            "path": cohort_info["path"],
            "sha256": cohort_info["sha256"],
            "calls": len(cohort),
            "cik_blocks": len({row["company_id"] for row in cohort}),
            "call_ids_sha256": hashlib.sha256(
                "\n".join(sorted(cohort_by_id)).encode("utf-8")
            ).hexdigest(),
        },
        "features": {
            "index_path": FEATURE_INDEX.as_posix(),
            "index_sha256": sha256(paths["index"]),
            "extraction_report_path": EXTRACTION_REPORT.as_posix(),
            "extraction_report_sha256": sha256(paths["report"]),
            "artifact_set_sha256": canonical_artifact_digest(artifacts),
            "calls": len(artifacts),
            "feature_rows": total_rows,
            "missing_cells": total_missing,
            "total_feature_cells": total_cells,
            "global_missing_cell_fraction": missing_fraction,
            "calls_with_extraction_errors": calls_with_errors,
            "acceptance_rule_passed": True,
        },
        "upstream": {
            "forward_confirmation_freeze_sha256": expected_hashes["forward_confirmation_freeze"],
            "acoustic_extraction_freeze_sha256": expected_hashes["acoustic_extraction_freeze"],
            "accepted_v3_pilot_sha256": expected_hashes["accepted_v3_pilot"],
        },
        "frozen_market_and_target_recipe": {
            "provider": "Twelve Data time_series",
            "identifier": "frozen SEC ticker; symbol-only request with returned-symbol validation",
            "expected_exchange_retained_for_audit": True,
            "interval": "1day",
            "adjust": "splits",
            "order": "asc",
            "pre_call_calendar_days": 150,
            "post_call_calendar_days": 20,
            "pre_call_price_windows_trading_days": [5, 20, 60],
            "pre_call_cutoff": "returns dated strictly before call_date",
            "target": "sum of squared log returns for first five trading dates strictly after call_date",
            "first_target_return_reference": "previous available trading close",
            "timezone_rule": "date-only; no intraday timing imputed",
            "one_raw_daily_series_per_frozen_company": True,
            "raw_provider_responses_retained": True,
            "cohort_membership_changes_allowed": False,
        },
        "data_use_guards": {
            "prices_retrieved_before_this_freeze": False,
            "targets_constructed_before_this_freeze": False,
            "model_fitted_or_evaluated_on_earnings25": False,
        },
        "authorization": {
            "market_and_target_retrieval_for_frozen_primary_calls": True,
            "model_training_selection_or_evaluation": False,
        },
        "next_step": "Retrieve and retain the frozen daily market series once; derive price inputs and targets without model access.",
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()