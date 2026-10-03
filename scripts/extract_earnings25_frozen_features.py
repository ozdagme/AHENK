#!/usr/bin/env python3
"""Extract frozen Earnings25 text and Praat features with safe resumption.

The script applies the already frozen same-speaker 15-second segmentation and
29-column Praat interface.  Each call is written atomically with source and
output hashes.  Completed calls are verified and skipped on rerun.

No market prices, targets, model fitting, or predictions are accessed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import parselmouth
import pyarrow.parquet as pq

from run_earnings25_acoustic_pilot import (
    DECODE_SAMPLE_RATE,
    FEATURE_COLUMNS,
    MAX_AMPLITUDE_FACTOR,
    MAX_PERIOD_FACTOR,
    PITCH_CEILING,
    PITCH_FLOOR,
    SILENCE_THRESHOLD,
    VOICING_THRESHOLD,
    csv_value,
    decode_audio,
    extract_features,
    ffmpeg_version,
    merge_segments,
)


FORWARD_FREEZE = Path(
    "reports/confirmation/earnings25_forward_confirmation_freeze.json"
)
ACOUSTIC_FREEZE = Path(
    "reports/confirmation/earnings25_acoustic_extraction_freeze.json"
)
PILOT_V3 = Path("reports/confirmation/earnings25_acoustic_pilot_v3.json")
PARQUET_DIR = Path("data/raw/earnings25/full")
SHARDS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--cohort", choices=("primary", "secondary"), default="primary")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Process at most N incomplete calls this run; 0 means all.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_settings(acoustic: dict[str, Any]) -> tuple[float, float]:
    policy = acoustic.get("segmentation_policy", {})
    if policy.get("name") != "same_speaker_max_15s":
        raise SystemExit("Unexpected frozen segmentation policy.")
    gap = float(policy["maximum_intersegment_gap_s"])
    maximum_duration = float(policy["maximum_merged_chunk_duration_s"])
    expected = {
        "pitch_floor_hz": PITCH_FLOOR,
        "pitch_ceiling_hz": PITCH_CEILING,
        "maximum_period_factor": MAX_PERIOD_FACTOR,
        "maximum_amplitude_factor": MAX_AMPLITUDE_FACTOR,
        "silence_threshold": SILENCE_THRESHOLD,
        "voicing_threshold": VOICING_THRESHOLD,
    }
    settings = acoustic.get("praat_settings", {})
    for key, value in expected.items():
        if float(settings.get(key, math.nan)) != value:
            raise SystemExit(f"Frozen Praat setting changed: {key}")
    if acoustic.get("feature_schema") != FEATURE_COLUMNS:
        raise SystemExit("Frozen acoustic feature schema changed.")
    return gap, maximum_duration


def completion_valid(
    directory: Path,
    call_id: str,
    acoustic_freeze_hash: str,
) -> bool:
    completion_path = directory / "completion.json"
    features_path = directory / "features.csv"
    text_path = directory / "text.txt"
    if not all(path.is_file() for path in (completion_path, features_path, text_path)):
        return False
    try:
        completion = read_json(completion_path)
    except Exception:
        return False
    return bool(
        completion.get("status") == "PASS"
        and completion.get("call_id") == call_id
        and completion.get("acoustic_freeze_sha256") == acoustic_freeze_hash
        and completion.get("features_sha256") == sha256(features_path)
        and completion.get("text_sha256") == sha256(text_path)
        and int(completion.get("feature_rows", 0)) > 0
        and completion.get("feature_rows") == completion.get("text_lines")
    )


def locate_calls(
    shard_paths: list[Path],
    call_ids: set[str],
) -> dict[str, tuple[Path, int]]:
    locations: dict[str, tuple[Path, int]] = {}
    for path in shard_paths:
        identifiers = pq.read_table(path, columns=["id"])["id"].to_pylist()
        for index, value in enumerate(identifiers):
            call_id = str(value)
            if call_id in call_ids:
                locations[call_id] = (path, index)
    missing = sorted(call_ids - set(locations))
    if missing:
        raise SystemExit(f"Frozen calls missing from Earnings25 shards: {missing[:10]}")
    return locations


def process_call(
    root: Path,
    output_root: Path,
    source_row: dict[str, Any],
    cohort_row: dict[str, str],
    gap_tolerance: float,
    maximum_duration: float,
    acoustic_freeze_hash: str,
    source_shard: str,
    row_index: int,
) -> dict[str, Any]:
    call_id = cohort_row["call_id"]
    audio = source_row["audio"]
    raw_segments = source_row["segments"]
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise RuntimeError(f"{call_id}: missing embedded audio bytes")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RuntimeError(f"{call_id}: missing supplied segments")
    segments = merge_segments(raw_segments, gap_tolerance, maximum_duration)
    if not segments:
        raise RuntimeError(f"{call_id}: frozen policy produced no segments")

    target = output_root / call_id
    if target.exists():
        raise RuntimeError(f"{call_id}: existing output is incomplete or invalid: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=f".{call_id}-building-", dir=output_root))
    rows: list[dict[str, float]] = []
    texts: list[str] = []
    error_counts: dict[str, int] = defaultdict(int)
    error_examples: list[str] = []
    audio_bytes = audio["bytes"]
    suffix = Path(str(audio.get("path") or f"{call_id}.mp3")).suffix or ".mp3"
    try:
        with tempfile.TemporaryDirectory(prefix="ahenk-earnings25-batch-") as temp_name:
            wav = decode_audio(audio_bytes, suffix, Path(temp_name))
            sound = parselmouth.Sound(str(wav))
            decoded_duration = float(sound.get_total_duration())
            for item in segments:
                start = max(0.0, float(item["start"]))
                end = min(decoded_duration, float(item["end"]))
                if end <= start:
                    continue
                duration = end - start
                segment_sound = sound.extract_part(
                    from_time=start,
                    to_time=end,
                    preserve_times=False,
                )
                values, errors = extract_features(segment_sound, duration)
                for error in errors:
                    parts = error.split(":", 2)
                    key = ":".join(parts[:2])
                    error_counts[key] += 1
                    if error not in error_examples and len(error_examples) < 10:
                        error_examples.append(error)
                rows.append(values)
                text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
                texts.append(text)
        if not rows or len(rows) != len(texts):
            raise RuntimeError(f"{call_id}: invalid text/feature row count")

        features_path = build / "features.csv"
        text_path = build / "text.txt"
        with features_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {column: csv_value(row[column]) for column in FEATURE_COLUMNS}
                )
        text_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
        missing_by_column = {
            column: sum(not math.isfinite(row[column]) for row in rows)
            for column in FEATURE_COLUMNS
        }
        missing_cells = sum(missing_by_column.values())
        total_cells = len(rows) * len(FEATURE_COLUMNS)
        completion = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "call_id": call_id,
            "company_id": cohort_row["company_id"],
            "ticker": cohort_row["ticker"],
            "call_date": cohort_row["call_date"],
            "source_shard": source_shard,
            "source_row_index": row_index,
            "source_audio_sha256": sha256_bytes(audio_bytes),
            "acoustic_freeze_sha256": acoustic_freeze_hash,
            "supplied_segments": len(raw_segments),
            "feature_rows": len(rows),
            "text_lines": len(texts),
            "missing_cells": missing_cells,
            "total_feature_cells": total_cells,
            "missing_cell_fraction": missing_cells / total_cells,
            "missing_by_column": missing_by_column,
            "extraction_error_counts": dict(error_counts),
            "extraction_error_examples": error_examples,
            "features_sha256": sha256(features_path),
            "text_sha256": sha256(text_path),
        }
        (build / "completion.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(build, target)
        return completion
    except Exception:
        shutil.rmtree(build, ignore_errors=True)
        raise


def finalize(
    root: Path,
    cohort_name: str,
    cohort_rows: list[dict[str, str]],
    output_root: Path,
    report_path: Path,
    index_path: Path,
    forward_path: Path,
    acoustic_path: Path,
    pilot_path: Path,
    acoustic_hash: str,
) -> None:
    completions: list[dict[str, Any]] = []
    index_rows: list[dict[str, str]] = []
    for row in cohort_rows:
        directory = output_root / row["call_id"]
        if not completion_valid(directory, row["call_id"], acoustic_hash):
            raise SystemExit(f"Cannot finalize; incomplete call: {row['call_id']}")
        completion = read_json(directory / "completion.json")
        completions.append(completion)
        index_rows.append(
            {
                **row,
                "text_path": (directory / "text.txt").relative_to(root).as_posix(),
                "audio_feature_path": (
                    directory / "features.csv"
                ).relative_to(root).as_posix(),
                "acoustic_segment_count": str(completion["feature_rows"]),
                "source_audio_sha256": completion["source_audio_sha256"],
            }
        )
    total_missing = sum(int(item["missing_cells"]) for item in completions)
    total_cells = sum(int(item["total_feature_cells"]) for item in completions)
    global_missing = total_missing / total_cells
    acceptance = read_json(acoustic_path)["batch_acceptance_rule"]
    if global_missing > float(acceptance["maximum_global_missing_cell_fraction"]):
        raise SystemExit(f"Global missing-cell fraction failed: {global_missing:.6f}")
    if index_path.exists() or report_path.exists():
        raise SystemExit("Final feature index/report already exists; refusing to overwrite it.")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(index_path, index_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_FROZEN_FEATURE_EXTRACTION",
        "cohort": cohort_name,
        "calls": len(cohort_rows),
        "cik_blocks": len({row["company_id"] for row in cohort_rows}),
        "feature_rows": sum(int(item["feature_rows"]) for item in completions),
        "missing_cells": total_missing,
        "total_feature_cells": total_cells,
        "global_missing_cell_fraction": global_missing,
        "calls_with_extraction_errors": sum(bool(item["extraction_error_counts"]) for item in completions),
        "source_and_protocol_hashes": {
            "forward_confirmation_freeze": sha256(forward_path),
            "acoustic_extraction_freeze": acoustic_hash,
            "accepted_v3_pilot": sha256(pilot_path),
        },
        "software": {
            "parselmouth": str(parselmouth.VERSION),
            "praat": str(parselmouth.PRAAT_VERSION),
            "ffmpeg": ffmpeg_version(),
        },
        "feature_index": {
            "path": index_path.relative_to(root).as_posix(),
            "sha256": sha256(index_path),
        },
        "data_use_guards": {
            "prices_retrieved": False,
            "targets_constructed": False,
            "model_fitted_or_evaluated": False,
        },
        "next_step": "Freeze extracted features, then retrieve pre-call and post-call market data once.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {index_path.relative_to(root)}")
    print(f"Wrote {report_path.relative_to(root)}")


def write_progress(
    path: Path,
    cohort_name: str,
    cohort_rows: list[dict[str, str]],
    output_root: Path,
    acoustic_hash: str,
) -> None:
    completions: list[dict[str, Any]] = []
    for row in cohort_rows:
        directory = output_root / row["call_id"]
        if completion_valid(directory, row["call_id"], acoustic_hash):
            completions.append(read_json(directory / "completion.json"))
    total_cells = sum(int(item["total_feature_cells"]) for item in completions)
    missing_cells = sum(int(item["missing_cells"]) for item in completions)
    report = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_READY_TO_FINALIZE" if len(completions) == len(cohort_rows) else "IN_PROGRESS",
        "cohort": cohort_name,
        "completed_calls": len(completions),
        "total_calls": len(cohort_rows),
        "remaining_calls": len(cohort_rows) - len(completions),
        "feature_rows": sum(int(item["feature_rows"]) for item in completions),
        "missing_cells": missing_cells,
        "total_feature_cells": total_cells,
        "global_missing_cell_fraction": (
            missing_cells / total_cells if total_cells else None
        ),
        "calls_with_extraction_errors": sum(
            bool(item["extraction_error_counts"]) for item in completions
        ),
        "data_use_guards": {
            "prices_retrieved": False,
            "targets_constructed": False,
            "model_fitted_or_evaluated": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.max_calls < 0:
        raise SystemExit("--max-calls cannot be negative.")
    root = Path(args.project_root).resolve()
    forward_path = root / FORWARD_FREEZE
    acoustic_path = root / ACOUSTIC_FREEZE
    pilot_path = root / PILOT_V3
    for path in (forward_path, acoustic_path, pilot_path):
        if not path.is_file():
            raise SystemExit(f"Missing frozen input: {path.relative_to(root)}")
    forward = read_json(forward_path)
    acoustic = read_json(acoustic_path)
    pilot = read_json(pilot_path)
    if forward.get("status") != "FROZEN_BEFORE_ANY_PRICE_OR_TARGET_ACCESS":
        raise SystemExit("Unexpected forward freeze status.")
    if acoustic.get("status") != "FROZEN_ACOUSTIC_EXTRACTION_BEFORE_BATCH_AND_TARGETS":
        raise SystemExit("Unexpected acoustic freeze status.")
    pilot_quality = pilot.get("quality", {})
    if (
        pilot.get("run_label") != "v3"
        or pilot_quality.get("exact_text_feature_row_alignment") is not True
        or float(pilot_quality.get("missing_cell_fraction", 1.0)) > 0.05
        or float(pilot_quality.get("complete_feature_row_fraction", 0.0)) < 0.80
        or bool(pilot_quality.get("extraction_error_counts"))
    ):
        raise SystemExit("Accepted v3 pilot is missing or failed its frozen rules.")
    gap_tolerance, maximum_duration = validate_settings(acoustic)
    acoustic_hash = sha256(acoustic_path)

    info_key = "primary_cohort" if args.cohort == "primary" else "secondary_cohort"
    cohort_info = forward[info_key]
    cohort_path = root / cohort_info["path"]
    if sha256(cohort_path) != cohort_info["sha256"]:
        raise SystemExit("Frozen cohort hash changed.")
    cohort_rows = read_csv(cohort_path)
    output_root = root / f"data/forward_confirmation/features_{args.cohort}"
    report_path = root / f"reports/confirmation/earnings25_{args.cohort}_feature_extraction.json"
    progress_path = root / f"reports/confirmation/earnings25_{args.cohort}_feature_extraction_progress.json"
    index_path = root / f"data/forward_confirmation/earnings25_{args.cohort}_feature_index.csv"
    if report_path.exists() or index_path.exists():
        raise SystemExit("Frozen feature extraction is already finalized.")

    completed = {
        row["call_id"]
        for row in cohort_rows
        if completion_valid(output_root / row["call_id"], row["call_id"], acoustic_hash)
    }
    invalid_existing = [
        row["call_id"]
        for row in cohort_rows
        if (output_root / row["call_id"]).exists() and row["call_id"] not in completed
    ]
    if invalid_existing:
        raise SystemExit(
            "Incomplete or invalid existing call outputs require inspection: "
            + ", ".join(invalid_existing[:10])
        )
    incomplete = [row for row in cohort_rows if row["call_id"] not in completed]
    selected = incomplete[: args.max_calls] if args.max_calls else incomplete
    print(
        f"Cohort={args.cohort}: completed={len(completed)}, "
        f"remaining={len(incomplete)}, processing={len(selected)}"
    )

    shard_paths = [
        root / PARQUET_DIR / f"test-{index:05d}-of-{SHARDS:05d}.parquet"
        for index in range(SHARDS)
    ]
    locations = locate_calls(shard_paths, {row["call_id"] for row in selected}) if selected else {}
    rows_by_id = {row["call_id"]: row for row in cohort_rows}
    grouped: dict[Path, list[tuple[str, int]]] = defaultdict(list)
    for call_id, (path, index) in locations.items():
        grouped[path].append((call_id, index))

    processed_now = 0
    for shard in shard_paths:
        items = sorted(grouped.get(shard, []), key=lambda item: item[1])
        if not items:
            continue
        table = pq.read_table(
            shard,
            columns=["id", "audio", "segments", "duration_s", "sample_rate", "channels"],
        )
        for call_id, row_index in items:
            if str(table["id"][row_index].as_py()) != call_id:
                raise RuntimeError("Parquet row identity changed during batch extraction.")
            source_row = {
                "audio": table["audio"][row_index].as_py(),
                "segments": table["segments"][row_index].as_py(),
            }
            completion = process_call(
                root,
                output_root,
                source_row,
                rows_by_id[call_id],
                gap_tolerance,
                maximum_duration,
                acoustic_hash,
                shard.name,
                row_index,
            )
            processed_now += 1
            print(
                f"[{len(completed) + processed_now}/{len(cohort_rows)}] {call_id}: "
                f"rows={completion['feature_rows']}, "
                f"missing={completion['missing_cell_fraction']:.4%}"
            )
        del table
        gc.collect()

    total_complete = sum(
        completion_valid(output_root / row["call_id"], row["call_id"], acoustic_hash)
        for row in cohort_rows
    )
    write_progress(
        progress_path,
        args.cohort,
        cohort_rows,
        output_root,
        acoustic_hash,
    )
    print(f"Wrote {progress_path.relative_to(root)}")
    if total_complete < len(cohort_rows):
        print(
            f"Progress saved: {total_complete}/{len(cohort_rows)} calls complete. "
            "Rerun the same command to continue."
        )
        return
    finalize(
        root,
        args.cohort,
        cohort_rows,
        output_root,
        report_path,
        index_path,
        forward_path,
        acoustic_path,
        pilot_path,
        acoustic_hash,
    )
    print("No prices, targets, model fitting, or predictions were accessed.")


if __name__ == "__main__":
    main()