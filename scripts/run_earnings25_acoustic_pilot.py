#!/usr/bin/env python3
"""Run a one-call, target-blind Earnings25 Praat feature pilot.

The pilot uses the frozen cohort, the dataset's supplied segment boundaries,
FFmpeg 16-kHz mono decoding, and Praat Voice report measurements.  It writes
one text line and one 29-column feature row per valid supplied segment.

No market prices, targets, model fitting, or predictions are accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import parselmouth
import pyarrow.parquet as pq
from parselmouth.praat import call


FREEZE = Path("reports/confirmation/earnings25_forward_confirmation_freeze.json")
ACOUSTIC_FREEZE = Path(
    "reports/confirmation/earnings25_acoustic_extraction_freeze.json"
)
PARQUET_DIR = Path("data/raw/earnings25/full")
PILOT_ROOT = Path("data/forward_confirmation/acoustic_pilot")
REPORT_BASENAME = "earnings25_acoustic_pilot"
SHARDS = 24

PITCH_FLOOR = 75.0
PITCH_CEILING = 600.0
MAX_PERIOD_FACTOR = 1.3
MAX_AMPLITUDE_FACTOR = 1.6
SILENCE_THRESHOLD = 0.03
VOICING_THRESHOLD = 0.45
DECODE_SAMPLE_RATE = 16_000

FEATURE_COLUMNS = [
    "Mean pitch",
    "Standard deviation",
    "Minimum pitch",
    "Maximum pitch",
    "Mean intensity",
    "Minimum intensity",
    "Maximum intensity",
    "Number of pulses",
    "Number of periods",
    "Mean period",
    "Standard deviation of period",
    "Fraction of unvoiced",
    "Number of voice breaks",
    "Degree of voice breaks",
    "Jitter local",
    "Jitter local absolute",
    "Jitter rap",
    "Jitter ppq5",
    "Jitter ddp",
    "Shimmer local",
    "Shimmer local dB",
    "Shimmer apq3",
    "Shimmer apq5",
    "Shimmer apq11",
    "Shimmer dda",
    "Mean autocorrelation",
    "Mean NHR",
    "Mean HNR",
    "Audio Length",
]

REPORT_LABELS = {
    "Mean pitch": "Mean pitch",
    "Standard deviation": "Standard deviation",
    "Minimum pitch": "Minimum pitch",
    "Maximum pitch": "Maximum pitch",
    "Number of pulses": "Number of pulses",
    "Number of periods": "Number of periods",
    "Mean period": "Mean period",
    "Standard deviation of period": "Standard deviation of period",
    "Fraction of unvoiced": "Fraction of locally unvoiced frames",
    "Number of voice breaks": "Number of voice breaks",
    "Degree of voice breaks": "Degree of voice breaks",
    "Jitter local": "Jitter (local)",
    "Jitter local absolute": "Jitter (local, absolute)",
    "Jitter rap": "Jitter (rap)",
    "Jitter ppq5": "Jitter (ppq5)",
    "Jitter ddp": "Jitter (ddp)",
    "Shimmer local": "Shimmer (local)",
    "Shimmer local dB": "Shimmer (local, dB)",
    "Shimmer apq3": "Shimmer (apq3)",
    "Shimmer apq5": "Shimmer (apq5)",
    "Shimmer apq11": "Shimmer (apq11)",
    "Shimmer dda": "Shimmer (dda)",
    "Mean autocorrelation": "Mean autocorrelation",
    "Mean NHR": "Mean noise-to-harmonics ratio",
    "Mean HNR": "Mean harmonics-to-noise ratio",
}

NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--call-id", default="", help="Defaults to the first frozen primary call.")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Diagnostic cap; 0 processes all supplied segments.",
    )
    parser.add_argument(
        "--run-label",
        default="v3",
        help="Safe output suffix; the failed first pilot remains preserved.",
    )
    parser.add_argument(
        "--acoustic-freeze",
        default=ACOUSTIC_FREEZE.as_posix(),
        help="Frozen acoustic-extraction manifest.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def locate_call(paths: list[Path], call_id: str) -> tuple[Path, int]:
    for path in paths:
        identifiers = pq.read_table(path, columns=["id"])["id"].to_pylist()
        for index, value in enumerate(identifiers):
            if str(value) == call_id:
                return path, index
    raise SystemExit(f"Call ID not found in Earnings25 shards: {call_id}")


def read_call(path: Path, row_index: int, call_id: str) -> dict[str, Any]:
    table = pq.read_table(
        path,
        columns=["id", "audio", "segments", "duration_s", "sample_rate", "channels"],
    )
    if str(table["id"][row_index].as_py()) != call_id:
        raise RuntimeError("Parquet row identity changed during pilot extraction.")
    audio = table["audio"][row_index].as_py()
    segments = table["segments"][row_index].as_py()
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise RuntimeError("Pilot call lacks embedded audio bytes.")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Pilot call lacks supplied segments.")
    return {
        "audio_bytes": audio["bytes"],
        "audio_path": str(audio.get("path") or f"{call_id}.mp3"),
        "segments": segments,
        "duration_s": float(table["duration_s"][row_index].as_py()),
        "source_sample_rate": int(table["sample_rate"][row_index].as_py()),
        "source_channels": int(table["channels"][row_index].as_py()),
    }


def merge_segments(
    segments: list[dict[str, Any]],
    gap_tolerance: float,
    maximum_duration: float,
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for item in segments:
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or 0.0)
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if end <= start or not text:
            continue
        valid.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker_id": str(item.get("speaker_id") or ""),
            }
        )
    valid.sort(key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in valid:
        if current is None:
            current = dict(item)
            continue
        same_speaker = item["speaker_id"] == current["speaker_id"]
        close_gap = item["start"] - current["end"] <= gap_tolerance
        within_cap = item["end"] - current["start"] <= maximum_duration
        if same_speaker and close_gap and within_cap:
            current["end"] = item["end"]
            current["text"] = current["text"] + " " + item["text"]
        else:
            merged.append(current)
            current = dict(item)
    if current is not None:
        merged.append(current)
    return merged


def ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.splitlines()[0]


def decode_audio(audio_bytes: bytes, suffix: str, temporary: Path) -> Path:
    compressed = temporary / f"input{suffix}"
    wav = temporary / "decoded.wav"
    compressed.write_bytes(audio_bytes)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(compressed),
            "-ac",
            "1",
            "-ar",
            str(DECODE_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or not wav.is_file():
        raise RuntimeError(f"FFmpeg decoding failed: {completed.stderr[-2000:]}")
    return wav


def first_number_after_label(report: str, label: str) -> float:
    prefix = label + ":"
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        value_text = line[len(prefix):].strip()
        if "undefined" in value_text.lower():
            return math.nan
        match = NUMBER.search(value_text.replace("−", "-"))
        return float(match.group(0)) if match else math.nan
    return math.nan


def safe_praat_value(function: Any) -> float:
    try:
        value = float(function())
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def extract_features(
    segment: parselmouth.Sound,
    duration: float,
) -> tuple[dict[str, float], list[str]]:
    values = {column: math.nan for column in FEATURE_COLUMNS}
    values["Audio Length"] = duration
    errors: list[str] = []
    try:
        pitch = segment.to_pitch(
            pitch_floor=PITCH_FLOOR,
            pitch_ceiling=PITCH_CEILING,
        )
        values["Mean pitch"] = safe_praat_value(
            lambda: call(pitch, "Get mean", 0.0, 0.0, "Hertz")
        )
        values["Standard deviation"] = safe_praat_value(
            lambda: call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz")
        )
        values["Minimum pitch"] = safe_praat_value(
            lambda: call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic")
        )
        values["Maximum pitch"] = safe_praat_value(
            lambda: call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "Parabolic")
        )
        pulses = call([segment, pitch], "To PointProcess (cc)")
        report = call(
            [segment, pitch, pulses],
            "Voice report",
            0.0,
            0.0,
            PITCH_FLOOR,
            PITCH_CEILING,
            MAX_PERIOD_FACTOR,
            MAX_AMPLITUDE_FACTOR,
            SILENCE_THRESHOLD,
            VOICING_THRESHOLD,
        )
        for column, label in REPORT_LABELS.items():
            parsed = first_number_after_label(str(report), label)
            if math.isfinite(parsed):
                values[column] = parsed
    except Exception as exc:
        errors.append(f"voice_report:{type(exc).__name__}:{exc}")

    try:
        intensity = segment.to_intensity(minimum_pitch=PITCH_FLOOR)
        values["Mean intensity"] = safe_praat_value(
            lambda: call(intensity, "Get mean", 0.0, 0.0, "energy")
        )
        values["Minimum intensity"] = safe_praat_value(
            lambda: call(intensity, "Get minimum", 0.0, 0.0, "Parabolic")
        )
        values["Maximum intensity"] = safe_praat_value(
            lambda: call(intensity, "Get maximum", 0.0, 0.0, "Parabolic")
        )
    except Exception as exc:
        errors.append(f"intensity:{type(exc).__name__}:{exc}")
    return values, errors


def csv_value(value: float) -> str:
    return "--undefined--" if not math.isfinite(value) else format(value, ".12g")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    freeze_path = root / FREEZE
    acoustic_freeze_path = root / args.acoustic_freeze
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        raise SystemExit("--run-label may contain only letters, digits, underscore, and hyphen.")
    report_path = (
        root
        / "reports/confirmation"
        / f"{REPORT_BASENAME}_{args.run_label}.json"
    )
    if not freeze_path.is_file():
        raise SystemExit(f"Missing freeze manifest: {FREEZE}")
    if not acoustic_freeze_path.is_file():
        raise SystemExit(f"Missing acoustic extraction freeze: {args.acoustic_freeze}")
    if report_path.exists():
        raise SystemExit("Acoustic pilot report already exists; refusing to overwrite it.")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    acoustic_freeze = json.loads(acoustic_freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_ANY_PRICE_OR_TARGET_ACCESS":
        raise SystemExit("Unexpected forward-confirmation freeze status.")
    if acoustic_freeze.get("status") != "FROZEN_ACOUSTIC_EXTRACTION_BEFORE_BATCH_AND_TARGETS":
        raise SystemExit("Unexpected acoustic-extraction freeze status.")
    upstream_hash = acoustic_freeze.get("upstream", {}).get(
        "forward_confirmation_freeze", {}
    ).get("sha256")
    if upstream_hash != sha256(freeze_path):
        raise SystemExit("Acoustic freeze references a different forward-cohort freeze.")
    policy = acoustic_freeze.get("segmentation_policy", {})
    if policy.get("name") != "same_speaker_max_15s":
        raise SystemExit("Unexpected frozen segmentation policy.")
    gap_tolerance = float(policy["maximum_intersegment_gap_s"])
    maximum_duration = float(policy["maximum_merged_chunk_duration_s"])
    praat_settings = acoustic_freeze.get("praat_settings", {})
    expected_settings = {
        "pitch_floor_hz": PITCH_FLOOR,
        "pitch_ceiling_hz": PITCH_CEILING,
        "maximum_period_factor": MAX_PERIOD_FACTOR,
        "maximum_amplitude_factor": MAX_AMPLITUDE_FACTOR,
        "silence_threshold": SILENCE_THRESHOLD,
        "voicing_threshold": VOICING_THRESHOLD,
    }
    for key, expected in expected_settings.items():
        if float(praat_settings.get(key, math.nan)) != expected:
            raise SystemExit(f"Frozen Praat setting changed: {key}")
    primary_info = freeze["primary_cohort"]
    primary_path = root / primary_info["path"]
    if sha256(primary_path) != primary_info["sha256"]:
        raise SystemExit("Frozen primary cohort hash changed.")
    primary = read_csv(primary_path)
    allowed = {row["call_id"] for row in primary}
    call_id = args.call_id or primary[0]["call_id"]
    if call_id not in allowed:
        raise SystemExit("Pilot call must belong to the frozen primary cohort.")

    shard_paths = [
        root / PARQUET_DIR / f"test-{index:05d}-of-{SHARDS:05d}.parquet"
        for index in range(SHARDS)
    ]
    shard, row_index = locate_call(shard_paths, call_id)
    source = read_call(shard, row_index, call_id)
    raw_segments = source["segments"]
    segments = merge_segments(raw_segments, gap_tolerance, maximum_duration)
    if args.max_segments > 0:
        segments = segments[: args.max_segments]

    output_dir = root / PILOT_ROOT / call_id / args.run_label
    if output_dir.exists():
        raise SystemExit(f"Pilot output directory already exists: {output_dir.relative_to(root)}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    texts: list[str] = []
    invalid_segments = 0
    durations: list[float] = []
    extraction_errors: Counter[str] = Counter()
    extraction_error_examples: list[str] = []
    suffix = Path(source["audio_path"]).suffix or ".mp3"
    with tempfile.TemporaryDirectory(prefix="ahenk-earnings25-") as temp_name:
        wav = decode_audio(source["audio_bytes"], suffix, Path(temp_name))
        sound = parselmouth.Sound(str(wav))
        decoded_duration = float(sound.get_total_duration())
        for index, item in enumerate(segments):
            start = max(0.0, float(item.get("start") or 0.0))
            end = min(decoded_duration, float(item.get("end") or 0.0))
            if end <= start:
                invalid_segments += 1
                continue
            duration = end - start
            try:
                segment_sound = sound.extract_part(
                    from_time=start,
                    to_time=end,
                    preserve_times=False,
                )
                values, errors = extract_features(segment_sound, duration)
                for error in errors:
                    key = error.split(":", 2)[0] + ":" + error.split(":", 2)[1]
                    extraction_errors[key] += 1
                    if error not in extraction_error_examples and len(extraction_error_examples) < 20:
                        extraction_error_examples.append(error)
            except Exception as exc:
                values = {column: math.nan for column in FEATURE_COLUMNS}
                values["Audio Length"] = duration
                key = f"segment:{type(exc).__name__}"
                extraction_errors[key] += 1
                rendered = f"segment:{type(exc).__name__}:{exc}"
                if rendered not in extraction_error_examples and len(extraction_error_examples) < 20:
                    extraction_error_examples.append(rendered)
            rows.append(values)
            durations.append(duration)
            text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            texts.append(text or f"[EMPTY_SEGMENT_{index}]")

    if not rows:
        raise SystemExit("No valid supplied segments were extracted.")
    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{call_id}-building-", dir=output_dir.parent)
    )
    build_features = build_dir / "features.csv"
    build_text = build_dir / "text.txt"
    try:
        with build_features.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {column: csv_value(row[column]) for column in FEATURE_COLUMNS}
                )
        build_text.write_text("\n".join(texts) + "\n", encoding="utf-8")
        feature_hash = sha256(build_features)
        text_hash = sha256(build_text)
        os.replace(build_dir, output_dir)
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    features_path = output_dir / "features.csv"
    text_path = output_dir / "text.txt"

    missing_by_column = {
        column: sum(not math.isfinite(row[column]) for row in rows)
        for column in FEATURE_COLUMNS
    }
    total_cells = len(rows) * len(FEATURE_COLUMNS)
    missing_cells = sum(missing_by_column.values())
    complete_rows = sum(
        all(math.isfinite(row[column]) for column in FEATURE_COLUMNS) for row in rows
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PILOT_COMPLETE_REVIEW_BEFORE_FULL_EXTRACTION",
        "run_label": args.run_label,
        "supersedes_failed_pilot": (
            "reports/confirmation/earnings25_acoustic_pilot.json"
            if args.run_label == "v2"
            else ""
        ),
        "correction": (
            "Uses the frozen target-blind same-speaker 15-second merge policy; "
            "the v2 pilot established the corrected Praat calculation engine."
        ),
        "acoustic_extraction_freeze": {
            "path": acoustic_freeze_path.relative_to(root).as_posix(),
            "sha256": sha256(acoustic_freeze_path),
        },
        "call_id": call_id,
        "source": {
            "shard": shard.name,
            "row_index_in_shard": row_index,
            "source_duration_s": source["duration_s"],
            "decoded_duration_s": decoded_duration,
            "source_sample_rate": source["source_sample_rate"],
            "source_channels": source["source_channels"],
            "supplied_segments": len(raw_segments),
            "policy_segments_before_diagnostic_cap": len(
                merge_segments(raw_segments, gap_tolerance, maximum_duration)
            ),
            "processed_segments": len(rows),
            "invalid_segments": invalid_segments,
        },
        "frozen_extraction_settings": {
            "decoded_sample_rate": DECODE_SAMPLE_RATE,
            "decoded_channels": 1,
            "pitch_floor_hz": PITCH_FLOOR,
            "pitch_ceiling_hz": PITCH_CEILING,
            "maximum_period_factor": MAX_PERIOD_FACTOR,
            "maximum_amplitude_factor": MAX_AMPLITUDE_FACTOR,
            "silence_threshold": SILENCE_THRESHOLD,
            "voicing_threshold": VOICING_THRESHOLD,
            "source_segment_boundaries": "Earnings25 supplied start/end values",
            "segmentation_policy": policy["name"],
            "same_speaker_gap_tolerance_s": gap_tolerance,
            "maximum_merged_chunk_duration_s": maximum_duration,
            "undefined_policy": "retain row and write --undefined--",
        },
        "software": {
            "parselmouth": str(parselmouth.VERSION),
            "praat": str(parselmouth.PRAAT_VERSION),
            "ffmpeg": ffmpeg_version(),
        },
        "schema": FEATURE_COLUMNS,
        "quality": {
            "feature_rows": len(rows),
            "text_nonempty_lines": len(texts),
            "exact_text_feature_row_alignment": len(rows) == len(texts),
            "complete_feature_rows": complete_rows,
            "complete_feature_row_fraction": complete_rows / len(rows),
            "missing_cells": missing_cells,
            "total_feature_cells": total_cells,
            "missing_cell_fraction": missing_cells / total_cells,
            "missing_by_column": missing_by_column,
            "extraction_error_counts": dict(extraction_errors),
            "extraction_error_examples": extraction_error_examples,
            "segment_duration_s": {
                "min": min(durations),
                "median": median(durations),
                "max": max(durations),
            },
        },
        "outputs": {
            "features": {
                "path": features_path.relative_to(root).as_posix(),
                "sha256": feature_hash,
            },
            "text": {
                "path": text_path.relative_to(root).as_posix(),
                "sha256": text_hash,
            },
        },
        "data_use_guards": {
            "prices_retrieved": False,
            "targets_constructed": False,
            "model_fitted_or_evaluated": False,
        },
        "next_step": (
            "Inspect undefined rates and segment-duration compatibility, then "
            "freeze or reject these extraction settings before batch processing."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {features_path.relative_to(root)}")
    print(f"Wrote {text_path.relative_to(root)}")
    print(f"Wrote {report_path.relative_to(root)}")
    print(json.dumps(report["quality"], indent=2))
    print("No prices, targets, model fitting, or predictions were accessed.")


if __name__ == "__main__":
    main()