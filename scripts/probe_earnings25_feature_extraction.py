#!/usr/bin/env python3
"""Probe the frozen Earnings25 text/audio interface without target access.

The probe verifies the frozen cohort hash, locates one frozen call inside the
pinned Parquet shards, records transcript/audio storage characteristics, and
reports locally available alignment and acoustic-extraction dependencies.
It does not write decoded audio, retrieve market data, construct targets,
train a model, or make predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


FREEZE = Path("reports/confirmation/earnings25_forward_confirmation_freeze.json")
PARQUET_DIR = Path("data/raw/earnings25/full")
OUTPUT = Path("reports/confirmation/earnings25_feature_extraction_probe.json")
SHARDS = 24


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audio_format(value: bytes) -> str:
    if value.startswith(b"RIFF") and value[8:12] == b"WAVE":
        return "wav"
    if value.startswith(b"ID3") or (
        len(value) >= 2 and value[0] == 0xFF and value[1] & 0xE0 == 0xE0
    ):
        return "mp3"
    if value.startswith(b"fLaC"):
        return "flac"
    if value.startswith(b"OggS"):
        return "ogg"
    if len(value) >= 12 and value[4:8] == b"ftyp":
        return "mp4_or_m4a"
    return "unknown"


def dependency_report() -> dict[str, Any]:
    modules = (
        "pyarrow",
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "parselmouth",
        "pydub",
        "aeneas",
        "soundfile",
        "librosa",
    )
    binaries = ("ffmpeg", "ffprobe", "praat")
    return {
        "python_modules": {
            name: importlib.util.find_spec(name) is not None for name in modules
        },
        "executables": {name: shutil.which(name) or "" for name in binaries},
    }


def locate_call(paths: list[Path], call_id: str) -> tuple[Path, int]:
    for path in paths:
        identifiers = pq.read_table(path, columns=["id"])["id"].to_pylist()
        for index, value in enumerate(identifiers):
            if str(value) == call_id:
                return path, index
    raise SystemExit(f"Frozen call ID not found in Earnings25 shards: {call_id}")


def inspect_row(path: Path, row_index: int, call_id: str) -> dict[str, Any]:
    text_table = pq.read_table(path, columns=["id", "text"])
    if str(text_table["id"][row_index].as_py()) != call_id:
        raise RuntimeError("Parquet row identity changed between scans.")
    text_value = text_table["text"][row_index].as_py()
    if isinstance(text_value, str):
        text_bytes = text_value.encode("utf-8")
        text_kind = "string"
        text_characters = len(text_value)
        text_lines = len(text_value.splitlines())
    else:
        serialized = json.dumps(text_value, ensure_ascii=False, sort_keys=True)
        text_bytes = serialized.encode("utf-8")
        text_kind = type(text_value).__name__
        text_characters = len(serialized)
        text_lines = 0

    audio_table = pq.read_table(path, columns=["id", "audio"])
    if str(audio_table["id"][row_index].as_py()) != call_id:
        raise RuntimeError("Parquet audio row identity changed between scans.")
    audio_value = audio_table["audio"][row_index].as_py()
    if not isinstance(audio_value, dict):
        raise RuntimeError(f"Unexpected audio value type: {type(audio_value).__name__}")
    audio_bytes = audio_value.get("bytes") or b""
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise RuntimeError("The probe row does not contain embedded audio bytes.")

    return {
        "call_id": call_id,
        "shard": path.name,
        "row_index_in_shard": row_index,
        "transcript": {
            "storage_type": text_kind,
            "characters": text_characters,
            "nonempty_lines": text_lines,
            "sha256": sha256_bytes(text_bytes),
        },
        "audio": {
            "storage_type": "embedded_bytes",
            "source_path_field": str(audio_value.get("path") or ""),
            "bytes": len(audio_bytes),
            "detected_container": audio_format(audio_bytes[:16]),
            "first_16_bytes_hex": audio_bytes[:16].hex(),
            "sha256": sha256_bytes(audio_bytes),
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    freeze_path = root / FREEZE
    output_path = root / OUTPUT
    if not freeze_path.is_file():
        raise SystemExit(f"Missing frozen protocol: {FREEZE}")
    if output_path.exists():
        raise SystemExit("Feature-extraction probe already exists; refusing to overwrite it.")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_ANY_PRICE_OR_TARGET_ACCESS":
        raise SystemExit("Unexpected Earnings25 freeze status.")
    primary_info = freeze.get("primary_cohort", {})
    primary_path = root / str(primary_info.get("path") or "")
    if not primary_path.is_file() or sha256(primary_path) != primary_info.get("sha256"):
        raise SystemExit("Frozen primary cohort is missing or its hash changed.")
    primary = read_csv(primary_path)
    if len(primary) != int(primary_info.get("calls", -1)):
        raise SystemExit("Frozen primary call count changed.")
    probe_call = primary[0]["call_id"]

    shard_paths = [
        root / PARQUET_DIR / f"test-{index:05d}-of-{SHARDS:05d}.parquet"
        for index in range(SHARDS)
    ]
    missing = [str(path.relative_to(root)) for path in shard_paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing Earnings25 shards: " + ", ".join(missing[:5]))
    shard, row_index = locate_call(shard_paths, probe_call)
    row_report = inspect_row(shard, row_index, probe_call)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FEATURE_EXTRACTION_DEPENDENCY_REVIEW_REQUIRED",
        "freeze_manifest": {
            "path": FREEZE.as_posix(),
            "sha256": sha256(freeze_path),
        },
        "primary_cohort_calls": len(primary),
        "parquet_shards": {
            "count": len(shard_paths),
            "total_bytes": sum(path.stat().st_size for path in shard_paths),
            "schema": str(pq.ParquetFile(shard_paths[0]).schema_arrow),
        },
        "probe_row": row_report,
        "dependencies": dependency_report(),
        "required_maec_interface": {
            "segment_level_alignment": True,
            "acoustic_streams": 29,
            "acoustic_summary_inputs": 175,
            "alignment_reference": "MAEC iterative forced alignment",
            "feature_family": "Praat voice-report pitch/intensity/pulse/period/jitter/shimmer/harmonicity",
        },
        "data_use_guards": {
            "audio_written_or_decoded": False,
            "prices_retrieved": False,
            "targets_constructed": False,
            "model_fitted_or_evaluated": False,
        },
        "next_step": (
            "Choose an extraction route after checking the reported audio "
            "container and locally available Praat/alignment dependencies."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path.relative_to(root)}")
    print(json.dumps(report["dependencies"], indent=2))
    print("No prices, targets, model fitting, or predictions were accessed.")


if __name__ == "__main__":
    main()