#!/usr/bin/env python3
"""Compare target-blind Earnings25 segment policies with the MAEC interface.

The frozen model was trained from one acoustic row per nonempty MAEC text
line.  Earnings25 supplies much finer timestamped segments.  This audit
compares raw segments and deterministic same-speaker merge policies against
the MAEC training distribution before any market prices, targets, model
fitting, or predictions are accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pyarrow.parquet as pq


FREEZE = Path("reports/confirmation/earnings25_forward_confirmation_freeze.json")
MODEL_INDEX = Path("data/processed/ahenk_dataset_index.csv")
PARQUET_DIR = Path("data/raw/earnings25/full")
OUTPUT = Path("reports/confirmation/earnings25_segmentation_compatibility.json")
SHARDS = 24
GAP_TOLERANCE_S = 2.0
POLICIES: dict[str, float | None] = {
    "raw_supplied_segments": 0.0,
    "same_speaker_turn_uncapped": None,
    "same_speaker_max_15s": 15.0,
    "same_speaker_max_30s": 30.0,
    "same_speaker_max_60s": 60.0,
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", text)


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def load_maec_training(root: Path) -> dict[str, Any]:
    index_path = root / MODEL_INDEX
    if not index_path.is_file():
        raise SystemExit(f"Missing model index: {MODEL_INDEX}")
    rows = read_csv(index_path)
    split_key = "temporal_split" if "temporal_split" in rows[0] else "split"
    training = [row for row in rows if row.get(split_key) == "train"]
    if len(training) != 1725:
        raise SystemExit(f"Expected 1,725 MAEC training calls, found {len(training)}.")
    line_counts: list[float] = []
    token_counts: list[float] = []
    missing_paths = 0
    for row in training:
        text_path = root / row["text_path"]
        if not text_path.is_file():
            missing_paths += 1
            continue
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        line_counts.append(float(len(lines)))
        token_counts.extend(float(len(tokens(line))) for line in lines)
    if missing_paths:
        raise SystemExit(f"Missing {missing_paths} MAEC training transcript paths.")
    return {
        "calls": len(training),
        "rows_per_call": summary(line_counts),
        "tokens_per_row": summary(token_counts),
        "source_index": {
            "path": MODEL_INDEX.as_posix(),
            "sha256": sha256(index_path),
        },
    }


def valid_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in segments:
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or 0.0)
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if end <= start or not text:
            continue
        output.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker_id": str(item.get("speaker_id") or ""),
            }
        )
    return sorted(output, key=lambda item: (item["start"], item["end"]))


def merge_segments(
    segments: list[dict[str, Any]],
    max_duration: float | None,
) -> list[dict[str, Any]]:
    if max_duration == 0.0:
        return segments
    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in segments:
        if current is None:
            current = dict(item)
            continue
        same_speaker = item["speaker_id"] == current["speaker_id"]
        close_gap = item["start"] - current["end"] <= GAP_TOLERANCE_S
        candidate_duration = item["end"] - current["start"]
        within_cap = max_duration is None or candidate_duration <= max_duration
        if same_speaker and close_gap and within_cap:
            current["end"] = item["end"]
            current["text"] = current["text"] + " " + item["text"]
        else:
            merged.append(current)
            current = dict(item)
    if current is not None:
        merged.append(current)
    return merged


def load_earnings_segments(
    root: Path,
    allowed_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for index in range(SHARDS):
        path = root / PARQUET_DIR / f"test-{index:05d}-of-{SHARDS:05d}.parquet"
        table = pq.read_table(path, columns=["id", "segments"])
        for call_id, segments in zip(table["id"].to_pylist(), table["segments"].to_pylist()):
            key = str(call_id)
            if key in allowed_ids:
                found[key] = valid_segments(segments or [])
    missing = sorted(allowed_ids - set(found))
    if missing:
        raise SystemExit(f"Frozen calls missing from Earnings25 shards: {missing[:10]}")
    return found


def policy_report(
    calls: dict[str, list[dict[str, Any]]],
    max_duration: float | None,
) -> dict[str, Any]:
    rows_per_call: list[float] = []
    token_counts: list[float] = []
    durations: list[float] = []
    for segments in calls.values():
        chunks = merge_segments(segments, max_duration)
        rows_per_call.append(float(len(chunks)))
        for chunk in chunks:
            token_counts.append(float(len(tokens(chunk["text"]))))
            durations.append(float(chunk["end"] - chunk["start"]))
    return {
        "rows_per_call": summary(rows_per_call),
        "tokens_per_row": summary(token_counts),
        "duration_s": summary(durations),
    }


def compatibility_score(candidate: dict[str, Any], maec: dict[str, Any]) -> float:
    components = (
        (candidate["rows_per_call"]["median"], maec["rows_per_call"]["median"]),
        (candidate["rows_per_call"]["q90"], maec["rows_per_call"]["q90"]),
        (candidate["tokens_per_row"]["median"], maec["tokens_per_row"]["median"]),
        (candidate["tokens_per_row"]["q75"], maec["tokens_per_row"]["q75"]),
    )
    return float(sum(abs(math.log(max(left, 1.0) / max(right, 1.0))) for left, right in components))


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    freeze_path = root / FREEZE
    output_path = root / OUTPUT
    if output_path.exists():
        raise SystemExit("Segmentation compatibility report already exists; refusing to overwrite it.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_ANY_PRICE_OR_TARGET_ACCESS":
        raise SystemExit("Unexpected forward-confirmation freeze status.")
    primary_info = freeze["primary_cohort"]
    primary_path = root / primary_info["path"]
    if sha256(primary_path) != primary_info["sha256"]:
        raise SystemExit("Frozen primary cohort hash changed.")
    primary = read_csv(primary_path)
    allowed_ids = {row["call_id"] for row in primary}

    maec = load_maec_training(root)
    earnings = load_earnings_segments(root, allowed_ids)
    policies: dict[str, Any] = {}
    for name, cap in POLICIES.items():
        report = policy_report(earnings, cap)
        report["same_speaker_gap_tolerance_s"] = None if cap == 0.0 else GAP_TOLERANCE_S
        report["maximum_chunk_duration_s"] = cap
        report["compatibility_score_lower_is_better"] = compatibility_score(report, maec)
        policies[name] = report
    ranking = sorted(
        policies,
        key=lambda name: (policies[name]["compatibility_score_lower_is_better"], name),
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SEGMENTATION_POLICY_REVIEW_REQUIRED",
        "freeze_manifest": {
            "path": FREEZE.as_posix(),
            "sha256": sha256(freeze_path),
        },
        "frozen_primary_calls": len(primary),
        "maec_training_interface": maec,
        "earnings25_policies": policies,
        "compatibility_ranking": ranking,
        "selection_rule": (
            "Choose the deterministic policy with the lowest prespecified "
            "log-ratio compatibility score before batch feature extraction; "
            "do not inspect prices, targets, or model predictions."
        ),
        "data_use_guards": {
            "audio_decoded": False,
            "prices_retrieved": False,
            "targets_constructed": False,
            "model_fitted_or_evaluated": False,
        },
        "next_step": "Freeze the top-ranked segmentation policy and run a second acoustic pilot.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path.relative_to(root)}")
    print("Compatibility ranking:")
    for name in ranking:
        print(f"  {name}: {policies[name]['compatibility_score_lower_is_better']:.6f}")
    print("No audio, prices, targets, model fitting, or predictions were accessed.")


if __name__ == "__main__":
    main()