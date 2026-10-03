#!/usr/bin/env python3
"""Freeze the target-blind Earnings25 acoustic extraction policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORWARD_FREEZE = Path(
    "reports/confirmation/earnings25_forward_confirmation_freeze.json"
)
COMPATIBILITY = Path(
    "reports/confirmation/earnings25_segmentation_compatibility.json"
)
PILOT_V2 = Path("reports/confirmation/earnings25_acoustic_pilot_v2.json")
OUTPUT = Path("reports/confirmation/earnings25_acoustic_extraction_freeze.json")

EXPECTED_POLICY = "same_speaker_max_15s"
EXPECTED_SCHEMA_COLUMNS = 29


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


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    paths = {
        "forward": root / FORWARD_FREEZE,
        "compatibility": root / COMPATIBILITY,
        "pilot_v2": root / PILOT_V2,
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
        raise SystemExit("Acoustic extraction freeze already exists; refusing to overwrite it.")

    forward = read_json(paths["forward"])
    compatibility = read_json(paths["compatibility"])
    pilot = read_json(paths["pilot_v2"])
    if forward.get("status") != "FROZEN_BEFORE_ANY_PRICE_OR_TARGET_ACCESS":
        raise SystemExit("Unexpected forward-cohort freeze status.")
    if compatibility.get("status") != "SEGMENTATION_POLICY_REVIEW_REQUIRED":
        raise SystemExit("Unexpected segmentation compatibility status.")
    ranking = compatibility.get("compatibility_ranking") or []
    if not ranking or ranking[0] != EXPECTED_POLICY:
        raise SystemExit(
            f"Prespecified ranking did not select {EXPECTED_POLICY!r}: {ranking[:3]}"
        )
    if pilot.get("run_label") != "v2":
        raise SystemExit("Corrected v2 acoustic pilot was not supplied.")
    if len(pilot.get("schema") or []) != EXPECTED_SCHEMA_COLUMNS:
        raise SystemExit("Corrected pilot does not contain the required 29-column schema.")
    if pilot.get("quality", {}).get("missing_cell_fraction", 1.0) > 0.05:
        raise SystemExit("Corrected pilot missing-cell fraction exceeds 5%.")
    for report in (forward, compatibility, pilot):
        guards = report.get("data_use_guards", {})
        for key in ("prices_retrieved", "targets_constructed", "model_fitted_or_evaluated"):
            if key in guards and guards[key] is not False:
                raise SystemExit(f"Data-use guard failed before extraction freeze: {key}")

    selected = compatibility["earnings25_policies"][EXPECTED_POLICY]
    settings = pilot["frozen_extraction_settings"]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_ACOUSTIC_EXTRACTION_BEFORE_BATCH_AND_TARGETS",
        "upstream": {
            "forward_confirmation_freeze": {
                "path": FORWARD_FREEZE.as_posix(),
                "sha256": sha256(paths["forward"]),
            },
            "segmentation_compatibility": {
                "path": COMPATIBILITY.as_posix(),
                "sha256": sha256(paths["compatibility"]),
            },
            "corrected_raw_segment_pilot": {
                "path": PILOT_V2.as_posix(),
                "sha256": sha256(paths["pilot_v2"]),
                "missing_cell_fraction": pilot["quality"]["missing_cell_fraction"],
            },
        },
        "segmentation_policy": {
            "name": EXPECTED_POLICY,
            "selection_basis": "lowest prespecified compatibility score",
            "compatibility_score": selected["compatibility_score_lower_is_better"],
            "same_speaker_required": True,
            "maximum_intersegment_gap_s": selected["same_speaker_gap_tolerance_s"],
            "maximum_merged_chunk_duration_s": selected["maximum_chunk_duration_s"],
            "source_segment_longer_than_cap": "retain unsplit",
            "expected_interface_from_audit": {
                "rows_per_call": selected["rows_per_call"],
                "tokens_per_row": selected["tokens_per_row"],
                "duration_s": selected["duration_s"],
            },
        },
        "audio_decode": {
            "container": "embedded MP3",
            "decoder": pilot["software"]["ffmpeg"],
            "sample_rate_hz": settings["decoded_sample_rate"],
            "channels": settings["decoded_channels"],
            "temporary_pcm_only": True,
        },
        "praat_settings": {
            "parselmouth": pilot["software"]["parselmouth"],
            "praat": pilot["software"]["praat"],
            "pitch_floor_hz": settings["pitch_floor_hz"],
            "pitch_ceiling_hz": settings["pitch_ceiling_hz"],
            "maximum_period_factor": settings["maximum_period_factor"],
            "maximum_amplitude_factor": settings["maximum_amplitude_factor"],
            "silence_threshold": settings["silence_threshold"],
            "voicing_threshold": settings["voicing_threshold"],
        },
        "feature_schema": pilot["schema"],
        "undefined_policy": "retain segment row and write --undefined--",
        "v3_pilot_acceptance_rule": {
            "exact_text_feature_row_alignment": True,
            "maximum_missing_cell_fraction": 0.05,
            "minimum_complete_feature_row_fraction": 0.80,
            "unexpected_extraction_errors_allowed": False,
        },
        "batch_acceptance_rule": {
            "all_frozen_primary_calls_extracted": True,
            "all_text_feature_row_counts_aligned": True,
            "maximum_global_missing_cell_fraction": 0.05,
            "no_call_with_zero_feature_rows": True,
            "source_audio_and_output_hashes_required": True,
        },
        "data_use_guards": {
            "prices_retrieved_before_freeze": False,
            "targets_constructed_before_freeze": False,
            "model_evaluated_before_freeze": False,
        },
        "next_step": "Run a v3 one-call pilot using the frozen 15-second merge policy.",
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen segmentation policy: {EXPECTED_POLICY}")
    print(f"Wrote {paths['output'].relative_to(root)}")
    print("No prices, targets, model fitting, or predictions were accessed.")


if __name__ == "__main__":
    main()