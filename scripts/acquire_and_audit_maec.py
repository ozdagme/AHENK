"""P02-AHENK için MAEC küçük paketini edinir ve otomatik olarak denetler.

İnsan etiketli MAEC_Dataset_Person_Label klasörü bu boru hattında kullanılmaz.
GPU kullanmaz. Mevcut bir veri kopyasının üzerine yazmaz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_URL = (
    "https://github.com/Earnings-Call-Dataset/"
    "MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-"
    "Financial-Risk-Prediction.git"
)
FOLDER_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<ticker>[A-Za-z0-9.-]+)$")

CALL_FIELDS = [
    "call_id",
    "company_id",
    "ticker",
    "call_date",
    "publication_time",
    "publication_time_confidence",
    "text_path",
    "audio_feature_path",
    "market_price_path",
    "has_text",
    "has_audio_features",
    "has_market_history",
    "text_quality",
    "audio_quality",
    "time_quality",
    "target_realized_variance_5d",
    "split",
    "source_sha256",
]


def run(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def text_quality(path: Path) -> float | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return 0.0
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique_ratio = len(set(lines)) / max(len(lines), 1)
    length_score = min(len(text) / 2000, 1.0)
    score = 0.45 * length_score + 0.35 * unique_ratio + 0.20 * (1 - replacement_ratio)
    return round(max(0.0, min(score, 1.0)), 6)


def audio_quality(path: Path) -> float | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        first_row = next(reader, [])
    header_score = 1.0 if len(header) >= 2 else 0.0
    row_score = 1.0 if len(first_row) == len(header) and first_row else 0.0
    size_score = min(path.stat().st_size / 2048, 1.0)
    return round(0.4 * header_score + 0.4 * row_score + 0.2 * size_score, 6)


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def update_source_registry(path: Path, commit: str) -> None:
    fields = [
        "source_id",
        "role",
        "status",
        "license_status",
        "source_url",
        "retrieved_at_utc",
        "sha256",
        "notes",
    ]
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    updated = False
    for row in rows:
        if row.get("source_id") == "maec_small_features":
            row.update(
                {
                    "status": "acquired_and_audited",
                    "license_status": "CC-BY-SA-4.0_verified_in_repository",
                    "source_url": REPOSITORY_URL,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": f"git-commit:{commit}",
                    "notes": (
                        "MAEC_Dataset only; MAEC_Dataset_Person_Label excluded; "
                        "59 GB MFCC archive deferred."
                    ),
                }
            )
            updated = True
            break

    if not updated:
        rows.append(
            {
                "source_id": "maec_small_features",
                "role": "text_and_acoustic_features",
                "status": "acquired_and_audited",
                "license_status": "CC-BY-SA-4.0_verified_in_repository",
                "source_url": REPOSITORY_URL,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": f"git-commit:{commit}",
                "notes": "Human-label folder excluded; 59 GB MFCC archive deferred.",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Yalnız ilk N çağrıyı denetler; 0 bütün veri demektir.",
    )
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    destination = root / "data/raw/maec/repository"
    report_dir = root / "reports/data_quality"
    manifest_path = root / "data/manifests/calls.csv"
    source_path = root / "data/manifests/sources.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        print("MAEC deposu indiriliyor (küçük paket yaklaşık 147.7 MB)...")
        run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(destination)])
    elif not (destination / ".git").is_dir():
        raise SystemExit(f"Hedef var fakat Git deposu değil: {destination}")
    else:
        print("Mevcut MAEC deposu korunuyor; yeniden indirme yapılmadı.")

    commit = run(["git", "rev-parse", "HEAD"], cwd=destination)
    license_path = destination / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8", errors="replace")
    license_ok = "Attribution-ShareAlike 4.0" in license_text
    if not license_ok:
        raise SystemExit("Beklenen CC BY-SA 4.0 lisans metni doğrulanamadı.")

    dataset = destination / "MAEC_Dataset"
    if not dataset.is_dir():
        raise SystemExit("MAEC_Dataset klasörü bulunamadı.")

    folders = sorted(path for path in dataset.iterdir() if path.is_dir())
    if args.limit > 0:
        folders = folders[: args.limit]

    rows: list[dict[str, str | int | float | None]] = []
    invalid_names: list[str] = []
    missing_text = 0
    missing_audio = 0

    for folder in folders:
        match = FOLDER_PATTERN.fullmatch(folder.name)
        if not match:
            invalid_names.append(folder.name)
            continue

        raw_date = match.group("date")
        call_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        ticker = match.group("ticker").upper()
        text_path = folder / "text.txt"
        feature_path = folder / "features.csv"
        has_text = int(text_path.is_file() and text_path.stat().st_size > 0)
        has_audio = int(feature_path.is_file() and feature_path.stat().st_size > 0)
        missing_text += 1 - has_text
        missing_audio += 1 - has_audio

        rows.append(
            {
                "call_id": folder.name,
                "company_id": ticker,
                "ticker": ticker,
                "call_date": call_date,
                "publication_time": "",
                "publication_time_confidence": "date_only",
                "text_path": relative(text_path, root) if has_text else "",
                "audio_feature_path": relative(feature_path, root) if has_audio else "",
                "market_price_path": "",
                "has_text": has_text,
                "has_audio_features": has_audio,
                "has_market_history": 0,
                "text_quality": text_quality(text_path),
                "audio_quality": audio_quality(feature_path),
                "time_quality": 0.5,
                "target_realized_variance_5d": "",
                "split": "unassigned",
                "source_sha256": sha256_files([text_path, feature_path]),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CALL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    update_source_registry(source_path, commit)

    complete = sum(
        int(row["has_text"] == 1 and row["has_audio_features"] == 1) for row in rows
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": REPOSITORY_URL,
        "git_commit": commit,
        "license": "CC-BY-SA-4.0",
        "license_verified": license_ok,
        "human_label_folder_used": False,
        "mfcc_59gb_archive_used": False,
        "folders_scanned": len(folders),
        "valid_calls": len(rows),
        "complete_text_audio_calls": complete,
        "missing_text_calls": missing_text,
        "missing_audio_feature_calls": missing_audio,
        "invalid_folder_names": invalid_names,
        "maec_access_status": "PASS" if complete >= 50 else "FAIL",
        "g0_status": "PENDING_MARKET_DATA" if complete >= 50 else "FAIL",
    }
    report_path = report_dir / "maec_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Rapor: {report_path}")
    if report["maec_access_status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()