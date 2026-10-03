#!/usr/bin/env python3
"""Fetch and audit Earnings25 metadata without touching market outcomes.

This is the first stage of a prospective, forward-time confirmation study.
The lightweight Dataset Viewer route is retained for diagnostics, but the
Earnings25 row groups currently exceed that service's scan limit.  The
supported ``--source parquet`` route downloads the pinned ``full`` split and
reads only metadata columns locally.  It does not retrieve prices, construct
targets, train a model, or inspect predictive performance.

Outputs are created once and are never overwritten:

* data/forward_confirmation/earnings25_metadata.csv
* reports/confirmation/earnings25_metadata_audit.json

The next protocol stage must resolve historical issuer identities and freeze
an eligible cohort before any pre-call prices or post-call targets are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET = "florencejiang/earnings25"
CONFIG = "full"
SPLIT = "test"
LICENSE = "cc-by-4.0"
MINIMUM_FORWARD_DATE = "2018-06-22"
EXPECTED_CALLS = 514
PAGE_SIZE = 10
HTTP_RETRIES = 4
PARQUET_SHARDS = 24
MINIMUM_FREE_GIB = 14


class DatasetViewerHTTPError(RuntimeError):
    """An HTTP response that may be recoverable with a smaller row slice."""

    def __init__(self, status: int, url: str, body: str) -> None:
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--minimum-forward-date",
        default=MINIMUM_FORWARD_DATE,
        help="Earliest admissible call date in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--expected-revision",
        default="",
        help="Optional Hugging Face repository commit SHA to require.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=PAGE_SIZE,
        help="Dataset Viewer rows per request (1--100; default: 10).",
    )
    parser.add_argument(
        "--source",
        choices=("rows", "parquet"),
        default="rows",
        help=(
            "Metadata source. Use 'parquet' for Earnings25 because its large "
            "row groups exceed the Dataset Viewer scan limit."
        ),
    )
    parser.add_argument(
        "--download-dir",
        default="data/raw/earnings25/full",
        help="Directory for the 24 pinned full-split Parquet shards.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test row parsing locally without network access or file writes.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "AHENK-forward-confirmation/1.1"})
    last_error: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = DatasetViewerHTTPError(exc.code, url, body)
            if exc.code not in {429, 500, 502, 503, 504}:
                raise last_error from exc
        except (TimeoutError, URLError) as exc:
            last_error = exc
        if attempt + 1 < HTTP_RETRIES:
            wait_seconds = 2**attempt
            print(
                f"Temporary metadata-service error; retrying in {wait_seconds}s "
                f"({attempt + 1}/{HTTP_RETRIES})..."
            )
            time.sleep(wait_seconds)
    else:
        if isinstance(last_error, DatasetViewerHTTPError):
            raise last_error
        raise RuntimeError(f"Could not retrieve {url}: {last_error}") from last_error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def repository_metadata(timeout: float) -> dict[str, str]:
    payload = get_json(
        f"https://huggingface.co/api/datasets/{DATASET}",
        timeout,
    )
    card = payload.get("cardData") or {}
    license_name = str(card.get("license") or payload.get("license") or "").lower()
    if license_name != LICENSE:
        raise RuntimeError(
            f"Expected dataset license {LICENSE!r}, received {license_name!r}."
        )
    revision = str(payload.get("sha") or "")
    if not revision:
        raise RuntimeError("Hugging Face repository revision was not reported.")
    return {"license": license_name, "revision": revision}


def normalize_timestamp(value: Any) -> tuple[str, str, str]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("release_date is empty")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    timezone_status = (
        "explicit_utc_offset"
        if parsed.tzinfo is not None and parsed.utcoffset() is not None
        else "missing_utc_offset"
    )
    return raw, parsed.date().isoformat(), timezone_status


def parse_row(item: dict[str, Any]) -> dict[str, str]:
    row_index = item.get("row_idx")
    row = item.get("row")
    if not isinstance(row, dict):
        raise ValueError(f"Row {row_index!r} has no row object")
    call_id = str(row.get("id") or "").strip()
    company = str(row.get("company") or "").strip()
    if not call_id or not company:
        raise ValueError(f"Row {row_index!r} lacks id or company")
    timestamp, call_date, timezone_status = normalize_timestamp(row.get("release_date"))
    audio = row.get("audio")
    audio_available = isinstance(audio, dict) and bool(audio.get("src"))
    return {
        "source_row_index": str(row_index),
        "call_id": call_id,
        "company_name": company,
        "country": str(row.get("country") or "").strip(),
        "industry": str(row.get("industry") or "").strip(),
        "call_timestamp": timestamp,
        "call_date": call_date,
        "timestamp_timezone_status": timezone_status,
        "duration_s": str(row.get("duration_s") or ""),
        "market_cap": str(row.get("market_cap") or ""),
        "audio_available": "1" if audio_available else "0",
        "identity_status": "unresolved",
        "historical_ticker": "",
        "cik": "",
        "isin": "",
        "eligibility_status": "pending_identity_review",
        "exclusion_reason": "",
    }


def dataset_rows(timeout: float, requested_page_size: int) -> tuple[list[dict[str, str]], int]:
    if not 1 <= requested_page_size <= 100:
        raise ValueError("--page-size must be between 1 and 100.")
    output: list[dict[str, str]] = []
    offset = 0
    total: int | None = None
    page_size = requested_page_size
    while total is None or offset < total:
        length = page_size if total is None else min(page_size, total - offset)
        while True:
            query = urlencode(
                {
                    "dataset": DATASET,
                    "config": CONFIG,
                    "split": SPLIT,
                    "offset": offset,
                    "length": length,
                }
            )
            try:
                payload = get_json(
                    f"https://datasets-server.huggingface.co/rows?{query}",
                    timeout,
                )
                break
            except DatasetViewerHTTPError as exc:
                if exc.status not in {429, 500, 502, 503, 504} or length == 1:
                    raise
                smaller = max(1, length // 2)
                print(
                    f"Dataset Viewer rejected offset={offset}, length={length}; "
                    f"retrying with length={smaller}."
                )
                length = smaller
                page_size = min(page_size, smaller)
        if payload.get("partial") is True:
            raise RuntimeError("Dataset Viewer returned a partial row index.")
        reported_total = int(payload.get("num_rows_total", -1))
        if reported_total <= 0:
            raise RuntimeError("Dataset Viewer did not report a valid row count.")
        if total is None:
            total = reported_total
        elif total != reported_total:
            raise RuntimeError("Dataset row count changed during metadata retrieval.")
        page = payload.get("rows")
        if not isinstance(page, list) or not page:
            raise RuntimeError(f"No rows returned at offset {offset}.")
        output.extend(parse_row(item) for item in page)
        offset += len(page)
        print(f"Metadata rows: {offset}/{total}")
    assert total is not None
    return output, total


def shard_name(index: int) -> str:
    return f"test-{index:05d}-of-{PARQUET_SHARDS:05d}.parquet"


def download_file(url: str, target: Path, timeout: float) -> None:
    if target.is_file():
        print(f"Already downloaded: {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "AHENK-forward-confirmation/1.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Could not download {target.name}: HTTP {exc.code}: {body[:300]}") from exc

    status = getattr(response, "status", response.getcode())
    append = offset > 0 and status == 206
    if offset and not append:
        print(f"Server did not accept resume for {target.name}; restarting shard.")
        offset = 0
    mode = "ab" if append else "wb"
    content_length = int(response.headers.get("Content-Length") or 0)
    expected = offset + content_length if content_length else 0
    downloaded = offset
    last_report = time.monotonic()
    with response, partial.open(mode) as handle:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            downloaded += len(block)
            if time.monotonic() - last_report >= 10:
                if expected:
                    print(
                        f"  {target.name}: {downloaded / 2**20:.1f}/"
                        f"{expected / 2**20:.1f} MiB"
                    )
                else:
                    print(f"  {target.name}: {downloaded / 2**20:.1f} MiB")
                last_report = time.monotonic()
    if expected and downloaded != expected:
        raise RuntimeError(
            f"Incomplete download for {target.name}: {downloaded} of {expected} bytes. "
            "Rerun the same command to resume."
        )
    os.replace(partial, target)
    print(f"Downloaded: {target.name} ({downloaded / 2**20:.1f} MiB)")


def parquet_metadata_rows(
    root: Path,
    download_dir: str,
    revision: str,
    timeout: float,
) -> tuple[list[dict[str, str]], int]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyArrow is required for --source parquet. Activate the ahenk "
            "environment defined by environment.yml."
        ) from exc

    destination = root / download_dir
    free_gib = shutil.disk_usage(destination.parent if destination.parent.exists() else root).free / 2**30
    print(f"Free disk space: {free_gib:.1f} GiB")
    if free_gib < MINIMUM_FREE_GIB:
        raise SystemExit(
            f"At least {MINIMUM_FREE_GIB} GiB free space is required before "
            "downloading the full evaluation split."
        )

    paths: list[Path] = []
    for index in range(PARQUET_SHARDS):
        name = shard_name(index)
        path = destination / name
        url = (
            f"https://huggingface.co/datasets/{DATASET}/resolve/"
            f"{revision}/full/{name}?download=true"
        )
        download_file(url, path, timeout)
        paths.append(path)

    selected = [
        "id",
        "company",
        "country",
        "industry",
        "release_date",
        "duration_s",
        "market_cap",
    ]
    rows: list[dict[str, str]] = []
    global_index = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        missing = sorted(set(selected + ["audio", "text"]) - names)
        if missing:
            raise RuntimeError(f"{path.name} lacks required columns: {missing}")
        table = parquet.read(columns=selected)
        for source in table.to_pylist():
            timestamp, call_date, timezone_status = normalize_timestamp(
                source.get("release_date")
            )
            call_id = str(source.get("id") or "").strip()
            company = str(source.get("company") or "").strip()
            if not call_id or not company:
                raise RuntimeError(f"{path.name} contains a row without id or company.")
            rows.append(
                {
                    "source_row_index": str(global_index),
                    "call_id": call_id,
                    "company_name": company,
                    "country": str(source.get("country") or "").strip(),
                    "industry": str(source.get("industry") or "").strip(),
                    "call_timestamp": timestamp,
                    "call_date": call_date,
                    "timestamp_timezone_status": timezone_status,
                    "duration_s": str(source.get("duration_s") or ""),
                    "market_cap": str(source.get("market_cap") or ""),
                    "audio_available": "stored_in_parquet_not_decoded",
                    "identity_status": "unresolved",
                    "historical_ticker": "",
                    "cik": "",
                    "isin": "",
                    "eligibility_status": "pending_identity_review",
                    "exclusion_reason": "",
                }
            )
            global_index += 1
        print(f"Read metadata: {global_index} calls")
    return rows, len(rows)


def audit_rows(
    rows: list[dict[str, str]],
    reported_total: int,
    minimum_forward_date: str,
) -> dict[str, Any]:
    if reported_total != EXPECTED_CALLS or len(rows) != EXPECTED_CALLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CALLS} full calls, found "
            f"reported={reported_total}, parsed={len(rows)}."
        )
    call_ids = [row["call_id"] for row in rows]
    if len(set(call_ids)) != len(call_ids):
        raise RuntimeError("Duplicate Earnings25 call IDs were found.")
    dates = [row["call_date"] for row in rows]
    if min(dates) < minimum_forward_date:
        raise RuntimeError(
            f"Earliest call {min(dates)} precedes required date {minimum_forward_date}."
        )
    missing_audio = sum(row["audio_available"] == "0" for row in rows)
    if missing_audio:
        raise RuntimeError(f"{missing_audio} metadata rows explicitly lack audio.")
    audio_not_decoded = sum(
        row["audio_available"] == "stored_in_parquet_not_decoded" for row in rows
    )
    missing_timezone = sum(
        row["timestamp_timezone_status"] == "missing_utc_offset" for row in rows
    )
    explicit_timezone = sum(
        row["timestamp_timezone_status"] == "explicit_utc_offset" for row in rows
    )
    companies = {row["company_name"] for row in rows}
    return {
        "calls": len(rows),
        "company_names": len(companies),
        "date_min": min(dates),
        "date_max": max(dates),
        "minimum_forward_date": minimum_forward_date,
        "all_calls_strictly_forward": True,
        "missing_audio_rows": missing_audio,
        "audio_rows_not_decoded_during_metadata_audit": audio_not_decoded,
        "timestamps_with_explicit_utc_offset": explicit_timezone,
        "timestamps_missing_utc_offset": missing_timezone,
        "timestamp_timezone_resolution_complete": missing_timezone == 0,
        "historical_identity_resolution_complete": False,
        "market_prices_or_targets_retrieved": False,
        "model_training_or_prediction_performed": False,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    example = {
        "row_idx": 7,
        "row": {
            "id": "example-call",
            "company": "Example Corporation",
            "country": "US",
            "industry": "Example",
            "release_date": "2025-10-01T20:30:00+00:00",
            "duration_s": 3600.0,
            "market_cap": 1_000_000_000,
            "audio": {"src": "https://example.invalid/audio.mp3"},
        },
    }
    parsed = parse_row(example)
    assert parsed["call_date"] == "2025-10-01"
    assert parsed["timestamp_timezone_status"] == "explicit_utc_offset"
    assert parsed["audio_available"] == "1"
    assert parsed["eligibility_status"] == "pending_identity_review"
    naive = dict(example)
    naive["row"] = dict(example["row"])
    naive["row"]["release_date"] = "2025-10-28T13:00:00"
    parsed_naive = parse_row(naive)
    assert parsed_naive["call_date"] == "2025-10-28"
    assert parsed_naive["timestamp_timezone_status"] == "missing_utc_offset"
    print("Earnings25 metadata parser self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    root = Path(args.project_root).resolve()
    csv_path = root / "data/forward_confirmation/earnings25_metadata.csv"
    report_path = root / "reports/confirmation/earnings25_metadata_audit.json"
    if csv_path.exists() or report_path.exists():
        raise SystemExit(
            "Earnings25 metadata outputs already exist; refusing to overwrite them."
        )

    repository = repository_metadata(args.timeout)
    if args.expected_revision and repository["revision"] != args.expected_revision:
        raise SystemExit(
            "Dataset revision differs from --expected-revision: "
            f"{repository['revision']}"
        )
    if args.source == "rows":
        try:
            rows, reported_total = dataset_rows(args.timeout, args.page_size)
        except DatasetViewerHTTPError as exc:
            if "Scan size limit exceeded" in exc.body:
                raise SystemExit(
                    "The Hugging Face Dataset Viewer cannot scan the Earnings25 "
                    "row groups. Rerun with --source parquet to download the 24 "
                    "pinned full-split shards and read metadata locally."
                ) from exc
            raise
    else:
        rows, reported_total = parquet_metadata_rows(
            root,
            args.download_dir,
            repository["revision"],
            args.timeout,
        )
    audit = audit_rows(rows, reported_total, args.minimum_forward_date)
    write_csv(csv_path, rows)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "CANDIDATE_METADATA_ONLY_IDENTITY_AND_TIMEZONE_RESOLUTION_REQUIRED"
            if audit["timestamps_missing_utc_offset"]
            else "CANDIDATE_METADATA_ONLY_IDENTITY_RESOLUTION_REQUIRED"
        ),
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "metadata_source": args.source,
        "license": repository["license"],
        "repository_revision": repository["revision"],
        "metadata_csv": {
            "path": csv_path.relative_to(root).as_posix(),
            "sha256": sha256(csv_path),
        },
        "audit": audit,
        "data_use_guards": {
            "audio_downloaded": args.source == "parquet",
            "audio_decoded_or_features_extracted": False,
            "prices_retrieved": False,
            "targets_constructed": False,
            "model_fitted_or_evaluated": False,
        },
        "next_step": (
            "Resolve historical issuer identifiers, missing timestamp timezones, "
            "and overlap with all prior modeling/evaluation cohorts; then freeze "
            "eligibility before prices."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path.relative_to(root)}")
    print(f"Wrote {report_path.relative_to(root)}")
    if args.source == "parquet":
        print(
            "Audio-bearing Parquet shards were downloaded but audio was not "
            "decoded; no prices, targets, training, or predictions were accessed."
        )
    else:
        print("No audio, prices, targets, training, or predictions were accessed.")


if __name__ == "__main__":
    main()