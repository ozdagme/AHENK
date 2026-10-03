"""MAEC deposundaki fiyat/etiket adaylarını güvenli biçimde inceler.

Pickle dosyalarını yüklemez; yalnız dosya karması ve pickle opcode/global
özetini çıkarır. CSV dosyalarının şema ve tarih aralığını raporlar. GPU kullanmaz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickletools
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        sample_rows = []
        row_count = 0
        first_values: list[str] = []
        last_values: list[str] = []
        for row in reader:
            row_count += 1
            if len(sample_rows) < 3:
                sample_rows.append(row[:8])
            if row:
                if len(first_values) < 5:
                    first_values.append(row[0])
                last_values.append(row[0])
                if len(last_values) > 5:
                    last_values.pop(0)
    return {
        "header": header,
        "column_count": len(header),
        "row_count": row_count,
        "first_column_first_values": first_values,
        "first_column_last_values": last_values,
        "sample_rows_first_8_columns": sample_rows,
    }


def inspect_pickle_without_loading(path: Path) -> dict:
    opcode_counts: Counter[str] = Counter()
    globals_found: list[str] = []
    parse_error = None
    try:
        with path.open("rb") as handle:
            for opcode, argument, _position in pickletools.genops(handle):
                opcode_counts[opcode.name] += 1
                if opcode.name in {"GLOBAL", "STACK_GLOBAL"}:
                    globals_found.append(str(argument))
                    if len(globals_found) >= 100:
                        break
    except Exception as exc:  # Rapor üretimini kesmeden bozuk/uyumsuz pickle'ı göster.
        parse_error = f"{type(exc).__name__}: {exc}"

    return {
        "loaded_or_executed": False,
        "opcode_counts": dict(opcode_counts.most_common(30)),
        "global_references_first_100": globals_found,
        "parse_error": parse_error,
        "security_note": "Pickle güvenlik nedeniyle yüklenmedi.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--sample-calls", type=int, default=10)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    repository = root / "data/raw/maec/repository"
    dataset = repository / "MAEC_Dataset"
    if not dataset.is_dir():
        raise SystemExit(f"MAEC veri klasörü bulunamadı: {dataset}")

    top_level_files = []
    csv_reports: dict[str, dict] = {}
    pickle_reports: dict[str, dict] = {}

    for path in sorted(repository.iterdir()):
        if not path.is_file():
            continue
        entry = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        top_level_files.append(entry)
        if path.suffix.lower() == ".csv":
            csv_reports[path.name] = inspect_csv(path)
        elif path.suffix.lower() in {".pkl", ".pickle"}:
            pickle_reports[path.name] = inspect_pickle_without_loading(path)

    sample_folders = sorted(path for path in dataset.iterdir() if path.is_dir())[
        : args.sample_calls
    ]
    call_folder_samples = []
    discovered_names: Counter[str] = Counter()
    for folder in sample_folders:
        files = []
        for path in sorted(folder.iterdir()):
            if path.is_file():
                discovered_names[path.name] += 1
                files.append({"name": path.name, "size_bytes": path.stat().st_size})
        call_folder_samples.append({"call_id": folder.name, "files": files})

    name_scan_limit = min(200, sum(1 for path in dataset.iterdir() if path.is_dir()))
    for folder in sorted(path for path in dataset.iterdir() if path.is_dir())[
        len(sample_folders) : name_scan_limit
    ]:
        for path in folder.iterdir():
            if path.is_file():
                discovered_names[path.name] += 1

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository.relative_to(root).as_posix(),
        "pickle_files_loaded": False,
        "top_level_files": top_level_files,
        "csv_reports": csv_reports,
        "pickle_reports": pickle_reports,
        "sample_call_folders": call_folder_samples,
        "file_names_in_first_calls": dict(discovered_names),
        "decision": "REVIEW_PRICE_FILES_BEFORE_MARKET_SOURCE_SELECTION",
    }

    output = root / "reports/data_quality/maec_financial_files.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Rapor yazıldı: {output}")


if __name__ == "__main__":
    main()