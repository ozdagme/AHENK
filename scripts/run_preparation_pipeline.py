"""P02-AHENK CPU tabanlı proje ve veri boru hattı hazırlığı.

Bu aşama GPU kullanmaz, veri indirmez ve insan/uzman etiketi üretmez.
Proje dizinlerini, kaynak kayıtlarını, çağrı manifestini ve veri sözleşmesini
oluşturur. Mevcut dosyaların üzerine yazmaz.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


DIRECTORIES = (
    "config",
    "data/raw/maec",
    "data/raw/market",
    "data/interim/aligned_calls",
    "data/processed/features",
    "data/manifests",
    "logs",
    "reports/environment",
    "reports/data_quality",
    "artifacts/models",
    "artifacts/predictions",
)


def write_json_if_missing(path: Path, payload: dict) -> str:
    if path.exists():
        return "KEPT"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return "CREATED"


def write_csv_if_missing(path: Path, fieldnames: list[str], rows: list[dict]) -> str:
    if path.exists():
        return "KEPT"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return "CREATED"


def gib(value: int) -> float:
    return round(value / 2**30, 3)


def system_snapshot(root: Path, check_gpu: bool) -> dict:
    disk = shutil.disk_usage(root)
    memory = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        memory = {
            "total_gib": gib(vm.total),
            "available_gib": gib(vm.available),
        }
    except ImportError:
        memory = {"status": "psutil_not_installed"}

    torch_status: dict = {"installed": False, "gpu_check_skipped": not check_gpu}
    try:
        torch_status = {
            "installed": True,
            "version": importlib.metadata.version("torch"),
            "gpu_check_skipped": not check_gpu,
        }
        if check_gpu:
            import torch

            torch_status.update(
                {
                    "cuda_build": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                }
            )
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                torch_status.update(
                    {
                        "device": torch.cuda.get_device_name(0),
                        "free_vram_gib": gib(free),
                        "total_vram_gib": gib(total),
                    }
                )
    except importlib.metadata.PackageNotFoundError:
        pass

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory": memory,
        "disk": {
            "total_gib": gib(disk.total),
            "free_gib": gib(disk.free),
        },
        "pytorch": torch_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=".",
        help="AHENK proje kökü; varsayılan mevcut dizindir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dosya oluşturmadan yapılacak işlemleri gösterir.",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="GPU boşken CUDA durumunu da rapora ekler; varsayılan kapalıdır.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    if args.dry_run:
        print("DRY RUN — oluşturulacak dizinler:")
        for relative in DIRECTORIES:
            print(f"  {root / relative}")
        print("Oluşturulacak temel dosyalar:")
        for relative in (
            "config/pipeline.json",
            "config/data_contract.json",
            "data/manifests/sources.csv",
            "data/manifests/calls.csv",
            "reports/environment/preparation_snapshot.json",
        ):
            print(f"  {root / relative}")
        return

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root.mkdir(parents=True, exist_ok=True)
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)

    pipeline_config = {
        "project": "P02-AHENK",
        "human_or_expert_labeling": False,
        "primary_task": "post_call_realized_variance_prediction",
        "label": {
            "type": "automatic_market_derived",
            "formula": "sum_of_squared_daily_log_returns",
            "horizon_trading_days": 5,
            "minimum_start": "first_full_trading_day_after_publication",
        },
        "splits": {
            "primary": "forward_time",
            "secondary": "unseen_company",
            "group_key": "call_id",
        },
        "missing_modality_tests": [0.0, 0.25, 0.5],
        "primary_metric": "QLIKE",
        "random_seeds": [3407, 2026, 2709],
        "gpu_policy": "single_heavy_job_only",
    }

    data_contract = {
        "required_fields": {
            "call_id": "string",
            "company_id": "string",
            "ticker": "string_or_null",
            "call_date": "ISO-8601 date",
            "publication_time": "ISO-8601 datetime_or_null",
            "publication_time_confidence": "exact|date_only|unknown",
            "text_path": "relative_path_or_null",
            "audio_feature_path": "relative_path_or_null",
            "market_price_path": "relative_path_or_null",
            "has_text": "0|1",
            "has_audio_features": "0|1",
            "has_market_history": "0|1",
            "text_quality": "float_0_1_or_null",
            "audio_quality": "float_0_1_or_null",
            "time_quality": "float_0_1_or_null",
            "target_realized_variance_5d": "positive_float_or_null",
            "split": "train|dev|test|unassigned",
            "source_sha256": "hex_string_or_null",
        },
        "invariants": [
            "call_id is unique",
            "all paths are relative to project root",
            "future prices are labels only and never model inputs",
            "all sentences from one call remain in one split",
            "missing modality has mask value zero",
            "no human or expert labels are permitted",
        ],
    }

    source_fields = [
        "source_id",
        "role",
        "status",
        "license_status",
        "source_url",
        "retrieved_at_utc",
        "sha256",
        "notes",
    ]
    source_rows = [
        {
            "source_id": "maec_small_features",
            "role": "text_and_acoustic_features",
            "status": "pending_access_check",
            "license_status": "pending_review",
            "source_url": "",
            "retrieved_at_utc": "",
            "sha256": "",
            "notes": "Use small feature package first; defer 59 GB MFCC archive.",
        },
        {
            "source_id": "market_prices",
            "role": "automatic_target_generation",
            "status": "pending_source_selection",
            "license_status": "pending_review",
            "source_url": "",
            "retrieved_at_utc": "",
            "sha256": "",
            "notes": "Adjusted prices and trading calendar are required.",
        },
    ]

    call_fields = list(data_contract["required_fields"].keys())
    statuses = {
        "config/pipeline.json": write_json_if_missing(
            root / "config/pipeline.json", pipeline_config
        ),
        "config/data_contract.json": write_json_if_missing(
            root / "config/data_contract.json", data_contract
        ),
        "data/manifests/sources.csv": write_csv_if_missing(
            root / "data/manifests/sources.csv", source_fields, source_rows
        ),
        "data/manifests/calls.csv": write_csv_if_missing(
            root / "data/manifests/calls.csv", call_fields, []
        ),
    }

    snapshot_path = root / "reports/environment/preparation_snapshot.json"
    snapshot = system_snapshot(root, check_gpu=args.check_gpu)
    snapshot["file_statuses"] = statuses
    snapshot["preparation_status"] = "PASS"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("P02-AHENK hazırlık boru hattı tamamlandı.")
    for path, status in statuses.items():
        print(f"{status:7} {path}")
    print(f"UPDATED reports/environment/preparation_snapshot.json")
    print("Sonraki kapı: MAEC erişimi ve lisans durumunun doğrulanması (G0).")


if __name__ == "__main__":
    main()