"""P02-AHENK yfinance kapsama ve otomatik hedef üretim pilotu.

GPU kullanmaz. Ana çağrı manifestinin üzerine yazmaz. Zaman çizelgesine eşit
aralıklı yayılan çağrıları seçer, yerel fiyat önbelleği oluşturur ve çağrı
sonrasındaki ilk beş işlem günü için gerçekleşen varyansı hesaplar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "yfinance kurulu değil. Önce ./scripts/install_market_stack.sh çalıştırın."
    ) from exc


def safe_name(symbol: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in symbol)


def yahoo_symbol(ticker: str) -> str:
    """Yahoo sınıf hisselerinde genellikle tire kullanır: BRK.B -> BRK-B."""
    return ticker.strip().upper().replace(".", "-")


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count >= len(rows):
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = {
        round(index * (len(rows) - 1) / (count - 1)) for index in range(count)
    }
    return [rows[index] for index in sorted(indices)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index, utc=True).tz_convert(None).normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    normalized.index.name = "Date"
    return normalized


def select_price_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("Adj Close", "Close"):
        if candidate in frame.columns:
            return candidate
    return None


def realized_variance_5d(
    frame: pd.DataFrame, call_date: date
) -> tuple[float | None, list[str], str | None]:
    price_column = select_price_column(frame)
    if price_column is None:
        return None, [], "missing_adjusted_or_close_column"

    prices = pd.to_numeric(frame[price_column], errors="coerce").dropna()
    prices = prices[prices > 0]
    if prices.empty:
        return None, [], "no_positive_prices"

    call_timestamp = pd.Timestamp(call_date)
    future_dates = prices.index[prices.index > call_timestamp]
    if len(future_dates) < 5:
        return None, [], "fewer_than_five_post_call_trading_days"

    selected_dates = future_dates[:5]
    squared_returns: list[float] = []
    for trading_day in selected_dates:
        position = prices.index.get_loc(trading_day)
        if not isinstance(position, int) or position == 0:
            return None, [], "missing_previous_close"
        previous = float(prices.iloc[position - 1])
        current = float(prices.iloc[position])
        squared_returns.append(math.log(current / previous) ** 2)

    return (
        float(sum(squared_returns)),
        [timestamp.date().isoformat() for timestamp in selected_dates],
        None,
    )


def download_symbol(
    symbol: str,
    start: date,
    end: date,
    cache_path: Path,
    sleep_seconds: float,
    retries: int,
    retry_backoff: float,
) -> tuple[pd.DataFrame, str, int]:
    cached_frame: pd.DataFrame | None = None
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
        cached_frame = normalize_history(cached)
        if not cached_frame.empty:
            cached_start = cached_frame.index.min().date()
            cached_end = cached_frame.index.max().date()
            start_ok = cached_start <= start + timedelta(days=7)
            end_ok = cached_end >= end - timedelta(days=7)
            if start_ok and end_ok:
                return cached_frame, "cache", 0
            print(
                f"  {symbol} önbelleği yetersiz "
                f"({cached_start}..{cached_end}); tam aralık yeniden indirilecek."
            )

    last_history = pd.DataFrame()
    for attempt in range(1, max(1, retries) + 1):
        last_history = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=True,
            timeout=30,
            raise_errors=False,
        )
        last_history = normalize_history(last_history)
        if not last_history.empty:
            last_history.to_csv(cache_path)
            time.sleep(max(0.0, sleep_seconds))
            return last_history, "download", attempt
        if attempt < retries:
            wait = retry_backoff * attempt
            print(f"  {symbol} boş döndü; {wait:.1f} saniye sonra tekrar denenecek.")
            time.sleep(wait)

    time.sleep(max(0.0, sleep_seconds))
    if cached_frame is not None and not cached_frame.empty:
        return cached_frame, "partial_cache", max(1, retries)
    return last_history, "download", max(1, retries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help="pilot: zaman-yayılı örnek; full: bütün MAEC çağrıları.",
    )
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')

    root = Path(args.project_root).resolve()
    calls_path = root / "data/manifests/calls.csv"
    if not calls_path.is_file():
        raise SystemExit(f"Çağrı manifesti bulunamadı: {calls_path}")

    with calls_path.open("r", encoding="utf-8", newline="") as handle:
        all_calls = list(csv.DictReader(handle))
    all_calls.sort(key=lambda row: (row["call_date"], row["call_id"]))
    selected_calls = (
        all_calls
        if args.mode == "full"
        else evenly_spaced(all_calls, args.max_calls)
    )
    if not selected_calls:
        raise SystemExit("Manifestte çağrı bulunamadı.")

    cache_dir = root / "data/raw/market/yfinance"
    report_dir = root / "reports/data_quality"
    output_manifest = root / (
        "data/manifests/market_full_calls.csv"
        if args.mode == "full"
        else "data/manifests/market_pilot_calls.csv"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    calls_by_symbol: dict[str, list[dict[str, str]]] = {}
    for row in selected_calls:
        symbol = yahoo_symbol(row["ticker"])
        calls_by_symbol.setdefault(symbol, []).append(row)

    histories: dict[str, pd.DataFrame] = {}
    symbol_reports: dict[str, dict] = {}
    for number, (symbol, rows) in enumerate(sorted(calls_by_symbol.items()), start=1):
        dates = [date.fromisoformat(row["call_date"]) for row in rows]
        start = min(dates) - timedelta(days=90)
        end = max(dates) + timedelta(days=20)
        cache_path = cache_dir / f"{safe_name(symbol)}.csv"
        try:
            frame, source, attempts = download_symbol(
                symbol,
                start,
                end,
                cache_path,
                args.sleep,
                args.retries,
                args.retry_backoff,
            )
            histories[symbol] = frame
            symbol_reports[symbol] = {
                "status": "ok" if not frame.empty else "empty",
                "rows": int(len(frame)),
                "repaired_rows": int(
                    frame["Repaired?"].fillna(False).astype(bool).sum()
                )
                if "Repaired?" in frame.columns
                else 0,
                "source": source,
                "attempts": attempts,
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
                "cache_path": cache_path.relative_to(root).as_posix()
                if cache_path.exists()
                else None,
                "cache_sha256": sha256(cache_path) if cache_path.exists() else None,
            }
        except Exception as exc:  # Tek sembol hatası bütün pilotu durdurmaz.
            histories[symbol] = pd.DataFrame()
            symbol_reports[symbol] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
            }
        print(f"[{number}/{len(calls_by_symbol)}] {symbol}: {symbol_reports[symbol]['status']}")

    output_rows = []
    failures: dict[str, int] = {}
    for row in selected_calls:
        symbol = yahoo_symbol(row["ticker"])
        frame = histories[symbol]
        target, target_dates, error = realized_variance_5d(
            frame, date.fromisoformat(row["call_date"])
        )
        if error:
            failures[error] = failures.get(error, 0) + 1
        output_rows.append(
            {
                **row,
                "yahoo_symbol": symbol,
                "target_realized_variance_5d": ""
                if target is None
                else format(target, ".12g"),
                "target_trading_dates": "|".join(target_dates),
                "market_status": "PASS" if target is not None else "FAIL",
                "market_error": error or "",
            }
        )

    output_fields = list(output_rows[0].keys())
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    successful_calls = sum(row["market_status"] == "PASS" for row in output_rows)
    coverage = successful_calls / len(output_rows)
    coverage_by_year: dict[str, dict[str, int | float]] = {}
    for row in output_rows:
        year = row["call_date"][:4]
        bucket = coverage_by_year.setdefault(
            year, {"calls": 0, "successful": 0, "failed": 0, "coverage": 0.0}
        )
        bucket["calls"] += 1
        if row["market_status"] == "PASS":
            bucket["successful"] += 1
        else:
            bucket["failed"] += 1
    for bucket in coverage_by_year.values():
        bucket["coverage"] = round(bucket["successful"] / bucket["calls"], 6)

    if args.mode == "pilot":
        minimum_coverage = 0.90
        status = "PASS" if successful_calls >= 50 and coverage >= minimum_coverage else "FAIL"
        g0_status = (
            "PILOT_PASS_FULL_DOWNLOAD_PENDING"
            if status == "PASS"
            else "MARKET_SOURCE_REVIEW_REQUIRED"
        )
    else:
        minimum_coverage = 0.70
        status = (
            "PASS_WITH_SCOPE_LIMITATION"
            if successful_calls >= 1000 and coverage >= minimum_coverage
            else "FAIL"
        )
        g0_status = (
            "PASS_YAHOO_AVAILABLE_SUBSET"
            if status == "PASS_WITH_SCOPE_LIMITATION"
            else "INSUFFICIENT_LABELED_CALLS"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "provider": "Yahoo Finance via yfinance",
        "usage": "research_pilot_no_raw_redistribution",
        "yfinance_version": yf.__version__,
        "selection": "evenly_spaced_over_call_timeline",
        "selected_calls": len(output_rows),
        "unique_requested_symbols": len(calls_by_symbol),
        "successful_calls": successful_calls,
        "failed_calls": len(output_rows) - successful_calls,
        "call_coverage": round(coverage, 6),
        "failure_reasons": failures,
        "coverage_by_year": coverage_by_year,
        "failed_symbols": sorted(
            symbol
            for symbol, details in symbol_reports.items()
            if details.get("status") != "ok"
        ),
        "symbol_reports": symbol_reports,
        "human_or_expert_labels_used": False,
        "target_definition": "sum of five squared daily log returns after call date",
        "minimum_required_coverage": minimum_coverage,
        "status": status,
        "g0_status": g0_status,
        "scope_note": (
            "Analysis is conditional on historical prices being available from "
            "Yahoo Finance at the recorded retrieval date."
        ),
    }
    report_path = report_dir / (
        "market_full.json" if args.mode == "full" else "market_pilot.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.mode == "full":
        eligible_path = root / "data/manifests/market_eligible_calls.csv"
        eligible_rows = [row for row in output_rows if row["market_status"] == "PASS"]
        with eligible_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(eligible_rows)
        print(f"Uygun çağrı manifesti: {eligible_path}")
    print(f"Manifest: {output_manifest}")
    print(f"Rapor: {report_path}")


if __name__ == "__main__":
    main()