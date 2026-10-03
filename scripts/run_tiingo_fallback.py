"""P02-AHENK Tiingo Starter uyumlu fiyat tamamlama boru hattı.

Tiingo fiyatları yalnız bellekte tutulur. Ham fiyat, günlük getiri veya API
yanıtı diske yazılmaz. Kalıcı çıktı yalnız çağrı başına beş günlük gerçekleşen
varyans gibi geri döndürülemeyen türetilmiş sonuçlardır.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


API_ROOT = "https://api.tiingo.com/tiingo/daily"


def tiingo_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def normalize_token(value: str) -> str:
    token = value.strip().strip("'\"")
    lowered = token.lower()
    if lowered.startswith("token "):
        token = token.split(None, 1)[1].strip()
    elif lowered.startswith("token="):
        token = token.split("=", 1)[1].strip()
    return token


def validate_token(token: str) -> str | None:
    request = urllib.request.Request(
        f"{API_ROOT}/AAPL",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "User-Agent": "P02-AHENK-research/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            json.loads(response.read().decode("utf-8"))
        return None
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return "authentication_or_permission_error"
        if exc.code == 429:
            return "rate_limited"
        return f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return "network_or_response_error"


def parse_tiingo_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def fetch_prices(
    symbol: str,
    start: date,
    end: date,
    token: str,
) -> tuple[list[tuple[date, float]], str | None]:
    encoded_symbol = urllib.parse.quote(symbol, safe="-")
    query = urllib.parse.urlencode(
        {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "resampleFreq": "daily",
            "format": "json",
        }
    )
    request = urllib.request.Request(
        f"{API_ROOT}/{encoded_symbol}/prices?{query}",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "User-Agent": "P02-AHENK-research/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return [], "not_found"
        if exc.code == 429:
            return [], "rate_limited"
        if exc.code in {401, 403}:
            return [], "authentication_or_permission_error"
        return [], f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError):
        return [], "network_error"

    if not isinstance(payload, list):
        return [], "unexpected_response"

    prices: list[tuple[date, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        value = item.get("adjClose")
        timestamp = item.get("date")
        try:
            price = float(value)
            trading_date = parse_tiingo_date(str(timestamp))
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0:
            prices.append((trading_date, price))

    prices = sorted(dict(prices).items())
    return prices, None if prices else "no_positive_prices"


def realized_variance_5d(
    prices: list[tuple[date, float]], call_date: date
) -> tuple[float | None, str | None]:
    future_positions = [
        index for index, (trading_date, _price) in enumerate(prices) if trading_date > call_date
    ]
    if len(future_positions) < 5:
        return None, "fewer_than_five_post_call_trading_days"

    squared_returns: list[float] = []
    for position in future_positions[:5]:
        if position == 0:
            return None, "missing_previous_close"
        previous = prices[position - 1][1]
        current = prices[position][1]
        squared_returns.append(math.log(current / previous) ** 2)
    return float(sum(squared_returns)), None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=40,
        help="Bu çalıştırmada işlenecek en fazla sembol; Starter saatlik limitine göre 40.",
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')
    token = normalize_token(os.environ.get("TIINGO_API_KEY", ""))
    if not token:
        raise SystemExit("TIINGO_API_KEY ortam değişkeni tanımlı değil.")

    token_error = validate_token(token)
    if token_error:
        raise SystemExit(
            "Tiingo anahtar ön testi başarısız: "
            f"{token_error}. Hesap etkinliğini ve kopyalanan API tokenını kontrol edin."
        )

    root = Path(args.project_root).resolve()
    source = root / "data/manifests/market_full_calls.csv"
    combined = root / "data/manifests/market_combined_calls.csv"
    progress_path = root / "reports/data_quality/tiingo_fallback_progress.json"
    if not source.is_file():
        raise SystemExit(f"Kaynak manifest bulunamadı: {source}")

    if combined.exists():
        rows = read_rows(combined)
    else:
        rows = read_rows(source)
        for row in rows:
            row["market_provider"] = (
                "Yahoo Finance via yfinance" if row.get("market_status") == "PASS" else ""
            )
            row["tiingo_attempted"] = "0"
            row["tiingo_error"] = ""

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {
            "processed_symbols": {},
            "raw_tiingo_data_persisted": False,
            "human_or_expert_labels_used": False,
        }

    processed_symbols = progress.setdefault("processed_symbols", {})
    pending_by_symbol: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("market_status") == "PASS":
            continue
        symbol = tiingo_symbol(row.get("ticker", ""))
        if not symbol:
            continue
        previous = processed_symbols.get(symbol, {})
        if previous.get("status") in {"complete", "no_data", "not_found"}:
            continue
        pending_by_symbol[symbol].append(row)

    selected_symbols = sorted(pending_by_symbol)[: max(0, args.max_symbols)]
    if not selected_symbols:
        print("İşlenecek yeni Tiingo sembolü yok.")

    rate_limited = False
    for number, symbol in enumerate(selected_symbols, start=1):
        symbol_rows = pending_by_symbol[symbol]
        call_dates = [date.fromisoformat(row["call_date"]) for row in symbol_rows]
        start = min(call_dates) - timedelta(days=30)
        end = max(call_dates) + timedelta(days=20)

        prices, fetch_error = fetch_prices(symbol, start, end, token)
        if fetch_error == "rate_limited":
            processed_symbols[symbol] = {
                "status": "rate_limited",
                "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            rate_limited = True
            print(f"[{number}/{len(selected_symbols)}] {symbol}: rate_limited")
            break
        if fetch_error == "authentication_or_permission_error":
            processed_symbols[symbol] = {
                "status": "authentication_or_permission_error",
                "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            print("Tiingo kimlik doğrulama hatası; kalan istekler gönderilmedi.")
            break

        recovered = 0
        failure_counts: dict[str, int] = defaultdict(int)
        if fetch_error:
            failure_counts[fetch_error] += len(symbol_rows)
            for row in symbol_rows:
                row["tiingo_attempted"] = "1"
                row["tiingo_error"] = fetch_error
        else:
            for row in symbol_rows:
                target, target_error = realized_variance_5d(
                    prices, date.fromisoformat(row["call_date"])
                )
                row["tiingo_attempted"] = "1"
                if target is None:
                    row["tiingo_error"] = target_error or "unknown_target_error"
                    failure_counts[row["tiingo_error"]] += 1
                else:
                    row["target_realized_variance_5d"] = format(target, ".12g")
                    row["target_trading_dates"] = ""
                    row["market_status"] = "PASS"
                    row["market_error"] = ""
                    row["market_provider"] = "Tiingo EOD derived transiently"
                    row["tiingo_error"] = ""
                    recovered += 1

        status = "complete" if recovered > 0 else (fetch_error or "no_data")
        processed_symbols[symbol] = {
            "status": status,
            "calls_considered": len(symbol_rows),
            "calls_recovered": recovered,
            "failure_counts": dict(failure_counts),
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Ham Tiingo fiyat listesi bu noktadan sonra tutulmaz.
        del prices

        write_rows(combined, rows)
        successful = sum(row.get("market_status") == "PASS" for row in rows)
        write_progress(
            progress_path,
            {
                **progress,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_calls": len(rows),
                "successful_calls": successful,
                "coverage": round(successful / len(rows), 6),
                "raw_tiingo_data_persisted": False,
                "derived_target_only": True,
            },
        )
        print(
            f"[{number}/{len(selected_symbols)}] {symbol}: "
            f"{recovered}/{len(symbol_rows)} çağrı kurtarıldı"
        )
        time.sleep(max(0.0, args.sleep))

    successful = sum(row.get("market_status") == "PASS" for row in rows)
    remaining_symbols = {
        tiingo_symbol(row.get("ticker", ""))
        for row in rows
        if row.get("market_status") != "PASS" and row.get("ticker")
    }
    retryable_symbols = {
        symbol
        for symbol in remaining_symbols
        if processed_symbols.get(symbol, {}).get("status")
        not in {"complete", "no_data", "not_found"}
    }
    if not remaining_symbols:
        final_status = "COMPLETE"
    elif not retryable_symbols:
        final_status = "EXHAUSTED_WITH_GAPS"
    else:
        final_status = "IN_PROGRESS"
    final_progress = {
        **progress,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_calls": len(rows),
        "successful_calls": successful,
        "failed_calls": len(rows) - successful,
        "coverage": round(successful / len(rows), 6),
        "remaining_symbols": len(remaining_symbols),
        "retryable_or_unattempted_symbols": len(retryable_symbols),
        "rate_limited": rate_limited,
        "raw_tiingo_data_persisted": False,
        "derived_target_only": True,
        "status": final_status,
    }
    write_rows(combined, rows)
    write_progress(progress_path, final_progress)
    print(json.dumps(final_progress, ensure_ascii=False, indent=2))
    print(f"Birleşik manifest: {combined}")
    print(f"İlerleme raporu: {progress_path}")


if __name__ == "__main__":
    main()