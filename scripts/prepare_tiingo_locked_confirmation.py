"""AHENK V3 için Tiingo tabanlı yeni ve kilitli doğrulama kümesi hazırlar.

Yalnız Yahoo Finance çalışmasında hedefi üretilememiş ve mevcut model veri
indeksinde bulunmayan çağrılar sorgulanır. Seçim protokolü ilk API isteğinden
önce diske yazılır. Tiingo anahtarı, ham fiyatlar ve API yanıtları kaydedilmez;
yalnız beş günlük gerçekleşen varyans ile geçmiş fiyatların toplulaştırılmış
özellikleri kalıcılaştırılır.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


API_ROOT = "https://api.tiingo.com/tiingo/daily"
TARGET_FIELD = "target_realized_variance_5d"
TERMINAL_SYMBOL_STATUSES = {
    "complete",
    "not_found",
    "no_data",
    "no_positive_prices",
    "unexpected_price_response",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def normalize_token(value: str) -> str:
    token = value.strip().strip("'\"")
    lowered = token.lower()
    if lowered.startswith("token "):
        token = token.split(None, 1)[1].strip()
    elif lowered.startswith("token="):
        token = token.split("=", 1)[1].strip()
    return token


def tiingo_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def request_json(url: str, token: str, retries: int = 3) -> tuple[object | None, str | None]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "User-Agent": "P02-AHENK-research/2.0",
        },
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8")), None
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return None, "authentication_or_permission_error"
            if exc.code == 404:
                return None, "not_found"
            if exc.code == 429:
                if attempt == retries:
                    return None, "rate_limited"
                time.sleep(5.0 * attempt)
                continue
            if 500 <= exc.code < 600 and attempt < retries:
                time.sleep(3.0 * attempt)
                continue
            return None, f"http_{exc.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries:
                return None, "network_or_response_error"
            time.sleep(3.0 * attempt)
    return None, "unknown_request_error"


def validate_token(token: str) -> str | None:
    payload, error = request_json(f"{API_ROOT}/AAPL", token, retries=1)
    if error:
        return error
    if not isinstance(payload, dict) or not payload.get("ticker"):
        return "unexpected_metadata_response"
    return None


def fetch_prices(
    symbol: str, start: date, end: date, token: str
) -> tuple[list[tuple[date, float]], str | None]:
    query = urllib.parse.urlencode(
        {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "resampleFreq": "daily",
            "format": "json",
        }
    )
    encoded = urllib.parse.quote(symbol, safe="-")
    payload, error = request_json(f"{API_ROOT}/{encoded}/prices?{query}", token)
    if error:
        return [], error
    if not isinstance(payload, list):
        return [], "unexpected_price_response"

    prices: dict[date, float] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            trading_date = date.fromisoformat(str(item.get("date", ""))[:10])
            price = float(item.get("adjClose"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0:
            prices[trading_date] = price
    ordered = sorted(prices.items())
    return ordered, None if ordered else "no_positive_prices"


def log_returns(prices: list[tuple[date, float]]) -> list[tuple[date, float]]:
    output: list[tuple[date, float]] = []
    for index in range(1, len(prices)):
        output.append((prices[index][0], math.log(prices[index][1] / prices[index - 1][1])))
    return output


def derive_call_values(
    prices: list[tuple[date, float]], call_date: date
) -> tuple[dict[str, object] | None, str | None]:
    returns = log_returns(prices)
    future = [(day, value) for day, value in returns if day > call_date]
    if len(future) < 5:
        return None, "fewer_than_five_post_call_trading_days"

    selected = future[:5]
    output: dict[str, object] = {
        TARGET_FIELD: sum(value * value for _day, value in selected),
        "target_trading_dates": "|".join(day.isoformat() for day, _value in selected),
    }
    past = [value for day, value in returns if day < call_date]
    output["price_history_days"] = len([day for day, _price in prices if day < call_date])
    for window in (5, 20, 60):
        sample = past[-window:]
        prefix = f"price_{window}d"
        output[f"{prefix}_available_returns"] = len(sample)
        if not sample:
            output[f"{prefix}_log_return"] = ""
            output[f"{prefix}_realized_variance"] = ""
            output[f"{prefix}_volatility"] = ""
            output[f"{prefix}_mean_absolute_return"] = ""
            continue
        mean = sum(sample) / len(sample)
        output[f"{prefix}_log_return"] = sum(sample)
        output[f"{prefix}_realized_variance"] = sum(value * value for value in sample)
        output[f"{prefix}_volatility"] = math.sqrt(
            sum((value - mean) ** 2 for value in sample) / len(sample)
        )
        output[f"{prefix}_mean_absolute_return"] = sum(abs(value) for value in sample) / len(sample)
    return output, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=1.2)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "ahenk":
        raise SystemExit('Önce "conda activate ahenk" çalıştırılmalıdır.')
    token = normalize_token(os.environ.get("TIINGO_API_KEY", ""))
    if not token:
        raise SystemExit("TIINGO_API_KEY ortam değişkeni tanımlı değil.")

    token_error = validate_token(token)
    if token_error:
        raise SystemExit(f"Tiingo anahtar ön testi başarısız: {token_error}")
    print("Tiingo anahtarı geçerli; anahtar diske yazılmadı.")
    if args.probe_only:
        return

    root = Path(args.project_root).resolve()
    source = root / "data/manifests/market_full_calls.csv"
    model_index = root / "data/processed/ahenk_dataset_index.csv"
    protocol_path = root / "reports/confirmation/tiingo_confirmation_protocol.json"
    candidates_path = root / "data/confirmation/tiingo_confirmation_candidates.csv"
    progress_path = root / "reports/confirmation/tiingo_confirmation_progress.json"
    if not source.is_file() or not model_index.is_file():
        raise SystemExit("market_full_calls.csv veya ahenk_dataset_index.csv bulunamadı.")

    source_rows = read_csv(source)
    used_ids = {row["call_id"] for row in read_csv(model_index)}
    eligible = [
        row for row in source_rows
        if row.get("market_status") != "PASS"
        and row.get("call_id") not in used_ids
        and row.get("ticker")
    ]
    eligible.sort(key=lambda row: (tiingo_symbol(row["ticker"]), row["call_date"], row["call_id"]))
    eligible_ids = [row["call_id"] for row in eligible]

    protocol = {
        "created_at_utc": utc_now(),
        "stage": "locked_before_tiingo_target_retrieval",
        "selection_rule": "all Yahoo-failed MAEC calls absent from the frozen model index",
        "source_manifest_sha256": sha256(source),
        "frozen_model_index_sha256": sha256(model_index),
        "eligible_calls_before_api_query": len(eligible),
        "eligible_call_ids_sha256": text_sha256(eligible_ids),
        "target": "sum of squared log returns over first five trading days strictly after call date",
        "price_inputs": "aggregates from trading days strictly before call date",
        "provider": "Tiingo EOD adjusted close",
        "raw_prices_persisted": False,
        "human_or_expert_labels_used": False,
        "model_selection_on_confirmation_set_allowed": False,
        "primary_comparison": "frozen AHENK V3 OOF risk gate versus frozen fixed-mean fusion",
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("source_manifest_sha256", "frozen_model_index_sha256", "eligible_call_ids_sha256"):
            if existing.get(key) != protocol[key]:
                raise SystemExit(
                    "Kilitli protokol ile mevcut dosyalar uyuşmuyor. "
                    "Dosyaları değiştirmeden önce inceleme yapın."
                )
        protocol = existing
    else:
        write_json(protocol_path, protocol)

    previous_rows = read_csv(candidates_path) if candidates_path.exists() else []
    recovered_by_id = {row["call_id"]: row for row in previous_rows}
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"processed_symbols": {}}
    )
    processed = progress.setdefault("processed_symbols", {})

    by_symbol: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        symbol = tiingo_symbol(row["ticker"])
        if row["call_id"] in recovered_by_id:
            continue
        if processed.get(symbol, {}).get("status") in TERMINAL_SYMBOL_STATUSES:
            continue
        by_symbol[symbol].append(row)
    symbols = sorted(by_symbol)[: max(0, args.max_symbols)]

    errors: Counter[str] = Counter()
    rate_limited = False
    for number, symbol in enumerate(symbols, start=1):
        rows = by_symbol[symbol]
        call_dates = [date.fromisoformat(row["call_date"]) for row in rows]
        prices, error = fetch_prices(
            symbol,
            min(call_dates) - timedelta(days=150),
            max(call_dates) + timedelta(days=20),
            token,
        )
        if error in {"authentication_or_permission_error", "rate_limited"}:
            processed[symbol] = {"status": error, "attempted_at_utc": utc_now()}
            errors[error] += len(rows)
            rate_limited = error == "rate_limited"
            print(f"[{number}/{len(symbols)}] {symbol}: {error}; çalışma durduruldu")
            break

        recovered = 0
        local_errors: Counter[str] = Counter()
        if error:
            local_errors[error] += len(rows)
        else:
            for row in rows:
                values, target_error = derive_call_values(prices, date.fromisoformat(row["call_date"]))
                if target_error:
                    local_errors[target_error] += 1
                    continue
                assert values is not None
                recovered_by_id[row["call_id"]] = {
                    **row,
                    **{
                        key: format(value, ".12g") if isinstance(value, float) else value
                        for key, value in values.items()
                    },
                    "confirmation_provider": "Tiingo EOD adjusted close",
                    "confirmation_locked": "1",
                }
                recovered += 1

        status = "complete" if recovered else (error or "no_data")
        processed[symbol] = {
            "status": status,
            "calls_considered": len(rows),
            "calls_recovered": recovered,
            "failure_counts": dict(local_errors),
            "attempted_at_utc": utc_now(),
        }
        errors.update(local_errors)
        del prices

        ordered_rows = sorted(recovered_by_id.values(), key=lambda row: (row["call_date"], row["call_id"]))
        if ordered_rows:
            extra_fields = [
                TARGET_FIELD, "target_trading_dates", "price_history_days",
                *[
                    f"price_{window}d_{suffix}"
                    for window in (5, 20, 60)
                    for suffix in (
                        "available_returns", "log_return", "realized_variance",
                        "volatility", "mean_absolute_return",
                    )
                ],
                "confirmation_provider", "confirmation_locked",
            ]
            fields = list(source_rows[0].keys())
            fields.extend(field for field in extra_fields if field not in fields)
            write_csv(candidates_path, ordered_rows, fields)

        report = {
            **progress,
            "updated_at_utc": utc_now(),
            "protocol_sha256": sha256(protocol_path),
            "eligible_calls": len(eligible),
            "recovered_calls": len(recovered_by_id),
            "processed_symbol_count": len(processed),
            "last_run_failure_counts": dict(errors),
            "raw_prices_persisted": False,
            "api_key_persisted": False,
            "human_or_expert_labels_used": False,
            "rate_limited": rate_limited,
            "status": "IN_PROGRESS",
        }
        write_json(progress_path, report)
        print(f"[{number}/{len(symbols)}] {symbol}: {recovered}/{len(rows)} çağrı kurtarıldı")
        time.sleep(max(0.0, args.sleep))

    remaining = len(eligible) - len(recovered_by_id)
    remaining_symbols = {
        tiingo_symbol(row["ticker"])
        for row in eligible
        if row["call_id"] not in recovered_by_id
    }
    retryable_symbols = {
        symbol
        for symbol in remaining_symbols
        if processed.get(symbol, {}).get("status") not in TERMINAL_SYMBOL_STATUSES
    }
    final = {
        **progress,
        "updated_at_utc": utc_now(),
        "protocol_sha256": sha256(protocol_path),
        "eligible_calls": len(eligible),
        "recovered_calls": len(recovered_by_id),
        "remaining_calls_including_unavailable": remaining,
        "remaining_symbols_including_unavailable": len(remaining_symbols),
        "retryable_or_unattempted_symbols": len(retryable_symbols),
        "processed_symbol_count": len(processed),
        "last_run_failure_counts": dict(errors),
        "raw_prices_persisted": False,
        "api_key_persisted": False,
        "human_or_expert_labels_used": False,
        "rate_limited": rate_limited,
        "status": (
            "COMPLETE"
            if not remaining
            else "IN_PROGRESS"
            if retryable_symbols
            else "EXHAUSTED_WITH_GAPS"
        ),
    }
    write_json(progress_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"Kilitli protokol: {protocol_path}")
    print(f"Adaylar: {candidates_path}")
    print(f"İlerleme: {progress_path}")


if __name__ == "__main__":
    main()