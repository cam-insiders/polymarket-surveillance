#!/usr/bin/env python3
"""
Download USDC transfer history for wallets in trades.db.

This script prefetches raw USDC transfer events from Polygonscan and stores
them in a local SQLite database. Backtests can later query locally instead of
hitting the API repeatedly.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv


USDC_CONTRACT_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGONSCAN_API_URL = "https://api.etherscan.io/v2/api?chainid=137"


@dataclass
class WalletRow:
    wallet: str
    first_trade_ts: int
    last_trade_ts: int
    max_usd_amount: float


class RateLimiter:
    def __init__(self, max_requests_per_second: float) -> None:
        if max_requests_per_second <= 0:
            raise ValueError("max_requests_per_second must be > 0")
        self.min_interval = 1.0 / max_requests_per_second
        self._last_request_time = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()


class TransferStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS usdc_transfers (
                tx_hash TEXT NOT NULL,
                block_number INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                from_address TEXT NOT NULL,
                to_address TEXT NOT NULL,
                value_usdc REAL NOT NULL,
                PRIMARY KEY (tx_hash, from_address, to_address)
            );

            CREATE INDEX IF NOT EXISTS idx_usdc_from ON usdc_transfers(from_address);
            CREATE INDEX IF NOT EXISTS idx_usdc_to ON usdc_transfers(to_address);
            CREATE INDEX IF NOT EXISTS idx_usdc_timestamp ON usdc_transfers(timestamp);

            CREATE TABLE IF NOT EXISTS attribution_cache (
                wallet TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                queried_at INTEGER NOT NULL
            );
            """
        )
        self.conn.commit()

    def load_queried_wallets(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT wallet FROM attribution_cache WHERE status = 'queried'"
        ).fetchall()
        return {str(r[0]).lower() for r in rows}

    def insert_transfers(
        self,
        rows: Sequence[Tuple[str, int, int, str, str, float]],
        commit: bool = False,
    ) -> None:
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO usdc_transfers (
                tx_hash, block_number, timestamp, from_address, to_address, value_usdc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        if commit:
            self.conn.commit()

    def set_wallet_status(self, wallet: str, status: str, commit: bool = False) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO attribution_cache (wallet, status, queried_at)
            VALUES (?, ?, ?)
            """,
            (wallet.lower(), status, int(time.time())),
        )
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    print(f"[{ts_now()}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch USDC transfers for wallets in trades.db"
    )
    parser.add_argument("--trades-db", default="data/trades.db", help="Path to trades.db")
    parser.add_argument(
        "--output-db",
        default="data/usdc_transfers.db",
        help="Path to output DB containing usdc_transfers + attribution_cache",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N wallets")
    parser.add_argument("--resume", action="store_true", help="Skip wallets already marked queried")
    parser.add_argument(
        "--min-notional",
        type=float,
        default=0.0,
        help="Only include wallets with max trade usd_amount >= this value",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Include wallets active on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Include wallets active on/before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--wallet-source",
        choices=["maker", "both"],
        default="maker",
        help="Wallet source columns from trades table",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Polygonscan API key (defaults to POLYGONSCAN_API_KEY env var)",
    )
    parser.add_argument("--api-url", default=POLYGONSCAN_API_URL, help="Polygonscan API URL")
    parser.add_argument(
        "--usdc-contract",
        default=USDC_CONTRACT_POLYGON,
        help="USDC contract address on Polygon",
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=4.5,
        help="Maximum API requests per second",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1000,
        help="Polygonscan page size for tokentx pagination",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retry attempts per API request",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show wallet counts; do not call Polygonscan",
    )
    parser.add_argument(
        "--ensure-trades-index",
        action="store_true",
        help="Create wallet/timestamp indexes on trades.db before wallet discovery",
    )
    return parser.parse_args()


def parse_trade_timestamp(value: object) -> Optional[int]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        n = int(text)
        if n > 10**12:
            return n // 1000
        if n > 10**10:
            return n
        return n

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


def parse_ymd_to_epoch(value: Optional[str], end_of_day: bool) -> Optional[int]:
    if not value:
        return None
    dt = datetime.strptime(value, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def wallets_query(wallet_source: str) -> str:
    if wallet_source == "both":
        return """
            WITH wallets AS (
                SELECT LOWER(maker) AS wallet, timestamp, usd_amount
                FROM trades
                WHERE maker IS NOT NULL AND TRIM(maker) != ''
                UNION ALL
                SELECT LOWER(taker) AS wallet, timestamp, usd_amount
                FROM trades
                WHERE taker IS NOT NULL AND TRIM(taker) != ''
            )
            SELECT
                wallet,
                MIN(timestamp) AS first_trade_ts,
                MAX(timestamp) AS last_trade_ts,
                MAX(COALESCE(usd_amount, 0)) AS max_usd_amount
            FROM wallets
            GROUP BY wallet
        """

    return """
        SELECT
            LOWER(maker) AS wallet,
            MIN(timestamp) AS first_trade_ts,
            MAX(timestamp) AS last_trade_ts,
            MAX(COALESCE(usd_amount, 0)) AS max_usd_amount
        FROM trades
        WHERE maker IS NOT NULL AND TRIM(maker) != ''
        GROUP BY LOWER(maker)
    """


def wallets_query_fast_distinct(wallet_source: str) -> str:
    if wallet_source == "both":
        return """
            SELECT wallet FROM (
                SELECT LOWER(maker) AS wallet
                FROM trades
                WHERE maker IS NOT NULL AND TRIM(maker) != ''
                UNION
                SELECT LOWER(taker) AS wallet
                FROM trades
                WHERE taker IS NOT NULL AND TRIM(taker) != ''
            )
            WHERE wallet IS NOT NULL AND TRIM(wallet) != ''
            ORDER BY wallet ASC
        """

    return """
        SELECT DISTINCT LOWER(maker) AS wallet
        FROM trades
        WHERE maker IS NOT NULL AND TRIM(maker) != ''
        ORDER BY wallet ASC
    """


def load_wallets(
    trades_db: str,
    min_notional: float,
    date_from: Optional[int],
    date_to: Optional[int],
    wallet_source: str,
) -> List[WalletRow]:
    conn = sqlite3.connect(trades_db)
    conn.row_factory = sqlite3.Row
    try:
        can_use_fast_path = (min_notional <= 0) and (date_from is None) and (date_to is None)

        if can_use_fast_path:
            rows = conn.execute(wallets_query_fast_distinct(wallet_source)).fetchall()
            out: List[WalletRow] = []
            for row in rows:
                wallet = str(row["wallet"]).lower()
                if not wallet:
                    continue
                out.append(
                    WalletRow(
                        wallet=wallet,
                        first_trade_ts=0,
                        last_trade_ts=0,
                        max_usd_amount=0.0,
                    )
                )
            return out

        rows = conn.execute(wallets_query(wallet_source)).fetchall()
    finally:
        conn.close()

    out: List[WalletRow] = []
    for row in rows:
        wallet = str(row["wallet"]).lower()
        first_ts = parse_trade_timestamp(row["first_trade_ts"])
        last_ts = parse_trade_timestamp(row["last_trade_ts"])
        max_usd = float(row["max_usd_amount"] or 0.0)

        if not wallet or first_ts is None or last_ts is None:
            continue
        if min_notional > 0 and max_usd < min_notional:
            continue

        # Keep wallets with any trade activity overlap in [date_from, date_to].
        if date_from is not None and last_ts < date_from:
            continue
        if date_to is not None and first_ts > date_to:
            continue

        out.append(
            WalletRow(
                wallet=wallet,
                first_trade_ts=first_ts,
                last_trade_ts=last_ts,
                max_usd_amount=max_usd,
            )
        )

    out.sort(key=lambda w: (w.first_trade_ts, w.wallet))
    return out


def ensure_trades_indexes(trades_db: str, wallet_source: str) -> None:
    conn = sqlite3.connect(trades_db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")

        existing = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='trades'"
            ).fetchall()
        }

        statements: List[Tuple[str, str]] = []

        if "idx_trades_timestamp" not in existing:
            statements.append(("idx_trades_timestamp", "CREATE INDEX idx_trades_timestamp ON trades(timestamp)"))

        if wallet_source == "maker":
            if "idx_trades_maker" not in existing:
                statements.append(("idx_trades_maker", "CREATE INDEX idx_trades_maker ON trades(maker)"))
        else:
            if "idx_trades_maker" not in existing:
                statements.append(("idx_trades_maker", "CREATE INDEX idx_trades_maker ON trades(maker)"))
            if "idx_trades_taker" not in existing:
                statements.append(("idx_trades_taker", "CREATE INDEX idx_trades_taker ON trades(taker)"))

        if not statements:
            log("Trades indexes already exist; skipping index creation")
            return

        for name, stmt in statements:
            log(f"Creating index {name}...")
            t0 = time.time()
            conn.execute(stmt)
            conn.commit()
            log(f"Created index {name} in {format_eta(time.time() - t0)}")
    finally:
        conn.close()


def request_tokentx(
    session: requests.Session,
    rate_limiter: RateLimiter,
    api_url: str,
    params: Dict[str, object],
    max_attempts: int,
) -> Optional[List[Dict]]:
    for attempt in range(1, max_attempts + 1):
        try:
            rate_limiter.wait()
            resp = session.get(api_url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            status = str(payload.get("status", ""))
            message = str(payload.get("message", ""))
            result = payload.get("result", [])

            if status == "1":
                if isinstance(result, list):
                    return result
                return None

            if status == "0" and "No transactions found" in message:
                return []

            # Handle temporary API-side issues with retry.
            if attempt < max_attempts:
                time.sleep(2**attempt)
                continue
            return None
        except (requests.RequestException, ValueError):
            if attempt < max_attempts:
                time.sleep(2**attempt)
                continue
            return None
    return None


def normalize_transfer(raw: Dict) -> Optional[Tuple[str, int, int, str, str, float]]:
    try:
        tx_hash = str(raw.get("hash", "")).strip().lower()
        from_addr = str(raw.get("from", "")).strip().lower()
        to_addr = str(raw.get("to", "")).strip().lower()
        block_number = int(str(raw.get("blockNumber", "0")))
        timestamp = int(str(raw.get("timeStamp", "0")))
        value_int = int(str(raw.get("value", "0")))
        decimals = int(str(raw.get("tokenDecimal", "6")))

        if not tx_hash or not from_addr or not to_addr:
            return None
        if block_number <= 0 or timestamp <= 0:
            return None

        value_usdc = value_int / float(10**decimals)
        return (tx_hash, block_number, timestamp, from_addr, to_addr, value_usdc)
    except (TypeError, ValueError, OverflowError):
        return None


def fetch_and_store_wallet(
    wallet: str,
    store: TransferStore,
    session: requests.Session,
    rate_limiter: RateLimiter,
    api_url: str,
    api_key: str,
    usdc_contract: str,
    offset: int,
    max_attempts: int,
) -> Tuple[bool, int, int, int]:
    total_insert_candidates = 0
    api_calls = 0
    pages = 0
    start_block = 0
    end_block = 99_999_999

    while True:
        page = 1
        window_hit_cap = False
        last_block_seen = start_block
        any_rows_this_window = False

        while True:
            params: Dict[str, object] = {
                "module": "account",
                "action": "tokentx",
                "contractaddress": usdc_contract,
                "address": wallet,
                "startblock": start_block,
                "endblock": end_block,
                "page": page,
                "offset": offset,
                "sort": "asc",
                "apikey": api_key,
            }

            rows = request_tokentx(
                session=session,
                rate_limiter=rate_limiter,
                api_url=api_url,
                params=params,
                max_attempts=max_attempts,
            )
            api_calls += 1

            if rows is None:
                return False, total_insert_candidates, api_calls, pages

            if not rows:
                break

            any_rows_this_window = True
            pages += 1
            normalized_rows: List[Tuple[str, int, int, str, str, float]] = []
            for raw in rows:
                transfer_row = normalize_transfer(raw)
                if transfer_row is None:
                    continue
                normalized_rows.append(transfer_row)
                last_block_seen = max(last_block_seen, transfer_row[1])

            store.insert_transfers(normalized_rows, commit=False)
            total_insert_candidates += len(normalized_rows)

            # Keep paging normally.
            if len(rows) < offset:
                break

            # Defend against providers that cap to ~10k results per block range.
            if page * offset >= 10_000:
                window_hit_cap = True
                break

            page += 1

        if window_hit_cap:
            next_start = last_block_seen + 1
            if next_start <= start_block:
                next_start = start_block + 1
            start_block = next_start
            continue

        if not any_rows_this_window:
            break

        # Completed window without cap, so done for this wallet.
        break

    return True, total_insert_candidates, api_calls, pages


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = args.api_key or os.getenv("POLYGONSCAN_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set POLYGONSCAN_API_KEY or pass --api-key")

    date_from = parse_ymd_to_epoch(args.date_from, end_of_day=False)
    date_to = parse_ymd_to_epoch(args.date_to, end_of_day=True)
    if date_from is not None and date_to is not None and date_from > date_to:
        raise SystemExit("--date-from must be <= --date-to")

    if args.ensure_trades_index:
        log("Ensuring trades.db indexes for wallet discovery...")
        index_start = time.time()
        ensure_trades_indexes(args.trades_db, args.wallet_source)
        log(f"Finished index setup in {format_eta(time.time() - index_start)}")

    wallet_load_start = time.time()
    wallets = load_wallets(
        trades_db=args.trades_db,
        min_notional=float(args.min_notional or 0.0),
        date_from=date_from,
        date_to=date_to,
        wallet_source=args.wallet_source,
    )
    wallet_load_elapsed = time.time() - wallet_load_start

    store = TransferStore(args.output_db)
    try:
        if args.resume:
            queried_wallets = store.load_queried_wallets()
            wallets = [w for w in wallets if w.wallet not in queried_wallets]

        if args.limit is not None and args.limit > 0:
            wallets = wallets[: args.limit]

        log(f"Loaded wallets to process: {len(wallets):,}")
        log(f"Wallet discovery time: {format_eta(wallet_load_elapsed)}")
        log(f"Trades DB:  {args.trades_db}")
        log(f"Output DB:  {args.output_db}")
        log(f"Wallet source: {args.wallet_source}")
        log(f"Resume mode: {args.resume}")
        log(f"Min notional: ${float(args.min_notional):,.2f}")

        if args.dry_run:
            log("Dry run complete. No API calls were made.")
            return

        limiter = RateLimiter(args.max_rps)
        session = requests.Session()

        start_time = time.time()
        ok_count = 0
        fail_count = 0
        transfer_candidates = 0

        for idx, wallet_row in enumerate(wallets, start=1):
            ok, inserted, api_calls, pages = fetch_and_store_wallet(
                wallet=wallet_row.wallet,
                store=store,
                session=session,
                rate_limiter=limiter,
                api_url=args.api_url,
                api_key=api_key,
                usdc_contract=args.usdc_contract,
                offset=args.offset,
                max_attempts=args.max_attempts,
            )

            if ok:
                ok_count += 1
                transfer_candidates += inserted
                store.set_wallet_status(wallet_row.wallet, "queried", commit=False)
                store.commit()
            else:
                fail_count += 1
                store.set_wallet_status(wallet_row.wallet, "failed", commit=False)
                store.commit()

            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = len(wallets) - idx
            eta_seconds = (remaining / rate) if rate > 0 else 0.0

            if idx == 1 or idx % 10 == 0 or idx == len(wallets):
                log(
                    f"[{idx}/{len(wallets)}] wallet={wallet_row.wallet[:12]}... "
                    f"status={'ok' if ok else 'failed'} rows={inserted:,} "
                    f"calls={api_calls} pages={pages} "
                    f"ok={ok_count:,} failed={fail_count:,} "
                    f"eta={format_eta(eta_seconds)}"
                )

        total_time = time.time() - start_time
        log("Done.")
        log(f"Wallets queried: {ok_count:,}")
        log(f"Wallets failed:  {fail_count:,}")
        log(f"Transfer rows inserted/seen: {transfer_candidates:,}")
        log(f"Elapsed: {format_eta(total_time)}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
