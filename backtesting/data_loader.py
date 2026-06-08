"""
Historical data loader for backtesting.
Loads warproxxx's poly_data using SQLite for fast queries.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models import TradeBatch


class HistoricalDataLoader:
    """
    Loads and processes historical Polymarket data
    """

    def __init__(
        self,
        data_dir: str = "data",
        cache_size: int = 0,
        sqlite_page_cache_kb: Optional[int] = None,
        sqlite_mmap_size: Optional[int] = None,
        trade_start_time: Optional[Any] = None,
        trade_end_time: Optional[Any] = None,
        cap_trades_at_market_close: Optional[bool] = None,
    ):
        self.data_dir = data_dir
        self.cache_size = max(0, int(cache_size))
        self.sqlite_page_cache_kb = int(
            sqlite_page_cache_kb
            if sqlite_page_cache_kb is not None
            else os.environ.get("POLYMARKET_SQLITE_CACHE_KB", "64000")
        )
        self.sqlite_mmap_size = int(
            sqlite_mmap_size
            if sqlite_mmap_size is not None
            else os.environ.get("POLYMARKET_SQLITE_MMAP_SIZE", "0")
        )
        self.markets_df: Optional[pd.DataFrame] = None
        self._market_lookup: Dict[int, Dict] = {}

        self.db_path = f"{data_dir}/trades.db"
        self.use_sqlite = Path(self.db_path).exists()
        self.trades_path = f"{data_dir}/processed/trades.csv"

        self._conn: Optional[sqlite3.Connection] = None
        self._trade_cache: OrderedDict[Tuple[int, Optional[float], Optional[int], Optional[int], bool], TradeBatch] = OrderedDict()

        env_start = os.environ.get("POLYMARKET_TRADE_START_TIME")
        env_end = os.environ.get("POLYMARKET_TRADE_END_TIME")
        env_cap_close = os.environ.get("POLYMARKET_TRADE_END_AT_MARKET_CLOSE")
        self.trade_start_ms, self.trade_start_sql = self._normalize_trade_time_bound(
            trade_start_time if trade_start_time is not None else env_start,
            upper_bound=False,
        )
        self.trade_end_ms, self.trade_end_sql = self._normalize_trade_time_bound(
            trade_end_time if trade_end_time is not None else env_end,
            upper_bound=True,
        )
        self.cap_trades_at_market_close = (
            str(env_cap_close).strip().lower() in {"1", "true", "yes", "on"}
            if cap_trades_at_market_close is None
            else bool(cap_trades_at_market_close)
        )
        if (
            self.trade_start_ms is not None
            and self.trade_end_ms is not None
            and self.trade_start_ms > self.trade_end_ms
        ):
            raise ValueError("trade_start_time must be <= trade_end_time")

        if not self.use_sqlite:
            logging.warning(f"SQLite database not found at {self.db_path}")
            logging.warning("Falling back to CSV mode (slow). Run convert_to_sqlite.py first.")
        else:
            logging.info(f"Using SQLite database: {self.db_path}")

    def load_data(self):
        """Load markets metadata."""
        logging.info("Loading market metadata...")
        markets_path = f"{self.data_dir}/markets.csv"
        self.markets_df = pd.read_csv(markets_path)
        logging.info(f"Loaded {len(self.markets_df):,} markets")

        self._market_lookup.clear()
        for _, row in self.markets_df.iterrows():
            self._market_lookup[int(row["id"])] = {
                "condition_id": row["condition_id"],
                "market_slug": row["market_slug"],
                "token1": str(row["token1"]),
                "token2": str(row["token2"]),
                "answer1": row["answer1"],
                "answer2": row["answer2"],
                "volume": row["volume"],
                "closed_time": row["closedTime"],
            }

        if self.use_sqlite:
            logging.info("SQLite database ready.")
        else:
            logging.info("CSV mode active. Consider converting to SQLite for speed.")

    def close(self):
        """Close open DB connections."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path)
            # Read-path tuning. Safe for read-only workloads; none affect correctness.
            # Defaults are RAM-conservative for optimizer workers; use
            # POLYMARKET_SQLITE_CACHE_KB / POLYMARKET_SQLITE_MMAP_SIZE to opt into
            # larger speed-oriented settings for a single-process run.
            pragmas = [
                "PRAGMA query_only = 1",          # Defensive: accidental writes error loudly.
                "PRAGMA journal_mode = WAL",      # No-op if already WAL.
                "PRAGMA synchronous = NORMAL",    # Write-only tuning; harmless for reads.
                "PRAGMA temp_store = MEMORY",     # Keep sorts/temp b-trees in RAM.
                f"PRAGMA cache_size = -{max(1, self.sqlite_page_cache_kb)}",
            ]
            if self.sqlite_mmap_size > 0:
                pragmas.append(f"PRAGMA mmap_size = {self.sqlite_mmap_size}")
            for pragma in pragmas:
                conn.execute(pragma)
            self._conn = conn
        return self._conn

    def set_trade_time_bounds(
        self,
        trade_start_time: Optional[Any] = None,
        trade_end_time: Optional[Any] = None,
        cap_trades_at_market_close: Optional[bool] = None,
    ) -> None:
        """Restrict subsequent trade loads to an inclusive UTC timestamp window."""
        start_ms, start_sql = self._normalize_trade_time_bound(
            trade_start_time,
            upper_bound=False,
        )
        end_ms, end_sql = self._normalize_trade_time_bound(
            trade_end_time,
            upper_bound=True,
        )
        if start_ms is not None and end_ms is not None and start_ms > end_ms:
            raise ValueError("trade_start_time must be <= trade_end_time")
        self.trade_start_ms = start_ms
        self.trade_start_sql = start_sql
        self.trade_end_ms = end_ms
        self.trade_end_sql = end_sql
        if cap_trades_at_market_close is not None:
            self.cap_trades_at_market_close = bool(cap_trades_at_market_close)
        self._trade_cache.clear()

    @staticmethod
    def _normalize_trade_time_bound(
        value,
        *,
        upper_bound: bool = False,
    ) -> Tuple[Optional[int], Optional[str]]:
        if value is None:
            return None, None
        text = str(value).strip()
        if not text:
            return None, None

        if text.isdigit():
            raw = int(text)
            ms = raw * 1000 if raw < 10_000_000_000 else raw
            dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        else:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            ms = int(dt.timestamp() * 1000)

        # SQLite compares the source timestamp text lexicographically
        sql_text = dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        if upper_bound:
            sql_text += "Z"
        return ms, sql_text

    def _effective_trade_time_bounds_for_market(
        self,
        market_id: int,
        *,
        ignore_trade_time_bounds: bool = False,
    ) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
        if ignore_trade_time_bounds:
            return None, None, None, None

        start_ms = self.trade_start_ms
        start_sql = self.trade_start_sql
        end_ms = self.trade_end_ms
        end_sql = self.trade_end_sql

        if self.cap_trades_at_market_close:
            market = self._market_lookup.get(int(market_id)) or {}
            close_raw = market.get("closed_time", market.get("closedTime"))
            close_ms, close_sql = self._normalize_trade_time_bound(
                close_raw,
                upper_bound=True,
            )
            if close_ms is not None and (end_ms is None or close_ms < end_ms):
                end_ms = close_ms
                end_sql = close_sql

        return start_ms, start_sql, end_ms, end_sql

    def _cache_get(self, key: Tuple[int, Optional[float], Optional[int], Optional[int], bool]) -> Optional[TradeBatch]:
        if self.cache_size <= 0:
            return None
        trades = self._trade_cache.get(key)
        if trades is not None:
            self._trade_cache.move_to_end(key)
        return trades

    def _cache_put(self, key: Tuple[int, Optional[float], Optional[int], Optional[int], bool], trades: TradeBatch):
        if self.cache_size <= 0:
            return
        self._trade_cache[key] = trades
        self._trade_cache.move_to_end(key)
        while len(self._trade_cache) > self.cache_size:
            self._trade_cache.popitem(last=False)

    def get_market_metadata(self, market_id: int) -> Optional[Dict]:
        return self._market_lookup.get(int(market_id))

    def get_markets_by_volume(self, min_volume: float = 10000, limit: Optional[int] = None) -> List[int]:
        if self.markets_df is None:
            raise RuntimeError("Call load_data() first.")
        filtered = self.markets_df[self.markets_df["volume"] >= min_volume]
        sorted_markets = filtered.sort_values("volume", ascending=False)
        market_ids = [int(x) for x in sorted_markets["id"].tolist()]
        if limit is not None:
            market_ids = market_ids[:limit]
        return market_ids

    def get_trade_count(
        self,
        market_id: int,
        min_usd_amount: Optional[float] = None,
    ) -> int:
        market_id = int(market_id)
        _start_ms, start_sql, _end_ms, end_sql = self._effective_trade_time_bounds_for_market(market_id)
        if self.use_sqlite:
            conn = self._get_conn()
            params: List[Any] = [market_id]
            if min_usd_amount is None:
                query = "SELECT COUNT(*) FROM trades WHERE market_id = ?"
            else:
                query = "SELECT COUNT(*) FROM trades WHERE market_id = ? AND usd_amount >= ?"
                params.append(float(min_usd_amount))
            if start_sql is not None:
                query += " AND timestamp >= ?"
                params.append(start_sql)
            if end_sql is not None:
                query += " AND timestamp <= ?"
                params.append(end_sql)
            cur = conn.execute(query, tuple(params))
            base_count = int(cur.fetchone()[0])
            return base_count

        # CSV fallback
        total = 0
        for chunk in pd.read_csv(self.trades_path, chunksize=200_000):
            market_rows = chunk[chunk["market_id"] == market_id]
            if min_usd_amount is not None:
                market_rows = market_rows[market_rows["usd_amount"] >= float(min_usd_amount)]
            if _start_ms is not None or _end_ms is not None:
                ts_ms = (
                    pd.to_datetime(market_rows["timestamp"], utc=True)
                    .dt.tz_convert("UTC")
                    .dt.tz_localize(None)
                    .astype("datetime64[ms]")
                    .astype("int64")
                )
                if _start_ms is not None:
                    market_rows = market_rows[ts_ms >= _start_ms]
                    ts_ms = ts_ms[ts_ms >= _start_ms]
                if _end_ms is not None:
                    market_rows = market_rows[ts_ms <= _end_ms]
            total += len(market_rows)
        return total

    def get_trades_for_market(
        self,
        market_id: int,
        min_usd_amount: Optional[float] = None,
        use_cache: bool = True,
        ignore_trade_time_bounds: bool = False,
    ) -> TradeBatch:
        """
        Load trades for a specific market as a columnar wallet-side batch.
        """
        market_id = int(market_id)
        start_ms, _start_sql, end_ms, _end_sql = self._effective_trade_time_bounds_for_market(
            market_id,
            ignore_trade_time_bounds=ignore_trade_time_bounds,
        )
        key = (
            market_id,
            None if min_usd_amount is None else float(min_usd_amount),
            start_ms,
            end_ms,
            False if ignore_trade_time_bounds else self.cap_trades_at_market_close,
        )

        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        if self.use_sqlite:
            trades = self._get_trades_sqlite(
                market_id,
                min_usd_amount=min_usd_amount,
                ignore_trade_time_bounds=ignore_trade_time_bounds,
            )
        else:
            trades = self._get_trades_csv(
                market_id,
                min_usd_amount=min_usd_amount,
                ignore_trade_time_bounds=ignore_trade_time_bounds,
            )

        if use_cache:
            self._cache_put(key, trades)

        return trades

    @staticmethod
    def _timestamp_to_ms(value) -> int:
        if isinstance(value, (int, np.integer)):
            return int(value) * 1000 if int(value) < 10_000_000_000 else int(value)
        if isinstance(value, float):
            as_int = int(value)
            return as_int * 1000 if as_int < 10_000_000_000 else as_int

        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp() * 1000)

    def _get_trades_sqlite(
        self,
        market_id: int,
        min_usd_amount: Optional[float],
        *,
        ignore_trade_time_bounds: bool = False,
    ) -> TradeBatch:
        market = self._market_lookup.get(market_id)
        if not market:
            logging.error(f"Market metadata not found for market_id={market_id}")
            return TradeBatch.empty({})

        conn = self._get_conn()
        params: List[Any] = [market_id]
        _start_ms, start_sql, _end_ms, end_sql = self._effective_trade_time_bounds_for_market(
            market_id,
            ignore_trade_time_bounds=ignore_trade_time_bounds,
        )

        query = """
            SELECT timestamp, market_id, maker, taker, nonusdc_side,
                   maker_direction, taker_direction, price, usd_amount,
                   token_amount, transactionHash
            FROM trades
            WHERE market_id = ?
        """
        if min_usd_amount is not None:
            query += " AND usd_amount >= ?"
            params.append(float(min_usd_amount))
        if start_sql is not None:
            query += " AND timestamp >= ?"
            params.append(start_sql)
        if end_sql is not None:
            query += " AND timestamp <= ?"
            params.append(end_sql)
        query += " ORDER BY timestamp ASC"

        cur = conn.execute(query, tuple(params))
        batches: List[TradeBatch] = []
        fetch_size = 100_000

        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break

            n = len(rows)
            wallets: List[str] = [None] * n  # type: ignore[list-item]
            sides: List[str] = [None] * n  # type: ignore[list-item]
            tx_hashes: List[str] = [None] * n  # type: ignore[list-item]
            outcome_idx = np.empty(n, dtype=np.int8)
            size_tokens = np.empty(n, dtype=np.float64)
            prices = np.empty(n, dtype=np.float64)
            notionals = np.empty(n, dtype=np.float64)
            timestamps_ms = np.empty(n, dtype=np.int64)

            for i, row in enumerate(rows):
                (
                    timestamp,
                    _row_market_id,
                    maker,
                    _taker,
                    nonusdc_side,
                    maker_direction,
                    _taker_direction,
                    price,
                    usd_amount,
                    token_amount,
                    tx_hash,
                ) = row
                wallets[i] = str(maker)
                sides[i] = str(maker_direction)
                tx_hashes[i] = str(tx_hash)
                outcome_idx[i] = 0 if str(nonusdc_side) == "token1" else 1
                size_tokens[i] = float(token_amount)
                prices[i] = float(price)
                notionals[i] = float(usd_amount)
                timestamps_ms[i] = self._timestamp_to_ms(timestamp)

            batches.append(
                TradeBatch.from_columns(
                    wallets=wallets,
                    sides=sides,
                    outcome_index=outcome_idx,
                    size_tokens=size_tokens,
                    price=prices,
                    notional_usdc=notionals,
                    timestamp_ms=timestamps_ms,
                    tx_hashes=tx_hashes,
                    condition_id=market["condition_id"],
                    market_slug=market["market_slug"],
                    outcomes=(market["answer1"], market["answer2"]),
                    assets=(market["token1"], market["token2"]),
                )
            )

        if not batches:
            return TradeBatch.empty(market)

        return TradeBatch.concat(batches)

    def _get_trades_csv(
        self,
        market_id: int,
        min_usd_amount: Optional[float],
        *,
        ignore_trade_time_bounds: bool = False,
    ) -> TradeBatch:
        market = self._market_lookup.get(market_id)
        if not market:
            logging.error(f"Market metadata not found for market_id={market_id}")
            return TradeBatch.empty({})

        batches: List[TradeBatch] = []
        chunk_size = 200_000
        start_ms, _start_sql, end_ms, _end_sql = self._effective_trade_time_bounds_for_market(
            market_id,
            ignore_trade_time_bounds=ignore_trade_time_bounds,
        )

        try:
            for chunk in pd.read_csv(self.trades_path, chunksize=chunk_size):
                market_trades = chunk[chunk["market_id"] == market_id]
                if min_usd_amount is not None:
                    market_trades = market_trades[market_trades["usd_amount"] >= float(min_usd_amount)]
                if start_ms is not None or end_ms is not None:
                    ts_ms = (
                        pd.to_datetime(market_trades["timestamp"], utc=True)
                        .dt.tz_convert("UTC")
                        .dt.tz_localize(None)
                        .astype("datetime64[ms]")
                        .astype("int64")
                    )
                    if start_ms is not None:
                        market_trades = market_trades[ts_ms >= start_ms]
                        ts_ms = ts_ms[ts_ms >= start_ms]
                    if end_ms is not None:
                        market_trades = market_trades[ts_ms <= end_ms]
                if market_trades.empty:
                    continue
                batches.append(self._dataframe_to_trade_batch(market_trades, market))
        except Exception as exc:
            logging.error(f"Error reading CSV trades: {exc}")
            return TradeBatch.empty(market)

        if not batches:
            return TradeBatch.empty(market)
        return TradeBatch.concat(batches).sort_by_timestamp()

    def _dataframe_to_trade_batch(self, df: pd.DataFrame, market: Dict) -> TradeBatch:
        """
        Convert fills DataFrame to wallet-side trades.
        """
        is_token1 = (df["nonusdc_side"] == "token1").to_numpy()
        timestamps_ms = (
            pd.to_datetime(df["timestamp"], utc=True)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ms]")
            .astype("int64")
        ).to_numpy()

        makers = df["maker"].to_numpy()
        maker_dirs = df["maker_direction"].to_numpy()
        token_amounts = df["token_amount"].to_numpy()
        prices = df["price"].to_numpy()
        usd_amounts = df["usd_amount"].to_numpy()
        tx_hashes = df["transactionHash"].to_numpy()

        condition_id = market["condition_id"]
        market_slug = market["market_slug"]
        answer1 = market["answer1"]
        answer2 = market["answer2"]
        token1 = market["token1"]
        token2 = market["token2"]

        outcome_idx = np.where(is_token1, 0, 1).astype(np.int8)

        return TradeBatch.from_columns(
            wallets=makers,
            sides=maker_dirs,
            outcome_index=outcome_idx,
            size_tokens=token_amounts,
            price=prices,
            notional_usdc=usd_amounts,
            timestamp_ms=timestamps_ms,
            tx_hashes=tx_hashes,
            condition_id=condition_id,
            market_slug=market_slug,
            outcomes=(answer1, answer2),
            assets=(token1, token2),
        )
