"""
USDC Transfer Provider with local caching.

Provides USDC transfer data for attribution analysis. Uses a cache-first
approach: checks local SQLite database first, falls back to Polygonscan
API if not cached, then stores the result for future use.

This allows backtesting to replicate live behavior (lazy fetching) while
building up a local cache over time to avoid repeated API calls.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from backtesting.logging_utils import experiment_backtest_logs_quiet
from clustering.polygonscan_client import PolygonscanClient


logger = logging.getLogger(__name__)


class UsdcTransferProvider:
    """
    Cache-first USDC transfer provider for attribution analysis.
    """

    def __init__(
        self,
        cache_db_path: str,
        polygonscan_config: Optional[Dict] = None,
    ):
        """
        Initialize the transfer provider.
        """
        self.cache_db_path = cache_db_path
        self._ensure_cache_db_exists()
        
        self.conn = sqlite3.connect(cache_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize Polygonscan client if config provided
        if polygonscan_config and polygonscan_config.get("api_key"):
            self.polygonscan_client = PolygonscanClient(polygonscan_config)
            if not experiment_backtest_logs_quiet():
                logger.info(
                    f"UsdcTransferProvider initialized with Polygonscan API "
                    f"(cache: {cache_db_path})"
                )
        else:
            self.polygonscan_client = None
            if not experiment_backtest_logs_quiet():
                logger.info(
                    f"UsdcTransferProvider initialized in cache-only mode "
                    f"(cache: {cache_db_path})"
                )
        
        # Track statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_fetches": 0,
            "api_failures": 0,
        }

    def _ensure_cache_db_exists(self):
        """Create the cache database and tables if they don't exist."""
        Path(self.cache_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.cache_db_path)
        conn.executescript("""
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
        """)
        conn.commit()
        conn.close()

    def is_wallet_cached(self, wallet: str) -> bool:
        """Check if a wallet's transfers have been fetched and cached."""
        wallet = wallet.lower()
        cursor = self.conn.execute(
            "SELECT status FROM attribution_cache WHERE wallet = ?",
            (wallet,)
        )
        row = cursor.fetchone()
        return row is not None and row["status"] == "queried"

    def get_transfers_for_wallet(
        self,
        wallet: str,
        fetch_if_missing: bool = True,
    ) -> List[Dict]:
        """
        Get USDC transfers for a wallet.
        """
        wallet = wallet.lower()
        
        # Check if already cached
        if self.is_wallet_cached(wallet):
            self.stats["cache_hits"] += 1
            return self._get_transfers_from_cache(wallet)
        
        self.stats["cache_misses"] += 1
        
        # Fetch from API if allowed and client is configured
        if fetch_if_missing and self.polygonscan_client is not None:
            return self._fetch_and_cache(wallet)
        
        return []

    def _get_transfers_from_cache(self, wallet: str) -> List[Dict]:
        """Get transfers from local cache for a wallet."""
        cursor = self.conn.execute(
            """
            SELECT tx_hash, block_number, timestamp, from_address, to_address, value_usdc
            FROM usdc_transfers
            WHERE from_address = ? OR to_address = ?
            ORDER BY timestamp ASC
            """,
            (wallet, wallet)
        )
        
        return [
            {
                "tx_hash": row["tx_hash"],
                "block_number": row["block_number"],
                "timestamp": row["timestamp"],
                "from_address": row["from_address"],
                "to_address": row["to_address"],
                "value_usdc": row["value_usdc"],
            }
            for row in cursor
        ]

    def _fetch_and_cache(self, wallet: str) -> List[Dict]:
        """Fetch transfers from Polygonscan and store in cache."""
        self.stats["api_fetches"] += 1
        
        logging.debug(f"Fetching USDC transfers from Polygonscan for {wallet[:10]}...")
        
        raw_transfers = self.polygonscan_client.get_usdc_transfers(wallet)
        
        if raw_transfers is None:
            # API failure
            self.stats["api_failures"] += 1
            self._update_cache_status(wallet, "failed")
            logging.warning(f"Polygonscan API failed for wallet {wallet[:10]}...")
            return []
        
        # Parse and store transfers
        parsed_transfers = []
        rows_to_insert = []
        
        for tx in raw_transfers:
            try:
                tx_hash = str(tx.get("hash", "")).lower()
                block_number = int(tx.get("blockNumber", 0))
                timestamp = int(tx.get("timeStamp", 0))
                from_addr = str(tx.get("from", "")).lower()
                to_addr = str(tx.get("to", "")).lower()
                value_wei = int(tx.get("value", 0))
                decimals = int(tx.get("tokenDecimal", 6))
                value_usdc = value_wei / (10 ** decimals)
                
                if tx_hash and from_addr and to_addr and timestamp > 0:
                    rows_to_insert.append((
                        tx_hash, block_number, timestamp,
                        from_addr, to_addr, value_usdc
                    ))
                    parsed_transfers.append({
                        "tx_hash": tx_hash,
                        "block_number": block_number,
                        "timestamp": timestamp,
                        "from_address": from_addr,
                        "to_address": to_addr,
                        "value_usdc": value_usdc,
                    })
            except (ValueError, TypeError, KeyError) as e:
                logging.debug(f"Failed to parse transfer: {e}")
                continue
        
        # Insert into cache
        if rows_to_insert:
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO usdc_transfers
                (tx_hash, block_number, timestamp, from_address, to_address, value_usdc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert
            )
            if not experiment_backtest_logs_quiet():
                logger.info(
                    f"Cached {len(rows_to_insert):,} transfer rows for wallet {wallet[:10]}..."
                )
        
        self._update_cache_status(wallet, "queried")
        self.conn.commit()
        
        logging.debug(
            f"Cached {len(rows_to_insert)} transfers for wallet {wallet[:10]}..."
        )
        
        return parsed_transfers

    def _update_cache_status(self, wallet: str, status: str):
        """Update the cache status for a wallet."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO attribution_cache (wallet, status, queried_at)
            VALUES (?, ?, ?)
            """,
            (wallet.lower(), status, int(time.time()))
        )

    def ensure_wallets_cached(
        self,
        wallets: Set[str],
        fetch_if_missing: bool = True,
    ) -> int:
        """
        Ensure all wallets in the set have their transfers cached.
        """
        fetched_count = 0
        
        for wallet in wallets:
            wallet = wallet.lower()
            if not self.is_wallet_cached(wallet):
                if fetch_if_missing and self.polygonscan_client is not None:
                    self._fetch_and_cache(wallet)
                    fetched_count += 1

        if fetched_count > 0 and not experiment_backtest_logs_quiet():
            logger.info(f"Fetched and cached transfers for {fetched_count:,} wallet(s)")
        
        return fetched_count

    def get_cache_counts(self) -> Dict[str, int]:
        """Return current cache table counts for quick verification."""
        c1 = self.conn.execute("SELECT COUNT(*) AS n FROM attribution_cache").fetchone()["n"]
        c2 = self.conn.execute("SELECT COUNT(*) AS n FROM usdc_transfers").fetchone()["n"]
        return {
            "cached_wallet_rows": int(c1 or 0),
            "cached_transfer_rows": int(c2 or 0),
        }

    def get_intra_cluster_edges(
        self,
        wallets: Set[str],
        before_timestamp: Optional[int] = None,
    ) -> List[Tuple[str, str, float, int]]:
        """
        Get aggregated USDC transfer edges between wallets in a cluster.
        """
        if not wallets or len(wallets) < 2:
            return []
        
        wallets_lower = [w.lower() for w in wallets]
        placeholders = ",".join("?" * len(wallets_lower))
        
        query = f"""
            SELECT 
                from_address,
                to_address,
                SUM(value_usdc) as total_amount,
                COUNT(*) as tx_count
            FROM usdc_transfers
            WHERE from_address IN ({placeholders})
              AND to_address IN ({placeholders})
              AND from_address != to_address
        """
        params = wallets_lower + wallets_lower
        
        if before_timestamp is not None:
            query += " AND timestamp < ?"
            params.append(before_timestamp)
        
        query += " GROUP BY from_address, to_address"
        
        cursor = self.conn.execute(query, params)
        
        return [
            (row["from_address"], row["to_address"], row["total_amount"], row["tx_count"])
            for row in cursor
        ]

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return dict(self.stats)

    def close(self):
        """Close the database connection."""
        self.conn.close()
