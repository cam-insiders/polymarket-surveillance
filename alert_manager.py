"""Alert management, wallet stats tracking, and database storage"""

import logging
import sqlite3

from detectors.base import DetectionContext
from models import Alert, Trade


class AlertManager:
    """Manages flagged trades, wallet-level stats, and database storage"""

    def __init__(self, db_path: str, context: DetectionContext):
        self.db_path = db_path
        self.context = context
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                market_slug TEXT,
                side TEXT,
                outcome TEXT,
                outcome_index INTEGER,
                size_tokens REAL,
                price REAL,
                notional_usdc REAL,
                timestamp_ms INTEGER,
                tx_hash TEXT UNIQUE,
                total_score REAL,
                signals TEXT,
                alert_timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_wallet ON alerts(wallet);
            CREATE INDEX IF NOT EXISTS idx_condition ON alerts(condition_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp_ms);
            CREATE INDEX IF NOT EXISTS idx_score ON alerts(total_score);

            -- Per-(wallet, market) aggregate stats for classification
            CREATE TABLE IF NOT EXISTS wallet_market_stats (
                wallet TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                market_slug TEXT,
                trade_count INTEGER NOT NULL DEFAULT 0,
                flag_count INTEGER NOT NULL DEFAULT 0,
                total_buy_notional REAL NOT NULL DEFAULT 0.0,
                total_sell_notional REAL NOT NULL DEFAULT 0.0,
                first_trade_ts INTEGER,
                last_trade_ts INTEGER,
                last_flag_ts INTEGER,
                PRIMARY KEY (wallet, condition_id)
            );

            CREATE INDEX IF NOT EXISTS idx_wms_condition
                ON wallet_market_stats(condition_id);
            CREATE INDEX IF NOT EXISTS idx_wms_flag_count
                ON wallet_market_stats(flag_count);

            -- Per-(wallet, market, outcome) position tracking for PnL evaluation
            CREATE TABLE IF NOT EXISTS wallet_outcome_positions (
                wallet TEXT NOT NULL,
                condition_id TEXT NOT NULL,
                outcome_index INTEGER NOT NULL,
                net_shares REAL NOT NULL DEFAULT 0.0,
                buy_notional REAL NOT NULL DEFAULT 0.0,
                sell_notional REAL NOT NULL DEFAULT 0.0,
                buy_shares REAL NOT NULL DEFAULT 0.0,
                sell_shares REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (wallet, condition_id, outcome_index)
            );
        """)
        self.conn.commit()
        logging.info(f"Database initialized: {self.db_path}")

    def record_trade(self, trade: Trade):
        """
        Update wallet-level stats for every processed trade.

        Uses pure SQL UPSERTs so counters survive restarts. After a
        restart the first poll may re-process up to ~500 historical
        trades, slightly inflating trade_count (and position figures)
        for affected wallets. This is negligible for markets running
        for days/weeks and does NOT affect flag_count (alerts table
        deduplicates on tx_hash).
        """
        buy_notional = trade.notional_usdc if trade.side == "BUY" else 0.0
        sell_notional = trade.notional_usdc if trade.side == "SELL" else 0.0

        # 1. Wallet × market aggregate
        self.conn.execute("""
            INSERT INTO wallet_market_stats
                (wallet, condition_id, market_slug, trade_count, flag_count,
                 total_buy_notional, total_sell_notional,
                 first_trade_ts, last_trade_ts)
            VALUES (?, ?, ?, 1, 0, ?, ?, ?, ?)
            ON CONFLICT(wallet, condition_id) DO UPDATE SET
                trade_count = trade_count + 1,
                total_buy_notional = total_buy_notional + excluded.total_buy_notional,
                total_sell_notional = total_sell_notional + excluded.total_sell_notional,
                last_trade_ts = MAX(last_trade_ts, excluded.last_trade_ts),
                market_slug = excluded.market_slug
        """, (
            trade.wallet, trade.condition_id, trade.market_slug,
            buy_notional, sell_notional,
            trade.timestamp_ms, trade.timestamp_ms,
        ))

        # 2. Wallet × market × outcome position
        shares_delta = trade.size_tokens if trade.side == "BUY" else -trade.size_tokens
        buy_shares = trade.size_tokens if trade.side == "BUY" else 0.0
        sell_shares = trade.size_tokens if trade.side == "SELL" else 0.0

        self.conn.execute("""
            INSERT INTO wallet_outcome_positions
                (wallet, condition_id, outcome_index,
                 net_shares, buy_notional, sell_notional, buy_shares, sell_shares)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wallet, condition_id, outcome_index) DO UPDATE SET
                net_shares = net_shares + excluded.net_shares,
                buy_notional = buy_notional + excluded.buy_notional,
                sell_notional = sell_notional + excluded.sell_notional,
                buy_shares = buy_shares + excluded.buy_shares,
                sell_shares = sell_shares + excluded.sell_shares
        """, (
            trade.wallet, trade.condition_id, trade.outcome_index,
            shares_delta, buy_notional, sell_notional, buy_shares, sell_shares,
        ))

        self.conn.commit()

    def record_flag(self, wallet: str, condition_id: str, timestamp_ms: int):
        """
        Increment flag_count for a wallet×market after a successful alert save.

        Only called when save_alert returns True (tx_hash was new), so
        flag_count stays correct even if the same trade is re-processed.
        """
        self.conn.execute("""
            UPDATE wallet_market_stats
            SET flag_count = flag_count + 1,
                last_flag_ts = ?
            WHERE wallet = ? AND condition_id = ?
        """, (timestamp_ms, wallet, condition_id))
        self.conn.commit()

    def save_alert(self, alert: Alert) -> bool:
        """Save alert to database"""
        try:
            data = alert.to_db_dict()
            self.conn.execute("""
                INSERT OR IGNORE INTO alerts (
                    wallet, condition_id, market_slug, side, outcome, outcome_index,
                    size_tokens, price, notional_usdc, timestamp_ms, tx_hash,
                    total_score, signals, alert_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["wallet"], data["condition_id"], data["market_slug"],
                data["side"], data["outcome"], data["outcome_index"],
                data["size_tokens"], data["price"], data["notional_usdc"],
                data["timestamp_ms"], data["tx_hash"], data["total_score"],
                data["signals"], data["alert_timestamp"]
            ))
            self.conn.commit()

            self.context.record_wallet_flag(alert.trade.wallet)

            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            logging.error(f"Failed to save alert: {e}")
            return False

    def close(self):
        self.conn.close()