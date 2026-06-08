"""Persistent wallet cache with Polymarket API lookups"""

import logging
import sqlite3
import time
from typing import Optional, Dict
import requests


class WalletCacheManager:
    """
    Manages persistent cache of wallet metadata with API lookups.
    """
    
    def __init__(self, db_path: str, api_url: str, new_wallet_threshold_minutes: int = 60):
        self.db_path = db_path
        self.api_url = api_url
        self.new_wallet_threshold_minutes = new_wallet_threshold_minutes  # Consider wallets with only recent trades as "new" 
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
        # In-memory cache for current session (avoid repeated DB queries)
        self._session_cache: Dict[str, Dict] = {}
        
        # Rate limiting for API calls
        self.last_api_call_time = 0
        self.min_api_call_interval = 0.2  # 200ms between API calls
    
    def _init_db(self):
        """Create wallet cache table if it doesn't exist"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS wallet_cache (
                wallet_address TEXT PRIMARY KEY,
                is_new_to_polymarket BOOLEAN NOT NULL,
                first_trade_timestamp INTEGER,
                total_historical_trades INTEGER DEFAULT 0,
                last_api_check_timestamp INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_wallet_lookup 
            ON wallet_cache(wallet_address);
        """)
        self.conn.commit()
        logging.info(f"Wallet cache initialized: {self.db_path}")
    
    def get_wallet_info(self, wallet_address: str) -> Optional[Dict]:
        """
        Get wallet metadata, fetching from API if not cached.
        """
        # Check session cache first
        if wallet_address in self._session_cache:
            return self._session_cache[wallet_address]
        
        # Check database cache
        cursor = self.conn.execute(
            "SELECT is_new_to_polymarket, first_trade_timestamp, "
            "total_historical_trades FROM wallet_cache WHERE wallet_address = ?",
            (wallet_address,)
        )
        row = cursor.fetchone()
        
        if row:
            # Found in cache
            is_new, first_trade_ts, trade_count = row
            
            wallet_info = {
                "is_new": bool(is_new),
                "first_seen": first_trade_ts,
                "trade_count": trade_count,
                "age_hours": self._calculate_age_hours(first_trade_ts) if first_trade_ts else None
            }
            
            self._session_cache[wallet_address] = wallet_info
            return wallet_info
        
        # Not in cache - need to query API
        return self._fetch_and_cache_wallet(wallet_address)
    
    def _fetch_and_cache_wallet(self, wallet_address: str) -> Optional[Dict]:
        """
        Fetch wallet history from Polymarket API and cache result.
        """
        try:
            # Rate limiting
            self._respect_rate_limit()
            
            # Get first (oldest) trade
            params_first = {
                "user": wallet_address,
                "type": "TRADE",  # Only actual trades, not redemptions
                "sortBy": "TIMESTAMP",
                "sortDirection": "ASC",  # Ascending = oldest first
                "limit": 1
            }
            
            logging.debug(f"Fetching oldest trade for {wallet_address[:10]}...")
            response_first = requests.get(
                "https://data-api.polymarket.com/activity",  # Use /activity endpoint
                params=params_first,
                timeout=10
            )
            response_first.raise_for_status()
            
            oldest_trades = response_first.json()
            
            if not isinstance(oldest_trades, list):
                logging.warning(f"Unexpected API response for wallet {wallet_address[:10]}: {oldest_trades}")
                return None
            
            # Get total trade count
            params_count = {
                "user": wallet_address,
                "type": "TRADE",
                "limit": 500  # Get up to 500 to estimate count
            }
            
            logging.debug(f"Fetching trade count for {wallet_address[:10]}...")
            response_count = requests.get(
                "https://data-api.polymarket.com/activity",
                params=params_count,
                timeout=10
            )
            response_count.raise_for_status()
            
            all_trades = response_count.json()
            trade_count = len(all_trades) if isinstance(all_trades, list) else 0
            
            # If we got exactly 500, they likely have more
            if trade_count == 500:
                logging.debug(f"Wallet {wallet_address[:10]} has 500+ trades (exact count not fetched)")
            
            # Analyze results
            if len(oldest_trades) == 0:
                # No trades at all → definitely new
                is_new = True
                first_trade_ts = None
                
                logging.info(f"Wallet {wallet_address[:10]} has NO trade history - NEW wallet")
            
            else:
                # Has trades - check age of oldest trade
                oldest_trade = oldest_trades[0]
                timestamp = oldest_trade.get("timestamp")
                
                if not timestamp:
                    # No timestamp on oldest trade - treat conservatively as not new
                    is_new = False
                    first_trade_ts = None
                    
                    logging.warning(
                        f"Wallet {wallet_address[:10]} has trades but no timestamp on oldest trade"
                    )
                else:
                    # Got timestamp - check age
                    first_trade_ts = int(timestamp) * 1000  # Convert seconds to milliseconds
                    current_time_ms = int(time.time() * 1000)
                    age_minutes = (current_time_ms - first_trade_ts) / (1000 * 60)
                    
                    # If oldest trade is within threshold, wallet is "new"
                    if age_minutes < self.new_wallet_threshold_minutes:
                        is_new = True
                        
                        logging.info(
                            f"Wallet {wallet_address[:10]} has RECENT first trade "
                            f"(oldest: {age_minutes:.1f}min ago, ~{trade_count} total trades) - treating as NEW"
                        )
                    else:
                        is_new = False
                        
                        logging.info(
                            f"Wallet {wallet_address[:10]} has ESTABLISHED history "
                            f"(first trade: {age_minutes/60:.1f}h ago, ~{trade_count} total trades) - NOT new"
                        )
            
            # Cache in database
            self.conn.execute("""
                INSERT OR REPLACE INTO wallet_cache (
                    wallet_address, is_new_to_polymarket, first_trade_timestamp,
                    total_historical_trades, last_api_check_timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                wallet_address,
                is_new,
                first_trade_ts,
                trade_count,
                int(time.time() * 1000)
            ))
            self.conn.commit()
            
            wallet_info = {
                "is_new": is_new,
                "first_seen": first_trade_ts,
                "trade_count": trade_count,
                "age_hours": self._calculate_age_hours(first_trade_ts) if first_trade_ts else None
            }
            
            # Cache in session memory
            self._session_cache[wallet_address] = wallet_info
            
            return wallet_info
            
        except requests.RequestException as e:
            logging.error(f"API call failed for wallet {wallet_address[:10]}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error fetching wallet {wallet_address[:10]}: {e}")
            return None
    
    def _calculate_age_hours(self, first_trade_timestamp: Optional[int]) -> Optional[float]:
        """Calculate wallet age in hours since first trade"""
        if not first_trade_timestamp:
            return None
        
        current_time_ms = int(time.time() * 1000)
        age_ms = current_time_ms - first_trade_timestamp
        return age_ms / (1000 * 60 * 60)  # Convert to hours
    
    def _respect_rate_limit(self):
        """Ensure minimum interval between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        if time_since_last_call < self.min_api_call_interval:
            sleep_time = self.min_api_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call_time = time.time()
    
    def close(self):
        """Close database connection"""
        self.conn.close()