"""Polygonscan client for fetching wallet transaction data"""

import json
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional

import requests

from backtesting.logging_utils import experiment_backtest_logs_quiet


logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding window rate limiter - enforces strict rate limit"""

    def __init__(self, max_per_second: float):
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second  # e.g., 0.2s for 5 req/s
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        """Acquire a token, blocking if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            
            # If not enough time has passed, sleep
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
                now = time.time()
            
            self.last_request_time = now
    
class PolygonscanClient:
    """API client for Polygonscan with rate limiting and retry logic"""

    def __init__(self, config: Dict):
        self.api_url = config.get("api_url", "https://api.etherscan.io/v2/api")
        self.api_key = config.get("api_key", "")
        self.chain_id = str(config.get("chain_id", 137))
        self.usdc_contract = config.get(
            "usdc_contract", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        )

        max_rps = config.get("max_requests_per_second", 5)
        self.rate_limiter = RateLimiter(max_rps)

        self.max_attempts = config.get("retry_max_attempts", 3)
        self.backoff_base = config.get("retry_backoff_base", 2)

        self.lookback_days = config.get("transfer_lookback_days", None)

        if not self.api_key:
            logging.warning("Polygonscan API key not set; API requests may be limited")
        
        if not experiment_backtest_logs_quiet():
            logger.info(
                f"PolygonscanClient initialized (url={self.api_url}, chain_id={self.chain_id}, max_rps={max_rps} req/s)"
            )
        
    
    def get_usdc_transfers(self, wallet_address: str) -> Optional[List[Dict]]:
        """Query USDC transfer events for a given wallet address"""

        if not self.api_key:
            logging.error("Polygonscan API key not set; cannot perform API requests")
            return None
    
        params = {
            "chainid": self.chain_id,
            "module": "account",
            "action": "tokentx",
            "contractaddress": self.usdc_contract,
            "address": wallet_address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": self.api_key,
        }

        # TODO: Add pagination if wallet has > 10k transfers
        # Polygonscan returns max 10,000 results per query
        # For now, we just take the first 10k (sufficient for most wallets)
        

        for attempt in range(1, self.max_attempts + 1):
            try:
                self.rate_limiter.acquire()

                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                status = data.get("status")
                message = data.get("message", "")
                result = data.get("result", [])

                if status == "1":
                    if len(result) >= 10000:
                        logging.warning(
                            f"Wallet {wallet_address} has >=10,000 transfers; "
                            f"results may be truncated"
                        )
                    logging.debug(
                        f"Fetched {len(result)} USDC transfers for wallet {wallet_address}"
                    )
                    return result
                elif status == "0" and "No transactions found" in message:
                    logging.debug(f"No USDC transfers found for wallet {wallet_address}")
                    return []
                else:
                    logging.error(
                        f"Polygonscan API error for {wallet_address[:10]}...: "
                        f"{message} (result: {result})"
                    )

                    result_str = str(result)
                    if "deprecated V1 endpoint" in result_str:
                        logging.error(
                            "Detected deprecated V1 endpoint response. "
                            "Use V2 endpoint (e.g., https://api.etherscan.io/v2/api) and chainid=137."
                        )
                        return None
                    
                    if "Invalid API Key" in str(result):
                        return None
                    
                    if attempt < self.max_attempts:
                        sleep_time = self.backoff_base ** attempt
                        if not experiment_backtest_logs_quiet():
                            logger.info(
                                f"Retrying in {sleep_time}s "
                                f"(attempt {attempt}/{self.max_attempts})"
                            )
                        time.sleep(sleep_time)
                        continue
                    
                    return None
            
            except requests.Timeout:
                logging.warning(f"Polygonscan request timeout (attempt {attempt}/{self.max_attempts})")
                if attempt < self.max_attempts:
                    time.sleep(self.backoff_base ** attempt)
                    continue
                return None
            
            except requests.RequestException as e:
                logging.error(f"Polygonscan request failed: {e}")
                if attempt < self.max_attempts:
                    time.sleep(self.backoff_base ** attempt)
                    continue
                return None
            
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logging.error(f"Failed to parse Polygonscan response: {e}")
                return None
            
        return None

    def parse_transfers_to_edges(
        self,
        wallet_address: str,
        transfers: List[Dict]
    ) -> List[Tuple[str, str, float, int, int, int, str]]:
        """Parse raw transfer events into attribution edges"""

        if not transfers:
            return []
        
        edge_data: Dict[Tuple[str, str], Dict] = {}

        for transfer in transfers:
            try:
                from_addr = transfer.get("from", "").lower()
                to_addr = transfer.get("to", "").lower()
                value_str = transfer.get("value", "0")
                timestamp_str = transfer.get("timeStamp", "0")
                tx_hash = transfer.get("hash", "")

                if not from_addr or not to_addr or not value_str:
                    continue

                amount = int(value_str) / 1e6  # USDC has 6 decimals
                timestamp = int(timestamp_str)

                edge_key = (from_addr, to_addr)

                if edge_key not in edge_data:
                    edge_data[edge_key] = {
                        "total_amount": 0.0,
                        "tx_count": 0,
                        "first_tx": timestamp,
                        "last_tx": timestamp,
                        "tx_hashes": [],
                    }
                
                edge_data[edge_key]["total_amount"] += amount
                edge_data[edge_key]["tx_count"] += 1
                edge_data[edge_key]["first_tx"] = min(
                    edge_data[edge_key]["first_tx"], timestamp
                )
                edge_data[edge_key]["last_tx"] = max(
                    edge_data[edge_key]["last_tx"], timestamp
                )
                edge_data[edge_key]["tx_hashes"].append(tx_hash)
            
            except (ValueError, TypeError, KeyError) as e:
                logging.warning(f"Failed to parse transfer event: {e}")
                continue
        
        edges = []
        for (from_addr, to_addr), data in edge_data.items():
            edges.append((
                from_addr,
                to_addr,
                data["total_amount"],
                data["tx_count"],
                data["first_tx"],
                data["last_tx"],
                json.dumps(data["tx_hashes"]),
            ))
        
        logging.debug(
            f"Parsed {len(edges)} attribution edges from transfers for wallet {wallet_address}"
        )

        return edges
