"""Data fetching: REST API and WebSocket clients"""

import json
import logging
import threading
import time
from typing import Dict, List, Set, Tuple, Optional

import requests
from websocket import WebSocketApp

from models import Trade, OrderBookSnapshot
from detectors.base import DetectionContext


class PolymarketDataFetcher:
    """Handles API calls to Polymarket for trade data"""
    
    def __init__(self, api_url: str, min_notional: Optional[float], max_trades: int, enable_prefilter: bool = False):
        self.api_url = api_url
        self.min_notional = 0 if min_notional is None else min_notional
        self.max_trades = max_trades
        self.enable_prefilter = enable_prefilter
        self.condition_ids: List[str] = []

        # Single set with size cap. Bounded at max_trades * 100 entries
        # (~6.5MB at max_trades=500). On overflow, downstream guards
        # handle brief re-processing: clustering entries have a UNIQUE
        # constraint (INSERT OR IGNORE), alerts dedup on tx_hash.
        self.seen_trades: Set[Tuple[str, str]] = set()
    
    def set_condition_ids(self, condition_ids: List[str]):
        """Update condition IDs to monitor"""
        self.condition_ids = condition_ids
        logging.info(f"Trade fetcher configured for {len(condition_ids)} markets")
        
    def fetch_recent_trades(self) -> List[Trade]:
        """Fetch and parse recent trades, returning only new ones"""
        if not self.condition_ids:
            logging.warning("No condition IDs configured for trade fetcher")
            return []
        
        try:
            params = {
                "market": ",".join(self.condition_ids),
                "limit": self.max_trades,
            }

            if self.enable_prefilter and self.min_notional > 0:
                params["filterType"] = "CASH"
                params["filterAmount"] = self.min_notional
            
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            raw_trades = response.json()
            
            if not isinstance(raw_trades, list):
                logging.error(f"Unexpected API response: {raw_trades}")
                return []
            
            if len(self.seen_trades) > self.max_trades * 100:
                self.seen_trades.clear()
                logging.info("Dedup set cleared (exceeded memory bound)")

            new_trades = []
            # API returns newest first, so we reverse to process chronologically
            for raw in reversed(raw_trades):
                trade_key = (raw.get("transactionHash"), raw.get("asset"))
                if trade_key in self.seen_trades:
                    continue
                self.seen_trades.add(trade_key)
                
                trade = Trade.from_api_response(raw)
                if trade:
                    new_trades.append(trade)
            
            if len(raw_trades) >= self.max_trades and len(new_trades) > self.max_trades * 0.8:
                logging.warning(
                    f"Trade throughput near limit: {len(new_trades)} new trades out of "
                    f"{len(raw_trades)} fetched (limit={self.max_trades}). "
                    f"May be missing older trades. Consider increasing "
                    f"max_trades_per_request or decreasing poll_interval_seconds."
                )

            return new_trades
            
        except requests.RequestException as e:
            logging.error(f"API request failed: {e}")
            return []


class PolymarketOrderbookStream:
    """
    WebSocket client for real-time orderbook updates.
    Runs in background thread with auto-reconnection and keep-alive pings.
    """
    
    def __init__(self, asset_ids: List[str], context: DetectionContext):
        self.asset_ids = asset_ids
        self.context = context
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.ws = None
        self.running = False
        self.ws_thread = None
        
        self.last_message_time = time.time()
        self.message_count = 0
        self.book_update_count = 0
        
    def start(self):
        """Start websocket connection in background thread"""
        self.running = True
        
        # We run the reconnection loop in a separate thread
        self.ws_thread = threading.Thread(target=self._run_stream_loop, daemon=True)
        self.ws_thread.start()
        
        # Monitor thread for stats
        health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        health_thread.start()
        
        logging.info(f"WebSocket orderbook stream started for {len(self.asset_ids)} assets")
    
    def stop(self):
        """Stop websocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
            
    def _run_stream_loop(self):
        """Main loop that handles connection and auto-reconnection"""
        while self.running:
            try:
                logging.info("Connecting to Polymarket WebSocket...")
                self.ws = WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                
                # run_forever blocks until connection is lost or closed.
                # ping_interval=30 sends a Ping frame every 30s to keep connection alive.
                # ping_timeout=10 waits 10s for Pong before considering connection dead.
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
                
            except Exception as e:
                logging.error(f"WebSocket crash: {e}")
            
            # If we fall out of run_forever, the connection is closed.
            if self.running:
                logging.warning("WebSocket disconnected. Reconnecting in 5 seconds...")
                time.sleep(5)

    def _on_open(self, ws):
        """Subscribe to assets on connection"""
        logging.info("WebSocket connected. Sending subscription...")
        subscribe_msg = {
            "assets_ids": self.asset_ids,
            "type": "market"
        }
        ws.send(json.dumps(subscribe_msg))
    
    def _on_message(self, ws, message):
        """Handle incoming orderbook updates"""
        self.message_count += 1
        self.last_message_time = time.time()
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logging.warning(f"WebSocket: Non-JSON message: {message[:100]}")
            return
        
        if isinstance(data, list):
            for snapshot in data:
                self._process_initial_snapshot(snapshot)
        elif isinstance(data, dict):
            event_type = data.get("event_type")
            if event_type == "price_change":
                self._process_price_change(data)
            elif event_type == "error":
                logging.error(f"WebSocket Error Message: {data}")
            elif self.message_count <= 5:
                # Log first few messages to debug subscription
                logging.debug(f"WebSocket event: {event_type}")
    
    def _process_initial_snapshot(self, snapshot: dict):
        """Process initial full orderbook snapshot"""
        self.book_update_count += 1
        
        try:
            asset_id = snapshot.get("asset_id")
            if not asset_id:
                return
            
            timestamp = int(snapshot.get("timestamp", time.time() * 1000))
            
            bids = []
            for bid in snapshot.get("bids", []):
                price = float(bid.get("price", 0))
                size = float(bid.get("size", 0))
                if price > 0 and size > 0:
                    bids.append((price, size))
            
            asks = []
            for ask in snapshot.get("asks", []):
                price = float(ask.get("price", 0))
                size = float(ask.get("size", 0))
                if price > 0 and size > 0:
                    asks.append((price, size))
            
            # Sort: Bids DESC, Asks ASC
            bids.sort(reverse=True, key=lambda x: x[0])
            asks.sort(key=lambda x: x[0])
            
            orderbook = OrderBookSnapshot(
                asset_id=asset_id,
                timestamp_ms=timestamp,
                bids=bids,
                asks=asks,
            )
            
            self.context.update_orderbook(asset_id, orderbook)
            
        except (AttributeError, ValueError, TypeError) as e:
            logging.error(f"Failed to process initial snapshot: {e}")
    
    def _process_price_change(self, event: dict):
        """Process price_change event (incremental update)"""
        self.book_update_count += 1
        
        try:
            timestamp = int(event.get("timestamp", time.time() * 1000))
            
            for change in event.get("price_changes", []):
                asset_id = change.get("asset_id")
                if not asset_id:
                    continue
                
                # Polymarket sends the new BEST bid/ask
                best_bid_str = change.get("best_bid")
                best_ask_str = change.get("best_ask")
                
                if not best_bid_str or not best_ask_str:
                    continue
                
                best_bid = float(best_bid_str)
                best_ask = float(best_ask_str)
                
                existing = self.context.orderbooks.get(asset_id)
                
                if existing:
                    # Shallow copy lists so we don't mutate while reading elsewhere
                    bids = list(existing.bids)
                    asks = list(existing.asks)
                    
                    # Simple top-of-book update
                    if bids and bids[0][0] != best_bid:
                        bids[0] = (best_bid, bids[0][1])
                    elif not bids:
                        bids = [(best_bid, 0)]
                    
                    if asks and asks[0][0] != best_ask:
                        asks[0] = (best_ask, asks[0][1])
                    elif not asks:
                        asks = [(best_ask, 0)]
                else:
                    bids = [(best_bid, 0)]
                    asks = [(best_ask, 0)]
                
                updated_book = OrderBookSnapshot(
                    asset_id=asset_id,
                    timestamp_ms=timestamp,
                    bids=bids,
                    asks=asks,
                )
                
                self.context.update_orderbook(asset_id, updated_book)
            
        except (AttributeError, ValueError, TypeError) as e:
            logging.error(f"Failed to process price_change: {e}")
    
    def _on_error(self, ws, error):
        # We log error but don't raise it, letting the loop handle reconnection
        logging.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        logging.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def _health_monitor(self):
        """Monitor websocket connection health and log stats"""
        last_log_time = time.time()
        
        while self.running:
            time.sleep(30)
            
            current_time = time.time()
            time_since_last_msg = current_time - self.last_message_time
            elapsed = current_time - last_log_time
            
            msg_rate = self.message_count / elapsed if elapsed > 0 else 0
            book_rate = self.book_update_count / elapsed if elapsed > 0 else 0
            
            if time_since_last_msg > 60:
                logging.warning(f"[!] WebSocket stale: no messages in {time_since_last_msg:.0f}s")
            else:
                active_books = len(self.context.orderbooks)
                logging.info(
                    f"[STATS] WebSocket: {self.message_count} msgs ({msg_rate:.1f}/s), "
                    f"{self.book_update_count} book updates ({book_rate:.1f}/s), "
                    f"{active_books} active orderbooks"
                )
            
            self.message_count = 0
            self.book_update_count = 0
            last_log_time = current_time