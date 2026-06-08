"""Base detector framework and shared context"""

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from clustering.models import ClusteringState
from models import Trade, DetectionSignal, OrderBookSnapshot, WalletProfile
from wallet_cache import WalletCacheManager


class RollingStats:
    """
    Efficient rolling window statistics using incremental updates.
    Maintains mean, std dev without recomputing from scratch.
    """
    
    def __init__(self, window_hours: int = 24):
        self.window_ms = window_hours * 3600 * 1000
        self.trades: deque = deque()
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0  # Sum of squared deviations from mean

    def add_trade(self, timestamp_ms: int, notional: float):
        self.trades.append((timestamp_ms, notional))
        self.count += 1
        delta = notional - self._mean
        self._mean += delta / self.count
        delta2 = notional - self._mean
        self._m2 += delta * delta2

        cutoff = timestamp_ms - self.window_ms
        while self.trades and self.trades[0][0] < cutoff:
            old_ts, old_val = self.trades.popleft()
            self.count -= 1
            if self.count == 0:
                self._mean = 0.0
                self._m2 = 0.0
            else:
                delta = old_val - self._mean
                self._mean -= delta / self.count
                delta2 = old_val - self._mean
                self._m2 -= delta * delta2

    def get_mean(self):
        return self._mean if self.count > 0 else None

    def get_std(self):
        if self.count < 2:
            return None
        variance = self._m2 / (self.count - 1)
        if variance < 0.0:
            if variance > -1e-12:
                variance = 0.0
            else:
                return None
        return math.sqrt(variance)
    
    def get_count(self) -> int:
        return self.count


class DetectionContext:
    """
    Shared context accessible to all detectors.
    Stores historical data, market state, etc.
    """
    
    def __init__(self, wallet_cache: Optional['WalletCacheManager'] = None):
        # Market state
        self.last_prices: Dict[Tuple[str, int], float] = {}
        self.orderbooks: Dict[str, OrderBookSnapshot] = {}
        self.condition_to_assets: Dict[str, List[str]] = {}
        self.asset_to_outcome: Dict[Tuple[str, str], Tuple[str, int]] = {}
        
        # Trading activity
        self.wallet_activity: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.market_stats: Dict[str, RollingStats] = {}
        self.market_stats_window_hours: Dict[str, int] = {}

        self.recent_trades: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.last_trade_time_per_outcome: Dict[Tuple[str, int], int] = {}
        self.outcome_trade_gaps: Dict[Tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=200)
        )
        
        # Wallet state
        self.wallet_profiles: Dict[str, WalletProfile] = {}
        self.wallet_flag_counts: Dict[str, int] = defaultdict(int)
        
        # Persistent wallet metadata.
        self.wallet_cache: Optional['WalletCacheManager'] = wallet_cache

        # Clustering state
        self.clustering_state: Optional['ClusteringState'] = None
    
    def add_trade(self, trade: Trade):
        """Update context with a new trade"""

        profile = self.get_or_create_profile(trade.wallet)
        profile.update(trade)

        self.last_prices[(trade.condition_id, trade.outcome_index)] = trade.price

        self.wallet_activity[(trade.wallet, trade.condition_id)].append(
            (trade.timestamp_ms, trade.notional_usdc)
        )

        # Allow detectors to override the rolling window length (defaults to 24h).
        desired_hours = int(self.market_stats_window_hours.get(trade.condition_id, 24))
        desired_ms = desired_hours * 3600 * 1000

        stats = self.market_stats.get(trade.condition_id)
        if stats is None or getattr(stats, "window_ms", None) != desired_ms:
            stats = RollingStats(window_hours=desired_hours)
            self.market_stats[trade.condition_id] = stats

        stats.add_trade(trade.timestamp_ms, trade.notional_usdc)

        self.recent_trades[trade.condition_id].append(
            (trade.timestamp_ms, trade.notional_usdc, trade.wallet)
        )

        outcome_key = (trade.condition_id, trade.outcome_index)
        prev_outcome_trade_ts = self.last_trade_time_per_outcome.get(outcome_key)
        if prev_outcome_trade_ts is not None:
            gap_ms = trade.timestamp_ms - prev_outcome_trade_ts
            if gap_ms > 0:
                self.outcome_trade_gaps[outcome_key].append(gap_ms)
        self.last_trade_time_per_outcome[outcome_key] = trade.timestamp_ms


    def get_or_create_profile(self, wallet: str) -> WalletProfile:
        """Retrieve or create a WalletProfile for a given wallet"""
        if wallet not in self.wallet_profiles:
            self.wallet_profiles[wallet] = WalletProfile(wallet_address=wallet)
        return self.wallet_profiles[wallet]

    def register_market_assets(self, condition_id: str, asset_ids: List[str]):
        """Register which asset_ids belong to which market"""
        self.condition_to_assets[condition_id] = asset_ids
        for idx, asset_id in enumerate(asset_ids):
            self.asset_to_outcome[asset_id] = (condition_id, idx)
    
    def update_orderbook(self, asset_id: str, snapshot: OrderBookSnapshot):
        """Update orderbook state for an asset"""
        self.orderbooks[asset_id] = snapshot
    
    def get_orderbook_for_outcome(self, condition_id: str, outcome_index: int) -> Optional[OrderBookSnapshot]:
        """Get orderbook for a specific outcome"""
        asset_ids = self.condition_to_assets.get(condition_id, [])
        if outcome_index >= len(asset_ids):
            return None
        asset_id = asset_ids[outcome_index]
        return self.orderbooks.get(asset_id)
    
    def record_wallet_flag(self, wallet: str):
        """Increment flag count for a wallet"""
        self.wallet_flag_counts[wallet] += 1


class Detector(ABC):
    """Abstract base class for all detectors"""
    
    REQUIRES_DIRECTIONAL_POSITION = False

    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        """Analyze a trade and return a signal if suspicious."""
        pass
    
    @abstractmethod
    def update_state(self, trade: Trade) -> None:
        """Update internal state after processing a trade."""
        pass

    def _is_likely_flipper(self, trade: Trade, context: DetectionContext) -> bool:
        """
        Returns True if wallet appears to be market-making/arbitraging.
        """
        profile = context.wallet_profiles.get(trade.wallet)
        if not profile:
            return False
        
        # Get thresholds from config (detectors can override)
        config = getattr(self, 'config', {})
        flipper_config = config.get('flipper_filter', {})
        
        max_ratio = flipper_config.get('max_directional_ratio', 0.5)
        min_volume = flipper_config.get('min_volume_for_pattern', 10000.0)
        
        ratio = profile.get_directional_ratio(trade.condition_id)
        total_volume = profile.total_volume_usdc.get(trade.condition_id, 0)
        
        if total_volume < min_volume:
            return False  # Not enough data to determine pattern
        
        is_flipper = ratio < max_ratio
        
        if is_flipper:
            logging.debug(
                f"[{self.name}] Filtered flipper: {trade.wallet[:10]}... "
                f"ratio={ratio:.2f}, volume=${total_volume:.0f}"
            )
        
        return is_flipper
