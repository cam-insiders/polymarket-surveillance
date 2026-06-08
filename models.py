"""Data structures for Polymarket Insider Trading Detection System"""

from collections import defaultdict
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union, overload
from datetime import datetime

import numpy as np

@dataclass
class WalletProfile:
    """Profile and state for a trading wallet"""
    wallet_address: str

    # Global Stats
    first_seen: int = 0
    last_seen: int = 0
    total_trades: int = 0

    # Outcome Specific Tracking (Key = (condition_id, outcome_index))

    net_position_shares: Dict[Tuple[str, int], float] = field(
        default_factory=lambda: defaultdict(float)
    )

    last_price: Dict[Tuple[str, int], float] = field(
        default_factory=lambda: defaultdict(float)
    )

    # Volume Stats (Key = (condition_id, outcome_index))

    total_buy_volume_usdc: Dict[Tuple[str, int], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    total_sell_volume_usdc: Dict[Tuple[str, int], float] = field(
        default_factory=lambda: defaultdict(float)
    )

    # Market-Specific State (Key = condition_id)

    total_volume_usdc: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def update(self, trade: 'Trade'):
        """Process a new trade and update state."""

        if self.first_seen == 0:
            self.first_seen = trade.timestamp_ms
        self.last_seen = trade.timestamp_ms
        self.total_trades += 1

        cid = trade.condition_id
        outcome_key = (cid, trade.outcome_index)

        # 1. Update outcome-specific volume
        if trade.side == "BUY":
            self.net_position_shares[outcome_key] += trade.size_tokens
            self.total_buy_volume_usdc[outcome_key] += trade.notional_usdc
        elif trade.side == "SELL":
            self.net_position_shares[outcome_key] -= trade.size_tokens
            self.total_sell_volume_usdc[outcome_key] += trade.notional_usdc
        
        # 2. Update last price for outcome
        self.last_price[outcome_key] = trade.price

        # 3. Update market-level volume
        self.total_volume_usdc[cid] += trade.notional_usdc

    def get_net_exposure_usdc(self, condition_id: str) -> float:
        """Calculate net exposure in USDC for a market"""
        exposure = 0.0
        for (cid, outcome_idx), shares in self.net_position_shares.items():
            if cid == condition_id:
                outcome_key = (cid, outcome_idx)
                price = self.last_price.get(outcome_key, 0.0)
                exposure += abs(shares * price)
        return exposure
    
    def get_directional_ratio(self, condition_id: str) -> float:
        """
        Returns 0.0 to 1.0.
        0.1 = Flipper (High volume, zero net position)
        1.0 = Accumulator (All volume is building position)
        """
        vol = self.total_volume_usdc.get(condition_id, 0)
        if vol == 0:
            return 0.0
        
        exposure = self.get_net_exposure_usdc(condition_id)
        return min(1.0, exposure / vol)
    
    def get_outcome_directional_ratio(self, condition_id: str, outcome_index: int) -> float:
        """
        For THIS SPECIFIC OUTCOME: what's the directional ratio?
        (net position / total volume on this outcome)
        
        Returns 0.0 to 1.0:
        - 1.0 = Pure accumulation (all buys, no sells)
        - 0.0 = Pure flipping (equal buys and sells)
        """
        outcome_volume = self.get_outcome_volume(condition_id, outcome_index)
        
        if outcome_volume < 1e-6:
            return 0.0
        
        outcome_position_shares = self.get_outcome_position(condition_id, outcome_index)
        outcome_key = (condition_id, outcome_index)
        outcome_price = self.last_price.get(outcome_key, 0.0)
        outcome_exposure = abs(outcome_position_shares * outcome_price)
        
        return min(1.0, outcome_exposure / outcome_volume)

    def get_outcome_concentration(self, condition_id: str, outcome_index: int) -> float:
        """Return this outcome's share of the wallet's market volume."""
        market_volume = self.total_volume_usdc.get(condition_id, 0.0)
        
        if market_volume < 1e-6:
            return 0.0
        
        outcome_volume = self.get_outcome_volume(condition_id, outcome_index)
        
        return min(1.0, outcome_volume / market_volume)

    def get_outcome_position(self, condition_id: str, outcome_index: int) -> float:
        """Get net position in shares for a specific outcome"""
        return self.net_position_shares.get((condition_id, outcome_index), 0.0)
    
    def get_outcome_volume(self, condition_id: str, outcome_index: int) -> float:
        """Get total volume in USDC for a specific outcome"""
        outcome_key = (condition_id, outcome_index)
        return (self.total_buy_volume_usdc.get(outcome_key, 0.0) +
                self.total_sell_volume_usdc.get(outcome_key, 0.0))
    
    def is_hedged(self, condition_id: str, threshold: float = 0.3) -> bool:
        """Check if wallet has significant positions on BOTH outcomes (hedge)"""
        # Get volumes for all outcomes in this market
        outcome_volumes = []
        for (cid, outcome_idx) in self.net_position_shares.keys():
            if cid == condition_id:
                vol = self.get_outcome_volume(cid, outcome_idx)
                if vol > 0:
                    outcome_volumes.append(vol)
        
        if len(outcome_volumes) < 2:
            return False  # Only traded one outcome
        
        total_vol = sum(outcome_volumes)
        if total_vol == 0:
            return False
        
        # Check if all outcomes have significant volume
        # (not just one dominant outcome)
        min_outcome_vol = min(outcome_volumes)
        min_ratio = min_outcome_vol / total_vol
        
        return min_ratio >= threshold
    
    def get_position_summary(self, condition_id: str) -> Dict:
        """
        Get a complete summary of positions in this market.
        Useful for debugging and logging.
        """
        summary = {
            "total_volume": self.total_volume_usdc.get(condition_id, 0.0),
            "net_exposure": self.get_net_exposure_usdc(condition_id),
            "directional_ratio": self.get_directional_ratio(condition_id),
            "is_hedged": self.is_hedged(condition_id),
            "outcomes": {}
        }
        
        for (cid, outcome_idx), shares in self.net_position_shares.items():
            if cid == condition_id:
                outcome_key = (cid, outcome_idx)
                price = self.last_price.get(outcome_key, 0.0)
                summary["outcomes"][outcome_idx] = {
                    "shares": shares,
                    "last_price": price,
                    "position_value": shares * price,
                    "buy_volume": self.total_buy_volume_usdc.get(outcome_key, 0.0),
                    "sell_volume": self.total_sell_volume_usdc.get(outcome_key, 0.0),
                }
        
        return summary

@dataclass
class Trade:
    """Normalised trade object from Polymarket API"""
    wallet: str
    condition_id: str
    market_slug: str
    side: str  # "BUY" or "SELL"
    outcome: str
    outcome_index: int
    size_tokens: float
    price: float
    notional_usdc: float
    timestamp_ms: int
    tx_hash: str
    asset: str

    @classmethod
    def from_api_response(cls, raw: dict) -> Optional['Trade']:
        """Parse raw API response into Trade object"""
        try:
            size = float(raw["size"])
            price = float(raw["price"])
            return cls(
                wallet=raw["proxyWallet"],
                condition_id=raw["conditionId"],
                market_slug=raw.get("slug", ""),
                side=raw["side"],
                outcome=raw.get("outcome", ""),
                outcome_index=int(raw.get("outcomeIndex", -1)),
                size_tokens=size,
                price=price,
                notional_usdc=size * price,
                timestamp_ms=int(raw["timestamp"]) * 1000,
                tx_hash=raw["transactionHash"],
                asset=raw["asset"],
            )
        except (KeyError, ValueError, TypeError) as e:
            logging.warning(f"Failed to parse trade: {e}")
            return None


class TradeView:
    """Read-only compatibility proxy for one row in a TradeBatch."""

    __slots__ = ("_batch", "_idx")

    def __init__(self, batch: "TradeBatch", idx: int):
        self._batch = batch
        self._idx = int(idx)

    @property
    def wallet(self) -> str:
        return self._batch._wallet_pool[int(self._batch._wallet_codes[self._idx])]

    @property
    def condition_id(self) -> str:
        return self._batch.condition_id

    @property
    def market_slug(self) -> str:
        return self._batch.market_slug

    @property
    def side(self) -> str:
        return "BUY" if int(self._batch._side_codes[self._idx]) == 1 else "SELL"

    @property
    def outcome(self) -> str:
        return self._batch.outcomes[int(self._batch.outcome_index[self._idx])]

    @property
    def outcome_index(self) -> int:
        return int(self._batch.outcome_index[self._idx])

    @property
    def size_tokens(self) -> float:
        return float(self._batch.size_tokens[self._idx])

    @property
    def price(self) -> float:
        return float(self._batch.price[self._idx])

    @property
    def notional_usdc(self) -> float:
        return float(self._batch.notional_usdc[self._idx])

    @property
    def timestamp_ms(self) -> int:
        return int(self._batch.timestamp_ms[self._idx])

    @property
    def tx_hash(self) -> str:
        return self._batch._tx_pool[int(self._batch._tx_codes[self._idx])]

    @property
    def asset(self) -> str:
        return self._batch.assets[int(self._batch.outcome_index[self._idx])]

    def as_trade(self) -> Trade:
        return Trade(
            wallet=self.wallet,
            condition_id=self.condition_id,
            market_slug=self.market_slug,
            side=self.side,
            outcome=self.outcome,
            outcome_index=self.outcome_index,
            size_tokens=self.size_tokens,
            price=self.price,
            notional_usdc=self.notional_usdc,
            timestamp_ms=self.timestamp_ms,
            tx_hash=self.tx_hash,
            asset=self.asset,
        )


class TradeBatch(Sequence[TradeView]):
    """
    Columnar per-market trade collection.

    It behaves like a read-only sequence of Trade-like objects while storing the
    heavy per-row data in NumPy arrays and compact string pools.
    """

    __slots__ = (
        "condition_id",
        "market_slug",
        "outcomes",
        "assets",
        "_wallet_pool",
        "_tx_pool",
        "_wallet_codes",
        "_tx_codes",
        "_side_codes",
        "outcome_index",
        "size_tokens",
        "price",
        "notional_usdc",
        "timestamp_ms",
    )

    def __init__(
        self,
        *,
        condition_id: str,
        market_slug: str,
        outcomes: Tuple[str, str],
        assets: Tuple[str, str],
        wallet_pool: Sequence[str],
        tx_pool: Sequence[str],
        wallet_codes: np.ndarray,
        tx_codes: np.ndarray,
        side_codes: np.ndarray,
        outcome_index: np.ndarray,
        size_tokens: np.ndarray,
        price: np.ndarray,
        notional_usdc: np.ndarray,
        timestamp_ms: np.ndarray,
    ):
        self.condition_id = str(condition_id)
        self.market_slug = str(market_slug)
        self.outcomes = (str(outcomes[0]), str(outcomes[1]))
        self.assets = (str(assets[0]), str(assets[1]))
        self._wallet_pool = tuple(str(x) for x in wallet_pool)
        self._tx_pool = tuple(str(x) for x in tx_pool)
        self._wallet_codes = np.asarray(wallet_codes, dtype=np.int32)
        self._tx_codes = np.asarray(tx_codes, dtype=np.int32)
        self._side_codes = np.asarray(side_codes, dtype=np.int8)
        self.outcome_index = np.asarray(outcome_index, dtype=np.int8)
        self.size_tokens = np.asarray(size_tokens, dtype=np.float64)
        self.price = np.asarray(price, dtype=np.float64)
        self.notional_usdc = np.asarray(notional_usdc, dtype=np.float64)
        self.timestamp_ms = np.asarray(timestamp_ms, dtype=np.int64)

    @staticmethod
    def _pool_codes(values: Sequence[str]) -> Tuple[Tuple[str, ...], np.ndarray]:
        lookup: Dict[str, int] = {}
        pool: List[str] = []
        codes = np.empty(len(values), dtype=np.int32)
        for i, raw in enumerate(values):
            value = str(raw)
            code = lookup.get(value)
            if code is None:
                code = len(pool)
                lookup[value] = code
                pool.append(value)
            codes[i] = code
        return tuple(pool), codes

    @classmethod
    def from_columns(
        cls,
        *,
        wallets: Sequence[str],
        sides: Sequence[str],
        outcome_index: Sequence[int],
        size_tokens: Sequence[float],
        price: Sequence[float],
        notional_usdc: Sequence[float],
        timestamp_ms: Sequence[int],
        tx_hashes: Sequence[str],
        condition_id: str,
        market_slug: str,
        outcomes: Tuple[str, str],
        assets: Tuple[str, str],
    ) -> "TradeBatch":
        wallet_pool, wallet_codes = cls._pool_codes(wallets)
        tx_pool, tx_codes = cls._pool_codes(tx_hashes)
        side_codes = np.fromiter(
            (1 if str(side).upper() == "BUY" else 0 for side in sides),
            dtype=np.int8,
            count=len(wallet_codes),
        )
        return cls(
            condition_id=condition_id,
            market_slug=market_slug,
            outcomes=outcomes,
            assets=assets,
            wallet_pool=wallet_pool,
            tx_pool=tx_pool,
            wallet_codes=wallet_codes,
            tx_codes=tx_codes,
            side_codes=side_codes,
            outcome_index=np.asarray(outcome_index, dtype=np.int8),
            size_tokens=np.asarray(size_tokens, dtype=np.float64),
            price=np.asarray(price, dtype=np.float64),
            notional_usdc=np.asarray(notional_usdc, dtype=np.float64),
            timestamp_ms=np.asarray(timestamp_ms, dtype=np.int64),
        )

    @classmethod
    def empty(cls, market: Dict) -> "TradeBatch":
        return cls.from_columns(
            wallets=[],
            sides=[],
            outcome_index=[],
            size_tokens=[],
            price=[],
            notional_usdc=[],
            timestamp_ms=[],
            tx_hashes=[],
            condition_id=market.get("condition_id", ""),
            market_slug=market.get("market_slug", ""),
            outcomes=(market.get("answer1", ""), market.get("answer2", "")),
            assets=(market.get("token1", ""), market.get("token2", "")),
        )

    @classmethod
    def concat(cls, batches: Sequence["TradeBatch"]) -> "TradeBatch":
        non_empty = [b for b in batches if len(b) > 0]
        if not non_empty:
            if not batches:
                raise ValueError("Cannot concatenate an empty batch list without market metadata")
            return batches[0].take(np.empty(0, dtype=np.int64))

        first = non_empty[0]
        wallet_lookup: Dict[str, int] = {}
        tx_lookup: Dict[str, int] = {}
        wallet_pool: List[str] = []
        tx_pool: List[str] = []
        wallet_code_chunks: List[np.ndarray] = []
        tx_code_chunks: List[np.ndarray] = []

        for batch in non_empty:
            if (
                batch.condition_id != first.condition_id
                or batch.market_slug != first.market_slug
                or batch.outcomes != first.outcomes
                or batch.assets != first.assets
            ):
                raise ValueError("TradeBatch.concat only supports batches from the same market")

            wallet_remap = np.empty(len(batch._wallet_pool), dtype=np.int32)
            for old_code, wallet in enumerate(batch._wallet_pool):
                new_code = wallet_lookup.get(wallet)
                if new_code is None:
                    new_code = len(wallet_pool)
                    wallet_lookup[wallet] = new_code
                    wallet_pool.append(wallet)
                wallet_remap[old_code] = new_code
            wallet_code_chunks.append(wallet_remap[batch._wallet_codes])

            tx_remap = np.empty(len(batch._tx_pool), dtype=np.int32)
            for old_code, tx_hash in enumerate(batch._tx_pool):
                new_code = tx_lookup.get(tx_hash)
                if new_code is None:
                    new_code = len(tx_pool)
                    tx_lookup[tx_hash] = new_code
                    tx_pool.append(tx_hash)
                tx_remap[old_code] = new_code
            tx_code_chunks.append(tx_remap[batch._tx_codes])

        return cls(
            outcome_index=np.concatenate([b.outcome_index for b in non_empty]),
            size_tokens=np.concatenate([b.size_tokens for b in non_empty]),
            price=np.concatenate([b.price for b in non_empty]),
            notional_usdc=np.concatenate([b.notional_usdc for b in non_empty]),
            timestamp_ms=np.concatenate([b.timestamp_ms for b in non_empty]),
            side_codes=np.concatenate([b._side_codes for b in non_empty]),
            wallet_codes=np.concatenate(wallet_code_chunks),
            tx_codes=np.concatenate(tx_code_chunks),
            wallet_pool=wallet_pool,
            tx_pool=tx_pool,
            condition_id=first.condition_id,
            market_slug=first.market_slug,
            outcomes=first.outcomes,
            assets=first.assets,
        )

    def __len__(self) -> int:
        return int(self.timestamp_ms.shape[0])

    @overload
    def __getitem__(self, idx: int) -> TradeView:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "TradeBatch":
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[TradeView, "TradeBatch"]:
        if isinstance(idx, slice):
            return self._slice(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return TradeView(self, idx)

    def __iter__(self) -> Iterator[TradeView]:
        for idx in range(len(self)):
            yield TradeView(self, idx)

    def _slice(self, idx: slice) -> "TradeBatch":
        return TradeBatch(
            condition_id=self.condition_id,
            market_slug=self.market_slug,
            outcomes=self.outcomes,
            assets=self.assets,
            wallet_pool=self._wallet_pool,
            tx_pool=self._tx_pool,
            wallet_codes=self._wallet_codes[idx],
            tx_codes=self._tx_codes[idx],
            side_codes=self._side_codes[idx],
            outcome_index=self.outcome_index[idx],
            size_tokens=self.size_tokens[idx],
            price=self.price[idx],
            notional_usdc=self.notional_usdc[idx],
            timestamp_ms=self.timestamp_ms[idx],
        )

    def take(self, indices: Sequence[int]) -> "TradeBatch":
        idx = np.asarray(indices, dtype=np.int64)
        return TradeBatch(
            condition_id=self.condition_id,
            market_slug=self.market_slug,
            outcomes=self.outcomes,
            assets=self.assets,
            wallet_pool=self._wallet_pool,
            tx_pool=self._tx_pool,
            wallet_codes=self._wallet_codes[idx],
            tx_codes=self._tx_codes[idx],
            side_codes=self._side_codes[idx],
            outcome_index=self.outcome_index[idx],
            size_tokens=self.size_tokens[idx],
            price=self.price[idx],
            notional_usdc=self.notional_usdc[idx],
            timestamp_ms=self.timestamp_ms[idx],
        )

    def filter_mask(self, mask: Sequence[bool]) -> "TradeBatch":
        return self.take(np.flatnonzero(np.asarray(mask, dtype=bool)))

    def filter_notional(self, min_usd: float) -> "TradeBatch":
        return self.filter_mask(self.notional_usdc >= float(min_usd))

    def sort_by_timestamp(self) -> "TradeBatch":
        if len(self) < 2:
            return self
        return self.take(np.argsort(self.timestamp_ms, kind="stable"))

    def to_trade_list(self) -> List[Trade]:
        return [view.as_trade() for view in self]

    @property
    def nbytes(self) -> int:
        arrays = (
            self._wallet_codes,
            self._tx_codes,
            self._side_codes,
            self.outcome_index,
            self.size_tokens,
            self.price,
            self.notional_usdc,
            self.timestamp_ms,
        )
        pool_bytes = sum(sys.getsizeof(s) for s in self._wallet_pool)
        pool_bytes += sum(sys.getsizeof(s) for s in self._tx_pool)
        pool_bytes += sum(sys.getsizeof(s) for s in self.outcomes)
        pool_bytes += sum(sys.getsizeof(s) for s in self.assets)
        pool_bytes += sys.getsizeof(self.condition_id) + sys.getsizeof(self.market_slug)
        return int(sum(arr.nbytes for arr in arrays) + pool_bytes)


TradeLike = Union[Trade, TradeView]
TradeCollection = Sequence[TradeLike]


def filter_trades_by_notional(trades: TradeCollection, min_usd: float) -> TradeCollection:
    """Filter trades while preserving TradeBatch's columnar representation."""
    if isinstance(trades, TradeBatch):
        return trades.filter_notional(min_usd)
    return [t for t in trades if t.notional_usdc >= float(min_usd)]


@dataclass
class DetectionSignal:
    """Output from a detector indicating suspicious activity"""
    detector_name: str
    confidence_score: float
    reason: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Alert:
    """A flagged trade with detection signals"""
    trade: Trade
    signals: List[DetectionSignal]
    total_score: float
    timestamp: datetime
    
    def to_db_dict(self) -> Dict:
        """Convert to flat dict for database storage"""
        return {
            "wallet": self.trade.wallet,
            "condition_id": self.trade.condition_id,
            "market_slug": self.trade.market_slug,
            "side": self.trade.side,
            "outcome": self.trade.outcome,
            "outcome_index": self.trade.outcome_index,
            "size_tokens": self.trade.size_tokens,
            "price": self.trade.price,
            "notional_usdc": self.trade.notional_usdc,
            "timestamp_ms": self.trade.timestamp_ms,
            "tx_hash": self.trade.tx_hash,
            "total_score": self.total_score,
            "signals": " | ".join([f"{s.detector_name}({s.confidence_score:.2f}): {s.reason}" 
                                   for s in self.signals]),
            "alert_timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrderBookSnapshot:
    """Snapshot of orderbook at a point in time"""
    asset_id: str
    timestamp_ms: int
    bids: List[Tuple[float, float]]  # [(price, size), ...] sorted desc
    asks: List[Tuple[float, float]]  # [(price, size), ...] sorted asc
    
    def get_bid_depth(self, price_threshold: float = None) -> float:
        """Total size on bid side, optionally above price threshold"""
        if price_threshold is None:
            return sum(size for (_, size) in self.bids)
        return sum(size for (price, size) in self.bids if price >= price_threshold)
    
    def get_ask_depth(self, price_threshold: float = None) -> float:
        """Total size on ask side, optionally below price threshold"""
        if price_threshold is None:
            return sum(size for (_, size) in self.asks)
        return sum(size for (price, size) in self.asks if price <= price_threshold)
    
    def get_imbalance_ratio(self) -> float:
        """Bid depth / Ask depth. >1 = bid heavy, <1 = ask heavy"""
        bid_depth = self.get_bid_depth()
        ask_depth = self.get_ask_depth()
        if ask_depth < 1e-6:
            return float('inf') if bid_depth > 0 else 1.0
        return bid_depth / ask_depth
    
    def get_spread_bps(self) -> Optional[float]:
        """Bid-ask spread in basis points"""
        if not self.bids or not self.asks:
            return None
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        mid = (best_bid + best_ask) / 2
        if mid < 1e-6:
            return None
        return ((best_ask - best_bid) / mid) * 10000


@dataclass
class MarketMetadata:
    """Complete metadata for a Polymarket market"""
    slug: str
    condition_id: str
    clob_token_ids: List[str]
    question: str
    outcomes: List[str]
    end_date_iso: Optional[str] = None
    
    def __repr__(self):
        return f"Market(slug={self.slug}, question={self.question[:40]}..., tokens={len(self.clob_token_ids)})"
