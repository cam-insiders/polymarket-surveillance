"""Orderbook-based detectors for insider trading detection"""

from typing import Dict, Optional

from models import Trade, DetectionSignal
from .base import Detector, DetectionContext


class OrderbookConsumptionDetector(Detector):
    """
    Detects trades that "eat through" multiple orderbook levels.
    Large market orders that consume depth indicate urgency/confidence.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_levels_consumed = config.get("min_levels_consumed", 3)
        self.min_notional = config.get("min_notional", 1000.0)
        self.max_slippage_bps = config.get("max_slippage_bps", 50)
        self.max_confidence = config.get("max_confidence", 1.0)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None
        
        orderbook = context.get_orderbook_for_outcome(trade.condition_id, trade.outcome_index)
        
        if orderbook is None:
            return None
        
        side_to_consume = orderbook.asks if trade.side == "BUY" else orderbook.bids
        
        if not side_to_consume:
            return None
        
        remaining_size = trade.size_tokens
        levels_consumed = 0
        weighted_price_sum = 0.0
        total_consumed = 0.0
        
        for price, size in side_to_consume:
            if remaining_size <= 0:
                break
            
            consumed = min(remaining_size, size)
            weighted_price_sum += price * consumed
            total_consumed += consumed
            remaining_size -= consumed
            levels_consumed += 1
        
        if levels_consumed < self.min_levels_consumed:
            return None
        
        if total_consumed > 0:
            avg_fill_price = weighted_price_sum / total_consumed
            best_price = side_to_consume[0][0]
            slippage_bps = abs((avg_fill_price - best_price) / best_price) * 10000
        else:
            return None
        
        level_score = min(1.0, levels_consumed / 10)
        slippage_score = min(1.0, slippage_bps / self.max_slippage_bps)
        confidence = 0.5 * level_score + 0.5 * slippage_score
        confidence *= self.max_confidence

        return DetectionSignal(
            detector_name=self.name,
            confidence_score=confidence,
            reason=f"Consumed {levels_consumed} levels, {slippage_bps:.1f}bps slippage "
                   f"(${trade.notional_usdc:.0f})",
            metadata={
                "levels_consumed": levels_consumed,
                "slippage_bps": slippage_bps,
                "avg_fill_price": avg_fill_price,
                "best_price": best_price,
            }
        )
    
    def update_state(self, trade: Trade) -> None:
        pass


class OrderbookImbalanceDetector(Detector):
    """
    Detects trades on heavily imbalanced orderbooks.
    Trading against heavy side suggests non-public information.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_imbalance_ratio = config.get("min_imbalance_ratio", 3.0)
        self.min_notional = config.get("min_notional", 800.0)
        self.max_confidence = config.get("max_confidence", 1.0)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None
        
        orderbook = context.get_orderbook_for_outcome(trade.condition_id, trade.outcome_index)
        
        if orderbook is None:
            return None
        
        imbalance_ratio = orderbook.get_imbalance_ratio()
        
        trading_against_heavy_bid = trade.side == "SELL" and imbalance_ratio > self.min_imbalance_ratio
        trading_against_heavy_ask = trade.side == "BUY" and imbalance_ratio < (1 / self.min_imbalance_ratio)
        
        if not (trading_against_heavy_bid or trading_against_heavy_ask):
            return None
        
        if trading_against_heavy_bid:
            imbalance_score = min(1.0, imbalance_ratio / (2 * self.min_imbalance_ratio))
            side_desc = f"bid-heavy {imbalance_ratio:.1f}:1"
        else:
            imbalance_score = min(1.0, (1 / imbalance_ratio) / (2 * self.min_imbalance_ratio))
            side_desc = f"ask-heavy {1/imbalance_ratio:.1f}:1"
        
        size_score = min(1.0, trade.notional_usdc / 3000)
        confidence = 0.6 * imbalance_score + 0.4 * size_score
        confidence *= self.max_confidence

        return DetectionSignal(
            detector_name=self.name,
            confidence_score=confidence,
            reason=f"Trading against {side_desc} book (${trade.notional_usdc:.0f})",
            metadata={
                "imbalance_ratio": imbalance_ratio,
                "trade_side": trade.side,
                "bid_depth": orderbook.get_bid_depth(),
                "ask_depth": orderbook.get_ask_depth(),
            }
        )
    
    def update_state(self, trade: Trade) -> None:
        pass


class ThinLiquidityExploitDetector(Detector):
    """
    Flags large trades in markets with shallow orderbooks.
    Only informed traders accept high price impact in thin markets.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_depth_ratio = config.get("min_depth_ratio", 0.3)
        self.max_total_depth = config.get("max_total_depth", 5000.0)
        self.min_notional = config.get("min_notional", 500.0)
        self.max_confidence = config.get("max_confidence", 1.0)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None
        
        orderbook = context.get_orderbook_for_outcome(trade.condition_id, trade.outcome_index)
        
        if orderbook is None:
            return None
        
        total_depth = orderbook.get_bid_depth() + orderbook.get_ask_depth()
        relevant_side_depth = (orderbook.get_ask_depth() if trade.side == "BUY" 
                              else orderbook.get_bid_depth())
        
        is_thin = total_depth < self.max_total_depth
        if not is_thin:
            return None
        
        depth_ratio = trade.size_tokens / relevant_side_depth if relevant_side_depth > 0 else 0
        
        if depth_ratio < self.min_depth_ratio:
            return None
        
        thin_score = 1.0 - (total_depth / self.max_total_depth)
        consumption_score = min(1.0, depth_ratio / 0.5)
        confidence = 0.5 * thin_score + 0.5 * consumption_score
        confidence *= self.max_confidence
        
        return DetectionSignal(
            detector_name=self.name,
            confidence_score=confidence,
            reason=f"Thin market exploit: {depth_ratio:.1%} of side depth "
                   f"(total book: ${total_depth:.0f})",
            metadata={
                "total_depth": total_depth,
                "relevant_side_depth": relevant_side_depth,
                "depth_ratio": depth_ratio,
                "spread_bps": orderbook.get_spread_bps(),
            }
        )
    
    def update_state(self, trade: Trade) -> None:
        pass