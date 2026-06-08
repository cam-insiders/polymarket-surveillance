"""Trade-based detectors for insider trading detection"""

import math
import statistics
from collections import deque
from typing import Dict, Optional

from models import Trade, DetectionSignal
from .base import Detector, DetectionContext


class VolumeAnomalyDetector(Detector):
    """
    Statistical outlier detection for trade size.
    Uses z-score relative to rolling window to identify outliers.
    """

    REQUIRES_DIRECTIONAL_POSITION = True
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.lookback_hours = config.get("lookback_window_hours", 24)
        self.min_trades_baseline = config.get("min_trades_for_baseline", 10)
        self.z_score_threshold = config.get("z_score_threshold", 3.0)
        self.min_notional = config.get("min_absolute_notional", 1000.0)
        self.max_confidence = config.get("max_confidence", 1.0)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None

        if self._is_likely_flipper(trade, context):
            return None

        desired_hours = int(self.lookback_hours)
        if context.market_stats_window_hours.get(trade.condition_id) != desired_hours:
            context.market_stats_window_hours[trade.condition_id] = desired_hours

        stats = context.market_stats.get(trade.condition_id)

        desired_ms = desired_hours * 3600 * 1000
        if stats is not None and getattr(stats, "window_ms", None) != desired_ms:
            return None

        if stats is None or stats.get_count() < self.min_trades_baseline:
            return None

        mean = stats.get_mean()
        std = stats.get_std()

        if mean is None or std is None or std < 1e-6:
            return None

        z_score = (trade.notional_usdc - mean) / std

        if z_score >= self.z_score_threshold:
            confidence = 1.0 / (1.0 + math.exp(-(z_score - self.z_score_threshold) / 1.5))
            confidence *= self.max_confidence

            return DetectionSignal(
                detector_name=self.name,
                confidence_score=confidence,
                reason=f"Volume outlier: z={z_score:.2f}σ "
                    f"(${trade.notional_usdc:.0f} vs μ=${mean:.0f}, σ=${std:.0f})",
                metadata={
                    "z_score": z_score,
                    "trade_notional": trade.notional_usdc,
                    "mean": mean,
                    "std": std,
                    "baseline_trades": stats.get_count(),
                    "lookback_hours": desired_hours,
                },
            )

        return None

    
    def update_state(self, trade: Trade) -> None:
        pass


class ProbabilityImpactDetector(Detector):
    """
    Measures information content via price impact.
    Large probability shifts indicate new information entering the market.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_delta_prob = config.get("min_delta_prob", 0.03)
        self.min_delta_log_odds = config.get("min_delta_log_odds", 0.4)
        self.min_notional = config.get("min_notional", 500.0)
        self.max_confidence = config.get("max_confidence", 1.0)

    def _safe_logit(self, p: float) -> Optional[float]:
        if p <= 0.001 or p >= 0.999:
            return None
        return math.log(p / (1.0 - p))
    
    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None
        
        key = (trade.condition_id, trade.outcome_index)
        prev_price = context.last_prices.get(key)
        
        if prev_price is None or abs(prev_price - trade.price) < 1e-6:
            return None
        
        delta_prob = abs(trade.price - prev_price)
        
        prev_lo = self._safe_logit(prev_price)
        curr_lo = self._safe_logit(trade.price)
        delta_lo = None
        
        if prev_lo is not None and curr_lo is not None:
            delta_lo = abs(curr_lo - prev_lo)
        
        scores = []
        reasons = []
        
        if delta_prob >= self.min_delta_prob:
            prob_score = min(1.0, delta_prob / (2 * self.min_delta_prob))
            scores.append(prob_score)
            reasons.append(f"Δp={delta_prob:.1%}")
        
        if delta_lo is not None and delta_lo >= self.min_delta_log_odds:
            lo_score = min(1.0, delta_lo / (2 * self.min_delta_log_odds))
            scores.append(lo_score)
            reasons.append(f"Δlogit={delta_lo:.2f}")
        
        if scores:
            if len(scores) == 2:
                confidence = 0.4 * scores[0] + 0.6 * scores[1]
            else:
                confidence = scores[0]

            confidence *= self.max_confidence    

            return DetectionSignal(
                detector_name=self.name,
                confidence_score=confidence,
                reason=f"Price impact: {' & '.join(reasons)} "
                       f"[{prev_price:.1%}→{trade.price:.1%}]",
                metadata={
                    "delta_prob": delta_prob,
                    "delta_log_odds": delta_lo,
                    "prev_price": prev_price,
                    "new_price": trade.price,
                }
            )
        
        return None
    
    def update_state(self, trade: Trade) -> None:
        pass

class AccumulationDetector(Detector):
    """
    Flags wallets that are relentlessly building a position without exiting.
    Upgrade from 'BurstTradingDetector' because it ignores flippers/market makers.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_accumulation_usdc = config.get("min_accumulation_usdc", 5000.0)
        self.min_directional_ratio = config.get("min_directional_ratio", 0.8)
        self.max_confidence = config.get("max_confidence", 1.0)
        self.min_outcome_concentration = config.get("min_outcome_concentration", 0.9)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.side != "BUY":
            return None
        
        profile = context.wallet_profiles.get(trade.wallet)
        if not profile:
            return None
        
        # Check 1: Is this wallet focused on ONE outcome? (>90% concentration)
        outcome_concentration = profile.get_outcome_concentration(
            trade.condition_id, 
            trade.outcome_index
        )
        
        if outcome_concentration < self.min_outcome_concentration:
            # Wallet is spreading across multiple outcomes (hedging/arbitrage)
            return None
        
        # Check 2: Are they accumulating on this outcome? (>80% directional)
        outcome_directional = profile.get_outcome_directional_ratio(
            trade.condition_id,
            trade.outcome_index
        )
        
        if outcome_directional < self.min_directional_ratio:  # 0.8
            # They're flipping on this outcome
            return None
        
        # Check 3: Is the position large enough?
        outcome_position_shares = profile.get_outcome_position(
            trade.condition_id, 
            trade.outcome_index
        )
        outcome_price = profile.last_price.get(
            (trade.condition_id, trade.outcome_index), 
            trade.price
        )
        outcome_exposure = abs(outcome_position_shares * outcome_price)
        
        if outcome_exposure < self.min_accumulation_usdc:  # $5000
            return None
        
        # All checks passed: focused, accumulating, large position
        confidence = outcome_directional * self.max_confidence
        
        return DetectionSignal(
            detector_name="AccumulationDetector",
            confidence_score=confidence,
            reason=f"Focused accumulation on {trade.outcome}: "
                f"${outcome_exposure:,.0f} position, "
                f"{outcome_concentration:.1%} outcome focus, "
                f"{outcome_directional:.1%} directional",
            metadata={
                "outcome_index": trade.outcome_index,
                "outcome_exposure_usdc": outcome_exposure,
                "outcome_concentration": outcome_concentration,
                "outcome_directional_ratio": outcome_directional
            }
        )
        
    def update_state(self, trade: Trade) -> None:
        pass


class ExtremePositionDetector(Detector):
    """
    Flags trades at probability extremes (tails).
    Large bets on unlikely outcomes (<20% or >80%) are suspicious.
    """

    REQUIRES_DIRECTIONAL_POSITION = True
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.tail_threshold = config.get("tail_threshold", 0.20)
        self.min_notional = config.get("min_notional", 1000.0)
        self.max_confidence = config.get("max_confidence", 1.0)
        
    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None
        
        if self._is_likely_flipper(trade, context):
            return None
        
        in_low_tail = trade.price < self.tail_threshold and trade.side == "BUY"
        in_high_tail = trade.price > (1 - self.tail_threshold) and trade.side == "SELL"
        
        if not (in_low_tail or in_high_tail):
            return None
        
        if in_low_tail:
            tail_distance = self.tail_threshold - trade.price
            tail_type = "low"
        else:
            tail_distance = trade.price - (1 - self.tail_threshold)
            tail_type = "high"
        
        tail_score = min(1.0, tail_distance / self.tail_threshold)
        size_score = min(1.0, trade.notional_usdc / 5000)
        confidence = 0.5 * tail_score + 0.5 * size_score
        confidence *= self.max_confidence

        return DetectionSignal(
            detector_name=self.name,
            confidence_score=confidence,
            reason=f"Tail bet: ${trade.notional_usdc:.0f} {trade.side} at {trade.price:.1%} "
                   f"({tail_type} probability)",
            metadata={
                "tail_type": tail_type,
                "price": trade.price,
                "tail_distance": tail_distance,
            }
        )
    
    def update_state(self, trade: Trade) -> None:
        pass


class ContraOutcomeSilenceDetector(Detector):
    """
    Flags trades that arrive while the opposite outcome has gone unusually quiet.

    In binary markets, informed flow on one side often causes market makers to
    reduce participation on the contra side. We can observe that behavior in the
    trade stream as an abnormally long silence versus the contra side's recent
    median inter-trade gap.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_gap_samples = int(config.get("min_gap_samples", 10))
        self.silence_threshold = float(config.get("silence_threshold", 5.0))
        self.min_notional = float(config.get("min_notional", 1000.0))
        self.max_contra_age_minutes = float(config.get("max_contra_age_minutes", 120.0))
        self.max_confidence = float(config.get("max_confidence", 1.0))

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None

        if trade.outcome_index not in (0, 1):
            return None

        contra_outcome = 1 - trade.outcome_index
        contra_key = (trade.condition_id, contra_outcome)

        last_contra_trade_ts = context.last_trade_time_per_outcome.get(contra_key)
        if last_contra_trade_ts is None:
            return None

        time_since_contra_ms = trade.timestamp_ms - last_contra_trade_ts
        if time_since_contra_ms <= 0:
            return None

        max_contra_age_ms = self.max_contra_age_minutes * 60_000.0
        if time_since_contra_ms > max_contra_age_ms:
            return None

        contra_gaps = context.outcome_trade_gaps.get(contra_key)
        if contra_gaps is None or len(contra_gaps) < self.min_gap_samples:
            return None

        normal_gap_ms = float(statistics.median(contra_gaps))
        if normal_gap_ms <= 0:
            return None

        silence_ratio = time_since_contra_ms / normal_gap_ms
        if silence_ratio < self.silence_threshold:
            return None

        confidence = min(1.0, silence_ratio / (2.0 * self.silence_threshold))
        confidence *= self.max_confidence

        return DetectionSignal(
            detector_name=self.name,
            confidence_score=confidence,
            reason=(
                f"Contra outcome silent: {silence_ratio:.1f}x normal gap "
                f"({time_since_contra_ms / 60000.0:.1f}m vs {normal_gap_ms / 60000.0:.1f}m)"
            ),
            metadata={
                "contra_outcome_index": contra_outcome,
                "time_since_contra_ms": time_since_contra_ms,
                "normal_gap_ms": normal_gap_ms,
                "silence_ratio": silence_ratio,
                "gap_samples": len(contra_gaps),
            },
        )

    def update_state(self, trade: Trade) -> None:
        pass