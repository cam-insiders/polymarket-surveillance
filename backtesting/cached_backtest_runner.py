"""Backtest runner that reuses frozen detector signals during sweeps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from backtesting.backtest_runner import BacktestResult, BacktestRunner
from detectors.base import DetectionContext
from detectors.trade_detectors import (
    AccumulationDetector,
    ContraOutcomeSilenceDetector,
    ExtremePositionDetector,
    ProbabilityImpactDetector,
    VolumeAnomalyDetector,
)
from detectors.wallet_detectors import RecidivismDetector
from models import DetectionSignal, Trade

logger = logging.getLogger(__name__)

_DETECTOR_NAMES = [
    "VolumeAnomalyDetector",
    "ProbabilityImpactDetector",
    "AccumulationDetector",
    "ExtremePositionDetector",
    "ContraOutcomeSilenceDetector",
    "RecidivismDetector",
]

# Maps the ParameterGrid detector group key to the detector class name.
DETECTOR_GROUP_TO_CLASS = {
    "volume_anomaly": "VolumeAnomalyDetector",
    "probability_impact": "ProbabilityImpactDetector",
    "accumulation_detector": "AccumulationDetector",
    "extreme_position": "ExtremePositionDetector",
    "contra_outcome_silence": "ContraOutcomeSilenceDetector",
    "recidivism_detector": "RecidivismDetector",
    "alert_threshold": None,  # special case: no detector runs in hot path
}


@dataclass
class FrozenSignalCache:
    """Per-market signals cached for one detector sweep."""
    market_id: int
    detector_signals: Dict[str, Dict[int, DetectionSignal]] = field(default_factory=dict)
    combined_scores: List[float] = field(default_factory=list)
    n_trades: int = 0

    def get_signal(self, detector_class_name: str, trade_idx: int) -> Optional[DetectionSignal]:
        """Return cached signal for a detector at a given trade index, or None."""
        return self.detector_signals.get(detector_class_name, {}).get(trade_idx)

    @property
    def memory_summary(self) -> str:
        total_signals = sum(len(v) for v in self.detector_signals.values())
        return (
            f"FrozenSignalCache(market={self.market_id}, "
            f"n_trades={self.n_trades}, "
            f"cached_signals={total_signals}, "
            f"sparsity={1.0 - total_signals / max(1, self.n_trades * len(self.detector_signals)):.2%})"
        )


def build_frozen_signals(
    config: Dict,
    trades: List[Trade],
    market_metadata: Dict,
    include_recidivism: bool = True,
) -> FrozenSignalCache:
    """Run one pass and record per-detector signals for a market."""
    market_id = int(market_metadata.get("id", -1))

    # Reuse BacktestRunner's config normalisation so we don't duplicate that logic.
    normalised = BacktestRunner._normalize_config(config)
    detector_cfg = normalised["detectors"]
    alert_threshold = float(normalised.get("alert_threshold", 0.5))

    detectors = _build_detectors(detector_cfg, include_recidivism)
    detector_class_names = [d.__class__.__name__ for d in detectors]

    signal_maps: Dict[str, Dict[int, DetectionSignal]] = {
        name: {} for name in detector_class_names
    }
    combined_scores: List[float] = []

    context = DetectionContext(wallet_cache=None)
    wallet_flag_counts: Dict[str, int] = {}  # local, mirrors context.wallet_flag_counts

    for trade_idx, trade in enumerate(trades):
        per_detector_signals: List[Optional[DetectionSignal]] = []

        for detector in detectors:
            signal = detector.analyze(trade, context)
            per_detector_signals.append(signal)
            detector.update_state(trade)

            if signal is not None and signal.confidence_score > 0:
                signal_maps[detector.__class__.__name__][trade_idx] = signal

        context.add_trade(trade)

        fired = [s for s in per_detector_signals if s is not None and s.confidence_score > 0]
        combined = BacktestRunner._calculate_total_score(fired)
        combined_scores.append(combined)

        if combined >= alert_threshold:
            context.record_wallet_flag(trade.wallet)

    cache = FrozenSignalCache(
        market_id=market_id,
        detector_signals=signal_maps,
        combined_scores=combined_scores,
        n_trades=len(trades),
    )

    logger.debug(
        f"build_frozen_signals: market={market_id} "
        f"| {cache.memory_summary}"
    )

    return cache

class CachedBacktestRunner:
    """Backtest runner that refreshes only one detector group."""

    def __init__(
        self,
        config: Dict,
        target_detector_group: str,
        frozen_cache: FrozenSignalCache,
        include_recidivism: bool = True,
        score_multipliers: Optional[np.ndarray] = None,
        score_cap: float = 0.95,
        wallet_cluster_boost: Optional[Dict[str, float]] = None,
        wallet_has_common_ownership: Optional[Dict[str, bool]] = None,
    ):
        """Initialize a cached runner for one detector group."""
        if target_detector_group not in DETECTOR_GROUP_TO_CLASS:
            raise ValueError(
                f"Unknown target_detector_group {target_detector_group!r}. "
                f"Valid: {sorted(DETECTOR_GROUP_TO_CLASS.keys())}"
            )

        self.config = BacktestRunner._normalize_config(config)
        self.target_detector_group = target_detector_group
        self.target_class_name: Optional[str] = DETECTOR_GROUP_TO_CLASS[target_detector_group]
        self.frozen_cache = frozen_cache
        self.include_recidivism = include_recidivism
        self.score_cap = float(score_cap)
        self.wallet_cluster_boost = wallet_cluster_boost or {}
        self.wallet_has_common_ownership = wallet_has_common_ownership or {}
        self._score_multipliers: Optional[np.ndarray] = None
        if score_multipliers is not None:
            arr = np.asarray(score_multipliers, dtype=np.float32)
            if arr.shape[0] != self.frozen_cache.n_trades:
                raise ValueError(
                    "score_multipliers length mismatch: "
                    f"got {arr.shape[0]}, expected {self.frozen_cache.n_trades}"
                )
            self._score_multipliers = arr

        # The target detector instance — None for alert_threshold sweep.
        self._target_detector = self._build_target_detector()

    def _apply_multiplier(self, trade_idx: int, score: float) -> float:
        if self._score_multipliers is None:
            return score
        return min(score * float(self._score_multipliers[trade_idx]), self.score_cap)

    def _build_target_detector(self):
        """Instantiate only the target detector with the candidate config params."""
        if self.target_class_name is None:
            return None  # alert_threshold — no detector to build

        detector_cfg = self.config["detectors"]

        class_map = {
            "VolumeAnomalyDetector": VolumeAnomalyDetector,
            "ProbabilityImpactDetector": ProbabilityImpactDetector,
            "AccumulationDetector": AccumulationDetector,
            "ExtremePositionDetector": ExtremePositionDetector,
            "ContraOutcomeSilenceDetector": ContraOutcomeSilenceDetector,
            "RecidivismDetector": RecidivismDetector,
        }

        target_cls = class_map[self.target_class_name]

        # Map class name back to config key.
        cfg_key_map = {
            "VolumeAnomalyDetector": "volume_anomaly",
            "ProbabilityImpactDetector": "probability_impact",
            "AccumulationDetector": "accumulation_detector",
            "ExtremePositionDetector": "extreme_position",
            "ContraOutcomeSilenceDetector": "contra_outcome_silence",
            "RecidivismDetector": "recidivism_detector",
        }
        cfg_key = cfg_key_map[self.target_class_name]
        return target_cls(detector_cfg.get(cfg_key, {}))

    def run_backtest(
        self,
        trades: List[Trade],
        market_metadata: Dict,
        capture_alerts: bool = False,
    ) -> BacktestResult:
        """
        Cached hot-path evaluation.
        """
        if self.target_class_name is None:
            # Special path: alert_threshold sweep.
            return self._run_alert_threshold_sweep(trades, market_metadata)

        return self._run_detector_sweep(
            trades=trades,
            market_metadata=market_metadata,
            capture_alerts=capture_alerts,
        )

    def _run_alert_threshold_sweep(
        self,
        trades: List[Trade],
        market_metadata: Dict,
    ) -> BacktestResult:
        """
        Re-threshold pre-cached combined scores.  Zero detector computation.
        Wallet state is still tracked correctly from the trade stream.
        """
        alert_threshold = float(self.config.get("alert_threshold", 0.5))
        cache = self.frozen_cache

        if len(trades) != cache.n_trades:
            logger.warning(
                f"CachedBacktestRunner: trade count mismatch for market "
                f"{market_metadata.get('id')} — "
                f"cache has {cache.n_trades}, got {len(trades)}. "
                f"Falling back to full BacktestRunner."
            )
            return BacktestRunner(
                config=self.config, include_recidivism=self.include_recidivism
            ).run_backtest(trades, market_metadata)

        wallet_suspicion: Dict[str, float] = {}
        wallet_flags: Dict[str, List] = {}
        wallet_positions: Dict[str, Dict[int, float]] = {}
        wallet_costs: Dict[str, Dict[int, float]] = {}
        wallet_trade_counts: Dict[str, int] = {}
        wallet_notional: Dict[str, float] = {}
        wallet_gross_buy_notional: Dict[str, float] = {}

        alerts_generated = 0

        for trade_idx, trade in enumerate(trades):
            w = trade.wallet
            if w not in wallet_suspicion:
                wallet_suspicion[w] = 0.0
                wallet_flags[w] = []
                wallet_positions[w] = {}
                wallet_costs[w] = {}
                wallet_trade_counts[w] = 0
                wallet_notional[w] = 0.0
                wallet_gross_buy_notional[w] = 0.0

            wallet_trade_counts[w] += 1
            wallet_notional[w] += trade.notional_usdc

            oi = trade.outcome_index
            if oi not in wallet_positions[w]:
                wallet_positions[w][oi] = 0.0
                wallet_costs[w][oi] = 0.0

            side = trade.side.upper()
            if side == "BUY":
                wallet_positions[w][oi] += trade.size_tokens
                wallet_costs[w][oi] += trade.notional_usdc
                wallet_gross_buy_notional[w] += trade.notional_usdc
            else:
                wallet_positions[w][oi] -= trade.size_tokens
                wallet_costs[w][oi] -= trade.notional_usdc

            combined = self._apply_multiplier(trade_idx, cache.combined_scores[trade_idx])
            wallet_suspicion[w] += combined

            if combined >= alert_threshold:
                alerts_generated += 1
                wallet_flags[w].append({
                    "detectors": [],   # not tracked in cached path for perf
                    "score": combined,
                    "timestamp_ms": trade.timestamp_ms,
                })

        return BacktestResult(
            total_trades=len(trades),
            alerts_generated=alerts_generated,
            alerts=[],
            detector_stats={},
            all_trade_features=[],
            market_id=market_metadata.get("id"),
            market_slug=market_metadata.get("market_slug", ""),
            wallet_suspicion=wallet_suspicion,
            wallet_flags=wallet_flags,
            wallet_positions=wallet_positions,
            wallet_costs=wallet_costs,
            wallet_trade_counts=wallet_trade_counts,
            wallet_cluster_boost={
                wallet: float(self.wallet_cluster_boost.get(wallet, 1.0))
                for wallet in wallet_suspicion
            },
            wallet_has_common_ownership={
                wallet: bool(self.wallet_has_common_ownership.get(wallet, False))
                for wallet in wallet_suspicion
            },
            wallet_notional=wallet_notional,
            wallet_gross_buy_notional=wallet_gross_buy_notional,
        )

    def _run_detector_sweep(
        self,
        trades: List[Trade],
        market_metadata: Dict,
        capture_alerts: bool = False,
    ) -> BacktestResult:
        """
        Per-trade loop: look up frozen signals for N-1 detectors, run 1 fresh.
        context.add_trade() is called unconditionally to preserve context state.
        """
        alert_threshold = float(self.config.get("alert_threshold", 0.5))
        cache = self.frozen_cache
        target_cls_name = self.target_class_name
        target_detector = self._target_detector

        # All detector class names that are available in the cache.
        frozen_detector_names = [
            name for name in cache.detector_signals.keys()
            if name != target_cls_name
        ]

        if len(trades) != cache.n_trades:
            logger.warning(
                f"CachedBacktestRunner: trade count mismatch for market "
                f"{market_metadata.get('id')} — falling back to full BacktestRunner."
            )
            return BacktestRunner(
                config=self.config, include_recidivism=self.include_recidivism
            ).run_backtest(trades, market_metadata)

        # Context is rebuilt from scratch each hot-pass so WalletProfile state
        # is correct for the target detector's analyze() calls.
        context = DetectionContext(wallet_cache=None)

        wallet_suspicion: Dict[str, float] = {}
        wallet_flags: Dict[str, List] = {}
        wallet_positions: Dict[str, Dict[int, float]] = {}
        wallet_costs: Dict[str, Dict[int, float]] = {}
        wallet_trade_counts: Dict[str, int] = {}
        wallet_notional: Dict[str, float] = {}
        wallet_gross_buy_notional: Dict[str, float] = {}

        alerts_generated = 0

        for trade_idx, trade in enumerate(trades):

            frozen_signals: List[DetectionSignal] = []
            for det_name in frozen_detector_names:
                sig = cache.get_signal(det_name, trade_idx)
                if sig is not None:
                    frozen_signals.append(sig)

            target_signal = target_detector.analyze(trade, context)
            target_detector.update_state(trade)

            all_signals = frozen_signals
            if target_signal is not None and target_signal.confidence_score > 0:
                all_signals = frozen_signals + [target_signal]

            combined = self._apply_multiplier(
                trade_idx,
                BacktestRunner._calculate_total_score(all_signals),
            )

            context.add_trade(trade)

            w = trade.wallet
            if w not in wallet_suspicion:
                wallet_suspicion[w] = 0.0
                wallet_flags[w] = []
                wallet_positions[w] = {}
                wallet_costs[w] = {}
                wallet_trade_counts[w] = 0
                wallet_notional[w] = 0.0
                wallet_gross_buy_notional[w] = 0.0

            wallet_trade_counts[w] += 1
            wallet_notional[w] += trade.notional_usdc

            oi = trade.outcome_index
            if oi not in wallet_positions[w]:
                wallet_positions[w][oi] = 0.0
                wallet_costs[w][oi] = 0.0

            side = trade.side.upper()
            if side == "BUY":
                wallet_positions[w][oi] += trade.size_tokens
                wallet_costs[w][oi] += trade.notional_usdc
                wallet_gross_buy_notional[w] += trade.notional_usdc
            else:
                wallet_positions[w][oi] -= trade.size_tokens
                wallet_costs[w][oi] -= trade.notional_usdc

            wallet_suspicion[w] += combined

            if combined >= alert_threshold:
                alerts_generated += 1
                context.record_wallet_flag(trade.wallet)
                wallet_flags[w].append({
                    "detectors": (
                        [s.detector_name for s in all_signals]
                        if capture_alerts else []
                    ),
                    "score": combined,
                    "timestamp_ms": trade.timestamp_ms,
                })

        return BacktestResult(
            total_trades=len(trades),
            alerts_generated=alerts_generated,
            alerts=[],
            detector_stats={target_cls_name: 0} if target_cls_name else {},
            all_trade_features=[],
            market_id=market_metadata.get("id"),
            market_slug=market_metadata.get("market_slug", ""),
            wallet_suspicion=wallet_suspicion,
            wallet_flags=wallet_flags,
            wallet_positions=wallet_positions,
            wallet_costs=wallet_costs,
            wallet_trade_counts=wallet_trade_counts,
            wallet_cluster_boost={
                wallet: float(self.wallet_cluster_boost.get(wallet, 1.0))
                for wallet in wallet_suspicion
            },
            wallet_has_common_ownership={
                wallet: bool(self.wallet_has_common_ownership.get(wallet, False))
                for wallet in wallet_suspicion
            },
            wallet_notional=wallet_notional,
            wallet_gross_buy_notional=wallet_gross_buy_notional,
        )


def _build_detectors(detector_cfg: Dict, include_recidivism: bool) -> List:
    """Build the full set of detectors from a normalised config dict."""
    detectors = [
        VolumeAnomalyDetector(detector_cfg.get("volume_anomaly", {})),
        ProbabilityImpactDetector(detector_cfg.get("probability_impact", {})),
        AccumulationDetector(detector_cfg.get("accumulation_detector", {})),
        ExtremePositionDetector(detector_cfg.get("extreme_position", {})),
        ContraOutcomeSilenceDetector(detector_cfg.get("contra_outcome_silence", {})),
    ]
    if include_recidivism:
        detectors.append(RecidivismDetector(detector_cfg.get("recidivism_detector", {})))
    return detectors
