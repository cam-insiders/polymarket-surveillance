"""
Backtest runner for historical replay with wallet-level tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from backtesting.logging_utils import experiment_backtest_logs_quiet
from models import Alert, DetectionSignal, Trade
from detectors.base import DetectionContext
from detectors.trade_detectors import (
    AccumulationDetector,
    ContraOutcomeSilenceDetector,
    ExtremePositionDetector,
    ProbabilityImpactDetector,
    VolumeAnomalyDetector,
)
from detectors.wallet_detectors import RecidivismDetector


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int
    alerts_generated: int
    alerts: List[Alert]
    detector_stats: Dict[str, int]
    all_trade_features: List[Dict[str, Any]]

    market_id: Optional[int] = None
    market_slug: str = ""

    wallet_suspicion: Dict[str, float] = field(default_factory=dict)
    wallet_flags: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    wallet_positions: Dict[str, Dict[int, float]] = field(default_factory=dict)
    wallet_costs: Dict[str, Dict[int, float]] = field(default_factory=dict)

    wallet_trade_counts: Dict[str, int] = field(default_factory=dict)
    wallet_cluster_boost: Dict[str, float] = field(default_factory=dict)
    wallet_has_common_ownership: Dict[str, bool] = field(default_factory=dict)
    wallet_notional: Dict[str, float] = field(default_factory=dict)
    wallet_gross_buy_notional: Dict[str, float] = field(default_factory=dict)

    # Performance metrics
    detection_latencies_us: Optional[np.ndarray] = None   # per-trade detection-only time (μs)
    total_latencies_us: Optional[np.ndarray] = None       # per-trade full processing time (μs)
    wall_clock_seconds: float = 0.0                       # total run_backtest() wall clock


class BacktestRunner:
    """
    Replay historical trades through detection system.
    Tracks wallet-level suspicion and positions for validation.
    """

    def __init__(self, config: Dict, include_recidivism: bool = True):
        self.include_recidivism = include_recidivism
        self.config = self._normalize_config(config)

    @staticmethod
    def _normalize_config(config: Dict) -> Dict:
        """
        Accept either:
        1) nested format: {"detectors": {...}, "alert_threshold": ...}
        2) raw CONFIG format: {"volume_anomaly": {...}, ...}
        """
        if "detectors" in config:
            detector_cfg = dict(config.get("detectors", {}))
            alert_threshold = float(config.get("alert_threshold", 0.5))
            return {"detectors": detector_cfg, "alert_threshold": alert_threshold}

        detector_cfg = {
            "volume_anomaly": dict(config.get("volume_anomaly", {})),
            "probability_impact": dict(config.get("probability_impact", {})),
            "accumulation_detector": dict(config.get("accumulation_detector", {})),
            "extreme_position": dict(config.get("extreme_position", {})),
            "contra_outcome_silence": dict(config.get("contra_outcome_silence", {})),
            "recidivism_detector": dict(config.get("recidivism_detector", {})),
        }
        alert_threshold = float(config.get("alert_threshold", 0.5))
        return {"detectors": detector_cfg, "alert_threshold": alert_threshold}

    def _build_detectors(self) -> List:
        detector_cfg = self.config["detectors"]
        detectors = [
            VolumeAnomalyDetector(detector_cfg.get("volume_anomaly", {})),
            ProbabilityImpactDetector(detector_cfg.get("probability_impact", {})),
            AccumulationDetector(detector_cfg.get("accumulation_detector", {})),
            ExtremePositionDetector(detector_cfg.get("extreme_position", {})),
            ContraOutcomeSilenceDetector(detector_cfg.get("contra_outcome_silence", {})),
        ]

        if self.include_recidivism:
            detectors.append(RecidivismDetector(detector_cfg.get("recidivism_detector", {})))

        return detectors

    def run_backtest(
        self,
        trades: List[Trade],
        market_metadata: Dict,
        capture_alerts: bool = True,
        capture_trade_features: bool = True,
        progress_every: int = 100_000,
        score_multipliers: Optional[np.ndarray] = None,
        score_cap: float = 0.95,
        wallet_cluster_boost: Optional[Dict[str, float]] = None,
        wallet_has_common_ownership: Optional[Dict[str, bool]] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical trades with wallet-level tracking.
        Always captures per-trade latency for performance reporting.
        """
        import time as _time  # local import to avoid polluting module namespace
        logger = logging.getLogger(__name__)
        quiet_logs = experiment_backtest_logs_quiet()
        if not quiet_logs:
            logger.info(f"Running backtest on {len(trades):,} trades...")

        context = DetectionContext(wallet_cache=None)
        detectors = self._build_detectors()
        alert_threshold = float(self.config["alert_threshold"])

        alerts: List[Alert] = []
        all_trade_features: List[Dict[str, Any]] = []
        detector_stats = {d.__class__.__name__: 0 for d in detectors}

        wallet_suspicion: Dict[str, float] = {}
        wallet_flags: Dict[str, List[Dict[str, Any]]] = {}
        wallet_positions: Dict[str, Dict[int, float]] = {}
        wallet_costs: Dict[str, Dict[int, float]] = {}
        wallet_trade_counts: Dict[str, int] = {}
        wallet_notional: Dict[str, float] = {}
        wallet_gross_buy_notional: Dict[str, float] = {}

        alerts_generated = 0

        # TIMING: pre-allocate latency arrays
        n_trades = len(trades)
        _det_latencies = np.empty(n_trades, dtype=np.float64)
        _tot_latencies = np.empty(n_trades, dtype=np.float64)
        _wall_start = _time.perf_counter()
        effective_multipliers: Optional[np.ndarray] = None
        if score_multipliers is not None:
            effective_multipliers = np.asarray(score_multipliers, dtype=np.float32)
            if effective_multipliers.shape[0] != n_trades:
                raise ValueError(
                    "score_multipliers length mismatch: "
                    f"got {effective_multipliers.shape[0]}, expected {n_trades}"
                )

        for i, trade in enumerate(trades):
            _t0 = _time.perf_counter()  # TIMING: start

            signals: List[DetectionSignal] = []
            for detector in detectors:
                signal = detector.analyze(trade, context)
                if signal and signal.confidence_score > 0:
                    signals.append(signal)
                    detector_stats[detector.__class__.__name__] += 1
                detector.update_state(trade)

            context.add_trade(trade)
            total_score = self._calculate_total_score(signals)
            if effective_multipliers is not None:
                total_score = min(total_score * float(effective_multipliers[i]), float(score_cap))

            _t1 = _time.perf_counter()  # TIMING: detection complete
            _det_latencies[i] = (_t1 - _t0) * 1e6

            wallet = trade.wallet
            if wallet not in wallet_suspicion:
                wallet_suspicion[wallet] = 0.0
                wallet_flags[wallet] = []
                wallet_positions[wallet] = {}
                wallet_costs[wallet] = {}
                wallet_trade_counts[wallet] = 0
                wallet_notional[wallet] = 0.0
                wallet_gross_buy_notional[wallet] = 0.0

            wallet_trade_counts[wallet] += 1
            wallet_notional[wallet] += trade.notional_usdc
            wallet_suspicion[wallet] += total_score

            outcome_idx = trade.outcome_index
            if outcome_idx not in wallet_positions[wallet]:
                wallet_positions[wallet][outcome_idx] = 0.0
                wallet_costs[wallet][outcome_idx] = 0.0

            side = trade.side.upper()
            if side == "BUY":
                wallet_positions[wallet][outcome_idx] += trade.size_tokens
                wallet_costs[wallet][outcome_idx] += trade.notional_usdc
                wallet_gross_buy_notional[wallet] += trade.notional_usdc
            else:
                wallet_positions[wallet][outcome_idx] -= trade.size_tokens
                wallet_costs[wallet][outcome_idx] -= trade.notional_usdc

            if capture_trade_features:
                all_trade_features.append(
                    self._extract_features(
                        trade=trade,
                        signals=signals,
                        total_score=total_score,
                        detectors=detectors,
                        alert_threshold=alert_threshold,
                    )
                )

            if total_score >= alert_threshold:
                alerts_generated += 1
                context.record_wallet_flag(wallet)

                wallet_flags[wallet].append(
                    {
                        "detectors": [s.detector_name for s in signals],
                        "score": total_score,
                        "timestamp_ms": trade.timestamp_ms,
                    }
                )

                if capture_alerts:
                    alert_trade = trade.as_trade() if hasattr(trade, "as_trade") else trade
                    alerts.append(
                        Alert(
                            trade=alert_trade,
                            total_score=total_score,
                            signals=signals,
                            timestamp=datetime.fromtimestamp(trade.timestamp_ms / 1000.0),
                        )
                    )

            _tot_latencies[i] = (_time.perf_counter() - _t0) * 1e6  # TIMING: full trade done

            if not quiet_logs and progress_every > 0 and (i + 1) % progress_every == 0:
                logger.info(
                    f"  Processed {i + 1:,}/{n_trades:,} trades "
                    f"| alerts={alerts_generated:,}"
                )

        _wall_elapsed = _time.perf_counter() - _wall_start  # TIMING: wall clock

        if not quiet_logs:
            logger.info(
                f"Backtest complete: alerts={alerts_generated:,} trades={n_trades:,} "
                f"wallets={len(wallet_suspicion):,} wall_clock={_wall_elapsed:.2f}s"
            )

        result_wallet_cluster_boost: Dict[str, float] = {}
        if wallet_cluster_boost is not None:
            result_wallet_cluster_boost = {
                wallet: float(wallet_cluster_boost.get(wallet, 1.0))
                for wallet in wallet_suspicion
            }
        result_wallet_has_common_ownership: Dict[str, bool] = {}
        if wallet_has_common_ownership is not None:
            result_wallet_has_common_ownership = {
                wallet: bool(wallet_has_common_ownership.get(wallet, False))
                for wallet in wallet_suspicion
            }

        return BacktestResult(
            total_trades=n_trades,
            alerts_generated=alerts_generated,
            alerts=alerts,
            detector_stats=detector_stats,
            all_trade_features=all_trade_features,
            market_id=market_metadata.get("id"),
            market_slug=market_metadata.get("market_slug", ""),
            wallet_suspicion=wallet_suspicion,
            wallet_flags=wallet_flags,
            wallet_positions=wallet_positions,
            wallet_costs=wallet_costs,
            wallet_trade_counts=wallet_trade_counts,
            wallet_cluster_boost=result_wallet_cluster_boost,
            wallet_has_common_ownership=result_wallet_has_common_ownership,
            wallet_notional=wallet_notional,
            wallet_gross_buy_notional=wallet_gross_buy_notional,
            # TIMING fields
            detection_latencies_us=_det_latencies,
            total_latencies_us=_tot_latencies,
            wall_clock_seconds=_wall_elapsed,
        )

    @staticmethod
    def _calculate_total_score(signals: List[DetectionSignal]) -> float:
        if not signals:
            return 0.0
        prob_not_insider = 1.0
        for signal in signals:
            prob_not_insider *= (1.0 - signal.confidence_score)
        return 1.0 - prob_not_insider

    @staticmethod
    def _extract_features(
        trade: Trade,
        signals: List[DetectionSignal],
        total_score: float,
        detectors: List,
        alert_threshold: float,
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {
            "wallet": trade.wallet,
            "timestamp_ms": trade.timestamp_ms,
            "notional_usdc": trade.notional_usdc,
            "price": trade.price,
            "side": trade.side,
            "outcome_index": trade.outcome_index,
            "total_score": total_score,
            "num_detectors_fired": len(signals),
            "is_alert": 1 if total_score >= alert_threshold else 0,
        }

        signal_dict = {s.detector_name: s.confidence_score for s in signals}
        for detector in detectors:
            detector_name = detector.__class__.__name__
            features[f"{detector_name}_confidence"] = signal_dict.get(detector_name, 0.0)

        return features
