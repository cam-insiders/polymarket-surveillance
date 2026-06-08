"""
Wallet-level evaluation using statistical abnormal returns.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List
import math
import numpy as np

from backtesting.logging_utils import experiment_backtest_logs_quiet
from backtesting.market_resolutions import get_winning_outcome


logger = logging.getLogger(__name__)


def _is_experiments_run() -> bool:
    """True when the entry script lives under the experiments directory."""
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0:
        return False
    normalized = os.path.normpath(argv0).lower()
    parts = normalized.split(os.sep)
    return "experiments" in parts


def _warn(msg: str) -> None:
    """Suppress warning noise during experiments sweeps."""
    if _is_experiments_run():
        return
    logging.warning(msg)


class WalletEvaluator:
    """
    Evaluates wallets using post-resolution returns/PnL z-scores.
    """

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        min_wallet_notional: float = 500.0,
        label_metric: str = "return",  # "return" or "pnl"
    ):
        self.z_score_threshold = z_score_threshold
        self.min_wallet_notional = float(min_wallet_notional)
        if label_metric not in {"return", "pnl"}:
            raise ValueError("label_metric must be 'return' or 'pnl'")
        self.label_metric = label_metric

    def evaluate_wallets(
        self,
        backtest_result,
        market_metadata: Dict,
        winning_outcome_override: int | None = None,
    ) -> List[Dict]:
        market_id = market_metadata.get("id")
        winning_outcome = (
            int(winning_outcome_override)
            if winning_outcome_override is not None
            else get_winning_outcome(market_id)
        )

        if winning_outcome is None:
            _warn(
                f"Market {market_metadata.get('market_slug', market_id)} not in resolution dataset; skipping."
            )
            return []

        wallet_rows: List[Dict] = []
        for wallet, positions in backtest_result.wallet_positions.items():
            if not positions:
                continue

            total_notional = float(backtest_result.wallet_notional.get(wallet, 0.0))
            if total_notional < self.min_wallet_notional:
                continue

            costs = backtest_result.wallet_costs.get(wallet, {})
            gross_buy = float(backtest_result.wallet_gross_buy_notional.get(wallet, 0.0))

            total_payout = 0.0
            total_cost = 0.0
            for outcome_idx, shares in positions.items():
                if outcome_idx == winning_outcome:
                    total_payout += shares
                total_cost += costs.get(outcome_idx, 0.0)

            net_pnl = total_payout - total_cost
            wallet_return = (net_pnl / gross_buy) if gross_buy > 1e-9 else 0.0
            label_value = wallet_return if self.label_metric == "return" else net_pnl

            flags = backtest_result.wallet_flags.get(wallet, [])

            # Informed trading score: return efficiency × log-scaled capital commitment.
            informed_score = wallet_return * math.log2(1.0 + gross_buy / 1000.0) if gross_buy > 0 else 0.0

            wallet_rows.append(
                {
                    "wallet": wallet,
                    "suspicion_score": float(backtest_result.wallet_suspicion.get(wallet, 0.0)),
                    "num_flags": len(flags),
                    "has_alert": len(flags) > 0,
                    "trade_count": int(backtest_result.wallet_trade_counts.get(wallet, 0)),
                    "cluster_boost": float(backtest_result.wallet_cluster_boost.get(wallet, 1.0)),
                    "has_common_ownership": bool(
                        backtest_result.wallet_has_common_ownership.get(wallet, False)
                    ),
                    "total_notional": total_notional,
                    "gross_buy_notional": gross_buy,
                    "net_pnl": net_pnl,
                    "return": wallet_return,
                    "informed_score": informed_score,
                    "label_value": label_value,
                    "position": dict(positions),
                    "detector_breakdown": self._get_detector_breakdown(flags),
                }
            )

        if len(wallet_rows) < 3:
            _warn(f"Too few eligible wallets ({len(wallet_rows)}) for statistical labeling")
            return []

        label_values = np.array([row["label_value"] for row in wallet_rows], dtype=float)
        mean_value = float(np.mean(label_values))
        std_value = float(np.std(label_values, ddof=1)) if len(label_values) > 1 else 0.0

        evaluations: List[Dict] = []
        insider_count = 0

        for row in wallet_rows:
            if std_value > 1e-12:
                z_score = (row["label_value"] - mean_value) / std_value
            else:
                z_score = 0.0

            is_insider = bool(z_score > self.z_score_threshold)
            insider_count += int(is_insider)

            evaluations.append(
                {
                    **row,
                    "label_metric": self.label_metric,
                    "label_mean": mean_value,
                    "label_std": std_value,
                    "z_score": float(z_score),
                    "is_insider": is_insider,
                    "winning_outcome": winning_outcome,
                    "market_slug": market_metadata.get("market_slug", ""),
                }
            )

        insider_rate = insider_count / len(evaluations)
        if not experiment_backtest_logs_quiet():
            logger.info(
                f"Wallet eval | market={market_metadata.get('market_slug', market_id)} "
                f"| wallets={len(evaluations)} "
                f"| insiders={insider_count} ({insider_rate:.2%}) "
                f"| metric={self.label_metric}"
            )

        returns = np.array([e["return"] for e in evaluations], dtype=float)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        cutoff_return = mean_return + (self.z_score_threshold * std_return)

        insider_mask = np.array([bool(e["is_insider"]) for e in evaluations], dtype=bool)
        insider_returns = returns[insider_mask]
        insider_mean_return = float(np.mean(insider_returns)) if len(insider_returns) else 0.0
        insider_median_return = float(np.median(insider_returns)) if len(insider_returns) else 0.0

        if not experiment_backtest_logs_quiet():
            logger.info(
                f"Label return stats | market={market_metadata.get('market_slug', market_id)} "
                f"| mean={mean_return:.2%} std={std_return:.2%} "
                f"| cutoff(z>{self.z_score_threshold:g})={cutoff_return:.2%} "
                f"| insiders_avg={insider_mean_return:.2%} median={insider_median_return:.2%}"
            )


        return evaluations

    @staticmethod
    def _get_detector_breakdown(flagged_trades: List[Dict]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}

        for flag in flagged_trades:
            if "detectors" in flag:
                for detector_name in flag["detectors"]:
                    breakdown[detector_name] = breakdown.get(detector_name, 0) + 1
                continue

            # Backward compatibility with old format
            for signal in flag.get("signals", []):
                detector_name = signal.detector_name
                breakdown[detector_name] = breakdown.get(detector_name, 0) + 1

        return breakdown




