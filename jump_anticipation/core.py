"""Post-hoc wallet boost for trades that anticipate price jumps."""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from backtesting.backtest_runner import BacktestResult
from models import Trade

logger = logging.getLogger(__name__)


@dataclass
class JumpEvent:
    """A significant price move in a single outcome's price series."""
    outcome_index: int
    jump_start_ms: int    # timestamp of the first trade in the detected jump
    jump_end_ms: int      # timestamp of the last trade within the jump window
    price_change: float   # p_end - p_start (signed)
    direction: int        # +1 if price went up, -1 if price went down


def find_jumps(trades: List[Trade], config: Dict) -> List[JumpEvent]:
    """Find significant price moves in each outcome's price series."""
    jump_threshold = float(config.get("jump_threshold", 0.05))
    jump_window_minutes = float(config.get("jump_window_minutes", 30))
    jump_window_ms = int(jump_window_minutes * 60 * 1000)

    by_outcome: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for trade in trades:
        by_outcome[trade.outcome_index].append((trade.timestamp_ms, trade.price))

    jumps: List[JumpEvent] = []

    for outcome_idx, points in by_outcome.items():
        points.sort()  # chronological
        n = len(points)
        if n < 2:
            continue

        j = 0
        for i in range(n):
            ts_i, p_i = points[i]

            if j < i:
                j = i

            while j + 1 < n and points[j + 1][0] - ts_i <= jump_window_ms:
                j += 1

            if j == i:
                continue  # no second trade within window

            ts_j, p_j = points[j]
            price_change = p_j - p_i

            if abs(price_change) >= jump_threshold:
                jumps.append(JumpEvent(
                    outcome_index=outcome_idx,
                    jump_start_ms=ts_i,
                    jump_end_ms=ts_j,
                    price_change=price_change,
                    direction=1 if price_change > 0 else -1,
                ))

    logger.debug(
        f"find_jumps: {len(trades):,} trades -> {len(jumps):,} jump events "
        f"(threshold={jump_threshold:.3f}, window={jump_window_minutes:.0f}min)"
    )
    return jumps


def score_wallets_jump_anticipation(
    scoring_trades: List[Trade],
    jumps: List[JumpEvent],
    config: Dict,
) -> Dict[str, float]:
    """Score wallets by alignment with the largest nearby future jump."""
    pre_jump_lookback_minutes = float(config.get("pre_jump_lookback_minutes", 60))
    pre_jump_lookback_ms = int(pre_jump_lookback_minutes * 60 * 1000)
    min_pre_jump_trades = int(config.get("min_pre_jump_trades", 2))
    max_boost_factor = float(config.get("max_boost_factor", 2.0))
    min_trade_notional = float(config.get("min_trade_notional", 0.0))

    jumps_by_outcome: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
    for jump in jumps:
        jumps_by_outcome[jump.outcome_index].append(
            (jump.jump_start_ms, jump.direction, abs(jump.price_change))
        )
    for oi in jumps_by_outcome:
        jumps_by_outcome[oi].sort()  # sort by start time

    jump_start_times: Dict[int, List[int]] = {
        oi: [entry[0] for entry in jlist]
        for oi, jlist in jumps_by_outcome.items()
    }

    wallet_n_total: Dict[str, int] = defaultdict(int)
    wallet_n_pre_jump: Dict[str, int] = defaultdict(int)
    wallet_n_aligned: Dict[str, int] = defaultdict(int)

    for trade in scoring_trades:
        if trade.notional_usdc < min_trade_notional:
            continue

        wallet = trade.wallet
        oi = trade.outcome_index
        ts = trade.timestamp_ms
        wallet_n_total[wallet] += 1

        starts = jump_start_times.get(oi)
        if starts is None:
            continue

        lo = bisect.bisect_left(starts, ts)
        hi = bisect.bisect_right(starts, ts + pre_jump_lookback_ms)

        if lo >= hi:
            continue

        wallet_n_pre_jump[wallet] += 1

        jlist = jumps_by_outcome[oi]
        relevant = jlist[lo:hi]
        _, jump_direction, _ = max(relevant, key=lambda x: x[2])

        trade_direction = 1 if trade.side.upper() == "BUY" else -1
        if trade_direction == jump_direction:
            wallet_n_aligned[wallet] += 1

    boost_scores: Dict[str, float] = {}
    n_boosted = 0

    for wallet, n_total in wallet_n_total.items():
        n_pre = wallet_n_pre_jump.get(wallet, 0)
        n_aligned = wallet_n_aligned.get(wallet, 0)

        if n_pre < min_pre_jump_trades:
            boost_scores[wallet] = 1.0
            continue

        alignment_rate = n_aligned / n_pre
        concentration = n_pre / n_total if n_total > 0 else 0.0

        alignment_excess = max(0.0, (alignment_rate - 0.5) * 2.0)
        raw_score = alignment_excess * concentration

        boost = 1.0 + (max_boost_factor - 1.0) * raw_score
        boost_scores[wallet] = boost

        if boost > 1.001:
            n_boosted += 1

    logger.debug(
        f"score_wallets_jump_anticipation: {len(boost_scores)} wallets scored, "
        f"{n_boosted} received boost > 1.0"
    )
    return boost_scores


def apply_jump_boost(
    result: BacktestResult,
    jump_scores: Dict[str, float],
) -> None:
    """Multiply wallet_cluster_boost by jump anticipation scores in place."""
    for wallet, jump_boost in jump_scores.items():
        if jump_boost <= 1.001:
            continue
        existing = result.wallet_cluster_boost.get(wallet, 1.0)
        result.wallet_cluster_boost[wallet] = existing * jump_boost


def run_jump_anticipation_boost(
    result: BacktestResult,
    all_trades: List[Trade],
    config: Dict,
    scoring_trades: Optional[List[Trade]] = None,
) -> Dict:
    """Run jump detection, wallet scoring, and boost application."""
    jumps = find_jumps(all_trades, config)

    if not jumps:
        logger.debug("run_jump_anticipation_boost: no jumps detected — boost not applied.")
        return {
            "n_jumps": 0,
            "n_wallets_scored": 0,
            "n_wallets_boosted": 0,
            "mean_boost": 1.0,
            "max_boost": 1.0,
        }

    effective_scoring_trades = scoring_trades if scoring_trades is not None else all_trades
    jump_scores = score_wallets_jump_anticipation(effective_scoring_trades, jumps, config)
    apply_jump_boost(result, jump_scores)

    n_boosted = sum(1 for v in jump_scores.values() if v > 1.001)
    boosted_vals = [v for v in jump_scores.values() if v > 1.001]
    mean_boost = sum(boosted_vals) / len(boosted_vals) if boosted_vals else 1.0
    max_boost_val = max(jump_scores.values(), default=1.0)

    logger.debug(
        f"run_jump_anticipation_boost: {len(jumps):,} jumps | "
        f"{len(jump_scores)} wallets scored | "
        f"{n_boosted} boosted (mean={mean_boost:.3f}, max={max_boost_val:.3f})"
    )

    return {
        "n_jumps": len(jumps),
        "n_wallets_scored": len(jump_scores),
        "n_wallets_boosted": n_boosted,
        "mean_boost": mean_boost,
        "max_boost": max_boost_val,
    }
