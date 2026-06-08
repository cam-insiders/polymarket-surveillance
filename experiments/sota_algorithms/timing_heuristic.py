"""
Practitioner timing heuristic baseline (fresh wallet + large buy + near resolution).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats

from backtesting.data_loader import HistoricalDataLoader
from backtesting.trade_event_study import _compute_resolution_return, _cohens_d
from experiments.sota_algorithms.common import (
    build_wallet_prior_trade_counts_by_market,
    build_wallet_insider_labels,
    copytrade_trade_summary,
    get_market_resolution_timestamp_ms,
    summarize_pooled_returns,
    wallet_classification_metrics,
    wallet_flagged_pnl_from_wallet_data,
)
from models import Trade


def run_timing_heuristic_market_baseline(
    trades: List[Trade],
    resolution_timestamp_ms: Optional[int],
    params: Dict,
    initial_wallet_trade_counts: Optional[Dict[str, int]] = None,
) -> Tuple[Set[int], Dict[str, int]]:
    if not trades:
        return set(), {}

    sorted_pairs = sorted(enumerate(trades), key=lambda x: x[1].timestamp_ms)
    max_prior_trades = int(params["max_prior_trades"])
    min_notional = float(params["min_notional"])
    max_hours = float(params["max_hours"])
    max_horizon_ms = max_hours * 3600.0 * 1000.0

    wallet_trade_count: Dict[str, int] = defaultdict(int)
    if initial_wallet_trade_counts:
        for wallet, count in initial_wallet_trade_counts.items():
            wallet_trade_count[str(wallet)] = int(count)
    flagged_indices: Set[int] = set()
    wallet_flag_counts: Dict[str, int] = defaultdict(int)

    for original_idx, trade in sorted_pairs:
        wallet = trade.wallet
        is_fresh_wallet = int(wallet_trade_count[wallet]) <= max_prior_trades
        is_large_bet = float(trade.notional_usdc) >= min_notional
        is_buy = trade.side.upper() == "BUY"
        is_close_to_resolution = (
            resolution_timestamp_ms is not None
            and 0 <= (resolution_timestamp_ms - int(trade.timestamp_ms)) <= max_horizon_ms
        )

        if is_fresh_wallet and is_large_bet and is_buy and is_close_to_resolution:
            flagged_indices.add(original_idx)
            wallet_flag_counts[wallet] += 1

        wallet_trade_count[wallet] += 1

    return flagged_indices, dict(wallet_flag_counts)


def run_timing_heuristic_baseline(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    all_entries: List[Tuple[Trade, int]],
    winning_outcomes: Dict[int, Optional[int]],
    *,
    max_prior_trades: int,
    min_notional: float,
    max_hours: float,
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
    min_usd_amount: Optional[float] = None,
) -> Dict:
    start = time.time()

    params = {
        "max_prior_trades": int(max_prior_trades),
        "min_notional": float(min_notional),
        "max_hours": float(max_hours),
    }

    flagged_returns_all: List[float] = []
    flagged_notionals_all: List[float] = []
    unflagged_returns_all: List[float] = []
    per_market_flagged: Dict[int, List[float]] = defaultdict(list)
    per_market_unflagged: Dict[int, List[float]] = defaultdict(list)
    flagged_wallets_by_market: Dict[int, Set[str]] = defaultdict(set)
    wallet_flag_counts_by_market: Dict[int, Dict[str, int]] = {}
    prior_wallet_counts_by_market = build_wallet_prior_trade_counts_by_market(all_entries)
    n_buy_eval = 0
    n_flagged_buy = 0
    n_unflagged_buy = 0

    for market_id in market_ids:
        winning = winning_outcomes.get(market_id)
        try:
            trades = loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=min_usd_amount,
                use_cache=False,
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)

        resolution_ts = get_market_resolution_timestamp_ms(loader, market_id)
        flagged_indices, wallet_flag_counts = run_timing_heuristic_market_baseline(
            trades=trades,
            resolution_timestamp_ms=resolution_ts,
            params=params,
            initial_wallet_trade_counts=prior_wallet_counts_by_market.get(market_id, {}),
        )
        wallet_flag_counts_by_market[market_id] = wallet_flag_counts
        flagged_wallets_by_market[market_id] = set()

        if winning is None:
            continue

        for idx, trade in enumerate(trades):
            if idx in flagged_indices:
                flagged_wallets_by_market[market_id].add(trade.wallet)

            if trade.side.upper() != "BUY":
                continue

            n_buy_eval += 1
            ret = _compute_resolution_return(trade, winning)
            if idx in flagged_indices:
                flagged_returns_all.append(ret)
                flagged_notionals_all.append(float(trade.notional_usdc))
                per_market_flagged[market_id].append(ret)
                n_flagged_buy += 1
            else:
                unflagged_returns_all.append(ret)
                per_market_unflagged[market_id].append(ret)
                n_unflagged_buy += 1

    pooled_stats = summarize_pooled_returns(
        flagged_returns_all,
        unflagged_returns_all,
    )
    pooled_mean_return_diff = pooled_stats["mean_return_diff"]
    pooled_flagged_mean = pooled_stats["flagged_mean_return"]
    pooled_unflagged_mean = pooled_stats["unflagged_mean_return"]

    per_market_d: List[float] = []
    sig_welch = 0
    n_markets_eval = 0
    for market_id in market_ids:
        fr = np.array(per_market_flagged.get(market_id, []))
        ur = np.array(per_market_unflagged.get(market_id, []))
        if len(fr) < 2 or len(ur) < 2:
            continue
        per_market_d.append(_cohens_d(fr, ur))
        n_markets_eval += 1
        try:
            _, p_val = scipy_stats.ttest_ind(fr, ur, equal_var=False)
            if p_val < 0.05:
                sig_welch += 1
        except Exception:
            pass

    mean_cohens_d = float(np.mean(per_market_d)) if per_market_d else 0.0
    actual_flag_rate = n_flagged_buy / max(n_buy_eval, 1)

    wallet_data = build_wallet_insider_labels(
        all_entries,
        winning_outcomes,
        z_score_threshold,
        min_wallet_notional,
    )
    wallet_pnl_stats = wallet_flagged_pnl_from_wallet_data(
        wallet_data,
        flagged_wallets_by_market,
    )
    copytrade_stats = copytrade_trade_summary(flagged_returns_all, flagged_notionals_all)

    wallet_metrics = wallet_classification_metrics(wallet_data, flagged_wallets_by_market)

    elapsed = time.time() - start
    total_flagged_wallets = int(sum(
        len(wallets) for wallets in flagged_wallets_by_market.values()
    ))
    total_wallet_flags = int(sum(
        count
        for wallet_counts in wallet_flag_counts_by_market.values()
        for count in wallet_counts.values()
    ))

    logging.info(
        f"timing_heuristic: {n_flagged_buy:,} flagged BUY trades "
        f"({actual_flag_rate:.2%}), "
        f"TP={wallet_metrics['tp']}, FP={wallet_metrics['fp']}, FN={wallet_metrics['fn']}, "
        f"wall={elapsed:.1f}s"
    )

    return {
        "baseline": "timing_heuristic",
        "num_flags": total_wallet_flags,
        "flagged_trades": n_flagged_buy,
        "unflagged_trades": n_unflagged_buy,
        "flagged_mean_return": pooled_flagged_mean,
        "unflagged_mean_return": pooled_unflagged_mean,
        "mean_return_diff": pooled_mean_return_diff,
        "mean_cohens_d": mean_cohens_d,
        "sig_welch_p05": sig_welch,
        "n_markets": n_markets_eval,
        "tp": wallet_metrics["tp"],
        "fp": wallet_metrics["fp"],
        "fn": wallet_metrics["fn"],
        "flagged_avg_return": wallet_metrics["flagged_avg_return"],
        "tp_avg_return": wallet_metrics["tp_avg_return"],
        "fp_avg_return": wallet_metrics["fp_avg_return"],
        "trades_per_second": 0.0,
        "det_p95_us": 0.0,
        "wall_clock_s": elapsed,
        "flagged_wallet_mean_net_pnl": wallet_pnl_stats["flagged_wallet_mean_net_pnl"],
        "flagged_wallet_median_net_pnl": wallet_pnl_stats["flagged_wallet_median_net_pnl"],
        "copytrade_total_flagged_buys": copytrade_stats["copytrade_total_flagged_buys"],
        "copytrade_total_capital_deployed": copytrade_stats["copytrade_total_capital_deployed"],
        "copytrade_total_pnl": copytrade_stats["copytrade_total_pnl"],
        "copytrade_portfolio_roi": copytrade_stats["copytrade_portfolio_roi"],
        "copytrade_win_rate": copytrade_stats["copytrade_win_rate"],
        "copytrade_mean_trade_return": copytrade_stats["copytrade_mean_trade_return"],
        "copytrade_median_trade_return": copytrade_stats["copytrade_median_trade_return"],
        "timing_max_prior_trades": int(params["max_prior_trades"]),
        "timing_wallet_history_scope": "platform_prior_to_market_from_entries",
        "timing_min_notional": float(params["min_notional"]),
        "timing_max_hours": float(params["max_hours"]),
        "timing_flag_rate": actual_flag_rate,
        "timing_flagged_wallets": total_flagged_wallets,
        "timing_total_wallet_flags": total_wallet_flags,
        "min_usd_amount": min_usd_amount,
        "deployable_live": True,
    }
