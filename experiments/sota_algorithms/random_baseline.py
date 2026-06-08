"""
Random BUY flagging baseline (null model).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from backtesting.data_loader import HistoricalDataLoader
from backtesting.market_resolutions import get_winning_outcome
from backtesting.trade_event_study import _compute_resolution_return
from experiments.sota_algorithms.common import (
    build_wallet_insider_labels,
    copytrade_trade_summary,
    safe_mean,
    safe_median,
    wallet_classification_metrics,
    wallet_flagged_pnl_from_wallet_data,
)
from models import Trade


def run_random_baseline(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    flag_rate: float,
    seed: int = 42,
    n_trials: int = 5,
    min_usd_amount: Optional[float] = None,
    all_entries: Optional[List[Tuple[Trade, int]]] = None,
    winning_outcomes: Optional[Dict[int, Optional[int]]] = None,
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
) -> Dict:
    start = time.time()
    rng = np.random.default_rng(seed)

    buy_returns: List[float] = []
    buy_notionals: List[float] = []
    buy_market_ids: List[int] = []
    buy_wallets: List[str] = []
    n_markets_eval = 0

    for market_id in market_ids:
        winning = winning_outcomes.get(market_id) if winning_outcomes is not None else get_winning_outcome(market_id)
        if winning is None:
            continue

        n_markets_eval += 1
        try:
            trades = loader.get_trades_for_market(
                market_id=market_id, min_usd_amount=min_usd_amount, use_cache=True
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)

        for t in trades:
            if t.side.upper() != "BUY":
                continue
            buy_returns.append(_compute_resolution_return(t, winning))
            buy_notionals.append(float(t.notional_usdc))
            buy_market_ids.append(market_id)
            buy_wallets.append(t.wallet)

    wallet_data: Dict[int, Dict[str, Dict]] = {}
    if all_entries is not None and winning_outcomes is not None:
        wallet_data = build_wallet_insider_labels(
            all_entries,
            winning_outcomes,
            z_score_threshold=z_score_threshold,
            min_wallet_notional=min_wallet_notional,
        )

    if not buy_returns:
        return {
            "baseline": "random_flagging",
            "num_flags": 0,
            "flagged_trades": 0,
            "unflagged_trades": 0,
            "flagged_mean_return": 0.0,
            "unflagged_mean_return": 0.0,
            "mean_return_diff": 0.0,
            "mean_cohens_d": 0.0,
            "sig_welch_p05": 0,
            "n_markets": n_markets_eval,
            "tp": 0, "fp": 0, "fn": 0,
            "flagged_avg_return": 0.0,
            "tp_avg_return": 0.0, "fp_avg_return": 0.0,
            "trades_per_second": 0.0,
            "det_p95_us": 0.0,
            "wall_clock_s": time.time() - start,
            "flagged_wallet_mean_net_pnl": 0.0,
            "flagged_wallet_median_net_pnl": 0.0,
            "copytrade_total_flagged_buys": 0,
            "copytrade_total_capital_deployed": 0.0,
            "copytrade_total_pnl": 0.0,
            "copytrade_portfolio_roi": 0.0,
            "copytrade_win_rate": 0.0,
            "copytrade_mean_trade_return": 0.0,
            "copytrade_median_trade_return": 0.0,
            "random_flag_rate": flag_rate,
            "random_n_trials": n_trials,
            "min_usd_amount": min_usd_amount,
            "deployable_live": True,
        }

    returns = np.asarray(buy_returns, dtype=float)
    notionals = np.asarray(buy_notionals, dtype=float)

    trial_diffs: List[float] = []
    trial_cohens: List[float] = []
    trial_flagged_counts: List[int] = []
    trial_unflagged_counts: List[int] = []
    trial_flagged_means: List[float] = []
    trial_unflagged_means: List[float] = []
    trial_copytrade_roi: List[float] = []
    trial_copytrade_win_rate: List[float] = []
    trial_copytrade_mean_trade_return: List[float] = []
    trial_copytrade_median_trade_return: List[float] = []
    trial_copytrade_total_pnl: List[float] = []
    trial_copytrade_capital: List[float] = []
    trial_wallet_mean_pnl: List[float] = []
    trial_wallet_median_pnl: List[float] = []
    trial_flagged_avg_return: List[float] = []
    trial_tp: List[float] = []
    trial_fp: List[float] = []
    trial_fn: List[float] = []
    trial_tp_avg_return: List[float] = []
    trial_fp_avg_return: List[float] = []

    for _ in range(n_trials):
        is_flagged = rng.random(returns.shape[0]) < flag_rate
        f_arr = returns[is_flagged]
        u_arr = returns[~is_flagged]
        f_not = notionals[is_flagged]

        if f_arr.size > 0 and u_arr.size > 0:
            diff = float(np.mean(f_arr) - np.mean(u_arr))
            n_f, n_u = int(f_arr.size), int(u_arr.size)
            if n_f >= 2 and n_u >= 2:
                pooled_var = (
                    ((n_f - 1) * np.var(f_arr, ddof=1) + (n_u - 1) * np.var(u_arr, ddof=1))
                    / (n_f + n_u - 2)
                )
                d = diff / np.sqrt(pooled_var) if pooled_var > 1e-12 else 0.0
            else:
                d = 0.0
            trial_diffs.append(diff)
            trial_cohens.append(d)
            trial_flagged_counts.append(n_f)
            trial_unflagged_counts.append(n_u)
            trial_flagged_means.append(float(np.mean(f_arr)))
            trial_unflagged_means.append(float(np.mean(u_arr)))

            ct = copytrade_trade_summary(f_arr.tolist(), f_not.tolist())
            trial_copytrade_roi.append(float(ct["copytrade_portfolio_roi"]))
            trial_copytrade_win_rate.append(float(ct["copytrade_win_rate"]))
            trial_copytrade_mean_trade_return.append(float(ct["copytrade_mean_trade_return"]))
            trial_copytrade_median_trade_return.append(float(ct["copytrade_median_trade_return"]))
            trial_copytrade_total_pnl.append(float(ct["copytrade_total_pnl"]))
            trial_copytrade_capital.append(float(ct["copytrade_total_capital_deployed"]))

            if wallet_data:
                flagged_wallets_by_market: Dict[int, Set[str]] = defaultdict(set)
                flagged_idx = np.where(is_flagged)[0]
                for idx in flagged_idx:
                    flagged_wallets_by_market[int(buy_market_ids[idx])].add(buy_wallets[idx])
                pnl_stats = wallet_flagged_pnl_from_wallet_data(wallet_data, flagged_wallets_by_market)
                trial_wallet_mean_pnl.append(float(pnl_stats["flagged_wallet_mean_net_pnl"]))
                trial_wallet_median_pnl.append(float(pnl_stats["flagged_wallet_median_net_pnl"]))
                trial_flagged_avg_return.append(float(pnl_stats["flagged_wallet_mean_return"]))
                wallet_metrics = wallet_classification_metrics(wallet_data, flagged_wallets_by_market)
                trial_tp.append(float(wallet_metrics["tp"]))
                trial_fp.append(float(wallet_metrics["fp"]))
                trial_fn.append(float(wallet_metrics["fn"]))
                trial_tp_avg_return.append(float(wallet_metrics["tp_avg_return"]))
                trial_fp_avg_return.append(float(wallet_metrics["fp_avg_return"]))

    return {
        "baseline": "random_flagging",
        "num_flags": int(round(safe_mean(trial_flagged_counts))) if trial_flagged_counts else 0,
        "flagged_trades": int(round(float(np.mean(trial_flagged_counts)))) if trial_flagged_counts else 0,
        "unflagged_trades": int(round(float(np.mean(trial_unflagged_counts)))) if trial_unflagged_counts else 0,
        "flagged_mean_return": float(np.mean(trial_flagged_means)) if trial_flagged_means else 0.0,
        "unflagged_mean_return": float(np.mean(trial_unflagged_means)) if trial_unflagged_means else 0.0,
        "mean_return_diff": float(np.mean(trial_diffs)) if trial_diffs else 0.0,
        "mean_cohens_d": float(np.mean(trial_cohens)) if trial_cohens else 0.0,
        "sig_welch_p05": 0,
        "n_markets": n_markets_eval,
        "tp": int(round(safe_mean(trial_tp))) if trial_tp else 0,
        "fp": int(round(safe_mean(trial_fp))) if trial_fp else 0,
        "fn": int(round(safe_mean(trial_fn))) if trial_fn else 0,
        "flagged_avg_return": safe_mean(trial_flagged_avg_return),
        "tp_avg_return": safe_mean(trial_tp_avg_return),
        "fp_avg_return": safe_mean(trial_fp_avg_return),
        "trades_per_second": 0.0,
        "det_p95_us": 0.0,
        "wall_clock_s": time.time() - start,
        "flagged_wallet_mean_net_pnl": safe_mean(trial_wallet_mean_pnl),
        "flagged_wallet_median_net_pnl": safe_median(trial_wallet_median_pnl),
        "copytrade_total_flagged_buys": int(round(safe_mean(trial_flagged_counts))) if trial_flagged_counts else 0,
        "copytrade_total_capital_deployed": safe_mean(trial_copytrade_capital),
        "copytrade_total_pnl": safe_mean(trial_copytrade_total_pnl),
        "copytrade_portfolio_roi": safe_mean(trial_copytrade_roi),
        "copytrade_win_rate": safe_mean(trial_copytrade_win_rate),
        "copytrade_mean_trade_return": safe_mean(trial_copytrade_mean_trade_return),
        "copytrade_median_trade_return": safe_median(trial_copytrade_median_trade_return),
        "random_flag_rate": flag_rate,
        "random_n_trials": n_trials,
        "min_usd_amount": min_usd_amount,
        "deployable_live": True,
    }
