"""
Isolation Forest baseline (Liu et al., 2008) on per-trade features.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest

from backtesting.data_loader import HistoricalDataLoader
from backtesting.market_resolutions import get_market_info, get_winning_outcome
from backtesting.trade_event_study import _compute_resolution_return, _cohens_d
from experiments.sota_algorithms.common import (
    build_wallet_insider_labels,
    copytrade_trade_summary,
    parse_resolution_date,
    summarize_pooled_returns,
    wallet_flagged_pnl_from_wallet_data,
)
from models import Trade

IF_FEATURE_NAMES = [
    "log_notional",
    "price",
    "side_buy",
    "wallet_prior_trades",
    "wallet_prior_notional",
    "wallet_market_concentration",
    "time_to_resolution_hours",
    "outcome_index",
    "market_trade_count_so_far",
]


def _select_matched_buy_indices(
    scores: np.ndarray,
    buy_eval_indices: List[int],
    match_flag_rate: float,
    *,
    calibration_scores: Optional[np.ndarray] = None,
    calibration_buy_indices: Optional[List[int]] = None,
) -> Tuple[Set[int], str]:
    scores = np.asarray(scores, dtype=float)
    source = "eval_buy_scores_retrospective"
    threshold_scores = scores
    threshold_buy_indices = buy_eval_indices
    if calibration_scores is not None and calibration_buy_indices:
        threshold_scores = np.asarray(calibration_scores, dtype=float)
        threshold_buy_indices = calibration_buy_indices
        source = "train_buy_scores"

    n_flag_target = max(1, int(round(float(match_flag_rate) * len(threshold_buy_indices))))
    buy_scores = threshold_scores[threshold_buy_indices]
    sorted_buy_scores = np.sort(buy_scores)
    k = min(n_flag_target - 1, len(sorted_buy_scores) - 1)
    threshold = sorted_buy_scores[k]
    return {i for i in buy_eval_indices if scores[i] <= threshold}, source


def extract_isolation_forest_features(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    min_usd_amount: Optional[float] = None,
    winning_outcome_overrides: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Walk all trades chronologically across markets, build per-trade feature
    vectors with running wallet/market statistics for Isolation Forest input.
    """
    resolution_ts: Dict[int, Optional[int]] = {}
    winning_outcomes: Dict[int, Optional[int]] = {}
    all_entries: List[Tuple[Trade, int]] = []

    for market_id in market_ids:
        if winning_outcome_overrides and market_id in winning_outcome_overrides:
            winning_outcomes[market_id] = int(winning_outcome_overrides[market_id])
        else:
            winning_outcomes[market_id] = get_winning_outcome(market_id)

        info = get_market_info(market_id)
        res_date = info.get("resolution_date") if info else None
        if res_date:
            try:
                resolution_ts[market_id] = parse_resolution_date(str(res_date).strip())
            except (ValueError, TypeError):
                resolution_ts[market_id] = None
        else:
            resolution_ts[market_id] = None

        try:
            trades = loader.get_trades_for_market(
                market_id=market_id, min_usd_amount=min_usd_amount, use_cache=False
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)

        for t in trades:
            all_entries.append((t, market_id))

    all_entries.sort(key=lambda x: x[0].timestamp_ms)

    wallet_prior_trades: Dict[str, int] = defaultdict(int)
    wallet_prior_notional: Dict[str, float] = defaultdict(float)
    wallet_market_notional: Dict[str, Dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    market_trade_count: Dict[int, int] = defaultdict(int)

    feature_rows: List[np.ndarray] = []
    trade_info: List[Dict] = []

    for trade, market_id in all_entries:
        wallet = trade.wallet
        market_key = market_id

        wt = wallet_prior_notional[wallet]
        wm = wallet_market_notional[wallet][market_key]
        concentration = (wm / wt) if wt > 0 else 0.0

        res_ts = resolution_ts.get(market_id)
        time_to_res = (
            (res_ts - trade.timestamp_ms) / 3.6e6
        ) if res_ts is not None else 0.0

        features = np.array([
            np.log(trade.notional_usdc + 1.0),
            trade.price,
            1.0 if trade.side.upper() == "BUY" else 0.0,
            float(wallet_prior_trades[wallet]),
            wt,
            concentration,
            time_to_res,
            float(trade.outcome_index),
            float(market_trade_count[market_key]),
        ])

        feature_rows.append(features)
        trade_info.append({
            "trade": trade,
            "market_id": market_id,
            "winning_outcome": winning_outcomes.get(market_id),
        })

        wallet_prior_trades[wallet] += 1
        wallet_prior_notional[wallet] += trade.notional_usdc
        wallet_market_notional[wallet][market_key] += trade.notional_usdc
        market_trade_count[market_key] += 1

    feature_matrix = np.nan_to_num(
        np.stack(feature_rows) if feature_rows else np.empty((0, len(IF_FEATURE_NAMES))),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    logging.info(
        f"IF features: {feature_matrix.shape[0]} trades x "
        f"{feature_matrix.shape[1]} features"
    )
    return feature_matrix, trade_info


def run_isolation_forest_baseline(
    feature_matrix: np.ndarray,
    trade_info: List[Dict],
    market_ids: List[int],
    all_entries: List[Tuple[Trade, int]],
    winning_outcomes: Dict[int, Optional[int]],
    *,
    n_estimators: int = 100,
    contamination: str = "auto",
    random_state: int = 42,
    match_flag_rate: Optional[float] = None,
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
    min_usd_amount: Optional[float] = None,
    model: Optional[IsolationForest] = None,
    calibration_feature_matrix: Optional[np.ndarray] = None,
    calibration_trade_info: Optional[List[Dict]] = None,
) -> Dict:
    start = time.time()

    n_trades = len(trade_info)
    if n_trades == 0:
        return empty_if_result(match_flag_rate)

    if model is None:
        logging.info(f"IF baseline: fitting on {n_trades:,} trades...")
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(feature_matrix)
    else:
        logging.info(f"IF baseline: scoring {n_trades:,} eval trades with prefit model...")
    scores = model.decision_function(feature_matrix)

    fit_time = time.time() - start
    logging.info(f"IF baseline: model ready in {fit_time:.1f}s, scoring trades...")

    buy_eval_indices = [
        i for i, info in enumerate(trade_info)
        if info["trade"].side.upper() == "BUY"
        and info["winning_outcome"] is not None
    ]
    n_buy_eval = len(buy_eval_indices)

    natural_predictions = model.predict(feature_matrix)
    natural_anomaly_count = int(np.sum(natural_predictions == -1))

    flagged_indices: Set[int] = set()

    threshold_source = "natural_model_predict"
    if match_flag_rate is not None and n_buy_eval > 0:
        calibration_scores = None
        calibration_buy_indices = None
        if (
            calibration_feature_matrix is not None
            and calibration_trade_info is not None
            and len(calibration_trade_info) == calibration_feature_matrix.shape[0]
            and calibration_feature_matrix.shape[0] > 0
        ):
            candidate_scores = model.decision_function(calibration_feature_matrix)
            candidate_buy_indices = [
                i for i, info in enumerate(calibration_trade_info)
                if info["trade"].side.upper() == "BUY"
                and info["winning_outcome"] is not None
            ]
            if candidate_buy_indices:
                calibration_scores = candidate_scores
                calibration_buy_indices = candidate_buy_indices

        flagged_indices, threshold_source = _select_matched_buy_indices(
            scores,
            buy_eval_indices,
            match_flag_rate,
            calibration_scores=calibration_scores,
            calibration_buy_indices=calibration_buy_indices,
        )
    else:
        flagged_indices = {i for i, pred in enumerate(natural_predictions) if pred == -1}

    flagged_returns_all: List[float] = []
    flagged_notionals_all: List[float] = []
    unflagged_returns_all: List[float] = []
    per_market_flagged: Dict[int, List[float]] = defaultdict(list)
    per_market_unflagged: Dict[int, List[float]] = defaultdict(list)
    n_flagged_buy = 0
    n_unflagged_buy = 0

    for i, info in enumerate(trade_info):
        trade = info["trade"]
        market_id = info["market_id"]
        winning = info["winning_outcome"]

        if trade.side.upper() != "BUY" or winning is None:
            continue

        ret = _compute_resolution_return(trade, winning)

        if i in flagged_indices:
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
        all_entries, winning_outcomes, z_score_threshold, min_wallet_notional
    )

    if_wallet_flags: Dict[int, Set[str]] = defaultdict(set)
    for i, info in enumerate(trade_info):
        if i in flagged_indices:
            if_wallet_flags[info["market_id"]].add(info["trade"].wallet)

    wallet_pnl_stats = wallet_flagged_pnl_from_wallet_data(wallet_data, if_wallet_flags)
    copytrade_stats = copytrade_trade_summary(flagged_returns_all, flagged_notionals_all)

    tp_wallets: List[str] = []
    fp_wallets: List[str] = []
    fn_wallets: List[str] = []
    tp_returns: List[float] = []
    fp_returns: List[float] = []

    for market_id, w_evals in wallet_data.items():
        flagged_in_market = if_wallet_flags.get(market_id, set())
        for wallet, data in w_evals.items():
            is_insider = data["is_insider"]
            is_flagged = wallet in flagged_in_market
            if is_flagged and is_insider:
                tp_wallets.append(wallet)
                tp_returns.append(data["return"])
            elif is_flagged and not is_insider:
                fp_wallets.append(wallet)
                fp_returns.append(data["return"])
            elif not is_flagged and is_insider:
                fn_wallets.append(wallet)

    elapsed = time.time() - start

    logging.info(
        f"IF baseline: {n_flagged_buy:,} flagged BUY trades "
        f"({actual_flag_rate:.2%}), "
        f"TP={len(tp_wallets)}, FP={len(fp_wallets)}, FN={len(fn_wallets)}, "
        f"wall={elapsed:.1f}s"
    )

    return {
        "baseline": "isolation_forest",
        "num_flags": int(len(flagged_indices)),
        "flagged_trades": n_flagged_buy,
        "unflagged_trades": n_unflagged_buy,
        "flagged_mean_return": pooled_flagged_mean,
        "unflagged_mean_return": pooled_unflagged_mean,
        "mean_return_diff": pooled_mean_return_diff,
        "mean_cohens_d": mean_cohens_d,
        "sig_welch_p05": sig_welch,
        "n_markets": n_markets_eval,
        "tp": len(tp_wallets),
        "fp": len(fp_wallets),
        "fn": len(fn_wallets),
        "flagged_avg_return": wallet_pnl_stats["flagged_wallet_mean_return"],
        "tp_avg_return": float(np.mean(tp_returns)) if tp_returns else 0.0,
        "fp_avg_return": float(np.mean(fp_returns)) if fp_returns else 0.0,
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
        "if_n_estimators": n_estimators,
        "if_flag_rate": actual_flag_rate,
        "if_natural_anomalies": natural_anomaly_count,
        "if_contamination": contamination,
        "match_flag_rate": match_flag_rate,
        "if_match_threshold_source": threshold_source,
        "min_usd_amount": min_usd_amount,
        "deployable_live": False,
    }


def empty_if_result(match_flag_rate: Optional[float]) -> Dict:
    return {
        "baseline": "isolation_forest",
        "num_flags": 0,
        "flagged_trades": 0, "unflagged_trades": 0,
        "flagged_mean_return": 0.0, "unflagged_mean_return": 0.0,
        "mean_return_diff": 0.0, "mean_cohens_d": 0.0,
        "sig_welch_p05": 0, "n_markets": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "flagged_avg_return": 0.0,
        "tp_avg_return": 0.0, "fp_avg_return": 0.0,
        "trades_per_second": 0.0, "det_p95_us": 0.0, "wall_clock_s": 0.0,
        "flagged_wallet_mean_net_pnl": 0.0,
        "flagged_wallet_median_net_pnl": 0.0,
        "copytrade_total_flagged_buys": 0,
        "copytrade_total_capital_deployed": 0.0,
        "copytrade_total_pnl": 0.0,
        "copytrade_portfolio_roi": 0.0,
        "copytrade_win_rate": 0.0,
        "copytrade_mean_trade_return": 0.0,
        "copytrade_median_trade_return": 0.0,
        "if_n_estimators": 0, "if_flag_rate": 0.0,
        "if_natural_anomalies": 0, "if_contamination": "auto",
        "match_flag_rate": match_flag_rate,
        "if_match_threshold_source": "empty",
        "min_usd_amount": None,
        "deployable_live": False,
    }
