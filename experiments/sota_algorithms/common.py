"""
Shared helpers for SOTA baseline comparisons (compare_sota experiment).
"""

from __future__ import annotations

from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import predict_wallet_positive
from backtesting.market_resolutions import get_market_info

def parse_resolution_date(date_str: str) -> Optional[int]:
    dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def parse_datetime_to_ms(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return int(ts.timestamp() * 1000)


def parse_iso_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    return ts


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize_pooled_returns(
    flagged_returns: List[float],
    unflagged_returns: List[float],
) -> Dict[str, float]:
    flagged_mean = float(np.mean(flagged_returns)) if flagged_returns else 0.0
    unflagged_mean = float(np.mean(unflagged_returns)) if unflagged_returns else 0.0
    mean_return_diff = (
        flagged_mean - unflagged_mean
        if flagged_returns and unflagged_returns
        else 0.0
    )
    return {
        "flagged_mean_return": flagged_mean,
        "unflagged_mean_return": unflagged_mean,
        "mean_return_diff": mean_return_diff,
    }


def get_market_resolution_timestamp_ms(
    loader: HistoricalDataLoader,
    market_id: int,
) -> Optional[int]:
    metadata = loader.get_market_metadata(market_id) or {}
    closed_ts = parse_datetime_to_ms(metadata.get("closedTime"))
    if closed_ts is not None:
        return closed_ts

    info = get_market_info(market_id) or {}
    resolution_date = info.get("resolution_date")
    if resolution_date is None:
        return None

    parsed_ts = parse_datetime_to_ms(resolution_date)
    if parsed_ts is not None:
        return parsed_ts

    try:
        return parse_resolution_date(str(resolution_date))
    except (TypeError, ValueError):
        return None


def copytrade_trade_summary(flagged_returns: List[float], flagged_notionals: List[float]) -> Dict[str, float]:
    if not flagged_returns:
        return {
            "copytrade_total_flagged_buys": 0,
            "copytrade_total_capital_deployed": 0.0,
            "copytrade_total_pnl": 0.0,
            "copytrade_portfolio_roi": 0.0,
            "copytrade_win_rate": 0.0,
            "copytrade_mean_trade_return": 0.0,
            "copytrade_median_trade_return": 0.0,
        }

    r = np.asarray(flagged_returns, dtype=float)
    n = np.asarray(flagged_notionals, dtype=float)
    total_capital = float(np.sum(n))
    total_pnl = float(np.sum(r * n))
    win_rate = float(np.mean(r > -0.999999))
    return {
        "copytrade_total_flagged_buys": int(r.size),
        "copytrade_total_capital_deployed": total_capital,
        "copytrade_total_pnl": total_pnl,
        "copytrade_portfolio_roi": (total_pnl / total_capital) if total_capital > 1e-12 else 0.0,
        "copytrade_win_rate": win_rate,
        "copytrade_mean_trade_return": float(np.mean(r)),
        "copytrade_median_trade_return": float(np.median(r)),
    }


def wallet_flagged_pnl_from_wallet_data(
    wallet_data: Dict[int, Dict[str, Dict]],
    flagged_wallets_by_market: Dict[int, Set[str]],
) -> Dict[str, float]:
    flagged_net_pnls: List[float] = []
    flagged_returns: List[float] = []
    for market_id, wallets in flagged_wallets_by_market.items():
        w_evals = wallet_data.get(market_id, {})
        for wallet in wallets:
            if wallet in w_evals:
                flagged_net_pnls.append(float(w_evals[wallet].get("net_pnl", 0.0)))
                flagged_returns.append(float(w_evals[wallet].get("return", 0.0)))

    return {
        "flagged_wallet_mean_net_pnl": safe_mean(flagged_net_pnls),
        "flagged_wallet_median_net_pnl": safe_median(flagged_net_pnls),
        "flagged_wallet_mean_return": safe_mean(flagged_returns),
        "flagged_wallet_median_return": safe_median(flagged_returns),
    }


def wallet_classification_metrics(
    wallet_data: Dict[int, Dict[str, Dict]],
    flagged_wallets_by_market: Dict[int, Set[str]],
) -> Dict[str, object]:
    tp_wallets: List[str] = []
    fp_wallets: List[str] = []
    fn_wallets: List[str] = []
    flagged_returns: List[float] = []
    tp_returns: List[float] = []
    fp_returns: List[float] = []

    for market_id, w_evals in wallet_data.items():
        flagged_in_market = flagged_wallets_by_market.get(market_id, set())
        for wallet, data in w_evals.items():
            is_insider = data["is_insider"]
            is_flagged = wallet in flagged_in_market
            if is_flagged:
                flagged_returns.append(data["return"])
            if is_flagged and is_insider:
                tp_wallets.append(wallet)
                tp_returns.append(data["return"])
            elif is_flagged and not is_insider:
                fp_wallets.append(wallet)
                fp_returns.append(data["return"])
            elif not is_flagged and is_insider:
                fn_wallets.append(wallet)

    return {
        "tp": len(tp_wallets),
        "fp": len(fp_wallets),
        "fn": len(fn_wallets),
        "flagged_avg_return": safe_mean(flagged_returns),
        "tp_avg_return": safe_mean(tp_returns),
        "fp_avg_return": safe_mean(fp_returns),
    }


def build_wallet_prior_trade_counts_by_market(
    all_entries: List[Tuple[object, int]],
) -> Dict[int, Dict[str, int]]:
    """
    Count wallet trades before each market's first observed trade.

    This gives per-market heuristics a platform-history seed instead of treating
    every wallet's first trade in each market as a fresh wallet.
    """
    first_ts_by_market: Dict[int, int] = {}
    for trade, market_id in all_entries:
        mid = int(market_id)
        ts = int(trade.timestamp_ms)
        if mid not in first_ts_by_market or ts < first_ts_by_market[mid]:
            first_ts_by_market[mid] = ts

    result: Dict[int, Dict[str, int]] = {}
    if not first_ts_by_market:
        return result

    markets_by_first_ts: Dict[int, List[int]] = defaultdict(list)
    for market_id, first_ts in first_ts_by_market.items():
        markets_by_first_ts[int(first_ts)].append(int(market_id))

    sorted_entries = sorted(all_entries, key=lambda x: int(x[0].timestamp_ms))
    wallet_counts: Dict[str, int] = defaultdict(int)
    entry_idx = 0
    n_entries = len(sorted_entries)

    for first_ts in sorted(markets_by_first_ts):
        while entry_idx < n_entries and int(sorted_entries[entry_idx][0].timestamp_ms) < first_ts:
            trade, _market_id = sorted_entries[entry_idx]
            wallet_counts[str(trade.wallet)] += 1
            entry_idx += 1

        snapshot = dict(wallet_counts)
        for market_id in markets_by_first_ts[first_ts]:
            result[int(market_id)] = dict(snapshot)
    return result


def wallet_flagged_pnl_from_evaluations(
    wallet_evaluations: List[Dict],
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
) -> Dict[str, float]:
    flagged_net_pnls = [
        float(e.get("net_pnl", 0.0))
        for e in wallet_evaluations
        if predict_wallet_positive(e, prediction_mode, suspicion_threshold, flag_rate_threshold)
    ]
    return {
        "flagged_wallet_mean_net_pnl": safe_mean(flagged_net_pnls),
        "flagged_wallet_median_net_pnl": safe_median(flagged_net_pnls),
    }


def build_wallet_insider_labels(
    all_entries: List[tuple],
    winning_outcomes: Dict[int, Optional[int]],
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
) -> Dict[int, Dict[str, Dict]]:
    """
    Build wallet positions and compute z-score-based insider labels per market.

    Returns:
        {market_id: {wallet: {"is_insider": bool, "return": float, "gross_buy": float, ...}}}
    """
    from collections import defaultdict

    positions: Dict[int, Dict[str, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    costs: Dict[int, Dict[str, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    gross_buy: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_notional: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for trade, market_id in all_entries:
        if winning_outcomes.get(market_id) is None:
            continue
        wallet = trade.wallet
        oi = trade.outcome_index
        total_notional[market_id][wallet] += trade.notional_usdc

        if trade.side.upper() == "BUY":
            positions[market_id][wallet][oi] += trade.size_tokens
            costs[market_id][wallet][oi] += trade.notional_usdc
            gross_buy[market_id][wallet] += trade.notional_usdc
        else:
            positions[market_id][wallet][oi] -= trade.size_tokens
            costs[market_id][wallet][oi] -= trade.notional_usdc

    result: Dict[int, Dict[str, Dict]] = {}

    for market_id in total_notional:
        winning = winning_outcomes.get(market_id)
        if winning is None:
            continue

        wallet_rows: List[Dict] = []
        for wallet in total_notional[market_id]:
            if total_notional[market_id][wallet] < min_wallet_notional:
                continue
            gb = gross_buy[market_id][wallet]
            if gb < 1e-9:
                continue

            total_payout = sum(
                shares for oi, shares in positions[market_id][wallet].items()
                if oi == winning
            )
            total_cost = sum(costs[market_id][wallet].values())

            net_pnl = total_payout - total_cost
            wallet_return = net_pnl / gb if gb > 1e-9 else 0.0

            wallet_rows.append({
                "wallet": wallet,
                "return": wallet_return,
                "net_pnl": net_pnl,
                "gross_buy": gb,
                "total_notional": total_notional[market_id][wallet],
            })

        if len(wallet_rows) < 3:
            continue

        returns = np.array([r["return"] for r in wallet_rows])
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

        wallet_evals = {}
        for row in wallet_rows:
            z = (row["return"] - mean_ret) / std_ret if std_ret > 1e-12 else 0.0
            wallet_evals[row["wallet"]] = {
                "is_insider": bool(z > z_score_threshold),
                "return": row["return"],
                "net_pnl": row["net_pnl"],
                "gross_buy": row["gross_buy"],
                "z_score": float(z),
            }
        result[market_id] = wallet_evals

    return result
