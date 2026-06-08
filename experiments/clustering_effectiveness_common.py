"""
Clustering effectiveness experiment.

Typical usage::

    # Optimize, then backtest on the same timeframe:
    python -m experiments.clustering_effectiveness_common \\
        --start-date 2025-02-01 --end-date 2025-02-15 \\
        --enable-trade-prefilter --min-usd-amount 500 \\
        --enable-layer2-attribution

    # Backtest a pre-made config on a different timeframe:
    python -m experiments.clustering_effectiveness_common \\
        --start-date 2025-03-01 --end-date 2025-03-31 \\
        --config-path experiments/results/timeframe_optimize.../best_config.json \\
        --enable-layer2-attribution
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from backtesting.data_loader import HistoricalDataLoader
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode
from backtesting.market_resolutions import get_winning_outcome
from backtesting.parameter_optimizer import _calculate_metrics_from_wallet_evaluations
from backtesting.trade_event_study import infer_market_winning_outcome_from_last_prices
from models import filter_trades_by_notional

from experiments.timeframe_market_common import run_timeframe_backtest_evaluation
from experiments.timeframe_experiment_common import (
    add_multi_start_args,
    add_standard_timeframe_optimizer_args,
    prepare_timeframe_inference,
    setup_timeframe_logging,
)
from experiments.timeframe_optimizers import run_multi_start_alternating_timeframe


DEFAULT_OUTPUT_DIR = "experiments/results/clustering_effectiveness"
DEFAULT_BOOST_BUCKETS = "1.0,1.2,1.4,1.6,1.8,2.0"
BOOST_EPS = 1e-6

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest (and optionally optimize) a detector + clustering config, "
            "then report how much the clustering layer actually helps by "
            "slicing flagged wallets on cluster membership, boost magnitude, "
            "decisiveness, and Layer-2 common-ownership attribution."
        ),
    )

    # Backtest timeframe
    parser.add_argument("--start-date", type=str, required=True,
                        help="Inclusive ISO start date for the evaluation backtest.")
    parser.add_argument("--end-date", type=str, required=True,
                        help="Inclusive ISO end date for the evaluation backtest.")

    # Config source
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Path to a pre-made config JSON (e.g. the output of a prior "
            "multi-start optimizer run).  When provided, optimization is "
            "skipped entirely."
        ),
    )

    # Optional distinct optimization timeframe
    parser.add_argument(
        "--opt-start-date",
        type=str,
        default=None,
        help=(
            "Start date for the optimization timeframe.  Defaults to "
            "--start-date (optimize on the same markets we backtest on)."
        ),
    )
    parser.add_argument(
        "--opt-end-date",
        type=str,
        default=None,
        help="End date for the optimization timeframe.  Defaults to --end-date.",
    )

    # Market filters (applied to both opt and backtest timeframes)
    parser.add_argument(
        "--insider-plausible-only",
        action="store_true",
        help="Filter to markets classified as insider-plausible.",
    )
    parser.add_argument(
        "--non-insider-plausible-only",
        action="store_true",
        help="Filter to markets classified as non-insider-plausible.",
    )

    parser.add_argument(
        "--suspicion-threshold",
        type=float,
        default=2.0,
        help="Suspicion score threshold for suspicion_threshold prediction mode.",
    )
    parser.add_argument(
        "--copytrade-fixed-size",
        type=float,
        default=100.0,
        help="Fixed trade size for the copytrade simulation portion of the backtest.",
    )
    parser.add_argument(
        "--verbose-output",
        action="store_true",
        default=False,
        help="Print per-market progress and detailed backtest logging.",
    )

    parser.add_argument(
        "--boost-buckets",
        type=str,
        default=DEFAULT_BOOST_BUCKETS,
        help=(
            "Comma-separated bucket edges for the boost-magnitude histogram.  "
            f"Default: '{DEFAULT_BOOST_BUCKETS}'.  Buckets are half-open "
            "(left, right], except for the leading [1.0] exact-match bucket "
            "reserved for wallets with no cluster membership."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run artifacts are written.",
    )

    add_standard_timeframe_optimizer_args(parser)
    add_multi_start_args(parser)

    return parser

def _parse_boost_buckets(spec: str) -> List[float]:
    edges = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(edges) < 2:
        raise SystemExit(
            f"--boost-buckets needs at least two edges; got {spec!r}"
        )
    if any(b < a for a, b in zip(edges, edges[1:])):
        raise SystemExit(
            f"--boost-buckets must be non-decreasing; got {edges!r}"
        )
    return edges


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _cohort_summary(rows: List[Dict]) -> Dict[str, Any]:
    """Compute wallet-level economic and classification stats for a cohort."""
    n = len(rows)
    if n == 0:
        return {
            "count": 0,
            "total_net_pnl": 0.0,
            "mean_net_pnl": 0.0,
            "median_net_pnl": 0.0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "weighted_return": 0.0,
            "precision": 0.0,
            "insiders": 0,
            "mean_boost": 1.0,
            "total_gross_buy": 0.0,
            "mean_gross_buy": 0.0,
            "share_common_ownership": 0.0,
        }
    net_pnls = [float(r.get("net_pnl", 0.0) or 0.0) for r in rows]
    returns = [float(r.get("return", 0.0) or 0.0) for r in rows]
    boosts = [float(r.get("cluster_boost", 1.0) or 1.0) for r in rows]
    gross = [float(r.get("gross_buy_notional", 0.0) or 0.0) for r in rows]
    insiders = sum(1 for r in rows if bool(r.get("is_insider", False)))
    common_own = sum(1 for r in rows if bool(r.get("has_common_ownership", False)))
    total_net_pnl = float(sum(net_pnls))
    total_gross_buy = float(sum(gross))
    return {
        "count": n,
        "total_net_pnl": total_net_pnl,
        "mean_net_pnl": float(total_net_pnl / n),
        "median_net_pnl": float(statistics.median(net_pnls)),
        "mean_return": float(sum(returns) / n),
        "median_return": float(statistics.median(returns)),
        "weighted_return": (
            sum(ret * weight for ret, weight in zip(returns, gross)) / total_gross_buy
            if total_gross_buy > 1e-9 else 0.0
        ),
        "precision": float(insiders / n),
        "insiders": int(insiders),
        "mean_boost": float(sum(boosts) / n),
        "total_gross_buy": total_gross_buy,
        "mean_gross_buy": float(total_gross_buy / n),
        "share_common_ownership": float(common_own / n),
    }


def _print_cohort_table(
    title: str,
    rows_by_label: List[Tuple[str, List[Dict]]],
    *,
    total_rows: Optional[List[Dict]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Print a pretty per-cohort stats table and return the structured version."""
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)
    header = (
        f"{'cohort':<32s} {'n':>6s} {'mean_pnl':>11s} {'median_pnl':>12s} "
        f"{'mean_ret':>9s} {'med_ret':>9s} {'weighted':>9s} "
        f"{'precision':>10s} {'mean_boost':>10s}"
    )
    print(header)
    print("-" * len(header))

    out: Dict[str, Dict[str, Any]] = {}
    total = len(total_rows) if total_rows is not None else None

    for label, rows in rows_by_label:
        s = _cohort_summary(rows)
        share_str = ""
        if total is not None and total > 0:
            share_str = f"  ({100.0 * s['count'] / total:5.1f}% of flagged)"
        out[label] = s
        print(
            f"{label:<32s} "
            f"{s['count']:>6,d} "
            f"${s['mean_net_pnl']:>+10,.2f} "
            f"${s['median_net_pnl']:>+11,.2f} "
            f"{s['mean_return']:>+9.2%} "
            f"{s['median_return']:>+9.2%} "
            f"{s['weighted_return']:>+9.2%} "
            f"{s['precision']:>9.2%} "
            f"{s['mean_boost']:>10.3f}"
            f"{share_str}"
        )
    return out


def _bucket_label(edges: List[float], idx: int) -> str:
    if idx == -1:
        return "boost = 1.0 (no cluster)"
    lo = edges[idx]
    hi = edges[idx + 1]
    lo_sym = "(" if idx > 0 else "["
    return f"{lo_sym}{lo:.2f}, {hi:.2f}]"


def _assign_boost_bucket(boost: float, edges: List[float]) -> int:
    """Return bucket index for a boost value, or -1 for exactly 1.0 (no cluster)."""
    if boost <= 1.0 + BOOST_EPS:
        return -1
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0:
            if lo - BOOST_EPS <= boost <= hi + BOOST_EPS:
                return i
        else:
            if lo + BOOST_EPS < boost <= hi + BOOST_EPS:
                return i
    # Above last edge: lump into last bucket
    return len(edges) - 2


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class ClusteringEffectivenessReport:
    report: Dict[str, Any]
    per_wallet_rows: List[Dict[str, Any]]
    boost_bucket_rows: List[Dict[str, Any]]
    trade_cohort_rows: List[Dict[str, Any]]
    trade_alert_boost_wallet_rows: List[Dict[str, Any]]
    cluster_return_rows: List[Dict[str, Any]]


def _wallet_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(row.get("wallet", "")).lower(),
        str(row.get("market_slug", "")),
    )


def _predict_wallet_positive(
    row: Dict[str, Any],
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
) -> bool:
    mode = str(prediction_mode)
    if mode == "has_alert":
        return bool(row.get("has_alert", False))
    if mode == "suspicion_threshold":
        return float(row.get("suspicion_score", 0.0) or 0.0) >= float(suspicion_threshold)
    if mode in {"flag_rate", "boosted_flag_rate"}:
        trade_count = int(row.get("trade_count", 0) or 0)
        num_flags = int(row.get("num_flags", 0) or 0)
        rate = (num_flags / trade_count) if trade_count > 0 else 0.0
        return rate >= float(flag_rate_threshold)
    raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")


def _annotate_wallet_rows(
    with_clustering_wallet_evaluations: List[Dict],
    without_clustering_wallet_evaluations: List[Dict],
    clustering_diagnostic_wallet_evaluations: List[Dict],
    flag_rate_threshold: float,
    prediction_mode: str,
    suspicion_threshold: float,
    boost_edges: List[float],
) -> List[Dict[str, Any]]:
    """
    Build per-wallet annotations from causal replay counterfactuals:
    with-clustering vs without-clustering.
    """
    without_by_key = {
        _wallet_key(row): row for row in without_clustering_wallet_evaluations
    }
    diagnostic_by_key = {
        _wallet_key(row): row for row in clustering_diagnostic_wallet_evaluations
    }

    annotated: List[Dict[str, Any]] = []
    for with_row in with_clustering_wallet_evaluations:
        key = _wallet_key(with_row)
        without_row = without_by_key.get(key, {})
        diagnostic_row = diagnostic_by_key.get(key)
        if diagnostic_row is None:
            diagnostic_row = with_row
            missing_diagnostic = True
        else:
            missing_diagnostic = False
        missing_counterfactual = not bool(without_row)

        trade_count_with = int(with_row.get("trade_count", 0) or 0)
        num_flags_with = int(with_row.get("num_flags", 0) or 0)
        rate_with = (num_flags_with / trade_count_with) if trade_count_with > 0 else 0.0

        trade_count_without = int(
            without_row.get("trade_count", trade_count_with) or trade_count_with
        )
        num_flags_without = int(without_row.get("num_flags", 0) or 0)
        rate_without = (
            (num_flags_without / trade_count_without) if trade_count_without > 0 else 0.0
        )

        flagged_with = _predict_wallet_positive(
            with_row,
            prediction_mode,
            suspicion_threshold,
            flag_rate_threshold,
        )
        flagged_without = _predict_wallet_positive(
            {
                "trade_count": trade_count_without,
                "num_flags": num_flags_without,
                "has_alert": bool(without_row.get("has_alert", False)),
                "suspicion_score": float(without_row.get("suspicion_score", 0.0) or 0.0),
            },
            prediction_mode,
            suspicion_threshold,
            flag_rate_threshold,
        )

        boost = float(diagnostic_row.get("cluster_boost", 1.0) or 1.0)
        combined_boost = float(with_row.get("cluster_boost", 1.0) or 1.0)
        in_cluster = boost > 1.0 + BOOST_EPS
        bucket_idx = _assign_boost_bucket(boost, boost_edges)
        alerts_delta = int(num_flags_with - num_flags_without)

        annotated.append(
            {
                **with_row,
                "counterfactual_missing": missing_counterfactual,
                "diagnostic_missing": missing_diagnostic,
                "trade_count_with_clustering": trade_count_with,
                "trade_count_without_clustering": trade_count_without,
                "num_flags_with_clustering": num_flags_with,
                "num_flags_without_clustering": num_flags_without,
                "flag_rate_with_clustering": rate_with,
                "flag_rate_without_clustering": rate_without,
                "suspicion_with_clustering": float(with_row.get("suspicion_score", 0.0) or 0.0),
                "suspicion_without_clustering": float(
                    without_row.get("suspicion_score", 0.0) or 0.0
                ),
                "flagged_with_clustering": bool(flagged_with),
                "flagged_without_clustering": bool(flagged_without),
                "decisive": bool(flagged_with and not flagged_without),
                "suppressed": bool(flagged_without and not flagged_with),
                "alerts_delta": alerts_delta,
                "combined_boost": combined_boost,
                "cluster_boost": boost,
                "has_common_ownership": bool(
                    diagnostic_row.get("has_common_ownership", with_row.get("has_common_ownership", False))
                ),
                "in_cluster": in_cluster,
                "boost_bucket_idx": bucket_idx,
                "boost_bucket_label": _bucket_label(boost_edges, bucket_idx),
            }
        )

    return annotated


def _trade_key(wallet: str, timestamp_ms: int) -> Tuple[str, int]:
    return (str(wallet).lower(), int(timestamp_ms))


def _build_flagged_info_by_key(
    backtest_result: Any,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    info_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if backtest_result is None:
        return info_by_key
    for wallet, flags in getattr(backtest_result, "wallet_flags", {}).items():
        w = str(wallet).lower()
        for flag_entry in flags or []:
            ts = int(flag_entry.get("timestamp_ms", 0) or 0)
            info_by_key[(w, ts)] = {
                "score": float(flag_entry.get("score", 0.0) or 0.0),
                "detectors": list(flag_entry.get("detectors", []) or []),
            }
    return info_by_key


def _compute_buy_trade_resolution_pnl_and_return(
    trade: Any,
    winning_outcome: int,
) -> Tuple[float, float, bool]:
    capital = float(getattr(trade, "notional_usdc", 0.0) or 0.0)
    is_win = int(getattr(trade, "outcome_index")) == int(winning_outcome)
    payout = float(getattr(trade, "size_tokens", 0.0) or 0.0) if is_win else 0.0
    pnl = payout - capital
    trade_return = pnl / capital if capital > 1e-9 else 0.0
    return pnl, trade_return, is_win


def _resolve_winning_outcome_for_market(
    market_id: int,
    all_trades: List[Any],
    resolution_threshold: float,
) -> Optional[int]:
    winning = get_winning_outcome(int(market_id))
    if winning is None:
        winning = infer_market_winning_outcome_from_last_prices(
            all_trades,
            threshold=float(resolution_threshold),
        )
    if winning is None:
        return None
    return int(winning)


def _trade_cohort_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "count": 0,
            "total_notional": 0.0,
            "total_pnl": 0.0,
            "mean_pnl": 0.0,
            "median_pnl": 0.0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "weighted_return": 0.0,
            "win_rate": 0.0,
        }

    pnls = [float(r.get("trade_pnl", 0.0) or 0.0) for r in rows]
    returns = [float(r.get("trade_return", 0.0) or 0.0) for r in rows]
    notionals = [float(r.get("notional_usdc", 0.0) or 0.0) for r in rows]
    wins = sum(1 for r in rows if bool(r.get("is_win", False)))
    total_notional = float(sum(notionals))
    total_pnl = float(sum(pnls))
    return {
        "count": n,
        "total_notional": total_notional,
        "total_pnl": total_pnl,
        "mean_pnl": float(sum(pnls) / n),
        "median_pnl": float(statistics.median(pnls)),
        "mean_return": float(sum(returns) / n),
        "median_return": float(statistics.median(returns)),
        "weighted_return": (total_pnl / total_notional) if total_notional > 1e-9 else 0.0,
        "win_rate": float(wins / n),
    }


def _print_trade_cohort_table(
    title: str,
    rows_by_label: List[Tuple[str, List[Dict[str, Any]]]],
) -> Dict[str, Dict[str, Any]]:
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)
    header = (
        f"{'cohort':<40s} {'n':>8s} {'total_pnl':>14s} {'mean_pnl':>12s} "
        f"{'mean_ret':>10s} {'med_ret':>10s} {'weighted':>10s} {'win_rate':>10s}"
    )
    print(header)
    print("-" * len(header))
    out: Dict[str, Dict[str, Any]] = {}
    for label, rows in rows_by_label:
        s = _trade_cohort_summary(rows)
        out[label] = s
        print(
            f"{label:<40s} "
            f"{int(s['count']):>8,d} "
            f"${float(s['total_pnl']):>+13,.2f} "
            f"${float(s['mean_pnl']):>+11,.2f} "
            f"{float(s['mean_return']):>+10.2%} "
            f"{float(s['median_return']):>+10.2%} "
            f"{float(s['weighted_return']):>+10.2%} "
            f"{float(s['win_rate']):>9.2%}"
        )
    return out


def _print_trade_alert_boost_wallet_table(
    title: str,
    cohorts: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]],
) -> Dict[str, Dict[str, Any]]:
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)
    header = (
        f"{'cohort':<40s} {'trades':>8s} {'wallets':>8s} "
        f"{'mean_pnl':>12s} {'median_pnl':>12s} {'mean_ret':>10s} "
        f"{'weighted':>10s} {'precision':>10s} {'trade_boost':>11s}"
    )
    print(header)
    print("-" * len(header))

    out: Dict[str, Dict[str, Any]] = {}
    for label, trade_rows, wallet_rows in cohorts:
        s = _cohort_summary(wallet_rows)
        multipliers = [
            float(r.get("cluster_multiplier", 1.0) or 1.0)
            for r in trade_rows
        ]
        s["trade_count"] = int(len(trade_rows))
        s["wallet_count"] = int(s["count"])
        s["mean_trade_cluster_multiplier"] = (
            float(sum(multipliers) / len(multipliers)) if multipliers else 1.0
        )
        out[label] = s
        print(
            f"{label:<40s} "
            f"{int(s['trade_count']):>8,d} "
            f"{int(s['wallet_count']):>8,d} "
            f"${float(s['mean_net_pnl']):>+11,.2f} "
            f"${float(s['median_net_pnl']):>+11,.2f} "
            f"{float(s['mean_return']):>+10.2%} "
            f"{float(s['weighted_return']):>+10.2%} "
            f"{float(s['precision']):>9.2%} "
            f"{float(s['mean_trade_cluster_multiplier']):>11.3f}"
        )
    return out


def _build_trade_buy_alert_counterfactual_report(
    *,
    loader: HistoricalDataLoader,
    market_ids: Sequence[int],
    with_backtest_results: Dict[int, Any],
    without_backtest_results: Dict[int, Any],
    min_usd_amount: Optional[float],
    resolution_threshold: float,
    winning_outcomes_override: Optional[Dict[int, int]] = None,
    clustering_config: Optional[Dict[str, Any]] = None,
    clustering_min_trade_size: float = 5000.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compare BUY-trade economics for:
      - boosted into alert (flagged with clustering, not flagged without)
      - boosted but would alert anyway (flagged in both, score increased)
      - not flagged (with clustering)
    """
    boosted_to_alert_rows: List[Dict[str, Any]] = []
    boosted_anyway_rows: List[Dict[str, Any]] = []
    cluster_boosted_any_rows: List[Dict[str, Any]] = []
    not_flagged_rows: List[Dict[str, Any]] = []

    markets_with_backtests = 0
    markets_with_resolution = 0
    buy_trades_scanned = 0

    for market_id in market_ids:
        with_result = with_backtest_results.get(int(market_id))
        without_result = without_backtest_results.get(int(market_id))
        if with_result is None or without_result is None:
            continue
        markets_with_backtests += 1

        try:
            all_trades = loader.get_trades_for_market(
                market_id=int(market_id),
                min_usd_amount=None,
                use_cache=False,
            )
        except TypeError:
            all_trades = loader.get_trades_for_market(int(market_id))

        if min_usd_amount is not None:
            detector_trades = filter_trades_by_notional(all_trades, min_usd_amount)
        else:
            detector_trades = all_trades

        winning_outcome = None
        if winning_outcomes_override is not None and int(market_id) in winning_outcomes_override:
            winning_outcome = int(winning_outcomes_override[int(market_id)])
        if winning_outcome is None:
            winning_outcome = _resolve_winning_outcome_for_market(
                int(market_id),
                all_trades,
                resolution_threshold,
            )
        if winning_outcome is None:
            continue
        markets_with_resolution += 1

        metadata = loader.get_market_metadata(int(market_id)) or {}
        market_slug = str(metadata.get("market_slug", market_id))
        flagged_with = _build_flagged_info_by_key(with_result)
        flagged_without = _build_flagged_info_by_key(without_result)
        cluster_multipliers = None
        if clustering_config:
            from backtesting.causal_boost_replay import build_live_parity_boost_schedule

            schedule = build_live_parity_boost_schedule(
                detector_trades=detector_trades,
                market_id=str(market_id),
                clustering_config=clustering_config,
                clustering_min_trade_size=float(clustering_min_trade_size),
                jump_anticipation_config=None,
                attribution_provider=None,
                fetch_if_missing=False,
            )
            cluster_multipliers = schedule.cluster_multiplier_by_trade_idx

        for idx, trade in enumerate(detector_trades):
            if str(getattr(trade, "side", "")).upper() != "BUY":
                continue
            buy_trades_scanned += 1
            cluster_multiplier = (
                float(cluster_multipliers[idx])
                if cluster_multipliers is not None
                else 1.0
            )
            cluster_boosted = bool(cluster_multiplier > 1.0 + BOOST_EPS)

            key = _trade_key(
                str(getattr(trade, "wallet", "")),
                int(getattr(trade, "timestamp_ms", 0) or 0),
            )
            flagged_in_with = key in flagged_with
            flagged_in_without = key in flagged_without
            score_with = float(flagged_with.get(key, {}).get("score", 0.0) or 0.0)
            score_without = float(flagged_without.get(key, {}).get("score", 0.0) or 0.0)
            score_delta = float(score_with - score_without)
            score_boosted = bool(flagged_in_with and score_delta > BOOST_EPS)
            trade_pnl, trade_return, is_win = _compute_buy_trade_resolution_pnl_and_return(
                trade,
                winning_outcome,
            )
            row = {
                "market_id": int(market_id),
                "market_slug": market_slug,
                "wallet": str(getattr(trade, "wallet", "")),
                "timestamp_ms": int(getattr(trade, "timestamp_ms", 0) or 0),
                "notional_usdc": float(getattr(trade, "notional_usdc", 0.0) or 0.0),
                "price": float(getattr(trade, "price", 0.0) or 0.0),
                "size_tokens": float(getattr(trade, "size_tokens", 0.0) or 0.0),
                "outcome_index": int(getattr(trade, "outcome_index")),
                "winning_outcome": int(winning_outcome),
                "trade_pnl": float(trade_pnl),
                "trade_return": float(trade_return),
                "is_win": bool(is_win),
                "flagged_with_clustering": bool(flagged_in_with),
                "flagged_without_clustering": bool(flagged_in_without),
                "score_with_clustering": score_with,
                "score_without_clustering": score_without,
                "score_delta_with_minus_without": score_delta,
                "score_boosted_by_clustering": score_boosted,
                "cluster_multiplier": cluster_multiplier,
                "cluster_boosted": cluster_boosted,
                "boosted_to_alert": bool(flagged_in_with and not flagged_in_without),
                "boosted_but_would_alert_anyway": bool(
                    flagged_in_with and flagged_in_without and score_boosted
                ),
            }
            if row["boosted_to_alert"]:
                boosted_to_alert_rows.append(row)
            if row["boosted_but_would_alert_anyway"]:
                boosted_anyway_rows.append(row)
            if row["cluster_boosted"]:
                cluster_boosted_any_rows.append(row)
            if not row["flagged_with_clustering"]:
                not_flagged_rows.append(row)

    cohorts = [
        ("boosted_to_buy_alert", boosted_to_alert_rows),
        ("boosted_but_would_buy_alert_anyway", boosted_anyway_rows),
        ("cluster_boosted_any_trade", cluster_boosted_any_rows),
        ("not_flagged_with_clustering", not_flagged_rows),
    ]
    cohort_summaries = _print_trade_cohort_table(
        "TRADE-LEVEL BUY ALERT EFFECT (counterfactual clustering off)",
        cohorts,
    )

    boosted_s = cohort_summaries.get("boosted_to_buy_alert", {})
    boosted_anyway_s = cohort_summaries.get("boosted_but_would_buy_alert_anyway", {})
    cluster_boosted_any_s = cohort_summaries.get("cluster_boosted_any_trade", {})
    not_flagged_s = cohort_summaries.get("not_flagged_with_clustering", {})
    comparisons = {
        "mean_return_diff_boosted_minus_not_flagged": float(
            boosted_s.get("mean_return", 0.0) - not_flagged_s.get("mean_return", 0.0)
        ),
        "weighted_return_diff_boosted_minus_not_flagged": float(
            boosted_s.get("weighted_return", 0.0) - not_flagged_s.get("weighted_return", 0.0)
        ),
        "mean_pnl_diff_boosted_minus_not_flagged": float(
            boosted_s.get("mean_pnl", 0.0) - not_flagged_s.get("mean_pnl", 0.0)
        ),
        "mean_return_diff_boosted_anyway_minus_not_flagged": float(
            boosted_anyway_s.get("mean_return", 0.0)
            - not_flagged_s.get("mean_return", 0.0)
        ),
        "weighted_return_diff_boosted_anyway_minus_not_flagged": float(
            boosted_anyway_s.get("weighted_return", 0.0)
            - not_flagged_s.get("weighted_return", 0.0)
        ),
        "mean_pnl_diff_boosted_anyway_minus_not_flagged": float(
            boosted_anyway_s.get("mean_pnl", 0.0) - not_flagged_s.get("mean_pnl", 0.0)
        ),
        "mean_return_diff_cluster_boosted_any_minus_not_flagged": float(
            cluster_boosted_any_s.get("mean_return", 0.0)
            - not_flagged_s.get("mean_return", 0.0)
        ),
        "weighted_return_diff_cluster_boosted_any_minus_not_flagged": float(
            cluster_boosted_any_s.get("weighted_return", 0.0)
            - not_flagged_s.get("weighted_return", 0.0)
        ),
        "mean_pnl_diff_cluster_boosted_any_minus_not_flagged": float(
            cluster_boosted_any_s.get("mean_pnl", 0.0)
            - not_flagged_s.get("mean_pnl", 0.0)
        ),
    }

    report = {
        "cohorts": cohort_summaries,
        "comparisons": comparisons,
        "meta": {
            "markets_requested": int(len(market_ids)),
            "markets_with_backtests": int(markets_with_backtests),
            "markets_with_resolution": int(markets_with_resolution),
            "buy_trades_scanned": int(buy_trades_scanned),
        },
    }
    trade_cohort_rows = [
        {"cohort": label, **stats} for label, stats in cohort_summaries.items()
    ]
    return report, trade_cohort_rows


def _wallet_eval_by_market_wallet(
    wallet_evaluations: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in wallet_evaluations:
        market_slug = str(row.get("market_slug", ""))
        wallet = str(row.get("wallet", "")).lower()
        if market_slug and wallet:
            rows[(market_slug, wallet)] = row
    return rows


def _unique_wallet_eval_rows_for_trade_rows(
    trade_rows: List[Dict[str, Any]],
    wallet_rows_by_key: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    seen: Set[Tuple[str, str]] = set()
    wallet_rows: List[Dict[str, Any]] = []
    missing_trade_rows = 0
    for trade_row in trade_rows:
        key = (
            str(trade_row.get("market_slug", "")),
            str(trade_row.get("wallet", "")).lower(),
        )
        wallet_row = wallet_rows_by_key.get(key)
        if wallet_row is None:
            missing_trade_rows += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        wallet_rows.append(wallet_row)
    return wallet_rows, missing_trade_rows


def _build_trade_alert_boost_wallet_eval_report(
    *,
    loader: HistoricalDataLoader,
    market_ids: Sequence[int],
    with_backtest_results: Dict[int, Any],
    without_backtest_results: Dict[int, Any],
    with_clustering_wallet_evaluations: List[Dict[str, Any]],
    min_usd_amount: Optional[float],
    clustering_config: Optional[Dict[str, Any]],
    clustering_min_trade_size: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Wallet-eval stats for all trades receiving a causal clustering multiplier.
    """
    if not clustering_config:
        cohorts = [
            ("cluster_boosted_to_alert", [], []),
            ("cluster_boosted_would_alert_anyway", [], []),
            ("cluster_boosted_any_trade", [], []),
            ("cluster_not_boosted_alert", [], []),
        ]
        summaries = _print_trade_alert_boost_wallet_table(
            "TRADE ALERT CLUSTER BOOST EFFECT (wallet-eval stats)",
            cohorts,
        )
        return {
            "cohorts": summaries,
            "meta": {
                "markets_requested": int(len(market_ids)),
                "markets_with_backtests": 0,
                "trades_scanned": 0,
                "note": "clustering_config missing",
            },
        }, [{"cohort": label, **stats} for label, stats in summaries.items()]

    from backtesting.causal_boost_replay import build_live_parity_boost_schedule

    wallet_rows_by_key = _wallet_eval_by_market_wallet(with_clustering_wallet_evaluations)
    boosted_to_alert_rows: List[Dict[str, Any]] = []
    boosted_anyway_alert_rows: List[Dict[str, Any]] = []
    boosted_any_trade_rows: List[Dict[str, Any]] = []
    not_boosted_alert_rows: List[Dict[str, Any]] = []

    markets_with_backtests = 0
    trades_scanned = 0

    for market_id in market_ids:
        with_result = with_backtest_results.get(int(market_id))
        without_result = without_backtest_results.get(int(market_id))
        if with_result is None or without_result is None:
            continue
        markets_with_backtests += 1

        try:
            all_trades = loader.get_trades_for_market(
                market_id=int(market_id),
                min_usd_amount=None,
                use_cache=False,
            )
        except TypeError:
            all_trades = loader.get_trades_for_market(int(market_id))

        detector_trades = (
            filter_trades_by_notional(all_trades, min_usd_amount)
            if min_usd_amount is not None
            else all_trades
        )
        if not detector_trades:
            continue

        schedule = build_live_parity_boost_schedule(
            detector_trades=detector_trades,
            market_id=str(market_id),
            clustering_config=clustering_config,
            clustering_min_trade_size=float(clustering_min_trade_size),
            jump_anticipation_config=None,
            attribution_provider=None,
            fetch_if_missing=False,
        )
        flagged_with = _build_flagged_info_by_key(with_result)
        flagged_without = _build_flagged_info_by_key(without_result)
        metadata = loader.get_market_metadata(int(market_id)) or {}
        market_slug = str(metadata.get("market_slug", market_id))

        for idx, trade in enumerate(detector_trades):
            trades_scanned += 1
            cluster_multiplier = float(schedule.cluster_multiplier_by_trade_idx[idx])
            key = _trade_key(
                str(getattr(trade, "wallet", "")),
                int(getattr(trade, "timestamp_ms", 0) or 0),
            )
            flagged_in_with = key in flagged_with
            flagged_in_without = key in flagged_without
            is_cluster_boosted = cluster_multiplier > 1.0 + BOOST_EPS
            row = {
                "market_id": int(market_id),
                "market_slug": market_slug,
                "wallet": str(getattr(trade, "wallet", "")),
                "timestamp_ms": int(getattr(trade, "timestamp_ms", 0) or 0),
                "side": str(getattr(trade, "side", "")),
                "notional_usdc": float(getattr(trade, "notional_usdc", 0.0) or 0.0),
                "cluster_multiplier": cluster_multiplier,
                "flagged_with_clustering": bool(flagged_in_with),
                "flagged_without_clustering": bool(flagged_in_without),
                "score_with_clustering": float(
                    flagged_with.get(key, {}).get("score", 0.0) or 0.0
                ),
                "score_without_clustering": float(
                    flagged_without.get(key, {}).get("score", 0.0) or 0.0
                ),
            }
            if is_cluster_boosted:
                boosted_any_trade_rows.append(row)
                if flagged_in_with and not flagged_in_without:
                    boosted_to_alert_rows.append(row)
                if flagged_in_with and flagged_in_without:
                    boosted_anyway_alert_rows.append(row)
            elif flagged_in_with:
                not_boosted_alert_rows.append(row)

    boosted_to_alert_wallets, missing_to_alert = _unique_wallet_eval_rows_for_trade_rows(
        boosted_to_alert_rows,
        wallet_rows_by_key,
    )
    boosted_anyway_wallets, missing_anyway = _unique_wallet_eval_rows_for_trade_rows(
        boosted_anyway_alert_rows,
        wallet_rows_by_key,
    )
    boosted_any_wallets, missing_any = _unique_wallet_eval_rows_for_trade_rows(
        boosted_any_trade_rows,
        wallet_rows_by_key,
    )
    not_boosted_alert_wallets, missing_not_boosted_alert = (
        _unique_wallet_eval_rows_for_trade_rows(
            not_boosted_alert_rows,
            wallet_rows_by_key,
        )
    )
    cohorts = [
        ("cluster_boosted_to_alert", boosted_to_alert_rows, boosted_to_alert_wallets),
        (
            "cluster_boosted_would_alert_anyway",
            boosted_anyway_alert_rows,
            boosted_anyway_wallets,
        ),
        ("cluster_boosted_any_trade", boosted_any_trade_rows, boosted_any_wallets),
        (
            "cluster_not_boosted_alert",
            not_boosted_alert_rows,
            not_boosted_alert_wallets,
        ),
    ]
    summaries = _print_trade_alert_boost_wallet_table(
        "TRADE ALERT CLUSTER BOOST EFFECT (wallet-eval stats)",
        cohorts,
    )
    report = {
        "cohorts": summaries,
        "meta": {
            "markets_requested": int(len(market_ids)),
            "markets_with_backtests": int(markets_with_backtests),
            "trades_scanned": int(trades_scanned),
            "wallet_eval_missing_trade_rows": {
                "cluster_boosted_to_alert": int(missing_to_alert),
                "cluster_boosted_would_alert_anyway": int(missing_anyway),
                "cluster_boosted_any_trade": int(missing_any),
                "cluster_not_boosted_alert": int(missing_not_boosted_alert),
            },
            "jump_anticipation_in_diagnostic": False,
            "attribution_in_diagnostic": False,
        },
    }
    rows = [{"cohort": label, **stats} for label, stats in summaries.items()]
    return report, rows


def _print_cluster_return_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"\n{'=' * 90}")
    print("ANONYMOUS CLUSTER RETURNS (wallet ground truth, full market history)")
    print("=" * 90)
    header = (
        f"{'cluster':>7s} {'strength':>9s} {'wallets':>8s} {'gt_wallets':>10s} "
        f"{'total_gross':>14s} {'total_pnl':>14s} {'mean_pnl':>12s} "
        f"{'mean_ret':>10s} {'cluster_ret':>11s} {'peak_pnl':>13s} {'peak_ret':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row.get('cluster_index', 0)):>7,d} "
            f"{float(row.get('cluster_strength_score', 1.0)):>9.3f} "
            f"{int(row.get('wallets_in_cluster', 0)):>8,d} "
            f"{int(row.get('wallets_with_ground_truth', 0)):>10,d} "
            f"${float(row.get('total_gross_buy_notional', 0.0)):>13,.2f} "
            f"${float(row.get('total_net_pnl', 0.0)):>+13,.2f} "
            f"${float(row.get('mean_net_pnl', 0.0)):>+11,.2f} "
            f"{float(row.get('mean_return', 0.0)):>+10.2%} "
            f"{float(row.get('weighted_return', 0.0)):>+11.2%} "
            f"${float(row.get('peak_wallet_net_pnl', 0.0)):>+12,.2f} "
            f"{float(row.get('peak_wallet_return', 0.0)):>+10.2%}"
        )

    if not rows:
        print("(no clusters found)")
        return {"cluster_count": 0}

    total_pnl = float(sum(float(r.get("total_net_pnl", 0.0) or 0.0) for r in rows))
    total_gross = float(
        sum(float(r.get("total_gross_buy_notional", 0.0) or 0.0) for r in rows)
    )
    return {
        "cluster_count": len(rows),
        "clusters_with_ground_truth": sum(
            1 for r in rows if int(r.get("wallets_with_ground_truth", 0) or 0) > 0
        ),
        "total_net_pnl": total_pnl,
        "total_gross_buy_notional": total_gross,
        "weighted_return": (total_pnl / total_gross) if total_gross > 1e-9 else 0.0,
    }


def _build_cluster_return_report(
    *,
    loader: HistoricalDataLoader,
    market_ids: Sequence[int],
    wallet_evaluations: List[Dict[str, Any]],
    clustering_config: Optional[Dict[str, Any]],
    clustering_min_trade_size: float,
    min_usd_amount: Optional[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Recompute anonymous cluster membership for the replay window and join each
    cluster to wallet-level ground-truth PnL/returns from the evaluation result.

    The output intentionally omits market IDs, slugs, cluster IDs, and wallet
    addresses; ``cluster_index`` is just a run-local row number.
    """
    if not clustering_config:
        summary = _print_cluster_return_table([])
        return {"summary": summary, "rows": []}, []

    from backtesting.bucket_clustering_backtest_runner import BucketClusteringBacktestRunner

    wallet_rows_by_market_wallet: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in wallet_evaluations:
        wallet = str(row.get("wallet", "")).lower()
        market_slug = str(row.get("market_slug", ""))
        if wallet and market_slug:
            wallet_rows_by_market_wallet[(market_slug, wallet)] = row

    runner = BucketClusteringBacktestRunner(
        detector_config={},
        clustering_config=clustering_config,
        attribution_provider=None,
    )
    rows: List[Dict[str, Any]] = []
    cluster_index = 0
    markets_with_clusters = 0

    for market_id in market_ids:
        metadata = loader.get_market_metadata(int(market_id)) or {}
        market_slug = str(metadata.get("market_slug", market_id))
        try:
            all_trades = loader.get_trades_for_market(
                int(market_id),
                min_usd_amount=None,
                use_cache=False,
            )
        except TypeError:
            all_trades = loader.get_trades_for_market(int(market_id))

        detector_trades = (
            filter_trades_by_notional(all_trades, min_usd_amount)
            if min_usd_amount is not None
            else all_trades
        )
        graph_trades = filter_trades_by_notional(
            detector_trades,
            clustering_min_trade_size,
        )
        if not graph_trades:
            continue

        clustering_state = runner._build_clustering_state(
            graph_trades=graph_trades,
            market_id=str(market_id),
            fetch_if_missing=False,
        )
        if not clustering_state.cluster_metadata:
            continue
        markets_with_clusters += 1

        for cluster in sorted(
            clustering_state.cluster_metadata.values(),
            key=lambda c: (int(c.cluster_id), int(c.size)),
        ):
            cluster_index += 1
            wallet_rows = [
                wallet_rows_by_market_wallet.get((market_slug, str(wallet).lower()))
                for wallet in cluster.wallets
            ]
            wallet_rows = [r for r in wallet_rows if r is not None]
            stats = _cohort_summary(wallet_rows)
            peak_wallet_row = max(
                wallet_rows,
                key=lambda r: float(r.get("net_pnl", 0.0) or 0.0),
                default=None,
            )
            rows.append(
                {
                    "cluster_index": int(cluster_index),
                    "cluster_strength_score": float(runner._compute_cluster_boost(cluster)),
                    "wallets_in_cluster": int(cluster.size),
                    "wallets_with_ground_truth": int(stats["count"]),
                    "density": float(cluster.density),
                    "total_edge_weight": float(cluster.total_edge_weight),
                    "total_gross_buy_notional": float(stats["total_gross_buy"]),
                    "total_net_pnl": float(stats["total_net_pnl"]),
                    "mean_net_pnl": float(stats["mean_net_pnl"]),
                    "median_net_pnl": float(stats["median_net_pnl"]),
                    "mean_return": float(stats["mean_return"]),
                    "median_return": float(stats["median_return"]),
                    "weighted_return": float(stats["weighted_return"]),
                    "peak_wallet_net_pnl": (
                        float(peak_wallet_row.get("net_pnl", 0.0) or 0.0)
                        if peak_wallet_row is not None
                        else 0.0
                    ),
                    "peak_wallet_return": (
                        float(peak_wallet_row.get("return", 0.0) or 0.0)
                        if peak_wallet_row is not None
                        else 0.0
                    ),
                    "peak_wallet_gross_buy_notional": (
                        float(peak_wallet_row.get("gross_buy_notional", 0.0) or 0.0)
                        if peak_wallet_row is not None
                        else 0.0
                    ),
                    "insider_precision": float(stats["precision"]),
                    "insider_wallets": int(stats["insiders"]),
                }
            )

    summary = _print_cluster_return_table(rows)
    summary["markets_with_clusters"] = int(markets_with_clusters)
    report = {"summary": summary, "rows": rows}
    return report, rows


def _build_report(
    *,
    annotated: List[Dict[str, Any]],
    with_clustering_wallet_evaluations: List[Dict[str, Any]],
    without_clustering_wallet_evaluations: List[Dict[str, Any]],
    boost_edges: List[float],
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
    layer2_enabled: bool,
) -> ClusteringEffectivenessReport:
    flagged = [r for r in annotated if r["flagged_with_clustering"]]
    flagged_no_cluster = [r for r in annotated if r["flagged_without_clustering"]]
    total_alerts_with = int(sum(int(r.get("num_flags_with_clustering", 0) or 0) for r in annotated))
    total_alerts_without = int(
        sum(int(r.get("num_flags_without_clustering", 0) or 0) for r in annotated)
    )
    missing_counterfactual = int(sum(1 for r in annotated if r.get("counterfactual_missing", False)))
    missing_diagnostic = int(sum(1 for r in annotated if r.get("diagnostic_missing", False)))

    report: Dict[str, Any] = {
        "config": {
            "prediction_mode": prediction_mode,
            "suspicion_threshold": suspicion_threshold,
            "flag_rate_threshold": flag_rate_threshold,
            "boost_buckets": boost_edges,
            "layer2_enabled": layer2_enabled,
            "counterfactual": "clustering_disabled_causal_replay",
            "cluster_bucket_source": "clustering_only_replay_when_available",
        },
        "counts": {
            "total_wallets": len(annotated),
            "insiders": sum(1 for r in annotated if r.get("is_insider", False)),
            "flagged_with_clustering": len(flagged),
            "flagged_without_clustering": len(flagged_no_cluster),
            "decisive_flips": sum(1 for r in flagged if r["decisive"]),
            "suppressed_flips": sum(1 for r in flagged_no_cluster if r.get("suppressed", False)),
            "in_any_cluster": sum(1 for r in annotated if r["in_cluster"]),
            "common_ownership_wallets": sum(
                1 for r in annotated if r.get("has_common_ownership", False)
            ),
            "total_alerts_with_clustering": total_alerts_with,
            "total_alerts_without_clustering": total_alerts_without,
            "alerts_delta": int(total_alerts_with - total_alerts_without),
            "counterfactual_missing_wallets": missing_counterfactual,
            "diagnostic_missing_wallets": missing_diagnostic,
        },
    }

    # 1. Cluster membership split (restricted to flagged wallets)
    membership = [
        ("flagged & in_cluster",
         [r for r in flagged if r["in_cluster"]]),
        ("flagged & NOT in_cluster",
         [r for r in flagged if not r["in_cluster"]]),
    ]
    report["membership_split"] = _print_cohort_table(
        "CLUSTER MEMBERSHIP SPLIT (flagged-with-clustering wallets)",
        membership,
        total_rows=flagged,
    )

    # 2. Decisiveness split
    decisive = [
        ("clustering was decisive",
         [r for r in flagged if r["decisive"]]),
        ("would-have-been-flagged anyway",
         [r for r in flagged if not r["decisive"]]),
    ]
    report["decisiveness_split"] = _print_cohort_table(
        "DECISIVENESS SPLIT (counterfactual clustering off)",
        decisive,
        total_rows=flagged,
    )

    # 3. Boost magnitude buckets
    bucket_rows: List[Tuple[str, List[Dict]]] = []
    bucket_rows.append(("boost = 1.0 (no cluster)",
                        [r for r in flagged if r["boost_bucket_idx"] == -1]))
    for i in range(len(boost_edges) - 1):
        lbl = _bucket_label(boost_edges, i)
        bucket_rows.append(
            (lbl, [r for r in flagged if r["boost_bucket_idx"] == i])
        )
    report["boost_buckets"] = _print_cohort_table(
        "BOOST MAGNITUDE BUCKETS (flagged-with-clustering wallets)",
        bucket_rows,
        total_rows=flagged,
    )

    # 4. Common ownership split
    ownership = [
        ("has_common_ownership = True",
         [r for r in flagged if r.get("has_common_ownership", False)]),
        ("has_common_ownership = False, in_cluster",
         [r for r in flagged
          if not r.get("has_common_ownership", False) and r["in_cluster"]]),
        ("no cluster",
         [r for r in flagged if not r["in_cluster"]]),
    ]
    tag = (
        "LAYER-2 COMMON-OWNERSHIP SPLIT (flagged-with-clustering wallets)"
        if layer2_enabled
        else
        "LAYER-2 COMMON-OWNERSHIP SPLIT (flagged-with-clustering wallets) "
        "-- Layer 2 attribution disabled, all rows will show False"
    )
    report["common_ownership_split"] = _print_cohort_table(
        tag, ownership, total_rows=flagged,
    )

    # 5. Cross-tab: boost bucket x has_common_ownership
    crosstab_title = (
        "CROSS-TAB: boost bucket x has_common_ownership (flagged-with-clustering wallets)"
    )
    print(f"\n{'=' * 90}")
    print(crosstab_title)
    print("=" * 90)
    crosstab_header = (
        f"{'bucket':<28s} {'own=False n':>12s} {'own=False med_pnl':>18s} "
        f"{'own=False prec':>15s} {'own=True n':>11s} {'own=True med_pnl':>17s} "
        f"{'own=True prec':>14s}"
    )
    print(crosstab_header)
    print("-" * len(crosstab_header))
    crosstab_out: Dict[str, Dict[str, Any]] = {}
    crosstab_labels = [
        ("boost = 1.0 (no cluster)", -1),
        *((_bucket_label(boost_edges, i), i) for i in range(len(boost_edges) - 1)),
    ]
    for lbl, idx in crosstab_labels:
        sub = [r for r in flagged if r["boost_bucket_idx"] == idx]
        own_true = [r for r in sub if r.get("has_common_ownership", False)]
        own_false = [r for r in sub if not r.get("has_common_ownership", False)]
        s_true = _cohort_summary(own_true)
        s_false = _cohort_summary(own_false)
        crosstab_out[lbl] = {"own_true": s_true, "own_false": s_false}
        print(
            f"{lbl:<28s} "
            f"{s_false['count']:>12,d} "
            f"${s_false['median_net_pnl']:>+17,.2f} "
            f"{s_false['precision']:>14.2%} "
            f"{s_true['count']:>11,d} "
            f"${s_true['median_net_pnl']:>+16,.2f} "
            f"{s_true['precision']:>13.2%}"
        )
    report["boost_bucket_x_ownership"] = crosstab_out

    # 6. Baseline: all flagged-with-clustering vs flagged-without-clustering
    baseline = [
        ("flagged WITH clustering", flagged),
        ("flagged WITHOUT clustering (counterfactual replay)", flagged_no_cluster),
    ]
    report["baseline_comparison"] = _print_cohort_table(
        "BASELINE COMPARISON: with-clustering vs no-clustering replay",
        baseline,
    )

    # 7. Classification metrics
    metrics_with = _calculate_metrics_from_wallet_evaluations(
        with_clustering_wallet_evaluations,
        prediction_mode,
        suspicion_threshold,
        flag_rate_threshold,
    )
    metrics_without = _calculate_metrics_from_wallet_evaluations(
        without_clustering_wallet_evaluations,
        prediction_mode,
        suspicion_threshold,
        flag_rate_threshold,
    )
    print(f"\n{'=' * 90}")
    print("CLASSIFICATION METRICS (full wallet universe)")
    print("=" * 90)

    def _fmt_m(label: str, m: Dict[str, Any]) -> None:
        print(
            f"  {label:<28s} "
            f"precision={float(m.get('precision', 0.0)):.4f}  "
            f"recall={float(m.get('recall', 0.0)):.4f}  "
            f"F1={float(m.get('f1', 0.0)):.4f}  "
            f"F0.5={float(m.get('f0_5', 0.0)):.4f}  "
            f"F2={float(m.get('f2', 0.0)):.4f}  "
            f"TP={int(m.get('true_positives', 0)):,}  "
            f"FP={int(m.get('false_positives', 0)):,}  "
            f"FN={int(m.get('false_negatives', 0)):,}"
        )

    _fmt_m("with clustering", metrics_with)
    _fmt_m("without clustering (counterfactual)", metrics_without)
    report["classification_metrics"] = {
        "with_clustering": metrics_with,
        "without_clustering": metrics_without,
    }

    # Per-wallet CSV rows (keep selected columns)
    per_wallet_rows = [
        {
            "wallet": r.get("wallet", ""),
            "market_slug": r.get("market_slug", ""),
            "trade_count_with_clustering": r.get("trade_count_with_clustering", 0),
            "trade_count_without_clustering": r.get("trade_count_without_clustering", 0),
            "num_flags_with_clustering": r.get("num_flags_with_clustering", 0),
            "num_flags_without_clustering": r.get("num_flags_without_clustering", 0),
            "flag_rate_with_clustering": r.get("flag_rate_with_clustering", 0.0),
            "flag_rate_without_clustering": r.get("flag_rate_without_clustering", 0.0),
            "cluster_boost": r.get("cluster_boost", 1.0),
            "in_cluster": r.get("in_cluster", False),
            "has_common_ownership": r.get("has_common_ownership", False),
            "flagged_with_clustering": r.get("flagged_with_clustering", False),
            "flagged_without_clustering": r.get("flagged_without_clustering", False),
            "decisive": r.get("decisive", False),
            "suppressed": r.get("suppressed", False),
            "alerts_delta": r.get("alerts_delta", 0),
            "counterfactual_missing": r.get("counterfactual_missing", False),
            "diagnostic_missing": r.get("diagnostic_missing", False),
            "combined_boost": r.get("combined_boost", 1.0),
            "boost_bucket": r.get("boost_bucket_label", ""),
            "net_pnl": r.get("net_pnl", 0.0),
            "return": r.get("return", 0.0),
            "gross_buy_notional": r.get("gross_buy_notional", 0.0),
            "is_insider": r.get("is_insider", False),
        }
        for r in annotated
    ]

    # Boost bucket CSV rows (long form)
    boost_bucket_rows: List[Dict[str, Any]] = []
    for lbl, stats in report["boost_buckets"].items():
        boost_bucket_rows.append({"bucket": lbl, **stats})

    return ClusteringEffectivenessReport(
        report=report,
        per_wallet_rows=per_wallet_rows,
        boost_bucket_rows=boost_bucket_rows,
        trade_cohort_rows=[],
        trade_alert_boost_wallet_rows=[],
        cluster_return_rows=[],
    )

def _resolve_config(
    loader: HistoricalDataLoader,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (config, source_metadata).  Runs the optimizer when needed."""
    if args.config_path:
        cfg = _load_config(args.config_path)
        return cfg, {
            "source": "provided",
            "config_path": str(Path(args.config_path).resolve()),
        }

    opt_start = args.opt_start_date or args.start_date
    opt_end = args.opt_end_date or args.end_date
    logging.info(
        "No --config-path provided; running multi-start optimizer on "
        "timeframe %s .. %s (n_starts=%d).",
        opt_start, opt_end, int(args.n_starts or 1),
    )

    opt_args = copy.copy(args)
    opt_args.start_date = opt_start
    opt_args.end_date = opt_end
    opt_out_dir = output_dir / "optimization"
    opt_out_dir.mkdir(parents=True, exist_ok=True)
    opt_args.output_dir = str(opt_out_dir)

    opt_prep = prepare_timeframe_inference(
        loader,
        output_dir=str(opt_out_dir),
        start_date=opt_start,
        end_date=opt_end,
        min_market_volume=opt_args.min_market_volume,
        classifications_path=opt_args.classifications_path,
        insider_plausible_only=opt_args.insider_plausible_only,
        non_insider_plausible_only=opt_args.non_insider_plausible_only,
        market_categories=opt_args.market_categories,
        exclude_categories=opt_args.exclude_categories,
        resolution_threshold=opt_args.resolution_threshold,
        min_trades=opt_args.min_trades,
        inferred_resolutions_db=opt_args.inferred_resolutions_db,
        enable_trade_prefilter=opt_args.enable_trade_prefilter,
        min_usd_amount=opt_args.min_usd_amount,
    )

    if not opt_prep.market_ids:
        raise SystemExit(
            "No inferred-resolved markets in optimization timeframe; cannot optimize."
        )

    out = run_multi_start_alternating_timeframe(loader, opt_prep, opt_args)
    best_config_path = str(out["best_config_path"])
    cfg = _load_config(best_config_path)
    source_meta = {
        "source": "optimized",
        "opt_start_date": opt_start,
        "opt_end_date": opt_end,
        "best_config_path": best_config_path,
        "multi_start_summary": out.get("multi_start_summary", {}),
        "optimizer_artifacts": {
            k: str(v) for k, v in out.items() if isinstance(v, (str, Path))
        },
    }
    return cfg, source_meta

def _save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _save_artifacts(
    output_dir: Path,
    ts: str,
    report_obj: ClusteringEffectivenessReport,
    run_metadata: Dict[str, Any],
) -> Dict[str, str]:
    saved: Dict[str, str] = {}

    summary_path = output_dir / f"clustering_effectiveness_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_metadata": run_metadata, "report": report_obj.report},
            f, indent=2, default=str,
        )
    saved["summary"] = str(summary_path)

    wallets_path = output_dir / f"clustering_effectiveness_wallets_{ts}.csv"
    wallet_fields = [
        "wallet", "market_slug",
        "trade_count_with_clustering", "trade_count_without_clustering",
        "num_flags_with_clustering", "num_flags_without_clustering",
        "flag_rate_with_clustering", "flag_rate_without_clustering",
        "cluster_boost", "in_cluster", "has_common_ownership",
        "flagged_with_clustering", "flagged_without_clustering",
        "decisive", "suppressed", "alerts_delta", "counterfactual_missing", "diagnostic_missing",
        "combined_boost",
        "boost_bucket",
        "net_pnl", "return", "gross_buy_notional", "is_insider",
    ]
    _save_csv(wallets_path, report_obj.per_wallet_rows, wallet_fields)
    saved["wallets_csv"] = str(wallets_path)

    buckets_path = output_dir / f"clustering_effectiveness_boost_buckets_{ts}.csv"
    bucket_fields = [
        "bucket", "count", "total_net_pnl", "mean_net_pnl", "median_net_pnl",
        "mean_return", "median_return", "weighted_return", "precision", "insiders",
        "mean_boost", "total_gross_buy", "mean_gross_buy", "share_common_ownership",
    ]
    _save_csv(buckets_path, report_obj.boost_bucket_rows, bucket_fields)
    saved["boost_buckets_csv"] = str(buckets_path)

    trade_cohorts_path = output_dir / f"clustering_effectiveness_trade_buy_alert_cohorts_{ts}.csv"
    trade_cohort_fields = [
        "cohort",
        "count",
        "total_notional",
        "total_pnl",
        "mean_pnl",
        "median_pnl",
        "mean_return",
        "median_return",
        "weighted_return",
        "win_rate",
    ]
    _save_csv(trade_cohorts_path, report_obj.trade_cohort_rows, trade_cohort_fields)
    saved["trade_buy_alert_cohorts_csv"] = str(trade_cohorts_path)

    alert_boost_path = output_dir / f"clustering_effectiveness_trade_alert_boost_wallet_eval_{ts}.csv"
    alert_boost_fields = [
        "cohort",
        "trade_count",
        "wallet_count",
        "count",
        "total_net_pnl",
        "mean_net_pnl",
        "median_net_pnl",
        "mean_return",
        "median_return",
        "weighted_return",
        "precision",
        "insiders",
        "mean_boost",
        "mean_trade_cluster_multiplier",
        "total_gross_buy",
        "mean_gross_buy",
        "share_common_ownership",
    ]
    _save_csv(
        alert_boost_path,
        report_obj.trade_alert_boost_wallet_rows,
        alert_boost_fields,
    )
    saved["trade_alert_boost_wallet_eval_csv"] = str(alert_boost_path)

    cluster_returns_path = output_dir / f"clustering_effectiveness_cluster_returns_{ts}.csv"
    cluster_return_fields = [
        "cluster_index",
        "cluster_strength_score",
        "wallets_in_cluster",
        "wallets_with_ground_truth",
        "density",
        "total_edge_weight",
        "total_gross_buy_notional",
        "total_net_pnl",
        "mean_net_pnl",
        "median_net_pnl",
        "mean_return",
        "median_return",
        "weighted_return",
        "peak_wallet_net_pnl",
        "peak_wallet_return",
        "peak_wallet_gross_buy_notional",
        "insider_precision",
        "insider_wallets",
    ]
    _save_csv(cluster_returns_path, report_obj.cluster_return_rows, cluster_return_fields)
    saved["cluster_returns_csv"] = str(cluster_returns_path)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.insider_plausible_only and args.non_insider_plausible_only:
        raise SystemExit(
            "--insider-plausible-only and --non-insider-plausible-only are mutually exclusive"
        )
    if int(args.n_starts or 1) < 1:
        raise SystemExit("--n-starts must be >= 1")

    boost_edges = _parse_boost_buckets(args.boost_buckets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_timeframe_logging(str(output_dir), "clustering_effectiveness")
    set_experiment_backtest_log_quiet_mode(enabled=not args.verbose_output)

    logging.info("Loading historical data...")
    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    run_start = time.time()

    # Resolve the config to backtest (either provided or via optimization).
    try:
        config, config_source = _resolve_config(loader, args, output_dir)
    except SystemExit:
        loader.close()
        raise
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to resolve config: %s", exc)
        loader.close()
        sys.exit(1)

    logging.info("Config source: %s", config_source.get("source"))

    # Build the args namespace that run_timeframe_backtest_evaluation expects.
    bt_args = copy.copy(args)
    bt_args.no_clustering = not getattr(args, "enable_clustering", True)
    bt_args.no_jump_anticipation = not getattr(args, "enable_jump_anticipation", False)
    # Backtest always uses the primary start/end date.
    bt_args.start_date = args.start_date
    bt_args.end_date = args.end_date
    bt_args.dry_run = False
    # run_timeframe_backtest_evaluation expects these attributes:
    bt_args.copytrade_fixed_size = args.copytrade_fixed_size
    bt_args.suspicion_threshold = args.suspicion_threshold
    bt_args.verbose_output = args.verbose_output

    logging.info(
        "Running evaluation backtest (with clustering) on %s .. %s ...",
        args.start_date,
        args.end_date,
    )

    try:
        eval_result, eval_meta = run_timeframe_backtest_evaluation(
            config, loader, bt_args, quiet=not args.verbose_output,
        )
    except RuntimeError as exc:
        logging.error("Backtest failed: %s", exc)
        loader.close()
        sys.exit(1)

    logging.info(
        "With-clustering backtest complete: wallets=%d, markets=%d",
        len(eval_result.wallet_evaluations),
        len(eval_result.market_ids),
    )

    bt_args_no_cluster = copy.copy(bt_args)
    bt_args_no_cluster.no_clustering = True
    logging.info(
        "Running counterfactual backtest (clustering disabled) on %s .. %s ...",
        args.start_date,
        args.end_date,
    )

    try:
        no_cluster_result, no_cluster_meta = run_timeframe_backtest_evaluation(
            config, loader, bt_args_no_cluster, quiet=not args.verbose_output,
        )
    except RuntimeError as exc:
        logging.error("Counterfactual no-clustering backtest failed: %s", exc)
        loader.close()
        sys.exit(1)

    logging.info(
        "No-clustering backtest complete: wallets=%d, markets=%d",
        len(no_cluster_result.wallet_evaluations),
        len(no_cluster_result.market_ids),
    )

    clustering_diag_result = eval_result
    clustering_diag_meta = eval_meta
    used_cluster_only_diagnostics = False
    if not bt_args.no_jump_anticipation:
        bt_args_cluster_diag = copy.copy(bt_args)
        bt_args_cluster_diag.no_jump_anticipation = True
        logging.info(
            "Running clustering-only diagnostic replay (JA disabled) for clean cluster buckets..."
        )
        try:
            clustering_diag_result, clustering_diag_meta = run_timeframe_backtest_evaluation(
                config, loader, bt_args_cluster_diag, quiet=not args.verbose_output,
            )
            used_cluster_only_diagnostics = True
        except RuntimeError as exc:
            logging.warning(
                "Clustering-only diagnostic replay failed (%s); "
                "falling back to combined cluster+JA diagnostics.",
                exc,
            )
            clustering_diag_result = eval_result
            clustering_diag_meta = eval_meta

    # Annotate wallet rows + build report.
    annotated = _annotate_wallet_rows(
        eval_result.wallet_evaluations,
        no_cluster_result.wallet_evaluations,
        clustering_diag_result.wallet_evaluations,
        flag_rate_threshold=args.flag_rate_threshold,
        prediction_mode=args.prediction_mode,
        suspicion_threshold=args.suspicion_threshold,
        boost_edges=boost_edges,
    )
    report_obj = _build_report(
        annotated=annotated,
        with_clustering_wallet_evaluations=eval_result.wallet_evaluations,
        without_clustering_wallet_evaluations=no_cluster_result.wallet_evaluations,
        boost_edges=boost_edges,
        prediction_mode=args.prediction_mode,
        suspicion_threshold=args.suspicion_threshold,
        flag_rate_threshold=args.flag_rate_threshold,
        layer2_enabled=bool(args.enable_layer2_attribution),
    )
    trade_buy_alert_report, trade_cohort_rows = _build_trade_buy_alert_counterfactual_report(
        loader=loader,
        market_ids=eval_result.market_ids,
        with_backtest_results=eval_result.backtest_results,
        without_backtest_results=no_cluster_result.backtest_results,
        min_usd_amount=args.min_usd_amount,
        resolution_threshold=float(args.resolution_threshold),
        clustering_config=(
            config.get("clustering_config")
            if not getattr(bt_args, "no_clustering", False)
            else None
        ),
        clustering_min_trade_size=float(args.clustering_min_trade_size),
    )
    report_obj.report["trade_buy_alert_counterfactual"] = trade_buy_alert_report
    report_obj.trade_cohort_rows = trade_cohort_rows
    trade_alert_boost_report, trade_alert_boost_rows = (
        _build_trade_alert_boost_wallet_eval_report(
            loader=loader,
            market_ids=eval_result.market_ids,
            with_backtest_results=eval_result.backtest_results,
            without_backtest_results=no_cluster_result.backtest_results,
            with_clustering_wallet_evaluations=eval_result.wallet_evaluations,
            min_usd_amount=args.min_usd_amount,
            clustering_config=(
                config.get("clustering_config")
                if not getattr(bt_args, "no_clustering", False)
                else None
            ),
            clustering_min_trade_size=float(args.clustering_min_trade_size),
        )
    )
    report_obj.report["trade_alert_boost_wallet_eval"] = trade_alert_boost_report
    report_obj.trade_alert_boost_wallet_rows = trade_alert_boost_rows

    total_elapsed = time.time() - run_start
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_metadata = {
        "timestamp": ts,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "opt_start_date": args.opt_start_date or args.start_date,
        "opt_end_date": args.opt_end_date or args.end_date,
        "config_source": config_source,
        "prediction_mode": args.prediction_mode,
        "flag_rate_threshold": args.flag_rate_threshold,
        "suspicion_threshold": args.suspicion_threshold,
        "enable_layer2_attribution": bool(args.enable_layer2_attribution),
        "enable_clustering": bool(getattr(args, "enable_clustering", True)),
        "enable_jump_anticipation": bool(getattr(args, "enable_jump_anticipation", False)),
        "clustering_min_trade_size": args.clustering_min_trade_size,
        "min_usd_amount": args.min_usd_amount,
        "min_wallet_notional": args.min_wallet_notional,
        "include_recidivism": bool(args.include_recidivism),
        "boost_buckets": boost_edges,
        "market_counts": {
            "with_clustering_candidate": eval_meta.get("candidate_markets", 0),
            "with_clustering_resolved": eval_meta.get("resolved_markets", 0),
            "without_clustering_candidate": no_cluster_meta.get("candidate_markets", 0),
            "without_clustering_resolved": no_cluster_meta.get("resolved_markets", 0),
            "clustering_diagnostic_candidate": clustering_diag_meta.get("candidate_markets", 0),
            "clustering_diagnostic_resolved": clustering_diag_meta.get("resolved_markets", 0),
        },
        "resolution_stats": {
            "with_clustering": eval_meta.get("resolution_stats", {}),
            "without_clustering": no_cluster_meta.get("resolution_stats", {}),
            "clustering_diagnostic": clustering_diag_meta.get("resolution_stats", {}),
        },
        "counterfactual": {
            "mode": "clustering_disabled_causal_replay",
            "missing_wallet_rows": int(
                report_obj.report.get("counts", {}).get("counterfactual_missing_wallets", 0)
            ),
            "missing_diagnostic_rows": int(
                report_obj.report.get("counts", {}).get("diagnostic_missing_wallets", 0)
            ),
            "clustering_diagnostic_ja_disabled": used_cluster_only_diagnostics,
        },
        "trade_buy_alert_counterfactual_meta": (
            report_obj.report.get("trade_buy_alert_counterfactual", {}).get("meta", {})
        ),
        "trade_alert_boost_wallet_eval_meta": (
            report_obj.report.get("trade_alert_boost_wallet_eval", {}).get("meta", {})
        ),
        "total_elapsed_seconds": total_elapsed,
        "config": config,
    }

    saved = _save_artifacts(output_dir, ts, report_obj, run_metadata)

    print("\n" + "=" * 90)
    print("CLUSTERING EFFECTIVENESS RUN COMPLETE")
    print("=" * 90)
    print(f"Config source:          {config_source.get('source')}")
    if config_source.get("best_config_path"):
        print(f"Optimizer best config:  {config_source['best_config_path']}")
    print(f"Backtest timeframe:     {args.start_date} .. {args.end_date}")
    print(
        "Markets evaluated:      "
        f"with={eval_meta.get('resolved_markets', 0):,} "
        f"without={no_cluster_meta.get('resolved_markets', 0):,}"
    )
    print(
        "Wallets evaluated:      "
        f"with={len(eval_result.wallet_evaluations):,} "
        f"without={len(no_cluster_result.wallet_evaluations):,}"
    )
    if used_cluster_only_diagnostics:
        print(
            "Cluster diagnostics:    "
            f"JA-disabled replay used ({len(clustering_diag_result.wallet_evaluations):,} wallets)"
        )
    elif not bt_args.no_jump_anticipation:
        print("Cluster diagnostics:    fallback to combined cluster+JA replay")
    else:
        print("Cluster diagnostics:    sourced from with-clustering replay (JA already off)")
    counts = report_obj.report.get("counts", {})
    print(
        "Wallet flips:           "
        f"decisive={int(counts.get('decisive_flips', 0)):,} "
        f"suppressed={int(counts.get('suppressed_flips', 0)):,}"
    )
    print(
        "Alert delta:            "
        f"with={int(counts.get('total_alerts_with_clustering', 0)):,} "
        f"without={int(counts.get('total_alerts_without_clustering', 0)):,} "
        f"delta={int(counts.get('alerts_delta', 0)):+,}"
    )
    print(
        "Join quality:           "
        f"counterfactual_missing={int(counts.get('counterfactual_missing_wallets', 0)):,} "
        f"diagnostic_missing={int(counts.get('diagnostic_missing_wallets', 0)):,}"
    )
    trade_cf = report_obj.report.get("trade_buy_alert_counterfactual", {})
    trade_cohorts = trade_cf.get("cohorts", {})
    boosted_stats = trade_cohorts.get("boosted_to_buy_alert", {})
    boosted_anyway_stats = trade_cohorts.get(
        "boosted_but_would_buy_alert_anyway", {}
    )
    cluster_boosted_any_stats = trade_cohorts.get("cluster_boosted_any_trade", {})
    not_flagged_stats = trade_cohorts.get("not_flagged_with_clustering", {})
    print(
        "Trade cohorts (BUY):    "
        f"boosted_to_alert={int(boosted_stats.get('count', 0)):,} "
        f"boosted_anyway={int(boosted_anyway_stats.get('count', 0)):,} "
        f"cluster_boosted_any={int(cluster_boosted_any_stats.get('count', 0)):,} "
        f"not_flagged={int(not_flagged_stats.get('count', 0)):,}"
    )
    print(
        "Trade mean return:      "
        f"boosted={float(boosted_stats.get('mean_return', 0.0)):+.2%} "
        f"boosted_anyway={float(boosted_anyway_stats.get('mean_return', 0.0)):+.2%} "
        f"cluster_any={float(cluster_boosted_any_stats.get('mean_return', 0.0)):+.2%} "
        f"not_flagged={float(not_flagged_stats.get('mean_return', 0.0)):+.2%}"
    )
    print(
        "Trade mean pnl:         "
        f"boosted=${float(boosted_stats.get('mean_pnl', 0.0)):+,.2f} "
        f"boosted_anyway=${float(boosted_anyway_stats.get('mean_pnl', 0.0)):+,.2f} "
        f"not_flagged=${float(not_flagged_stats.get('mean_pnl', 0.0)):+,.2f}"
    )
    alert_boost = report_obj.report.get("trade_alert_boost_wallet_eval", {})
    alert_boost_cohorts = alert_boost.get("cohorts", {})
    alert_to = alert_boost_cohorts.get("cluster_boosted_to_alert", {})
    alert_anyway = alert_boost_cohorts.get("cluster_boosted_would_alert_anyway", {})
    alert_any = alert_boost_cohorts.get("cluster_boosted_any_trade", {})
    alert_not_boosted = alert_boost_cohorts.get("cluster_not_boosted_alert", {})
    print(
        "Cluster-boosted flags:  "
        f"to_alert={int(alert_to.get('trade_count', 0)):,} "
        f"anyway={int(alert_anyway.get('trade_count', 0)):,} "
        f"any_trade={int(alert_any.get('trade_count', 0)):,} "
        f"not_boosted_alert={int(alert_not_boosted.get('trade_count', 0)):,}"
    )
    print(
        "Boosted wallet return:  "
        f"to_alert={float(alert_to.get('mean_return', 0.0)):+.2%} "
        f"anyway={float(alert_anyway.get('mean_return', 0.0)):+.2%} "
        f"any_trade={float(alert_any.get('mean_return', 0.0)):+.2%} "
        f"not_boosted_alert={float(alert_not_boosted.get('mean_return', 0.0)):+.2%}"
    )
    print(f"Layer 2 attribution:    {'enabled' if args.enable_layer2_attribution else 'disabled'}")
    print(f"Total wall time:        {total_elapsed:.1f}s")
    print("\nFiles:")
    print(f"  - {saved['summary']}")
    print(f"  - {saved['wallets_csv']}")
    print(f"  - {saved['boost_buckets_csv']}")
    print(f"  - {saved['trade_buy_alert_cohorts_csv']}")
    print(f"  - {saved['trade_alert_boost_wallet_eval_csv']}")
    print(f"  - {log_path}")

    loader.close()


if __name__ == "__main__":
    main()
