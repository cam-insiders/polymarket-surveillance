"""
Experiment: Compare system against SOTA baselines with trade-window replay.

Usage:
    python -m experiments.compare_sota_timeframe \\
        --train-start 2025-02-01 --train-end 2025-02-10 \\
        --test-start 2025-02-11 --test-end 2025-02-20

    python -m experiments.compare_sota_timeframe \\
        --start-date 2025-02-01 --end-date 2025-02-20 --train-split 0.3
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import DEFAULT_CLUSTERING_CONFIG, EvaluationResult, evaluate_config
from backtesting.market_resolutions import get_resolved_market_ids, get_winning_outcome
from backtesting.parameter_optimizer import _calculate_metrics_from_wallet_evaluations
from experiments.timeframe_market_common import infer_resolutions, select_market_ids_in_timeframe
from experiments.timeframe_experiment_common import (
    filter_markets_by_window_trade_count,
    materialize_timeframe_prep,
    prepare_timeframe_inference,
    scoped_trade_time_filter,
)
from experiments.timeframe_optimizers import run_timeframe_optimizer
from experiments.sota_algorithms.common import wallet_flagged_pnl_from_evaluations
from experiments.sota_algorithms.consob_pca_faithful import (
    parse_n_components,
    run_consob_pca_faithful_baseline,
)
from experiments.sota_algorithms.isolation_forest import (
    IF_FEATURE_NAMES,
    extract_isolation_forest_features,
    run_isolation_forest_baseline,
)
from experiments.sota_algorithms.mitts_ofir_faithful import (
    run_mitts_ofir_faithful_causal,
    run_mitts_ofir_faithful_retrospective,
)
from experiments.sota_algorithms.random_baseline import run_random_baseline
from experiments.sota_algorithms.timing_heuristic import run_timing_heuristic_baseline
from models import Trade


def _load_full_market_ground_truth_entries(
    loader: HistoricalDataLoader,
    market_ids: List[int],
) -> List[Tuple[Trade, int]]:
    """
    Build the wallet-label tape from complete market history.

    Timeframe replay scopes detector inputs to the active train/test window, but
    wallet ground truth must match ``evaluate_config``: all trades in the
    evaluated market, unfiltered by replay window or detector min-USD prefilter.
    """
    entries: List[Tuple[Trade, int]] = []
    for mid in market_ids:
        try:
            trades = loader.get_trades_for_market(
                market_id=mid,
                min_usd_amount=None,
                use_cache=False,
                ignore_trade_time_bounds=True,
            )
        except TypeError:
            trades = loader.get_trades_for_market(mid)
        for trade in trades:
            entries.append((trade, int(mid)))
    entries.sort(key=lambda x: x[0].timestamp_ms)
    return entries


def _effective_min_usd_amount(args: argparse.Namespace) -> Optional[float]:
    """Match ``timeframe_trade_window_train_backtest``: filter only when prefilter is on."""
    return args.min_usd_amount if args.enable_trade_prefilter else None


def _prepare_window_markets(
    loader: HistoricalDataLoader,
    args: argparse.Namespace,
    *,
    start_date: str,
    end_date: str,
    override_filename_prefix: str,
) -> Tuple[List[int], Dict[int, int], List[int], Dict[str, Any]]:
    """Select markets by close time and infer resolutions on full trade history."""
    prep = prepare_timeframe_inference(
        loader,
        output_dir=args.output_dir,
        start_date=start_date,
        end_date=end_date,
        min_market_volume=args.min_market_volume,
        classifications_path=args.classifications_path,
        insider_plausible_only=args.insider_plausible_only,
        non_insider_plausible_only=args.non_insider_plausible_only,
        market_categories=args.market_categories,
        exclude_categories=args.exclude_categories,
        resolution_threshold=args.resolution_threshold,
        min_trades=args.min_trades,
        inferred_resolutions_db=args.inferred_resolutions_db,
        enable_trade_prefilter=args.enable_trade_prefilter,
        min_usd_amount=args.min_usd_amount,
        override_filename_prefix=override_filename_prefix,
    )
    return (
        list(prep.market_ids),
        dict(prep.inferred_winners),
        list(prep.candidate_market_ids),
        dict(prep.res_stats),
    )


def _full_system_metrics_row(
    result: EvaluationResult,
    elapsed: float,
    baseline_name: str,
    min_usd_amount: Optional[float],
) -> Dict:
    pooled = result.event_study_pooled.get("pooled", {})
    wallet_metrics = _calculate_metrics_from_wallet_evaluations(
        result.wallet_evaluations,
        result.prediction_mode,
        result.suspicion_threshold,
        result.flag_rate_threshold,
    )
    cs = result.copytrade_summary
    flagged_wallet_pnl = wallet_flagged_pnl_from_evaluations(
        result.wallet_evaluations,
        result.prediction_mode,
        result.suspicion_threshold,
        result.flag_rate_threshold,
    )
    copytrade_metrics = {
        "copytrade_total_flagged_buys": 0,
        "copytrade_total_capital_deployed": 0.0,
        "copytrade_total_pnl": 0.0,
        "copytrade_portfolio_roi": 0.0,
        "copytrade_win_rate": 0.0,
        "copytrade_mean_trade_return": 0.0,
        "copytrade_median_trade_return": 0.0,
    }
    if result.copytrade_result is not None:
        copytrade_metrics = {
            "copytrade_total_flagged_buys": int(result.copytrade_result.total_flagged_buys),
            "copytrade_total_capital_deployed": float(result.copytrade_result.total_capital_deployed),
            "copytrade_total_pnl": float(result.copytrade_result.total_pnl),
            "copytrade_portfolio_roi": float(result.copytrade_result.portfolio_roi),
            "copytrade_win_rate": float(result.copytrade_result.win_rate),
            "copytrade_mean_trade_return": float(result.copytrade_result.mean_trade_return),
            "copytrade_median_trade_return": float(result.copytrade_result.median_trade_return),
        }
    num_flags = int(sum(
        int(getattr(backtest_result, "alerts_generated", 0) or 0)
        for backtest_result in result.backtest_results.values()
    ))
    return {
        "baseline": baseline_name,
        "num_flags": num_flags,
        "flagged_trades": pooled.get("total_flagged_trades", 0),
        "unflagged_trades": pooled.get("total_unflagged_trades", 0),
        "flagged_mean_return": pooled.get("pooled_flagged_mean_return", 0),
        "unflagged_mean_return": pooled.get("pooled_unflagged_mean_return", 0),
        "mean_return_diff": pooled.get("pooled_mean_return_diff", 0),
        "mean_cohens_d": pooled.get("mean_cohens_d", 0),
        "sig_welch_p05": pooled.get("markets_significant_welch_p05", 0),
        "n_markets": pooled.get("n_markets", 0),
        "num_wallets_evaluated": int(wallet_metrics.get("num_wallets", 0)),
        "num_flagged_wallets": int(wallet_metrics.get("num_predicted_positive", 0)),
        "tp": int(wallet_metrics.get("true_positives", 0)),
        "fp": int(wallet_metrics.get("false_positives", 0)),
        "fn": int(wallet_metrics.get("false_negatives", 0)),
        "tn": int(wallet_metrics.get("true_negatives", 0)),
        "wallet_precision": float(wallet_metrics.get("precision", 0.0)),
        "wallet_recall": float(wallet_metrics.get("recall", 0.0)),
        "wallet_f1": float(wallet_metrics.get("f1", 0.0)),
        "wallet_f0_5": float(wallet_metrics.get("f0_5", 0.0)),
        "flagged_avg_return": cs.get("flagged", {}).get("avg_return", 0),
        "tp_avg_return": cs.get("tp", {}).get("avg_return", 0),
        "fp_avg_return": cs.get("fp", {}).get("avg_return", 0),
        "trades_per_second": result.aggregate_performance.overall_trades_per_second,
        "det_p95_us": result.aggregate_performance.detection_latency_p95_us,
        "wall_clock_s": elapsed,
        "min_usd_amount": min_usd_amount,
        "flagged_wallet_mean_net_pnl": flagged_wallet_pnl["flagged_wallet_mean_net_pnl"],
        "flagged_wallet_median_net_pnl": flagged_wallet_pnl["flagged_wallet_median_net_pnl"],
        "deployable_live": True,
        **copytrade_metrics,
    }


def _add_wallet_classification_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wallet-level precision/recall/F-scores from tp/fp/fn counts.

    This keeps a consistent metric definition across full-system and all SOTA rows.
    """
    if df.empty:
        return df
    required = {"tp", "fp", "fn"}
    if not required.issubset(df.columns):
        return df

    tp = pd.to_numeric(df["tp"], errors="coerce").fillna(0.0)
    fp = pd.to_numeric(df["fp"], errors="coerce").fillna(0.0)
    fn = pd.to_numeric(df["fn"], errors="coerce").fillna(0.0)

    precision_den = tp + fp
    recall_den = tp + fn
    precision = np.where(precision_den > 0.0, tp / precision_den, 0.0)
    recall = np.where(recall_den > 0.0, tp / recall_den, 0.0)
    f1_den = precision + recall
    f1 = np.where(f1_den > 0.0, (2.0 * precision * recall) / f1_den, 0.0)
    f05_den = (0.25 * precision) + recall
    f05 = np.where(f05_den > 0.0, (1.25 * precision * recall) / f05_den, 0.0)

    df = df.copy()
    predicted_positive = (tp + fp).astype(float)
    df["wallet_precision"] = precision.astype(float)
    df["wallet_recall"] = recall.astype(float)
    df["wallet_f1"] = f1.astype(float)
    df["wallet_f0_5"] = f05.astype(float)
    if "num_flagged_wallets" not in df.columns:
        df["num_flagged_wallets"] = predicted_positive.astype(int)
    else:
        existing_flagged = pd.to_numeric(df["num_flagged_wallets"], errors="coerce")
        df["num_flagged_wallets"] = existing_flagged.fillna(predicted_positive).astype(int)
    return df


def _split_market_ids_by_close_time(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    eval_fraction: float,
) -> Tuple[List[int], List[int]]:
    if eval_fraction <= 0.0 or len(market_ids) < 2:
        return [], list(market_ids)
    if loader.markets_df is None:
        raise RuntimeError("Call load_data() first.")

    df = loader.markets_df.copy()
    df["id_int"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id_int"].notna()].copy()
    df["id_int"] = df["id_int"].astype(int)
    df["closed_dt"] = pd.to_datetime(df["closedTime"], utc=True, errors="coerce")
    selected = df[df["id_int"].isin(market_ids)].copy()
    selected = selected[selected["closed_dt"].notna()].sort_values("closed_dt")
    ordered = [int(x) for x in selected["id_int"].tolist()]
    missing = [int(mid) for mid in market_ids if int(mid) not in ordered]
    ordered.extend(sorted(missing))

    if len(ordered) < 2:
        return [], ordered

    eval_n = int(np.floor(len(ordered) * float(eval_fraction)))
    eval_n = max(1, min(len(ordered) - 1, eval_n))
    return ordered[:-eval_n], ordered[-eval_n:]


def _resolved_winning_outcomes(
    market_ids: List[int],
    winning_overrides: Optional[Dict[int, int]],
) -> Dict[int, int]:
    resolved: Dict[int, int] = {}
    for mid in market_ids:
        if winning_overrides is not None and mid in winning_overrides:
            resolved[int(mid)] = int(winning_overrides[mid])
            continue
        winning = get_winning_outcome(mid)
        if winning is not None:
            resolved[int(mid)] = int(winning)
    return resolved


def _optimize_full_system_variant(
    loader: HistoricalDataLoader,
    args: argparse.Namespace,
    train_market_ids: List[int],
    train_inferred_winners: Dict[int, int],
    train_res_stats: Dict[str, Any],
    *,
    variant_name: str,
    enable_clustering: bool,
    enable_layer2_attribution: bool,
    enable_jump_anticipation: bool,
) -> Dict[str, Any]:
    variant_output_dir = str(Path(args.output_dir) / "train" / variant_name)
    variant_res_stats = dict(train_res_stats)
    variant_res_stats.update(
        {
            "variant": variant_name,
            "enable_clustering": bool(enable_clustering),
            "enable_layer2_attribution": bool(enable_layer2_attribution),
            "enable_jump_anticipation": bool(enable_jump_anticipation),
        }
    )
    variant_prep = materialize_timeframe_prep(
        loader=loader,
        output_dir=variant_output_dir,
        candidate_market_ids=train_market_ids,
        inferred_winners=train_inferred_winners,
        res_stats=variant_res_stats,
        market_ids=sorted(train_inferred_winners.keys()),
        override_filename_prefix=f"{variant_name}_resolution_overrides",
    )
    logging.info(
        "Optimizing %s on %s train markets via %s "
        "(clustering=%s, layer2=%s, jump=%s)...",
        variant_name,
        f"{len(variant_prep.market_ids):,}",
        args.optimizer_mode,
        enable_clustering,
        enable_layer2_attribution,
        enable_jump_anticipation,
    )

    opt_args = argparse.Namespace(**vars(args).copy())
    opt_args.output_dir = variant_output_dir
    opt_args.enable_clustering = enable_clustering
    opt_args.enable_layer2_attribution = enable_layer2_attribution
    opt_args.enable_jump_anticipation = enable_jump_anticipation
    optimizer_out = run_timeframe_optimizer(loader, variant_prep, opt_args)
    best_config_path = Path(optimizer_out["best_config_path"])
    logging.info("Optimized %s config: %s", variant_name, best_config_path)
    with open(best_config_path, "r", encoding="utf-8") as f:
        best_config = json.load(f)
    return {
        "prep": variant_prep,
        "best_config": best_config,
        "best_config_path": best_config_path,
    }


def _load_full_system_config(config_path: str) -> Tuple[Dict[str, Any], Path]:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise ValueError(f"--full-system-config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"--full-system-config must point to a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--full-system-config must be valid JSON ({path}): {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"Could not read --full-system-config {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("--full-system-config JSON must be an object.")

    required_keys = ("clustering_config", "jump_anticipation_config")
    missing_keys = [key for key in required_keys if config.get(key) is None]
    if missing_keys:
        raise ValueError(
            "--full-system-config missing required non-null key(s): "
            + ", ".join(missing_keys)
        )

    invalid_keys = [key for key in required_keys if not isinstance(config.get(key), dict)]
    if invalid_keys:
        raise ValueError(
            "--full-system-config key(s) must be JSON objects: "
            + ", ".join(invalid_keys)
        )

    empty_keys = [key for key in required_keys if len(config.get(key, {})) == 0]
    if empty_keys:
        raise ValueError(
            "--full-system-config key(s) must contain parameter definitions: "
            + ", ".join(empty_keys)
        )

    return config, path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare system against SOTA baselines using only in-window trades "
            "for train optimization and eval replay."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/compare_sota_timeframe",
    )
    parser.add_argument(
        "--min-window-trades",
        type=int,
        default=1,
        help=(
            "Drop resolved markets with fewer than this many trades inside the "
            "train/eval replay window."
        ),
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Inclusive ISO start date for timeframe market selection")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Inclusive ISO end date for timeframe market selection")
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help=(
            "Explicit train-window start date. When paired with --train-end, "
            "--test-start, and --test-end, this overrides --train-split."
        ),
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Explicit train-window end date (requires --train-start/--test-start/--test-end).",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        help="Explicit eval/test-window start date (requires --train-start/--train-end/--test-end).",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Explicit eval/test-window end date (requires --train-start/--train-end/--test-start).",
    )
    parser.add_argument("--min-market-volume", type=float, default=0.0,
                        help="Minimum market volume when selecting timeframe markets")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.3,
        help="Chronological eval fraction for single-window timeframe runs. "
        "Example: --train-split 0.3 evaluates on the final 30%% of selected markets "
        "(training uses the first 70%%). For convenience, values >1 are interpreted as "
        "percentages, e.g. 30 == 0.30. Set 0 to disable split (train==eval). "
        "Ignored when explicit --train-start/--train-end/--test-start/--test-end are provided.",
    )
    parser.add_argument(
        "--skip-resolution-inference",
        action="store_true",
        default=False,
        help="Do not call infer_resolutions / SQLite cache; use static market_resolutions.py only "
        "(usually yields empty metrics for arbitrary timeframe IDs).",
    )
    parser.add_argument("--resolution-threshold", type=float, default=0.99,
                        help="Winning outcome inference threshold (last-price >= threshold)")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Passed to infer_resolutions (parity with other timeframe experiments)")
    parser.add_argument("--inferred-resolutions-db", type=str, default="inferred_resolutions.db",
                        help="SQLite cache of precomputed inferred resolutions")
    parser.add_argument(
        "--optimizer-mode",
        choices=("coordinate_descent", "alternating_det_clust"),
        default="alternating_det_clust",
        help="How to optimize the train-slice full-system config before evaluation.",
    )
    parser.add_argument("--n-passes", type=int, default=2)
    parser.add_argument("--objective", type=str, default="f0_5")
    parser.add_argument("--coarse-top-k", type=int, default=100)
    parser.add_argument("--coarse-trade-cap", type=int, default=500000)
    parser.add_argument("--enable-trade-prefilter", action="store_true", default=False)
    parser.add_argument(
        "--insider-plausible-only",
        action="store_true",
        help="Filter timeframe markets to insider-plausible (requires --classifications-path)",
    )
    parser.add_argument(
        "--non-insider-plausible-only",
        action="store_true",
        help="Filter timeframe markets to non-insider-plausible",
    )
    parser.add_argument(
        "--market-categories",
        type=str,
        nargs="+",
        default=None,
        help="Filter to categories (e.g. ELECTION SPORTS)",
    )
    parser.add_argument(
        "--exclude-categories",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--classifications-path",
        type=str,
        default="data/market_classifications.json",
    )
    parser.add_argument("--no-clustering", action="store_true", default=False,
                        help="Legacy flag. compare_sota now always trains/evaluates the base full-system "
                        "row with clustering off and the stacked row with clustering on.")
    parser.add_argument("--clustering-min-trade-size", type=float, default=5000.0)
    parser.add_argument("--no-jump-anticipation", action="store_true", default=False,
                        help="Disable jump anticipation for the base full-system run only; "
                        "the stacked run always enables it.")
    parser.add_argument(
        "--enable-layer2-attribution",
        action="store_true",
        default=False,
        help="Legacy flag. compare_sota now always trains/evaluates the base full-system "
        "row with Layer 2 off and the stacked row with Layer 2 on.",
    )
    parser.add_argument("--usdc-cache", type=str, default="data/usdc_transfers.db")
    parser.add_argument("--polygonscan-api-key", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--include-recidivism", action="store_true", default=False)
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--flag-rate-threshold", type=float, default=0.2)
    parser.add_argument("--if-n-estimators", type=int, default=200)
    parser.add_argument("--if-contamination", type=str, default="auto")
    parser.add_argument("--if-random-state", type=int, default=42)
    parser.add_argument("--timing-max-prior-trades", type=int, default=5,
                        help="Max prior trades for the naive timing heuristic")
    parser.add_argument("--timing-min-notional", type=float, default=5000.0,
                        help="Absolute notional threshold for the naive timing heuristic")
    parser.add_argument("--timing-max-hours", type=float, default=48.0,
                        help="Max hours to resolution for the naive timing heuristic")
    parser.add_argument("--mo-faithful-flag-percentile", type=float, default=5.0,
                        help="Top composite percentile flagged by the faithful Mitts-Ofir "
                             "screen when no matched flag rate is in effect")
    parser.add_argument("--mo-faithful-low-price-threshold", type=float, default=0.15,
                        help="Implied-price ceiling for the causal faithful Mitts-Ofir "
                             "pre-event timing proxy (low-price accumulation share)")
    parser.add_argument("--consob-n-components", type=str, default="3",
                        help="Faithful CONSOB PCA: K components (int) or explained-variance "
                             "ratio in (0,1) (e.g. 0.9). Fixed a priori, not label-tuned.")
    parser.add_argument("--consob-bucket-hours", type=int, default=6,
                        help="Faithful CONSOB: trajectory bucket width in hours (rescaled "
                             "daily->6h for the 24/7 prediction-market horizon)")
    parser.add_argument("--consob-investigation-hours", type=int, default=24,
                        help="Faithful CONSOB: investigation window length ending at the PSE bucket")
    parser.add_argument("--consob-d-theta", type=int, default=3,
                        help="Faithful CONSOB: max active buckets for the few-active-buckets clause")
    parser.add_argument("--consob-min-wallets-for-kde", type=int, default=8,
                        help="Faithful CONSOB: min eligible wallets before attempting a bimodal "
                             "KDE trough for eps_theta (else percentile fallback)")
    parser.add_argument("--consob-percentile-fallback", type=float, default=90.0,
                        help="Faithful CONSOB: s* percentile used for eps_theta when the KDE is "
                             "not cleanly bimodal or N is small")
    parser.add_argument("--min-usd-amount", type=float, default=None)
    parser.add_argument("--random-n-trials", type=int, default=5)
    parser.add_argument("--skip-random-baseline", action="store_true",
                        help="Skip the random flagging baseline to speed up testing.")
    parser.add_argument("--skip-full-system", action="store_true",
                        help="Skip full system evaluation (use saved results if available).")
    parser.add_argument(
        "--full-system-config",
        type=str,
        default=None,
        help=(
            "Path to a pre-optimized full-system config JSON. When provided, "
            "compare_sota skips both full-system optimization runs and skips the "
            "base full-system evaluation, then backtests one full_system row from "
            "this config. The JSON must include non-null "
            "clustering_config and jump_anticipation_config."
        ),
    )
    parser.set_defaults(
        enable_layer2_attribution=True,
        enable_jump_anticipation=True,
    )
    parser.add_argument(
        "--disable-layer2-attribution",
        dest="enable_layer2_attribution",
        action="store_false",
        help="Disable Layer 2 attribution (enabled by default for this experiment).",
    )
    parser.add_argument(
        "--disable-jump-anticipation",
        dest="enable_jump_anticipation",
        action="store_false",
        help="Disable jump anticipation (enabled by default for this experiment).",
    )
    args = parser.parse_args()

    if args.skip_full_system and args.full_system_config:
        parser.error("--skip-full-system and --full-system-config are mutually exclusive.")

    provided_full_system_config: Optional[Dict[str, Any]] = None
    provided_full_system_config_path: Optional[Path] = None
    if args.full_system_config:
        try:
            provided_full_system_config, provided_full_system_config_path = _load_full_system_config(
                args.full_system_config
            )
        except ValueError as exc:
            parser.error(str(exc))

    explicit_train_test_mode = any(
        value is not None
        for value in (args.train_start, args.train_end, args.test_start, args.test_end)
    )
    if explicit_train_test_mode and not all(
        value is not None
        for value in (args.train_start, args.train_end, args.test_start, args.test_end)
    ):
        parser.error(
            "When using explicit train/test windows, provide all four args: "
            "--train-start, --train-end, --test-start, --test-end."
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )

    if provided_full_system_config_path is not None:
        logging.info(
            "Using provided full-system config: %s "
            "(skipping full-system optimization and base full-system evaluation).",
            provided_full_system_config_path,
        )

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    def _select_and_optionally_infer(
        *,
        label: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[List[int], Optional[Dict[int, int]], List[int], Optional[Dict[str, Any]]]:
        candidate_market_ids = select_market_ids_in_timeframe(
            loader=loader,
            start_date=start_date,
            end_date=end_date,
            min_volume=args.min_market_volume,
            classifications_path=args.classifications_path,
            insider_plausible_only=args.insider_plausible_only,
            non_insider_plausible_only=args.non_insider_plausible_only,
            market_categories=args.market_categories,
            exclude_categories=args.exclude_categories,
        )
        logging.info("%s candidate markets: %s", label, f"{len(candidate_market_ids):,}")

        if args.skip_resolution_inference:
            logging.info(
                "%s: skipping resolution inference; candidate markets use static "
                "get_winning_outcome() only.",
                label,
            )
            return candidate_market_ids, None, candidate_market_ids, None

        inferred_winners, resolution_stats = infer_resolutions(
            loader=loader,
            market_ids=candidate_market_ids,
            resolution_threshold=args.resolution_threshold,
            min_trades=args.min_trades,
            min_usd_amount=_effective_min_usd_amount(args),
            inferred_resolutions_db=args.inferred_resolutions_db,
            save_cache=True,
        )
        resolved_market_ids = sorted(inferred_winners.keys())
        logging.info(
            "%s resolution inference: resolved=%s/%s, with_trades=%s, unresolved=%s",
            label,
            f"{resolution_stats['resolved']:,}",
            f"{resolution_stats['total_markets']:,}",
            f"{resolution_stats['with_trades']:,}",
            f"{resolution_stats['unresolved']:,}",
        )
        return resolved_market_ids, inferred_winners, candidate_market_ids, resolution_stats

    if args.enable_trade_prefilter and args.min_usd_amount is None:
        logging.error("--enable-trade-prefilter requires --min-usd-amount.")
        loader.close()
        sys.exit(1)

    timeframe_selection_mode = "all_resolved_static"
    eval_fraction_effective = float(args.train_split)
    train_candidate_market_ids: List[int] = []
    eval_candidate_market_ids: List[int] = []
    train_resolution_stats: Optional[Dict[str, Any]] = None
    eval_resolution_stats: Optional[Dict[str, Any]] = None
    train_winning_overrides: Optional[Dict[int, int]] = None

    if explicit_train_test_mode:
        if args.start_date is not None or args.end_date is not None:
            logging.info(
                "Explicit train/test windows provided; ignoring --start-date/--end-date and --train-split."
            )
        else:
            logging.info(
                "Explicit train/test windows provided; ignoring --train-split."
            )
        timeframe_selection_mode = "explicit_train_test_windows"
        eval_fraction_effective = 0.0

        train_market_ids, train_winning_overrides, train_candidate_market_ids, train_resolution_stats = (
            _prepare_window_markets(
                loader,
                args,
                start_date=str(args.train_start),
                end_date=str(args.train_end),
                override_filename_prefix="compare_sota_train_resolution_overrides",
            )
        )
        market_ids, winning_overrides, eval_candidate_market_ids, eval_resolution_stats = (
            _prepare_window_markets(
                loader,
                args,
                start_date=str(args.test_start),
                end_date=str(args.test_end),
                override_filename_prefix="compare_sota_test_resolution_overrides",
            )
        )
        logging.info(
            "Using explicit windows: train=%s markets, eval=%s markets.",
            f"{len(train_market_ids):,}",
            f"{len(market_ids):,}",
        )
        if not market_ids:
            logging.error("No markets selected for evaluation in explicit test window.")
            loader.close()
            sys.exit(1)
    elif args.start_date is not None or args.end_date is not None:
        timeframe_selection_mode = "single_window"
        market_ids, winning_overrides, eval_candidate_market_ids, eval_resolution_stats = (
            _select_and_optionally_infer(
                label="Timeframe",
                start_date=str(args.start_date),
                end_date=str(args.end_date),
            )
        )
        if not market_ids:
            logging.error(
                "No markets with inferred resolutions in this timeframe. "
                "Widen the window or fix data; use --skip-resolution-inference only if you "
                "have static resolutions for these IDs."
            )
            loader.close()
            sys.exit(1)

        split_arg = float(args.train_split)
        if split_arg < 0.0:
            logging.error("--train-split must be non-negative.")
            loader.close()
            sys.exit(1)
        if split_arg > 1.0:
            if split_arg >= 100.0:
                logging.error(
                    "--train-split=%s is invalid. Use eval fraction in [0,1), "
                    "or a percentage in (0,100).",
                    split_arg,
                )
                loader.close()
                sys.exit(1)
            logging.warning(
                "--train-split=%s interpreted as %.4f eval fraction "
                "(percentage compatibility mode).",
                split_arg,
                split_arg / 100.0,
            )
            split_arg = split_arg / 100.0
        if split_arg >= 1.0:
            logging.error("--train-split must be < 1.0 when provided as a fraction.")
            loader.close()
            sys.exit(1)

        eval_fraction_effective = split_arg
        if eval_fraction_effective > 0.0:
            timeframe_selection_mode = "single_window_chronological_split"
            train_market_ids, market_ids = _split_market_ids_by_close_time(
                loader=loader,
                market_ids=market_ids,
                eval_fraction=eval_fraction_effective,
            )
            if not train_market_ids:
                train_market_ids = list(market_ids)
                logging.info(
                    "Not enough markets for a strict chronological split; "
                    "falling back to train==eval on %s markets.",
                    f"{len(market_ids):,}",
                )
            if not market_ids:
                logging.error("Train/eval split produced an empty eval set.")
                loader.close()
                sys.exit(1)
            logging.info(
                "Chronological split by market close time: train=%s markets, eval=%s markets, eval_fraction=%.3f (eval tail %.1f%%)",
                f"{len(train_market_ids):,}",
                f"{len(market_ids):,}",
                float(eval_fraction_effective),
                float(eval_fraction_effective * 100.0),
            )
            if winning_overrides is not None:
                train_winning_overrides = {
                    int(mid): int(winning_overrides[mid])
                    for mid in train_market_ids
                    if mid in winning_overrides
                }
        else:
            train_market_ids = list(market_ids)
            train_winning_overrides = (
                {int(mid): int(win) for mid, win in winning_overrides.items()}
                if winning_overrides is not None
                else None
            )
            logging.info(
                "Train/eval split disabled: optimizing and evaluating on the same %s markets.",
                f"{len(market_ids):,}",
            )
    else:
        logging.error(
            "compare_sota_timeframe requires explicit train/test windows or "
            "--start-date/--end-date. All-resolved static mode has no trade replay window."
        )
        loader.close()
        sys.exit(1)

    if explicit_train_test_mode:
        train_trade_start = str(args.train_start)
        train_trade_end = str(args.train_end)
        eval_trade_start = str(args.test_start)
        eval_trade_end = str(args.test_end)
    else:
        train_trade_start = str(args.start_date)
        train_trade_end = str(args.end_date)
        eval_trade_start = str(args.start_date)
        eval_trade_end = str(args.end_date)

    if not market_ids:
        logging.error("No markets selected for evaluation.")
        loader.close()
        sys.exit(1)

    logging.info(f"Markets selected for compare_sota_timeframe (eval): {len(market_ids):,}")

    if winning_overrides is not None:
        logging.info(
            f"Inferred/cached eval winners: {len(winning_overrides):,} markets "
            f"(db={args.inferred_resolutions_db})."
        )
    elif (
        explicit_train_test_mode
        or args.start_date is not None
        or args.end_date is not None
    ):
        n_static = sum(1 for m in market_ids if get_winning_outcome(m) is not None)
        logging.warning(
            "Resolution inference skipped: static market_resolutions.py covers "
            f"{n_static:,} / {len(market_ids):,} selected eval markets."
        )

    train_inferred_winners = _resolved_winning_outcomes(
        train_market_ids,
        train_winning_overrides,
    )

    if not train_inferred_winners:
        if provided_full_system_config is None:
            logging.error("No resolved train markets available for optimization.")
            loader.close()
            sys.exit(1)
        logging.warning(
            "No resolved train markets available for optimization, but "
            "--full-system-config was provided so proceeding with config-driven "
            "full-system evaluation and eval-only fit-once baselines."
        )

    train_res_stats = dict(train_resolution_stats or {})
    train_res_stats.update(
        {
            "total_markets": len(train_market_ids),
            "resolved": len(train_inferred_winners),
            "unresolved": max(0, len(train_market_ids) - len(train_inferred_winners)),
            "source": "inferred" if train_winning_overrides is not None else "static",
            "eval_fraction": float(eval_fraction_effective),
            "eval_split_pct": float(eval_fraction_effective * 100.0),
            "train_split_pct": (
                float((1.0 - eval_fraction_effective) * 100.0)
                if eval_fraction_effective > 0.0
                else 0.0
            ),
            "selection_mode": timeframe_selection_mode,
        }
    )
    baseline_train_market_ids = sorted(train_inferred_winners.keys())
    if len(baseline_train_market_ids) != len(train_market_ids):
        logging.info(
            "Resolved train markets usable for fitting: %s / %s.",
            f"{len(baseline_train_market_ids):,}",
            f"{len(train_market_ids):,}",
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: List[Dict] = []
    base_config_path: Optional[Path] = None
    stack_config_path: Optional[Path] = None
    train_trade_filter_meta: Dict[str, Any] = {}
    eval_trade_filter_meta: Dict[str, Any] = {}
    train_window_trade_stats: Optional[Dict[str, Any]] = None
    eval_window_trade_stats: Optional[Dict[str, Any]] = None
    trade_filter_min_usd = args.min_usd_amount if args.enable_trade_prefilter else None
    base_config: Optional[Dict[str, Any]] = None
    stack_config: Optional[Dict[str, Any]] = None
    base_enable_jump = not args.no_jump_anticipation

    if not args.skip_full_system and provided_full_system_config is None:
        logging.info(
            "Optimizing full-system variants on train-window trades only (%s .. %s)",
            train_trade_start,
            train_trade_end,
        )
        with scoped_trade_time_filter(
            loader,
            start_date=train_trade_start,
            end_date=train_trade_end,
        ) as train_trade_filter_meta:
            kept, train_window_trade_stats = filter_markets_by_window_trade_count(
                loader,
                baseline_train_market_ids,
                min_window_trades=args.min_window_trades,
                min_usd_amount=trade_filter_min_usd,
                label="Train",
            )
            baseline_train_market_ids = kept
            train_market_ids = kept
            train_inferred_winners = {
                int(mid): int(train_inferred_winners[mid])
                for mid in kept
                if mid in train_inferred_winners
            }
            if not baseline_train_market_ids:
                logging.error(
                    "No train markets have enough trades inside the train replay window."
                )
                loader.close()
                sys.exit(1)
            base_variant = _optimize_full_system_variant(
                loader=loader,
                args=args,
                train_market_ids=train_market_ids,
                train_inferred_winners=train_inferred_winners,
                train_res_stats=train_res_stats,
                variant_name="full_system_base",
                enable_clustering=False,
                enable_layer2_attribution=False,
                enable_jump_anticipation=base_enable_jump,
            )
            stack_variant = _optimize_full_system_variant(
                loader=loader,
                args=args,
                train_market_ids=train_market_ids,
                train_inferred_winners=train_inferred_winners,
                train_res_stats=train_res_stats,
                variant_name="full_system_clustering_layer2",
                enable_clustering=True,
                enable_layer2_attribution=True,
                enable_jump_anticipation=True,
            )
            base_config = base_variant["best_config"]
            stack_config = stack_variant["best_config"]
            base_config_path = base_variant["best_config_path"]
            stack_config_path = stack_variant["best_config_path"]

    separate_train_eval_trade_windows = (
        train_trade_start != eval_trade_start or train_trade_end != eval_trade_end
    )

    logging.info(
        "Evaluating all baselines on test-window trades only (%s .. %s)",
        eval_trade_start,
        eval_trade_end,
    )
    with scoped_trade_time_filter(
        loader,
        start_date=eval_trade_start,
        end_date=eval_trade_end,
    ) as eval_trade_filter_meta:
        kept, eval_window_trade_stats = filter_markets_by_window_trade_count(
            loader,
            market_ids,
            min_window_trades=args.min_window_trades,
            min_usd_amount=trade_filter_min_usd,
            label="Eval",
        )
        market_ids = kept
        if winning_overrides is not None:
            winning_overrides = {
                int(mid): int(winning_overrides[mid])
                for mid in kept
                if mid in winning_overrides
            }
        if not market_ids:
            logging.error(
                "No eval markets have enough trades inside the eval replay window."
            )
            loader.close()
            sys.exit(1)

        # ------------------------------------------------------------------
        # Baseline 1: Full system (optimized rows or config-driven row)
        # ------------------------------------------------------------------
        if not args.skip_full_system:
            if provided_full_system_config is not None:
                if provided_full_system_config_path is None:
                    raise RuntimeError("Internal error: missing provided full-system config path.")

                provided_config = provided_full_system_config
                provided_clustering = provided_config["clustering_config"]
                provided_jump = provided_config["jump_anticipation_config"]
                base_config_path = provided_full_system_config_path

                logging.info(
                    f"\n{'='*80}\nBASELINE: full_system "
                    f"(provided config: clustering + jump + Layer 2 on)\n{'='*80}"
                )
                start = time.time()
                result = evaluate_config(
                    config=provided_config,
                    loader=loader,
                    market_ids=market_ids,
                    prediction_mode=args.prediction_mode,
                    flag_rate_threshold=args.flag_rate_threshold,
                    suspicion_threshold=args.suspicion_threshold,
                    z_score_threshold=args.z_score_threshold,
                    min_wallet_notional=args.min_wallet_notional,
                    min_usd_amount=_effective_min_usd_amount(args),
                    include_recidivism=args.include_recidivism,
                    clustering_config=provided_clustering,
                    clustering_min_trade_size=args.clustering_min_trade_size,
                    jump_anticipation_config=provided_jump,
                    measure_memory=False,
                    winning_outcomes_override=winning_overrides,
                    enable_layer2_attribution=args.enable_layer2_attribution,
                    usdc_cache_db=args.usdc_cache,
                    polygonscan_api_key=args.polygonscan_api_key,
                    quiet_per_market=True,
                )
                elapsed = time.time() - start
                result.save(args.output_dir, tag="full_system")
                row_cli = _full_system_metrics_row(result, elapsed, "full_system", args.min_usd_amount)
                row_cli["full_system_variant"] = "provided_config"
                row_cli["uses_clustering"] = True
                row_cli["uses_layer2"] = True
                row_cli["uses_jump"] = True
                row_cli["trained_with_clustering"] = True
                row_cli["trained_with_layer2"] = True
                row_cli["trained_with_jump"] = True
                row_cli["optimized_config_path"] = str(base_config_path)
                results.append(row_cli)
                logging.info(
                    f"  -> wallet_f0.5={results[-1]['wallet_f0_5']:.4f}, "
                    f"flagged_wallet_avg_return={results[-1]['flagged_avg_return']:.2%}, "
                    f"flagged_trades={results[-1]['flagged_trades']:,}, "
                    f"elapsed={elapsed:.1f}s"
                )
            else:
                if base_config is None or stack_config is None:
                    raise RuntimeError(
                        "Full-system configs missing after train-window optimization."
                    )

                logging.info(
                    f"\n{'='*80}\nBASELINE: full_system "
                    f"(separately optimized: clustering off, Layer 2 off)\n{'='*80}"
                )
                start = time.time()
                result = evaluate_config(
                    config=base_config,
                    loader=loader,
                    market_ids=market_ids,
                    prediction_mode=args.prediction_mode,
                    flag_rate_threshold=args.flag_rate_threshold,
                    suspicion_threshold=args.suspicion_threshold,
                    z_score_threshold=args.z_score_threshold,
                    min_wallet_notional=args.min_wallet_notional,
                    min_usd_amount=_effective_min_usd_amount(args),
                    include_recidivism=args.include_recidivism,
                    clustering_config=None,
                    clustering_min_trade_size=args.clustering_min_trade_size,
                    jump_anticipation_config=base_config.get("jump_anticipation_config", None),
                    measure_memory=False,
                    winning_outcomes_override=winning_overrides,
                    enable_layer2_attribution=False,
                    usdc_cache_db=args.usdc_cache,
                    polygonscan_api_key=args.polygonscan_api_key,
                    quiet_per_market=True,
                )
                elapsed = time.time() - start
                result.save(args.output_dir, tag="full_system")
                row_cli = _full_system_metrics_row(result, elapsed, "full_system", args.min_usd_amount)
                row_cli["full_system_variant"] = "base"
                row_cli["uses_clustering"] = False
                row_cli["uses_layer2"] = False
                row_cli["uses_jump"] = base_config.get("jump_anticipation_config", None) is not None
                row_cli["trained_with_clustering"] = False
                row_cli["trained_with_layer2"] = False
                row_cli["trained_with_jump"] = bool(base_enable_jump)
                row_cli["optimized_config_path"] = str(base_config_path)
                results.append(row_cli)
                logging.info(
                    f"  -> wallet_f0.5={results[-1]['wallet_f0_5']:.4f}, "
                    f"flagged_wallet_avg_return={results[-1]['flagged_avg_return']:.2%}, "
                    f"flagged_trades={results[-1]['flagged_trades']:,}, "
                    f"elapsed={elapsed:.1f}s"
                )

                stack_clustering = stack_config.get("clustering_config", DEFAULT_CLUSTERING_CONFIG)
                stack_jump = stack_config.get("jump_anticipation_config", None)
                logging.info(
                    f"\n{'='*80}\nBASELINE: full_system_clustering_layer2 "
                    f"(separately optimized: clustering + jump + Layer 2 on)\n{'='*80}"
                )
                start = time.time()
                result_stack = evaluate_config(
                    config=stack_config,
                    loader=loader,
                    market_ids=market_ids,
                    prediction_mode=args.prediction_mode,
                    flag_rate_threshold=args.flag_rate_threshold,
                    suspicion_threshold=args.suspicion_threshold,
                    z_score_threshold=args.z_score_threshold,
                    min_wallet_notional=args.min_wallet_notional,
                    min_usd_amount=_effective_min_usd_amount(args),
                    include_recidivism=args.include_recidivism,
                    clustering_config=stack_clustering,
                    clustering_min_trade_size=args.clustering_min_trade_size,
                    jump_anticipation_config=stack_jump,
                    measure_memory=False,
                    winning_outcomes_override=winning_overrides,
                    enable_layer2_attribution=True,
                    usdc_cache_db=args.usdc_cache,
                    polygonscan_api_key=args.polygonscan_api_key,
                    quiet_per_market=True,
                )
                elapsed_stack = time.time() - start
                result_stack.save(args.output_dir, tag="full_system_clustering_layer2")
                row_stack = _full_system_metrics_row(
                    result_stack, elapsed_stack, "full_system_clustering_layer2", args.min_usd_amount,
                )
                row_stack["full_system_variant"] = "stacked"
                row_stack["uses_clustering"] = True
                row_stack["uses_layer2"] = True
                row_stack["uses_jump"] = stack_jump is not None
                row_stack["trained_with_clustering"] = True
                row_stack["trained_with_layer2"] = True
                row_stack["trained_with_jump"] = True
                row_stack["optimized_config_path"] = str(stack_config_path)
                results.append(row_stack)
                logging.info(
                    f"  -> wallet_f0.5={results[-1]['wallet_f0_5']:.4f}, "
                    f"flagged_wallet_avg_return={results[-1]['flagged_avg_return']:.2%}, "
                    f"flagged_trades={results[-1]['flagged_trades']:,}, "
                    f"elapsed={elapsed_stack:.1f}s"
                )
        else:
            logging.info("Skipping full system evaluation (--skip-full-system).")

        # ------------------------------------------------------------------
        # Baseline 2: Isolation Forest (Liu et al., 2008)
        # ------------------------------------------------------------------
        logging.info(f"\n{'='*80}\nBASELINE: isolation_forest\n{'='*80}")
        logging.info("Extracting features and fitting Isolation Forest...")

        match_flag_rate: Optional[float] = None
        full_system_row = next((r for r in results if r["baseline"] == "full_system"), None)
        if full_system_row and full_system_row["flagged_trades"] > 0:
            total_buy_estimate = (
                full_system_row["flagged_trades"]
                + full_system_row.get("unflagged_trades", 0)
            )
            if total_buy_estimate > 0:
                match_flag_rate = full_system_row["flagged_trades"] / total_buy_estimate
                logging.info(f"Matched flag rate from full system: {match_flag_rate:.4f}")

        feature_matrix, trade_info = extract_isolation_forest_features(
            loader,
            market_ids,
            min_usd_amount=_effective_min_usd_amount(args),
            winning_outcome_overrides=winning_overrides,
        )
        if_prefit_model = None
        train_feature_matrix = None
        train_trade_info = None
        if baseline_train_market_ids:
            train_if_overrides = {
                int(mid): int(train_inferred_winners[mid])
                for mid in baseline_train_market_ids
                if mid in train_inferred_winners
            }
            if separate_train_eval_trade_windows:
                with scoped_trade_time_filter(
                    loader,
                    start_date=train_trade_start,
                    end_date=train_trade_end,
                ):
                    train_feature_matrix, train_trade_info = extract_isolation_forest_features(
                        loader,
                        baseline_train_market_ids,
                        min_usd_amount=_effective_min_usd_amount(args),
                        winning_outcome_overrides=train_if_overrides,
                    )
            else:
                train_feature_matrix, train_trade_info = extract_isolation_forest_features(
                    loader,
                    baseline_train_market_ids,
                    min_usd_amount=_effective_min_usd_amount(args),
                    winning_outcome_overrides=train_if_overrides,
                )
            if train_feature_matrix.shape[0] > 0:
                logging.info(
                    "IF train/eval split: fitting on %s train trades, scoring %s eval trades.",
                    f"{train_feature_matrix.shape[0]:,}",
                    f"{feature_matrix.shape[0]:,}",
                )
                if_prefit_model = IsolationForest(
                    n_estimators=args.if_n_estimators,
                    contamination=args.if_contamination,
                    random_state=args.if_random_state,
                    n_jobs=-1,
                )
                if_prefit_model.fit(train_feature_matrix)
            else:
                logging.warning(
                    "IF train split is enabled but produced no train trades; falling back to eval-only fitting."
                )

        winning_outcomes: Dict[int, Optional[int]] = {}
        for mid in market_ids:
            if winning_overrides is not None and mid in winning_overrides:
                winning_outcomes[mid] = int(winning_overrides[mid])
            else:
                winning_outcomes[mid] = get_winning_outcome(mid)

        all_entries = _load_full_market_ground_truth_entries(loader, market_ids)
        logging.info(
            "SOTA wallet ground truth: using %s full-market trades across %s eval markets "
            "(unfiltered by replay window/min-USD), matching full-system wallet labels.",
            f"{len(all_entries):,}",
            f"{len(market_ids):,}",
        )

        if_result = run_isolation_forest_baseline(
            feature_matrix, trade_info, market_ids, all_entries, winning_outcomes,
            n_estimators=args.if_n_estimators,
            contamination=args.if_contamination,
            random_state=args.if_random_state,
            match_flag_rate=None,
            min_usd_amount=_effective_min_usd_amount(args),
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
            model=if_prefit_model,
        )
        if_result["baseline"] = "isolation_forest"
        results.append(if_result)

        # ------------------------------------------------------------------
        # Baseline 3: Naive timing heuristic
        # ------------------------------------------------------------------
        logging.info(f"\n{'='*80}\nBASELINE: timing_heuristic\n{'='*80}")

        timing_result = run_timing_heuristic_baseline(
            loader=loader,
            market_ids=market_ids,
            all_entries=all_entries,
            winning_outcomes=winning_outcomes,
            max_prior_trades=args.timing_max_prior_trades,
            min_notional=args.timing_min_notional,
            max_hours=args.timing_max_hours,
            min_usd_amount=_effective_min_usd_amount(args),
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
        )
        results.append(timing_result)

        # ------------------------------------------------------------------
        # Baseline 4: CONSOB / Ravagnani four-condition screen
        # (event-anchored, per-market eval-fit; non-causal by design)
        # ------------------------------------------------------------------
        logging.info(
            f"\n{'='*80}\nBASELINE: consob_pca (Ravagnani et al., 2024)\n{'='*80}"
        )

        consob_faithful = run_consob_pca_faithful_baseline(
            loader=loader,
            market_ids=market_ids,
            all_entries=all_entries,
            winning_outcomes=winning_outcomes,
            bucket_hours=args.consob_bucket_hours,
            investigation_hours=args.consob_investigation_hours,
            d_theta=args.consob_d_theta,
            n_components=parse_n_components(args.consob_n_components),
            min_wallets_for_kde=args.consob_min_wallets_for_kde,
            percentile_fallback=args.consob_percentile_fallback,
            min_usd_amount=_effective_min_usd_amount(args),
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
        )
        results.append(consob_faithful)

        # ------------------------------------------------------------------
        # Baseline 5: Mitts & Ofir five-signal pair-level screen
        # ------------------------------------------------------------------
        logging.info(f"\n{'='*80}\nBASELINE: mitts_ofir\n{'='*80}")

        mitts_faithful_retrospective = run_mitts_ofir_faithful_retrospective(
            loader=loader,
            market_ids=market_ids,
            all_entries=all_entries,
            winning_outcomes=winning_outcomes,
            flag_percentile=args.mo_faithful_flag_percentile,
            match_flag_rate=match_flag_rate,
            min_usd_amount=_effective_min_usd_amount(args),
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
        )
        results.append(mitts_faithful_retrospective)

        mitts_faithful_causal = run_mitts_ofir_faithful_causal(
            loader=loader,
            market_ids=market_ids,
            all_entries=all_entries,
            winning_outcomes=winning_outcomes,
            flag_percentile=args.mo_faithful_flag_percentile,
            match_flag_rate=match_flag_rate,
            low_price_threshold=args.mo_faithful_low_price_threshold,
            min_usd_amount=_effective_min_usd_amount(args),
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
        )
        results.append(mitts_faithful_causal)

        # ------------------------------------------------------------------
        # Baseline 6: Random flagging at matched rate
        # ------------------------------------------------------------------
        if args.skip_random_baseline:
            logging.info("Skipping random baseline (--skip-random-baseline).")
        else:
            logging.info(f"\n{'='*80}\nBASELINE: random_flagging\n{'='*80}")

            if match_flag_rate is not None:
                random_result = run_random_baseline(
                    loader, market_ids,
                    flag_rate=max(match_flag_rate, 0.01),
                    n_trials=args.random_n_trials,
                    min_usd_amount=_effective_min_usd_amount(args),
                    all_entries=all_entries,
                    winning_outcomes=winning_outcomes,
                    z_score_threshold=args.z_score_threshold,
                    min_wallet_notional=args.min_wallet_notional,
                )
            else:
                random_result = run_random_baseline(
                    loader,
                    market_ids,
                    flag_rate=0.01,
                    n_trials=args.random_n_trials,
                    min_usd_amount=_effective_min_usd_amount(args),
                    all_entries=all_entries,
                    winning_outcomes=winning_outcomes,
                    z_score_threshold=args.z_score_threshold,
                    min_wallet_notional=args.min_wallet_notional,
                )
            results.append(random_result)

        # ------------------------------------------------------------------
    # Save and display results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    df = _add_wallet_classification_metrics(df)
    eval_start_date = args.test_start if explicit_train_test_mode else args.start_date
    eval_end_date = args.test_end if explicit_train_test_mode else args.end_date
    train_start_date = (
        args.train_start
        if explicit_train_test_mode
        else (args.start_date if (args.start_date is not None or args.end_date is not None) else None)
    )
    train_end_date = (
        args.train_end
        if explicit_train_test_mode
        else (args.end_date if (args.start_date is not None or args.end_date is not None) else None)
    )
    date_based_selection = bool(
        explicit_train_test_mode or args.start_date is not None or args.end_date is not None
    )

    df["start_date"] = eval_start_date
    df["end_date"] = eval_end_date
    df["train_start_date"] = train_start_date
    df["train_end_date"] = train_end_date
    df["test_start_date"] = eval_start_date
    df["test_end_date"] = eval_end_date
    df["date_selection_mode"] = timeframe_selection_mode
    df["timeframe_infer_resolutions"] = bool(
        date_based_selection and not args.skip_resolution_inference
    )
    df["optimizer_mode"] = args.optimizer_mode
    df["objective_metric"] = args.objective
    using_provided_full_system_config = provided_full_system_config_path is not None
    full_system_mode = (
        "provided_config"
        if using_provided_full_system_config
        else ("optimized_dual" if not args.skip_full_system else "skipped")
    )
    df["full_system_mode"] = full_system_mode
    df["full_system_base_config_path"] = (
        str(base_config_path) if base_config_path is not None else None
    )
    df["full_system_clustering_layer2_config_path"] = (
        str(stack_config_path) if stack_config_path is not None else None
    )
    df["full_system_config_override_path"] = (
        str(provided_full_system_config_path)
        if provided_full_system_config_path is not None
        else None
    )
    if full_system_mode == "optimized_dual":
        df["base_forces_clustering_off"] = True
        df["base_forces_layer2_off"] = True
        df["base_jump_enabled"] = bool(not args.no_jump_anticipation)
        df["stack_forces_clustering_on"] = True
        df["stack_forces_layer2_on"] = True
        df["stack_forces_jump_on"] = True
    else:
        df["base_forces_clustering_off"] = None
        df["base_forces_layer2_off"] = None
        df["base_jump_enabled"] = None
        df["stack_forces_clustering_on"] = None
        df["stack_forces_layer2_on"] = None
        df["stack_forces_jump_on"] = None
    df["train_split_arg"] = float(args.train_split)
    df["eval_fraction"] = float(eval_fraction_effective)
    df["eval_split_pct"] = float(eval_fraction_effective * 100.0)
    df["train_split_pct"] = (
        float((1.0 - eval_fraction_effective) * 100.0)
        if eval_fraction_effective > 0.0
        else 0.0
    )
    df["train_markets"] = int(len(baseline_train_market_ids))
    df["eval_markets"] = int(len(market_ids))
    df["train_eval_market_overlap"] = int(
        len(set(int(mid) for mid in baseline_train_market_ids) & set(int(mid) for mid in market_ids))
    )
    df["fit_on_eval_markets"] = df["train_eval_market_overlap"] > 0
    df["train_candidate_markets"] = int(len(train_candidate_market_ids))
    df["eval_candidate_markets"] = int(len(eval_candidate_market_ids))
    df["experiment"] = "compare_sota_timeframe"
    df["train_trade_window_start"] = train_trade_start
    df["train_trade_window_end"] = train_trade_end
    df["eval_trade_window_start"] = eval_trade_start
    df["eval_trade_window_end"] = eval_trade_end
    df["wallet_ground_truth_trade_history"] = "full_market_unfiltered"
    df["min_window_trades"] = int(args.min_window_trades)
    df["train_window_trade_stats"] = json.dumps(train_window_trade_stats or {}, default=str)
    df["eval_window_trade_stats"] = json.dumps(eval_window_trade_stats or {}, default=str)
    if "copytrade_total_flagged_buys" in df.columns:
        df["n_copied_trades"] = df["copytrade_total_flagged_buys"]
    elif "flagged_trades" in df.columns:
        df["n_copied_trades"] = df["flagged_trades"]
    if "flagged_trades" in df.columns:
        df = df.drop(columns=["flagged_trades"])
    if "n_copied_trades" in df.columns and "copytrade_portfolio_roi" in df.columns:
        ordered_cols = [c for c in df.columns if c != "n_copied_trades"]
        roi_idx = ordered_cols.index("copytrade_portfolio_roi")
        ordered_cols.insert(roi_idx, "n_copied_trades")
        df = df[ordered_cols]
    path = f"{args.output_dir}/sota_comparison_timeframe_{timestamp}.csv"
    df.to_csv(path, index=False)

    try:
        if_model = IsolationForest(
            n_estimators=args.if_n_estimators,
            contamination=args.if_contamination,
            random_state=args.if_random_state,
            n_jobs=-1,
        )
        if_model.fit(feature_matrix)
        feature_means = np.mean(feature_matrix, axis=0)
        score_corr = np.array([
            np.corrcoef(feature_matrix[:, j], if_model.decision_function(feature_matrix))[0, 1]
            for j in range(feature_matrix.shape[1])
        ])
        feature_info = pd.DataFrame({
            "feature": IF_FEATURE_NAMES,
            "mean": feature_means,
            "score_correlation": score_corr,
        })
        feature_path = f"{args.output_dir}/if_feature_info_{timestamp}.csv"
        feature_info.to_csv(feature_path, index=False)
        logging.info(f"IF feature info saved: {feature_path}")
    except Exception as e:
        logging.warning(f"Could not compute IF feature info: {e}")

    print(f"\n{'='*80}")
    print("SOTA COMPARISON RESULTS (TRADE-WINDOW REPLAY)")
    print(f"{'='*80}")
    display_cols = [
        "baseline", "deployable_live", "wallet_precision", "wallet_f1", "wallet_f0_5", "num_flagged_wallets",
        "num_flags", "flagged_avg_return", "tp_avg_return", "fp_avg_return",
        "tp", "fp", "fn",
        "flagged_wallet_mean_net_pnl", "flagged_wallet_median_net_pnl",
        "n_copied_trades", "copytrade_portfolio_roi", "copytrade_mean_trade_return",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    display_df = df[existing_cols].copy()
    for col in ("flagged_avg_return", "tp_avg_return", "fp_avg_return"):
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0.0).map(
                lambda v: f"{float(v):.2%}"
            )
    for col in ("copytrade_portfolio_roi", "copytrade_mean_trade_return"):
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(0.0).map(
                lambda v: f"{float(v):.2%}"
            )
    print(display_df.to_string(index=False))
    print(f"\nSaved: {path}")
    loader.close()


if __name__ == "__main__":
    main()
