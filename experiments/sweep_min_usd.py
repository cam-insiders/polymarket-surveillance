"""
Experiment: Sweep min_usd_amount (trade prefilter threshold).
Usage:
    python -m experiments.sweep_min_usd path/to/config.json \\
        --start-date 2025-01-01 --end-date 2025-03-31

    python -m experiments.sweep_min_usd path/to/config.json \\
        --start-date 2025-01-01 --end-date 2025-01-31 --insider-plausible-only

    python -m experiments.sweep_min_usd path/to/config.json \\
        --start-date 2025-01-01 --end-date 2025-03-31 \\
        --values 0 100 500 1000 5000

Charting:
    python -m experiments.charting.sweep_min_usd_charts --list
    python -m experiments.charting.sweep_min_usd_charts --csv experiments/results/sweep_min_usd/sweep_min_usd_comparison_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import DEFAULT_CLUSTERING_CONFIG, evaluate_config
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode

from experiments.timeframe_market_common import (
    _normalize_category_list,
)
from experiments.common.timeframe import (
    filter_markets_by_window_trade_count,
    prepare_timeframe_inference,
    scoped_trade_time_filter,
)

DEFAULT_VALUES = [None, 50, 100, 250, 500, 1000, 2500, 5000, 10000]


def _evaluate_at_min_usd(
    config: dict,
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes_override: Dict[int, int],
    min_usd: Optional[float],
    args: argparse.Namespace,
):
    """Single ``evaluate_config`` call inside the active trade-window scope."""
    clustering_config = None
    if not args.no_clustering:
        clustering_config = config.get("clustering_config", DEFAULT_CLUSTERING_CONFIG)

    jump_anticipation_config = None
    if not args.no_jump_anticipation:
        jump_anticipation_config = config.get("jump_anticipation_config", None)

    return evaluate_config(
        config=config,
        loader=loader,
        market_ids=market_ids,
        prediction_mode=args.prediction_mode,
        flag_rate_threshold=args.flag_rate_threshold,
        suspicion_threshold=args.suspicion_threshold,
        z_score_threshold=args.z_score_threshold,
        min_wallet_notional=args.min_wallet_notional,
        min_usd_amount=min_usd,
        include_recidivism=args.include_recidivism,
        clustering_config=clustering_config,
        clustering_min_trade_size=args.clustering_min_trade_size,
        jump_anticipation_config=jump_anticipation_config,
        copytrade_fixed_size=args.copytrade_fixed_size,
        measure_memory=False,
        winning_outcomes_override=winning_outcomes_override,
        enable_layer2_attribution=args.enable_layer2_attribution,
        usdc_cache_db=args.usdc_cache,
        polygonscan_api_key=args.polygonscan_api_key,
    )


def run_sweep(
    config: dict,
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes_override: Dict[int, int],
    values: List[Optional[float]],
    output_dir: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run evaluation for each min_usd_amount value. Returns comparison DataFrame."""
    rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with scoped_trade_time_filter(
        loader,
        start_date=args.start_date,
        end_date=args.end_date,
    ) as trade_filter:
        for i, min_usd in enumerate(values):
            label = f"min_usd_{min_usd}" if min_usd is not None else "min_usd_none"
            logging.info(f"\n{'='*80}")
            logging.info(f"SWEEP [{i+1}/{len(values)}]: min_usd_amount = {min_usd}")
            logging.info(f"{'='*80}")

            kept_market_ids, window_stats = filter_markets_by_window_trade_count(
                loader,
                market_ids,
                min_window_trades=int(args.min_window_trades),
                min_usd_amount=min_usd,
                label=label,
            )
            base_row = {
                "min_usd_amount": min_usd if min_usd is not None else 0,
                "min_usd_label": label,
                "input_resolved_markets": len(market_ids),
                "markets_after_trade_filter": len(kept_market_ids),
                "min_window_trades": int(args.min_window_trades),
                "trade_window_start": trade_filter["start_time"],
                "trade_window_end": trade_filter["end_time"],
                "window_total_trades": window_stats["total_window_trades"],
                "window_zero_trade_markets": window_stats["zero_trade_markets"],
            }
            if not kept_market_ids:
                rows.append({**base_row, "error": "no_markets_after_trade_filter"})
                logging.warning("Skipping %s: no markets have enough in-window trades.", label)
                continue

            kept_winners = {
                int(mid): int(winning_outcomes_override[mid])
                for mid in kept_market_ids
                if mid in winning_outcomes_override
            }
            start = time.time()
            result = _evaluate_at_min_usd(
                config=config,
                loader=loader,
                market_ids=kept_market_ids,
                winning_outcomes_override=kept_winners,
                min_usd=min_usd,
                args=args,
            )
            elapsed = time.time() - start

            result.save(output_dir, tag=label)

            pooled = result.event_study_pooled.get("pooled", {})
            cs = result.copytrade_summary
            agg = result.aggregate_performance

            rows.append({
                **base_row,
                "flagged_trades": pooled.get("total_flagged_trades", 0),
                "unflagged_trades": pooled.get("total_unflagged_trades", 0),
                "flagged_mean_return": pooled.get("pooled_flagged_mean_return", 0),
                "unflagged_mean_return": pooled.get("pooled_unflagged_mean_return", 0),
                "mean_return_diff": pooled.get("pooled_mean_return_diff", 0),
                "mean_cohens_d": pooled.get("mean_cohens_d", 0),
                "flagged_wallets": cs.get("flagged", {}).get("count", 0),
                "tp_wallets": cs.get("tp", {}).get("count", 0),
                "fp_wallets": cs.get("fp", {}).get("count", 0),
                "fn_wallets": cs.get("fn", {}).get("count", 0),
                "flagged_avg_return": cs.get("flagged", {}).get("avg_return", 0),
                "tp_avg_return": cs.get("tp", {}).get("avg_return", 0),
                "fp_avg_return": cs.get("fp", {}).get("avg_return", 0),
                "total_trades_processed": agg.total_trades,
                "trades_per_second": agg.overall_trades_per_second,
                "detection_p95_us": agg.detection_latency_p95_us,
                "wall_clock_seconds": elapsed,
            })

            logging.info(f"  -> {label}: diff={pooled.get('pooled_mean_return_diff', 0):+.4f}, "
                         f"flagged={pooled.get('total_flagged_trades', 0):,}, "
                         f"d={pooled.get('mean_cohens_d', 0):.3f}, "
                         f"elapsed={elapsed:.1f}s")

    df = pd.DataFrame(rows)
    comparison_path = f"{output_dir}/sweep_min_usd_comparison_{timestamp}.csv"
    df.to_csv(comparison_path, index=False)
    logging.info(f"\nComparison saved: {comparison_path}")

    return df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep min_usd_amount trade prefilter on a timeframe "
            "(markets resolve in-window; replay uses only in-window trades)."
        )
    )
    parser.add_argument("config_path", type=str, help="Path to detector config JSON")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Inclusive ISO start date (market closedTime)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Inclusive ISO end date (market closedTime)",
    )
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="min_usd values to sweep. Use 0 for None (no filter).")
    parser.add_argument(
        "--min-window-trades",
        type=int,
        default=1,
        help="Drop resolved markets with fewer than this many in-window trades.",
    )
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-market-volume", type=float, default=0.0)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--inferred-resolutions-db", type=str, default="inferred_resolutions.db")
    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--flag-rate-threshold", type=float, default=0.2)
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--include-recidivism", action="store_true", default=False)
    parser.add_argument("--clustering-min-trade-size", type=float, default=5000.0)
    parser.add_argument("--no-clustering", action="store_true", default=False)
    parser.add_argument(
        "--enable-layer2-attribution",
        action="store_true",
        default=False,
        help="Enable Layer 2 attribution analysis for clustering backtests.",
    )
    parser.add_argument("--usdc-cache", type=str, default="data/usdc_transfers.db")
    parser.add_argument("--polygonscan-api-key", type=str, default=None)
    parser.add_argument("--no-jump-anticipation", action="store_true", default=False)
    parser.add_argument("--copytrade-fixed-size", type=float, default=100.0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--insider-plausible-only",
        action="store_true",
        help="Filter to markets classified as insider-plausible",
    )
    parser.add_argument(
        "--non-insider-plausible-only",
        action="store_true",
        help="Filter to markets classified as non-insider-plausible",
    )
    parser.add_argument(
        "--market-categories",
        type=str,
        nargs="+",
        default=None,
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
    parser.add_argument("--output-dir", type=str, default="experiments/results/sweep_min_usd")
    parser.add_argument(
        "--verbose-output",
        action="store_true",
        default=False,
        help="Verbose evaluation logging (default: quieter per-iteration logs).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.insider_plausible_only and args.non_insider_plausible_only:
        raise SystemExit(
            "--insider-plausible-only and --non-insider-plausible-only are mutually exclusive"
        )

    args.market_categories = _normalize_category_list(args.market_categories)
    args.exclude_categories = _normalize_category_list(args.exclude_categories)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    set_experiment_backtest_log_quiet_mode(enabled=not args.verbose_output)

    if args.values is not None:
        values = [None if v == 0 else v for v in args.values]
    else:
        values = DEFAULT_VALUES

    with open(args.config_path, encoding="utf-8") as f:
        config = json.load(f)

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    # Infer winners once on the full trade tape so labels do not depend on sweep values.
    prep = prepare_timeframe_inference(
        loader,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        min_market_volume=args.min_market_volume,
        classifications_path=args.classifications_path,
        insider_plausible_only=args.insider_plausible_only,
        non_insider_plausible_only=args.non_insider_plausible_only,
        market_categories=args.market_categories,
        exclude_categories=args.exclude_categories,
        resolution_threshold=args.resolution_threshold,
        min_trades=args.min_trades,
        min_usd_amount=None,
        inferred_resolutions_db=args.inferred_resolutions_db,
        enable_trade_prefilter=False,
        override_filename_prefix="sweep_min_usd_resolution_overrides",
    )
    market_ids = list(prep.market_ids)
    winning_overrides = dict(prep.inferred_winners)

    if not market_ids:
        raise SystemExit("No markets resolved in the selected timeframe; nothing to evaluate.")

    if not args.verbose_output:
        logging.getLogger("backtesting.evaluation").setLevel(logging.WARNING)

    print(
        f"\nSweeping min_usd_amount across {len(values)} values on {len(market_ids):,} markets "
        f"(resolve + trade window {args.start_date} .. {args.end_date}, "
        f"insider_plausible_only={args.insider_plausible_only})"
    )
    df = run_sweep(
        config,
        loader,
        market_ids,
        winning_overrides,
        values,
        args.output_dir,
        args,
    )

    print(f"\n{'='*80}")
    print("SWEEP RESULTS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    loader.close()


if __name__ == "__main__":
    main()
