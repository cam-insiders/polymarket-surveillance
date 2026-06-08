"""
Experiment: optimize and backtest using only trades inside each timeframe.

Market selection still uses markets whose ``closedTime`` falls inside the
requested train/test windows. Resolution inference also uses the full market
trade history so winner labels are not distorted by the trade-window ablation.
Only the detector/evaluation replay is restricted to trades inside the
corresponding train or test window. Wallet ground-truth labels/PnL are still
computed from each selected market's complete trade history, so pre-window
positions and exits are not truncated.

Typical usage::

    python -m experiments.timeframe_trade_window_train_backtest \
        --train-start 2025-02-01 --train-end 2025-02-10 \
        --test-start 2025-02-11 --test-end 2025-02-20 \
        --n-starts 8 --n-passes 1 --val-fraction 0.2 \
        --optimizer-mode coordinate_descent

Backtest-only usage::

    python -m experiments.timeframe_trade_window_train_backtest \
        --config-path experiments/results/.../timeframe_best_config.json \
        --test-start 2025-02-11 --test-end 2025-02-20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import (
    DEFAULT_CLUSTERING_CONFIG,
    evaluate_config,
    print_copytrade_summary,
)
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode
from experiments.timeframe_market_common import (
    _normalize_category_list,
    infer_resolutions,
    print_wallet_classification_summary,
    select_market_ids_in_timeframe,
)
from experiments.timeframe_experiment_common import (
    add_multi_start_args,
    add_standard_timeframe_optimizer_args,
    prepare_timeframe_inference,
    scoped_trade_time_filter,
    setup_timeframe_logging,
)
from experiments.timeframe_optimizers import run_timeframe_optimizer


DEFAULT_OUTPUT_DIR = "experiments/results/timeframe_trade_window_train_backtest"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize on markets resolved in a train window using only train-window "
            "trades, then backtest the selected config on markets resolved in a "
            "test window using only test-window trades."
        )
    )
    parser.add_argument("--train-start", "--train-start-date", dest="train_start_date", default=None)
    parser.add_argument("--train-end", "--train-end-date", dest="train_end_date", default=None)
    parser.add_argument("--test-start", "--test-start-date", dest="test_start_date", required=True)
    parser.add_argument("--test-end", "--test-end-date", dest="test_end_date", required=True)
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Backtest-only mode: evaluate this existing config on the test "
            "timeframe and skip optimization. When set, --train-start/--train-end "
            "are not required."
        ),
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        default=False,
        help="Skip optimization and only run the timeframe-restricted backtest. Requires --config-path.",
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=("coordinate_descent", "alternating_det_clust"),
        default="coordinate_descent",
        help=(
            "Optimization runner. coordinate_descent is the cheapest detector-only "
            "multi-start path; alternating_det_clust also alternates clustering."
        ),
    )
    parser.add_argument(
        "--insider-plausible-only",
        action="store_true",
        help="Filter train/test market sets to markets classified as insider-plausible.",
    )
    parser.add_argument(
        "--non-insider-plausible-only",
        action="store_true",
        help="Filter train/test market sets to markets classified as non-insider-plausible.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where optimization, backtest, and run metadata are written.",
    )
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--copytrade-fixed-size", type=float, default=100.0)
    parser.add_argument(
        "--verbose-output",
        action="store_true",
        default=False,
        help="Print per-market backtest logs during the final test evaluation.",
    )
    parser.add_argument(
        "--min-window-trades",
        type=int,
        default=1,
        help=(
            "Drop resolved markets with fewer than this many trades inside the "
            "train/test replay window. Default 1 avoids spending optimizer time "
            "on markets that have no in-window trades."
        ),
    )
    add_standard_timeframe_optimizer_args(parser)
    add_multi_start_args(parser)
    parser.set_defaults(
        enable_layer2_attribution=True,
        enable_jump_anticipation=True,
        enable_ja_optimization=True,
    )
    parser.add_argument(
        "--disable-layer2-attribution",
        dest="enable_layer2_attribution",
        action="store_false",
        help="Disable Layer 2 attribution. It is enabled by default for this experiment.",
    )
    parser.add_argument(
        "--disable-jump-anticipation",
        dest="enable_jump_anticipation",
        action="store_false",
        help="Disable jump anticipation. It is enabled by default for this experiment.",
    )
    parser.add_argument(
        "--disable-ja-optimization",
        dest="enable_ja_optimization",
        action="store_false",
        help="Disable jump-anticipation parameter optimization. It is enabled by default.",
    )
    return parser


def _filter_markets_by_window_trade_count(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    *,
    min_window_trades: int,
    min_usd_amount: float | None,
    label: str,
) -> Tuple[List[int], Dict[str, Any]]:
    kept: List[int] = []
    counts: List[int] = []
    zero_count = 0
    for market_id in market_ids:
        count = int(
            loader.get_trade_count(
                market_id=market_id,
                min_usd_amount=min_usd_amount,
            )
        )
        counts.append(count)
        if count == 0:
            zero_count += 1
        if count >= int(min_window_trades):
            kept.append(int(market_id))

    sorted_counts = sorted(counts)
    stats = {
        "label": label,
        "input_markets": len(market_ids),
        "kept_markets": len(kept),
        "dropped_markets": len(market_ids) - len(kept),
        "zero_trade_markets": zero_count,
        "min_window_trades": int(min_window_trades),
        "total_window_trades": int(sum(counts)),
        "median_window_trades": (
            float(sorted_counts[len(sorted_counts) // 2]) if sorted_counts else 0.0
        ),
        "max_window_trades": max(counts) if counts else 0,
    }
    logging.info(
        "%s window-trade filter: kept=%s/%s, dropped=%s, zero_trade=%s, "
        "total_window_trades=%s, median=%s, max=%s",
        label,
        f"{stats['kept_markets']:,}",
        f"{stats['input_markets']:,}",
        f"{stats['dropped_markets']:,}",
        f"{stats['zero_trade_markets']:,}",
        f"{stats['total_window_trades']:,}",
        f"{stats['median_window_trades']:,.0f}",
        f"{stats['max_window_trades']:,}",
    )
    return kept, stats


def _select_and_infer_test_markets(
    loader: HistoricalDataLoader,
    args: argparse.Namespace,
) -> Tuple[List[int], Dict[int, int], Dict[str, Any], List[int]]:
    candidate_market_ids = select_market_ids_in_timeframe(
        loader=loader,
        start_date=args.test_start_date,
        end_date=args.test_end_date,
        min_volume=args.min_market_volume,
        classifications_path=args.classifications_path,
        insider_plausible_only=args.insider_plausible_only,
        non_insider_plausible_only=args.non_insider_plausible_only,
        market_categories=args.market_categories,
        exclude_categories=args.exclude_categories,
    )
    logging.info("Test candidate markets in timeframe: %s", f"{len(candidate_market_ids):,}")

    winning_overrides, resolution_stats = infer_resolutions(
        loader=loader,
        market_ids=candidate_market_ids,
        resolution_threshold=args.resolution_threshold,
        min_trades=args.min_trades,
        min_usd_amount=args.min_usd_amount if args.enable_trade_prefilter else None,
        inferred_resolutions_db=args.inferred_resolutions_db,
    )
    market_ids = sorted(winning_overrides.keys())
    logging.info(
        "Test resolution inference: resolved=%s / %s, with_trades=%s, unresolved=%s",
        f"{resolution_stats['resolved']:,}",
        f"{resolution_stats['total_markets']:,}",
        f"{resolution_stats['with_trades']:,}",
        f"{resolution_stats['unresolved']:,}",
    )
    return market_ids, winning_overrides, resolution_stats, candidate_market_ids


def _evaluate_on_test_window(
    *,
    config: Dict[str, Any],
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_overrides: Dict[int, int],
    args: argparse.Namespace,
):
    eval_logger = logging.getLogger("backtesting.evaluation")
    previous_level = eval_logger.level
    if not args.verbose_output:
        eval_logger.setLevel(logging.WARNING)

    clustering_config = (
        config.get("clustering_config", DEFAULT_CLUSTERING_CONFIG)
        if args.enable_clustering
        else None
    )
    jump_anticipation_config = (
        config.get("jump_anticipation_config")
        if args.enable_jump_anticipation
        else None
    )

    try:
        return evaluate_config(
            config=config,
            loader=loader,
            market_ids=market_ids,
            prediction_mode=args.prediction_mode,
            flag_rate_threshold=args.flag_rate_threshold,
            suspicion_threshold=args.suspicion_threshold,
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
            min_usd_amount=args.min_usd_amount if args.enable_trade_prefilter else None,
            include_recidivism=args.include_recidivism,
            clustering_config=clustering_config,
            clustering_min_trade_size=args.clustering_min_trade_size,
            jump_anticipation_config=jump_anticipation_config,
            copytrade_fixed_size=args.copytrade_fixed_size,
            measure_memory=False,
            winning_outcomes_override=winning_overrides,
            enable_layer2_attribution=args.enable_layer2_attribution,
            usdc_cache_db=args.usdc_cache,
            polygonscan_api_key=args.polygonscan_api_key,
        )
    finally:
        eval_logger.setLevel(previous_level)


def _print_trade_level_reports(result, *, fixed_trade_size: float) -> None:
    pooled = result.event_study_pooled.get("pooled", {})
    if pooled:
        print("\nTRADE-LEVEL EVENT STUDY (POOLED)")
        print(f"  Markets in study: {pooled.get('n_markets', 0):,}")
        print(f"  Flagged trades: {pooled.get('total_flagged_trades', 0):,}")
        print(f"  Unflagged trades: {pooled.get('total_unflagged_trades', 0):,}")
        print(f"  Flagged mean return: {pooled.get('pooled_flagged_mean_return', 0):+.4f}")
        print(f"  Unflagged mean return: {pooled.get('pooled_unflagged_mean_return', 0):+.4f}")
        print(f"  Mean diff: {pooled.get('pooled_mean_return_diff', 0):+.4f}")
        print(f"  Mean Cohen's d: {pooled.get('mean_cohens_d', 0):.3f}")

    ct = result.copytrade_result
    if ct is not None:
        print("\nTRADE-LEVEL COPYTRADE SIMULATION (POOLED)")
        print(f"  Total trades copied: {ct.total_flagged_buys:,}")
        print(
            "  [Notional-matched]  "
            f"capital=${ct.total_capital_deployed:,.2f}  "
            f"P&L=${ct.total_pnl:+,.2f}  "
            f"ROI={ct.portfolio_roi:+.2%}  "
            f"win_rate={ct.win_rate:.2%}"
        )
        if ct.fixed_roi is not None:
            print(
                f"  [Fixed ${fixed_trade_size:.0f}/trade]    "
                f"capital=${ct.fixed_capital_deployed:,.2f}  "
                f"P&L=${ct.fixed_total_pnl:+,.2f}  "
                f"ROI={ct.fixed_roi:+.2%}  "
                f"median={ct.fixed_median_return:+.4f}"
            )


def main() -> None:
    args = _build_arg_parser().parse_args()
    backtest_only = bool(args.backtest_only or args.config_path)
    if backtest_only and not args.config_path:
        raise SystemExit("--backtest-only requires --config-path")
    if not backtest_only and (not args.train_start_date or not args.train_end_date):
        raise SystemExit(
            "--train-start and --train-end are required unless --config-path/--backtest-only is used"
        )
    if args.insider_plausible_only and args.non_insider_plausible_only:
        raise SystemExit(
            "--insider-plausible-only and --non-insider-plausible-only are mutually exclusive"
        )
    if args.n_starts < 1:
        raise SystemExit("--n-starts must be >= 1")
    args.market_categories = _normalize_category_list(args.market_categories)
    args.exclude_categories = _normalize_category_list(args.exclude_categories)

    # Existing optimizer metadata expects start_date/end_date on args.
    args.start_date = args.train_start_date if not backtest_only else args.test_start_date
    args.end_date = args.train_end_date if not backtest_only else args.test_end_date

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = setup_timeframe_logging(args.output_dir, "timeframe_trade_window_train_backtest")
    set_experiment_backtest_log_quiet_mode(enabled=not args.verbose_output)
    run_start = time.time()

    logging.info("Loading historical data...")
    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    try:
        train_prep = None
        train_trade_filter: Dict[str, Any] = {}
        train_window_stats: Dict[str, Any] = {}
        optimizer_out: Dict[str, Any] = {}

        if backtest_only:
            best_config_path = Path(args.config_path).expanduser().resolve()
            logging.info("Backtest-only mode: using existing config %s", best_config_path)
        else:
            logging.info(
                "Preparing train markets resolved in %s .. %s (resolution inference uses full market history)",
                args.train_start_date,
                args.train_end_date,
            )
            train_prep = prepare_timeframe_inference(
                loader,
                output_dir=args.output_dir,
                start_date=args.train_start_date,
                end_date=args.train_end_date,
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
                override_filename_prefix="trade_window_train_resolution_overrides",
            )
            if not train_prep.market_ids:
                raise RuntimeError("No inferred-resolved train markets in timeframe; aborting.")

            logging.info(
                "Filtering train markets by trades in replay window %s .. %s",
                args.train_start_date,
                args.train_end_date,
            )
            with scoped_trade_time_filter(
                loader,
                start_date=args.train_start_date,
                end_date=args.train_end_date,
            ) as train_trade_filter:
                train_market_ids, train_window_stats = _filter_markets_by_window_trade_count(
                    loader,
                    train_prep.market_ids,
                    min_window_trades=args.min_window_trades,
                    min_usd_amount=args.min_usd_amount if args.enable_trade_prefilter else None,
                    label="Train",
                )
                if not train_market_ids:
                    raise RuntimeError(
                        "No train markets have enough trades inside the train replay window; "
                        "try lowering --min-window-trades or using a wider train window."
                    )
                train_prep = replace(
                    train_prep,
                    market_ids=train_market_ids,
                    inferred_winners={
                        market_id: train_prep.inferred_winners[market_id]
                        for market_id in train_market_ids
                        if market_id in train_prep.inferred_winners
                    },
                )
                logging.info(
                    "Optimizing on %d train markets after window-trade filtering",
                    len(train_prep.market_ids),
                )
                optimizer_out = run_timeframe_optimizer(loader, train_prep, args)

            best_config_path = Path(optimizer_out["best_config_path"])

        with open(best_config_path, "r", encoding="utf-8") as f:
            best_config = json.load(f)

        logging.info(
            "Preparing test markets resolved in %s .. %s (resolution inference uses full market history)",
            args.test_start_date,
            args.test_end_date,
        )
        (
            test_market_ids,
            test_winners,
            test_resolution_stats,
            test_candidate_market_ids,
        ) = _select_and_infer_test_markets(loader, args)
        if not test_market_ids:
            raise RuntimeError("No inferred-resolved test markets in timeframe; aborting.")

        logging.info(
            "Filtering test markets by trades in replay window %s .. %s",
            args.test_start_date,
            args.test_end_date,
        )
        with scoped_trade_time_filter(
            loader,
            start_date=args.test_start_date,
            end_date=args.test_end_date,
        ) as test_trade_filter:
            test_market_ids, test_window_stats = _filter_markets_by_window_trade_count(
                loader,
                test_market_ids,
                min_window_trades=args.min_window_trades,
                min_usd_amount=args.min_usd_amount if args.enable_trade_prefilter else None,
                label="Test",
            )
            if not test_market_ids:
                raise RuntimeError(
                    "No test markets have enough trades inside the test replay window; "
                    "try lowering --min-window-trades or using a wider test window."
                )
            test_winners = {
                market_id: test_winners[market_id]
                for market_id in test_market_ids
                if market_id in test_winners
            }
            logging.info(
                "Backtesting best config on %d test markets after window-trade filtering",
                len(test_market_ids),
            )
            result = _evaluate_on_test_window(
                config=best_config,
                loader=loader,
                market_ids=test_market_ids,
                winning_overrides=test_winners,
                args=args,
            )

        tag_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = result.save(args.output_dir, tag=f"trade_window_test_{tag_ts}")

        meta = {
            "experiment": (
                "timeframe_trade_window_backtest_only"
                if backtest_only
                else "timeframe_trade_window_train_backtest"
            ),
            "backtest_only": backtest_only,
            "optimizer_mode": args.optimizer_mode,
            "best_config_path": str(best_config_path),
            "train_start_date": args.train_start_date,
            "train_end_date": args.train_end_date,
            "test_start_date": args.test_start_date,
            "test_end_date": args.test_end_date,
            "train_candidate_markets": len(train_prep.candidate_market_ids) if train_prep else None,
            "train_resolved_markets": len(train_prep.market_ids) if train_prep else None,
            "train_resolution_stats": train_prep.res_stats if train_prep else None,
            "test_candidate_markets": len(test_candidate_market_ids),
            "test_resolved_markets": len(test_market_ids),
            "test_resolution_stats": test_resolution_stats,
            "train_trade_filter": train_trade_filter,
            "test_trade_filter": test_trade_filter,
            "ground_truth_trade_history": "full_market_unfiltered",
            "train_window_trade_stats": train_window_stats,
            "test_window_trade_stats": test_window_stats,
            "n_starts": args.n_starts,
            "val_fraction": args.val_fraction,
            "objective": args.objective,
            "optimizer_output": {k: str(v) for k, v in optimizer_out.items() if k.endswith("_path")},
            "log_path": log_path,
        }
        meta_prefix = (
            "trade_window_backtest_only_meta"
            if backtest_only
            else "trade_window_train_backtest_meta"
        )
        meta_path = Path(args.output_dir) / f"{meta_prefix}_{tag_ts}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        agg = result.aggregate_performance
        ms_summary = optimizer_out.get("multi_start_summary", {})
        print("\n" + "=" * 80)
        print(
            "TRADE-WINDOW BACKTEST-ONLY COMPLETE"
            if backtest_only
            else "TRADE-WINDOW TRAIN -> TEST EXPERIMENT COMPLETE"
        )
        print("=" * 80)
        if train_prep is not None:
            print(f"Train markets resolved:      {len(train_prep.market_ids):,}")
        print(f"Test markets resolved:       {len(test_market_ids):,}")
        if train_trade_filter:
            print(f"Train trade window:          {train_trade_filter['start_time']} .. {train_trade_filter['end_time']}")
        print(f"Test trade window:           {test_trade_filter['start_time']} .. {test_trade_filter['end_time']}")
        print("Wallet ground truth scope:   full market history")
        if ms_summary:
            print(
                "Multi-start winner:          "
                f"start_idx={ms_summary.get('best_start_idx')} "
                f"selected_on={'VAL' if ms_summary.get('used_validation') else 'TRAIN'} "
                f"objective={float(ms_summary.get('best_final_objective', 0.0)):.4f}"
            )
        print(f"Test trades processed:       {agg.total_trades:,}")
        print(f"Test wall clock time:        {agg.total_wall_clock_seconds:.2f}s")
        print(f"Best config:                 {best_config_path}")

        print_wallet_classification_summary(result)
        print_copytrade_summary("TEST WINDOW COPYTRADE REPORT", result.copytrade_summary)
        _print_trade_level_reports(result, fixed_trade_size=args.copytrade_fixed_size)

        print("\nFiles:")
        for label, path in saved.items():
            print(f"  - {label}: {path}")
        print(f"  - run_meta: {meta_path}")
        print(f"  - log: {log_path}")
    except Exception as exc:
        logging.exception("Trade-window train/backtest experiment failed: %s", exc)
        loader.close()
        sys.exit(1)

    loader.close()
    logging.info("Total wall time: %.1fs", time.time() - run_start)


if __name__ == "__main__":
    main()
