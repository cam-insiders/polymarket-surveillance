"""
Optimize on markets resolving in a timeframe using only in-window trades, then
evaluate the saved config on multiple eval slices using only eval-window trades:
  - all markets in the eval window (no classification filter),
  - insider-plausible only,
  - each classifier category (ELECTION, SPORTS, …) one at a time.

Example:
  python -m experiments.timeframe_optimize_category_eval \\
    --optimizer-mode alternating_det_clust \\
    --output-dir experiments/results/category_eval

Bar charts from ``category_eval_summary.json`` (four 1×2 PNGs per train pool if ``--train-both-domains``):
  python -m experiments.charting.category_eval_charts --list
  python -m experiments.charting.category_eval_charts --run category_eval_<timestamp>
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode

from experiments.timeframe_market_common import _normalize_category_list
from experiments.common.timeframe import (
    add_standard_timeframe_optimizer_args,
    filter_markets_by_window_trade_count,
    prepare_timeframe_inference,
    run_timeframe_trade_window_backtest_evaluation,
    scoped_trade_time_filter,
    setup_timeframe_logging,
)
from experiments.timeframe_domain_matrix import summarize_evaluation
from experiments.timeframe_optimizers import run_timeframe_optimizer
from scripts.classify_markets import ALL_CATEGORIES

# Default windows (override with CLI). Eval defaults are a later window than train by default.
DEFAULT_TRAIN_START = "2025-01-01"
DEFAULT_TRAIN_END = "2025-01-10"
DEFAULT_TEST_START = "2025-03-01"
DEFAULT_TEST_END = "2025-03-28"


def _clone_ns(ns: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**copy.deepcopy(vars(ns)))


def _eval_slices() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Named eval slices: (name, kwargs for select_market_ids / backtest namespace fields).
    """
    slices: List[Tuple[str, Dict[str, Any]]] = [
        (
            "all",
            {
                "insider_plausible_only": False,
                "non_insider_plausible_only": False,
                "market_categories": None,
            },
        ),
        (
            "insider_plausible",
            {
                "insider_plausible_only": True,
                "non_insider_plausible_only": False,
                "market_categories": None,
            },
        ),
    ]
    for cat in sorted(ALL_CATEGORIES):
        slices.append(
            (
                f"category_{cat}",
                {
                    "insider_plausible_only": False,
                    "non_insider_plausible_only": False,
                    "market_categories": [cat],
                },
            )
        )
    return slices


def _train_run_specs(
    train_both_domains: bool,
    train_insider_plausible_only: bool,
) -> List[Tuple[str, bool, bool]]:
    """(train_domain label, insider_plausible_only, non_insider_plausible_only)."""
    full = ("full_domain", False, False)
    insider = ("insider_plausible_only", True, False)
    if train_both_domains:
        return [full, insider]
    if train_insider_plausible_only:
        return [insider]
    return [full]


def _run_category_eval_slices(
    loader: HistoricalDataLoader,
    args: argparse.Namespace,
    config: Dict[str, Any],
    best_path: Path,
    train_mode: str,
    run_root: Path,
    run_ts: str,
) -> List[Dict[str, Any]]:
    """Backtest ``config`` on every eval slice; returns rows (incl. errors)."""
    rows: List[Dict[str, Any]] = []
    for slice_name, flags in _eval_slices():
        bt = argparse.Namespace(
            config_path=str(best_path),
            start_date=args.test_start_date,
            end_date=args.test_end_date,
            resolution_threshold=args.resolution_threshold,
            min_market_volume=args.min_market_volume,
            min_trades=args.min_trades,
            inferred_resolutions_db=args.inferred_resolutions_db,
            prediction_mode=args.prediction_mode,
            flag_rate_threshold=args.flag_rate_threshold,
            suspicion_threshold=2.0,
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
            min_usd_amount=args.min_usd_amount,
            enable_trade_prefilter=args.enable_trade_prefilter,
            include_recidivism=args.include_recidivism,
            clustering_min_trade_size=args.clustering_min_trade_size,
            no_clustering=False,
            enable_layer2_attribution=args.enable_layer2_attribution,
            usdc_cache=args.usdc_cache,
            polygonscan_api_key=args.polygonscan_api_key,
            no_jump_anticipation=False,
            copytrade_fixed_size=100.0,
            data_dir=args.data_dir,
            insider_plausible_only=flags["insider_plausible_only"],
            non_insider_plausible_only=flags["non_insider_plausible_only"],
            market_categories=_normalize_category_list(flags.get("market_categories")),
            exclude_categories=args.exclude_categories,
            classifications_path=args.classifications_path,
            dry_run=False,
            verbose_output=False,
        )

        try:
            result, eval_meta = run_timeframe_trade_window_backtest_evaluation(
                config=config,
                loader=loader,
                args=bt,
                market_start=args.test_start_date,
                market_end=args.test_end_date,
                trade_start=args.test_start_date,
                trade_end=args.test_end_date,
                output_dir=str(run_root),
                min_window_trades=int(args.min_window_trades),
                trade_filter_label=f"Eval {slice_name}",
                quiet=True,
                override_filename_prefix=f"category_eval_{slice_name}_resolution_overrides",
            )
        except RuntimeError as exc:
            logging.warning(
                "Eval slice %s (train=%s) skipped: %s", slice_name, train_mode, exc
            )
            rows.append(
                {
                    "train_domain": train_mode,
                    "test_domain": slice_name,
                    "error": str(exc),
                }
            )
            continue

        summary_row = summarize_evaluation(result, train_mode, slice_name)
        summary_row["objective_metric"] = args.objective
        summary_row["candidate_markets"] = eval_meta["candidate_markets"]
        summary_row["resolved_markets"] = eval_meta["resolved_markets"]
        summary_row["resolved_markets_after_trade_filter"] = eval_meta[
            "resolved_markets_after_trade_filter"
        ]
        summary_row["trade_window_start"] = eval_meta["trade_start"]
        summary_row["trade_window_end"] = eval_meta["trade_end"]
        summary_row["config_path"] = str(best_path)
        rows.append(summary_row)

        if args.save_eval_artifacts:
            tag = f"eval_{train_mode}_{slice_name}_{run_ts}"
            saved = result.save(str(run_root), tag=tag)
            summary_row["saved_eval_paths"] = saved

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timeframe optimize (full-domain by default; optional insider pool) then per-slice category eval"
    )
    parser.add_argument("--train-start-date", type=str, default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end-date", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--test-start-date", type=str, default=DEFAULT_TEST_START)
    parser.add_argument("--test-end-date", type=str, default=DEFAULT_TEST_END)
    train_pool = parser.add_mutually_exclusive_group()
    train_pool.add_argument(
        "--train-both-domains",
        action="store_true",
        default=False,
        help="Run two optimisations: all train-window markets, then insider-plausible only; "
        "append eval rows for both (train_domain column distinguishes them).",
    )
    train_pool.add_argument(
        "--train-insider-plausible-only",
        action="store_true",
        default=False,
        help="Optimise only on insider-plausible train-window markets (single run).",
    )
    parser.add_argument(
        "--eval-only-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Load this JSON config and skip optimisation (category eval only).",
    )
    parser.add_argument(
        "--eval-train-domain-label",
        type=str,
        default="provided_config",
        help="train_domain value in metrics when using --eval-only-config (default: provided_config).",
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=("coordinate_descent", "alternating_det_clust"),
        default="alternating_det_clust",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/timeframe_optimize_category_eval",
    )
    parser.add_argument(
        "--save-eval-artifacts",
        action="store_true",
        default=False,
        help="Save full EvaluationResult artifacts for each eval slice.",
    )
    parser.add_argument(
        "--min-window-trades",
        type=int,
        default=1,
        help="Drop resolved train/eval markets with fewer than this many in-window trades.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="wallet_f1",
        choices=(
            "wallet_f1",
            "wallet_f0_5",
            "event_study_mean_diff",
            "event_study_mean_cohens_d",
            "copytrade_portfolio_roi",
            "copytrade_fixed_median_return",
        ),
        help="Metric to sort the printed category leaderboard by.",
    )
    add_standard_timeframe_optimizer_args(parser)
    args = parser.parse_args()

    args.market_categories = _normalize_category_list(args.market_categories)
    args.exclude_categories = _normalize_category_list(args.exclude_categories)

    if args.eval_only_config and (args.train_both_domains or args.train_insider_plausible_only):
        parser.error(
            "--eval-only-config cannot be combined with --train-both-domains or "
            "--train-insider-plausible-only"
        )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"category_eval_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = setup_timeframe_logging(str(run_root), "category_eval")
    set_experiment_backtest_log_quiet_mode(enabled=True)

    eval_only_path: Optional[Path] = None
    if args.eval_only_config:
        eval_only_path = Path(args.eval_only_config).expanduser().resolve()
        if not eval_only_path.is_file():
            logging.error("eval-only config not found: %s", eval_only_path)
            sys.exit(1)

    eval_only = eval_only_path is not None
    specs: List[Tuple[str, bool, bool]] = []
    if eval_only:
        schedule_label = args.eval_train_domain_label
        logging.info(
            "Run root: %s | eval_only_config=%s train_domain_label=%s",
            run_root,
            eval_only_path,
            schedule_label,
        )
    else:
        specs = _train_run_specs(args.train_both_domains, args.train_insider_plausible_only)
        schedule_label = "both" if len(specs) > 1 else specs[0][0]
        logging.info("Run root: %s | train_schedule=%s", run_root, schedule_label)

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    rows: List[Dict[str, Any]] = []
    best_paths: Dict[str, str] = {}

    if eval_only:
        train_mode = args.eval_train_domain_label
        best_paths[train_mode] = str(eval_only_path)
        with open(eval_only_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        rows.extend(
            _run_category_eval_slices(
                loader,
                args,
                config,
                eval_only_path,
                train_mode,
                run_root,
                run_ts,
            )
        )
    else:
        for train_mode, insider_only, non_insider_only in specs:
            targs = _clone_ns(args)
            targs.start_date = args.train_start_date
            targs.end_date = args.train_end_date
            targs.output_dir = str(run_root / f"train_{train_mode}")
            targs.insider_plausible_only = insider_only
            targs.non_insider_plausible_only = non_insider_only

            prep = prepare_timeframe_inference(
                loader,
                output_dir=targs.output_dir,
                start_date=targs.start_date,
                end_date=targs.end_date,
                min_market_volume=targs.min_market_volume,
                classifications_path=targs.classifications_path,
                insider_plausible_only=targs.insider_plausible_only,
                non_insider_plausible_only=targs.non_insider_plausible_only,
                market_categories=targs.market_categories,
                exclude_categories=targs.exclude_categories,
                resolution_threshold=targs.resolution_threshold,
                min_trades=targs.min_trades,
                inferred_resolutions_db=targs.inferred_resolutions_db,
                enable_trade_prefilter=targs.enable_trade_prefilter,
                min_usd_amount=targs.min_usd_amount,
            )
            if not prep.market_ids:
                logging.error(
                    "No resolved train markets for train_domain=%s; aborting.", train_mode
                )
                loader.close()
                sys.exit(1)

            trade_filter_min_usd = (
                float(targs.min_usd_amount) if targs.enable_trade_prefilter else None
            )
            with scoped_trade_time_filter(
                loader,
                start_date=args.train_start_date,
                end_date=args.train_end_date,
            ):
                train_market_ids, train_window_stats = filter_markets_by_window_trade_count(
                    loader,
                    prep.market_ids,
                    min_window_trades=int(args.min_window_trades),
                    min_usd_amount=trade_filter_min_usd,
                    label=f"Train {train_mode}",
                )
                if not train_market_ids:
                    logging.error(
                        "No train markets for train_domain=%s have enough in-window trades; aborting.",
                        train_mode,
                    )
                    loader.close()
                    sys.exit(1)
                prep = replace(
                    prep,
                    market_ids=train_market_ids,
                    inferred_winners={
                        int(mid): int(prep.inferred_winners[mid])
                        for mid in train_market_ids
                        if mid in prep.inferred_winners
                    },
                    res_stats={**prep.res_stats, "train_window_trade_stats": train_window_stats},
                )
                out = run_timeframe_optimizer(loader, prep, targs)

            best_path = Path(out["best_config_path"])
            best_paths[train_mode] = str(best_path)
            logging.info("Best config (%s): %s", train_mode, best_path)

            with open(best_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            rows.extend(
                _run_category_eval_slices(
                    loader,
                    args,
                    config,
                    best_path,
                    train_mode,
                    run_root,
                    run_ts,
                )
            )

    table_path = run_root / "category_eval_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(table_path, index=False)

    summary_path = run_root / "category_eval_summary.json"
    primary_best = best_paths.get("full_domain") or next(iter(best_paths.values()))
    meta = {
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
        "test_start_date": args.test_start_date,
        "test_end_date": args.test_end_date,
        "train_mode": schedule_label,
        "train_both_domains": bool(args.train_both_domains) and not eval_only,
        "train_insider_plausible_only": bool(args.train_insider_plausible_only)
        and not eval_only,
        "eval_only_config": str(eval_only_path) if eval_only_path else None,
        "skipped_optimization": eval_only,
        "best_config_path": primary_best,
        "best_config_paths": best_paths,
        "optimizer_mode": args.optimizer_mode,
        "objective_metric": args.objective,
        "prediction_mode": args.prediction_mode,
        "rank_by": args.rank_by,
        "rows": rows,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 80)
    print("TIMEFRAME OPTIMIZE + CATEGORY EVAL COMPLETE")
    print("=" * 80)
    if eval_only:
        print(
            f"  Train: skipped (eval-only config: {eval_only_path})  "
            f"[train_domain label: {args.eval_train_domain_label}]"
        )
    else:
        print(
            f"  Train: {args.train_start_date} .. {args.train_end_date} ({schedule_label})"
        )
    print(f"  Eval:  {args.test_start_date} .. {args.test_end_date}")
    print(f"  Objective metric: {args.objective}")
    print(f"  Metrics CSV: {table_path}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Log: {log_path}")

    if "test_domain" in df.columns:
        cat_mask = df["test_domain"].astype(str).str.startswith("category_")
        if eval_only:
            leaderboard_modes = [args.eval_train_domain_label]
        elif schedule_label == "both":
            leaderboard_modes = ["full_domain", "insider_plausible_only"]
        else:
            leaderboard_modes = [specs[0][0]]
        for tm in leaderboard_modes:
            cat_df = df[cat_mask & (df["train_domain"].astype(str) == tm)].copy()
            if cat_df.empty or args.rank_by not in cat_df.columns:
                continue
            cat_df = cat_df.sort_values(args.rank_by, ascending=False, na_position="last")
            label = f"train_domain={tm}" if schedule_label == "both" else "categories"
            print(f"\nCategory leaderboard ({label}, sorted by {args.rank_by}, top 15):")
            show_cols = [
                c
                for c in ["test_domain", args.rank_by, "resolved_markets", "wallet_f1"]
                if c in cat_df.columns
            ]
            print(cat_df[show_cols].head(15).to_string(index=False))

    loader.close()


if __name__ == "__main__":
    main()
