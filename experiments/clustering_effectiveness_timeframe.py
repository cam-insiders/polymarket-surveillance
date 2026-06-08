"""
Clustering effectiveness experiment with trade-window replay.

Same analysis as ``clustering_effectiveness_common``, but aligned with
``timeframe_trade_window_train_backtest``

Typical usage::

    python -m experiments.clustering_effectiveness_timeframe \\
        --train-start 2025-02-01 --train-end 2025-02-28 \\
        --test-start 2025-03-01 --test-end 2025-03-14 \\
        --enable-trade-prefilter --min-usd-amount 500

    python -m experiments.clustering_effectiveness_timeframe \\
        --test-start 2025-03-01 --test-end 2025-03-14 \\
        --config-path experiments/results/.../timeframe_best_config.json
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from backtesting.data_loader import HistoricalDataLoader
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode

from experiments import clustering_effectiveness_common as ce
from experiments.timeframe_market_common import _normalize_category_list
from experiments.timeframe_experiment_common import (
    add_multi_start_args,
    add_standard_timeframe_optimizer_args,
    filter_markets_by_window_trade_count,
    prepare_timeframe_inference,
    run_timeframe_trade_window_backtest_evaluation,
    scoped_trade_time_filter,
    setup_timeframe_logging,
)
from experiments.timeframe_optimizers import run_multi_start_alternating_timeframe


DEFAULT_OUTPUT_DIR = "experiments/results/clustering_effectiveness_timeframe"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clustering effectiveness with train/test trade-window replay: "
            "optimize on train-window trades (optional), analyze clustering "
            "impact on test-window trades only."
        ),
    )
    parser.add_argument(
        "--train-start",
        "--train-start-date",
        dest="train_start_date",
        default=None,
        help="Inclusive train window start (market close + train trade replay).",
    )
    parser.add_argument(
        "--train-end",
        "--train-end-date",
        dest="train_end_date",
        default=None,
        help="Inclusive train window end.",
    )
    parser.add_argument(
        "--test-start",
        "--test-start-date",
        dest="test_start_date",
        required=True,
        help="Inclusive test window start (market close + eval trade replay).",
    )
    parser.add_argument(
        "--test-end",
        "--test-end-date",
        dest="test_end_date",
        required=True,
        help="Inclusive test window end.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Pre-optimized config JSON; skips train-window optimization.",
    )
    parser.add_argument(
        "--min-window-trades",
        type=int,
        default=1,
        help="Drop markets with fewer than this many trades inside each replay window.",
    )
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
        help="Fixed trade size for copytrade simulation during backtest.",
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
        default=ce.DEFAULT_BOOST_BUCKETS,
        help="Comma-separated bucket edges for boost-magnitude histogram.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
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
        help="Disable Layer 2 attribution (enabled by default).",
    )
    parser.add_argument(
        "--disable-jump-anticipation",
        dest="enable_jump_anticipation",
        action="store_false",
        help="Disable jump anticipation (enabled by default).",
    )
    parser.add_argument(
        "--disable-ja-optimization",
        dest="enable_ja_optimization",
        action="store_false",
        help="Disable jump-anticipation parameter optimization (enabled by default).",
    )
    return parser


def _resolve_config(
    loader: HistoricalDataLoader,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if args.config_path:
        cfg = ce._load_config(args.config_path)
        return cfg, {
            "source": "provided",
            "config_path": str(Path(args.config_path).resolve()),
        }

    if not args.train_start_date or not args.train_end_date:
        raise SystemExit(
            "--train-start and --train-end are required when --config-path is not set."
        )

    train_start = str(args.train_start_date)
    train_end = str(args.train_end_date)
    logging.info(
        "No --config-path; optimizing on train-window trades %s .. %s (n_starts=%d).",
        train_start,
        train_end,
        int(args.n_starts or 1),
    )

    opt_out_dir = output_dir / "optimization"
    opt_out_dir.mkdir(parents=True, exist_ok=True)
    opt_args = copy.copy(args)
    opt_args.output_dir = str(opt_out_dir)
    opt_args.start_date = train_start
    opt_args.end_date = train_end

    train_prep = prepare_timeframe_inference(
        loader,
        output_dir=str(opt_out_dir),
        start_date=train_start,
        end_date=train_end,
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
        override_filename_prefix="clustering_effectiveness_train_resolution_overrides",
    )
    if not train_prep.market_ids:
        raise SystemExit("No inferred-resolved train markets in timeframe; cannot optimize.")

    trade_filter_min_usd = (
        float(opt_args.min_usd_amount) if opt_args.enable_trade_prefilter else None
    )
    train_trade_filter: Dict[str, Any] = {}
    with scoped_trade_time_filter(
        loader,
        start_date=train_start,
        end_date=train_end,
    ) as train_trade_filter:
        kept, train_window_stats = filter_markets_by_window_trade_count(
            loader,
            train_prep.market_ids,
            min_window_trades=int(args.min_window_trades),
            min_usd_amount=trade_filter_min_usd,
            label="Train",
        )
        if not kept:
            raise SystemExit(
                "No train markets have enough trades inside the train replay window."
            )
        train_prep = replace(
            train_prep,
            market_ids=kept,
            inferred_winners={
                int(mid): int(train_prep.inferred_winners[mid])
                for mid in kept
                if mid in train_prep.inferred_winners
            },
        )
        optimizer_out = run_multi_start_alternating_timeframe(loader, train_prep, opt_args)

    best_config_path = str(optimizer_out["best_config_path"])
    cfg = ce._load_config(best_config_path)
    return cfg, {
        "source": "optimized",
        "train_start_date": train_start,
        "train_end_date": train_end,
        "train_trade_filter": train_trade_filter,
        "train_window_trade_stats": train_window_stats,
        "best_config_path": best_config_path,
        "multi_start_summary": optimizer_out.get("multi_start_summary", {}),
        "optimizer_artifacts": {
            k: str(v) for k, v in optimizer_out.items() if isinstance(v, (str, Path))
        },
    }


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.insider_plausible_only and args.non_insider_plausible_only:
        raise SystemExit(
            "--insider-plausible-only and --non-insider-plausible-only are mutually exclusive"
        )
    if int(args.n_starts or 1) < 1:
        raise SystemExit("--n-starts must be >= 1")

    args.market_categories = _normalize_category_list(args.market_categories)
    args.exclude_categories = _normalize_category_list(args.exclude_categories)

    boost_edges = ce._parse_boost_buckets(args.boost_buckets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_timeframe_logging(
        str(output_dir),
        "clustering_effectiveness_timeframe",
    )
    set_experiment_backtest_log_quiet_mode(enabled=not args.verbose_output)

    test_start = str(args.test_start_date)
    test_end = str(args.test_end_date)

    logging.info("Loading historical data...")
    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    run_start = time.time()

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

    bt_args = copy.copy(args)
    bt_args.no_clustering = not bool(getattr(args, "enable_clustering", True))
    bt_args.no_jump_anticipation = not bool(getattr(args, "enable_jump_anticipation", False))
    bt_args.start_date = test_start
    bt_args.end_date = test_end
    bt_args.dry_run = False
    bt_args.verbose_output = args.verbose_output

    eval_kw = dict(
        config=config,
        loader=loader,
        market_start=test_start,
        market_end=test_end,
        trade_start=test_start,
        trade_end=test_end,
        output_dir=str(output_dir),
        min_window_trades=int(args.min_window_trades),
        quiet=not args.verbose_output,
    )

    logging.info(
        "Running with-clustering backtest on test-window trades %s .. %s ...",
        test_start,
        test_end,
    )
    try:
        eval_result, eval_meta = run_timeframe_trade_window_backtest_evaluation(
            **eval_kw,
            args=bt_args,
            trade_filter_label="Test",
            override_filename_prefix="clustering_effectiveness_test_resolution_overrides",
        )
    except RuntimeError as exc:
        logging.error("Backtest failed: %s", exc)
        loader.close()
        sys.exit(1)

    bt_args_no_cluster = copy.copy(bt_args)
    bt_args_no_cluster.no_clustering = True
    logging.info("Running counterfactual backtest (clustering off) on test window ...")
    try:
        no_cluster_result, no_cluster_meta = run_timeframe_trade_window_backtest_evaluation(
            **eval_kw,
            args=bt_args_no_cluster,
            trade_filter_label="Test (no clustering)",
            override_filename_prefix="clustering_effectiveness_test_no_cluster_overrides",
        )
    except RuntimeError as exc:
        logging.error("Counterfactual backtest failed: %s", exc)
        loader.close()
        sys.exit(1)

    clustering_diag_result = eval_result
    clustering_diag_meta = eval_meta
    used_cluster_only_diagnostics = False
    if not bt_args.no_jump_anticipation:
        bt_args_cluster_diag = copy.copy(bt_args)
        bt_args_cluster_diag.no_jump_anticipation = True
        logging.info("Running clustering-only diagnostic replay (JA off) on test window ...")
        try:
            clustering_diag_result, clustering_diag_meta = (
                run_timeframe_trade_window_backtest_evaluation(
                    **eval_kw,
                    args=bt_args_cluster_diag,
                    trade_filter_label="Test (cluster diag)",
                    override_filename_prefix="clustering_effectiveness_test_cluster_diag_overrides",
                )
            )
            used_cluster_only_diagnostics = True
        except RuntimeError as exc:
            logging.warning(
                "Clustering-only diagnostic replay failed (%s); using combined replay.",
                exc,
            )

    annotated = ce._annotate_wallet_rows(
        eval_result.wallet_evaluations,
        no_cluster_result.wallet_evaluations,
        clustering_diag_result.wallet_evaluations,
        flag_rate_threshold=args.flag_rate_threshold,
        prediction_mode=args.prediction_mode,
        suspicion_threshold=args.suspicion_threshold,
        boost_edges=boost_edges,
    )
    report_obj = ce._build_report(
        annotated=annotated,
        with_clustering_wallet_evaluations=eval_result.wallet_evaluations,
        without_clustering_wallet_evaluations=no_cluster_result.wallet_evaluations,
        boost_edges=boost_edges,
        prediction_mode=args.prediction_mode,
        suspicion_threshold=args.suspicion_threshold,
        flag_rate_threshold=args.flag_rate_threshold,
        layer2_enabled=bool(args.enable_layer2_attribution),
    )
    trade_filter_min_usd = (
        float(args.min_usd_amount)
        if bool(getattr(args, "enable_trade_prefilter", False))
        and args.min_usd_amount is not None
        else None
    )
    with scoped_trade_time_filter(
        loader,
        start_date=test_start,
        end_date=test_end,
    ):
        trade_buy_alert_report, trade_cohort_rows = (
            ce._build_trade_buy_alert_counterfactual_report(
                loader=loader,
                market_ids=eval_result.market_ids,
                with_backtest_results=eval_result.backtest_results,
                without_backtest_results=no_cluster_result.backtest_results,
                min_usd_amount=trade_filter_min_usd,
                resolution_threshold=float(args.resolution_threshold),
                winning_outcomes_override=eval_meta.get("winning_outcomes", {}),
                clustering_config=(
                    config.get("clustering_config")
                    if not getattr(bt_args, "no_clustering", False)
                    else None
                ),
                clustering_min_trade_size=float(args.clustering_min_trade_size),
            )
        )
        trade_alert_boost_report, trade_alert_boost_rows = (
            ce._build_trade_alert_boost_wallet_eval_report(
                loader=loader,
                market_ids=eval_result.market_ids,
                with_backtest_results=eval_result.backtest_results,
                without_backtest_results=no_cluster_result.backtest_results,
                with_clustering_wallet_evaluations=eval_result.wallet_evaluations,
                min_usd_amount=trade_filter_min_usd,
                clustering_config=(
                    config.get("clustering_config")
                    if not getattr(bt_args, "no_clustering", False)
                    else None
                ),
                clustering_min_trade_size=float(args.clustering_min_trade_size),
            )
        )
    report_obj.report["trade_buy_alert_counterfactual"] = trade_buy_alert_report
    report_obj.trade_cohort_rows = trade_cohort_rows
    report_obj.report["trade_alert_boost_wallet_eval"] = trade_alert_boost_report
    report_obj.trade_alert_boost_wallet_rows = trade_alert_boost_rows

    with scoped_trade_time_filter(
        loader,
        start_date=test_start,
        end_date=test_end,
    ):
        cluster_return_report, cluster_return_rows = ce._build_cluster_return_report(
            loader=loader,
            market_ids=eval_result.market_ids,
            wallet_evaluations=eval_result.wallet_evaluations,
            clustering_config=config.get("clustering_config"),
            clustering_min_trade_size=float(args.clustering_min_trade_size),
            min_usd_amount=trade_filter_min_usd,
        )
    report_obj.report["anonymous_cluster_returns"] = cluster_return_report
    report_obj.cluster_return_rows = cluster_return_rows

    total_elapsed = time.time() - run_start
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_metadata = {
        "experiment": "clustering_effectiveness_timeframe",
        "timestamp": ts,
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
        "test_start_date": test_start,
        "test_end_date": test_end,
        "config_source": config_source,
        "prediction_mode": args.prediction_mode,
        "flag_rate_threshold": args.flag_rate_threshold,
        "suspicion_threshold": args.suspicion_threshold,
        "enable_layer2_attribution": bool(args.enable_layer2_attribution),
        "enable_clustering": bool(getattr(args, "enable_clustering", True)),
        "enable_jump_anticipation": bool(getattr(args, "enable_jump_anticipation", False)),
        "min_window_trades": int(args.min_window_trades),
        "clustering_min_trade_size": args.clustering_min_trade_size,
        "min_usd_amount": args.min_usd_amount,
        "min_wallet_notional": args.min_wallet_notional,
        "include_recidivism": bool(args.include_recidivism),
        "boost_buckets": boost_edges,
        "market_counts": {
            "with_clustering_candidate": eval_meta.get("candidate_markets", 0),
            "with_clustering_resolved": eval_meta.get("resolved_markets", 0),
            "with_clustering_after_trade_filter": eval_meta.get(
                "resolved_markets_after_trade_filter", 0
            ),
            "without_clustering_after_trade_filter": no_cluster_meta.get(
                "resolved_markets_after_trade_filter", 0
            ),
            "clustering_diagnostic_after_trade_filter": clustering_diag_meta.get(
                "resolved_markets_after_trade_filter", 0
            ),
        },
        "trade_filters": {
            "with_clustering": eval_meta.get("trade_filter"),
            "without_clustering": no_cluster_meta.get("trade_filter"),
            "clustering_diagnostic": clustering_diag_meta.get("trade_filter"),
        },
        "window_trade_stats": {
            "with_clustering": eval_meta.get("window_trade_stats"),
            "without_clustering": no_cluster_meta.get("window_trade_stats"),
            "clustering_diagnostic": clustering_diag_meta.get("window_trade_stats"),
            "train": config_source.get("train_window_trade_stats"),
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
        "anonymous_cluster_returns_meta": (
            report_obj.report.get("anonymous_cluster_returns", {}).get("summary", {})
        ),
        "total_elapsed_seconds": total_elapsed,
        "config": config,
    }

    saved = ce._save_artifacts(output_dir, ts, report_obj, run_metadata)

    print("\n" + "=" * 90)
    print("CLUSTERING EFFECTIVENESS (TRADE-WINDOW) COMPLETE")
    print("=" * 90)
    print(f"Config source:          {config_source.get('source')}")
    if config_source.get("best_config_path"):
        print(f"Optimizer best config:  {config_source['best_config_path']}")
    if config_source.get("config_path"):
        print(f"Provided config:        {config_source['config_path']}")
    if args.train_start_date:
        print(f"Train trade window:     {args.train_start_date} .. {args.train_end_date}")
    print(f"Test trade window:      {test_start} .. {test_end}")
    print(
        "Test markets:           "
        f"resolved={eval_meta.get('resolved_markets', 0):,} "
        f"after_filter={eval_meta.get('resolved_markets_after_trade_filter', 0):,}"
    )
    print(
        "Wallets evaluated:      "
        f"with={len(eval_result.wallet_evaluations):,} "
        f"without={len(no_cluster_result.wallet_evaluations):,}"
    )
    counts = report_obj.report.get("counts", {})
    print(
        "Wallet flips:           "
        f"decisive={int(counts.get('decisive_flips', 0)):,} "
        f"suppressed={int(counts.get('suppressed_flips', 0)):,}"
    )
    print(f"Layer 2 attribution:    {'enabled' if args.enable_layer2_attribution else 'disabled'}")
    print(f"Jump anticipation:      {'enabled' if args.enable_jump_anticipation else 'disabled'}")
    cluster_summary = report_obj.report.get("anonymous_cluster_returns", {}).get("summary", {})
    print(
        "Anonymous clusters:     "
        f"{int(cluster_summary.get('cluster_count', 0)):,} "
        f"(with_gt={int(cluster_summary.get('clusters_with_ground_truth', 0)):,}, "
        f"weighted_return={float(cluster_summary.get('weighted_return', 0.0)):+.2%})"
    )
    print(f"Total wall time:        {total_elapsed:.1f}s")
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
    print("\nFiles:")
    for label, path in saved.items():
        print(f"  - {label}: {path}")
    print(f"  - log: {log_path}")

    loader.close()


if __name__ == "__main__":
    main()
