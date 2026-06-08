"""
Component ablation with trade-window replay.

Same leave-one-out ablations as ``ablation_common``, aligned with
``timeframe_trade_window_train_backtest``

Usage::

    python -m experiments.ablation_detectors_timeframe path/to/config.json \\
        --test-start 2025-02-15 --test-end 2025-02-28

    python -m experiments.ablation_detectors_timeframe \\
        experiments/results/timeframe_trade_window_train_backtest/timeframe_best_config_....json \\
        --test-start 2025-03-01 --test-end 2025-03-14 \\
        --enable-trade-prefilter --min-usd-amount 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import evaluate_config
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode
from backtesting.parameter_optimizer import _calculate_metrics_from_wallet_evaluations
from experiments import ablation_common as abl
from experiments.timeframe_market_common import _normalize_category_list
from experiments.timeframe_experiment_common import (
    filter_markets_by_window_trade_count,
    prepare_timeframe_inference,
    scoped_trade_time_filter,
    setup_timeframe_logging,
)

DEFAULT_OUTPUT_DIR = "experiments/results/ablation_detectors_timeframe"


def _effective_min_usd_amount(args: argparse.Namespace) -> Optional[float]:
    return args.min_usd_amount if args.enable_trade_prefilter else None


def _evaluate_variant(
    *,
    variant: abl.AblationVariant,
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_overrides: Dict[int, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    start = time.time()
    fixed_copytrade_size = 100.0
    if float(args.copytrade_fixed_size) != fixed_copytrade_size:
        logging.warning(
            "Ignoring --copytrade-fixed-size=%s for ablation comparison; "
            "using fixed $100 per trade for the fixed-size ROI metric.",
            args.copytrade_fixed_size,
        )

    result = evaluate_config(
        config=variant.config,
        loader=loader,
        market_ids=market_ids,
        prediction_mode=args.prediction_mode,
        flag_rate_threshold=args.flag_rate_threshold,
        suspicion_threshold=args.suspicion_threshold,
        z_score_threshold=args.z_score_threshold,
        min_wallet_notional=args.min_wallet_notional,
        min_usd_amount=_effective_min_usd_amount(args),
        include_recidivism=variant.include_recidivism,
        clustering_config=variant.clustering_config,
        clustering_min_trade_size=args.clustering_min_trade_size,
        jump_anticipation_config=variant.jump_anticipation_config,
        copytrade_fixed_size=fixed_copytrade_size,
        measure_memory=False,
        winning_outcomes_override=winning_overrides,
        enable_layer2_attribution=(
            bool(args.enable_layer2_attribution) and variant.clustering_config is not None
        ),
        usdc_cache_db=args.usdc_cache,
        polygonscan_api_key=args.polygonscan_api_key,
    )
    elapsed = time.time() - start

    wallet_metrics = _calculate_metrics_from_wallet_evaluations(
        result.wallet_evaluations,
        result.prediction_mode,
        result.suspicion_threshold,
        result.flag_rate_threshold,
    )
    cs = result.copytrade_summary
    ct = result.copytrade_result
    flagged_wallet = cs.get("flagged", {})
    row = {
        "variant": variant.name,
        "component_type": variant.component_type,
        "component_name": variant.component_name,
        "include_recidivism": bool(variant.include_recidivism),
        "clustering_enabled": bool(variant.clustering_config is not None),
        "jump_anticipation_enabled": bool(variant.jump_anticipation_config is not None),
        "wallet_f1": float(wallet_metrics.get("f1", 0.0)),
        "wallet_precision": float(wallet_metrics.get("precision", 0.0)),
        "wallet_recall": float(wallet_metrics.get("recall", 0.0)),
        "wallet_flagged_mean_net_pnl": float(flagged_wallet.get("avg_net_pnl", 0.0)),
        "wallet_flagged_mean_return": float(flagged_wallet.get("avg_return", 0.0)),
        "trade_copytrade_notional_portfolio_roi": (
            float(ct.portfolio_roi) if ct is not None else 0.0
        ),
        "trade_copytrade_fixed_100_roi": (
            float(ct.fixed_roi) if (ct is not None and ct.fixed_roi is not None) else 0.0
        ),
        "num_flagged_wallets": int(wallet_metrics.get("num_predicted_positive", 0)),
        "wall_clock_s": elapsed,
    }
    row.update(
        abl._flatten_detector_diagnostics(
            abl._build_detector_diagnostics(
                result,
                winning_overrides,
                loader=loader,
                market_ids=market_ids,
                min_usd_amount=_effective_min_usd_amount(args),
                jump_anticipation_config=variant.jump_anticipation_config,
            )
        )
    )
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Component ablation on a fixed config using test-window trade replay only."
        )
    )
    parser.add_argument("config_path", type=str)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--test-start",
        "--test-start-date",
        dest="test_start_date",
        required=True,
        help="Inclusive test window start (market close + trade replay).",
    )
    parser.add_argument(
        "--test-end",
        "--test-end-date",
        dest="test_end_date",
        required=True,
        help="Inclusive test window end.",
    )
    parser.add_argument(
        "--min-window-trades",
        type=int,
        default=1,
        help="Drop markets with fewer than this many trades inside the test replay window.",
    )
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-market-volume", type=float, default=0.0)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument(
        "--inferred-resolutions-db",
        type=str,
        default="inferred_resolutions.db",
    )
    parser.add_argument("--enable-trade-prefilter", action="store_true", default=False)
    parser.add_argument("--min-usd-amount", type=float, default=300.0)
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
    parser.add_argument("--market-categories", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-categories", type=str, nargs="+", default=None)
    parser.add_argument(
        "--classifications-path",
        type=str,
        default="data/market_classifications.json",
    )

    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--flag-rate-threshold", type=float, default=0.2)
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--clustering-min-trade-size", type=float, default=5000.0)
    parser.add_argument(
        "--copytrade-fixed-size",
        type=float,
        default=100.0,
        help="Kept for CLI compatibility; ablation comparison always uses fixed $100.",
    )

    parser.set_defaults(include_recidivism=False)
    parser.add_argument(
        "--include-recidivism",
        dest="include_recidivism",
        action="store_true",
        help="Include RecidivismDetector in the baseline/full-system run.",
    )
    parser.add_argument(
        "--exclude-recidivism",
        dest="include_recidivism",
        action="store_false",
        help="Disable RecidivismDetector in the baseline/full-system run.",
    )
    parser.add_argument(
        "--disable-clustering",
        action="store_true",
        default=False,
        help="Ignore clustering_config from the config and skip clustering ablation.",
    )
    parser.add_argument(
        "--disable-jump-anticipation",
        action="store_true",
        default=False,
        help="Ignore jump_anticipation_config from the config and skip JA ablation.",
    )
    parser.set_defaults(enable_layer2_attribution=True)
    parser.add_argument(
        "--disable-layer2-attribution",
        dest="enable_layer2_attribution",
        action="store_false",
        help="Disable Layer 2 attribution (enabled by default when clustering is on).",
    )
    parser.add_argument("--usdc-cache", type=str, default="data/usdc_transfers.db")
    parser.add_argument("--polygonscan-api-key", type=str, default=None)
    parser.add_argument("--verbose-output", action="store_true", default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.insider_plausible_only and args.non_insider_plausible_only:
        raise SystemExit(
            "--insider-plausible-only and --non-insider-plausible-only are mutually exclusive"
        )
    args.market_categories = _normalize_category_list(args.market_categories)
    args.exclude_categories = _normalize_category_list(args.exclude_categories)

    test_start = str(args.test_start_date)
    test_end = str(args.test_end_date)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = setup_timeframe_logging(str(args.output_dir), "ablation_detectors_timeframe")
    set_experiment_backtest_log_quiet_mode(enabled=not args.verbose_output)

    with open(args.config_path, encoding="utf-8") as f:
        base_config = json.load(f)

    base_clustering_config = (
        None if args.disable_clustering else deepcopy(base_config.get("clustering_config"))
    )
    base_ja_config = (
        None if args.disable_jump_anticipation else deepcopy(base_config.get("jump_anticipation_config"))
    )

    variants = abl._build_variants(
        base_config=base_config,
        include_recidivism=bool(args.include_recidivism),
        clustering_config=base_clustering_config,
        jump_anticipation_config=base_ja_config,
    )

    logging.info(
        "Prepared %d ablation variants (detectors=%d, clustering=%s, jump_anticipation=%s)",
        len(variants),
        sum(1 for v in variants if v.component_type == "detector"),
        bool(base_clustering_config is not None),
        bool(base_ja_config is not None),
    )

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    prep = prepare_timeframe_inference(
        loader,
        output_dir=args.output_dir,
        start_date=test_start,
        end_date=test_end,
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
        override_filename_prefix="ablation_test_resolution_overrides",
    )
    if not prep.market_ids:
        loader.close()
        raise RuntimeError(
            f"No inferred-resolved markets closed in {test_start} .. {test_end}; nothing to ablate."
        )

    trade_filter_min_usd = _effective_min_usd_amount(args)
    test_trade_filter: Dict[str, Any] = {}
    with scoped_trade_time_filter(
        loader,
        start_date=test_start,
        end_date=test_end,
    ) as test_trade_filter:
        market_ids, window_stats = filter_markets_by_window_trade_count(
            loader,
            prep.market_ids,
            min_window_trades=int(args.min_window_trades),
            min_usd_amount=trade_filter_min_usd,
            label="Test",
        )
        if not market_ids:
            loader.close()
            raise RuntimeError(
                "No markets have enough trades inside the test replay window; "
                "try lowering --min-window-trades or widening the test window."
            )
        winning_overrides = {
            int(mid): int(prep.inferred_winners[mid])
            for mid in market_ids
            if mid in prep.inferred_winners
        }

        logging.info(
            "Test trade replay %s .. %s: %d markets after window filter (from %d resolved)",
            test_start,
            test_end,
            len(market_ids),
            len(prep.market_ids),
        )

        rows: List[Dict[str, Any]] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for variant in variants:
            logging.info("\n%s\nABLATION: %s\n%s", "=" * 72, variant.name, "=" * 72)
            row = _evaluate_variant(
                variant=variant,
                loader=loader,
                market_ids=market_ids,
                winning_overrides=winning_overrides,
                args=args,
            )
            rows.append(row)
            logging.info(
                "  -> wallet_f1=%.4f wallet_mean_net_pnl=%+.2f wallet_mean_return=%+.4f "
                "roi_notional=%+.4f roi_fixed100=%+.4f elapsed=%.1fs",
                float(row["wallet_f1"]),
                float(row["wallet_flagged_mean_net_pnl"]),
                float(row["wallet_flagged_mean_return"]),
                float(row["trade_copytrade_notional_portfolio_roi"]),
                float(row["trade_copytrade_fixed_100_roi"]),
                float(row["wall_clock_s"]),
            )

    df = pd.DataFrame(rows)
    df = abl._add_change_after_removing_columns(df)

    csv_path = f"{args.output_dir}/component_ablation_timeframe_{timestamp}.csv"
    meta_path = f"{args.output_dir}/component_ablation_timeframe_{timestamp}_meta.json"
    df.to_csv(csv_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "ablation_detectors_timeframe",
                "config_path": str(Path(args.config_path).resolve()),
                "test_start_date": test_start,
                "test_end_date": test_end,
                "test_trade_filter": test_trade_filter,
                "test_window_trade_stats": window_stats,
                "candidate_markets": len(prep.candidate_market_ids),
                "resolved_markets": len(prep.market_ids),
                "markets_evaluated": len(market_ids),
                "resolution_stats": prep.res_stats,
                "resolution_override_path": str(prep.override_path),
                "prediction_mode": args.prediction_mode,
                "flag_rate_threshold": args.flag_rate_threshold,
                "include_recidivism": bool(args.include_recidivism),
                "clustering_enabled": bool(base_clustering_config is not None),
                "jump_anticipation_enabled": bool(base_ja_config is not None),
                "enable_layer2_attribution": bool(args.enable_layer2_attribution),
                "enable_trade_prefilter": bool(args.enable_trade_prefilter),
                "min_usd_amount": args.min_usd_amount,
                "min_window_trades": int(args.min_window_trades),
                "comparison_metrics": abl.COMPARISON_METRICS,
                "classification_filters": {
                    "insider_plausible_only": bool(args.insider_plausible_only),
                    "non_insider_plausible_only": bool(args.non_insider_plausible_only),
                    "market_categories": args.market_categories,
                    "exclude_categories": args.exclude_categories,
                    "classifications_path": args.classifications_path,
                },
                "variants": [v.name for v in variants],
                "log_path": log_path,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n{'=' * 88}")
    print("COMPONENT ABLATION (TRADE-WINDOW REPLAY)")
    print(f"{'=' * 88}")
    print(f"Test window:  {test_start} .. {test_end}")
    print(
        f"Markets:      {len(market_ids):,} evaluated / "
        f"{len(prep.market_ids):,} resolved / "
        f"{len(prep.candidate_market_ids):,} candidates"
    )
    abl._print_compact_terminal_summary(df)
    abl._print_full_system_detector_diagnostics(df)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {meta_path}")
    print(f"Log:   {log_path}")

    loader.close()


if __name__ == "__main__":
    main()
