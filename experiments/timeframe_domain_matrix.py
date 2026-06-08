"""
3×3 domain matrix: train an optimizer on {all, insider-plausible, non-insider-plausible}
markets, then backtest each saved config on each test domain.

Example:
  python -m experiments.timeframe_domain_matrix \\
    --train-start-date 2025-02-01 --train-end-date 2025-02-15 \\
    --test-start-date 2025-03-01 --test-end-date 2025-03-15 \\
    --optimizer-mode alternating_det_clust \\
    --output-dir experiments/results/domain_matrix_run

Train only (writes train_manifest.json with paths + objective_metric; skips 9 backtests):
  python -m experiments.timeframe_domain_matrix --train-only ...
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import EvaluationResult
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode

from experiments.timeframe_market_common import (
    _normalize_category_list,
    run_timeframe_backtest_evaluation,
)
from experiments.timeframe_experiment_common import (
    add_standard_timeframe_optimizer_args,
    prepare_timeframe_inference,
    setup_timeframe_logging,
)
from experiments.timeframe_optimizers import run_timeframe_optimizer

TRAIN_TEST_DOMAINS: Tuple[Tuple[str, Dict[str, bool]], ...] = (
    ("all", {"insider_plausible_only": False, "non_insider_plausible_only": False}),
    ("insider", {"insider_plausible_only": True, "non_insider_plausible_only": False}),
    ("non_insider", {"insider_plausible_only": False, "non_insider_plausible_only": True}),
)


def _clone_ns(ns: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**copy.deepcopy(vars(ns)))


def _wallet_prf_from_copytrade_summary(summary: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Precision, recall, F1, F0.5 from wallet-level TP/FP/FN counts."""
    tp = int(summary.get("tp", {}).get("count", 0) or 0)
    fp = int(summary.get("fp", {}).get("count", 0) or 0)
    fn = int(summary.get("fn", {}).get("count", 0) or 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    b2 = 0.25
    f05 = ((1.0 + b2) * prec * rec / (b2 * prec + rec)) if (b2 * prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1), float(f05)


def summarize_evaluation(
    result: EvaluationResult,
    train_domain: str,
    test_domain: str,
) -> Dict[str, Any]:
    prec, rec, f1, f05 = _wallet_prf_from_copytrade_summary(result.copytrade_summary)
    pooled = result.event_study_pooled.get("pooled", {}) or {}
    ct = result.copytrade_result
    flagged = result.copytrade_summary.get("flagged", {}) or {}

    row: Dict[str, Any] = {
        "train_domain": train_domain,
        "test_domain": test_domain,
        "n_markets_eval": result.aggregate_performance.total_markets,
        "n_trades": result.aggregate_performance.total_trades,
        "wallet_precision": prec,
        "wallet_recall": rec,
        "wallet_f1": f1,
        "wallet_f0_5": f05,
        "flagged_wallet_count": int(flagged.get("count", 0) or 0),
        "flagged_mean_return": float(flagged.get("avg_return", 0.0) or 0.0),
        "flagged_median_return": float(flagged.get("median_return", 0.0) or 0.0),
        "flagged_avg_net_pnl": float(flagged.get("avg_net_pnl", 0.0) or 0.0),
        "flagged_mean_net_pnl": float(flagged.get("avg_net_pnl", 0.0) or 0.0),
        "flagged_median_net_pnl": float(flagged.get("median_net_pnl", 0.0) or 0.0),
        "event_study_mean_diff": float(pooled.get("pooled_mean_return_diff", 0.0) or 0.0),
        "event_study_mean_cohens_d": float(pooled.get("mean_cohens_d", 0.0) or 0.0),
    }
    if ct is not None:
        row["copytrade_total_pnl"] = float(ct.total_pnl)
        row["copytrade_portfolio_roi"] = float(ct.portfolio_roi)
        row["copytrade_median_trade_return"] = float(ct.median_trade_return)
        row["copytrade_win_rate"] = float(ct.win_rate)
        if ct.fixed_median_return is not None:
            row["copytrade_fixed_median_return"] = float(ct.fixed_median_return)
    return row


def _add_backtest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bt-prediction-mode", type=str, default=None, help="Override; default = train prediction-mode")
    p.add_argument("--bt-flag-rate-threshold", type=float, default=None)
    p.add_argument("--bt-suspicion-threshold", type=float, default=2.0)
    p.add_argument("--bt-z-score-threshold", type=float, default=None)
    p.add_argument("--bt-min-wallet-notional", type=float, default=None)
    p.add_argument("--bt-min-usd-amount", type=float, default=None)
    p.add_argument("--bt-clustering-min-trade-size", type=float, default=None)
    p.add_argument("--bt-no-clustering", action="store_true", default=False)
    p.add_argument("--bt-enable-layer2-attribution", action="store_true", default=False)
    p.add_argument("--bt-no-jump-anticipation", action="store_true", default=False)
    p.add_argument("--bt-copytrade-fixed-size", type=float, default=100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="3×3 train/test domain matrix for timeframe experiments")
    parser.add_argument("--train-start-date", type=str, required=True)
    parser.add_argument("--train-end-date", type=str, required=True)
    parser.add_argument("--test-start-date", type=str, default=None)
    parser.add_argument("--test-end-date", type=str, default=None)
    parser.add_argument(
        "--optimizer-mode",
        choices=("coordinate_descent", "alternating_det_clust"),
        default="alternating_det_clust",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/timeframe_domain_matrix",
    )
    parser.add_argument(
        "--save-eval-artifacts",
        action="store_true",
        default=False,
        help="Call EvaluationResult.save() for each backtest (nine JSON/CSV sets).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        default=False,
        help="Run the three training jobs only; write train_manifest.json and skip all backtests.",
    )
    add_standard_timeframe_optimizer_args(parser)
    _add_backtest_args(parser)
    args = parser.parse_args()

    test_start = args.test_start_date or args.train_start_date
    test_end = args.test_end_date or args.train_end_date

    args.market_categories = _normalize_category_list(args.market_categories)
    args.exclude_categories = _normalize_category_list(args.exclude_categories)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"domain_matrix_{run_ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = setup_timeframe_logging(str(run_root), "domain_matrix")
    set_experiment_backtest_log_quiet_mode(enabled=True)

    logging.info("Run root: %s", run_root)

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    train_config_paths: Dict[str, Path] = {}

    for domain_name, flags in TRAIN_TEST_DOMAINS:
        targs = _clone_ns(args)
        targs.start_date = args.train_start_date
        targs.end_date = args.train_end_date
        targs.insider_plausible_only = flags["insider_plausible_only"]
        targs.non_insider_plausible_only = flags["non_insider_plausible_only"]
        targs.output_dir = str(run_root / f"train_{domain_name}")

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
            logging.error("Train domain %s: no resolved markets; aborting.", domain_name)
            loader.close()
            sys.exit(1)

        out = run_timeframe_optimizer(loader, prep, targs)
        best_path = out["best_config_path"]

        train_config_paths[domain_name] = Path(best_path)
        logging.info("Train domain %s -> %s", domain_name, best_path)

    manifest_path = run_root / "train_manifest.json"
    manifest = {
        "created_at": datetime.now().isoformat(),
        "train_only": bool(args.train_only),
        "objective_metric": args.objective,
        "prediction_mode": args.prediction_mode,
        "optimizer_mode": args.optimizer_mode,
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
        "train_runs": [
            {
                "domain": name,
                "best_config_path": str(path),
                "objective_metric": args.objective,
            }
            for name, path in train_config_paths.items()
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Wrote train manifest: %s", manifest_path)

    if args.train_only:
        print("\n" + "=" * 80)
        print("DOMAIN MATRIX TRAIN-ONLY COMPLETE")
        print("=" * 80)
        print(f"  Train manifest: {manifest_path}")
        print(f"  Log: {log_path}")
        print(f"  Objective metric: {args.objective}")
        loader.close()
        return

    if len(train_config_paths) != len(TRAIN_TEST_DOMAINS):
        logging.error(
            "Expected %s train configs, got %s. Aborting backtest phase.",
            len(TRAIN_TEST_DOMAINS),
            len(train_config_paths),
        )
        loader.close()
        sys.exit(1)

    rows: List[Dict[str, Any]] = []

    for train_name, config_path in train_config_paths.items():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for test_name, test_flags in TRAIN_TEST_DOMAINS:
            bt = argparse.Namespace(
                config_path=str(config_path),
                start_date=test_start,
                end_date=test_end,
                resolution_threshold=args.resolution_threshold,
                min_market_volume=args.min_market_volume,
                min_trades=args.min_trades,
                inferred_resolutions_db=args.inferred_resolutions_db,
                prediction_mode=args.bt_prediction_mode or args.prediction_mode,
                flag_rate_threshold=(
                    args.bt_flag_rate_threshold
                    if args.bt_flag_rate_threshold is not None
                    else args.flag_rate_threshold
                ),
                suspicion_threshold=args.bt_suspicion_threshold,
                z_score_threshold=(
                    args.bt_z_score_threshold if args.bt_z_score_threshold is not None else args.z_score_threshold
                ),
                min_wallet_notional=(
                    args.bt_min_wallet_notional
                    if args.bt_min_wallet_notional is not None
                    else args.min_wallet_notional
                ),
                min_usd_amount=(
                    args.bt_min_usd_amount if args.bt_min_usd_amount is not None else args.min_usd_amount
                ),
                include_recidivism=args.include_recidivism,
                clustering_min_trade_size=(
                    args.bt_clustering_min_trade_size
                    if args.bt_clustering_min_trade_size is not None
                    else args.clustering_min_trade_size
                ),
                no_clustering=args.bt_no_clustering,
                enable_layer2_attribution=args.bt_enable_layer2_attribution,
                usdc_cache=args.usdc_cache,
                polygonscan_api_key=args.polygonscan_api_key,
                no_jump_anticipation=args.bt_no_jump_anticipation,
                copytrade_fixed_size=args.bt_copytrade_fixed_size,
                data_dir=args.data_dir,
                insider_plausible_only=test_flags["insider_plausible_only"],
                non_insider_plausible_only=test_flags["non_insider_plausible_only"],
                market_categories=args.market_categories,
                exclude_categories=args.exclude_categories,
                classifications_path=args.classifications_path,
                dry_run=False,
                verbose_output=False,
            )

            try:
                result, _eval_meta = run_timeframe_backtest_evaluation(
                    config, loader, bt, quiet=True
                )
            except RuntimeError as exc:
                logging.error(
                    "Backtest train=%s test=%s failed: %s",
                    train_name,
                    test_name,
                    exc,
                )
                rows.append(
                    {
                        "train_domain": train_name,
                        "test_domain": test_name,
                        "error": str(exc),
                    }
                )
                continue

            summary_row = summarize_evaluation(result, train_name, test_name)
            summary_row["config_path"] = str(config_path)
            rows.append(summary_row)

            if args.save_eval_artifacts:
                tag = f"bt_{train_name}_{test_name}_{run_ts}"
                saved = result.save(str(run_root), tag=tag)
                summary_row["saved_eval_paths"] = saved

    table_path = run_root / "domain_matrix_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(table_path, index=False)

    summary_path = run_root / "domain_matrix_summary.json"
    meta = {
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
        "test_start_date": test_start,
        "test_end_date": test_end,
        "optimizer_mode": args.optimizer_mode,
        "objective_metric": args.objective,
        "prediction_mode": args.prediction_mode,
        "train_manifest": str(manifest_path),
        "rows": rows,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 80)
    print("DOMAIN MATRIX COMPLETE")
    print("=" * 80)
    print(f"  Metrics CSV: {table_path}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Log: {log_path}")
    if not df.empty and "wallet_f1" in df.columns:
        print("\nWallet F1 (rows = train domain, cols = test domain):")
        try:
            pivot = df.pivot(index="train_domain", columns="test_domain", values="wallet_f1")
            print(pivot.to_string())
        except Exception:
            print(df.to_string())

    loader.close()


if __name__ == "__main__":
    main()
