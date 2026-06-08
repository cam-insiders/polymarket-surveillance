"""
Experiment: Post-hoc sweep of flag_rate_threshold.

Runs evaluation ONCE on markets that resolve in the timeframe, replaying only
trades inside that same timeframe, then re-thresholds wallet evaluations at
many flag_rate values. Generates a precision-recall curve without re-running
any detectors.

Usage:
    python -m experiments.sweep_flag_rate backtest_results/best_config_xxx.json \\
        --start-date 2025-01-01 --end-date 2025-03-31

    python -m experiments.sweep_flag_rate path/to/config.json \\
        --start-date 2025-01-01 --end-date 2025-01-31 --insider-plausible-only
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import predict_wallet_positive
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode

from experiments.timeframe_market_common import (
    _normalize_category_list,
)
from experiments.common.timeframe import (
    run_timeframe_trade_window_backtest_evaluation,
)

DEFAULT_THRESHOLDS = [
    0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18,
    0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


def rethreshold(
    wallet_evaluations: List[dict],
    thresholds: List[float],
    prediction_mode: str = "flag_rate",
    suspicion_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Re-threshold wallet evaluations at many flag_rate values.
    Returns DataFrame with one row per threshold.

    This is O(n_wallets * n_thresholds) — essentially free.
    """
    y_true = [bool(e["is_insider"]) for e in wallet_evaluations]
    rows = []

    for thr in thresholds:
        y_pred = [predict_wallet_positive(e, prediction_mode, suspicion_threshold, thr)
                  for e in wallet_evaluations]

        if sum(y_pred) == 0 and sum(y_true) == 0:
            rows.append({"flag_rate_threshold": thr, "tp": 0, "fp": 0, "fn": 0, "tn": len(y_true),
                         "precision": 0, "recall": 0, "f1": 0, "f0_5": 0, "f2": 0,
                         "n_flagged": 0, "flagged_avg_return": 0, "flagged_avg_pnl": 0})
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f0_5 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

        flagged = [e for e, p in zip(wallet_evaluations, y_pred) if p]
        flagged_returns = [float(e.get("return", 0)) for e in flagged]
        flagged_pnls = [float(e.get("net_pnl", 0)) for e in flagged]

        rows.append({
            "flag_rate_threshold": thr,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "f0_5": float(f0_5), "f2": float(f2),
            "n_flagged": len(flagged),
            "flagged_avg_return": float(np.mean(flagged_returns)) if flagged_returns else 0.0,
            "flagged_avg_pnl": float(np.mean(flagged_pnls)) if flagged_pnls else 0.0,
        })

    return pd.DataFrame(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc flag_rate_threshold sweep on a timeframe "
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
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-market-volume", type=float, default=0.0)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--inferred-resolutions-db", type=str, default="inferred_resolutions.db")
    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--min-usd-amount", type=float, default=None)
    parser.add_argument(
        "--min-window-trades",
        type=int,
        default=1,
        help="Drop resolved markets with fewer than this many in-window trades.",
    )
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
        help="Filter to specific categories: ELECTION, EARNINGS, POLICY, etc.",
    )
    parser.add_argument(
        "--exclude-categories",
        type=str,
        nargs="+",
        default=None,
        help="Exclude specific categories: CRYPTO_PRICE, SPORTS, etc.",
    )
    parser.add_argument(
        "--classifications-path",
        type=str,
        default="data/market_classifications.json",
        help="Path to market classifications JSON",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/results/sweep_flag_rate")
    parser.add_argument(
        "--verbose-output",
        action="store_true",
        default=False,
        help="Verbose evaluation logging (default: quiet for the single evaluation pass).",
    )
    return parser


def main():
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

    with open(args.config_path, encoding="utf-8") as f:
        config = json.load(f)

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    # Single eval uses a very low flag_rate so wallet_evaluations are populated; sweep is post-hoc.
    args.flag_rate_threshold = 0.01
    args.enable_trade_prefilter = args.min_usd_amount is not None

    print(
        f"Running single trade-window evaluation on timeframe {args.start_date} .. {args.end_date} "
        f"(insider_plausible_only={args.insider_plausible_only})..."
    )
    result, meta = run_timeframe_trade_window_backtest_evaluation(
        config=config,
        loader=loader,
        args=args,
        market_start=args.start_date,
        market_end=args.end_date,
        trade_start=args.start_date,
        trade_end=args.end_date,
        output_dir=args.output_dir,
        min_window_trades=int(args.min_window_trades),
        trade_filter_label="Eval",
        override_filename_prefix="sweep_flag_rate_resolution_overrides",
        quiet=not args.verbose_output,
    )
    print(
        f"Resolved markets: {meta['resolved_markets_after_trade_filter']:,} after trade filter / "
        f"{meta['resolved_markets']:,} resolved "
        f"(candidates: {meta['candidate_markets']:,})"
    )

    print(f"\nSweeping {len(DEFAULT_THRESHOLDS)} flag_rate thresholds...")
    df = rethreshold(
        result.wallet_evaluations,
        DEFAULT_THRESHOLDS,
        args.prediction_mode,
        args.suspicion_threshold,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{args.output_dir}/flag_rate_sweep_{ts}.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*80}")
    print("FLAG RATE THRESHOLD SWEEP")
    print(f"{'='*80}")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved: {path}")
    loader.close()


if __name__ == "__main__":
    main()
