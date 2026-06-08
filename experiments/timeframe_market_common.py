"""
Experiment: Backtest one config across all markets in a timeframe.
Usage:
    python -m experiments.timeframe_market_common path/to/config.json
    python -m experiments.timeframe_market_common path/to/config.json --start-date 2025-01-01 --end-date 2025-12-31
    python -m experiments.timeframe_market_common backtest_results/best_config_20260407_190206.json --start-date 2025-01-01 --end-date 2025-01-05 --min-usd-amount 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import (
    DEFAULT_CLUSTERING_CONFIG,
    EvaluationResult,
    evaluate_config,
    print_copytrade_summary,
)
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode
from backtesting.parameter_optimizer import _calculate_metrics_from_wallet_evaluations
from backtesting.trade_event_study import infer_market_winning_outcome_from_last_prices
from experiments.inferred_resolution_cache import (
    load_cached_resolution_rows,
    upsert_market_resolution_cache,
)


def _fmt(value: Any, spec: str = ".4f", *, missing: str = "n/a") -> str:
    """Format a metric value that may be None / NaN / numpy scalar."""
    if value is None:
        return missing
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fval != fval:  # NaN
        return missing
    return format(fval, spec)


def print_wallet_classification_summary(result: EvaluationResult) -> None:
    """Print a compact wallet-level classification + PnL summary.
    """
    metrics = _calculate_metrics_from_wallet_evaluations(
        result.wallet_evaluations,
        result.prediction_mode,
        result.suspicion_threshold,
        result.flag_rate_threshold,
    )

    num_wallets = int(metrics.get("num_wallets", 0) or 0)
    num_predicted = int(metrics.get("num_predicted_positive", 0) or 0)
    num_true = int(metrics.get("num_true_insiders", 0) or 0)
    tp = int(metrics.get("true_positives", 0) or 0)
    fp = int(metrics.get("false_positives", 0) or 0)
    fn = int(metrics.get("false_negatives", 0) or 0)
    tn = int(metrics.get("true_negatives", 0) or 0)

    print(f"\n{'=' * 80}")
    print("WALLET CLASSIFICATION & PNL QUALITY")
    print(f"{'=' * 80}")
    print(f"  Wallets evaluated:         {num_wallets:,}")
    print(f"  Flagged (predicted pos):   {num_predicted:,}")
    print(f"  True insiders (labels):    {num_true:,}")
    print(f"  Confusion:                 TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(
        "  Precision / Recall:        "
        f"{_fmt(metrics.get('precision'))} / {_fmt(metrics.get('recall'))}"
    )
    print(
        "  F-scores:                  "
        f"F0.5={_fmt(metrics.get('f0_5'))}  "
        f"F1={_fmt(metrics.get('f1'))}  "
        f"F2={_fmt(metrics.get('f2'))}"
    )
    print(
        "  Flagged PnL:               "
        f"mean_net=${_fmt(metrics.get('mean_net_pnl_flagged'), '+.2f')}  "
        f"median_net=${_fmt(metrics.get('median_net_pnl_flagged'), '+.2f')}  "
        f"median_return={_fmt(metrics.get('median_return_flagged'), '+.4f')}"
    )
    print(
        "  Flagged informed score:    "
        f"median={_fmt(metrics.get('median_informed_score_flagged'))}"
    )
    print(
        "  Trade-level (if computed): "
        f"mean_return_diff={_fmt(metrics.get('trade_mean_return_diff'), '+.4f')}  "
        f"t_stat={_fmt(metrics.get('trade_t_stat'), '+.2f')}  "
        f"cohens_d={_fmt(metrics.get('trade_cohens_d'), '+.3f')}"
    )

    composite = _compute_quality_composite(metrics)
    if composite is not None:
        print(
            "  Quality composite:         "
            f"{composite:+.4f}   "
            "(F0.5 * tanh(median_net_pnl/100) — signed PnL-weighted precision)"
        )


def _compute_quality_composite(metrics: Dict[str, Any]) -> Optional[float]:
    """Cheap single-number summary: F0.5 weighted by a bounded PnL signal.
    """
    import math

    f0_5 = metrics.get("f0_5")
    pnl = metrics.get("median_net_pnl_flagged")
    if f0_5 is None or pnl is None:
        return None
    try:
        f0_5_val = float(f0_5)
        pnl_val = float(pnl)
    except (TypeError, ValueError):
        return None
    if f0_5_val != f0_5_val or pnl_val != pnl_val:
        return None
    return f0_5_val * math.tanh(pnl_val / 100.0)


def _parse_iso_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    return ts


def _normalize_category_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    normalized = [str(value).strip().upper() for value in values if str(value).strip()]
    return normalized or None


def select_market_ids_in_timeframe(
    loader: HistoricalDataLoader,
    start_date: Optional[str],
    end_date: Optional[str],
    min_volume: float,
    classifications_path: Optional[str] = None,
    insider_plausible_only: bool = False,
    non_insider_plausible_only: bool = False,
    market_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
) -> List[int]:
    if loader.markets_df is None:
        raise RuntimeError("Call load_data() first.")
    if insider_plausible_only and non_insider_plausible_only:
        raise ValueError(
            "insider_plausible_only and non_insider_plausible_only are mutually exclusive"
        )

    df = loader.markets_df.copy()
    df["closed_dt"] = pd.to_datetime(df["closedTime"], utc=True, errors="coerce")
    df["volume_num"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    start_ts = _parse_iso_date(start_date)
    end_ts = _parse_iso_date(end_date)

    mask = df["closed_dt"].notna() & (df["volume_num"] >= float(min_volume))
    if start_ts is not None:
        mask &= df["closed_dt"] >= start_ts
    if end_ts is not None:
        mask &= df["closed_dt"] <= end_ts

    selected = [int(x) for x in df.loc[mask, "id"].tolist()]

    normalized_market_categories = _normalize_category_list(market_categories)
    normalized_exclude_categories = _normalize_category_list(exclude_categories)

    if (
        insider_plausible_only
        or non_insider_plausible_only
        or normalized_market_categories
        or normalized_exclude_categories
    ):
        if not classifications_path or not Path(classifications_path).exists():
            logging.warning(
                "Classification filters requested but file not found: %s. Skipping classification filter.",
                classifications_path,
            )
        else:
            with open(classifications_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            classifications = data.get("classifications", {})

            original_count = len(selected)
            filtered: List[int] = []

            for market_id in selected:
                mid_str = str(market_id)
                if mid_str not in classifications:
                    continue

                classification = classifications[mid_str]
                if not isinstance(classification, dict):
                    continue
                category = classification.get("category")
                category = str(category).upper() if category is not None else None
                if not category:
                    continue

                if insider_plausible_only and not classification.get("insider_plausible", False):
                    continue
                if non_insider_plausible_only and classification.get("insider_plausible", False):
                    continue
                if normalized_market_categories and category not in normalized_market_categories:
                    continue
                if normalized_exclude_categories and category in normalized_exclude_categories:
                    continue

                filtered.append(market_id)

            selected = filtered
            logging.info(
                "Classification filter: %s -> %s markets (insider_only=%s, non_insider_only=%s, categories=%s, exclude=%s)",
                original_count,
                len(selected),
                insider_plausible_only,
                non_insider_plausible_only,
                normalized_market_categories,
                normalized_exclude_categories,
            )

    return selected


def infer_resolutions(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    resolution_threshold: float,
    min_trades: int,
    min_usd_amount: Optional[float] = None,
    inferred_resolutions_db: str = "inferred_resolutions.db",
    save_cache: bool = True,
) -> Tuple[Dict[int, int], Dict[str, int]]:
    overrides: Dict[int, int] = {}
    inferred_rows: List[Dict] = []
    cached_rows = load_cached_resolution_rows(
        db_path=inferred_resolutions_db,
        market_ids=market_ids,
    )
    _rp = Path(inferred_resolutions_db).expanduser().resolve()
    logging.info(
        "Resolution cache: arg=%s resolved=%s exists=%s rows_loaded=%s requested_markets=%s",
        inferred_resolutions_db,
        _rp,
        _rp.is_file(),
        len(cached_rows),
        len(market_ids),
    )
    stats = {
        "total_markets": len(market_ids),
        "with_trades": sum(1 for row in cached_rows.values() if int(row.get("n_trades", 0) or 0) > 0),
        "too_few_trades": 0,
        "resolved": sum(
            1
            for row in cached_rows.values()
            if row.get("inference_status") == "resolved"
            and row.get("inferred_winning_outcome") is not None
        ),
        "unresolved": sum(
            1
            for row in cached_rows.values()
            if row.get("inference_status") != "resolved"
        ),
        "cache_used": 1 if cached_rows else 0,
        "cache_job_id": None,
        "cache_skipped_missing": max(0, len(market_ids) - len(cached_rows)),
        "saved_job_id": None,
        "saved_cache_rows": 0,
    }
    for market_id, row in cached_rows.items():
        if row.get("inference_status") == "resolved" and row.get("inferred_winning_outcome") is not None:
            overrides[int(market_id)] = int(row["inferred_winning_outcome"])

    missing_market_ids = [market_id for market_id in market_ids if market_id not in cached_rows]
    if not missing_market_ids:
        return overrides, stats

    progress_iter = missing_market_ids
    if _TQDM_AVAILABLE:
        progress_iter = tqdm(missing_market_ids, desc="Inferring resolutions", unit="market")

    for idx, market_id in enumerate(progress_iter, start=1):
        metadata = dict(loader.get_market_metadata(market_id) or {})
        market_slug = metadata.get("market_slug")
        closed_time_utc = metadata.get("closedTime")
        try:
            volume = float(metadata.get("volume")) if metadata.get("volume") is not None else None
        except (TypeError, ValueError):
            volume = None
        try:
            trades = loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=None,
                use_cache=False,
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)

        if not trades:
            stats["unresolved"] += 1
            inferred_rows.append(
                {
                    "market_id": int(market_id),
                    "market_slug": market_slug,
                    "closed_time_utc": closed_time_utc,
                    "volume": volume,
                    "n_trades": 0,
                    "inferred_winning_outcome": None,
                    "inference_status": "no_trades",
                    "latest_trade_ts_ms": None,
                }
            )
            continue

        latest_trade_ts_ms = max(int(t.timestamp_ms) for t in trades)
        stats["with_trades"] += 1
        winning = infer_market_winning_outcome_from_last_prices(
            trades=trades,
            threshold=resolution_threshold,
        )
        if winning is None:
            stats["unresolved"] += 1
            inferred_rows.append(
                {
                    "market_id": int(market_id),
                    "market_slug": market_slug,
                    "closed_time_utc": closed_time_utc,
                    "volume": volume,
                    "n_trades": len(trades),
                    "inferred_winning_outcome": None,
                    "inference_status": "unresolved",
                    "latest_trade_ts_ms": latest_trade_ts_ms,
                }
            )
            continue

        overrides[int(market_id)] = int(winning)
        stats["resolved"] += 1
        inferred_rows.append(
            {
                "market_id": int(market_id),
                "market_slug": market_slug,
                "closed_time_utc": closed_time_utc,
                "volume": volume,
                "n_trades": len(trades),
                "inferred_winning_outcome": int(winning),
                "inference_status": "resolved",
                "latest_trade_ts_ms": latest_trade_ts_ms,
            }
        )

        if (not _TQDM_AVAILABLE) and (idx % 25 == 0 or idx == len(missing_market_ids)):
            logging.info(
                "Resolution inference progress: "
                f"{idx:,}/{len(missing_market_ids):,} markets | "
                f"resolved={stats['resolved']:,}"
            )

    if save_cache:
        stats["saved_cache_rows"] = upsert_market_resolution_cache(
            db_path=inferred_resolutions_db,
            rows=inferred_rows,
            resolution_threshold=resolution_threshold,
            source="manual_infer",
        )
        if stats["saved_cache_rows"]:
            logging.info(
                "Saved %s inferred resolution row(s) to %s",
                stats["saved_cache_rows"],
                inferred_resolutions_db,
            )

    return overrides, stats


def run_timeframe_backtest_evaluation(
    config: Dict,
    loader: HistoricalDataLoader,
    args: Any,
    *,
    quiet: bool = False,
) -> Tuple[EvaluationResult, Dict]:
    """
    Run select → infer resolutions → evaluate_config for a timeframe backtest.
    """
    clustering_config = None
    if not args.no_clustering:
        clustering_config = config.get("clustering_config", DEFAULT_CLUSTERING_CONFIG)

    jump_anticipation_config = None
    if not args.no_jump_anticipation:
        jump_anticipation_config = config.get("jump_anticipation_config", None)

    candidate_market_ids = select_market_ids_in_timeframe(
        loader=loader,
        start_date=args.start_date,
        end_date=args.end_date,
        min_volume=args.min_market_volume,
        classifications_path=args.classifications_path,
        insider_plausible_only=args.insider_plausible_only,
        non_insider_plausible_only=args.non_insider_plausible_only,
        market_categories=args.market_categories,
        exclude_categories=args.exclude_categories,
    )
    logging.info(f"Candidate markets in timeframe: {len(candidate_market_ids):,}")

    winning_overrides, resolution_stats = infer_resolutions(
        loader=loader,
        market_ids=candidate_market_ids,
        resolution_threshold=args.resolution_threshold,
        min_trades=args.min_trades,
        min_usd_amount=args.min_usd_amount,
        inferred_resolutions_db=args.inferred_resolutions_db,
        save_cache=not getattr(args, "dry_run", False),
    )
    market_ids = sorted(winning_overrides.keys())

    logging.info(
        "Resolution inference: "
        f"resolved={resolution_stats['resolved']:,} / {resolution_stats['total_markets']:,}, "
        f"with_trades={resolution_stats['with_trades']:,}, "
        f"too_few_trades={resolution_stats['too_few_trades']:,}, "
        f"unresolved={resolution_stats['unresolved']:,}"
    )

    if not market_ids:
        raise RuntimeError("No markets resolved in the selected timeframe; nothing to evaluate.")

    verbose_out = getattr(args, "verbose_output", False)
    if quiet or not verbose_out:
        logging.getLogger("backtesting.evaluation").setLevel(logging.WARNING)

    result = evaluate_config(
        config=config,
        loader=loader,
        market_ids=market_ids,
        prediction_mode=args.prediction_mode,
        flag_rate_threshold=args.flag_rate_threshold,
        suspicion_threshold=args.suspicion_threshold,
        z_score_threshold=args.z_score_threshold,
        min_wallet_notional=args.min_wallet_notional,
        min_usd_amount=args.min_usd_amount,
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

    meta = {
        "candidate_markets": len(candidate_market_ids),
        "resolved_markets": len(market_ids),
        "resolution_stats": resolution_stats,
    }
    return result, meta


def main():
    parser = argparse.ArgumentParser(
        description="Backtest one config on all markets in a timeframe using inferred resolutions."
    )
    parser.add_argument("config_path", type=str, help="Path to config JSON")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive ISO start date")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive ISO end date")
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-market-volume", type=float, default=0.0)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--inferred-resolutions-db", type=str, default="inferred_resolutions.db")
    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--flag-rate-threshold", type=float, default=0.2)
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--min-usd-amount", type=float, default=None)
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
    parser.add_argument("--output-dir", type=str, default="experiments/results/timeframe_backtest")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show filtered market counts and inferred resolutions without running evaluation.",
    )
    parser.add_argument(
        "--verbose-output",
        action="store_true",
        default=False,
        help="Print per-market and per-event detailed output.",
    )
    args = parser.parse_args()
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

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    if args.dry_run:
        candidate_market_ids = select_market_ids_in_timeframe(
            loader=loader,
            start_date=args.start_date,
            end_date=args.end_date,
            min_volume=args.min_market_volume,
            classifications_path=args.classifications_path,
            insider_plausible_only=args.insider_plausible_only,
            non_insider_plausible_only=args.non_insider_plausible_only,
            market_categories=args.market_categories,
            exclude_categories=args.exclude_categories,
        )
        logging.info(f"Candidate markets in timeframe: {len(candidate_market_ids):,}")

        winning_overrides, resolution_stats = infer_resolutions(
            loader=loader,
            market_ids=candidate_market_ids,
            resolution_threshold=args.resolution_threshold,
            min_trades=args.min_trades,
            min_usd_amount=args.min_usd_amount,
            inferred_resolutions_db=args.inferred_resolutions_db,
            save_cache=False,
        )
        market_ids = sorted(winning_overrides.keys())

        logging.info(
            "Resolution inference: "
            f"resolved={resolution_stats['resolved']:,} / {resolution_stats['total_markets']:,}, "
            f"with_trades={resolution_stats['with_trades']:,}, "
            f"too_few_trades={resolution_stats['too_few_trades']:,}, "
            f"unresolved={resolution_stats['unresolved']:,}"
        )

        print("\nDRY RUN")
        print(f"  Candidate markets: {len(candidate_market_ids):,}")
        print(f"  Resolved markets:  {len(market_ids):,}")
        print(
            "  Classification filters: "
            f"insider_plausible_only={args.insider_plausible_only}, "
            f"non_insider_plausible_only={args.non_insider_plausible_only}, "
            f"categories={args.market_categories}, "
            f"exclude={args.exclude_categories}"
        )
        if market_ids:
            preview = market_ids[:20]
            print(f"  Sample resolved market IDs (up to 20): {preview}")
        loader.close()
        return

    try:
        result, eval_meta = run_timeframe_backtest_evaluation(config, loader, args, quiet=not args.verbose_output)
    except RuntimeError as exc:
        logging.error("%s", exc)
        loader.close()
        sys.exit(1)

    agg = result.aggregate_performance
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"  Markets evaluated:       {agg.total_markets:,}")
    print(f"  Total trades processed:  {agg.total_trades:,}")
    print(f"  Wall clock time:         {agg.total_wall_clock_seconds:.2f}s")
    print(f"  Throughput:              {agg.overall_trades_per_second:,.0f} trades/sec")
    print(f"  Detection p95 latency:   {agg.detection_latency_p95_us:.1f}us")

    print(
        "\nCopytrade eval settings: "
        f"prediction_mode={result.prediction_mode}, "
        f"flag_rate_threshold={result.flag_rate_threshold}"
    )

    print_wallet_classification_summary(result)

    print_copytrade_summary("WALLET-LEVEL COPYTRADE REPORT", result.copytrade_summary)

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
        print("\nPOOLED across all markets:")
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
                f"  [Fixed ${args.copytrade_fixed_size:.0f}/trade]    "
                f"capital=${ct.fixed_capital_deployed:,.2f}  "
                f"P&L=${ct.fixed_total_pnl:+,.2f}  "
                f"ROI={ct.fixed_roi:+.2%}  "
                f"median={ct.fixed_median_return:+.4f}"
            )

    tag_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = result.save(args.output_dir, tag=f"timeframe_{tag_ts}")

    meta = {
        "config_path": args.config_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "resolution_threshold": args.resolution_threshold,
        "min_market_volume": args.min_market_volume,
        "min_trades": args.min_trades,
        "candidate_markets": eval_meta["candidate_markets"],
        "resolved_markets": eval_meta["resolved_markets"],
        "resolution_stats": eval_meta["resolution_stats"],
        "classification_filters": {
            "insider_plausible_only": args.insider_plausible_only,
            "non_insider_plausible_only": args.non_insider_plausible_only,
            "market_categories": args.market_categories,
            "exclude_categories": args.exclude_categories,
            "classifications_path": args.classifications_path,
        },
    }
    meta_path = Path(args.output_dir) / f"timeframe_run_meta_{tag_ts}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nResults saved:")
    for label, path in saved.items():
        print(f"  {label}: {path}")
    print(f"  run_meta: {meta_path}")

    loader.close()


if __name__ == "__main__":
    main()
