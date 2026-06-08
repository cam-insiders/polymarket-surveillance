"""
Shared helpers for timeframe-based experiments (optimization + backtests).

Centralises resolution-override wiring and market inference prep so temporary
optimizer scripts and matrix runners stay small and consistent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import DEFAULT_CLUSTERING_CONFIG, EvaluationResult, evaluate_config

from experiments.timeframe_market_common import infer_resolutions, select_market_ids_in_timeframe


def normalize_trade_time_bound(value: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """Return ``(epoch_ms, sqlite_iso_utc)`` for an inclusive trade-time bound."""
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None

    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid trade timestamp bound: {value}")
    ts = ts.tz_convert("UTC")
    ms = int(ts.timestamp() * 1000)
    sql_text = ts.tz_localize(None).isoformat(timespec="microseconds")
    return ms, sql_text


@contextmanager
def scoped_trade_time_filter(
    loader: HistoricalDataLoader,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    cap_at_market_close: bool = False,
) -> Iterator[Dict[str, Optional[Any]]]:
    """
    Temporarily restrict trade loads in this process and optimizer subprocesses.
    """
    start_ms, start_sql = normalize_trade_time_bound(start_date)
    end_ms, end_sql = normalize_trade_time_bound(end_date)
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValueError("trade filter start_date must be <= end_date")

    old_loader_bounds = (
        getattr(loader, "trade_start_ms", None),
        getattr(loader, "trade_start_sql", None),
        getattr(loader, "trade_end_ms", None),
        getattr(loader, "trade_end_sql", None),
        getattr(loader, "cap_trades_at_market_close", False),
    )
    old_env_start = os.environ.get("POLYMARKET_TRADE_START_TIME")
    old_env_end = os.environ.get("POLYMARKET_TRADE_END_TIME")
    old_env_cap = os.environ.get("POLYMARKET_TRADE_END_AT_MARKET_CLOSE")

    if start_sql is None:
        os.environ.pop("POLYMARKET_TRADE_START_TIME", None)
    else:
        os.environ["POLYMARKET_TRADE_START_TIME"] = start_sql
    if end_sql is None:
        os.environ.pop("POLYMARKET_TRADE_END_TIME", None)
    else:
        os.environ["POLYMARKET_TRADE_END_TIME"] = end_sql
    os.environ["POLYMARKET_TRADE_END_AT_MARKET_CLOSE"] = "1" if cap_at_market_close else "0"

    loader.set_trade_time_bounds(
        start_sql,
        end_sql,
        cap_trades_at_market_close=cap_at_market_close,
    )
    try:
        yield {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_time": start_sql,
            "end_time": end_sql,
            "cap_at_market_close": bool(cap_at_market_close),
        }
    finally:
        if old_env_start is None:
            os.environ.pop("POLYMARKET_TRADE_START_TIME", None)
        else:
            os.environ["POLYMARKET_TRADE_START_TIME"] = old_env_start
        if old_env_end is None:
            os.environ.pop("POLYMARKET_TRADE_END_TIME", None)
        else:
            os.environ["POLYMARKET_TRADE_END_TIME"] = old_env_end
        if old_env_cap is None:
            os.environ.pop("POLYMARKET_TRADE_END_AT_MARKET_CLOSE", None)
        else:
            os.environ["POLYMARKET_TRADE_END_AT_MARKET_CLOSE"] = old_env_cap
        (
            loader.trade_start_ms,
            loader.trade_start_sql,
            loader.trade_end_ms,
            loader.trade_end_sql,
            loader.cap_trades_at_market_close,
        ) = old_loader_bounds
        if hasattr(loader, "_trade_cache"):
            loader._trade_cache.clear()


def setup_timeframe_logging(output_dir: str, log_filename_prefix: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{output_dir}/{log_filename_prefix}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def build_resolution_payload(
    loader: HistoricalDataLoader, overrides: Dict[int, int]
) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for market_id, winning_outcome in overrides.items():
        meta = loader.get_market_metadata(market_id) or {}
        payload[str(market_id)] = {
            "market_slug": str(meta.get("market_slug", market_id)),
            "winning_outcome": int(winning_outcome),
            "resolution_date": str(meta.get("closed_time", meta.get("closedTime", "inferred"))),
            "notes": "TEMP inferred from last traded outcome prices",
        }
    return payload


@dataclass(frozen=True)
class TimeframePrepResult:
    ts: str
    out_base: Path
    candidate_market_ids: List[int]
    inferred_winners: Dict[int, int]
    res_stats: Dict[str, Any]
    market_ids: List[int]
    override_path: Path


def materialize_timeframe_prep(
    loader: HistoricalDataLoader,
    *,
    output_dir: str,
    candidate_market_ids: List[int],
    inferred_winners: Dict[int, int],
    res_stats: Dict[str, Any],
    market_ids: Optional[List[int]] = None,
    override_filename_prefix: str = "timeframe_resolution_overrides",
) -> TimeframePrepResult:
    """
    Persist resolution overrides and return a reusable TimeframePrepResult.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    override_path = out_base / f"{override_filename_prefix}_{ts}.json"
    with open(override_path, "w", encoding="utf-8") as f:
        json.dump(build_resolution_payload(loader, inferred_winners), f, indent=2)

    os.environ["POLYMARKET_RESOLUTIONS_OVERRIDE_PATH"] = str(override_path)
    logging.info("Using runtime resolution overrides: %s", override_path)

    resolved_market_ids = sorted(market_ids) if market_ids is not None else sorted(inferred_winners.keys())
    return TimeframePrepResult(
        ts=ts,
        out_base=out_base,
        candidate_market_ids=list(candidate_market_ids),
        inferred_winners=dict(inferred_winners),
        res_stats=dict(res_stats),
        market_ids=resolved_market_ids,
        override_path=override_path,
    )


def prepare_timeframe_inference(
    loader: HistoricalDataLoader,
    *,
    output_dir: str,
    start_date: str,
    end_date: str,
    min_market_volume: float,
    classifications_path: str,
    insider_plausible_only: bool,
    non_insider_plausible_only: bool,
    market_categories: Optional[List[str]],
    exclude_categories: Optional[List[str]],
    resolution_threshold: float,
    min_trades: int,
    inferred_resolutions_db: str,
    enable_trade_prefilter: bool,
    min_usd_amount: Optional[float],
    override_filename_prefix: str = "timeframe_resolution_overrides",
) -> TimeframePrepResult:
    """
    Select markets, infer resolutions, write override JSON, set POLYMARKET_RESOLUTIONS_OVERRIDE_PATH.
    """
    candidate_market_ids = select_market_ids_in_timeframe(
        loader=loader,
        start_date=start_date,
        end_date=end_date,
        min_volume=min_market_volume,
        classifications_path=classifications_path,
        insider_plausible_only=insider_plausible_only,
        non_insider_plausible_only=non_insider_plausible_only,
        market_categories=market_categories,
        exclude_categories=exclude_categories,
    )
    logging.info("Candidate markets in timeframe: %s", f"{len(candidate_market_ids):,}")

    inferred_winners, res_stats = infer_resolutions(
        loader=loader,
        market_ids=candidate_market_ids,
        resolution_threshold=resolution_threshold,
        min_trades=min_trades,
        min_usd_amount=min_usd_amount if enable_trade_prefilter else None,
        inferred_resolutions_db=inferred_resolutions_db,
    )
    market_ids = sorted(inferred_winners.keys())

    logging.info(
        "Resolution inference: "
        f"resolved={res_stats['resolved']:,} / {res_stats['total_markets']:,}, "
        f"with_trades={res_stats['with_trades']:,}, "
        f"too_few_trades={res_stats['too_few_trades']:,}, "
        f"unresolved={res_stats['unresolved']:,}"
    )

    return materialize_timeframe_prep(
        loader=loader,
        output_dir=output_dir,
        candidate_market_ids=candidate_market_ids,
        inferred_winners=inferred_winners,
        res_stats=res_stats,
        market_ids=market_ids,
        override_filename_prefix=override_filename_prefix,
    )


def time_ordered_train_val_split(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    *,
    val_fraction: float,
    min_val_markets: int = 10,
) -> Tuple[List[int], List[int]]:
    """
    Split ``market_ids`` chronologically into (train, val) by market close time.
    """
    if not (0.0 <= float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction}")
    if val_fraction == 0.0 or not market_ids:
        return list(market_ids), []

    rows: List[Tuple[int, Optional[pd.Timestamp]]] = []
    for market_id in market_ids:
        meta = loader.get_market_metadata(market_id) or {}
        closed_raw = meta.get("closed_time", meta.get("closedTime"))
        if closed_raw is None:
            closed_ts: Optional[pd.Timestamp] = None
        else:
            closed_ts = pd.to_datetime(closed_raw, utc=True, errors="coerce")
            if pd.isna(closed_ts):
                closed_ts = None
        rows.append((int(market_id), closed_ts))

    n_with_time = sum(1 for _, ts in rows if ts is not None)
    if n_with_time == 0:
        logging.warning(
            "time_ordered_train_val_split: none of the %d input markets had a "
            "parseable close timestamp (looked for 'closed_time' / 'closedTime' "
            "in loader.get_market_metadata). Disabling split.",
            len(market_ids),
        )
        return list(market_ids), []

    with_time_pairs: List[Tuple[int, pd.Timestamp]] = sorted(
        ((mid, ts) for mid, ts in rows if ts is not None),
        key=lambda x: x[1],
    )
    with_time = [mid for mid, _ in with_time_pairs]
    without_time = [mid for mid, ts in rows if ts is None]

    n_val_target = int(round(len(market_ids) * float(val_fraction)))
    n_val = min(n_val_target, len(with_time))

    if n_val < min_val_markets:
        logging.warning(
            "time_ordered_train_val_split: requested val_fraction=%.3f would yield %d "
            "validation markets which is below min_val_markets=%d; disabling split.",
            val_fraction,
            n_val,
            min_val_markets,
        )
        return list(market_ids), []

    train_from_timed = with_time[:-n_val]
    val_markets = with_time[-n_val:]
    train_markets = without_time + train_from_timed

    train_boundary_ts = with_time_pairs[-n_val - 1][1] if n_val < len(with_time) else None
    val_boundary_ts = with_time_pairs[-n_val][1]
    val_last_ts = with_time_pairs[-1][1]

    logging.info(
        "Time-ordered train/val split: train=%d markets (latest train closedTime=%s), "
        "val=%d markets (%s .. %s)%s",
        len(train_markets),
        train_boundary_ts,
        len(val_markets),
        val_boundary_ts,
        val_last_ts,
        f", {len(without_time)} markets lacked closedTime and were routed to train"
        if without_time
        else "",
    )

    return sorted(train_markets), sorted(val_markets)


def add_standard_timeframe_optimizer_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags for timeframe coordinate-descent / alternating optimizers."""
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-market-volume", type=float, default=0.0)
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
    parser.add_argument("--classifications-path", type=str, default="data/market_classifications.json")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--inferred-resolutions-db", type=str, default="inferred_resolutions.db")

    parser.add_argument("--n-passes", type=int, default=2)
    parser.add_argument("--objective", type=str, default="f0_5")
    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--flag-rate-threshold", type=float, default=0.2)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--coarse-top-k", type=int, default=100)
    parser.add_argument("--coarse-trade-cap", type=int, default=500000)
    parser.add_argument("--enable-trade-prefilter", action="store_true", default=False)
    parser.add_argument("--min-usd-amount", type=float, default=300.0)
    parser.add_argument("--include-recidivism", action="store_true", default=False)
    parser.add_argument("--clustering-min-trade-size", type=float, default=5000.0)
    parser.add_argument("--data-dir", type=str, default="data")

    parser.add_argument("--enable-clustering", action="store_true", default=True)
    parser.add_argument(
        "--disable-clustering",
        dest="enable_clustering",
        action="store_false",
        help=(
            "Turn off the clustering stage. In the alternating runner this "
            "skips Stage 2 on every pass so only detector parameters are "
            "optimised."
        ),
    )
    parser.add_argument("--enable-jump-anticipation", action="store_true", default=False)
    parser.add_argument(
        "--enable-ja-optimization",
        action="store_true",
        default=False,
        help=(
            "Also optimise jump-anticipation parameters as part of each "
            "alternating pass (runs after the clustering stage). Requires "
            "--enable-jump-anticipation; otherwise ignored."
        ),
    )
    parser.add_argument(
        "--ja-n-passes",
        type=int,
        default=1,
        help=(
            "Coordinate-descent passes for the per-pass JA optimisation stage "
            "(only used when --enable-ja-optimization is set). The JA grid is "
            "tiny so 1 pass is usually sufficient."
        ),
    )

    parser.add_argument("--enable-layer2-attribution", action="store_true", default=False)
    parser.add_argument("--usdc-cache", type=str, default="data/usdc_transfers.db")
    parser.add_argument("--polygonscan-api-key", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)


def add_multi_start_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags controlling multi-start coordinate-descent behaviour.
    """
    parser.add_argument(
        "--n-starts",
        type=int,
        default=1,
        help=(
            "Number of coordinate-descent trajectories. 1 = plain single-start "
            "behaviour. Any value >1 triggers the multi-start runner."
        ),
    )
    parser.add_argument(
        "--start-strategy",
        type=str,
        default="perturb",
        choices=("perturb", "random", "mixed"),
        help=(
            "How to generate non-baseline starts. 'perturb' (default) replaces "
            "each baseline parameter with a random grid value independently "
            "with probability --perturb-prob. 'random' samples uniformly from "
            "the full joint grid. 'mixed' alternates the two."
        ),
    )
    parser.add_argument(
        "--perturb-prob",
        type=float,
        default=0.3,
        help="Per-parameter replacement probability when --start-strategy=perturb.",
    )
    parser.add_argument(
        "--exclude-baseline-start",
        dest="include_baseline_start",
        action="store_false",
        default=True,
        help=(
            "Skip the guaranteed baseline trajectory. Makes every start "
            "perturbed/random instead of keeping index 0 anchored at baseline."
        ),
    )
    parser.add_argument(
        "--no-shuffle-order",
        dest="shuffle_order",
        action="store_false",
        default=True,
        help=(
            "Disable per-start shuffling of the detector optimisation order. "
            "Order shuffling adds a second dimension of exploration for free."
        ),
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="Seed for the start-generation RNG (deterministic starts across reruns).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of timeframe markets (by closedTime) reserved for validation. "
            "Inner coordinate-descent runs on the earlier (1 - val_fraction) markets; "
            "each trajectory's final config is scored once on the held-out tail and "
            "the winner is picked on VALIDATION objective. Set to 0 to disable the "
            "split and rank purely on training objective (legacy behaviour). Default "
            "0.2 = last ~20%% of markets chronologically."
        ),
    )
    parser.add_argument(
        "--resume-skip-starts-before-idx",
        type=int,
        default=0,
        help=(
            "Resume: skip executing multi-start indices below N and load their "
            "results from --resume-from-json instead. The master RNG is still "
            "advanced as if those starts ran (same perturbations + shuffles), so "
            "starts N and beyond are bit-reproduced from the original run. "
            "Requires --resume-from-json when >0."
        ),
    )
    parser.add_argument(
        "--resume-from-json",
        type=str,
        default=None,
        help=(
            "Path to a reconstructed per-start JSON produced by "
            "`python -m experiments.recover_multi_start_det_clust`. Required when "
            "--resume-skip-starts-before-idx > 0."
        ),
    )


def filter_markets_by_window_trade_count(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    *,
    min_window_trades: int,
    min_usd_amount: Optional[float],
    label: str,
) -> Tuple[List[int], Dict[str, Any]]:
    """Keep markets with at least ``min_window_trades`` trades in the loader's current window."""
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


def run_timeframe_trade_window_backtest_evaluation(
    config: Dict[str, Any],
    loader: HistoricalDataLoader,
    args: Any,
    *,
    market_start: str,
    market_end: str,
    trade_start: str,
    trade_end: str,
    output_dir: str,
    min_window_trades: int = 1,
    trade_filter_label: str = "Eval",
    quiet: bool = False,
    override_filename_prefix: str = "trade_window_resolution_overrides",
) -> Tuple[EvaluationResult, Dict[str, Any]]:
    """
    Select markets by close time, infer resolutions on full history, then replay
    ``evaluate_config`` using only trades inside ``[trade_start, trade_end]``.
    """
    enable_trade_prefilter = bool(getattr(args, "enable_trade_prefilter", False))
    configured_min_usd = getattr(args, "min_usd_amount", None)
    trade_filter_min_usd = (
        float(configured_min_usd)
        if enable_trade_prefilter and configured_min_usd is not None
        else None
    )

    prep = prepare_timeframe_inference(
        loader,
        output_dir=output_dir,
        start_date=market_start,
        end_date=market_end,
        min_market_volume=float(args.min_market_volume),
        classifications_path=str(args.classifications_path),
        insider_plausible_only=bool(args.insider_plausible_only),
        non_insider_plausible_only=bool(args.non_insider_plausible_only),
        market_categories=getattr(args, "market_categories", None),
        exclude_categories=getattr(args, "exclude_categories", None),
        resolution_threshold=float(args.resolution_threshold),
        min_trades=int(args.min_trades),
        inferred_resolutions_db=str(args.inferred_resolutions_db),
        enable_trade_prefilter=enable_trade_prefilter,
        min_usd_amount=trade_filter_min_usd,
        override_filename_prefix=override_filename_prefix,
    )
    if not prep.market_ids:
        raise RuntimeError(
            f"No inferred-resolved markets closed in {market_start} .. {market_end}; "
            "nothing to evaluate."
        )

    clustering_config = None
    if not getattr(args, "no_clustering", False):
        clustering_config = config.get("clustering_config", DEFAULT_CLUSTERING_CONFIG)

    jump_anticipation_config = None
    if not getattr(args, "no_jump_anticipation", False):
        jump_anticipation_config = config.get("jump_anticipation_config", None)

    verbose_out = bool(getattr(args, "verbose_output", False))
    if quiet or not verbose_out:
        logging.getLogger("backtesting.evaluation").setLevel(logging.WARNING)

    with scoped_trade_time_filter(
        loader,
        start_date=trade_start,
        end_date=trade_end,
    ) as trade_filter:
        market_ids, window_stats = filter_markets_by_window_trade_count(
            loader,
            prep.market_ids,
            min_window_trades=int(min_window_trades),
            min_usd_amount=trade_filter_min_usd,
            label=trade_filter_label,
        )
        if not market_ids:
            raise RuntimeError(
                f"No markets have enough trades inside the {trade_filter_label.lower()} "
                f"replay window {trade_start} .. {trade_end}; try lowering "
                "--min-window-trades or widening the window."
            )
        winning_overrides = {
            int(mid): int(prep.inferred_winners[mid])
            for mid in market_ids
            if mid in prep.inferred_winners
        }
        result = evaluate_config(
            config=config,
            loader=loader,
            market_ids=market_ids,
            prediction_mode=args.prediction_mode,
            flag_rate_threshold=args.flag_rate_threshold,
            suspicion_threshold=args.suspicion_threshold,
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
            min_usd_amount=trade_filter_min_usd,
            include_recidivism=bool(args.include_recidivism),
            clustering_config=clustering_config,
            clustering_min_trade_size=args.clustering_min_trade_size,
            jump_anticipation_config=jump_anticipation_config,
            copytrade_fixed_size=getattr(args, "copytrade_fixed_size", None),
            measure_memory=False,
            winning_outcomes_override=winning_overrides,
            enable_layer2_attribution=bool(args.enable_layer2_attribution),
            usdc_cache_db=args.usdc_cache,
            polygonscan_api_key=args.polygonscan_api_key,
        )

    meta: Dict[str, Any] = {
        "candidate_markets": len(prep.candidate_market_ids),
        "resolved_markets": len(prep.market_ids),
        "resolved_markets_after_trade_filter": len(market_ids),
        "resolution_stats": prep.res_stats,
        "window_trade_stats": window_stats,
        "trade_filter": trade_filter,
        "winning_outcomes": winning_overrides,
        "resolution_override_path": str(prep.override_path),
        "market_start": market_start,
        "market_end": market_end,
        "trade_start": trade_start,
        "trade_end": trade_end,
    }
    return result, meta
