"""Reusable optimizer runners for timeframe experiments."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtesting.cached_optimizer import CachedCoordinateDescentOptimizer
from backtesting.clustering_optimizer import ClusteringOptimizer
from backtesting.clustering_parameter_grid import ClusteringParameterGrid
from backtesting.data_loader import HistoricalDataLoader
from backtesting.multi_start_optimizer import MultiStartCoordinateDescentOptimizer
from backtesting.parameter_grid import ParameterGrid
from config import CONFIG
from models import filter_trades_by_notional

from experiments.timeframe_experiment_common import (
    TimeframePrepResult,
    time_ordered_train_val_split,
)


DETECTOR_OPTIMIZE_ORDER: List[str] = [
    "volume_anomaly",
    "probability_impact",
    "accumulation_detector",
    "extreme_position",
    "contra_outcome_silence",
    "alert_threshold",
]

CLUSTERING_OPTIMIZE_ORDER: List[str] = [
    "boost_magnitude",
    "boost_weights",
    "boost_normalizer",
    "clustering",
    "time_window",
    "size",
]


def _detector_optimizer_kwargs(
    args: Any,
    *,
    clustering_config: Optional[Dict],
    max_workers: int,
    ja_config_override: Any = "UNSET",
) -> Dict[str, Any]:
    """Build shared kwargs for detector optimizers."""
    if ja_config_override == "UNSET":
        ja_config = CONFIG.get("jump_anticipation_config") if args.enable_jump_anticipation else None
    else:
        ja_config = ja_config_override if args.enable_jump_anticipation else None
    return {
        "z_score_threshold": args.z_score_threshold,
        "min_wallet_notional": args.min_wallet_notional,
        "label_metric": "return",
        "prediction_mode": args.prediction_mode,
        "suspicion_threshold": 2.0,
        "flag_rate_threshold": args.flag_rate_threshold,
        "coarse_top_k": args.coarse_top_k,
        "coarse_trade_cap": args.coarse_trade_cap,
        "min_usd_amount": args.min_usd_amount,
        "enable_trade_prefilter": args.enable_trade_prefilter,
        "data_dir": args.data_dir,
        "max_workers": max_workers,
        "parallelize_coarse": True,
        "parallelize_full": True,
        "show_progress": True,
        "objective_metric": args.objective,
        "clustering_config": clustering_config,
        "clustering_min_trade_size": args.clustering_min_trade_size,
        "ja_config": ja_config,
        "enable_layer2_attribution": getattr(args, "enable_layer2_attribution", False),
        "usdc_cache_db": getattr(args, "usdc_cache", "data/usdc_transfers.db"),
        "polygonscan_api_key": getattr(args, "polygonscan_api_key", None),
    }


def _clustering_optimizer_kwargs(
    args: Any,
    *,
    detector_config: Dict[str, Any],
    max_workers: int,
    ja_config_override: Any = "UNSET",
) -> Dict[str, Any]:
    if ja_config_override == "UNSET":
        ja_config = CONFIG.get("jump_anticipation_config") if args.enable_jump_anticipation else None
    else:
        ja_config = ja_config_override if args.enable_jump_anticipation else None
    return {
        "detector_config": detector_config,
        "z_score_threshold": args.z_score_threshold,
        "min_wallet_notional": args.min_wallet_notional,
        "label_metric": "return",
        "prediction_mode": args.prediction_mode,
        "suspicion_threshold": 2.0,
        "flag_rate_threshold": args.flag_rate_threshold,
        "coarse_top_k": args.coarse_top_k,
        "coarse_trade_cap": args.coarse_trade_cap,
        "min_trade_size": args.clustering_min_trade_size,
        "min_usd_amount": args.min_usd_amount,
        "enable_trade_prefilter": args.enable_trade_prefilter,
        "data_dir": args.data_dir,
        "max_workers": max_workers,
        "parallelize_coarse": True,
        "parallelize_full": True,
        "show_progress": True,
        "objective_metric": args.objective,
        "include_recidivism": args.include_recidivism,
        "enable_layer2_attribution": getattr(args, "enable_layer2_attribution", False),
        "usdc_cache_db": getattr(args, "usdc_cache", "data/usdc_transfers.db"),
        "polygonscan_api_key": getattr(args, "polygonscan_api_key", None),
        "jump_anticipation_config": ja_config,
    }


def _run_ja_optimization_for_markets(
    loader: HistoricalDataLoader,
    args: Any,
    *,
    detector_config: Dict[str, Any],
    clustering_config: Dict[str, Any],
    initial_ja_config: Optional[Dict[str, Any]],
    market_ids: List[int],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Optimize jump anticipation for a fixed detector and clustering config."""
    from backtesting.backtest_runner import BacktestRunner
    from backtesting.cached_evaluator import precompute_ground_truth
    from jump_anticipation.optimizer import (
        JumpAnticipationOptimizer,
        get_ja_baseline_config,
    )

    effective_min = float(args.min_usd_amount) if args.enable_trade_prefilter else None

    base_results: Dict[int, Any] = {}
    all_trades_map: Dict[int, Any] = {}
    scoring_trades_map: Dict[int, Any] = {}
    precomputed_gts: Dict[int, Any] = {}

    for market_id in market_ids:
        try:
            replay_trades = loader.get_trades_for_market(
                market_id=market_id, min_usd_amount=None, use_cache=False
            )
        except TypeError:
            replay_trades = loader.get_trades_for_market(market_id)
        try:
            ground_truth_trades = loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=None,
                use_cache=False,
                ignore_trade_time_bounds=True,
            )
        except TypeError:
            ground_truth_trades = replay_trades
        metadata = dict(loader.get_market_metadata(market_id) or {})
        metadata["id"] = market_id

        if effective_min is not None:
            detector_trades = filter_trades_by_notional(replay_trades, effective_min)
        else:
            detector_trades = replay_trades

        runner = BacktestRunner(
            config=detector_config, include_recidivism=args.include_recidivism
        )
        base_result = runner.run_backtest(
            trades=detector_trades, market_metadata=metadata,
            capture_alerts=False, capture_trade_features=False, progress_every=0,
        )

        base_results[market_id] = base_result
        all_trades_map[market_id] = replay_trades
        scoring_trades_map[market_id] = detector_trades

        gt = precompute_ground_truth(
            trades=ground_truth_trades,
            market_metadata=metadata,
            label_metric="return",
            z_score_threshold=float(args.z_score_threshold),
            min_wallet_notional=float(args.min_wallet_notional),
        )
        if gt is not None:
            precomputed_gts[market_id] = gt

    ja_optimizer = JumpAnticipationOptimizer(
        detector_config=detector_config,
        clustering_config=clustering_config,
        clustering_min_trade_size=float(args.clustering_min_trade_size),
        include_recidivism=bool(args.include_recidivism),
        prediction_mode=args.prediction_mode,
        flag_rate_threshold=float(args.flag_rate_threshold),
        suspicion_threshold=2.0,
        objective_metric=args.objective,
        n_passes=int(getattr(args, "ja_n_passes", 1) or 1),
        show_progress=False,
        poll_interval_seconds=float(getattr(args, "poll_interval_seconds", 5.0) or 5.0),
        enable_layer2_attribution=bool(getattr(args, "enable_layer2_attribution", False)),
        usdc_cache_db=str(getattr(args, "usdc_cache", "data/usdc_transfers.db")),
        polygonscan_api_key=getattr(args, "polygonscan_api_key", None),
    )
    starting_ja_config = deepcopy(initial_ja_config) if initial_ja_config else get_ja_baseline_config()
    best_ja_config, ja_df = ja_optimizer.optimize(
        base_results=base_results,
        all_trades=all_trades_map,
        scoring_trades=scoring_trades_map,
        precomputed_gts=precomputed_gts,
        initial_config=starting_ja_config,
    )
    return best_ja_config, ja_df


def _multi_start_kwargs(args: Any) -> Dict[str, Any]:
    """Constructor kwargs for :class:`MultiStartCoordinateDescentOptimizer`."""
    return {
        "n_starts": int(getattr(args, "n_starts", 1) or 1),
        "start_strategy": str(getattr(args, "start_strategy", "perturb")),
        "perturb_prob": float(getattr(args, "perturb_prob", 0.3)),
        "include_baseline_start": bool(getattr(args, "include_baseline_start", True)),
        "shuffle_order": bool(getattr(args, "shuffle_order", True)),
        "random_seed": int(getattr(args, "start_seed", 42)),
    }


def _resolve_max_workers(args: Any, cap: int = 12) -> int:
    cpu_count = os.cpu_count() or 1
    explicit = getattr(args, "max_workers", None)
    if explicit is None:
        return min(cpu_count, cap)
    return max(1, min(int(explicit), cpu_count))


def _build_timeframe_meta(args: Any, prep: TimeframePrepResult, *, extra: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the ``timeframe_optimization_meta`` block stored alongside
    the best-config JSON.  ``extra`` lets callers inject runner-specific keys
    (``max_workers``, ``multi_start``, ``alternating_passes``, etc.).
    """
    meta: Dict[str, Any] = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "resolution_threshold": args.resolution_threshold,
        "min_market_volume": args.min_market_volume,
        "min_trades": args.min_trades,
        "inferred_resolutions_db": args.inferred_resolutions_db,
        "candidate_markets": len(prep.candidate_market_ids),
        "optimized_markets": len(prep.market_ids),
        "resolution_stats": prep.res_stats,
        "classification_filters": {
            "insider_plausible_only": args.insider_plausible_only,
            "non_insider_plausible_only": args.non_insider_plausible_only,
            "market_categories": args.market_categories,
            "exclude_categories": args.exclude_categories,
            "classifications_path": args.classifications_path,
        },
        "enable_trade_prefilter": args.enable_trade_prefilter,
        "min_usd_amount": args.min_usd_amount if args.enable_trade_prefilter else None,
        "resolutions_override_path": str(prep.override_path),
        "objective_metric": args.objective,
        "prediction_mode": args.prediction_mode,
    }
    meta.update(extra)
    return meta


def _dump_inferred_resolutions(path: Path, inferred_winners: Dict[int, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in inferred_winners.items()}, f, indent=2)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _best_row_from_df(results_df: pd.DataFrame) -> Dict[str, Any]:
    rank_col = "objective_score" if "objective_score" in results_df.columns else "f1"
    if rank_col not in results_df.columns or results_df.empty:
        return {}
    best_idx = results_df[rank_col].idxmax()
    return results_df.loc[best_idx].to_dict()

def run_coordinate_descent_timeframe(
    loader: HistoricalDataLoader,
    prep: TimeframePrepResult,
    args: Any,
) -> Dict[str, Any]:
    """Single-start CachedCoordinateDescentOptimizer (detectors only)."""
    if not prep.market_ids:
        raise RuntimeError("No inferred-resolved markets in timeframe; cannot optimize.")

    clustering_config = CONFIG.get("clustering_config") if args.enable_clustering else None
    ja_config = CONFIG.get("jump_anticipation_config") if args.enable_jump_anticipation else None
    max_workers = _resolve_max_workers(args, cap=12)

    optimizer_kwargs = _detector_optimizer_kwargs(
        args, clustering_config=clustering_config, max_workers=max_workers
    )
    optimizer = CachedCoordinateDescentOptimizer(**optimizer_kwargs)
    optimizer.include_recidivism = args.include_recidivism

    logging.info("Starting coordinate descent optimization on timeframe market set...")
    best_config, results_df, detector_summaries = optimizer.optimize(
        loader=loader,
        market_ids=prep.market_ids,
        n_passes=args.n_passes,
        optimize_order=DETECTOR_OPTIMIZE_ORDER,
    )

    combined_config = {
        **best_config,
        "clustering_config": clustering_config,
        "jump_anticipation_config": ja_config,
        "timeframe_optimization_meta": _build_timeframe_meta(
            args,
            prep,
            extra={
                "max_workers": max_workers,
                "temporary_script": True,
            },
        ),
    }

    out_base = prep.out_base
    ts = prep.ts
    best_config_path = out_base / f"timeframe_best_config_{ts}.json"
    results_path = out_base / f"timeframe_coordinate_descent_{ts}.csv"
    summaries_path = out_base / f"timeframe_detector_summaries_{ts}.json"
    inferred_res_path = out_base / f"timeframe_inferred_resolutions_{ts}.json"

    _write_json(best_config_path, combined_config)
    results_df.to_csv(results_path, index=False)
    _write_json(summaries_path, detector_summaries)
    _dump_inferred_resolutions(inferred_res_path, prep.inferred_winners)

    return {
        "best_config_path": best_config_path,
        "results_path": results_path,
        "summaries_path": summaries_path,
        "inferred_res_path": inferred_res_path,
        "best_row": _best_row_from_df(results_df),
        "max_workers": max_workers,
    }

@dataclass
class _AlternatingTrajectory:
    detector_config: Dict[str, Any]
    clustering_config: Optional[Dict[str, Any]]
    detector_df: pd.DataFrame
    clustering_df: pd.DataFrame
    detector_summaries: Dict[str, Any]
    clustering_summaries: Dict[str, Any]
    final_objective: float
    ja_config: Optional[Dict[str, Any]] = None
    ja_df: Optional[pd.DataFrame] = None


def _run_alternating_trajectory(
    loader: HistoricalDataLoader,
    prep: TimeframePrepResult,
    args: Any,
    *,
    initial_detector_config: Dict[str, Any],
    initial_clustering_config: Optional[Dict[str, Any]],
    initial_ja_config: Optional[Dict[str, Any]] = None,
    market_ids: Optional[List[int]] = None,
    detector_order: Sequence[str] = DETECTOR_OPTIMIZE_ORDER,
    clustering_order: Sequence[str] = CLUSTERING_OPTIMIZE_ORDER,
    max_workers: int,
    clustering_workers: int,
    detector_optimizer_attributes: Optional[Dict[str, Any]] = None,
) -> _AlternatingTrajectory:
    """Run alternating optimizer passes from a supplied starting point."""
    detector_optimizer_attributes = dict(detector_optimizer_attributes or {})
    current_detector_config = deepcopy(initial_detector_config)
    current_clustering_config = deepcopy(initial_clustering_config)
    current_ja_config = deepcopy(initial_ja_config) if initial_ja_config is not None else None
    effective_market_ids = list(market_ids) if market_ids is not None else list(prep.market_ids)

    optimize_ja = bool(
        getattr(args, "enable_ja_optimization", False)
        and getattr(args, "enable_jump_anticipation", False)
    )
    clustering_enabled = bool(getattr(args, "enable_clustering", True))

    detector_pass_rows: List[pd.DataFrame] = []
    clustering_pass_rows: List[pd.DataFrame] = []
    ja_pass_rows: List[pd.DataFrame] = []
    detector_summaries: Dict[str, Any] = {}
    clustering_summaries: Dict[str, Any] = {}
    final_objective = float("-inf")

    for pass_idx in range(1, args.n_passes + 1):
        logging.info("\n" + "#" * 80)
        logging.info(
            "ALTERNATING PASS %s/%s — STAGE 1 DETECTORS", pass_idx, args.n_passes
        )
        logging.info("#" * 80)

        detector_optimizer = CachedCoordinateDescentOptimizer(
            **_detector_optimizer_kwargs(
                args,
                clustering_config=current_clustering_config,
                max_workers=max_workers,
                ja_config_override=current_ja_config,
            )
        )
        detector_optimizer.include_recidivism = args.include_recidivism
        for attr, value in detector_optimizer_attributes.items():
            setattr(detector_optimizer, attr, value)

        current_detector_config, detector_df, det_summaries = detector_optimizer.optimize(
            loader=loader,
            market_ids=effective_market_ids,
            n_passes=1,
            optimize_order=list(detector_order),
            initial_config=current_detector_config,
        )
        detector_df = detector_df.copy()
        detector_df["unified_pass"] = pass_idx
        detector_pass_rows.append(detector_df)
        detector_summaries = det_summaries

        if clustering_enabled:
            logging.info("\n" + "#" * 80)
            logging.info(
                "ALTERNATING PASS %s/%s — STAGE 2 CLUSTERING", pass_idx, args.n_passes
            )
            logging.info("#" * 80)

            clustering_optimizer = ClusteringOptimizer(
                **_clustering_optimizer_kwargs(
                    args,
                    detector_config=current_detector_config,
                    max_workers=clustering_workers,
                    ja_config_override=current_ja_config,
                )
            )

            (
                current_clustering_config,
                clustering_df,
                clust_summaries,
            ) = clustering_optimizer.optimize(
                loader=loader,
                market_ids=effective_market_ids,
                n_passes=1,
                optimize_order=list(clustering_order),
                initial_config=current_clustering_config,
            )
            clustering_df = clustering_df.copy()
            clustering_df["unified_pass"] = pass_idx
            clustering_pass_rows.append(clustering_df)
            clustering_summaries = clust_summaries

            # Track best final objective from the last pass' clustering results
            if "objective_score" in clustering_df.columns and not clustering_df.empty:
                pass_best = float(clustering_df["objective_score"].max())
                if pass_best > final_objective:
                    final_objective = pass_best
        else:
            logging.info(
                "ALTERNATING PASS %s/%s — STAGE 2 CLUSTERING skipped "
                "(--disable-clustering)", pass_idx, args.n_passes,
            )
            # Fall back to the detector stage's best score so final_objective
            # still tracks progress when clustering is the usual source of
            # the pass-level objective.
            if (
                "objective_score" in detector_df.columns
                and not detector_df.empty
            ):
                pass_best_det = float(detector_df["objective_score"].max())
                if pass_best_det > final_objective:
                    final_objective = pass_best_det

        if optimize_ja:
            logging.info("\n" + "#" * 80)
            logging.info(
                "ALTERNATING PASS %s/%s — STAGE 3 JUMP ANTICIPATION",
                pass_idx,
                args.n_passes,
            )
            logging.info("#" * 80)

            try:
                current_ja_config, ja_df = _run_ja_optimization_for_markets(
                    loader,
                    args,
                    detector_config=current_detector_config,
                    clustering_config=current_clustering_config,
                    initial_ja_config=current_ja_config,
                    market_ids=effective_market_ids,
                )
                ja_df = ja_df.copy()
                ja_df["unified_pass"] = pass_idx
                ja_pass_rows.append(ja_df)
                if "objective_score" in ja_df.columns and not ja_df.empty:
                    pass_best_ja = float(ja_df["objective_score"].max())
                    if pass_best_ja > final_objective:
                        final_objective = pass_best_ja
            except Exception:
                logging.exception(
                    "JA optimisation stage failed in pass %d; keeping previous "
                    "ja_config and continuing",
                    pass_idx,
                )

    detector_df_all = (
        pd.concat(detector_pass_rows, ignore_index=True) if detector_pass_rows else pd.DataFrame()
    )
    clustering_df_all = (
        pd.concat(clustering_pass_rows, ignore_index=True)
        if clustering_pass_rows
        else pd.DataFrame()
    )
    ja_df_all = (
        pd.concat(ja_pass_rows, ignore_index=True) if ja_pass_rows else None
    )

    return _AlternatingTrajectory(
        detector_config=current_detector_config,
        clustering_config=current_clustering_config,
        detector_df=detector_df_all,
        clustering_df=clustering_df_all,
        detector_summaries=detector_summaries,
        clustering_summaries=clustering_summaries,
        final_objective=final_objective,
        ja_config=current_ja_config,
        ja_df=ja_df_all,
    )


def run_alternating_detectors_clustering_timeframe(
    loader: HistoricalDataLoader,
    prep: TimeframePrepResult,
    args: Any,
) -> Dict[str, Any]:
    """Single-start alternating detector-then-clustering optimisation, kept
    backwards-compatible with the original experiment script output layout.
    """
    if not prep.market_ids:
        raise RuntimeError("No inferred-resolved markets in timeframe; cannot optimize.")

    max_workers = _resolve_max_workers(args, cap=8)
    clustering_workers = 1 if args.enable_layer2_attribution else max_workers
    ja_config = CONFIG.get("jump_anticipation_config") if args.enable_jump_anticipation else None
    optimize_ja = bool(
        getattr(args, "enable_ja_optimization", False)
        and getattr(args, "enable_jump_anticipation", False)
    )

    baseline_detector_config = ParameterGrid.get_baseline_config()
    baseline_clustering_config = (
        CONFIG.get(
            "clustering_config", ClusteringParameterGrid.get_baseline_config()
        )
        if args.enable_clustering
        else None
    )
    initial_ja_config: Optional[Dict[str, Any]] = None
    if args.enable_jump_anticipation:
        if optimize_ja:
            from jump_anticipation.optimizer import get_ja_baseline_config
            initial_ja_config = get_ja_baseline_config()
        else:
            initial_ja_config = ja_config

    trajectory = _run_alternating_trajectory(
        loader=loader,
        prep=prep,
        args=args,
        initial_detector_config=baseline_detector_config,
        initial_clustering_config=baseline_clustering_config,
        initial_ja_config=initial_ja_config,
        max_workers=max_workers,
        clustering_workers=clustering_workers,
    )

    final_ja_config = (
        trajectory.ja_config if optimize_ja and trajectory.ja_config is not None
        else ja_config
    )
    combined_config = {
        **trajectory.detector_config,
        "clustering_config": trajectory.clustering_config,
        "jump_anticipation_config": final_ja_config,
        "timeframe_optimization_meta": _build_timeframe_meta(
            args,
            prep,
            extra={
                "max_workers": max_workers,
                "clustering_workers": clustering_workers,
                "temporary_script": True,
                "alternating_passes": True,
                "layer2_attribution": args.enable_layer2_attribution,
                "jump_anticipation_enabled": bool(final_ja_config),
                "jump_anticipation_optimized": optimize_ja,
                "ja_n_passes": int(getattr(args, "ja_n_passes", 1) or 1) if optimize_ja else None,
            },
        ),
    }

    out_base = prep.out_base
    ts = prep.ts
    best_config_path = out_base / f"timeframe_best_config_det_clust_{ts}.json"
    detector_results_path = out_base / f"timeframe_detector_results_det_clust_{ts}.csv"
    clustering_results_path = out_base / f"timeframe_clustering_results_det_clust_{ts}.csv"
    inferred_res_path = out_base / f"timeframe_inferred_resolutions_det_clust_{ts}.json"
    detector_summaries_path = out_base / f"timeframe_detector_summaries_det_clust_{ts}.json"
    clustering_summaries_path = out_base / f"timeframe_clustering_summaries_det_clust_{ts}.json"

    _write_json(best_config_path, combined_config)
    if not trajectory.detector_df.empty:
        trajectory.detector_df.to_csv(detector_results_path, index=False)
    if not trajectory.clustering_df.empty:
        trajectory.clustering_df.to_csv(clustering_results_path, index=False)
    _dump_inferred_resolutions(inferred_res_path, prep.inferred_winners)
    _write_json(detector_summaries_path, trajectory.detector_summaries)
    _write_json(clustering_summaries_path, trajectory.clustering_summaries)

    return {
        "best_config_path": best_config_path,
        "detector_results_path": detector_results_path,
        "clustering_results_path": clustering_results_path,
        "inferred_res_path": inferred_res_path,
        "detector_summaries_path": detector_summaries_path,
        "clustering_summaries_path": clustering_summaries_path,
        "max_workers": max_workers,
        "clustering_workers": clustering_workers,
        "final_objective": trajectory.final_objective,
    }

def run_multi_start_coordinate_descent_timeframe(
    loader: HistoricalDataLoader,
    prep: TimeframePrepResult,
    args: Any,
) -> Dict[str, Any]:
    """Multi-start coordinate descent over the detector parameter space."""
    if not prep.market_ids:
        raise RuntimeError("No inferred-resolved markets in timeframe; cannot optimize.")

    clustering_config = CONFIG.get("clustering_config") if args.enable_clustering else None
    ja_config = CONFIG.get("jump_anticipation_config") if args.enable_jump_anticipation else None
    max_workers = _resolve_max_workers(args, cap=12)

    train_market_ids, val_market_ids = time_ordered_train_val_split(
        loader,
        prep.market_ids,
        val_fraction=float(getattr(args, "val_fraction", 0.0) or 0.0),
    )

    optimizer_kwargs = _detector_optimizer_kwargs(
        args, clustering_config=clustering_config, max_workers=max_workers
    )

    multi_start_kwargs = _multi_start_kwargs(args)
    multi_start = MultiStartCoordinateDescentOptimizer(
        **multi_start_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_attributes={"include_recidivism": args.include_recidivism},
    )

    logging.info(
        "Starting multi-start coordinate descent (n_starts=%d, strategy=%s) "
        "on %d train markets (val=%d markets)...",
        multi_start.n_starts,
        multi_start.start_strategy,
        len(train_market_ids),
        len(val_market_ids),
    )
    best_config, combined_df, ms_summary = multi_start.optimize(
        loader=loader,
        market_ids=train_market_ids,
        val_market_ids=val_market_ids,
        n_passes=args.n_passes,
        optimize_order=DETECTOR_OPTIMIZE_ORDER,
    )

    combined_config = {
        **best_config,
        "clustering_config": clustering_config,
        "jump_anticipation_config": ja_config,
        "timeframe_optimization_meta": _build_timeframe_meta(
            args,
            prep,
            extra={
                "max_workers": max_workers,
                "multi_start": {
                    "n_starts": ms_summary["n_starts_completed"],
                    "start_strategy": ms_summary["start_strategy"],
                    "perturb_prob": ms_summary["perturb_prob"],
                    "shuffle_order": ms_summary["shuffle_order"],
                    "include_baseline_start": ms_summary["include_baseline_start"],
                    "random_seed": ms_summary["random_seed"],
                    "best_start_idx": ms_summary["best_start_idx"],
                    "best_final_objective": ms_summary["best_final_objective"],
                    "median_final_objective": ms_summary["median_final_objective"],
                    "final_objective_spread": ms_summary["final_objective_spread"],
                    "total_elapsed_seconds": ms_summary["total_elapsed_seconds"],
                    "used_validation": ms_summary["used_validation"],
                    "n_train_markets": ms_summary["n_train_markets"],
                    "n_val_markets": ms_summary["n_val_markets"],
                    "val_fraction": float(getattr(args, "val_fraction", 0.0) or 0.0),
                    "train_val_gap_mean": ms_summary["train_val_gap_mean"],
                    "train_val_gap_max": ms_summary["train_val_gap_max"],
                    "train_objective_aggregates": ms_summary["train_objective_aggregates"],
                    "val_objective_aggregates": ms_summary["val_objective_aggregates"],
                },
            },
        ),
    }

    out_base = prep.out_base
    ts = prep.ts
    best_config_path = out_base / f"timeframe_best_config_multi_start_{ts}.json"
    results_path = out_base / f"timeframe_multi_start_{ts}.csv"
    summaries_path = out_base / f"timeframe_multi_start_summary_{ts}.json"
    inferred_res_path = out_base / f"timeframe_inferred_resolutions_multi_start_{ts}.json"

    _write_json(best_config_path, combined_config)
    combined_df.to_csv(results_path, index=False)
    _write_json(summaries_path, ms_summary)
    _dump_inferred_resolutions(inferred_res_path, prep.inferred_winners)

    return {
        "best_config_path": best_config_path,
        "results_path": results_path,
        "summaries_path": summaries_path,
        "inferred_res_path": inferred_res_path,
        "best_row": _best_row_from_df(combined_df),
        "max_workers": max_workers,
        "multi_start_summary": ms_summary,
    }

def _load_resume_json(
    path: str,
    *,
    expected_n_starts: int,
    expected_seed: int,
    expected_strategy: str,
    expected_shuffle_order: bool,
    expected_include_baseline_start: bool,
) -> Dict[int, Dict[str, Any]]:
    """Load a reconstruction JSON produced by
    :mod:`experiments.recover_multi_start_det_clust` and return a
    ``{start_idx: entry}`` mapping for the completed starts.

    Raises ``RuntimeError`` when the JSON's multi-start knobs disagree with
    the current invocation — any mismatch would silently break the RNG
    reproduction contract that makes this resume valid.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Strict knob matching: anything that affects _generate_starts or the
    # per-start shuffle MUST match byte-for-byte.
    if int(payload.get("n_starts", -1)) != int(expected_n_starts):
        raise RuntimeError(
            f"resume-from-json n_starts={payload.get('n_starts')} != "
            f"invocation --n-starts={expected_n_starts}"
        )
    if int(payload.get("random_seed", -1)) != int(expected_seed):
        raise RuntimeError(
            f"resume-from-json random_seed={payload.get('random_seed')} != "
            f"invocation --start-seed={expected_seed}"
        )
    if str(payload.get("start_strategy", "")) != str(expected_strategy):
        raise RuntimeError(
            f"resume-from-json start_strategy={payload.get('start_strategy')!r} != "
            f"invocation --start-strategy={expected_strategy!r}"
        )
    if bool(payload.get("shuffle_order", True)) != bool(expected_shuffle_order):
        raise RuntimeError(
            f"resume-from-json shuffle_order={payload.get('shuffle_order')} != "
            f"invocation shuffle_order={expected_shuffle_order}"
        )
    if bool(payload.get("include_baseline_start", True)) != bool(expected_include_baseline_start):
        raise RuntimeError(
            f"resume-from-json include_baseline_start={payload.get('include_baseline_start')} != "
            f"invocation include_baseline_start={expected_include_baseline_start}"
        )

    by_idx: Dict[int, Dict[str, Any]] = {}
    for entry in payload.get("per_start", []):
        by_idx[int(entry["start_idx"])] = entry
    return by_idx


def _stub_trajectory_from_resume(
    entry: Dict[str, Any],
) -> "_AlternatingTrajectory":
    """Build an ``_AlternatingTrajectory`` with empty DataFrames carrying just
    the reconstructed final configs.  Used so the rest of the runner can
    treat reconstructed starts uniformly with freshly-executed ones when a
    reconstructed start turns out to be the overall winner."""
    train_obj = float(entry.get("train_final_objective", float("nan")))
    return _AlternatingTrajectory(
        detector_config=deepcopy(entry["final_detector_config"]),
        clustering_config=deepcopy(entry["final_clustering_config"]),
        detector_df=pd.DataFrame(),
        clustering_df=pd.DataFrame(),
        detector_summaries={},
        clustering_summaries={},
        final_objective=train_obj,
    )


def _evaluate_alternating_trajectory_on_markets(
    loader: HistoricalDataLoader,
    args: Any,
    *,
    detector_config: Dict[str, Any],
    clustering_config: Dict[str, Any],
    market_ids: List[int],
    max_workers: int,
    ja_config: Any = "UNSET",
) -> Tuple[float, Dict[str, Any], float]:
    """Score a finalised ``(detector_config, clustering_config)`` pair on an
    independent market set.  Returns ``(objective, metrics_dict, elapsed)``.
    """
    start = time.time()
    optimizer = CachedCoordinateDescentOptimizer(
        **_detector_optimizer_kwargs(
            args,
            clustering_config=clustering_config,
            max_workers=max_workers,
            ja_config_override=ja_config,
        )
    )
    optimizer.include_recidivism = args.include_recidivism
    eval_result = optimizer._evaluate_config(
        config=detector_config,
        loader=loader,
        market_ids=market_ids,
    )
    metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
    objective = float(metrics.get(args.objective, float("nan")))
    return objective, metrics, time.time() - start


def run_multi_start_alternating_timeframe(
    loader: HistoricalDataLoader,
    prep: TimeframePrepResult,
    args: Any,
) -> Dict[str, Any]:
    """Run alternating detector and clustering optimization from multiple starts."""
    if not prep.market_ids:
        raise RuntimeError("No inferred-resolved markets in timeframe; cannot optimize.")

    max_workers = _resolve_max_workers(args, cap=8)
    clustering_workers = 1 if args.enable_layer2_attribution else max_workers
    ja_config = CONFIG.get("jump_anticipation_config") if args.enable_jump_anticipation else None
    optimize_ja = bool(
        getattr(args, "enable_ja_optimization", False)
        and getattr(args, "enable_jump_anticipation", False)
    )

    train_market_ids, val_market_ids = time_ordered_train_val_split(
        loader,
        prep.market_ids,
        val_fraction=float(getattr(args, "val_fraction", 0.0) or 0.0),
    )
    use_val = bool(val_market_ids)

    baseline_detector_config = ParameterGrid.get_baseline_config()
    baseline_clustering_config = (
        CONFIG.get(
            "clustering_config", ClusteringParameterGrid.get_baseline_config()
        )
        if args.enable_clustering
        else None
    )
    baseline_ja_config: Optional[Dict[str, Any]] = None
    if args.enable_jump_anticipation:
        if optimize_ja:
            from jump_anticipation.optimizer import get_ja_baseline_config
            baseline_ja_config = get_ja_baseline_config()
        else:
            baseline_ja_config = ja_config

    n_starts = int(getattr(args, "n_starts", 1) or 1)
    start_strategy = str(getattr(args, "start_strategy", "perturb"))
    perturb_prob = float(getattr(args, "perturb_prob", 0.3))
    include_baseline_start = bool(getattr(args, "include_baseline_start", True))
    shuffle_order = bool(getattr(args, "shuffle_order", True))
    seed = int(getattr(args, "start_seed", 42))

    rng = random.Random(seed)

    # Pre-generate the starting detector configs via the same machinery the
    # reusable optimiser uses.
    msopt = MultiStartCoordinateDescentOptimizer(
        n_starts=n_starts,
        start_strategy=start_strategy,
        perturb_prob=perturb_prob,
        include_baseline_start=include_baseline_start,
        shuffle_order=shuffle_order,
        random_seed=seed,
    )
    starts = msopt._generate_starts(baseline_detector_config, rng)

    # Resume support
    resume_skip_before = int(getattr(args, "resume_skip_starts_before_idx", 0) or 0)
    resume_from_json = getattr(args, "resume_from_json", None)
    if resume_skip_before < 0:
        raise ValueError(
            f"--resume-skip-starts-before-idx must be >= 0, got {resume_skip_before}"
        )
    if resume_skip_before > 0 and not resume_from_json:
        raise ValueError(
            "--resume-skip-starts-before-idx > 0 requires --resume-from-json"
        )

    resume_entries: Dict[int, Dict[str, Any]] = {}
    if resume_skip_before > 0:
        resume_entries = _load_resume_json(
            str(resume_from_json),
            expected_n_starts=n_starts,
            expected_seed=seed,
            expected_strategy=start_strategy,
            expected_shuffle_order=shuffle_order,
            expected_include_baseline_start=include_baseline_start,
        )
        missing = [i for i in range(resume_skip_before) if i not in resume_entries]
        if missing:
            raise RuntimeError(
                f"resume-from-json is missing entries for skipped start indices {missing}; "
                "cannot resume without the full prefix."
            )
        # Validate initial_detector_config for each skipped start against the
        # replayed start config; catches grid/baseline drift immediately.
        for i in range(resume_skip_before):
            replayed_label, replayed_strategy, replayed_initial = starts[i]
            entry = resume_entries[i]
            if entry["label"] != replayed_label:
                raise RuntimeError(
                    f"Resume start {i}: label mismatch json='{entry['label']}' replayed='{replayed_label}'"
                )
            if entry["strategy"] != replayed_strategy:
                raise RuntimeError(
                    f"Resume start {i}: strategy mismatch json='{entry['strategy']}' replayed='{replayed_strategy}'"
                )
            if entry.get("initial_detector_config") != replayed_initial:
                raise RuntimeError(
                    f"Resume start {i} ({replayed_label}): initial_detector_config mismatch "
                    "between JSON and replayed start — grid/baseline drifted since the original run."
                )
        logging.info(
            "RESUME MODE: will synthesise per-start entries for indices [0..%d) from %s "
            "and execute indices [%d..%d).",
            resume_skip_before,
            resume_from_json,
            resume_skip_before,
            len(starts),
        )

    logging.info("\n" + "=" * 80)
    logging.info(
        "MULTI-START ALTERNATING DETECTOR+CLUSTERING | n_starts=%d | strategy=%s "
        "| train_markets=%d | val_markets=%d",
        len(starts),
        start_strategy,
        len(train_market_ids),
        len(val_market_ids),
    )
    if use_val:
        logging.info("Trajectory ranking: VALIDATION objective.")
    else:
        logging.info(
            "No validation set (val_fraction=0 or too few markets); ranking on TRAIN."
        )
    logging.info("=" * 80)

    best_trajectory: Optional[_AlternatingTrajectory] = None
    best_score = float("-inf")
    best_start_idx = -1
    best_label = ""
    per_start: List[Dict[str, Any]] = []
    total_start = time.time()

    for start_idx, (label, strategy, start_detector_config) in enumerate(starts):
        detector_order = list(DETECTOR_OPTIMIZE_ORDER)
        if shuffle_order and start_idx > 0 and not (include_baseline_start and start_idx == 0):
            rng.shuffle(detector_order)

        if start_idx < resume_skip_before:
            entry = resume_entries[start_idx]
            logged_order = entry.get("detector_order_replayed") or entry.get("detector_order_logged")
            if logged_order != detector_order:
                raise RuntimeError(
                    f"Resume start {start_idx} ({label}): detector_order mismatch.\n"
                    f"  json:     {logged_order}\n"
                    f"  replayed: {detector_order}\n"
                    "RNG reproduction failed — aborting."
                )
            train_obj = float(entry.get("train_final_objective", float("nan")))
            val_obj = float(entry.get("val_final_objective", float("nan")))
            used_val_entry = bool(entry.get("used_validation", False))
            ranking_obj = val_obj if (used_val_entry and val_obj == val_obj) else train_obj

            per_start.append({
                "start_idx": start_idx,
                "label": label,
                "strategy": strategy,
                "detector_order": detector_order,
                "train_final_objective": train_obj,
                "val_final_objective": val_obj,
                "final_objective": ranking_obj,
                "train_val_gap": (
                    train_obj - val_obj if (used_val_entry and val_obj == val_obj) else float("nan")
                ),
                "elapsed_seconds": float(entry.get("train_elapsed_seconds", 0.0)),
                "val_elapsed_seconds": float(entry.get("val_elapsed_seconds", 0.0)),
                "val_final_metrics": {},
                "reconstructed": True,
                "reconstruction_source": str(resume_from_json),
            })
            logging.info("\n" + "#" * 80)
            logging.info(
                "Alternating multi-start run %d/%d | %s (strategy=%s)  [RESUMED from JSON]",
                start_idx + 1,
                len(starts),
                label,
                strategy,
            )
            logging.info("  detector_order: %s", detector_order)
            logging.info(
                "  -> %s: train_obj=%.4f  val_obj=%.4f  (reconstructed; not re-executed)",
                label,
                train_obj,
                val_obj,
            )
            logging.info("#" * 80)

            if ranking_obj == ranking_obj and ranking_obj > best_score:
                best_score = ranking_obj
                best_trajectory = _stub_trajectory_from_resume(entry)
                best_start_idx = start_idx
                best_label = label
            continue

        logging.info("\n" + "#" * 80)
        logging.info(
            "Alternating multi-start run %d/%d | %s (strategy=%s)",
            start_idx + 1,
            len(starts),
            label,
            strategy,
        )
        logging.info("  detector_order: %s", detector_order)
        logging.info("#" * 80)

        traj_start = time.time()
        try:
            trajectory = _run_alternating_trajectory(
                loader=loader,
                prep=prep,
                args=args,
                initial_detector_config=start_detector_config,
                initial_clustering_config=baseline_clustering_config,
                initial_ja_config=baseline_ja_config,
                market_ids=train_market_ids,
                detector_order=detector_order,
                max_workers=max_workers,
                clustering_workers=clustering_workers,
            )
        except Exception:
            logging.exception(
                "Alternating multi-start trajectory %d (%s) failed; skipping",
                start_idx,
                label,
            )
            continue
        train_elapsed = time.time() - traj_start

        train_obj = trajectory.final_objective

        val_obj = float("nan")
        val_metrics: Dict[str, Any] = {}
        val_elapsed = 0.0
        if use_val:
            try:
                val_obj, val_metrics, val_elapsed = (
                    _evaluate_alternating_trajectory_on_markets(
                        loader,
                        args,
                        detector_config=trajectory.detector_config,
                        clustering_config=trajectory.clustering_config,
                        market_ids=val_market_ids,
                        max_workers=max_workers,
                        ja_config=trajectory.ja_config if optimize_ja else "UNSET",
                    )
                )
            except Exception:
                logging.exception(
                    "Validation pass failed for alternating trajectory %d (%s); "
                    "falling back to train objective for this start",
                    start_idx,
                    label,
                )

        ranking_obj = val_obj if (use_val and val_obj == val_obj) else train_obj

        per_start.append({
            "start_idx": start_idx,
            "label": label,
            "strategy": strategy,
            "detector_order": detector_order,
            "train_final_objective": train_obj,
            "val_final_objective": val_obj,
            "final_objective": ranking_obj,
            "train_val_gap": (
                train_obj - val_obj if (use_val and val_obj == val_obj) else float("nan")
            ),
            "elapsed_seconds": train_elapsed,
            "val_elapsed_seconds": val_elapsed,
            "val_final_metrics": val_metrics,
            "reconstructed": False,
            "final_ja_config": trajectory.ja_config if optimize_ja else None,
        })

        if use_val:
            logging.info(
                "  -> %s: train_obj=%.4f  val_obj=%.4f  gap=%+.4f  (%.1fs train + %.1fs val)",
                label,
                train_obj,
                val_obj,
                train_obj - val_obj if val_obj == val_obj else float("nan"),
                train_elapsed,
                val_elapsed,
            )
        else:
            logging.info(
                "  -> %s: final_obj=%.4f  (%.1fs)",
                label,
                train_obj,
                train_elapsed,
            )

        if ranking_obj == ranking_obj and ranking_obj > best_score:
            best_score = ranking_obj
            best_trajectory = trajectory
            best_start_idx = start_idx
            best_label = label

    if best_trajectory is None:
        raise RuntimeError("All alternating multi-start trajectories failed; see logs.")

    total_elapsed = time.time() - total_start

    winner_reconstructed = bool(
        best_trajectory.detector_df.empty and best_trajectory.clustering_df.empty
    )

    combined_detector_df = best_trajectory.detector_df.copy()
    combined_clustering_df = best_trajectory.clustering_df.copy()
    if not combined_detector_df.empty:
        combined_detector_df["winning_start"] = best_label
    if not combined_clustering_df.empty:
        combined_clustering_df["winning_start"] = best_label

    train_objs = [s["train_final_objective"] for s in per_start if s["train_final_objective"] == s["train_final_objective"]]
    val_objs = [s["val_final_objective"] for s in per_start if s["val_final_objective"] == s["val_final_objective"]]
    gaps = [s["train_val_gap"] for s in per_start if s["train_val_gap"] == s["train_val_gap"]]

    ms_summary = {
        "n_starts_requested": n_starts,
        "n_starts_completed": len(per_start),
        "start_strategy": start_strategy,
        "perturb_prob": perturb_prob,
        "shuffle_order": shuffle_order,
        "include_baseline_start": include_baseline_start,
        "random_seed": seed,
        "best_start_idx": best_start_idx,
        "best_label": best_label,
        "best_final_objective": best_score,
        "used_validation": use_val,
        "n_train_markets": len(train_market_ids),
        "n_val_markets": len(val_market_ids),
        "val_fraction": float(getattr(args, "val_fraction", 0.0) or 0.0),
        "train_objective_aggregates": {
            "min": min(train_objs) if train_objs else float("nan"),
            "median": float(pd.Series(train_objs).median()) if train_objs else float("nan"),
            "max": max(train_objs) if train_objs else float("nan"),
            "spread": (max(train_objs) - min(train_objs)) if train_objs else float("nan"),
        },
        "val_objective_aggregates": {
            "min": min(val_objs) if val_objs else float("nan"),
            "median": float(pd.Series(val_objs).median()) if val_objs else float("nan"),
            "max": max(val_objs) if val_objs else float("nan"),
            "spread": (max(val_objs) - min(val_objs)) if val_objs else float("nan"),
        },
        "train_val_gap_mean": float(pd.Series(gaps).mean()) if gaps else float("nan"),
        "train_val_gap_max": max(gaps) if gaps else float("nan"),
        "total_elapsed_seconds": total_elapsed,
        "per_start": per_start,
        "resume_skip_starts_before_idx": resume_skip_before,
        "resume_from_json": str(resume_from_json) if resume_from_json else None,
        "reconstructed_winner": winner_reconstructed,
    }

    logging.info("\n" + "=" * 80)
    logging.info("ALTERNATING MULTI-START SUMMARY")
    logging.info("=" * 80)
    logging.info(
        "Completed %d/%d starts in %.1fs  |  train_markets=%d  val_markets=%d",
        len(per_start), n_starts, total_elapsed, len(train_market_ids), len(val_market_ids),
    )
    if use_val:
        logging.info(
            "Train across starts: best=%.4f median=%.4f min=%.4f spread=%.4f",
            ms_summary["train_objective_aggregates"]["max"],
            ms_summary["train_objective_aggregates"]["median"],
            ms_summary["train_objective_aggregates"]["min"],
            ms_summary["train_objective_aggregates"]["spread"],
        )
        logging.info(
            "Val   across starts: best=%.4f median=%.4f min=%.4f spread=%.4f",
            ms_summary["val_objective_aggregates"]["max"],
            ms_summary["val_objective_aggregates"]["median"],
            ms_summary["val_objective_aggregates"]["min"],
            ms_summary["val_objective_aggregates"]["spread"],
        )
        logging.info(
            "Train-val gap: mean=%+.4f max=%+.4f  (positive = trajectories overfit on train)",
            ms_summary["train_val_gap_mean"], ms_summary["train_val_gap_max"],
        )
    logging.info(
        "Winner: start_idx=%d (%s) @ ranking_obj=%.4f (selected on %s)",
        best_start_idx, best_label, best_score, "VAL" if use_val else "TRAIN",
    )
    for s in sorted(per_start, key=lambda x: -x["final_objective"]):
        if use_val and s["val_final_objective"] == s["val_final_objective"]:
            logging.info(
                "  [%2d] %-26s %-8s  train=%.4f  val=%.4f  gap=%+.4f  (%.1fs)",
                s["start_idx"], s["label"], s["strategy"],
                s["train_final_objective"], s["val_final_objective"], s["train_val_gap"],
                s["elapsed_seconds"] + s.get("val_elapsed_seconds", 0.0),
            )
        else:
            logging.info(
                "  [%2d] %-26s %-8s  final=%.4f  (%.1fs)",
                s["start_idx"], s["label"], s["strategy"],
                s["final_objective"], s["elapsed_seconds"],
            )

    final_ja_config = (
        best_trajectory.ja_config if optimize_ja and best_trajectory.ja_config is not None
        else ja_config
    )
    combined_config = {
        **best_trajectory.detector_config,
        "clustering_config": best_trajectory.clustering_config,
        "jump_anticipation_config": final_ja_config,
        "timeframe_optimization_meta": _build_timeframe_meta(
            args,
            prep,
            extra={
                "max_workers": max_workers,
                "clustering_workers": clustering_workers,
                "alternating_passes": True,
                "layer2_attribution": args.enable_layer2_attribution,
                "jump_anticipation_enabled": bool(final_ja_config),
                "jump_anticipation_optimized": optimize_ja,
                "ja_n_passes": int(getattr(args, "ja_n_passes", 1) or 1) if optimize_ja else None,
                "multi_start": {
                    "n_starts": ms_summary["n_starts_completed"],
                    "start_strategy": ms_summary["start_strategy"],
                    "perturb_prob": ms_summary["perturb_prob"],
                    "shuffle_order": ms_summary["shuffle_order"],
                    "include_baseline_start": ms_summary["include_baseline_start"],
                    "random_seed": ms_summary["random_seed"],
                    "best_start_idx": ms_summary["best_start_idx"],
                    "best_final_objective": ms_summary["best_final_objective"],
                    "total_elapsed_seconds": ms_summary["total_elapsed_seconds"],
                    "used_validation": ms_summary["used_validation"],
                    "n_train_markets": ms_summary["n_train_markets"],
                    "n_val_markets": ms_summary["n_val_markets"],
                    "val_fraction": ms_summary["val_fraction"],
                    "train_val_gap_mean": ms_summary["train_val_gap_mean"],
                    "train_val_gap_max": ms_summary["train_val_gap_max"],
                    "train_objective_aggregates": ms_summary["train_objective_aggregates"],
                    "val_objective_aggregates": ms_summary["val_objective_aggregates"],
                    "reconstructed_winner": winner_reconstructed,
                    "resume_skip_starts_before_idx": resume_skip_before,
                    "resume_from_json": str(resume_from_json) if resume_from_json else None,
                },
            },
        ),
    }

    out_base = prep.out_base
    ts = prep.ts
    best_config_path = out_base / f"timeframe_best_config_multi_start_det_clust_{ts}.json"
    detector_results_path = out_base / f"timeframe_detector_results_multi_start_det_clust_{ts}.csv"
    clustering_results_path = out_base / f"timeframe_clustering_results_multi_start_det_clust_{ts}.csv"
    inferred_res_path = out_base / f"timeframe_inferred_resolutions_multi_start_det_clust_{ts}.json"
    detector_summaries_path = out_base / f"timeframe_detector_summaries_multi_start_det_clust_{ts}.json"
    clustering_summaries_path = out_base / f"timeframe_clustering_summaries_multi_start_det_clust_{ts}.json"
    ms_summary_path = out_base / f"timeframe_multi_start_summary_det_clust_{ts}.json"
    winner_eval_path = out_base / f"timeframe_winner_evaluation_{ts}.json"
    ja_results_path = out_base / f"timeframe_ja_results_multi_start_det_clust_{ts}.csv"

    _write_json(best_config_path, combined_config)
    _dump_inferred_resolutions(inferred_res_path, prep.inferred_winners)

    # Per-candidate artefacts only exist when the winner was executed in
    # this process; reconstructed winners have empty stub DataFrames +
    # empty summary dicts and we skip writing those placeholder files to
    # avoid polluting downstream consumers that expect populated results.
    ja_df_written = False
    if not winner_reconstructed:
        if not combined_detector_df.empty:
            combined_detector_df.to_csv(detector_results_path, index=False)
        if not combined_clustering_df.empty:
            combined_clustering_df.to_csv(clustering_results_path, index=False)
        _write_json(detector_summaries_path, best_trajectory.detector_summaries)
        _write_json(clustering_summaries_path, best_trajectory.clustering_summaries)
        if (
            optimize_ja
            and best_trajectory.ja_df is not None
            and not best_trajectory.ja_df.empty
        ):
            winning_ja_df = best_trajectory.ja_df.copy()
            winning_ja_df["winning_start"] = best_label
            winning_ja_df.to_csv(ja_results_path, index=False)
            ja_df_written = True

    winner_evaluation: Optional[Dict[str, Any]] = None
    if winner_reconstructed:
        # One-shot evaluation on the full (train ∪ val) market set so the
        # reconstructed winner at least carries a trustworthy metrics dict.
        all_markets = sorted(set(train_market_ids) | set(val_market_ids))
        logging.info(
            "Reconstructed winner (start %d, %s): evaluating on all %d markets "
            "to produce winner_evaluation artefact...",
            best_start_idx, best_label, len(all_markets),
        )
        try:
            obj_all, metrics_all, eval_elapsed = _evaluate_alternating_trajectory_on_markets(
                loader,
                args,
                detector_config=best_trajectory.detector_config,
                clustering_config=best_trajectory.clustering_config,
                market_ids=all_markets,
                max_workers=max_workers,
            )
            winner_evaluation = {
                "source": f"reconstructed_start_{best_start_idx}",
                "label": best_label,
                "ranking_objective": best_score,
                "objective_metric": args.objective,
                "all_markets_objective": obj_all,
                "all_markets_metrics": metrics_all,
                "n_markets_evaluated": len(all_markets),
                "n_train_markets": len(train_market_ids),
                "n_val_markets": len(val_market_ids),
                "eval_elapsed_seconds": eval_elapsed,
            }
            _write_json(winner_eval_path, winner_evaluation)
            logging.info(
                "  winner all-markets %s=%.4f (train+val, %.1fs)",
                args.objective, obj_all, eval_elapsed,
            )
        except Exception:
            logging.exception(
                "Winner evaluation on all markets failed; skipping winner_evaluation artefact"
            )

    _write_json(ms_summary_path, ms_summary)

    result: Dict[str, Any] = {
        "best_config_path": best_config_path,
        "inferred_res_path": inferred_res_path,
        "multi_start_summary_path": ms_summary_path,
        "max_workers": max_workers,
        "clustering_workers": clustering_workers,
        "multi_start_summary": ms_summary,
        "final_objective": best_trajectory.final_objective,
        "reconstructed_winner": winner_reconstructed,
    }
    if not winner_reconstructed:
        result.update({
            "detector_results_path": detector_results_path,
            "clustering_results_path": clustering_results_path,
            "detector_summaries_path": detector_summaries_path,
            "clustering_summaries_path": clustering_summaries_path,
        })
        if ja_df_written:
            result["ja_results_path"] = ja_results_path
    if winner_evaluation is not None:
        result["winner_evaluation_path"] = winner_eval_path
        result["winner_evaluation"] = winner_evaluation
    return result


def run_timeframe_optimizer(
    loader: HistoricalDataLoader,
    prep: TimeframePrepResult,
    args: Any,
) -> Dict[str, Any]:
    """Dispatch to the requested timeframe optimizer mode."""
    optimizer_mode = str(getattr(args, "optimizer_mode", "alternating_det_clust"))
    n_starts = int(getattr(args, "n_starts", 1) or 1)

    if n_starts > 1:
        if optimizer_mode == "coordinate_descent":
            return run_multi_start_coordinate_descent_timeframe(loader, prep, args)
        if optimizer_mode == "alternating_det_clust":
            return run_multi_start_alternating_timeframe(loader, prep, args)
        raise ValueError(
            f"Unsupported optimizer_mode for multi-start: {optimizer_mode!r}"
        )

    if optimizer_mode == "coordinate_descent":
        return run_coordinate_descent_timeframe(loader, prep, args)
    if optimizer_mode == "alternating_det_clust":
        return run_alternating_detectors_clustering_timeframe(loader, prep, args)
    raise ValueError(f"Unsupported optimizer_mode: {optimizer_mode!r}")
