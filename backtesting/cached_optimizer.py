"""Cached coordinate-descent optimizer for repeated detector sweeps."""

from __future__ import annotations

import logging
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backtesting.backtest_runner import BacktestRunner
from backtesting.bucket_clustering_backtest_runner import BucketClusteringBacktestRunner
from backtesting.causal_boost_replay import BoostSchedule
from backtesting.cached_backtest_runner import (
    CachedBacktestRunner,
    FrozenSignalCache,
    build_frozen_signals,
)
from backtesting.cached_evaluator import (
    PrecomputedMarketGroundTruth,
    PrecomputedTradeReturns,
    build_trade_flag_mask,
    fast_evaluate_wallets,
    precompute_ground_truth,
    precompute_trade_returns,
)
from backtesting.data_loader import HistoricalDataLoader
from backtesting.fbeta_econ import finalize_fbeta_econ_metrics
from backtesting.trade_level_metrics import merge_trade_level_metrics
from backtesting.evaluation_support import (
    build_attribution_provider,
    get_backtest_worker_count,
)
from backtesting.parameter_optimizer import (
    CoordinateDescentOptimizer,
    _bar_progress,
    _calculate_metrics_from_wallet_evaluations,
    _iter_progress,
)
from backtesting.wallet_evaluator import WalletEvaluator
from jump_anticipation.core import apply_jump_boost
from models import filter_trades_by_notional

logger = logging.getLogger(__name__)


# Worker subprocess state populated by _init_cached_eval_worker.
_CW_LOADER: Optional[HistoricalDataLoader] = None
_CW_EVALUATOR: Optional[WalletEvaluator] = None          # fallback only
_CW_MIN_USD_AMOUNT: Optional[float] = None
_CW_INCLUDE_RECIDIVISM: bool = True
_CW_PREDICTION_MODE: str = "has_alert"
_CW_SUSPICION_THRESHOLD: float = 2.0
_CW_FLAG_RATE_THRESHOLD: float = 0.25
_CW_CLUSTERING_RUNNER: Optional[BucketClusteringBacktestRunner] = None
_CW_CLUSTERING_MIN_TRADE_SIZE: float = 5000.0
_CW_TARGET_DETECTOR_GROUP: str = ""
_CW_ATTRIBUTION_PROVIDER = None
_CW_PRECOMPUTED_CLUSTER_BOOSTS: Dict[int, Dict[str, float]] = {}
_CW_USE_PRECOMPUTED_CLUSTER_BOOSTS: bool = False
_CW_PRECOMPUTED_BOOST_SCHEDULES: Dict[int, BoostSchedule] = {}
_CW_USE_PRECOMPUTED_BOOST_SCHEDULES: bool = False

# Per-market data lives for the lifetime of one worker pool.
_CW_TRADE_DATA: Dict[int, List] = {}                              # market_id -> List[Trade]
_CW_MARKET_METADATA: Dict[int, Dict] = {}                         # market_id -> metadata dict
_CW_FROZEN_CACHES: Dict[int, FrozenSignalCache] = {}              # market_id -> cache
_CW_PRECOMPUTED_GT: Dict[int, Optional[PrecomputedMarketGroundTruth]] = {}  # market_id -> GT or None

_CW_PRECOMPUTED_TR: Dict[int, PrecomputedTradeReturns] = {}
_CW_ACTIVE_STAGE_TOKEN: Optional[str] = None


def _prepare_cached_worker_stage(
    *,
    base_config: Dict,
    target_detector_group: str,
    stage_token: Optional[str] = None,
) -> None:
    """
    Rebuild stage-dependent frozen caches from already-loaded detector trades.
    """
    global _CW_TARGET_DETECTOR_GROUP, _CW_FROZEN_CACHES, _CW_ACTIVE_STAGE_TOKEN

    _CW_TARGET_DETECTOR_GROUP = str(target_detector_group)
    rebuilt: Dict[int, FrozenSignalCache] = {}

    for market_id, detector_trades in _CW_TRADE_DATA.items():
        metadata = _CW_MARKET_METADATA.get(market_id)
        if metadata is None:
            continue
        rebuilt[market_id] = build_frozen_signals(
            config=base_config,
            trades=detector_trades,
            market_metadata=metadata,
            include_recidivism=_CW_INCLUDE_RECIDIVISM,
        )

    _CW_FROZEN_CACHES = rebuilt
    if stage_token is not None:
        _CW_ACTIVE_STAGE_TOKEN = str(stage_token)

def _init_cached_eval_worker(
    data_dir: str,
    min_usd_amount: Optional[float],
    include_recidivism: bool,
    z_score_threshold: float,
    min_wallet_notional: float,
    label_metric: str,
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
    clustering_config: Optional[Dict],
    clustering_min_trade_size: float,
    enable_layer2_attribution: bool,
    usdc_cache_db: str,
    polygonscan_api_key: Optional[str],
    precomputed_cluster_boosts: Optional[Dict[int, Dict[str, float]]],
    precomputed_boost_schedules: Optional[Dict[int, BoostSchedule]],
    market_ids_for_init: Tuple[int, ...],
    base_config: Dict,
    target_detector_group: str,
    stage_token: Optional[str] = None,
    ja_config: Optional[Dict] = None, 
) -> None:
    """Initialize a cached evaluation worker process."""
    global _CW_LOADER, _CW_EVALUATOR
    global _CW_MIN_USD_AMOUNT, _CW_INCLUDE_RECIDIVISM
    global _CW_PREDICTION_MODE, _CW_SUSPICION_THRESHOLD, _CW_FLAG_RATE_THRESHOLD
    global _CW_CLUSTERING_RUNNER, _CW_CLUSTERING_MIN_TRADE_SIZE
    global _CW_TARGET_DETECTOR_GROUP
    global _CW_ATTRIBUTION_PROVIDER
    global _CW_PRECOMPUTED_CLUSTER_BOOSTS, _CW_USE_PRECOMPUTED_CLUSTER_BOOSTS
    global _CW_PRECOMPUTED_BOOST_SCHEDULES, _CW_USE_PRECOMPUTED_BOOST_SCHEDULES
    global _CW_TRADE_DATA, _CW_MARKET_METADATA
    global _CW_FROZEN_CACHES, _CW_PRECOMPUTED_GT, _CW_PRECOMPUTED_TR
    global _CW_JA_SCORES, _CW_ACTIVE_STAGE_TOKEN

    logging.getLogger().setLevel(logging.WARNING)

    # Standard worker state
    _CW_MIN_USD_AMOUNT = min_usd_amount
    _CW_INCLUDE_RECIDIVISM = include_recidivism
    _CW_PREDICTION_MODE = prediction_mode
    _CW_SUSPICION_THRESHOLD = float(suspicion_threshold)
    _CW_FLAG_RATE_THRESHOLD = float(flag_rate_threshold)
    _CW_CLUSTERING_MIN_TRADE_SIZE = float(clustering_min_trade_size)
    _CW_TARGET_DETECTOR_GROUP = target_detector_group
    _CW_PRECOMPUTED_CLUSTER_BOOSTS = precomputed_cluster_boosts or {}
    _CW_USE_PRECOMPUTED_CLUSTER_BOOSTS = precomputed_cluster_boosts is not None
    _CW_PRECOMPUTED_BOOST_SCHEDULES = precomputed_boost_schedules or {}
    _CW_USE_PRECOMPUTED_BOOST_SCHEDULES = precomputed_boost_schedules is not None
    _CW_ACTIVE_STAGE_TOKEN = None

    _CW_LOADER = HistoricalDataLoader(data_dir=data_dir, cache_size=0)
    _CW_LOADER.load_data()

    # Fallback evaluator for unresolved markets
    _CW_EVALUATOR = WalletEvaluator(
        z_score_threshold=z_score_threshold,
        min_wallet_notional=min_wallet_notional,
        label_metric=label_metric,
    )

    _CW_ATTRIBUTION_PROVIDER = None

    if (
        clustering_config is not None
        and not _CW_USE_PRECOMPUTED_CLUSTER_BOOSTS
        and not _CW_USE_PRECOMPUTED_BOOST_SCHEDULES
    ):
        _CW_ATTRIBUTION_PROVIDER = build_attribution_provider(
            enable_layer2_attribution=enable_layer2_attribution,
            usdc_cache_db=usdc_cache_db,
            polygonscan_api_key=polygonscan_api_key,
        )
        _CW_CLUSTERING_RUNNER = BucketClusteringBacktestRunner(
            detector_config={},
            clustering_config=clustering_config,
            attribution_provider=_CW_ATTRIBUTION_PROVIDER,
        )
    else:
        _CW_CLUSTERING_RUNNER = None

    _CW_TRADE_DATA = {}      # filtered trades for detectors
    _CW_MARKET_METADATA = {}
    _CW_FROZEN_CACHES = {}
    _CW_PRECOMPUTED_GT = {}
    _CW_PRECOMPUTED_TR = {}
    _CW_JA_SCORES = {}

    for market_id in market_ids_for_init:
        # Replay trades obey any active trade-time window.
        try:
            replay_trades = _CW_LOADER.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=None,
                use_cache=False,
            )
        except TypeError:
            replay_trades = _CW_LOADER.get_trades_for_market(market_id)

        # Ground truth labels use complete market history, not the replay
        # window. This keeps timeframe-trade experiments causal for detection
        # while avoiding truncated wallet PnL/position labels.
        try:
            ground_truth_trades = _CW_LOADER.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=None,
                use_cache=False,
                ignore_trade_time_bounds=True,
            )
        except TypeError:
            ground_truth_trades = replay_trades

        # Filter in memory to avoid a second SQL query.
        if min_usd_amount is not None:
            detector_trades = filter_trades_by_notional(replay_trades, min_usd_amount)
        else:
            detector_trades = replay_trades

        metadata = dict(_CW_LOADER.get_market_metadata(market_id) or {})
        metadata["id"] = market_id

        _CW_TRADE_DATA[market_id] = detector_trades
        _CW_MARKET_METADATA[market_id] = metadata

        # Ground truth uses full market history.
        gt = precompute_ground_truth(
            trades=ground_truth_trades,
            market_metadata=metadata,
            label_metric=label_metric,
            z_score_threshold=z_score_threshold,
            min_wallet_notional=min_wallet_notional,
        )
        _CW_PRECOMPUTED_GT[market_id] = gt

        # Trade returns use the replay tape.
        tr = precompute_trade_returns(
            trades=replay_trades,
            market_metadata=metadata,
            winning_outcome=gt.winning_outcome if gt is not None else None,
        )
        _CW_PRECOMPUTED_TR[market_id] = tr

    # Jump detection needs the dense tape; wallet scoring uses detector trades.
    if ja_config is not None and not _CW_USE_PRECOMPUTED_BOOST_SCHEDULES:
        from jump_anticipation.core import find_jumps, score_wallets_jump_anticipation
        for market_id in market_ids_for_init:
            try:
                # Reload unfiltered trades for the jump price series.
                try:
                    all_trades_for_ja = _CW_LOADER.get_trades_for_market(
                        market_id=market_id, min_usd_amount=None, use_cache=False
                    )
                except TypeError:
                    all_trades_for_ja = _CW_LOADER.get_trades_for_market(market_id)

                jumps = find_jumps(all_trades_for_ja, ja_config)
                if jumps:
                    # Score wallets on FILTERED trades (same as detector pipeline)
                    scoring_trades = _CW_TRADE_DATA[market_id]
                    _CW_JA_SCORES[market_id] = score_wallets_jump_anticipation(
                        scoring_trades, jumps, ja_config
                    )
                else:
                    _CW_JA_SCORES[market_id] = {}
            except Exception as e:
                logging.warning(f"JA precomputation failed for market {market_id}: {e}")
                _CW_JA_SCORES[market_id] = {}

    # Build initial stage caches once after immutable market data is loaded.
    _prepare_cached_worker_stage(
        base_config=base_config,
        target_detector_group=target_detector_group,
        stage_token=stage_token,
    )


def _cached_evaluate_config_worker(
    config: Dict,
    market_ids_tuple: Tuple[int, ...],
    base_config: Dict,
    target_detector_group: str,
    stage_token: str,
) -> Dict:
    """Evaluate one candidate config against cached market data."""
    if _CW_ACTIVE_STAGE_TOKEN != stage_token:
        _prepare_cached_worker_stage(
            base_config=base_config,
            target_detector_group=target_detector_group,
            stage_token=stage_token,
        )

    start = time.time()
    all_wallet_evals: List[Dict] = []

    # Accumulators for pooled trade-level metrics
    all_flagged_returns: List[np.ndarray] = []
    all_unflagged_returns: List[np.ndarray] = []
    all_flagged_notionals: List[np.ndarray] = []
    all_unflagged_notionals: List[np.ndarray] = []

    for market_id in market_ids_tuple:
        trades = _CW_TRADE_DATA.get(market_id)
        metadata = _CW_MARKET_METADATA.get(market_id)
        frozen_cache = _CW_FROZEN_CACHES.get(market_id)
        precomputed_gt = _CW_PRECOMPUTED_GT.get(market_id)
        precomputed_tr = _CW_PRECOMPUTED_TR.get(market_id)

        if trades is None or metadata is None or frozen_cache is None:
            # Shouldn't happen, but fall back gracefully rather than crash.
            logger.warning(
                f"_cached_evaluate_config_worker: missing cache for market {market_id} "
                f"— falling back to BacktestRunner + WalletEvaluator."
            )
            _fallback_evals = _fallback_evaluate_market(config, market_id)
            all_wallet_evals.extend(_fallback_evals)
            continue

        schedule = (
            _CW_PRECOMPUTED_BOOST_SCHEDULES.get(market_id)
            if _CW_USE_PRECOMPUTED_BOOST_SCHEDULES
            else None
        )

        # Cached hot path
        runner = CachedBacktestRunner(
            config=config,
            target_detector_group=_CW_TARGET_DETECTOR_GROUP,
            frozen_cache=frozen_cache,
            include_recidivism=_CW_INCLUDE_RECIDIVISM,
            score_multipliers=(
                schedule.score_multiplier_by_trade_idx if schedule is not None else None
            ),
            score_cap=(schedule.score_cap if schedule is not None else 0.95),
            wallet_cluster_boost=(
                schedule.final_wallet_cluster_boost if schedule is not None else None
            ),
            wallet_has_common_ownership=(
                schedule.final_wallet_has_common_ownership if schedule is not None else None
            ),
        )

        backtest_result = runner.run_backtest(
            trades=trades,
            market_metadata=metadata,
        )

        # Legacy post-hoc boosts if no causal schedule is available.
        if schedule is None:
            if _CW_USE_PRECOMPUTED_CLUSTER_BOOSTS:
                BucketClusteringBacktestRunner.apply_precomputed_wallet_boosts(
                    backtest_result,
                    _CW_PRECOMPUTED_CLUSTER_BOOSTS.get(market_id, {}),
                )
            elif _CW_CLUSTERING_RUNNER is not None:
                graph_trades = filter_trades_by_notional(
                    trades,
                    _CW_CLUSTERING_MIN_TRADE_SIZE,
                )
                backtest_result = _CW_CLUSTERING_RUNNER.run_boost_only(
                    base_result=backtest_result,
                    graph_trades=graph_trades,
                    market_id=str(market_id),
                )

            # Jump anticipation boost — scores precomputed at worker init
            if _CW_JA_SCORES:
                ja_scores = _CW_JA_SCORES.get(market_id, {})
                if ja_scores:
                    apply_jump_boost(backtest_result, ja_scores)

        # Fast evaluation — no z-score recomputation, no financial recalculation.
        if precomputed_gt is not None:
            wallet_evals = fast_evaluate_wallets(precomputed_gt, backtest_result)
        else:
            # Unresolved market: fall back to WalletEvaluator 
            wallet_evals = _CW_EVALUATOR.evaluate_wallets(backtest_result, metadata) if _CW_EVALUATOR else []

        all_wallet_evals.extend(wallet_evals)

        # Trade-level: extract flagged/unflagged returns for this market
        if precomputed_tr is not None:
            is_flagged = build_trade_flag_mask(precomputed_tr, backtest_result)

            f_ret = precomputed_tr.returns[is_flagged]
            u_ret = precomputed_tr.returns[~is_flagged]
            if len(f_ret) > 0:
                all_flagged_returns.append(f_ret)
                all_flagged_notionals.append(precomputed_tr.notionals[is_flagged])
            if len(u_ret) > 0:
                all_unflagged_returns.append(u_ret)
                all_unflagged_notionals.append(precomputed_tr.notionals[~is_flagged])

    metrics = _calculate_metrics_from_wallet_evaluations(
        wallet_evaluations=all_wallet_evals,
        prediction_mode=_CW_PREDICTION_MODE,
        suspicion_threshold=_CW_SUSPICION_THRESHOLD,
        flag_rate_threshold=_CW_FLAG_RATE_THRESHOLD,
    )

    merge_trade_level_metrics(
        metrics,
        all_flagged_returns,
        all_unflagged_returns,
        all_flagged_notionals,
        all_unflagged_notionals,
    )
    finalize_fbeta_econ_metrics(metrics)

    return {
        "metrics": metrics,
        "elapsed_seconds": time.time() - start,
    }


def _fallback_evaluate_market(config: Dict, market_id: int) -> List[Dict]:
    """Full non-cached eval for one market. Only called if cache is missing."""
    if _CW_LOADER is None or _CW_EVALUATOR is None:
        return []
    
    # Load replay trades for detectors and a separate full-history tape for labels.
    try:
        replay_trades = _CW_LOADER.get_trades_for_market(
            market_id=market_id, min_usd_amount=None, use_cache=False
        )
    except TypeError:
        replay_trades = _CW_LOADER.get_trades_for_market(market_id)
    try:
        ground_truth_trades = _CW_LOADER.get_trades_for_market(
            market_id=market_id,
            min_usd_amount=None,
            use_cache=False,
            ignore_trade_time_bounds=True,
        )
    except TypeError:
        ground_truth_trades = replay_trades

    # Filter for detector path
    if _CW_MIN_USD_AMOUNT is not None:
        detector_trades = filter_trades_by_notional(replay_trades, _CW_MIN_USD_AMOUNT)
    else:
        detector_trades = replay_trades

    metadata = dict(_CW_LOADER.get_market_metadata(market_id) or {})
    metadata["id"] = market_id

    # Run detectors on filtered trades
    runner = BacktestRunner(config=config, include_recidivism=_CW_INCLUDE_RECIDIVISM)
    schedule = (
        _CW_PRECOMPUTED_BOOST_SCHEDULES.get(market_id)
        if _CW_USE_PRECOMPUTED_BOOST_SCHEDULES
        else None
    )
    result = runner.run_backtest(
        trades=detector_trades, market_metadata=metadata,
        capture_alerts=False, capture_trade_features=False, progress_every=0,
        score_multipliers=(schedule.score_multiplier_by_trade_idx if schedule is not None else None),
        score_cap=(schedule.score_cap if schedule is not None else 0.95),
        wallet_cluster_boost=(
            schedule.final_wallet_cluster_boost if schedule is not None else None
        ),
        wallet_has_common_ownership=(
            schedule.final_wallet_has_common_ownership if schedule is not None else None
        ),
    )

    if schedule is None:
        if _CW_USE_PRECOMPUTED_CLUSTER_BOOSTS:
            BucketClusteringBacktestRunner.apply_precomputed_wallet_boosts(
                result,
                _CW_PRECOMPUTED_CLUSTER_BOOSTS.get(market_id, {}),
            )
        elif _CW_CLUSTERING_RUNNER is not None:
            graph_trades = filter_trades_by_notional(
                detector_trades,
                _CW_CLUSTERING_MIN_TRADE_SIZE,
            )
            result = _CW_CLUSTERING_RUNNER.run_boost_only(
                base_result=result,
                graph_trades=graph_trades,
                market_id=str(market_id),
            )

    # Ground truth from complete market history — consistent with all other paths
    gt = precompute_ground_truth(
        trades=ground_truth_trades,
        market_metadata=metadata,
        label_metric=_CW_EVALUATOR.label_metric,
        z_score_threshold=_CW_EVALUATOR.z_score_threshold,
        min_wallet_notional=_CW_EVALUATOR.min_wallet_notional,
    )
    if gt is not None:
        return fast_evaluate_wallets(gt, result)

    # Unresolved market — WalletEvaluator returns [] anyway
    return _CW_EVALUATOR.evaluate_wallets(result, metadata)


class CachedCoordinateDescentOptimizer(CoordinateDescentOptimizer):
    """Coordinate-descent optimizer with cached signals and ground truth."""

    def _shutdown_cached_pool(self) -> None:
        executor = getattr(self, "_cached_pool_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                logger.exception("Failed to cleanly shut down cached worker pool")
        self._cached_pool_executor = None
        self._cached_pool_worker_count = 0
        self._cached_pool_preload_market_ids = tuple()

    def _build_stage_token(self, detector_name: str, base_config: Dict) -> str:
        payload = {
            "detector_name": detector_name,
            "include_recidivism": bool(self.include_recidivism),
            "base_config": base_config,
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _get_preload_market_ids(self, market_ids: List[int]) -> Tuple[int, ...]:
        configured = tuple(getattr(self, "_cached_pool_preload_market_ids", ()) or ())
        if configured and set(market_ids).issubset(set(configured)):
            return configured
        return tuple(market_ids)

    def optimize(
        self,
        loader,
        market_ids: List[int],
        n_passes: int = 1,
        optimize_order: Optional[List[str]] = None,
        initial_config: Optional[Dict] = None,
    ):
        # Reuse one worker pool for the whole optimize() call
        self._cached_pool_executor = None
        self._cached_pool_worker_count = 0
        self._cached_pool_preload_market_ids = tuple(market_ids)
        try:
            return super().optimize(
                loader=loader,
                market_ids=market_ids,
                n_passes=n_passes,
                optimize_order=optimize_order,
                initial_config=initial_config,
            )
        finally:
            self._shutdown_cached_pool()
            self._cached_pool_preload_market_ids = tuple()

    def _optimize_detector(
        self,
        detector_name: str,
        base_config: Dict,
        base_metrics: Dict,
        loader,
        full_market_ids: List[int],
        coarse_market_ids: List[int],
        pass_num: int,
        detector_idx: int,
        total_detectors: int,
    ) -> Dict:
        from backtesting.parameter_grid import ParameterGrid
        import json

        logging.info("\n" + "=" * 80)
        logging.info(f"[CACHED] OPTIMIZING: {detector_name} ({detector_idx + 1}/{total_detectors})")
        logging.info("=" * 80)

        baseline_obj = self._objective(base_metrics)
        baseline_f1 = float(base_metrics.get("f1", 0.0))
        configs_and_params = ParameterGrid.generate_configs_for_detector(detector_name, base_config)

        candidates = [
            {"config_id": idx, "config": cfg, "params": params}
            for idx, (cfg, params) in enumerate(configs_and_params)
        ]
        logging.info(f"Grid size: {len(candidates):,}")

        coarse_scores: Dict[int, float] = {}
        shortlisted = candidates

        use_coarse_stage = (
            self.coarse_top_k > 0
            and len(candidates) > self.coarse_top_k
            and len(coarse_market_ids) > 0
            and set(coarse_market_ids) != set(full_market_ids)
        )

        if use_coarse_stage:
            logging.info(f"[CACHED] Coarse stage on {len(coarse_market_ids)} market(s)...")
            coarse_start = time.time()

            # cached coarse evaluation
            coarse_eval_map = self._evaluate_candidates_cached(
                candidates=candidates,
                loader=loader,
                market_ids=coarse_market_ids,
                detector_name=detector_name,
                stage_label="coarse",
                allow_parallel=self.parallelize_coarse,
                base_config=base_config,
            )

            for item in candidates:
                cfg_id = item["config_id"]
                coarse_scores[cfg_id] = self._objective(coarse_eval_map[cfg_id]["metrics"])

            ranked = sorted(candidates, key=lambda x: coarse_scores[x["config_id"]], reverse=True)
            shortlisted = ranked[: self.coarse_top_k]

            logging.info(
                f"[CACHED] Coarse stage done in {(time.time() - coarse_start):.1f}s "
                f"| shortlisted={len(shortlisted)}"
            )

        logging.info(f"[CACHED] Full stage candidates: {len(shortlisted)}")

        # cached full evaluation
        full_eval_map = self._evaluate_candidates_cached(
            candidates=shortlisted,
            loader=loader,
            market_ids=full_market_ids,
            detector_name=detector_name,
            stage_label="full",
            allow_parallel=self.parallelize_full,
            base_config=base_config,
        )

        # Best config selection
        best_obj = baseline_obj
        best_config = base_config
        best_metrics = base_metrics
        best_params = None
        results: List[Dict] = []

        for item in shortlisted:
            cfg = item["config"]
            cfg_id = item["config_id"]
            params = item["params"]

            eval_result = full_eval_map[cfg_id]
            metrics = eval_result["metrics"]
            cfg_obj = self._objective(metrics)

            row = {
                "pass": pass_num,
                "detector": detector_name,
                "config_id": cfg_id,
                **params,
                **metrics,
                "objective_metric": self.objective_metric,
                "objective_score": cfg_obj,
                "coarse_objective": coarse_scores.get(cfg_id),
                "coarse_f1": None,
                "config_time_seconds": eval_result["elapsed_seconds"],
                "full_config_json": json.dumps(cfg, sort_keys=True),
            }
            results.append(row)

            if cfg_obj > best_obj:
                best_obj = cfg_obj
                best_config = cfg
                best_metrics = metrics
                best_params = params

        improvement_pct = (
            (best_obj - baseline_obj) / baseline_obj * 100.0
        ) if baseline_obj > 0 else 0.0

        logging.info(
            f"Done {detector_name}: baseline_{self.objective_metric}={baseline_obj:.4f}, "
            f"best_{self.objective_metric}={best_obj:.4f}, improvement={improvement_pct:+.2f}%"
        )
        logging.info("  Baseline cls: " + self.format_classification_metrics(base_metrics))
        logging.info("  Best cls:     " + self.format_classification_metrics(best_metrics))
        if best_params is not None:
            logging.info(f"  Best params:  {best_params}")

        return {
            "best_config": best_config,
            "best_metrics": best_metrics,
            "best_objective": best_obj,
            "baseline_objective": baseline_obj,
            "best_f1": float(best_metrics.get("f1", 0.0)),
            "baseline_f1": baseline_f1,
            "best_params": best_params,
            "configs_tested_full": len(shortlisted),
            "configs_tested_coarse": len(candidates),
            "improvement_pct": improvement_pct,
            "results": results,
        }

    def _evaluate_candidates_cached(
        self,
        candidates: List[Dict],
        loader,
        market_ids: List[int],
        detector_name: str,
        stage_label: str,
        allow_parallel: bool,
        base_config: Dict,
    ) -> Dict[int, Dict]:
        """Evaluate candidate configs using cached market data."""
        if len(candidates) == 0:
            return {}

        basic_worker_count = max(1, min(int(self.max_workers), int(len(candidates))))
        if not (allow_parallel and basic_worker_count > 1 and len(candidates) > 1):
            return self._evaluate_candidates_cached_serial(
                candidates=candidates,
                loader=loader,
                market_ids=market_ids,
                detector_name=detector_name,
                stage_label=stage_label,
                base_config=base_config,
            )

        market_ids_tuple = tuple(market_ids)
        preload_market_ids_tuple = self._get_preload_market_ids(market_ids)
        preload_market_ids = list(preload_market_ids_tuple)

        precomputed_boost_schedules = self._get_precomputed_boost_schedules(
            loader,
            preload_market_ids,
        )
        precomputed_cluster_boosts = None
        if precomputed_boost_schedules is None:
            precomputed_cluster_boosts = self._get_precomputed_cluster_boosts(
                loader,
                preload_market_ids,
            )

        worker_count = get_backtest_worker_count(
            self.max_workers,
            len(candidates),
            enable_layer2_attribution=self.enable_layer2_attribution,
            clustering_enabled=self.clustering_config is not None,
            live_layer2_fetches=(
                precomputed_boost_schedules is None and precomputed_cluster_boosts is None
            ),
        )
        if not (allow_parallel and worker_count > 1 and len(candidates) > 1):
            return self._evaluate_candidates_cached_serial(
                candidates=candidates,
                loader=loader,
                market_ids=market_ids,
                detector_name=detector_name,
                stage_label=stage_label,
                base_config=base_config,
                precomputed_boost_schedules=precomputed_boost_schedules,
                precomputed_cluster_boosts=precomputed_cluster_boosts,
            )

        stage_token = self._build_stage_token(detector_name=detector_name, base_config=base_config)

        logging.info(
            f"[CACHED] Parallel {stage_label} eval: {len(candidates)} configs "
            f"on {worker_count} worker(s) | target={detector_name}"
        )

        eval_map: Dict[int, Dict] = {}

        executor = getattr(self, "_cached_pool_executor", None)
        cached_worker_count = int(getattr(self, "_cached_pool_worker_count", 0) or 0)
        cached_preload_market_ids = tuple(getattr(self, "_cached_pool_preload_market_ids", ()) or ())

        pool_needs_reset = (
            executor is None
            or cached_worker_count < worker_count
            or cached_preload_market_ids != preload_market_ids_tuple
        )
        if pool_needs_reset:
            self._shutdown_cached_pool()
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_cached_eval_worker,
                initargs=(
                    self.data_dir,
                    self._effective_min_usd_amount(),
                    self.include_recidivism,
                    self.z_score_threshold,
                    self.min_wallet_notional,
                    self.label_metric,
                    self.prediction_mode,
                    self.suspicion_threshold,
                    self.flag_rate_threshold,
                    self.clustering_config,
                    self.clustering_min_trade_size,
                    self.enable_layer2_attribution,
                    self.usdc_cache_db,
                    self.polygonscan_api_key,
                    precomputed_cluster_boosts,
                    precomputed_boost_schedules,
                    preload_market_ids_tuple,   # immutable preload scope for this pool
                    base_config,                # initial frozen cache stage config
                    detector_name,              # initial target detector
                    stage_token,                # stage token matching initial frozen cache
                    self.ja_config,             # used to build JA score cache if applicable
                ),
            )
            self._cached_pool_executor = executor
            self._cached_pool_worker_count = worker_count
            self._cached_pool_preload_market_ids = preload_market_ids_tuple

        assert executor is not None
        future_to_cfg_id = {
            executor.submit(
                _cached_evaluate_config_worker,
                item["config"],
                market_ids_tuple,
                base_config,
                detector_name,
                stage_token,
            ): item["config_id"]
            for item in candidates
        }

        with _bar_progress(
            total=len(future_to_cfg_id),
            show_progress=self.show_progress,
            desc=f"[cached] {detector_name} {stage_label}",
            unit="cfg",
            leave=False,
        ) as pbar:
            for future in as_completed(future_to_cfg_id):
                cfg_id = future_to_cfg_id[future]
                try:
                    eval_map[cfg_id] = future.result()
                except Exception as exc:
                    self._shutdown_cached_pool()
                    raise RuntimeError(
                        f"[CACHED] {detector_name} {stage_label} eval failed "
                        f"for config_id={cfg_id}"
                    ) from exc
                pbar.update(1)

        return eval_map

    def _evaluate_candidates_cached_serial(
        self,
        candidates: List[Dict],
        loader,
        market_ids: List[int],
        detector_name: str,
        stage_label: str,
        base_config: Dict,
        precomputed_boost_schedules: Optional[Dict[int, BoostSchedule]] = None,
        precomputed_cluster_boosts: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> Dict[int, Dict]:
        """
        Serial cached evaluation path.
        """
        frozen_caches: Dict[int, FrozenSignalCache] = {}
        precomputed_gts: Dict[int, Optional[PrecomputedMarketGroundTruth]] = {}
        precomputed_trs: Dict[int, Optional[PrecomputedTradeReturns]] = {}
        detector_trade_data: Dict[int, List] = {}   # filtered — for detectors
        metadata_map: Dict[int, Dict] = {}
        if precomputed_boost_schedules is None:
            precomputed_boost_schedules = self._get_precomputed_boost_schedules(loader, market_ids)
        if precomputed_boost_schedules is None and precomputed_cluster_boosts is None:
            precomputed_cluster_boosts = self._get_precomputed_cluster_boosts(loader, market_ids)

        effective_min = self._effective_min_usd_amount()

        for market_id in market_ids:
            # Replay trades obey any active trade-time window.
            try:
                replay_trades = loader.get_trades_for_market(
                    market_id,
                    min_usd_amount=None,
                    use_cache=False,
                )
            except TypeError:
                replay_trades = loader.get_trades_for_market(market_id)
            ground_truth_trades = self._safe_get_trades_unfiltered(loader, market_id)
            metadata = dict(loader.get_market_metadata(market_id) or {})
            metadata["id"] = market_id

            # Filter in memory to avoid a second SQL query.
            if effective_min is not None:
                detector_trades = filter_trades_by_notional(replay_trades, effective_min)
            else:
                detector_trades = replay_trades

            detector_trade_data[market_id] = detector_trades
            metadata_map[market_id] = metadata

            # Frozen cache from detector trades.
            frozen_caches[market_id] = build_frozen_signals(
                config=base_config,
                trades=detector_trades,
                market_metadata=metadata,
                include_recidivism=self.include_recidivism,
            )
            # Ground truth from complete market history
            precomputed_gts[market_id] = precompute_ground_truth(
                trades=ground_truth_trades,
                market_metadata=metadata,
                label_metric=self.label_metric,
                z_score_threshold=self.z_score_threshold,
                min_wallet_notional=self.min_wallet_notional,
            )
            # Trade returns from the replay tape
            precomputed_trs[market_id] = precompute_trade_returns(
                trades=replay_trades,
                market_metadata=metadata,
                winning_outcome=precomputed_gts[market_id].winning_outcome
                    if precomputed_gts[market_id] is not None else None,
            )

        # Precompute JA scores once per market
        ja_scores_by_market: Dict[int, Dict[str, float]] = {}
        if self.ja_config is not None and precomputed_boost_schedules is None:
            from jump_anticipation.core import find_jumps, score_wallets_jump_anticipation
            for market_id in market_ids:
                try:
                    try:
                        all_trades_for_ja = loader.get_trades_for_market(
                            market_id,
                            min_usd_amount=None,
                            use_cache=False,
                        )
                    except TypeError:
                        all_trades_for_ja = loader.get_trades_for_market(market_id)
                    jumps = find_jumps(all_trades_for_ja, self.ja_config)
                    ja_scores_by_market[market_id] = (
                        score_wallets_jump_anticipation(
                            detector_trade_data[market_id], jumps, self.ja_config
                        )
                        if jumps else {}
                    )
                except Exception as e:
                    logger.warning(f"JA precomputation failed for market {market_id}: {e}")
                    ja_scores_by_market[market_id] = {}

        # Iterate candidates
        eval_map: Dict[int, Dict] = {}
        iterable = _iter_progress(
            candidates,
            show_progress=self.show_progress,
            desc=f"[cached] {detector_name} {stage_label}",
            unit="cfg",
            leave=False,
        )

        for item in iterable:
            start = time.time()
            config = item["config"]
            all_wallet_evals: List[Dict] = []

            all_flagged_returns: List[np.ndarray] = []
            all_unflagged_returns: List[np.ndarray] = []
            all_flagged_notionals: List[np.ndarray] = []
            all_unflagged_notionals: List[np.ndarray] = []

            for market_id in market_ids:
                detector_trades = detector_trade_data[market_id]
                metadata = metadata_map[market_id]
                frozen_cache = frozen_caches[market_id]
                precomputed_gt = precomputed_gts[market_id]
                precomputed_tr = precomputed_trs[market_id]
                schedule = (
                    precomputed_boost_schedules.get(market_id)
                    if precomputed_boost_schedules is not None
                    else None
                )

                # Backtest runs on filtered trades
                runner = CachedBacktestRunner(
                    config=config,
                    target_detector_group=detector_name,
                    frozen_cache=frozen_cache,
                    include_recidivism=self.include_recidivism,
                    score_multipliers=(
                        schedule.score_multiplier_by_trade_idx if schedule is not None else None
                    ),
                    score_cap=(schedule.score_cap if schedule is not None else 0.95),
                    wallet_cluster_boost=(
                        schedule.final_wallet_cluster_boost if schedule is not None else None
                    ),
                    wallet_has_common_ownership=(
                        schedule.final_wallet_has_common_ownership if schedule is not None else None
                    ),
                )
                backtest_result = runner.run_backtest(trades=detector_trades, market_metadata=metadata)

                if schedule is None:
                    if precomputed_cluster_boosts is not None:
                        BucketClusteringBacktestRunner.apply_precomputed_wallet_boosts(
                            backtest_result,
                            precomputed_cluster_boosts.get(market_id, {}),
                        )

                    # Jump anticipation boost
                    if ja_scores_by_market:
                        ja_scores = ja_scores_by_market.get(market_id, {})
                        if ja_scores:
                            apply_jump_boost(backtest_result, ja_scores)

                # Wallet evaluation uses precomputed ground truth
                if precomputed_gt is not None:
                    wallet_evals = fast_evaluate_wallets(precomputed_gt, backtest_result)
                else:
                    wallet_evals = self.evaluator.evaluate_wallets(backtest_result, metadata)

                all_wallet_evals.extend(wallet_evals)

                # Trade-level metrics from precomputed returns
                if precomputed_tr is not None:
                    is_flagged = build_trade_flag_mask(precomputed_tr, backtest_result)

                    f_ret = precomputed_tr.returns[is_flagged]
                    u_ret = precomputed_tr.returns[~is_flagged]
                    
                    if len(f_ret) > 0:
                        all_flagged_returns.append(f_ret)
                        all_flagged_notionals.append(precomputed_tr.notionals[is_flagged])
                    if len(u_ret) > 0:
                        all_unflagged_returns.append(u_ret)
                        all_unflagged_notionals.append(precomputed_tr.notionals[~is_flagged])

            metrics = _calculate_metrics_from_wallet_evaluations(
                wallet_evaluations=all_wallet_evals,
                prediction_mode=self.prediction_mode,
                suspicion_threshold=self.suspicion_threshold,
                flag_rate_threshold=self.flag_rate_threshold,
            )

            merge_trade_level_metrics(
                metrics,
                all_flagged_returns,
                all_unflagged_returns,
                all_flagged_notionals,
                all_unflagged_notionals,
            )
            finalize_fbeta_econ_metrics(metrics)

            eval_map[item["config_id"]] = {
                "metrics": metrics,
                "elapsed_seconds": time.time() - start,
            }

        return eval_map
