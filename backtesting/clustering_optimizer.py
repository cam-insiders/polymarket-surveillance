"""
Coordinate descent optimizer for clustering parameters.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score

from backtesting.backtest_runner import BacktestRunner, BacktestResult
from backtesting.causal_boost_replay import build_live_parity_boost_schedule
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
from backtesting.clustering_parameter_grid import ClusteringParameterGrid
from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation_support import (
    build_attribution_provider,
    get_backtest_worker_count,
)
from backtesting.fbeta_econ import finalize_fbeta_econ_metrics
from backtesting.trade_level_metrics import EXTENDED_TRADE_OBJECTIVES
from backtesting.logging_utils import experiment_backtest_logs_quiet
from backtesting.wallet_evaluator import WalletEvaluator
from models import Trade, filter_trades_by_notional

# Optional tqdm
try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _tqdm = None
    _TQDM_AVAILABLE = False


class _NoOpPbar:
    """No-op progress bar for when tqdm is unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def update(self, _n: int = 1):
        return None


def _iter_progress(iterable, show_progress: bool, **kwargs):
    """Wrap iterable with progress bar if available."""
    if show_progress and _TQDM_AVAILABLE:
        return _tqdm(iterable, **kwargs)
    return iterable


def _bar_progress(total: int, show_progress: bool, **kwargs):
    """Create progress bar if available."""
    if show_progress and _TQDM_AVAILABLE:
        return _tqdm(total=total, **kwargs)
    return _NoOpPbar()


def _median(values: List[float]) -> float:
    """Compute median of a list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _merge_trade_level_metrics(
    metrics: Dict,
    flagged_returns_chunks: List[np.ndarray],
    unflagged_returns_chunks: List[np.ndarray],
    flagged_notionals_chunks: Optional[List[np.ndarray]] = None,
    unflagged_notionals_chunks: Optional[List[np.ndarray]] = None,
) -> None:
    """Merge pooled flagged/unflagged trade-return statistics into metrics."""
    from backtesting.trade_level_metrics import merge_trade_level_metrics

    merge_trade_level_metrics(
        metrics,
        flagged_returns_chunks,
        unflagged_returns_chunks,
        flagged_notionals_chunks=flagged_notionals_chunks,
        unflagged_notionals_chunks=unflagged_notionals_chunks,
    )
    finalize_fbeta_econ_metrics(metrics)


def _calculate_metrics_from_wallet_evaluations(
    wallet_evaluations: List[Dict],
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
) -> Dict:
    """Calculate classification metrics from wallet evaluations."""
    if not wallet_evaluations:
        return {
            "num_wallets": 0,
            "num_predicted_positive": 0,
            "num_true_insiders": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f0_5": 0.0,
            "f2": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "mean_net_pnl_flagged": 0.0,
            "median_abs_return_flagged": 0.0,
            "median_net_pnl_flagged": 0.0,
            "median_abs_net_pnl_flagged": 0.0,
            "mean_return_flagged": 0.0,
            "trade_flagged_count": 0,
            "trade_unflagged_count": 0,
            "trade_flagged_mean_return": 0.0,
            "trade_flagged_mean_return_lcb": 0.0,
            "trade_flagged_mean_return_se": 0.0,
            "trade_flagged_weighted_return": 0.0,
            "trade_flagged_weighted_return_lcb": 0.0,
            "trade_flagged_weighted_return_se": 0.0,
            "trade_unflagged_mean_return": 0.0,
            "trade_mean_return_diff": 0.0,
            "trade_t_stat": 0.0,
            "trade_cohens_d": 0.0,
            "trade_cohens_d_lcb": 0.0,
            "trade_cohens_d_se": 0.0,
            "trade_flagged_win_rate": 0.0,
            "trade_weighted_mean_return_diff": 0.0,
            "trade_weighted_cohens_d": 0.0,
            "trade_weighted_cohens_d_lcb": 0.0,
            "trade_weighted_cohens_d_se": 0.0,
            "trade_winsorized_mean_return_diff": 0.0,
            "trade_winsorized_cohens_d": 0.0,
            "trade_winsorized_weighted_mean_return_diff": 0.0,
            "trade_winsorized_weighted_cohens_d": 0.0,
            "trade_winsorized_weighted_cohens_d_lcb": 0.0,
            "trade_winsorized_weighted_cohens_d_se": 0.0,
            "trade_weighted_flagged_effective_count": 0.0,
            "trade_weighted_unflagged_effective_count": 0.0,
            "econ_signal_winrate": 0.0,
            "econ_signal_mean_return_norm": 0.0,
            "econ_signal_return_sigmoid_norm": 0.0,
            "econ_signal_trade_mean_return_sigmoid_norm": 0.0,
            "econ_signal_trade_t_stat_norm": 0.0,
            "f_beta_econ_winrate": 0.0,
            "f_beta_econ_mean_return": 0.0,
            "f_beta_econ_winrate_f1": 0.0,
            "f_beta_econ_mean_return_f1": 0.0,
            "f_beta_econ_return_f1": 0.0,
            "f_beta_econ_t_stat_f1": 0.0,
            "f_beta_econ_winrate_f0_5": 0.0,
            "f_beta_econ_mean_return_f0_5": 0.0,
            "f_beta_econ_winrate_return": 0.0,
            "f_beta_econ_beta": 1.0,
            "f_beta_econ_t_stat_scale": 2.0,
            "f_beta_econ_return_scale": 1.0,
        }

    y_true = [bool(e["is_insider"]) for e in wallet_evaluations]

    if prediction_mode == "has_alert":
        y_pred = [bool(e["has_alert"]) for e in wallet_evaluations]
    elif prediction_mode == "suspicion_threshold":
        y_pred = [float(e["suspicion_score"]) >= float(suspicion_threshold) for e in wallet_evaluations]
    elif prediction_mode == "flag_rate":
        thr = float(flag_rate_threshold)
        y_pred = []
        for e in wallet_evaluations:
            trade_count = int(e.get("trade_count", 0) or 0)
            num_flags = int(e.get("num_flags", 0) or 0)
            rate = (num_flags / trade_count) if trade_count > 0 else 0.0
            y_pred.append(rate >= thr)
    elif prediction_mode == "boosted_flag_rate":
        # Backward-compatible alias. Causal trade-level boosts are already
        # reflected in per-trade alert decisions.
        thr = float(flag_rate_threshold)
        y_pred = []
        for e in wallet_evaluations:
            trade_count = int(e.get("trade_count", 0) or 0)
            num_flags = int(e.get("num_flags", 0) or 0)
            rate = (num_flags / trade_count) if trade_count > 0 else 0.0
            y_pred.append(rate >= thr)
    else:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f0_5 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    # Economic metrics for flagged wallets
    flagged_evals = [e for e, pred in zip(wallet_evaluations, y_pred) if pred]

    flagged_net_pnls = [float(e.get("net_pnl", 0.0) or 0.0) for e in flagged_evals]
    flagged_returns = [float(e.get("return", 0.0) or 0.0) for e in flagged_evals]
    flagged_informed = [float(e.get("informed_score", 0.0) or 0.0) for e in flagged_evals]

    result = {
        "num_wallets": len(wallet_evaluations),
        "num_predicted_positive": int(sum(y_pred)),
        "num_true_insiders": int(sum(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f0_5": float(f0_5),
        "f2": float(f2),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "mean_net_pnl_flagged": float(sum(flagged_net_pnls) / len(flagged_net_pnls)) if flagged_net_pnls else 0.0,
        "median_net_pnl_flagged": _median(flagged_net_pnls),
        "mean_return_flagged": float(sum(flagged_returns) / len(flagged_returns)) if flagged_returns else 0.0,
        "median_return_flagged": _median(flagged_returns),
        "median_informed_score_flagged": _median(flagged_informed),
        "trade_flagged_count": 0,
        "trade_unflagged_count": 0,
        "trade_flagged_mean_return": 0.0,
        "trade_flagged_mean_return_lcb": 0.0,
        "trade_flagged_mean_return_se": 0.0,
        "trade_flagged_weighted_return": 0.0,
        "trade_flagged_weighted_return_lcb": 0.0,
        "trade_flagged_weighted_return_se": 0.0,
        "trade_unflagged_mean_return": 0.0,
        "trade_mean_return_diff": 0.0,
        "trade_t_stat": 0.0,
        "trade_cohens_d": 0.0,
        "trade_cohens_d_lcb": 0.0,
        "trade_cohens_d_se": 0.0,
        "trade_flagged_win_rate": 0.0,
        "trade_weighted_mean_return_diff": 0.0,
        "trade_weighted_cohens_d": 0.0,
        "trade_weighted_cohens_d_lcb": 0.0,
        "trade_weighted_cohens_d_se": 0.0,
        "trade_winsorized_mean_return_diff": 0.0,
        "trade_winsorized_cohens_d": 0.0,
        "trade_winsorized_weighted_mean_return_diff": 0.0,
        "trade_winsorized_weighted_cohens_d": 0.0,
        "trade_winsorized_weighted_cohens_d_lcb": 0.0,
        "trade_winsorized_weighted_cohens_d_se": 0.0,
        "trade_weighted_flagged_effective_count": 0.0,
        "trade_weighted_unflagged_effective_count": 0.0,
    }
    finalize_fbeta_econ_metrics(result)
    return result


# ---------- Worker state for ProcessPool ----------
# Workers receive pre-computed detector results (lightweight) and filtered
# graph trades (small). No full trade lists, no detector re-runs.
_WORKER_BASE_RESULTS: Optional[Dict[int, BacktestResult]] = None
_WORKER_GRAPH_TRADES: Optional[Dict[int, List[Trade]]] = None
_WORKER_DETECTOR_TRADES: Optional[Dict[int, List[Trade]]] = None
_WORKER_FROZEN_CACHES: Optional[Dict[int, FrozenSignalCache]] = None
_WORKER_EVALUATOR: Optional[WalletEvaluator] = None
_WORKER_LOADER: Optional[HistoricalDataLoader] = None
_WORKER_INCLUDE_RECIDIVISM: bool = False
_WORKER_DETECTOR_CONFIG: Optional[Dict] = None
_WORKER_PREDICTION_MODE: str = "has_alert"
_WORKER_SUSPICION_THRESHOLD: float = 2.0
_WORKER_FLAG_RATE_THRESHOLD: float = 0.25
_WORKER_ATTRIBUTION_PROVIDER = None
_WORKER_PRECOMPUTED_GT: Optional[Dict[int, Optional[PrecomputedMarketGroundTruth]]] = None
_WORKER_PRECOMPUTED_TR: Optional[Dict[int, Optional[PrecomputedTradeReturns]]] = None
_WORKER_MIN_TRADE_SIZE: float = 5000.0
_WORKER_POLL_INTERVAL_SECONDS: float = 5.0
_WORKER_JUMP_ANTICIPATION_CONFIG: Optional[Dict] = None


def _init_clustering_worker(
    base_results: Dict[int, BacktestResult],
    graph_trades: Dict[int, List[Trade]],
    detector_trades: Dict[int, List[Trade]],
    frozen_caches: Dict[int, FrozenSignalCache],
    ground_truth_cache: Dict[int, Optional[PrecomputedMarketGroundTruth]],
    trade_returns_cache: Dict[int, Optional[PrecomputedTradeReturns]],
    detector_config: Dict,
    data_dir: str,
    include_recidivism: bool,
    z_score_threshold: float,
    min_wallet_notional: float,
    label_metric: str,
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
    enable_layer2_attribution: bool = False,
    usdc_cache_db: str = "data/usdc_transfers.db",
    polygonscan_api_key: Optional[str] = None,
    min_trade_size: float = 5000.0,
    poll_interval_seconds: float = 5.0,
    jump_anticipation_config: Optional[Dict] = None,
):
    """
    Initialize worker process for parallel clustering evaluation.

    Workers receive:
    - base_results: Pre-computed detector BacktestResults (one per market).
      These are lightweight dicts (~100KB per market) not full trade lists.
    - graph_trades: Size-filtered trades for graph construction only (~30k
      trades per market instead of 1.5M).
    """
    global _WORKER_BASE_RESULTS
    global _WORKER_GRAPH_TRADES
    global _WORKER_DETECTOR_TRADES
    global _WORKER_FROZEN_CACHES
    global _WORKER_EVALUATOR
    global _WORKER_LOADER
    global _WORKER_INCLUDE_RECIDIVISM
    global _WORKER_DETECTOR_CONFIG
    global _WORKER_PREDICTION_MODE
    global _WORKER_SUSPICION_THRESHOLD
    global _WORKER_FLAG_RATE_THRESHOLD
    global _WORKER_ATTRIBUTION_PROVIDER
    global _WORKER_PRECOMPUTED_GT
    global _WORKER_PRECOMPUTED_TR
    global _WORKER_MIN_TRADE_SIZE
    global _WORKER_POLL_INTERVAL_SECONDS
    global _WORKER_JUMP_ANTICIPATION_CONFIG

    logging.getLogger().setLevel(logging.WARNING)

    _WORKER_BASE_RESULTS = base_results
    _WORKER_GRAPH_TRADES = graph_trades
    _WORKER_DETECTOR_TRADES = detector_trades
    _WORKER_FROZEN_CACHES = frozen_caches
    _WORKER_PRECOMPUTED_GT = ground_truth_cache
    _WORKER_PRECOMPUTED_TR = trade_returns_cache
    _WORKER_DETECTOR_CONFIG = detector_config
    _WORKER_INCLUDE_RECIDIVISM = include_recidivism
    _WORKER_PREDICTION_MODE = prediction_mode
    _WORKER_SUSPICION_THRESHOLD = float(suspicion_threshold)
    _WORKER_FLAG_RATE_THRESHOLD = float(flag_rate_threshold)
    _WORKER_MIN_TRADE_SIZE = float(min_trade_size)
    _WORKER_POLL_INTERVAL_SECONDS = float(poll_interval_seconds)
    _WORKER_JUMP_ANTICIPATION_CONFIG = jump_anticipation_config

    _WORKER_LOADER = HistoricalDataLoader(data_dir=data_dir, cache_size=0)
    _WORKER_LOADER.load_data()

    _WORKER_EVALUATOR = WalletEvaluator(
        z_score_threshold=z_score_threshold,
        min_wallet_notional=min_wallet_notional,
        label_metric=label_metric,
    )

    _WORKER_ATTRIBUTION_PROVIDER = build_attribution_provider(
        enable_layer2_attribution=enable_layer2_attribution,
        usdc_cache_db=usdc_cache_db,
        polygonscan_api_key=polygonscan_api_key,
    )


def _evaluate_clustering_config_worker(
    clustering_config: Dict,
    market_ids: Tuple[int, ...],
) -> Dict:
    """
    Worker function to evaluate a clustering config.

    Uses pre-computed detector results (no detector re-runs).
    Only builds graph + clusters + applies boost.
    """
    if _WORKER_LOADER is None or _WORKER_EVALUATOR is None:
        raise RuntimeError("Worker state not initialized")
    if _WORKER_DETECTOR_TRADES is None or _WORKER_FROZEN_CACHES is None:
        raise RuntimeError("Worker detector_trades/frozen_caches not initialized")
    if _WORKER_PRECOMPUTED_GT is None or _WORKER_PRECOMPUTED_TR is None:
        raise RuntimeError("Worker ground truth/trade-return caches not initialized")

    start = time.time()

    all_wallet_evals: List[Dict] = []
    all_flagged_returns: List[np.ndarray] = []
    all_unflagged_returns: List[np.ndarray] = []
    all_flagged_notionals: List[np.ndarray] = []
    all_unflagged_notionals: List[np.ndarray] = []

    for market_id in market_ids:
        detector_trades = _WORKER_DETECTOR_TRADES.get(market_id, [])
        frozen_cache = _WORKER_FROZEN_CACHES.get(market_id)

        if frozen_cache is None:
            continue

        metadata = dict(_WORKER_LOADER.get_market_metadata(market_id) or {})
        metadata["id"] = market_id

        schedule = build_live_parity_boost_schedule(
            detector_trades=detector_trades,
            market_id=str(market_id),
            clustering_config=clustering_config,
            clustering_min_trade_size=_WORKER_MIN_TRADE_SIZE,
            jump_anticipation_config=_WORKER_JUMP_ANTICIPATION_CONFIG,
            poll_interval_seconds=_WORKER_POLL_INTERVAL_SECONDS,
            attribution_provider=_WORKER_ATTRIBUTION_PROVIDER,
            fetch_if_missing=_WORKER_ATTRIBUTION_PROVIDER is not None,
        )

        runner = CachedBacktestRunner(
            config=_WORKER_DETECTOR_CONFIG or {},
            target_detector_group="alert_threshold",
            frozen_cache=frozen_cache,
            include_recidivism=_WORKER_INCLUDE_RECIDIVISM,
            score_multipliers=schedule.score_multiplier_by_trade_idx,
            score_cap=schedule.score_cap,
            wallet_cluster_boost=schedule.final_wallet_cluster_boost,
            wallet_has_common_ownership=schedule.final_wallet_has_common_ownership,
        )
        boosted_result = runner.run_backtest(
            trades=detector_trades,
            market_metadata=metadata,
        )

        gt = _WORKER_PRECOMPUTED_GT.get(market_id)
        if gt is not None:
            wallet_evals = fast_evaluate_wallets(gt, boosted_result)
        else:
            wallet_evals = _WORKER_EVALUATOR.evaluate_wallets(boosted_result, metadata)
        all_wallet_evals.extend(wallet_evals)

        precomputed_tr = _WORKER_PRECOMPUTED_TR.get(market_id)
        if precomputed_tr is not None:
            is_flagged = build_trade_flag_mask(precomputed_tr, boosted_result)
            flagged = precomputed_tr.returns[is_flagged]
            unflagged = precomputed_tr.returns[~is_flagged]
            if len(flagged) > 0:
                all_flagged_returns.append(flagged)
                all_flagged_notionals.append(precomputed_tr.notionals[is_flagged])
            if len(unflagged) > 0:
                all_unflagged_returns.append(unflagged)
                all_unflagged_notionals.append(precomputed_tr.notionals[~is_flagged])

    metrics = _calculate_metrics_from_wallet_evaluations(
        wallet_evaluations=all_wallet_evals,
        prediction_mode=_WORKER_PREDICTION_MODE,
        suspicion_threshold=_WORKER_SUSPICION_THRESHOLD,
        flag_rate_threshold=_WORKER_FLAG_RATE_THRESHOLD,
    )
    _merge_trade_level_metrics(
        metrics,
        all_flagged_returns,
        all_unflagged_returns,
        all_flagged_notionals,
        all_unflagged_notionals,
    )

    return {
        "metrics": metrics,
        "elapsed_seconds": time.time() - start,
    }


class ClusteringOptimizer:
    """
    Coordinate descent optimizer for clustering parameters.
    """

    def __init__(
        self,
        detector_config: Dict,
        z_score_threshold: float = 2.0,
        min_wallet_notional: float = 500.0,
        label_metric: str = "return",
        prediction_mode: str = "has_alert",
        suspicion_threshold: float = 2.0,
        flag_rate_threshold: float = 0.25,
        coarse_top_k: int = 25,
        coarse_trade_cap: int = 250_000,
        min_trade_size: float = 10000.0,
        min_usd_amount: float = 300.0,
        enable_trade_prefilter: bool = False,
        data_dir: str = "data",
        max_workers: Optional[int] = None,
        parallelize_coarse: bool = True,
        parallelize_full: bool = True,
        show_progress: bool = True,
        objective_metric: str = "f0_5",
        include_recidivism: bool = False,
        enable_layer2_attribution: bool = False,
        usdc_cache_db: str = "data/usdc_transfers.db",
        polygonscan_api_key: Optional[str] = None,
        poll_interval_seconds: float = 5.0,
        jump_anticipation_config: Optional[Dict] = None,
    ):
        """Initialize clustering optimizer."""
        self.detector_config = detector_config
        self.z_score_threshold = z_score_threshold
        self.prediction_mode = prediction_mode
        self.suspicion_threshold = float(suspicion_threshold)
        self.flag_rate_threshold = float(flag_rate_threshold)
        self.coarse_top_k = int(coarse_top_k)
        self.coarse_trade_cap = int(coarse_trade_cap)
        self.min_trade_size = float(min_trade_size)
        self.min_usd_amount = None if min_usd_amount is None else float(min_usd_amount)
        self.enable_trade_prefilter = bool(enable_trade_prefilter)
        self.data_dir = data_dir

        self.parallelize_coarse = parallelize_coarse
        self.parallelize_full = parallelize_full
        self.show_progress = show_progress

        cpu_count = os.cpu_count() or 1
        default_workers = min(cpu_count, 8)
        self.max_workers = max(1, min(int(max_workers or default_workers), cpu_count))

        if self.show_progress and not _TQDM_AVAILABLE:
            logging.warning("tqdm is not installed; falling back to simple logging progress.")

        self.evaluator = WalletEvaluator(
            z_score_threshold=z_score_threshold,
            min_wallet_notional=min_wallet_notional,
            label_metric=label_metric,
        )

        self.min_wallet_notional = float(min_wallet_notional)
        self.label_metric = label_metric
        self.include_recidivism = include_recidivism
        self.enable_layer2_attribution = bool(enable_layer2_attribution)
        self.usdc_cache_db = usdc_cache_db
        self.polygonscan_api_key = polygonscan_api_key
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.jump_anticipation_config = jump_anticipation_config
        self._attribution_provider = None

        valid_metrics = {
            "f1",
            "f0_5",
            "f2",
            "precision",
            "recall",
            "mean_net_pnl_flagged",
            "median_net_pnl_flagged",
            "mean_return_flagged",
            "median_return_flagged",
            "median_informed_score_flagged",
            # Trade-level objectives
            "trade_mean_return_diff",
            "trade_t_stat",
            "trade_cohens_d",
            "trade_flagged_mean_return",
            "trade_flagged_win_rate",
            *EXTENDED_TRADE_OBJECTIVES,
            # F-beta econ objectives (require trade-level legs)
            "f_beta_econ_winrate",
            "f_beta_econ_mean_return",
            "f_beta_econ_winrate_f1",
            "f_beta_econ_mean_return_f1",
            "f_beta_econ_return_f1",
            "f_beta_econ_t_stat_f1",
            "f_beta_econ_winrate_f0_5",
            "f_beta_econ_mean_return_f0_5",
            "f_beta_econ_winrate_return",
        }

        if objective_metric not in valid_metrics:
            raise ValueError(f"objective_metric must be one of {sorted(valid_metrics)}")
        self.objective_metric = objective_metric

        # Caches populated during optimize()
        self.base_results_cache: Dict[int, BacktestResult] = {}  # detector results (lightweight)
        self.graph_trades_cache: Dict[int, List[Trade]] = {}     # size-filtered trades for graph
        self.detector_trades_cache: Dict[int, List[Trade]] = {}
        self.frozen_signal_cache: Dict[int, FrozenSignalCache] = {}
        self.ground_truth_cache: Dict[int, Optional[PrecomputedMarketGroundTruth]] = {}
        self.trade_returns_cache: Dict[int, Optional[PrecomputedTradeReturns]] = {}

    def _objective(self, metrics: Dict) -> float:
        """Extract objective value from metrics dict."""
        return float(metrics.get(self.objective_metric, 0.0))

    def _effective_min_usd_amount(self) -> Optional[float]:
        if not self.enable_trade_prefilter:
            return None
        return self.min_usd_amount

    @staticmethod
    def format_classification_metrics(metrics: Dict) -> str:
        """Format classification metrics for logging."""
        tp = int(metrics.get("true_positives", 0))
        fp = int(metrics.get("false_positives", 0))
        fn = int(metrics.get("false_negatives", 0))
        tn = int(metrics.get("true_negatives", 0))
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        f1 = float(metrics.get("f1", 0.0))
        f0_5 = float(metrics.get("f0_5", 0.0))

        insiders_total = tp + fn
        tp_pct = (tp / insiders_total * 100.0) if insiders_total > 0 else 0.0

        return (
            f"TP={tp} FP={fp} FN={fn} TN={tn} | "
            f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} F0.5={f0_5:.4f} | "
            f"Insiders correctly flagged={tp}/{insiders_total} ({tp_pct:.2f}%)"
        )

    def optimize(
        self,
        loader: HistoricalDataLoader,
        market_ids: List[int],
        n_passes: int = 1,
        optimize_order: Optional[List[str]] = None,
        initial_config: Optional[Dict] = None,
    ) -> Tuple[Dict, pd.DataFrame, Dict]:
        """Run coordinate descent optimization for clustering parameters."""
        logging.info("\n" + "=" * 80)
        logging.info("CLUSTERING COORDINATE DESCENT OPTIMIZATION")
        logging.info("=" * 80)

        if optimize_order is None:
            optimize_order = [
                "boost_magnitude",
                "boost_weights",
                "boost_normalizer",
                "clustering",
                "time_window",
                "size",
            ]

        valid_groups = set(ClusteringParameterGrid.get_parameter_groups().keys())
        optimize_order = [g for g in optimize_order if g in valid_groups]

        self.data_dir = getattr(loader, "data_dir", self.data_dir)

        logging.info(f"Markets: {market_ids}")
        logging.info(f"Prediction mode: {self.prediction_mode}")
        logging.info(f"Objective metric: {self.objective_metric}")
        logging.info(f"Detector config: FROZEN from Stage 1")
        logging.info(
            f"Parallel settings: max_workers={self.max_workers}, "
            f"coarse={self.parallelize_coarse}, full={self.parallelize_full}"
        )
        quiet_experiment_logs = experiment_backtest_logs_quiet()

        logging.getLogger("backtesting.wallet_evaluator").setLevel(logging.WARNING)
        logging.getLogger("clustering.cluster_computer").setLevel(logging.WARNING)
        logging.getLogger("clustering.ownership_analyser").setLevel(logging.WARNING)

        logging.info("\n" + "=" * 80)
        logging.info("PHASE 1: RUNNING DETECTORS (ONE-TIME, FULL TRADE SET)")
        logging.info("=" * 80)
        logging.info("Detector config is FROZEN. Results will be reused for all clustering configs.")

        detector_runner = BacktestRunner(
            config=self.detector_config,
            include_recidivism=self.include_recidivism,
        )

        phase1_start = time.time()
        for idx, market_id in enumerate(market_ids):
            if not quiet_experiment_logs:
                logging.info(f"\nMarket {idx + 1}/{len(market_ids)}: {market_id}")

            # Load replay trades, then derive detector trades in-memory
            try:
                replay_trades = loader.get_trades_for_market(
                    market_id,
                    min_usd_amount=None,
                    use_cache=False,
                )
            except TypeError:
                replay_trades = loader.get_trades_for_market(market_id)
            try:
                ground_truth_trades = loader.get_trades_for_market(
                    market_id,
                    min_usd_amount=None,
                    use_cache=False,
                    ignore_trade_time_bounds=True,
                )
            except TypeError:
                ground_truth_trades = replay_trades
            effective_min = self._effective_min_usd_amount()
            if effective_min is not None:
                detector_trades = filter_trades_by_notional(replay_trades, effective_min)
            else:
                detector_trades = replay_trades
            if not quiet_experiment_logs:
                logging.info(
                    f"  Loaded {len(replay_trades):,} replay trades "
                    f"({len(detector_trades):,} after detector prefilter)"
                )

            # Run detectors on full trade set
            base_result = detector_runner.run_backtest(
                trades=detector_trades,
                market_metadata={"id": market_id, **dict(loader.get_market_metadata(market_id) or {})},
                capture_alerts=False,
                capture_trade_features=False,
                progress_every=500_000,
            )
            self.base_results_cache[market_id] = base_result
            self.detector_trades_cache[market_id] = detector_trades
            self.frozen_signal_cache[market_id] = build_frozen_signals(
                config=self.detector_config,
                trades=detector_trades,
                market_metadata={"id": market_id, **dict(loader.get_market_metadata(market_id) or {})},
                include_recidivism=self.include_recidivism,
            )

            metadata = dict(loader.get_market_metadata(market_id) or {})
            metadata["id"] = market_id
            self.ground_truth_cache[market_id] = precompute_ground_truth(
                trades=ground_truth_trades,
                market_metadata=metadata,
                label_metric=self.label_metric,
                z_score_threshold=self.z_score_threshold,
                min_wallet_notional=self.min_wallet_notional,
            )
            gt = self.ground_truth_cache.get(market_id)
            self.trade_returns_cache[market_id] = precompute_trade_returns(
                trades=replay_trades,
                market_metadata=metadata,
                winning_outcome=(gt.winning_outcome if gt is not None else None),
            )

            # Filter trades for graph construction
            graph_trades = filter_trades_by_notional(detector_trades, self.min_trade_size)
            self.graph_trades_cache[market_id] = graph_trades

            if not quiet_experiment_logs:
                logging.info(
                    f"  Detector results: {base_result.alerts_generated:,} alerts, "
                    f"{len(base_result.wallet_suspicion):,} wallets"
                )
                logging.info(
                    f"  Graph trades: {len(graph_trades):,} "
                    f"(${self.min_trade_size:,.0f}+ from {len(detector_trades):,} detector trades)"
                )

            # Free the loaded trade lists — caches hold the pieces needed later.
            del replay_trades
            del ground_truth_trades

        phase1_time = time.time() - phase1_start
        logging.info(f"\nPhase 1 complete in {phase1_time:.1f}s")

        logging.info("\n" + "=" * 80)
        logging.info("PHASE 2: CLUSTERING PARAMETER OPTIMIZATION")
        logging.info("=" * 80)

        coarse_market_ids = self._choose_coarse_markets(loader, market_ids)
        logging.info(f"Markets (full): {market_ids}")
        logging.info(f"Markets (coarse shortlist): {coarse_market_ids}")

        # Evaluate baseline configs
        current_best_config = initial_config or ClusteringParameterGrid.get_baseline_config()
        no_clustering_config = ClusteringParameterGrid.get_no_clustering_config()

        logging.info("\nEvaluating baseline configs...")
        baseline_eval = self._evaluate_config(
            current_best_config,
            loader,
            market_ids,
            progress_desc="baseline markets",
        )
        no_clustering_eval = self._evaluate_config(
            no_clustering_config,
            loader,
            market_ids,
            progress_desc="no-clustering markets",
        )

        baseline_metrics = baseline_eval["metrics"]
        no_clustering_metrics = no_clustering_eval["metrics"]
        current_best_metrics = baseline_metrics

        logging.info("Baseline (with clustering):")
        logging.info("  " + self.format_classification_metrics(baseline_metrics))
        logging.info("No clustering baseline:")
        logging.info("  " + self.format_classification_metrics(no_clustering_metrics))

        all_results: List[Dict] = [
            {
                "pass": -1,
                "param_group": "no_clustering_baseline",
                "config_id": -2,
                **no_clustering_metrics,
                "objective_metric": self.objective_metric,
                "objective_score": self._objective(no_clustering_metrics),
                "coarse_objective": None,
                "config_time_seconds": no_clustering_eval["elapsed_seconds"],
                "full_config_json": json.dumps(no_clustering_config, sort_keys=True),
            },
            {
                "pass": -1,
                "param_group": "baseline",
                "config_id": -1,
                **baseline_metrics,
                "objective_metric": self.objective_metric,
                "objective_score": self._objective(baseline_metrics),
                "coarse_objective": None,
                "config_time_seconds": baseline_eval["elapsed_seconds"],
                "full_config_json": json.dumps(current_best_config, sort_keys=True),
            }
        ]

        group_summaries: Dict = {}
        overall_start = time.time()

        # Coordinate descent loop
        for pass_num in range(n_passes):
            logging.info("\n" + "=" * 80)
            logging.info(f"PASS {pass_num + 1}/{n_passes}")
            logging.info("=" * 80)

            for group_idx, param_group in enumerate(optimize_order):
                result = self._optimize_param_group(
                    param_group=param_group,
                    base_config=current_best_config,
                    base_metrics=current_best_metrics,
                    loader=loader,
                    full_market_ids=market_ids,
                    coarse_market_ids=coarse_market_ids,
                    pass_num=pass_num,
                    group_idx=group_idx,
                    total_groups=len(optimize_order),
                )

                current_best_config = result["best_config"]
                current_best_metrics = result["best_metrics"]
                all_results.extend(result["results"])

                key = f"pass{pass_num}_{param_group}"
                group_summaries[key] = {
                    "param_group": param_group,
                    "pass": pass_num,
                    "objective_metric": self.objective_metric,
                    "best_objective": result["best_objective"],
                    "baseline_objective": result["baseline_objective"],
                    "best_f1": result["best_f1"],
                    "baseline_f1": result["baseline_f1"],
                    "best_params": result["best_params"],
                    "configs_tested_full": result["configs_tested_full"],
                    "configs_tested_coarse": result["configs_tested_coarse"],
                    "improvement_pct": result["improvement_pct"],
                }

        total_time = time.time() - overall_start
        results_df = pd.DataFrame(all_results)

        self._print_final_summary(
            results_df=results_df,
            group_summaries=group_summaries,
            total_time_seconds=total_time,
            n_passes=n_passes,
        )

        return current_best_config, results_df, group_summaries

    def _optimize_param_group(
        self,
        param_group: str,
        base_config: Dict,
        base_metrics: Dict,
        loader: HistoricalDataLoader,
        full_market_ids: List[int],
        coarse_market_ids: List[int],
        pass_num: int,
        group_idx: int,
        total_groups: int,
    ) -> Dict:
        """Optimize a single parameter group using coordinate descent."""
        logging.info("\n" + "=" * 80)
        logging.info(f"OPTIMIZING: {param_group} ({group_idx + 1}/{total_groups})")
        logging.info("=" * 80)

        baseline_obj = self._objective(base_metrics)
        baseline_f1 = float(base_metrics.get("f1", 0.0))

        configs_and_params = ClusteringParameterGrid.generate_configs_for_param_group(
            param_group_name=param_group,
            base_config=base_config,
        )

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
            logging.info(f"Coarse stage on {len(coarse_market_ids)} market(s)...")
            coarse_start = time.time()

            coarse_eval_map = self._evaluate_candidates(
                candidates=candidates,
                loader=loader,
                market_ids=coarse_market_ids,
                param_group=param_group,
                stage_label="coarse",
                allow_parallel=self.parallelize_coarse,
            )

            for item in candidates:
                cfg_id = item["config_id"]
                coarse_scores[cfg_id] = self._objective(coarse_eval_map[cfg_id]["metrics"])

            ranked = sorted(candidates, key=lambda x: coarse_scores[x["config_id"]], reverse=True)
            shortlisted = ranked[: self.coarse_top_k]

            logging.info(
                f"Coarse stage done in {(time.time() - coarse_start):.1f}s "
                f"| shortlisted={len(shortlisted)}"
            )

        logging.info(f"Full stage candidates: {len(shortlisted)}")

        full_eval_map = self._evaluate_candidates(
            candidates=shortlisted,
            loader=loader,
            market_ids=full_market_ids,
            param_group=param_group,
            stage_label="full",
            allow_parallel=self.parallelize_full,
        )

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
                "param_group": param_group,
                "config_id": cfg_id,
                **params,
                **metrics,
                "objective_metric": self.objective_metric,
                "objective_score": cfg_obj,
                "coarse_objective": coarse_scores.get(cfg_id),
                "config_time_seconds": eval_result["elapsed_seconds"],
                "full_config_json": json.dumps(cfg, sort_keys=True),
            }
            results.append(row)

            if cfg_obj > best_obj:
                best_obj = cfg_obj
                best_config = cfg
                best_metrics = metrics
                best_params = params

        improvement_pct = ((best_obj - baseline_obj) / baseline_obj * 100.0) if baseline_obj > 0 else 0.0

        logging.info(
            f"Done {param_group}: baseline_{self.objective_metric}={baseline_obj:.4f}, "
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

    def _evaluate_candidates(
        self,
        candidates: List[Dict],
        loader: HistoricalDataLoader,
        market_ids: List[int],
        param_group: str,
        stage_label: str,
        allow_parallel: bool,
    ) -> Dict[int, Dict]:
        """Evaluate multiple configs in parallel or serial."""
        if len(candidates) == 0:
            return {}

        worker_count = get_backtest_worker_count(
            self.max_workers,
            len(candidates),
            enable_layer2_attribution=self.enable_layer2_attribution,
            clustering_enabled=True,
        )
        can_parallel = allow_parallel and worker_count > 1 and len(candidates) > 1

        if not can_parallel:
            eval_map: Dict[int, Dict] = {}
            iterable = _iter_progress(
                candidates,
                show_progress=self.show_progress,
                desc=f"{param_group} {stage_label}",
                unit="cfg",
                leave=False,
            )
            for item in iterable:
                eval_map[item["config_id"]] = self._evaluate_config(
                    clustering_config=item["config"],
                    loader=loader,
                    market_ids=market_ids,
                )
            return eval_map

        logging.info(
            f"Parallel {stage_label} eval: {len(candidates)} configs on {worker_count} worker(s)"
        )

        # Send only requested market caches to workers.
        worker_base_results = {
            mid: self.base_results_cache[mid]
            for mid in market_ids
            if mid in self.base_results_cache
        }
        worker_graph_trades = {
            mid: self.graph_trades_cache[mid]
            for mid in market_ids
            if mid in self.graph_trades_cache
        }
        worker_detector_trades = {
            mid: self.detector_trades_cache[mid]
            for mid in market_ids
            if mid in self.detector_trades_cache
        }
        worker_frozen_caches = {
            mid: self.frozen_signal_cache[mid]
            for mid in market_ids
            if mid in self.frozen_signal_cache
        }
        worker_trade_returns = {
            mid: self.trade_returns_cache.get(mid)
            for mid in market_ids
        }

        eval_map: Dict[int, Dict] = {}
        market_ids_tuple = tuple(market_ids)

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_clustering_worker,
            initargs=(
                worker_base_results,
                worker_graph_trades,
                worker_detector_trades,
                worker_frozen_caches,
                self.ground_truth_cache,
                worker_trade_returns,
                self.detector_config,
                self.data_dir,
                self.include_recidivism,
                self.z_score_threshold,
                self.min_wallet_notional,
                self.label_metric,
                self.prediction_mode,
                self.suspicion_threshold,
                self.flag_rate_threshold,
                self.enable_layer2_attribution,
                self.usdc_cache_db,
                self.polygonscan_api_key,
                self.min_trade_size,
                self.poll_interval_seconds,
                self.jump_anticipation_config,
            ),
        ) as executor:
            future_to_cfg_id = {
                executor.submit(
                    _evaluate_clustering_config_worker,
                    item["config"],
                    market_ids_tuple,
                ): item["config_id"]
                for item in candidates
            }

            with _bar_progress(
                total=len(future_to_cfg_id),
                show_progress=self.show_progress,
                desc=f"{param_group} {stage_label}",
                unit="cfg",
                leave=False,
            ) as pbar:
                for future in as_completed(future_to_cfg_id):
                    cfg_id = future_to_cfg_id[future]
                    try:
                        eval_map[cfg_id] = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"{param_group} {stage_label} evaluation failed for config_id={cfg_id}"
                        ) from exc
                    pbar.update(1)

        return eval_map

    def _evaluate_config(
        self,
        clustering_config: Dict,
        loader: HistoricalDataLoader,
        market_ids: List[int],
        progress_desc: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a single clustering config on all markets (serial, main process).
        """
        start = time.time()

        all_wallet_evals: List[Dict] = []
        all_flagged_returns: List[np.ndarray] = []
        all_unflagged_returns: List[np.ndarray] = []
        all_flagged_notionals: List[np.ndarray] = []
        all_unflagged_notionals: List[np.ndarray] = []
        attribution_provider = self._get_attribution_provider()

        iter_markets = _iter_progress(
            market_ids,
            show_progress=self.show_progress,
            desc=progress_desc or "clustering markets",
            unit="mkt",
            leave=False,
        )

        for market_id in iter_markets:
            detector_trades = self.detector_trades_cache.get(market_id, [])
            frozen_cache = self.frozen_signal_cache.get(market_id)

            if frozen_cache is None:
                logging.warning(f"No frozen detector cache for market {market_id}, skipping")
                continue

            metadata = dict(loader.get_market_metadata(market_id) or {})
            metadata["id"] = market_id

            schedule = build_live_parity_boost_schedule(
                detector_trades=detector_trades,
                market_id=str(market_id),
                clustering_config=clustering_config,
                clustering_min_trade_size=self.min_trade_size,
                jump_anticipation_config=self.jump_anticipation_config,
                poll_interval_seconds=self.poll_interval_seconds,
                attribution_provider=attribution_provider,
                fetch_if_missing=self.enable_layer2_attribution,
            )

            runner = CachedBacktestRunner(
                config=self.detector_config,
                target_detector_group="alert_threshold",
                frozen_cache=frozen_cache,
                include_recidivism=self.include_recidivism,
                score_multipliers=schedule.score_multiplier_by_trade_idx,
                score_cap=schedule.score_cap,
                wallet_cluster_boost=schedule.final_wallet_cluster_boost,
                wallet_has_common_ownership=schedule.final_wallet_has_common_ownership,
            )
            boosted_result = runner.run_backtest(
                trades=detector_trades,
                market_metadata=metadata,
            )

            gt = self.ground_truth_cache.get(market_id)
            if gt is not None:
                wallet_evals = fast_evaluate_wallets(gt, boosted_result)
            else:
                wallet_evals = self.evaluator.evaluate_wallets(boosted_result, metadata)
            all_wallet_evals.extend(wallet_evals)

            precomputed_tr = self.trade_returns_cache.get(market_id)
            if precomputed_tr is not None:
                is_flagged = build_trade_flag_mask(precomputed_tr, boosted_result)
                flagged = precomputed_tr.returns[is_flagged]
                unflagged = precomputed_tr.returns[~is_flagged]
                if len(flagged) > 0:
                    all_flagged_returns.append(flagged)
                    all_flagged_notionals.append(precomputed_tr.notionals[is_flagged])
                if len(unflagged) > 0:
                    all_unflagged_returns.append(unflagged)
                    all_unflagged_notionals.append(precomputed_tr.notionals[~is_flagged])

        metrics = _calculate_metrics_from_wallet_evaluations(
            wallet_evaluations=all_wallet_evals,
            prediction_mode=self.prediction_mode,
            suspicion_threshold=self.suspicion_threshold,
            flag_rate_threshold=self.flag_rate_threshold,
        )
        _merge_trade_level_metrics(
            metrics,
            all_flagged_returns,
            all_unflagged_returns,
            all_flagged_notionals,
            all_unflagged_notionals,
        )

        return {
            "metrics": metrics,
            "wallet_evaluations": all_wallet_evals,
            "elapsed_seconds": time.time() - start,
        }

    def _get_attribution_provider(self):
        if self._attribution_provider is not None:
            return self._attribution_provider
        try:
            self._attribution_provider = build_attribution_provider(
                enable_layer2_attribution=self.enable_layer2_attribution,
                usdc_cache_db=self.usdc_cache_db,
                polygonscan_api_key=self.polygonscan_api_key,
            )
            if self._attribution_provider is None:
                return None
            counts = self._attribution_provider.get_cache_counts()
            logging.info(
                "USDC cache ready: "
                f"wallet_rows={counts['cached_wallet_rows']:,}, "
                f"transfer_rows={counts['cached_transfer_rows']:,}"
            )
        except Exception:
            self._attribution_provider = None
        return self._attribution_provider

    def _choose_coarse_markets(
        self,
        loader: HistoricalDataLoader,
        market_ids: List[int],
    ) -> List[int]:
        """Choose subset of markets for coarse evaluation."""
        counts: List[Tuple[int, int]] = []

        for market_id in market_ids:
            graph_trades = self.graph_trades_cache.get(market_id, [])
            counts.append((market_id, len(graph_trades)))

        if not counts:
            return market_ids[:1]

        coarse = [mid for mid, c in counts if c <= self.coarse_trade_cap]
        if coarse:
            return coarse

        counts.sort(key=lambda x: x[1])
        return [counts[0][0]]

    def _print_final_summary(
        self,
        results_df: pd.DataFrame,
        group_summaries: Dict,
        total_time_seconds: float,
        n_passes: int,
    ):
        """Print final optimization summary."""
        logging.info("\n" + "=" * 80)
        logging.info("CLUSTERING OPTIMIZATION COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Total time: {total_time_seconds / 3600:.2f} hours")
        logging.info(f"Rows in results: {len(results_df):,}")
        logging.info(f"Passes: {n_passes}")

        no_clustering_rows = results_df[results_df["param_group"] == "no_clustering_baseline"]
        baseline_rows = results_df[(results_df["param_group"] == "baseline") & (results_df["pass"] == -1)]

        no_clustering_obj = float(no_clustering_rows.iloc[0].get("objective_score", 0.0)) if len(no_clustering_rows) else 0.0
        baseline_obj = float(baseline_rows.iloc[0].get("objective_score", 0.0)) if len(baseline_rows) else 0.0

        best_idx = results_df["objective_score"].idxmax()
        best_row = results_df.loc[best_idx].to_dict() if len(results_df) else {}
        best_obj = float(best_row.get("objective_score", 0.0))

        logging.info(f"No clustering {self.objective_metric}: {no_clustering_obj:.4f}")
        logging.info(f"Baseline {self.objective_metric}: {baseline_obj:.4f}")
        logging.info(f"Best {self.objective_metric}: {best_obj:.4f}")

        if no_clustering_obj > 0:
            clustering_lift = ((baseline_obj - no_clustering_obj) / no_clustering_obj * 100.0)
            logging.info(f"Clustering lift over no-clustering: {clustering_lift:+.2f}%")

        if baseline_obj > 0:
            optimization_improvement = ((best_obj - baseline_obj) / baseline_obj * 100.0)
            logging.info(f"Optimization improvement over baseline: {optimization_improvement:+.2f}%")

        if len(no_clustering_rows):
            logging.info("No clustering classification: " + self.format_classification_metrics(no_clustering_rows.iloc[0].to_dict()))
        if len(baseline_rows):
            logging.info("Baseline classification:     " + self.format_classification_metrics(baseline_rows.iloc[0].to_dict()))
        if best_row:
            logging.info("Best classification:         " + self.format_classification_metrics(best_row))

        for _, summary in sorted(group_summaries.items()):
            logging.info(
                f"{summary['param_group']:20s} pass={summary['pass']} "
                f"| best_{self.objective_metric}={summary['best_objective']:.4f} "
                f"| improvement={summary['improvement_pct']:+.2f}% "
                f"| coarse={summary['configs_tested_coarse']} "
                f"| full={summary['configs_tested_full']}"
            )
