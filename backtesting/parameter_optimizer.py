"""
Coordinate-descent-style parameter optimization for backtesting.
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

from backtesting.backtest_runner import BacktestRunner
from backtesting.causal_boost_replay import BoostSchedule
from backtesting.cached_evaluator import build_trade_flag_mask, precompute_trade_returns
from backtesting.fbeta_econ import finalize_fbeta_econ_metrics
from backtesting.trade_level_metrics import EXTENDED_TRADE_OBJECTIVES, merge_trade_level_metrics
from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation_support import (
    build_attribution_provider,
    evaluate_wallets_with_ground_truth,
    get_backtest_worker_count,
    load_all_trades_for_market,
    precompute_causal_boost_schedules,
    precompute_wallet_cluster_boosts,
)
from backtesting.parameter_grid import ParameterGrid
from backtesting.wallet_evaluator import WalletEvaluator
from backtesting.bucket_clustering_backtest_runner import BucketClusteringBacktestRunner
from models import filter_trades_by_notional

# Optional tqdm
try:
    from tqdm.auto import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except Exception:
    _tqdm = None
    _TQDM_AVAILABLE = False


class _NoOpPbar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, _n: int = 1):
        return None


def _iter_progress(iterable, show_progress: bool, **kwargs):
    if show_progress and _TQDM_AVAILABLE:
        return _tqdm(iterable, **kwargs)
    return iterable


def _bar_progress(total: int, show_progress: bool, **kwargs):
    if show_progress and _TQDM_AVAILABLE:
        return _tqdm(total=total, **kwargs)
    return _NoOpPbar()

def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _calculate_metrics_from_wallet_evaluations(
    wallet_evaluations: List[Dict],
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
) -> Dict:
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
            # Trade-level (populated separately, but need zero defaults)
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
            # F-beta econ composites (Recall/F1/F0.5 and t-stat+F1 legs)
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
        # Backward-compatible alias. In causal trade-level mode, boosts are
        # already reflected in which trades crossed alert_threshold.
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
        # Trade-level placeholders (populated by fast_trade_level_metrics in cached path)
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
_WORKER_LOADER: Optional[HistoricalDataLoader] = None
_WORKER_EVALUATOR: Optional[WalletEvaluator] = None
_WORKER_MIN_USD_AMOUNT: Optional[float] = None
_WORKER_INCLUDE_RECIDIVISM: bool = True
_WORKER_PREDICTION_MODE: str = "has_alert"
_WORKER_SUSPICION_THRESHOLD: float = 2.0
_WORKER_FLAG_RATE_THRESHOLD: float = 0.25
_WORKER_CLUSTERING_RUNNER: Optional[BucketClusteringBacktestRunner] = None
_WORKER_CLUSTERING_MIN_TRADE_SIZE: float = 5000.0
_WORKER_ATTRIBUTION_PROVIDER = None
_WORKER_PRECOMPUTED_CLUSTER_BOOSTS: Dict[int, Dict[str, float]] = {}
_WORKER_USE_PRECOMPUTED_CLUSTER_BOOSTS: bool = False
_WORKER_PRECOMPUTED_BOOST_SCHEDULES: Dict[int, BoostSchedule] = {}
_WORKER_USE_PRECOMPUTED_BOOST_SCHEDULES: bool = False


def _init_eval_worker(
    data_dir: str,
    min_usd_amount: Optional[float],
    include_recidivism: bool,
    z_score_threshold: float,
    min_wallet_notional: float,
    label_metric: str,
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
    clustering_config: Optional[Dict] = None,
    clustering_min_trade_size: float = 5000.0,
    enable_layer2_attribution: bool = False,
    usdc_cache_db: str = "data/usdc_transfers.db",
    polygonscan_api_key: Optional[str] = None,
    precomputed_cluster_boosts: Optional[Dict[int, Dict[str, float]]] = None,
    precomputed_boost_schedules: Optional[Dict[int, BoostSchedule]] = None,
):
    global _WORKER_LOADER
    global _WORKER_EVALUATOR
    global _WORKER_MIN_USD_AMOUNT
    global _WORKER_INCLUDE_RECIDIVISM
    global _WORKER_PREDICTION_MODE
    global _WORKER_SUSPICION_THRESHOLD
    global _WORKER_FLAG_RATE_THRESHOLD
    global _WORKER_CLUSTERING_RUNNER
    global _WORKER_CLUSTERING_MIN_TRADE_SIZE
    global _WORKER_ATTRIBUTION_PROVIDER
    global _WORKER_PRECOMPUTED_CLUSTER_BOOSTS
    global _WORKER_USE_PRECOMPUTED_CLUSTER_BOOSTS
    global _WORKER_PRECOMPUTED_BOOST_SCHEDULES
    global _WORKER_USE_PRECOMPUTED_BOOST_SCHEDULES

    logging.getLogger().setLevel(logging.WARNING)

    _WORKER_MIN_USD_AMOUNT = min_usd_amount
    _WORKER_INCLUDE_RECIDIVISM = include_recidivism
    _WORKER_PREDICTION_MODE = prediction_mode
    _WORKER_SUSPICION_THRESHOLD = float(suspicion_threshold)
    _WORKER_FLAG_RATE_THRESHOLD = float(flag_rate_threshold)
    _WORKER_CLUSTERING_MIN_TRADE_SIZE = float(clustering_min_trade_size)
    _WORKER_PRECOMPUTED_CLUSTER_BOOSTS = precomputed_cluster_boosts or {}
    _WORKER_USE_PRECOMPUTED_CLUSTER_BOOSTS = precomputed_cluster_boosts is not None
    _WORKER_PRECOMPUTED_BOOST_SCHEDULES = precomputed_boost_schedules or {}
    _WORKER_USE_PRECOMPUTED_BOOST_SCHEDULES = precomputed_boost_schedules is not None

    _WORKER_LOADER = HistoricalDataLoader(data_dir=data_dir, cache_size=0)
    _WORKER_LOADER.load_data()

    _WORKER_EVALUATOR = WalletEvaluator(
        z_score_threshold=z_score_threshold,
        min_wallet_notional=min_wallet_notional,
        label_metric=label_metric,
    )

    _WORKER_ATTRIBUTION_PROVIDER = None

    if (
        clustering_config is not None
        and not _WORKER_USE_PRECOMPUTED_CLUSTER_BOOSTS
        and not _WORKER_USE_PRECOMPUTED_BOOST_SCHEDULES
    ):
        _WORKER_ATTRIBUTION_PROVIDER = build_attribution_provider(
            enable_layer2_attribution=enable_layer2_attribution,
            usdc_cache_db=usdc_cache_db,
            polygonscan_api_key=polygonscan_api_key,
        )
        _WORKER_CLUSTERING_RUNNER = BucketClusteringBacktestRunner(
            detector_config={},  # not used for run_boost_only
            clustering_config=clustering_config,
            attribution_provider=_WORKER_ATTRIBUTION_PROVIDER,
        )
    else:
        _WORKER_CLUSTERING_RUNNER = None



def _safe_get_trades_worker(market_id: int):
    if _WORKER_LOADER is None:
        raise RuntimeError("Worker loader was not initialized")
    try:
        return _WORKER_LOADER.get_trades_for_market(
            market_id=market_id,
            min_usd_amount=_WORKER_MIN_USD_AMOUNT,
            use_cache=False,
        )
    except TypeError:
        return _WORKER_LOADER.get_trades_for_market(market_id)


def _evaluate_config_worker(config: Dict, market_ids: Tuple[int, ...]) -> Dict:
    if _WORKER_LOADER is None or _WORKER_EVALUATOR is None:
        raise RuntimeError("Worker state not initialized")

    start = time.time()
    runner = BacktestRunner(config=config, include_recidivism=_WORKER_INCLUDE_RECIDIVISM)
    all_wallet_evals: List[Dict] = []

    for market_id in market_ids:
        # Replay tape follows any active trade-time filter; ground truth always
        # uses the complete market history.
        try:
            replay_trades = _WORKER_LOADER.get_trades_for_market(
                market_id=market_id, min_usd_amount=None, use_cache=False
            )
        except TypeError:
            replay_trades = _WORKER_LOADER.get_trades_for_market(market_id)
        try:
            ground_truth_trades = _WORKER_LOADER.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=None,
                use_cache=False,
                ignore_trade_time_bounds=True,
            )
        except TypeError:
            ground_truth_trades = replay_trades

        if _WORKER_MIN_USD_AMOUNT is not None:
            detector_trades = filter_trades_by_notional(replay_trades, _WORKER_MIN_USD_AMOUNT)
        else:
            detector_trades = replay_trades

        metadata = dict(_WORKER_LOADER.get_market_metadata(market_id) or {})
        metadata["id"] = market_id

        schedule = (
            _WORKER_PRECOMPUTED_BOOST_SCHEDULES.get(market_id)
            if _WORKER_USE_PRECOMPUTED_BOOST_SCHEDULES
            else None
        )

        # Detectors see filtered trades; optional causal per-trade multiplier schedule.
        backtest_result = runner.run_backtest(
            trades=detector_trades,
            market_metadata=metadata,
            capture_alerts=False,
            capture_trade_features=False,
            progress_every=0,
            score_multipliers=(schedule.score_multiplier_by_trade_idx if schedule is not None else None),
            score_cap=(schedule.score_cap if schedule is not None else 0.95),
            wallet_cluster_boost=(
                schedule.final_wallet_cluster_boost if schedule is not None else None
            ),
            wallet_has_common_ownership=(
                schedule.final_wallet_has_common_ownership if schedule is not None else None
            ),
        )

        # Legacy post-hoc path if no causal schedule was provided.
        if schedule is None:
            if _WORKER_USE_PRECOMPUTED_CLUSTER_BOOSTS:
                BucketClusteringBacktestRunner.apply_precomputed_wallet_boosts(
                    backtest_result,
                    _WORKER_PRECOMPUTED_CLUSTER_BOOSTS.get(market_id, {}),
                )
            elif _WORKER_CLUSTERING_RUNNER is not None:
                graph_trades = filter_trades_by_notional(
                    detector_trades,
                    _WORKER_CLUSTERING_MIN_TRADE_SIZE,
                )
                backtest_result = _WORKER_CLUSTERING_RUNNER.run_boost_only(
                    base_result=backtest_result,
                    graph_trades=graph_trades,
                    market_id=str(market_id),
                )

        wallet_evals = evaluate_wallets_with_ground_truth(
            all_trades=ground_truth_trades,
            backtest_result=backtest_result,
            market_metadata=metadata,
            evaluator=_WORKER_EVALUATOR,
        )
        all_wallet_evals.extend(wallet_evals)

    metrics = _calculate_metrics_from_wallet_evaluations(
        wallet_evaluations=all_wallet_evals,
        prediction_mode=_WORKER_PREDICTION_MODE,
        suspicion_threshold=_WORKER_SUSPICION_THRESHOLD,
        flag_rate_threshold=_WORKER_FLAG_RATE_THRESHOLD,
    )

    return {
        "metrics": metrics,
        "elapsed_seconds": time.time() - start,
    }


class CoordinateDescentOptimizer:
    """
    Coordinate descent with:
    1) Coarse-to-full evaluation
    2) Optional parallel config evaluation
    3) Better end-of-run classification reporting
    """

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        min_wallet_notional: float = 500.0,
        label_metric: str = "return",
        prediction_mode: str = "has_alert",  # "has_alert" or "suspicion_threshold"
        suspicion_threshold: float = 2.0,
        coarse_top_k: int = 25,
        coarse_trade_cap: int = 250_000,
        min_usd_amount: Optional[float] = 500.0,
        enable_trade_prefilter: bool = False,
        data_dir: str = "data",
        max_workers: Optional[int] = None,
        parallelize_coarse: bool = True,
        parallelize_full: bool = True,
        show_progress: bool = True,
        objective_metric: str = "f1",  # f1, f0_5, f2, precision, recall
        flag_rate_threshold: float = 0.25,
        clustering_config: Optional[Dict] = None,
        clustering_min_trade_size: float = 5000.0,
        ja_config: Optional[Dict] = None,
        enable_layer2_attribution: bool = False,
        usdc_cache_db: str = "data/usdc_transfers.db",
        polygonscan_api_key: Optional[str] = None,
        use_causal_trade_level_boosts: bool = True,
        poll_interval_seconds: float = 5.0,
    ):
        self.z_score_threshold = z_score_threshold
        self.prediction_mode = prediction_mode
        self.suspicion_threshold = float(suspicion_threshold)
        self.coarse_top_k = int(coarse_top_k)
        self.coarse_trade_cap = int(coarse_trade_cap)
        self.min_usd_amount = min_usd_amount
        self.enable_trade_prefilter = enable_trade_prefilter
        self.data_dir = data_dir

        self.parallelize_coarse = parallelize_coarse
        self.parallelize_full = parallelize_full
        self.show_progress = show_progress
        self.flag_rate_threshold = float(flag_rate_threshold)

        self.clustering_config = clustering_config
        self.clustering_min_trade_size = float(clustering_min_trade_size)
        self.ja_config = ja_config
        self.enable_layer2_attribution = bool(enable_layer2_attribution)
        self.usdc_cache_db = usdc_cache_db
        self.polygonscan_api_key = polygonscan_api_key
        self.use_causal_trade_level_boosts = bool(use_causal_trade_level_boosts)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._attribution_provider = None
        self._precomputed_cluster_boosts_by_market: Dict[int, Dict[str, float]] = {}
        self._precomputed_boost_schedules_by_market: Dict[int, BoostSchedule] = {}
        self._precomputed_boost_schedule_cache_key: Optional[str] = None
        
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
        self.include_recidivism = True

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
            # F-beta econ objectives: harmonic F-beta over an economic signal
            # and a wallet-level classification leg (recall/f1/f0.5).
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

    def _objective(self, metrics: Dict) -> float:
        return float(metrics.get(self.objective_metric, 0.0))

    def _effective_min_usd_amount(self) -> Optional[float]:
        if not self.enable_trade_prefilter:
            return None
        return self.min_usd_amount

    def _get_attribution_provider(self):
        if self.clustering_config is None:
            return None
        if self._attribution_provider is not None:
            return self._attribution_provider
        self._attribution_provider = build_attribution_provider(
            enable_layer2_attribution=self.enable_layer2_attribution,
            usdc_cache_db=self.usdc_cache_db,
            polygonscan_api_key=self.polygonscan_api_key,
        )
        return self._attribution_provider

    def _get_precomputed_cluster_boosts(
        self,
        loader,
        market_ids: List[int],
    ) -> Optional[Dict[int, Dict[str, float]]]:
        if self.clustering_config is None:
            return None

        missing_market_ids = [
            market_id
            for market_id in market_ids
            if market_id not in self._precomputed_cluster_boosts_by_market
        ]
        if missing_market_ids:
            computed = precompute_wallet_cluster_boosts(
                loader=loader,
                market_ids=missing_market_ids,
                clustering_config=self.clustering_config,
                clustering_min_trade_size=self.clustering_min_trade_size,
                min_usd_amount=self._effective_min_usd_amount(),
                enable_layer2_attribution=self.enable_layer2_attribution,
                usdc_cache_db=self.usdc_cache_db,
                polygonscan_api_key=self.polygonscan_api_key,
            )
            self._precomputed_cluster_boosts_by_market.update(computed)

        return {
            market_id: self._precomputed_cluster_boosts_by_market.get(market_id, {})
            for market_id in market_ids
        }

    def _boost_schedule_cache_key(self) -> str:
        payload = {
            "clustering_config": self.clustering_config,
            "jump_anticipation_config": self.ja_config,
            "effective_min_usd_amount": self._effective_min_usd_amount(),
            "clustering_min_trade_size": self.clustering_min_trade_size,
            "poll_interval_seconds": self.poll_interval_seconds,
            "enable_layer2_attribution": self.enable_layer2_attribution,
            "use_causal_trade_level_boosts": self.use_causal_trade_level_boosts,
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _get_precomputed_boost_schedules(
        self,
        loader,
        market_ids: List[int],
    ) -> Optional[Dict[int, BoostSchedule]]:
        """
        Return per-market causal boost schedules for fixed clustering/JA settings.
        """
        if not self.use_causal_trade_level_boosts:
            return None
        if self.clustering_config is None and self.ja_config is None:
            return {}

        cache_key = self._boost_schedule_cache_key()
        if self._precomputed_boost_schedule_cache_key != cache_key:
            self._precomputed_boost_schedule_cache_key = cache_key
            self._precomputed_boost_schedules_by_market = {}

        missing_market_ids = [
            market_id
            for market_id in market_ids
            if market_id not in self._precomputed_boost_schedules_by_market
        ]
        if missing_market_ids:
            computed = precompute_causal_boost_schedules(
                loader=loader,
                market_ids=missing_market_ids,
                clustering_config=self.clustering_config,
                jump_anticipation_config=self.ja_config,
                min_usd_amount=self._effective_min_usd_amount(),
                clustering_min_trade_size=self.clustering_min_trade_size,
                poll_interval_seconds=self.poll_interval_seconds,
                enable_layer2_attribution=self.enable_layer2_attribution,
                usdc_cache_db=self.usdc_cache_db,
                polygonscan_api_key=self.polygonscan_api_key,
            )
            self._precomputed_boost_schedules_by_market.update(computed)

        return {
            market_id: self._precomputed_boost_schedules_by_market.get(market_id)
            for market_id in market_ids
            if market_id in self._precomputed_boost_schedules_by_market
        }

    @staticmethod
    def format_classification_metrics(metrics: Dict) -> str:
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
        loader,
        market_ids: List[int],
        n_passes: int = 1,
        optimize_order: Optional[List[str]] = None,
        initial_config: Optional[Dict] = None,
    ) -> Tuple[Dict, pd.DataFrame, Dict]:
        logging.info("\n" + "=" * 80)
        logging.info("COORDINATE DESCENT OPTIMIZATION")
        logging.info("=" * 80)

        if optimize_order is None:
            optimize_order = [
                "volume_anomaly",
                "probability_impact",
                "accumulation_detector",
                "recidivism_detector",
                "extreme_position",
                "contra_outcome_silence",
                "alert_threshold",
            ]

        valid_groups = set(ParameterGrid.get_detector_groups().keys())
        optimize_order = [d for d in optimize_order if d in valid_groups]

        self.data_dir = getattr(loader, "data_dir", self.data_dir)

        coarse_market_ids = self._choose_coarse_markets(loader, market_ids)

        logging.info(f"Markets (full): {market_ids}")
        logging.info(f"Markets (coarse shortlist): {coarse_market_ids}")
        logging.info(f"Prediction mode: {self.prediction_mode}")
        logging.info(f"Objective metric: {self.objective_metric}")
        logging.info(f"z-score threshold: {self.z_score_threshold}")
        logging.info(
            f"Trade prefilter: enabled={self.enable_trade_prefilter} "
            f"| threshold={self.min_usd_amount} "
            f"| effective_min_usd_amount={self._effective_min_usd_amount()}"
        )
        logging.info(
            f"Parallel settings: max_workers={self.max_workers}, "
            f"coarse={self.parallelize_coarse}, full={self.parallelize_full}"
        )

        if initial_config is None:
            current_best_config = ParameterGrid.get_baseline_config()
        else:
            current_best_config = initial_config

        baseline_eval = self._evaluate_config(current_best_config, loader, market_ids)
        current_best_metrics = baseline_eval["metrics"]

        all_results: List[Dict] = [
            {
                "pass": -1,
                "detector": "baseline",
                "config_id": -1,
                **current_best_metrics,
                "objective_metric": self.objective_metric,
                "objective_score": self._objective(current_best_metrics),
                "coarse_objective": None,
                "coarse_f1": None,
                "config_time_seconds": baseline_eval["elapsed_seconds"],
                "full_config_json": json.dumps(current_best_config, sort_keys=True),
            }
        ]

        detector_summaries: Dict = {}
        overall_start = time.time()

        for pass_num in range(n_passes):
            logging.info("\n" + "=" * 80)
            logging.info(f"PASS {pass_num + 1}/{n_passes}")
            logging.info("=" * 80)

            for detector_idx, detector_name in enumerate(optimize_order):
                result = self._optimize_detector(
                    detector_name=detector_name,
                    base_config=current_best_config,
                    base_metrics=current_best_metrics,
                    loader=loader,
                    full_market_ids=market_ids,
                    coarse_market_ids=coarse_market_ids,
                    pass_num=pass_num,
                    detector_idx=detector_idx,
                    total_detectors=len(optimize_order),
                )

                current_best_config = result["best_config"]
                current_best_metrics = result["best_metrics"]
                all_results.extend(result["results"])

                key = f"pass{pass_num}_{detector_name}"
                detector_summaries[key] = {
                    "detector": detector_name,
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
            detector_summaries=detector_summaries,
            total_time_seconds=total_time,
            n_passes=n_passes,
        )

        return current_best_config, results_df, detector_summaries

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
        logging.info("\n" + "=" * 80)
        logging.info(f"OPTIMIZING: {detector_name} ({detector_idx + 1}/{total_detectors})")
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
            logging.info(f"Coarse stage on {len(coarse_market_ids)} market(s)...")
            coarse_start = time.time()

            coarse_eval_map = self._evaluate_candidates(
                candidates=candidates,
                loader=loader,
                market_ids=coarse_market_ids,
                detector_name=detector_name,
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
            detector_name=detector_name,
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
            cfg_f1 = float(metrics.get("f1", 0.0))

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

        improvement_pct = ((best_obj - baseline_obj) / baseline_obj * 100.0) if baseline_obj > 0 else 0.0

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

    def _evaluate_candidates(
        self,
        candidates: List[Dict],
        loader,
        market_ids: List[int],
        detector_name: str,
        stage_label: str,
        allow_parallel: bool,
    ) -> Dict[int, Dict]:
        if len(candidates) == 0:
            return {}

        precomputed_boost_schedules = self._get_precomputed_boost_schedules(loader, market_ids)
        precomputed_cluster_boosts = None
        if precomputed_boost_schedules is None:
            precomputed_cluster_boosts = self._get_precomputed_cluster_boosts(loader, market_ids)
        worker_count = get_backtest_worker_count(
            self.max_workers,
            len(candidates),
            enable_layer2_attribution=self.enable_layer2_attribution,
            clustering_enabled=self.clustering_config is not None,
            live_layer2_fetches=(
                precomputed_boost_schedules is None and precomputed_cluster_boosts is None
            ),
        )
        can_parallel = allow_parallel and worker_count > 1 and len(candidates) > 1
        if not can_parallel:
            eval_map: Dict[int, Dict] = {}
            iterable = _iter_progress(
                candidates,
                show_progress=self.show_progress,
                desc=f"{detector_name} {stage_label}",
                unit="cfg",
                leave=False,
            )
            for item in iterable:
                eval_map[item["config_id"]] = self._evaluate_config(
                    config=item["config"],
                    loader=loader,
                    market_ids=market_ids,
                    precomputed_cluster_boosts=precomputed_cluster_boosts,
                    precomputed_boost_schedules=precomputed_boost_schedules,
                )
            return eval_map

        logging.info(
            f"Parallel {stage_label} eval: {len(candidates)} configs on {worker_count} worker(s)"
        )

        eval_map: Dict[int, Dict] = {}
        market_ids_tuple = tuple(market_ids)

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_eval_worker,
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
            ),
        ) as executor:
            future_to_cfg_id = {
                executor.submit(_evaluate_config_worker, item["config"], market_ids_tuple): item["config_id"]
                for item in candidates
            }

            with _bar_progress(
                total=len(future_to_cfg_id),
                show_progress=self.show_progress,
                desc=f"{detector_name} {stage_label}",
                unit="cfg",
                leave=False,
            ) as pbar:
                for future in as_completed(future_to_cfg_id):
                    cfg_id = future_to_cfg_id[future]
                    try:
                        eval_map[cfg_id] = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"{detector_name} {stage_label} evaluation failed for config_id={cfg_id}"
                        ) from exc
                    pbar.update(1)

        return eval_map

    def _safe_get_trades(self, loader, market_id: int):
        effective_min_usd_amount = self._effective_min_usd_amount()
        try:
            return loader.get_trades_for_market(
                market_id,
                min_usd_amount=effective_min_usd_amount,
                use_cache=False,
            )
        except TypeError:
            return loader.get_trades_for_market(market_id)
    
    def _safe_get_trades_unfiltered(self, loader, market_id: int):
        """Load ALL trades for a market with no prefilter. Used for ground truth."""
        return load_all_trades_for_market(loader, market_id)

    def _evaluate_config(
        self,
        config: Dict,
        loader,
        market_ids: List[int],
        precomputed_cluster_boosts: Optional[Dict[int, Dict[str, float]]] = None,
        precomputed_boost_schedules: Optional[Dict[int, BoostSchedule]] = None,
    ) -> Dict:
        start = time.time()

        runner = BacktestRunner(config=config, include_recidivism=self.include_recidivism)
        all_wallet_evals: List[Dict] = []
        if precomputed_boost_schedules is None:
            precomputed_boost_schedules = self._get_precomputed_boost_schedules(loader, market_ids)
        if precomputed_boost_schedules is None and precomputed_cluster_boosts is None:
            precomputed_cluster_boosts = self._get_precomputed_cluster_boosts(loader, market_ids)

        clustering_runner = None
        if (
            self.clustering_config is not None
            and precomputed_cluster_boosts is None
            and precomputed_boost_schedules is None
        ):
            clustering_runner = BucketClusteringBacktestRunner(
                detector_config={},
                clustering_config=self.clustering_config,
                attribution_provider=self._get_attribution_provider(),
            )

        all_flagged_returns: List = []
        all_unflagged_returns: List = []
        all_flagged_notionals: List = []
        all_unflagged_notionals: List = []

        for market_id in market_ids:
            # Replay tape follows any active trade-time filter; ground truth
            # always uses the complete market history.
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

            effective_min = self._effective_min_usd_amount()
            if effective_min is not None:
                detector_trades = filter_trades_by_notional(replay_trades, effective_min)
            else:
                detector_trades = replay_trades

            schedule = (
                precomputed_boost_schedules.get(market_id)
                if precomputed_boost_schedules is not None
                else None
            )

            # Detectors see filtered trades
            backtest_result = runner.run_backtest(
                trades=detector_trades,
                market_metadata=metadata,
                capture_alerts=False,
                capture_trade_features=False,
                progress_every=0,
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

            if schedule is None:
                if precomputed_cluster_boosts is not None:
                    BucketClusteringBacktestRunner.apply_precomputed_wallet_boosts(
                        backtest_result,
                        precomputed_cluster_boosts.get(market_id, {}),
                    )
                elif clustering_runner is not None:
                    graph_trades = filter_trades_by_notional(
                        detector_trades,
                        self.clustering_min_trade_size,
                    )
                    backtest_result = clustering_runner.run_boost_only(
                        base_result=backtest_result,
                        graph_trades=graph_trades,
                        market_id=str(market_id),
                    )

                # Legacy jump boost path (causal schedule disabled/unavailable).
                if getattr(self, "ja_config", None) is not None:
                    from jump_anticipation.core import run_jump_anticipation_boost
                    run_jump_anticipation_boost(
                        result=backtest_result,
                        all_trades=replay_trades,
                        config=self.ja_config,
                        scoring_trades=detector_trades,
                    )

            wallet_evals = evaluate_wallets_with_ground_truth(
                all_trades=ground_truth_trades,
                backtest_result=backtest_result,
                market_metadata=metadata,
                evaluator=self.evaluator,
            )
            all_wallet_evals.extend(wallet_evals)

            # Trade-level objective follows the replay tape.
            ptr = precompute_trade_returns(trades=replay_trades, market_metadata=metadata)
            if ptr is not None:
                is_flagged = build_trade_flag_mask(ptr, backtest_result)
                f_ret = ptr.returns[is_flagged]
                u_ret = ptr.returns[~is_flagged]
                if len(f_ret) > 0:
                    all_flagged_returns.append(f_ret)
                    all_flagged_notionals.append(ptr.notionals[is_flagged])
                if len(u_ret) > 0:
                    all_unflagged_returns.append(u_ret)
                    all_unflagged_notionals.append(ptr.notionals[~is_flagged])

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

        return {
            "metrics": metrics,
            "wallet_evaluations": all_wallet_evals,
            "elapsed_seconds": time.time() - start,
        }

    def _choose_coarse_markets(self, loader, market_ids: List[int]) -> List[int]:
        counts: List[Tuple[int, int]] = []
        effective_min_usd_amount = self._effective_min_usd_amount()

        for market_id in market_ids:
            try:
                count = loader.get_trade_count(
                    market_id=market_id,
                    min_usd_amount=effective_min_usd_amount,
                    expanded=True,
                )
            except TypeError:
                count = loader.get_trade_count(market_id)

            if count is not None:
                counts.append((market_id, int(count)))

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
        detector_summaries: Dict,
        total_time_seconds: float,
        n_passes: int,
    ):
        logging.info("\n" + "=" * 80)
        logging.info("COORDINATE DESCENT COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Total time: {total_time_seconds / 3600:.2f} hours")
        logging.info(f"Rows in results: {len(results_df):,}")
        logging.info(f"Passes: {n_passes}")

        baseline_rows = results_df[
            (results_df["detector"] == "baseline") & (results_df["pass"] == -1)
        ]

        baseline_f1 = float(baseline_rows.iloc[0].get("f1", 0.0)) if len(baseline_rows) else 0.0
        baseline_obj = float(baseline_rows.iloc[0].get("objective_score", 0.0)) if len(baseline_rows) else 0.0

        best_idx = results_df["objective_score"].idxmax() if "objective_score" in results_df.columns else results_df["f1"].idxmax()
        best_row = results_df.loc[best_idx].to_dict() if len(results_df) else {}
        best_f1 = float(best_row.get("f1", 0.0))
        best_obj = float(best_row.get("objective_score", 0.0))

        improvement_pct = ((best_obj - baseline_obj) / baseline_obj * 100.0) if baseline_obj > 0 else 0.0

        logging.info(f"Baseline {self.objective_metric}: {baseline_obj:.4f}")
        logging.info(f"Best {self.objective_metric}: {best_obj:.4f}")
        logging.info(f"Baseline F1: {baseline_f1:.4f}")
        logging.info(f"Best F1: {best_f1:.4f}")
        logging.info(f"Total improvement ({self.objective_metric}): {improvement_pct:+.2f}%")

        if len(baseline_rows):
            logging.info("Baseline classification: " + self.format_classification_metrics(baseline_rows.iloc[0].to_dict()))
        if best_row:
            logging.info("Best classification:     " + self.format_classification_metrics(best_row))

        for _, summary in sorted(detector_summaries.items()):
            logging.info(
                f"{summary['detector']} pass={summary['pass']} "
                f"| best_{self.objective_metric}={summary['best_objective']:.4f} "
                f"| best_f1={summary['best_f1']:.4f} "
                f"| improvement={summary['improvement_pct']:+.2f}% "
                f"| coarse={summary['configs_tested_coarse']} "
                f"| full={summary['configs_tested_full']}"
            )
