"""
jump_anticipation/optimizer.py

Stage 3: Coordinate descent over jump anticipation parameters.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from backtesting.backtest_runner import BacktestResult
from backtesting.cached_backtest_runner import (
    CachedBacktestRunner,
    FrozenSignalCache,
    build_frozen_signals,
)
from backtesting.causal_boost_replay import build_live_parity_boost_schedule
from backtesting.cached_evaluator import (
    PrecomputedMarketGroundTruth,
    PrecomputedTradeReturns,
    build_trade_flag_mask,
    fast_evaluate_wallets,
    precompute_trade_returns,
)
from backtesting.evaluation_support import build_attribution_provider
from backtesting.parameter_optimizer import _calculate_metrics_from_wallet_evaluations
from models import Trade

logger = logging.getLogger(__name__)


from backtesting.trade_level_metrics import EXTENDED_TRADE_OBJECTIVES, merge_trade_level_metrics


def get_ja_baseline_config() -> Dict:
    """
    Return the JA baseline config from the live CONFIG dict.
    """
    from config import CONFIG
    ja = CONFIG.get("jump_anticipation_config", {})
    return {
        "jump_threshold":            float(ja.get("jump_threshold", 0.05)),
        "jump_window_minutes":       float(ja.get("jump_window_minutes", 30)),
        "pre_jump_lookback_minutes": float(ja.get("pre_jump_lookback_minutes", 60)),
        "min_pre_jump_trades":       int(ja.get("min_pre_jump_trades", 2)),
        "max_boost_factor":          float(ja.get("max_boost_factor", 2.0)),
        "min_trade_notional":        float(ja.get("min_trade_notional", 0.0)),
    }

# Groups determine which parameters are swept together in one coordinate step.
JA_PARAMETER_GROUPS: Dict[str, List[str]] = {
    "boost":     ["max_boost_factor", "min_trade_notional"],
    "lookback":  ["pre_jump_lookback_minutes", "min_pre_jump_trades"],
    "detection": ["jump_threshold", "jump_window_minutes"],
}

JA_VARIATIONS: Dict[str, List] = {
    "jump_threshold":            [0.03, 0.05, 0.07, 0.10],
    "jump_window_minutes":       [15, 30, 60],
    "pre_jump_lookback_minutes": [30, 60, 90],
    "min_pre_jump_trades":       [1, 2, 3],
    "max_boost_factor":          [1.5, 2.0, 2.5, 3.0],
    "min_trade_notional":        [0.0, 500.0, 1000.0],
}

JA_VARIATIONS_SPARSE_RETURN: Dict[str, List] = {
    # A return-seeking jump signal should key off visible repricing, not every
    # small drift. 0.10 is a clean "ten cent move" threshold; 0.07 leaves one
    # exploratory lower step.
    "jump_threshold":            [0.05, 0.07, 0.10, 0.15],
    # Fifteen minutes catches very fast repricing; thirty minutes is the normal
    # short-horizon window; sixty minutes is a slower market reaction.
    "jump_window_minutes":       [15, 30, 60],
    # Keep attribution close to the jump so the signal is less likely to pick
    # up ordinary earlier trading.
    "pre_jump_lookback_minutes": [15, 30, 60],
    # Two or three pre-jump trades reduce single-print coincidences while still
    # allowing sparse markets to qualify.
    "min_pre_jump_trades":       [2, 3],
    # Keep boost as corroboration rather than the whole alert.
    "max_boost_factor":          [1.5, 2.0, 2.5],
    "min_trade_notional":        [0.0, 500.0, 1000.0],
}

class JumpAnticipationOptimizer:
    """
    Stage 3: coordinate descent over jump anticipation parameters.
    """

    def __init__(
        self,
        detector_config: Dict,
        clustering_config: Optional[Dict],
        clustering_min_trade_size: float = 5000.0,
        include_recidivism: bool = False,
        prediction_mode: str = "flag_rate",
        flag_rate_threshold: float = 0.2,
        suspicion_threshold: float = 2.0,
        objective_metric: str = "f0_5",
        n_passes: int = 1,
        show_progress: bool = True,
        poll_interval_seconds: float = 5.0,
        enable_layer2_attribution: bool = False,
        usdc_cache_db: str = "data/usdc_transfers.db",
        polygonscan_api_key: Optional[str] = None,
    ):
        self.detector_config = detector_config
        self.clustering_config = clustering_config
        self.clustering_min_trade_size = float(clustering_min_trade_size)
        self.include_recidivism = bool(include_recidivism)
        self.prediction_mode = prediction_mode
        self.flag_rate_threshold = flag_rate_threshold
        self.suspicion_threshold = suspicion_threshold
        self.objective_metric = objective_metric
        self.n_passes = n_passes
        self.show_progress = show_progress
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.enable_layer2_attribution = bool(enable_layer2_attribution)
        self.usdc_cache_db = usdc_cache_db
        self.polygonscan_api_key = polygonscan_api_key

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
            "trade_mean_return_diff",
            "trade_t_stat",
            "trade_cohens_d",
            "trade_flagged_mean_return",
            "trade_flagged_win_rate",
            *EXTENDED_TRADE_OBJECTIVES,
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

    def optimize(
        self,
        base_results: Dict[int, BacktestResult],
        all_trades: Dict[int, List[Trade]],
        scoring_trades: Dict[int, List[Trade]],
        precomputed_gts: Dict[int, PrecomputedMarketGroundTruth],
        initial_config: Optional[Dict] = None,
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Coordinate descent over JA parameters.

        Args:
            base_results:    Frozen Stage 1+2 BacktestResults per market.
                             Never mutated — each candidate copies wallet_cluster_boost.
            all_trades:      Unfiltered trade lists per market. Used for find_jumps
                             (dense price series).
            scoring_trades:  Filtered trade lists per market. Used for
                             score_wallets_jump_anticipation (consistency with
                             detector pipeline).
            precomputed_gts: Precomputed ground truth per market (from
                             precompute_ground_truth(), same as Stage 1).
            initial_config:  Starting JA config. Defaults to get_ja_baseline_config().

        Returns:
            (best_ja_config, results_df)
            results_df has one row per candidate evaluated, suitable for CSV export.
        """
        current_config = dict(initial_config or get_ja_baseline_config())
        rows: List[Dict] = []

        frozen_caches: Dict[int, FrozenSignalCache] = {}
        trade_returns_by_market: Dict[int, Optional[PrecomputedTradeReturns]] = {}
        for market_id, market_scoring_trades in scoring_trades.items():
            gt = precomputed_gts.get(market_id)
            market_slug = gt.market_slug if gt is not None else str(market_id)
            frozen_caches[market_id] = build_frozen_signals(
                config=self.detector_config,
                trades=market_scoring_trades,
                market_metadata={"id": market_id, "market_slug": market_slug},
                include_recidivism=self.include_recidivism,
            )
            market_all_trades = all_trades.get(market_id, market_scoring_trades)
            trade_returns_by_market[market_id] = precompute_trade_returns(
                trades=market_all_trades,
                market_metadata={"id": market_id, "market_slug": market_slug},
                winning_outcome=(gt.winning_outcome if gt is not None else None),
            )

        attribution_provider = build_attribution_provider(
            enable_layer2_attribution=self.enable_layer2_attribution,
            usdc_cache_db=self.usdc_cache_db,
            polygonscan_api_key=self.polygonscan_api_key,
        )

        # Evaluate baseline
        baseline_metrics = self._evaluate(
            current_config,
            base_results,
            all_trades,
            scoring_trades,
            precomputed_gts,
            frozen_caches,
            trade_returns_by_market,
            attribution_provider,
        )
        current_obj = baseline_metrics.get("objective_score", baseline_metrics.get(self.objective_metric, 0.0))

        logger.debug(
            f"[JA Stage 3] baseline: "
            f"{self.objective_metric}={current_obj:.4f} | "
            f"precision={baseline_metrics.get('precision', 0.0):.4f} | "
            f"recall={baseline_metrics.get('recall', 0.0):.4f} | "
            f"predicted={baseline_metrics.get('num_predicted', 0)} / "
            f"insiders={baseline_metrics.get('num_insiders', 0)}"
        )
        rows.append({
            "pass": -1, "group": "baseline", "param": "baseline", "value": "baseline",
            "objective_score": current_obj,
            **{k: v for k, v in baseline_metrics.items() if k != "objective_score"},
            **current_config,
        })

        for pass_idx in range(self.n_passes):
            pass_improved = False

            for group_name, param_names in JA_PARAMETER_GROUPS.items():
                for param_name in param_names:
                    best_value = current_config[param_name]
                    best_obj_for_param = current_obj

                    for value in JA_VARIATIONS.get(param_name, []):
                        if value == best_value:
                            continue

                        candidate = {**current_config, param_name: value}
                        metrics = self._evaluate(
                            candidate,
                            base_results,
                            all_trades,
                            scoring_trades,
                            precomputed_gts,
                            frozen_caches,
                            trade_returns_by_market,
                            attribution_provider,
                        )
                        obj = metrics.get("objective_score", metrics.get(self.objective_metric, 0.0))

                        rows.append({
                            "pass": pass_idx,
                            "group": group_name,
                            "param": param_name,
                            "value": value,
                            "objective_score": obj,
                            **{k: v for k, v in metrics.items() if k != "objective_score"},
                            **candidate,
                        })

                        if obj > best_obj_for_param:
                            best_obj_for_param = obj
                            best_value = value

                    if best_value != current_config[param_name]:
                        old = current_config[param_name]
                        current_config = {**current_config, param_name: best_value}
                        current_obj = best_obj_for_param
                        pass_improved = True
                        logger.debug(
                            f"[JA Stage 3] pass={pass_idx} | {param_name}: "
                            f"{old} -> {best_value} "
                            f"({self.objective_metric}={current_obj:.4f})"
                        )

            if not pass_improved:
                logger.debug(
                    f"[JA Stage 3] pass {pass_idx}: no improvement — stopping early"
                )
                break

        logger.debug(
            f"[JA Stage 3] complete: "
            f"{self.objective_metric}={current_obj:.4f} | "
            f"config={current_config}"
        )

        if attribution_provider is not None:
            attribution_provider.close()
        return current_config, pd.DataFrame(rows)

    def _evaluate(
        self,
        ja_config: Dict,
        base_results: Dict[int, BacktestResult],
        all_trades: Dict[int, List[Trade]],
        scoring_trades: Dict[int, List[Trade]],
        precomputed_gts: Dict[int, PrecomputedMarketGroundTruth],
        frozen_caches: Dict[int, FrozenSignalCache],
        trade_returns_by_market: Dict[int, Optional[PrecomputedTradeReturns]],
        attribution_provider,
    ) -> Dict[str, Any]:
        """Evaluate one JA config across all markets using trade-level replay."""
        all_wallet_evals: List[Dict] = []
        all_flagged_returns: List[np.ndarray] = []
        all_unflagged_returns: List[np.ndarray] = []
        all_flagged_notionals: List[np.ndarray] = []
        all_unflagged_notionals: List[np.ndarray] = []

        for market_id, _base_result in base_results.items():
            gt = precomputed_gts.get(market_id)
            if gt is None:
                continue

            market_all_trades = all_trades.get(market_id, [])
            market_scoring_trades = scoring_trades.get(market_id, market_all_trades)
            frozen_cache = frozen_caches.get(market_id)
            if frozen_cache is None:
                continue

            schedule = build_live_parity_boost_schedule(
                detector_trades=market_scoring_trades,
                market_id=str(market_id),
                clustering_config=self.clustering_config,
                clustering_min_trade_size=self.clustering_min_trade_size,
                jump_anticipation_config=ja_config,
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
            replay_result = runner.run_backtest(
                trades=market_scoring_trades,
                market_metadata={"id": market_id, "market_slug": gt.market_slug},
            )

            all_wallet_evals.extend(fast_evaluate_wallets(gt, replay_result))
            precomputed_tr = trade_returns_by_market.get(market_id)
            if precomputed_tr is not None:
                is_flagged = build_trade_flag_mask(precomputed_tr, replay_result)
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
        merge_trade_level_metrics(
            metrics,
            all_flagged_returns,
            all_unflagged_returns,
            all_flagged_notionals,
            all_unflagged_notionals,
        )
        return metrics


def print_ja_search_space():
    """Print summary of JA parameter search space."""
    print("\nJump Anticipation Parameter Search Space (Stage 3):")
    print("-" * 50)
    total = 0
    for group, params in JA_PARAMETER_GROUPS.items():
        group_total = sum(len(JA_VARIATIONS.get(p, [])) for p in params)
        total += group_total
        print(f"  {group:12s}: {len(params)} params = {group_total} configs")
    print("-" * 50)
    print(f"  {'Total':12s}:                 {total} configs per pass")

    baseline = get_ja_baseline_config()
    print("\n  Baseline (from CONFIG):")
    for k, v in baseline.items():
        print(f"    {k}: {v}")
