"""Multi-start coordinate descent over the detector parameter space."""

from __future__ import annotations

import logging
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtesting.cached_optimizer import CachedCoordinateDescentOptimizer
from backtesting.parameter_grid import ParameterGrid

logger = logging.getLogger(__name__)


VALID_START_STRATEGIES: Tuple[str, ...] = ("perturb", "random", "mixed")


_METRIC_COLUMNS: Tuple[str, ...] = (
    "num_wallets",
    "num_predicted_positive",
    "num_true_insiders",
    "precision",
    "recall",
    "f1",
    "f0_5",
    "f2",
    "true_positives",
    "false_positives",
    "false_negatives",
    "true_negatives",
    "mean_net_pnl_flagged",
    "median_net_pnl_flagged",
    "mean_return_flagged",
    "median_return_flagged",
    "median_informed_score_flagged",
    "trade_flagged_mean_return",
    "trade_flagged_mean_return_lcb",
    "trade_flagged_mean_return_se",
    "trade_flagged_weighted_return",
    "trade_flagged_weighted_return_lcb",
    "trade_flagged_weighted_return_se",
    "trade_mean_return_diff",
    "trade_t_stat",
    "trade_cohens_d",
    "trade_cohens_d_lcb",
    "trade_cohens_d_se",
    "trade_flagged_win_rate",
    "trade_weighted_mean_return_diff",
    "trade_weighted_cohens_d",
    "trade_weighted_cohens_d_lcb",
    "trade_weighted_cohens_d_se",
    "trade_winsorized_mean_return_diff",
    "trade_winsorized_cohens_d",
    "trade_winsorized_weighted_mean_return_diff",
    "trade_winsorized_weighted_cohens_d",
    "trade_winsorized_weighted_cohens_d_lcb",
    "trade_winsorized_weighted_cohens_d_se",
    "trade_weighted_flagged_effective_count",
    "trade_weighted_unflagged_effective_count",
    "econ_signal_winrate",
    "econ_signal_mean_return_norm",
    "econ_signal_return_sigmoid_norm",
    "econ_signal_trade_mean_return_sigmoid_norm",
    "econ_signal_trade_t_stat_norm",
    "f_beta_econ_winrate",
    "f_beta_econ_mean_return",
    "f_beta_econ_winrate_f1",
    "f_beta_econ_mean_return_f1",
    "f_beta_econ_return_f1",
    "f_beta_econ_t_stat_f1",
    "f_beta_econ_winrate_f0_5",
    "f_beta_econ_mean_return_f0_5",
    "f_beta_econ_winrate_return",
    "f_beta_econ_beta",
    "f_beta_econ_t_stat_scale",
    "f_beta_econ_return_scale",
)


@dataclass
class StartSummary:
    """Summary of one coordinate-descent trajectory."""

    start_idx: int
    label: str
    strategy: str
    baseline_objective: float
    train_final_objective: float
    final_objective: float
    elapsed_seconds: float
    configs_evaluated: int
    optimize_order: List[str] = field(default_factory=list)
    train_final_metrics: Dict[str, Any] = field(default_factory=dict)
    val_final_objective: float = float("nan")
    val_final_metrics: Dict[str, Any] = field(default_factory=dict)
    val_elapsed_seconds: float = 0.0
    initial_config: Dict[str, Any] = field(default_factory=dict)
    final_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_idx": self.start_idx,
            "label": self.label,
            "strategy": self.strategy,
            "baseline_objective": self.baseline_objective,
            "train_final_objective": self.train_final_objective,
            "val_final_objective": self.val_final_objective,
            "final_objective": self.final_objective,
            "train_val_gap": (
                self.train_final_objective - self.val_final_objective
                if _is_number(self.val_final_objective)
                else float("nan")
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "val_elapsed_seconds": self.val_elapsed_seconds,
            "configs_evaluated": self.configs_evaluated,
            "optimize_order": self.optimize_order,
            "train_final_metrics": self.train_final_metrics,
            "val_final_metrics": self.val_final_metrics,
            "initial_config": self.initial_config,
            "final_config": self.final_config,
        }


class MultiStartCoordinateDescentOptimizer:
    """Run cached coordinate descent from multiple starting configs."""

    def __init__(
        self,
        *,
        n_starts: int = 5,
        start_strategy: str = "perturb",
        perturb_prob: float = 0.3,
        include_baseline_start: bool = True,
        shuffle_order: bool = True,
        random_seed: Optional[int] = 42,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if n_starts < 1:
            raise ValueError(f"n_starts must be >= 1, got {n_starts}")
        if start_strategy not in VALID_START_STRATEGIES:
            raise ValueError(
                f"start_strategy must be one of {VALID_START_STRATEGIES}; got {start_strategy!r}"
            )
        if not (0.0 <= float(perturb_prob) <= 1.0):
            raise ValueError(f"perturb_prob must be in [0, 1]; got {perturb_prob}")

        self.n_starts = int(n_starts)
        self.start_strategy = start_strategy
        self.perturb_prob = float(perturb_prob)
        self.include_baseline_start = bool(include_baseline_start)
        self.shuffle_order = bool(shuffle_order)
        self.random_seed = random_seed
        self.optimizer_kwargs: Dict[str, Any] = dict(optimizer_kwargs or {})
        self.optimizer_attributes: Dict[str, Any] = dict(optimizer_attributes or {})

    def _generate_starts(
        self,
        baseline: Dict[str, Any],
        rng: random.Random,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return ``[(label, strategy, config)]`` of length ``n_starts``."""
        starts: List[Tuple[str, str, Dict[str, Any]]] = []

        if self.include_baseline_start:
            starts.append(("baseline", "baseline", deepcopy(baseline)))

        while len(starts) < self.n_starts:
            idx = len(starts)
            if self.start_strategy == "mixed":
                strategy = "perturb" if (idx % 2 == 1) else "random"
            else:
                strategy = self.start_strategy

            if strategy == "perturb":
                cfg = ParameterGrid.perturb_baseline(
                    rng=rng,
                    perturb_prob=self.perturb_prob,
                    base_config=baseline,
                )
            else:  # "random"
                cfg = ParameterGrid.sample_random_config(rng=rng)
            starts.append((f"{strategy}_start_{idx}", strategy, cfg))

        return starts

    def _build_inner_optimizer(self) -> CachedCoordinateDescentOptimizer:
        optimizer = CachedCoordinateDescentOptimizer(**self.optimizer_kwargs)
        for attr, value in self.optimizer_attributes.items():
            setattr(optimizer, attr, value)
        return optimizer

    def _extract_final_row(
        self,
        results_df: pd.DataFrame,
    ) -> Tuple[float, Dict[str, Any]]:
        """Return (best_objective, metrics_dict) for the trajectory's best row."""
        rank_col = "objective_score" if "objective_score" in results_df.columns else "f1"
        if rank_col not in results_df.columns or len(results_df) == 0:
            return float("nan"), {}

        best_idx = results_df[rank_col].idxmax()
        best_row = results_df.loc[best_idx]
        final_obj = float(best_row[rank_col])
        final_metrics = {
            k: _coerce_scalar(best_row[k])
            for k in _METRIC_COLUMNS
            if k in results_df.columns
        }
        return final_obj, final_metrics

    @staticmethod
    def _extract_baseline_objective(results_df: pd.DataFrame) -> float:
        if "detector" not in results_df.columns or "objective_score" not in results_df.columns:
            return float("nan")
        baseline_rows = results_df[results_df["detector"] == "baseline"]
        if baseline_rows.empty:
            return float("nan")
        return float(baseline_rows["objective_score"].iloc[0])

    def optimize(
        self,
        loader,
        market_ids: List[int],
        n_passes: int = 1,
        optimize_order: Optional[Sequence[str]] = None,
        initial_config: Optional[Dict[str, Any]] = None,
        val_market_ids: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
        """Run all starts and return the best trajectory."""
        if not market_ids:
            raise ValueError("market_ids is empty")

        rng = random.Random(self.random_seed)
        baseline = (
            deepcopy(initial_config)
            if initial_config is not None
            else ParameterGrid.get_baseline_config()
        )
        base_optimize_order = list(optimize_order) if optimize_order is not None else None

        val_market_ids = list(val_market_ids or [])
        use_val = bool(val_market_ids)
        # Detect accidental overlap between train and val rather than letting
        # it silently inflate validation scores.
        overlap = set(market_ids) & set(val_market_ids)
        if overlap:
            raise ValueError(
                f"train and validation market sets overlap on {len(overlap)} ids; "
                "splits must be disjoint"
            )

        starts = self._generate_starts(baseline, rng)

        logger.info("\n" + "=" * 80)
        logger.info("MULTI-START COORDINATE DESCENT")
        logger.info("=" * 80)
        logger.info(
            "n_starts=%d | strategy=%s | perturb_prob=%.2f | shuffle_order=%s | seed=%s",
            len(starts),
            self.start_strategy,
            self.perturb_prob,
            self.shuffle_order,
            self.random_seed,
        )
        logger.info(
            "include_baseline_start=%s | n_passes=%d | train_markets=%d | val_markets=%d",
            self.include_baseline_start,
            n_passes,
            len(market_ids),
            len(val_market_ids),
        )
        if use_val:
            logger.info(
                "Trajectory ranking will use VALIDATION objective "
                "(train objective retained for diagnostic only)."
            )
        else:
            logger.info(
                "No validation set supplied; trajectories will be ranked on "
                "TRAINING objective (legacy behaviour — be aware of selection bias)."
            )

        all_dfs: List[pd.DataFrame] = []
        per_start: List[StartSummary] = []
        winning_detector_summaries: Optional[Dict[str, Any]] = None
        best_config: Dict[str, Any] = deepcopy(baseline)
        best_objective = float("-inf")
        best_start_idx = -1

        overall_start = time.time()

        for start_idx, (label, strategy, start_config) in enumerate(starts):
            run_order = base_optimize_order
            if (
                self.shuffle_order
                and base_optimize_order
                and start_idx > 0
                and not (self.include_baseline_start and start_idx == 0)
            ):
                run_order = list(base_optimize_order)
                rng.shuffle(run_order)

            logger.info("\n" + "#" * 80)
            logger.info(
                "Multi-start run %d/%d  |  %s  (strategy=%s)",
                start_idx + 1,
                len(starts),
                label,
                strategy,
            )
            if run_order is not None:
                logger.info("  optimize_order: %s", run_order)
            logger.info("#" * 80)

            optimizer = self._build_inner_optimizer()
            trajectory_start = time.time()

            try:
                final_config, results_df, detector_summaries = optimizer.optimize(
                    loader=loader,
                    market_ids=market_ids,
                    n_passes=n_passes,
                    optimize_order=run_order,
                    initial_config=deepcopy(start_config),
                )
            except Exception:
                logger.exception(
                    "Multi-start trajectory %d (%s) failed; skipping", start_idx, label
                )
                continue

            elapsed = time.time() - trajectory_start

            results_df = results_df.copy()
            results_df["start_idx"] = start_idx
            results_df["start_label"] = label
            results_df["start_strategy"] = strategy
            all_dfs.append(results_df)

            baseline_obj = self._extract_baseline_objective(results_df)
            train_obj, train_metrics = self._extract_final_row(results_df)

            # Validation pass: score the final config ONCE on val markets
            val_obj = float("nan")
            val_metrics: Dict[str, Any] = {}
            val_elapsed = 0.0
            if use_val:
                val_start = time.time()
                try:
                    eval_result = optimizer._evaluate_config(
                        config=final_config,
                        loader=loader,
                        market_ids=val_market_ids,
                    )
                    raw_metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
                    val_metrics = {
                        k: _coerce_scalar(v)
                        for k, v in raw_metrics.items()
                        if k in _METRIC_COLUMNS
                    }
                    val_obj = optimizer._objective(raw_metrics)
                except Exception:
                    logger.exception(
                        "Validation pass failed for trajectory %d (%s); "
                        "falling back to training objective for this start",
                        start_idx,
                        label,
                    )
                val_elapsed = time.time() - val_start

            # The objective used for ranking: val when available, else train.
            ranking_obj = val_obj if (use_val and _is_number(val_obj)) else train_obj

            per_start.append(
                StartSummary(
                    start_idx=start_idx,
                    label=label,
                    strategy=strategy,
                    baseline_objective=baseline_obj,
                    train_final_objective=train_obj,
                    final_objective=ranking_obj,
                    elapsed_seconds=elapsed,
                    configs_evaluated=len(results_df),
                    optimize_order=list(run_order) if run_order else [],
                    train_final_metrics=train_metrics,
                    val_final_objective=val_obj,
                    val_final_metrics=val_metrics,
                    val_elapsed_seconds=val_elapsed,
                    initial_config=start_config,
                    final_config=final_config,
                )
            )

            if use_val:
                logger.info(
                    "  -> %s: train_obj=%.4f (base=%.4f, delta=%+.4f)  |  "
                    "val_obj=%.4f  |  gap=%+.4f  (%.1fs train + %.1fs val)",
                    label,
                    train_obj,
                    baseline_obj,
                    train_obj - baseline_obj if _is_number(baseline_obj) else float("nan"),
                    val_obj,
                    train_obj - val_obj if _is_number(val_obj) else float("nan"),
                    elapsed,
                    val_elapsed,
                )
            else:
                logger.info(
                    "  -> %s: baseline_obj=%.4f -> final_obj=%.4f  (delta=%+.4f, %.1fs)",
                    label,
                    baseline_obj,
                    train_obj,
                    train_obj - baseline_obj if _is_number(baseline_obj) else float("nan"),
                    elapsed,
                )

            if _is_number(ranking_obj) and ranking_obj > best_objective:
                best_objective = ranking_obj
                best_config = final_config
                best_start_idx = start_idx
                winning_detector_summaries = detector_summaries

        if not per_start:
            raise RuntimeError("All multi-start trajectories failed; see logs for details.")

        total_elapsed = time.time() - overall_start
        combined_df = (
            pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        )

        summary = self._build_summary(
            per_start=per_start,
            best_start_idx=best_start_idx,
            winning_detector_summaries=winning_detector_summaries,
            total_elapsed=total_elapsed,
            used_validation=use_val,
            n_train_markets=len(market_ids),
            n_val_markets=len(val_market_ids),
        )
        self._log_summary(summary, per_start)

        return best_config, combined_df, summary

    def _build_summary(
        self,
        per_start: List[StartSummary],
        best_start_idx: int,
        winning_detector_summaries: Optional[Dict[str, Any]],
        total_elapsed: float,
        used_validation: bool,
        n_train_markets: int,
        n_val_markets: int,
    ) -> Dict[str, Any]:
        train_objs = [
            s.train_final_objective for s in per_start if _is_number(s.train_final_objective)
        ]
        val_objs = [s.val_final_objective for s in per_start if _is_number(s.val_final_objective)]
        rank_objs = [s.final_objective for s in per_start if _is_number(s.final_objective)]

        gaps = [
            s.train_final_objective - s.val_final_objective
            for s in per_start
            if _is_number(s.train_final_objective) and _is_number(s.val_final_objective)
        ]

        def _aggregate(values: List[float]) -> Dict[str, float]:
            if not values:
                return {
                    "min": float("nan"),
                    "median": float("nan"),
                    "max": float("nan"),
                    "spread": float("nan"),
                }
            return {
                "min": min(values),
                "median": float(pd.Series(values).median()),
                "max": max(values),
                "spread": max(values) - min(values),
            }

        return {
            "n_starts_requested": self.n_starts,
            "n_starts_completed": len(per_start),
            "start_strategy": self.start_strategy,
            "perturb_prob": self.perturb_prob,
            "shuffle_order": self.shuffle_order,
            "include_baseline_start": self.include_baseline_start,
            "random_seed": self.random_seed,
            "used_validation": used_validation,
            "n_train_markets": n_train_markets,
            "n_val_markets": n_val_markets,
            "best_start_idx": best_start_idx,
            "best_final_objective": max(rank_objs) if rank_objs else float("nan"),
            "ranking_objective_aggregates": _aggregate(rank_objs),
            "train_objective_aggregates": _aggregate(train_objs),
            "val_objective_aggregates": _aggregate(val_objs),
            "train_val_gap_mean": float(pd.Series(gaps).mean()) if gaps else float("nan"),
            "train_val_gap_max": max(gaps) if gaps else float("nan"),
            "median_final_objective": (
                float(pd.Series(rank_objs).median()) if rank_objs else float("nan")
            ),
            "min_final_objective": min(rank_objs) if rank_objs else float("nan"),
            "max_final_objective": max(rank_objs) if rank_objs else float("nan"),
            "final_objective_spread": (
                (max(rank_objs) - min(rank_objs)) if rank_objs else float("nan")
            ),
            "total_elapsed_seconds": total_elapsed,
            "per_start": [s.to_dict() for s in per_start],
            "winning_detector_summaries": winning_detector_summaries,
        }

    @staticmethod
    def _log_summary(
        summary: Dict[str, Any],
        per_start: List[StartSummary],
    ) -> None:
        used_val = summary.get("used_validation", False)

        logger.info("\n" + "=" * 80)
        logger.info("MULTI-START SUMMARY")
        logger.info("=" * 80)
        logger.info(
            "Completed %d/%d starts in %.1fs  |  train_markets=%d  val_markets=%d",
            summary["n_starts_completed"],
            summary["n_starts_requested"],
            summary["total_elapsed_seconds"],
            summary.get("n_train_markets", 0),
            summary.get("n_val_markets", 0),
        )

        train_agg = summary["train_objective_aggregates"]
        val_agg = summary["val_objective_aggregates"]

        if used_val:
            logger.info(
                "Train objective across starts:  best=%.4f | median=%.4f | min=%.4f | spread=%.4f",
                train_agg["max"], train_agg["median"], train_agg["min"], train_agg["spread"],
            )
            logger.info(
                "Val   objective across starts:  best=%.4f | median=%.4f | min=%.4f | spread=%.4f",
                val_agg["max"], val_agg["median"], val_agg["min"], val_agg["spread"],
            )
            logger.info(
                "Train-val gap: mean=%+.4f | max=%+.4f  (positive = trajectories overfit on train)",
                summary["train_val_gap_mean"],
                summary["train_val_gap_max"],
            )
            logger.info(
                "Winner selected on VALIDATION objective. Winning start idx: %d",
                summary["best_start_idx"],
            )
        else:
            logger.info(
                "Objectives: best=%.4f | median=%.4f | min=%.4f | spread=%.4f",
                summary["max_final_objective"],
                summary["median_final_objective"],
                summary["min_final_objective"],
                summary["final_objective_spread"],
            )
            logger.info("Winning start idx: %d", summary["best_start_idx"])

        logger.info("Per-start ranking (best first):")
        for s in sorted(per_start, key=lambda x: -x.final_objective):
            if used_val and _is_number(s.val_final_objective):
                gap = s.train_final_objective - s.val_final_objective
                logger.info(
                    "  [%2d] %-26s %-8s  base=%.4f  train=%.4f  val=%.4f  gap=%+.4f  (%.1fs)",
                    s.start_idx,
                    s.label,
                    s.strategy,
                    s.baseline_objective,
                    s.train_final_objective,
                    s.val_final_objective,
                    gap,
                    s.elapsed_seconds + s.val_elapsed_seconds,
                )
            else:
                delta = (
                    s.train_final_objective - s.baseline_objective
                    if _is_number(s.baseline_objective)
                    else float("nan")
                )
                logger.info(
                    "  [%2d] %-26s %-8s  base=%.4f -> final=%.4f  (delta=%+.4f, %.1fs)",
                    s.start_idx,
                    s.label,
                    s.strategy,
                    s.baseline_objective,
                    s.train_final_objective,
                    delta,
                    s.elapsed_seconds,
                )

def _is_number(value: Any) -> bool:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return False
    return fval == fval and fval not in (float("inf"), float("-inf"))


def _coerce_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python for easy JSON/dict use."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value
