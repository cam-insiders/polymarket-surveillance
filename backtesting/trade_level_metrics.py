"""
Pooled trade-level metrics for flagged vs unflagged BUY resolution returns.

Used by coordinate-descent / clustering optimizers and cached evaluators.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# Clip each group's lower/upper tail before winsorized metrics (per group).
DEFAULT_WINSOR_TAIL_QUANTILE = 0.05

EXTENDED_TRADE_OBJECTIVES = (
    "trade_flagged_mean_return_lcb",
    "trade_flagged_weighted_return_lcb",
    "trade_cohens_d_lcb",
    "trade_weighted_mean_return_diff",
    "trade_weighted_cohens_d",
    "trade_weighted_cohens_d_lcb",
    "trade_winsorized_mean_return_diff",
    "trade_winsorized_cohens_d",
    "trade_winsorized_weighted_mean_return_diff",
    "trade_winsorized_weighted_cohens_d",
    "trade_winsorized_weighted_cohens_d_lcb",
)


def trade_level_zero_defaults() -> Dict[str, float]:
    """Default zeros for trade-level keys (including extended objectives)."""
    return {
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


def winsorize_group(
    values: np.ndarray,
    tail: float = DEFAULT_WINSOR_TAIL_QUANTILE,
) -> np.ndarray:
    """Clip values to [q_tail, q_{1-tail}] within a single group."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return arr
    lo = float(np.quantile(arr, tail))
    hi = float(np.quantile(arr, 1.0 - tail))
    if lo > hi:
        return arr
    return np.clip(arr, lo, hi)


def _unweighted_cohens_d(flagged: np.ndarray, unflagged: np.ndarray) -> float:
    n_f, n_u = len(flagged), len(unflagged)
    if n_f < 2 or n_u < 2:
        return 0.0
    f_mean = float(np.mean(flagged))
    u_mean = float(np.mean(unflagged))
    diff = f_mean - u_mean
    f_var = float(np.var(flagged, ddof=1))
    u_var = float(np.var(unflagged, ddof=1))
    pooled_var = ((n_f - 1) * f_var + (n_u - 1) * u_var) / (n_f + n_u - 2)
    pooled_std = float(np.sqrt(pooled_var))
    if pooled_std < 1e-12:
        return 0.0
    return diff / pooled_std


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum < 1e-12 or x.size == 0:
        return float(np.mean(x)) if x.size > 0 else 0.0
    return float(np.average(x, weights=w))


def _weighted_var(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    w = np.asarray(weights, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum < 1e-12 or x.size < 2:
        return 0.0
    return float(np.average((x - mean) ** 2, weights=w))


def _mean_standard_error(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def _weighted_cohens_d(
    flagged: np.ndarray,
    unflagged: np.ndarray,
    flagged_weights: np.ndarray,
    unflagged_weights: np.ndarray,
) -> float:
    n_f, n_u = len(flagged), len(unflagged)
    if n_f < 2 or n_u < 2:
        return 0.0
    f_mean = _weighted_mean(flagged, flagged_weights)
    u_mean = _weighted_mean(unflagged, unflagged_weights)
    diff = f_mean - u_mean
    f_var = _weighted_var(flagged, flagged_weights, f_mean)
    u_var = _weighted_var(unflagged, unflagged_weights, u_mean)
    pooled_var = ((n_f - 1) * f_var + (n_u - 1) * u_var) / (n_f + n_u - 2)
    pooled_std = float(np.sqrt(pooled_var))
    if pooled_std < 1e-12:
        return 0.0
    return diff / pooled_std


def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w ** 2))
    if denom <= 1e-12:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def _cohens_d_standard_error(effect_size: float, n_f: float, n_u: float) -> float:
    """
    Approximate SE for a two-sample standardized mean difference.

    Weighted objectives pass effective sample sizes in place of raw counts.
    """
    if n_f <= 1.0 or n_u <= 1.0 or (n_f + n_u) <= 2.0:
        return 0.0
    first = (n_f + n_u) / (n_f * n_u)
    second = (effect_size ** 2) / (2.0 * (n_f + n_u - 2.0))
    return float(np.sqrt(max(0.0, first + second)))


def compute_pooled_trade_metrics(
    pooled_flagged: np.ndarray,
    pooled_unflagged: np.ndarray,
    pooled_flagged_notionals: Optional[np.ndarray] = None,
    pooled_unflagged_notionals: Optional[np.ndarray] = None,
    winsor_tail: float = DEFAULT_WINSOR_TAIL_QUANTILE,
) -> Dict[str, float]:
    """
    Compute standard and extended trade-level metrics on pooled BUY returns.
    """
    pooled_f = np.asarray(pooled_flagged, dtype=np.float64)
    pooled_u = np.asarray(pooled_unflagged, dtype=np.float64)
    n_f, n_u = len(pooled_f), len(pooled_u)

    f_mean = float(np.mean(pooled_f)) if n_f > 0 else 0.0
    u_mean = float(np.mean(pooled_u)) if n_u > 0 else 0.0
    diff = f_mean - u_mean

    if n_f >= 2 and n_u >= 2:
        f_var = float(np.var(pooled_f, ddof=1))
        u_var = float(np.var(pooled_u, ddof=1))
        se = np.sqrt(f_var / n_f + u_var / n_u) if (f_var + u_var) > 1e-15 else 0.0
        t_stat = diff / se if se > 1e-12 else 0.0
        cohens_d = _unweighted_cohens_d(pooled_f, pooled_u)
        cohens_d_se = _cohens_d_standard_error(cohens_d, float(n_f), float(n_u))
        cohens_d_lcb = cohens_d - (1.64 * cohens_d_se)
    else:
        t_stat = 0.0
        cohens_d = 0.0
        cohens_d_se = 0.0
        cohens_d_lcb = 0.0

    wins_f = winsorize_group(pooled_f, tail=winsor_tail)
    wins_u = winsorize_group(pooled_u, tail=winsor_tail)
    wins_diff = float(np.mean(wins_f) - np.mean(wins_u)) if (len(wins_f) and len(wins_u)) else 0.0
    wins_d = _unweighted_cohens_d(wins_f, wins_u)

    weighted_diff = 0.0
    weighted_d = 0.0
    weighted_d_se = 0.0
    weighted_d_lcb = 0.0
    wins_weighted_diff = 0.0
    wins_weighted_d = 0.0
    wins_weighted_d_se = 0.0
    wins_weighted_d_lcb = 0.0
    n_f_eff = 0.0
    n_u_eff = 0.0
    flagged_weighted_return = 0.0
    flagged_weighted_return_se = 0.0
    flagged_weighted_return_lcb = 0.0
    flagged_mean_return_se = _mean_standard_error(pooled_f)
    flagged_mean_return_lcb = f_mean - (1.64 * flagged_mean_return_se)
    if (
        pooled_flagged_notionals is not None
        and pooled_unflagged_notionals is not None
        and len(pooled_flagged_notionals) == n_f
        and len(pooled_unflagged_notionals) == n_u
        and n_f > 0
        and n_u > 0
    ):
        wf = np.asarray(pooled_flagged_notionals, dtype=np.float64)
        wu = np.asarray(pooled_unflagged_notionals, dtype=np.float64)
        flagged_weighted_return = _weighted_mean(pooled_f, wf)
        weighted_diff = _weighted_mean(pooled_f, wf) - _weighted_mean(pooled_u, wu)
        weighted_d = _weighted_cohens_d(pooled_f, pooled_u, wf, wu)
        n_f_eff = _effective_sample_size(wf)
        n_u_eff = _effective_sample_size(wu)
        flagged_weighted_var = _weighted_var(pooled_f, wf, flagged_weighted_return)
        flagged_weighted_return_se = (
            float(np.sqrt(flagged_weighted_var / n_f_eff)) if n_f_eff > 1.0 else 0.0
        )
        flagged_weighted_return_lcb = flagged_weighted_return - (
            1.64 * flagged_weighted_return_se
        )
        weighted_d_se = _cohens_d_standard_error(weighted_d, n_f_eff, n_u_eff)
        weighted_d_lcb = weighted_d - (1.64 * weighted_d_se)
        wins_weighted_diff = _weighted_mean(wins_f, wf) - _weighted_mean(wins_u, wu)
        wins_weighted_d = _weighted_cohens_d(wins_f, wins_u, wf, wu)
        wins_weighted_d_se = _cohens_d_standard_error(wins_weighted_d, n_f_eff, n_u_eff)
        wins_weighted_d_lcb = wins_weighted_d - (1.64 * wins_weighted_d_se)

    return {
        "trade_flagged_count": int(n_f),
        "trade_unflagged_count": int(n_u),
        "trade_flagged_mean_return": f_mean,
        "trade_flagged_mean_return_lcb": float(flagged_mean_return_lcb),
        "trade_flagged_mean_return_se": float(flagged_mean_return_se),
        "trade_flagged_weighted_return": float(flagged_weighted_return),
        "trade_flagged_weighted_return_lcb": float(flagged_weighted_return_lcb),
        "trade_flagged_weighted_return_se": float(flagged_weighted_return_se),
        "trade_unflagged_mean_return": u_mean,
        "trade_mean_return_diff": diff,
        "trade_t_stat": float(t_stat),
        "trade_cohens_d": float(cohens_d),
        "trade_cohens_d_lcb": float(cohens_d_lcb),
        "trade_cohens_d_se": float(cohens_d_se),
        "trade_flagged_win_rate": float(np.mean(pooled_f > 0.0)) if n_f > 0 else 0.0,
        "trade_weighted_mean_return_diff": float(weighted_diff),
        "trade_weighted_cohens_d": float(weighted_d),
        "trade_weighted_cohens_d_lcb": float(weighted_d_lcb),
        "trade_weighted_cohens_d_se": float(weighted_d_se),
        "trade_winsorized_mean_return_diff": wins_diff,
        "trade_winsorized_cohens_d": float(wins_d),
        "trade_winsorized_weighted_mean_return_diff": float(wins_weighted_diff),
        "trade_winsorized_weighted_cohens_d": float(wins_weighted_d),
        "trade_winsorized_weighted_cohens_d_lcb": float(wins_weighted_d_lcb),
        "trade_winsorized_weighted_cohens_d_se": float(wins_weighted_d_se),
        "trade_weighted_flagged_effective_count": float(n_f_eff),
        "trade_weighted_unflagged_effective_count": float(n_u_eff),
    }


def merge_trade_level_metrics(
    metrics: Dict,
    flagged_returns_chunks: List[np.ndarray],
    unflagged_returns_chunks: List[np.ndarray],
    flagged_notionals_chunks: Optional[List[np.ndarray]] = None,
    unflagged_notionals_chunks: Optional[List[np.ndarray]] = None,
    winsor_tail: float = DEFAULT_WINSOR_TAIL_QUANTILE,
) -> None:
    """Merge pooled flagged/unflagged trade stats into ``metrics`` in place."""
    if not flagged_returns_chunks or not unflagged_returns_chunks:
        metrics.update(trade_level_zero_defaults())
        return

    pooled_f = np.concatenate(flagged_returns_chunks)
    pooled_u = np.concatenate(unflagged_returns_chunks)

    pooled_f_not: Optional[np.ndarray] = None
    pooled_u_not: Optional[np.ndarray] = None
    if flagged_notionals_chunks and unflagged_notionals_chunks:
        if len(flagged_notionals_chunks) == len(flagged_returns_chunks) and len(
            unflagged_notionals_chunks
        ) == len(unflagged_returns_chunks):
            pooled_f_not = np.concatenate(flagged_notionals_chunks)
            pooled_u_not = np.concatenate(unflagged_notionals_chunks)

    metrics.update(
        compute_pooled_trade_metrics(
            pooled_f,
            pooled_u,
            pooled_flagged_notionals=pooled_f_not,
            pooled_unflagged_notionals=pooled_u_not,
            winsor_tail=winsor_tail,
        )
    )
