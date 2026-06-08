"""
F-beta over (EconSignal, ClassificationLeg) — an objective metric that forces
BOTH the economic informativeness of flagged trades AND a wallet-level
classification score to be high.
"""

from __future__ import annotations

import math
from typing import Dict


DEFAULT_BETA: float = 1.0
DEFAULT_T_STAT_SCALE: float = 2.0
DEFAULT_RETURN_SCALE: float = 1.0


def _fbeta_harmonic(econ_signal: float, second_leg: float, beta: float) -> float:
    """Harmonic F_beta between EconSignal in [0,1] and a second score in [0,1]."""
    e = float(econ_signal)
    s = float(second_leg)
    b = float(beta)

    if e <= 0.0 or s <= 0.0:
        return 0.0

    b2 = b * b
    denom = b2 * e + s
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * (e * s) / denom


def _normalize_trade_t_stat(
    trade_t_stat: float,
    t_stat_scale: float = DEFAULT_T_STAT_SCALE,
) -> float:
    """
    Map trade t-stat to [0, 1] as an economic-signal input.
    """
    scale = float(t_stat_scale)
    if scale <= 1e-12:
        scale = DEFAULT_T_STAT_SCALE
    z = float(trade_t_stat) / scale
    return max(0.0, min(1.0, math.tanh(z / 2.0)))


def _normalize_return_sigmoid(
    mean_return_flagged: float,
    return_scale: float = DEFAULT_RETURN_SCALE,
) -> float:
    """
    Map mean return to [0, 1] with a one-sided sigmoid (not hard clipping).
    """
    scale = float(return_scale)
    if scale <= 1e-12:
        scale = DEFAULT_RETURN_SCALE
    z = float(mean_return_flagged) / scale
    return max(0.0, min(1.0, math.tanh(z / 2.0)))


def compute_trade_flagged_win_rate(flagged_returns) -> float:
    """
    Fraction of flagged BUY trades with strictly positive return.
    """
    try:
        n = len(flagged_returns)
    except TypeError:
        flagged_returns = list(flagged_returns)
        n = len(flagged_returns)

    if n == 0:
        return 0.0

    wins = 0
    for r in flagged_returns:
        if float(r) > 0.0:
            wins += 1
    return wins / n


def compute_fbeta_econ_metrics(
    *,
    recall: float,
    f1: float,
    f0_5: float,
    flagged_win_rate: float,
    mean_return_flagged: float,
    trade_flagged_mean_return: float,
    trade_t_stat: float,
    beta: float = DEFAULT_BETA,
    t_stat_scale: float = DEFAULT_T_STAT_SCALE,
    return_scale: float = DEFAULT_RETURN_SCALE,
) -> Dict[str, float]:
    """
    Return F-beta econ objectives (Recall, F1, and F0.5 legs), normalised
    EconSignal inputs (win rate, mean return variants, and trade t-stat), and
    ``f_beta_econ_beta``, as a flat dict for merging.
    """
    mean_return_norm = max(0.0, min(1.0, float(mean_return_flagged)))
    mean_return_sigmoid_norm = _normalize_return_sigmoid(
        mean_return_flagged,
        return_scale=return_scale,
    )
    trade_mean_return_sigmoid_norm = _normalize_return_sigmoid(
        trade_flagged_mean_return,
        return_scale=return_scale,
    )
    win_rate = max(0.0, min(1.0, float(flagged_win_rate)))
    t_stat_norm = _normalize_trade_t_stat(trade_t_stat, t_stat_scale=t_stat_scale)
    r = float(recall)
    f1v = max(0.0, min(1.0, float(f1)))
    f05v = max(0.0, min(1.0, float(f0_5)))

    return {
        "econ_signal_winrate": win_rate,
        "econ_signal_mean_return_norm": mean_return_norm,
        "econ_signal_return_sigmoid_norm": mean_return_sigmoid_norm,
        "econ_signal_trade_mean_return_sigmoid_norm": trade_mean_return_sigmoid_norm,
        "econ_signal_trade_t_stat_norm": t_stat_norm,
        "f_beta_econ_winrate": _fbeta_harmonic(win_rate, r, beta),
        "f_beta_econ_mean_return": _fbeta_harmonic(mean_return_norm, r, beta),
        "f_beta_econ_winrate_f1": _fbeta_harmonic(win_rate, f1v, beta),
        "f_beta_econ_mean_return_f1": _fbeta_harmonic(mean_return_norm, f1v, beta),
        "f_beta_econ_return_f1": _fbeta_harmonic(mean_return_sigmoid_norm, f1v, beta),
        "f_beta_econ_t_stat_f1": _fbeta_harmonic(t_stat_norm, f1v, beta),
        "f_beta_econ_winrate_f0_5": _fbeta_harmonic(win_rate, f05v, beta),
        "f_beta_econ_mean_return_f0_5": _fbeta_harmonic(mean_return_norm, f05v, beta),
        "f_beta_econ_winrate_return": _fbeta_harmonic(
            win_rate,
            trade_mean_return_sigmoid_norm,
            beta,
        ),
        "f_beta_econ_beta": float(beta),
        "f_beta_econ_t_stat_scale": float(t_stat_scale),
        "f_beta_econ_return_scale": float(return_scale),
    }


def finalize_fbeta_econ_metrics(metrics: Dict, beta: float = DEFAULT_BETA) -> None:
    """
    In-place: read recall, f1, f0_5, trade_flagged_win_rate,
    mean_return_flagged, trade_flagged_mean_return, and trade_t_stat from
    ``metrics`` and write all F-beta econ variants plus normalised EconSignal
    inputs into the same dict
    """
    fbeta = compute_fbeta_econ_metrics(
        recall=float(metrics.get("recall", 0.0) or 0.0),
        f1=float(metrics.get("f1", 0.0) or 0.0),
        f0_5=float(metrics.get("f0_5", 0.0) or 0.0),
        flagged_win_rate=float(metrics.get("trade_flagged_win_rate", 0.0) or 0.0),
        mean_return_flagged=float(metrics.get("mean_return_flagged", 0.0) or 0.0),
        trade_flagged_mean_return=float(
            metrics.get("trade_flagged_mean_return", 0.0) or 0.0
        ),
        trade_t_stat=float(metrics.get("trade_t_stat", 0.0) or 0.0),
        beta=beta,
    )
    metrics.update(fbeta)
