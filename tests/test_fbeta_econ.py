"""Tests for F-beta econ composite objectives."""

from __future__ import annotations

from backtesting.fbeta_econ import (
    _fbeta_harmonic,
    _normalize_return_sigmoid,
    compute_fbeta_econ_metrics,
    finalize_fbeta_econ_metrics,
)
from backtesting.parameter_optimizer import CoordinateDescentOptimizer


def test_f_beta_econ_winrate_return_penalizes_high_winrate_negative_return() -> None:
    high_wr_negative_return = compute_fbeta_econ_metrics(
        recall=0.5,
        f1=0.5,
        f0_5=0.5,
        flagged_win_rate=0.95,
        mean_return_flagged=0.0,
        trade_flagged_mean_return=-0.05,
        trade_t_stat=1.0,
    )
    moderate_both = compute_fbeta_econ_metrics(
        recall=0.5,
        f1=0.5,
        f0_5=0.5,
        flagged_win_rate=0.65,
        mean_return_flagged=0.08,
        trade_flagged_mean_return=0.08,
        trade_t_stat=1.0,
    )

    assert high_wr_negative_return["f_beta_econ_winrate_return"] == 0.0
    assert moderate_both["f_beta_econ_winrate_return"] > 0.0


def test_f_beta_econ_winrate_return_prefers_balanced_legs_over_winrate_only() -> None:
    winrate_only = compute_fbeta_econ_metrics(
        recall=0.5,
        f1=0.5,
        f0_5=0.5,
        flagged_win_rate=1.0,
        mean_return_flagged=0.01,
        trade_flagged_mean_return=0.01,
        trade_t_stat=1.0,
    )
    balanced = compute_fbeta_econ_metrics(
        recall=0.5,
        f1=0.5,
        f0_5=0.5,
        flagged_win_rate=0.70,
        mean_return_flagged=0.12,
        trade_flagged_mean_return=0.12,
        trade_t_stat=1.0,
    )

    assert balanced["f_beta_econ_winrate_return"] > winrate_only["f_beta_econ_winrate_return"]


def test_finalize_fbeta_econ_metrics_uses_trade_flagged_mean_return() -> None:
    metrics = {
        "recall": 0.4,
        "f1": 0.5,
        "f0_5": 0.45,
        "trade_flagged_win_rate": 0.8,
        "mean_return_flagged": 0.05,
        "trade_flagged_mean_return": 0.10,
        "trade_t_stat": 2.0,
    }
    finalize_fbeta_econ_metrics(metrics)

    expected_return_norm = _normalize_return_sigmoid(0.10)
    expected = _fbeta_harmonic(0.8, expected_return_norm, beta=1.0)

    assert metrics["econ_signal_trade_mean_return_sigmoid_norm"] == expected_return_norm
    assert metrics["f_beta_econ_winrate_return"] == expected


def test_coordinate_descent_accepts_f_beta_econ_winrate_return_objective() -> None:
    optimizer = CoordinateDescentOptimizer(objective_metric="f_beta_econ_winrate_return")
    assert optimizer.objective_metric == "f_beta_econ_winrate_return"
