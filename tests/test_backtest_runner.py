import copy

import numpy as np
import pytest

from backtesting.backtest_runner import BacktestRunner
from models import DetectionSignal


def test_config_normalization_accepts_nested_and_raw_configs(tiny_config):
    nested = BacktestRunner._normalize_config(tiny_config)
    raw = BacktestRunner._normalize_config(
        {
            "alert_threshold": 0.7,
            "probability_impact": {"min_delta_prob": 0.1},
            "volume_anomaly": {"z_score_threshold": 4.0},
        }
    )

    assert nested["alert_threshold"] == 0.5
    assert nested["detectors"]["probability_impact"]["max_confidence"] == 0.6
    assert raw["alert_threshold"] == 0.7
    assert raw["detectors"]["probability_impact"]["min_delta_prob"] == 0.1


def test_noisy_or_score_combines_detector_confidences():
    score = BacktestRunner._calculate_total_score(
        [
            DetectionSignal("A", 0.2, "a"),
            DetectionSignal("B", 0.5, "b"),
        ]
    )

    assert score == pytest.approx(0.6)
    assert BacktestRunner._calculate_total_score([]) == 0.0


def test_backtest_runner_generates_alerts_and_tracks_wallet_state(make_trade, tiny_config):
    trades = [
        make_trade(wallet="0xa", price=0.40, timestamp_ms=0, notional_usdc=100),
        make_trade(wallet="0xa", price=0.50, timestamp_ms=60_000, notional_usdc=200),
    ]

    result = BacktestRunner(tiny_config, include_recidivism=False).run_backtest(
        trades,
        {"id": 1, "market_slug": "market-1"},
        capture_alerts=True,
        capture_trade_features=True,
        progress_every=0,
    )

    assert result.total_trades == 2
    assert result.alerts_generated == 1
    assert result.detector_stats["ProbabilityImpactDetector"] == 1
    assert result.wallet_trade_counts["0xa"] == 2
    assert result.wallet_positions["0xa"][0] == pytest.approx(650.0)
    assert result.wallet_costs["0xa"][0] == pytest.approx(300.0)
    assert result.wallet_gross_buy_notional["0xa"] == pytest.approx(300.0)
    assert result.alerts[0].signals[0].detector_name == "ProbabilityImpactDetector"
    assert result.all_trade_features[1]["is_alert"] == 1


def test_score_multipliers_are_applied_with_cap(make_trade, tiny_config):
    config = copy.deepcopy(tiny_config)
    config["detectors"]["probability_impact"]["max_confidence"] = 0.4
    trades = [
        make_trade(wallet="0xa", price=0.40, timestamp_ms=0),
        make_trade(wallet="0xa", price=0.50, timestamp_ms=60_000),
    ]

    result = BacktestRunner(config, include_recidivism=False).run_backtest(
        trades,
        {"id": 1, "market_slug": "market-1"},
        capture_alerts=True,
        progress_every=0,
        score_multipliers=np.array([1.0, 2.0], dtype=np.float32),
        score_cap=0.7,
    )

    assert result.alerts_generated == 1
    assert result.wallet_suspicion["0xa"] == pytest.approx(0.7)
    assert result.alerts[0].total_score == pytest.approx(0.7)


def test_recidivism_uses_prior_alerts_causally(make_trade, tiny_config):
    trades = [
        make_trade(wallet="0xa", price=0.40, timestamp_ms=0),
        make_trade(wallet="0xa", price=0.50, timestamp_ms=60_000),
        make_trade(wallet="0xa", price=0.50, timestamp_ms=120_000),
    ]

    result = BacktestRunner(tiny_config, include_recidivism=True).run_backtest(
        trades,
        {"id": 1, "market_slug": "market-1"},
        capture_alerts=False,
        capture_trade_features=True,
        progress_every=0,
    )

    assert result.all_trade_features[1]["RecidivismDetector_confidence"] == 0.0
    assert result.all_trade_features[2]["RecidivismDetector_confidence"] > 0.0


def test_runner_validates_multiplier_length(make_trade, tiny_config):
    with pytest.raises(ValueError, match="score_multipliers length mismatch"):
        BacktestRunner(tiny_config, include_recidivism=False).run_backtest(
            [make_trade()],
            {"id": 1},
            progress_every=0,
            score_multipliers=np.ones(2),
        )
