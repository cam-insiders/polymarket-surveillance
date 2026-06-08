import pytest

from detectors.base import DetectionContext
from detectors.trade_detectors import (
    AccumulationDetector,
    ContraOutcomeSilenceDetector,
    ExtremePositionDetector,
    ProbabilityImpactDetector,
    VolumeAnomalyDetector,
)


def test_volume_anomaly_detector_uses_existing_rolling_baseline(make_trade):
    context = DetectionContext()
    detector = VolumeAnomalyDetector(
        {
            "lookback_window_hours": 24,
            "min_trades_for_baseline": 5,
            "z_score_threshold": 3.0,
            "min_absolute_notional": 0.0,
            "max_confidence": 0.8,
        }
    )
    for idx, notional in enumerate([100, 120, 80, 100, 110]):
        context.add_trade(make_trade(wallet=f"0xbase{idx}", notional_usdc=notional, timestamp_ms=idx))

    signal = detector.analyze(make_trade(wallet="0xwhale", notional_usdc=500, timestamp_ms=10), context)

    assert signal is not None
    assert signal.detector_name == "VolumeAnomalyDetector"
    assert signal.metadata["baseline_trades"] == 5
    assert signal.metadata["z_score"] > 3.0


def test_volume_anomaly_detector_suppresses_likely_flippers(make_trade):
    context = DetectionContext()
    detector = VolumeAnomalyDetector(
        {
            "min_trades_for_baseline": 3,
            "z_score_threshold": 1.0,
            "min_absolute_notional": 0.0,
            "flipper_filter": {"max_directional_ratio": 0.5, "min_volume_for_pattern": 1000.0},
        }
    )
    for idx, notional in enumerate([100, 120, 80]):
        context.add_trade(make_trade(wallet=f"0xbase{idx}", notional_usdc=notional, timestamp_ms=idx))
    context.add_trade(make_trade(wallet="0xflip", side="BUY", notional_usdc=10_000, price=0.5, timestamp_ms=10))
    context.add_trade(make_trade(wallet="0xflip", side="SELL", notional_usdc=9_900, price=0.5, timestamp_ms=11))

    assert detector.analyze(make_trade(wallet="0xflip", notional_usdc=50_000, timestamp_ms=12), context) is None


def test_probability_impact_detector_scores_probability_and_log_odds_moves(make_trade):
    context = DetectionContext()
    context.add_trade(make_trade(price=0.40, timestamp_ms=0))
    detector = ProbabilityImpactDetector(
        {
            "min_delta_prob": 0.05,
            "min_delta_log_odds": 0.2,
            "min_notional": 0.0,
            "max_confidence": 1.0,
        }
    )

    signal = detector.analyze(make_trade(price=0.55, timestamp_ms=1000), context)

    assert signal is not None
    assert signal.metadata["delta_prob"] == pytest.approx(0.15)
    assert signal.metadata["delta_log_odds"] > 0.2
    assert 0.0 < signal.confidence_score <= 1.0


def test_accumulation_detector_requires_focus_directionality_and_size(make_trade):
    context = DetectionContext()
    detector = AccumulationDetector(
        {
            "min_accumulation_usdc": 5_000.0,
            "min_directional_ratio": 0.8,
            "min_outcome_concentration": 0.9,
            "max_confidence": 0.7,
        }
    )
    context.add_trade(make_trade(wallet="0xacc", side="BUY", outcome_index=0, notional_usdc=6_000, price=0.5))

    signal = detector.analyze(make_trade(wallet="0xacc", side="BUY", outcome_index=0, notional_usdc=100, price=0.5), context)

    assert signal is not None
    assert signal.metadata["outcome_exposure_usdc"] == pytest.approx(6_000)
    assert signal.metadata["outcome_concentration"] == pytest.approx(1.0)

    hedged_context = DetectionContext()
    hedged_context.add_trade(make_trade(wallet="0xhedge", side="BUY", outcome_index=0, notional_usdc=6_000, price=0.5))
    hedged_context.add_trade(make_trade(wallet="0xhedge", side="BUY", outcome_index=1, notional_usdc=6_000, price=0.5))
    assert detector.analyze(make_trade(wallet="0xhedge", side="BUY", outcome_index=0), hedged_context) is None


def test_extreme_position_detector_flags_directional_tail_trades(make_trade):
    detector = ExtremePositionDetector({"tail_threshold": 0.2, "min_notional": 100.0, "max_confidence": 1.0})

    low_tail = detector.analyze(make_trade(side="BUY", price=0.10, notional_usdc=1_000), DetectionContext())
    high_tail = detector.analyze(make_trade(side="SELL", price=0.90, notional_usdc=1_000), DetectionContext())
    ordinary = detector.analyze(make_trade(side="BUY", price=0.50, notional_usdc=1_000), DetectionContext())

    assert low_tail is not None
    assert low_tail.metadata["tail_type"] == "low"
    assert high_tail is not None
    assert high_tail.metadata["tail_type"] == "high"
    assert ordinary is None


def test_contra_outcome_silence_detector_uses_contra_gap_history(make_trade):
    context = DetectionContext()
    detector = ContraOutcomeSilenceDetector(
        {
            "min_gap_samples": 4,
            "silence_threshold": 3.0,
            "min_notional": 1000.0,
            "max_contra_age_minutes": 60.0,
            "max_confidence": 0.7,
        }
    )
    for minute in [0, 5, 10, 15, 20]:
        context.add_trade(make_trade(outcome_index=0, timestamp_ms=minute * 60_000))

    signal = detector.analyze(
        make_trade(outcome_index=1, timestamp_ms=40 * 60_000, notional_usdc=2_000),
        context,
    )

    assert signal is not None
    assert signal.metadata["contra_outcome_index"] == 0
    assert signal.metadata["silence_ratio"] == pytest.approx(4.0)


def test_contra_outcome_silence_detector_ignores_stale_or_under_sampled_contra_side(make_trade):
    context = DetectionContext()
    detector = ContraOutcomeSilenceDetector(
        {
            "min_gap_samples": 4,
            "silence_threshold": 3.0,
            "min_notional": 1000.0,
            "max_contra_age_minutes": 10.0,
        }
    )
    for minute in [0, 5, 10, 15, 20]:
        context.add_trade(make_trade(outcome_index=0, timestamp_ms=minute * 60_000))

    assert detector.analyze(make_trade(outcome_index=1, timestamp_ms=40 * 60_000, notional_usdc=2_000), context) is None
    assert detector.analyze(make_trade(outcome_index=2, timestamp_ms=21 * 60_000, notional_usdc=2_000), context) is None
