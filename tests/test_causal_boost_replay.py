import numpy as np
import pytest

from backtesting.backtest_runner import BacktestRunner
from backtesting.causal_boost_replay import build_live_parity_boost_schedule, build_trade_batches


def test_build_trade_batches_uses_event_time_windows(make_trade):
    trades = [
        make_trade(timestamp_ms=0),
        make_trade(timestamp_ms=1_000),
        make_trade(timestamp_ms=5_000),
        make_trade(timestamp_ms=9_000),
        make_trade(timestamp_ms=12_000),
    ]

    assert build_trade_batches(trades, poll_interval_seconds=5.0) == [(0, 2), (2, 4), (4, 5)]
    assert build_trade_batches([], poll_interval_seconds=5.0) == []


def test_empty_boost_schedule_has_empty_arrays():
    schedule = build_live_parity_boost_schedule(
        detector_trades=[],
        market_id="1",
        clustering_config=None,
        jump_anticipation_config=None,
    )

    assert schedule.score_multiplier_by_trade_idx.shape == (0,)
    assert schedule.cluster_multiplier_by_trade_idx.shape == (0,)
    assert schedule.ja_multiplier_by_trade_idx.shape == (0,)


def test_jump_anticipation_multiplier_is_batch_delayed(monkeypatch, make_trade):
    monkeypatch.setattr("backtesting.causal_boost_replay.find_jumps", lambda trades, cfg: [{"jump": 1}])
    monkeypatch.setattr("backtesting.causal_boost_replay.score_wallets_jump_anticipation", lambda trades, jumps, cfg: {"0xa": 1.8})
    trades = [
        make_trade(wallet=f"0x{i}", timestamp_ms=i * 100, notional_usdc=2_000)
        for i in range(10)
    ]
    trades.append(make_trade(wallet="0xa", timestamp_ms=7_000, notional_usdc=2_000))

    schedule = build_live_parity_boost_schedule(
        detector_trades=trades,
        market_id="1",
        clustering_config=None,
        jump_anticipation_config={"scoring_interval_minutes": 0.0, "buffer_hours": 24.0},
        poll_interval_seconds=5.0,
    )

    assert schedule.ja_multiplier_by_trade_idx[:10].tolist() == pytest.approx([1.0] * 10)
    assert schedule.ja_multiplier_by_trade_idx[10] == pytest.approx(1.8)
    assert schedule.ja_score_batch_indices == [0, 1]


def test_cluster_recluster_respects_event_time_interval_gate(make_trade):
    trades = [
        make_trade(wallet="0xa", timestamp_ms=0, notional_usdc=6_000),
        make_trade(wallet="0xb", timestamp_ms=60_000, notional_usdc=6_000),
        make_trade(wallet="0xc", timestamp_ms=120_000, notional_usdc=6_000),
        make_trade(wallet="0xa", timestamp_ms=360_000, notional_usdc=6_000),
    ]
    clustering_cfg = {
        "bucket_size": 300,
        "size_normalizer": 10_000,
        "max_size_mult": 5.0,
        "k_core": 2,
        "min_edge_weight": 0.5,
        "min_cluster_interval": 300,
        "max_cluster_interval": 3600,
        "significant_change_threshold": 2,
        "boost": {"max_boost_factor": 2.0},
    }

    schedule = build_live_parity_boost_schedule(
        detector_trades=trades,
        market_id="1",
        clustering_config=clustering_cfg,
        clustering_min_trade_size=1000.0,
        jump_anticipation_config=None,
        poll_interval_seconds=5.0,
    )

    assert schedule.cluster_update_trade_indices == [3]


def test_no_boost_schedule_is_equivalent_to_baseline_runner(make_trade, tiny_config):
    trades = [
        make_trade(wallet="0xa", price=0.40, timestamp_ms=0),
        make_trade(wallet="0xa", price=0.50, timestamp_ms=60_000),
        make_trade(wallet="0xb", price=0.30, timestamp_ms=120_000),
    ]
    runner = BacktestRunner(tiny_config, include_recidivism=False)
    baseline = runner.run_backtest(trades, {"id": 1}, capture_alerts=False, capture_trade_features=False, progress_every=0)
    schedule = build_live_parity_boost_schedule(
        detector_trades=trades,
        market_id="1",
        clustering_config=None,
        jump_anticipation_config=None,
        poll_interval_seconds=5.0,
    )
    replay = runner.run_backtest(
        trades,
        {"id": 1},
        capture_alerts=False,
        capture_trade_features=False,
        progress_every=0,
        score_multipliers=schedule.score_multiplier_by_trade_idx,
        score_cap=schedule.score_cap,
    )

    assert np.all(schedule.score_multiplier_by_trade_idx == 1.0)
    assert replay.alerts_generated == baseline.alerts_generated
    assert replay.wallet_flags == baseline.wallet_flags
    assert replay.wallet_suspicion == baseline.wallet_suspicion
