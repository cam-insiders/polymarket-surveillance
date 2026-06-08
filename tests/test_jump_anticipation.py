import pytest

from backtesting.backtest_runner import BacktestResult
from jump_anticipation.core import (
    JumpEvent,
    apply_jump_boost,
    find_jumps,
    run_jump_anticipation_boost,
    score_wallets_jump_anticipation,
)
from jump_anticipation.manager import JumpAnticipationManager


def test_find_jumps_detects_signed_price_moves_by_outcome(make_trade):
    trades = [
        make_trade(outcome_index=0, price=0.40, timestamp_ms=0),
        make_trade(outcome_index=0, price=0.46, timestamp_ms=10 * 60_000),
        make_trade(outcome_index=1, price=0.70, timestamp_ms=0),
        make_trade(outcome_index=1, price=0.62, timestamp_ms=5 * 60_000),
    ]

    jumps = find_jumps(trades, {"jump_threshold": 0.05, "jump_window_minutes": 30})

    assert {(j.outcome_index, j.direction) for j in jumps} == {(0, 1), (1, -1)}
    assert all(abs(j.price_change) >= 0.05 for j in jumps)


def test_score_wallets_counts_each_trade_once_against_largest_relevant_jump(make_trade):
    jumps = [
        JumpEvent(outcome_index=0, jump_start_ms=10 * 60_000, jump_end_ms=11 * 60_000, price_change=0.05, direction=1),
        JumpEvent(outcome_index=0, jump_start_ms=20 * 60_000, jump_end_ms=21 * 60_000, price_change=0.20, direction=-1),
    ]
    trades = [
        make_trade(wallet="0xa", side="SELL", outcome_index=0, timestamp_ms=0, notional_usdc=1_000),
        make_trade(wallet="0xb", side="BUY", outcome_index=0, timestamp_ms=0, notional_usdc=1_000),
        make_trade(wallet="0xc", side="BUY", outcome_index=1, timestamp_ms=0, notional_usdc=1_000),
    ]

    scores = score_wallets_jump_anticipation(
        trades,
        jumps,
        {"pre_jump_lookback_minutes": 30, "min_pre_jump_trades": 1, "max_boost_factor": 2.0},
    )

    assert scores["0xa"] == pytest.approx(2.0)
    assert scores["0xb"] == pytest.approx(1.0)
    assert scores["0xc"] == pytest.approx(1.0)


def test_score_wallets_applies_minimum_evidence_and_notional_filters(make_trade):
    jumps = [JumpEvent(0, 10 * 60_000, 11 * 60_000, 0.10, 1)]
    trades = [
        make_trade(wallet="0xa", side="BUY", outcome_index=0, timestamp_ms=0, notional_usdc=1_000),
        make_trade(wallet="0xb", side="BUY", outcome_index=0, timestamp_ms=0, notional_usdc=10),
    ]

    scores = score_wallets_jump_anticipation(
        trades,
        jumps,
        {
            "pre_jump_lookback_minutes": 30,
            "min_pre_jump_trades": 2,
            "max_boost_factor": 2.0,
            "min_trade_notional": 100.0,
        },
    )

    assert scores == {"0xa": 1.0}


def test_apply_and_run_jump_boost_mutate_result_boosts(make_trade):
    result = BacktestResult(
        total_trades=0,
        alerts_generated=0,
        alerts=[],
        detector_stats={},
        all_trade_features=[],
        wallet_suspicion={"0xa": 1.0, "0xb": 1.0},
        wallet_cluster_boost={"0xa": 1.5},
    )
    apply_jump_boost(result, {"0xa": 1.2, "0xb": 1.4, "0xc": 1.0})

    assert result.wallet_cluster_boost["0xa"] == pytest.approx(1.8)
    assert result.wallet_cluster_boost["0xb"] == pytest.approx(1.4)
    assert "0xc" not in result.wallet_cluster_boost

    pipeline_result = BacktestResult(0, 0, [], {}, [], wallet_suspicion={"0xa": 1.0})
    trades = [
        make_trade(wallet="0xa", side="BUY", outcome_index=0, price=0.40, timestamp_ms=0),
        make_trade(wallet="0xprice", side="BUY", outcome_index=0, price=0.50, timestamp_ms=10 * 60_000),
    ]
    diag = run_jump_anticipation_boost(
        pipeline_result,
        all_trades=trades,
        scoring_trades=[trades[0]],
        config={
            "jump_threshold": 0.05,
            "jump_window_minutes": 30,
            "pre_jump_lookback_minutes": 30,
            "min_pre_jump_trades": 1,
            "max_boost_factor": 2.0,
        },
    )

    assert diag["n_jumps"] >= 1
    assert pipeline_result.wallet_cluster_boost["0xa"] == pytest.approx(2.0)


def test_jump_anticipation_manager_prunes_buffer_and_scores_periodically(make_trade, monkeypatch):
    monkeypatch.setattr("jump_anticipation.manager.time.time", lambda: 100.0)
    monkeypatch.setattr("jump_anticipation.manager.find_jumps", lambda trades, cfg: [JumpEvent(0, 1, 2, 0.1, 1)])
    monkeypatch.setattr("jump_anticipation.manager.score_wallets_jump_anticipation", lambda trades, jumps, cfg: {"0x0": 1.6})

    manager = JumpAnticipationManager(
        {
            "scoring_interval_minutes": 0.0,
            "buffer_hours": 0.001,
            "jump_window_minutes": 0.01,
            "pre_jump_lookback_minutes": 0.01,
        }
    )
    manager.on_trade(make_trade(wallet="0xold", timestamp_ms=0))
    for idx in range(10):
        manager.on_trade(make_trade(wallet=f"0x{idx}", timestamp_ms=10_000 + idx))

    assert all(trade.wallet != "0xold" for trade in manager._buffer)
    assert manager.maybe_score() is True
    assert manager.get_wallet_boost("0x0") == pytest.approx(1.6)
    assert manager.get_wallet_boost("0xmissing") == pytest.approx(1.0)
