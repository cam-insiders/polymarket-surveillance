import sqlite3

import pandas as pd
import pytest

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import evaluate_config
from experiments.compare_sota_timeframe import _load_full_market_ground_truth_entries
from experiments.sota_algorithms.common import build_wallet_insider_labels


def _write_timeframe_ground_truth_fixture(data_dir):
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "id": 101,
                "condition_id": "condition-101",
                "market_slug": "timeframe-ground-truth-market",
                "token1": "asset-no",
                "token2": "asset-yes",
                "answer1": "NO",
                "answer2": "YES",
                "volume": 10_000.0,
                "closedTime": "2024-01-12T00:00:00Z",
            }
        ]
    ).to_csv(data_dir / "markets.csv", index=False)

    conn = sqlite3.connect(data_dir / "trades.db")
    conn.execute(
        """
        CREATE TABLE trades (
            timestamp TEXT,
            market_id INTEGER,
            maker TEXT,
            taker TEXT,
            nonusdc_side TEXT,
            maker_direction TEXT,
            taker_direction TEXT,
            price REAL,
            usd_amount REAL,
            token_amount REAL,
            transactionHash TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # Pre-window trade: this is the position history the regression
            # protects. Without it, alpha's return is only 1.0 and its z-score
            # falls below the test threshold.
            (
                "2024-01-01T00:00:00",
                101,
                "0xalpha",
                "0xt",
                "token1",
                "BUY",
                "SELL",
                0.20,
                200.0,
                1000.0,
                "tx-alpha-pre",
            ),
            (
                "2024-01-10T00:00:00",
                101,
                "0xalpha",
                "0xt",
                "token1",
                "BUY",
                "SELL",
                0.50,
                100.0,
                200.0,
                "tx-alpha-window",
            ),
            (
                "2024-01-10T00:01:00",
                101,
                "0xbeta",
                "0xt",
                "token2",
                "BUY",
                "SELL",
                0.50,
                100.0,
                200.0,
                "tx-beta-window",
            ),
            (
                "2024-01-10T00:02:00",
                101,
                "0xgamma",
                "0xt",
                "token1",
                "BUY",
                "SELL",
                0.50,
                100.0,
                200.0,
                "tx-gamma-window",
            ),
        ],
    )
    conn.commit()
    conn.close()


def test_timeframe_replay_uses_full_market_history_for_wallet_ground_truth(tmp_path, tiny_config):
    data_dir = tmp_path / "data"
    _write_timeframe_ground_truth_fixture(data_dir)

    loader = HistoricalDataLoader(
        str(data_dir),
        cache_size=0,
        trade_start_time="2024-01-10T00:00:00",
        trade_end_time="2024-01-10T23:59:59",
    )
    loader.load_data()

    # Sanity check: the active replay window excludes alpha's pre-window buy.
    assert loader.get_trade_count(101) == 3
    assert loader.get_trade_count(101, min_usd_amount=None) == 3
    assert len(loader.get_trades_for_market(101, ignore_trade_time_bounds=True)) == 4

    result = evaluate_config(
        config=tiny_config,
        loader=loader,
        market_ids=[101],
        prediction_mode="flag_rate",
        flag_rate_threshold=0.2,
        suspicion_threshold=2.0,
        z_score_threshold=0.9,
        min_wallet_notional=0.0,
        min_usd_amount=None,
        include_recidivism=False,
        clustering_config=None,
        jump_anticipation_config=None,
        measure_memory=False,
        winning_outcomes_override={101: 0},
    )

    rows = {row["wallet"]: row for row in result.wallet_evaluations}
    assert set(rows) == {"0xalpha", "0xbeta", "0xgamma"}

    alpha = rows["0xalpha"]
    assert alpha["gross_buy_notional"] == pytest.approx(300.0)
    assert alpha["total_notional"] == pytest.approx(300.0)
    assert alpha["net_pnl"] == pytest.approx(900.0)
    assert alpha["return"] == pytest.approx(3.0)
    assert alpha["z_score"] > 0.9
    assert alpha["is_insider"] is True

    assert rows["0xbeta"]["net_pnl"] == pytest.approx(-100.0)
    assert rows["0xbeta"]["return"] == pytest.approx(-1.0)
    assert rows["0xbeta"]["is_insider"] is False
    assert rows["0xgamma"]["net_pnl"] == pytest.approx(100.0)
    assert rows["0xgamma"]["return"] == pytest.approx(1.0)
    assert rows["0xgamma"]["is_insider"] is False

    # The detector replay itself remains timeframe-scoped.
    assert result.aggregate_performance.total_trades == 3
    loader.close()


def test_compare_sota_timeframe_uses_full_market_history_for_wallet_ground_truth(tmp_path):
    data_dir = tmp_path / "data"
    _write_timeframe_ground_truth_fixture(data_dir)

    loader = HistoricalDataLoader(
        str(data_dir),
        cache_size=0,
        trade_start_time="2024-01-10T00:00:00",
        trade_end_time="2024-01-10T23:59:59",
    )
    loader.load_data()

    entries = _load_full_market_ground_truth_entries(loader, [101])
    assert len(entries) == 4

    wallet_data = build_wallet_insider_labels(
        entries,
        {101: 0},
        z_score_threshold=0.9,
        min_wallet_notional=0.0,
    )
    rows = wallet_data[101]

    assert set(rows) == {"0xalpha", "0xbeta", "0xgamma"}
    assert rows["0xalpha"]["gross_buy"] == pytest.approx(300.0)
    assert rows["0xalpha"]["return"] == pytest.approx(3.0)
    assert rows["0xalpha"]["is_insider"] is True
    loader.close()
