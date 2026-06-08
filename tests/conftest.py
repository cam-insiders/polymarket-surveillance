import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from models import OrderBookSnapshot, Trade, TradeBatch


@pytest.fixture
def make_trade():
    def _make_trade(
        *,
        wallet="0xabc",
        condition_id="condition-1",
        market_slug="market-1",
        side="BUY",
        outcome=None,
        outcome_index=0,
        size_tokens=None,
        price=0.5,
        notional_usdc=1000.0,
        timestamp_ms=0,
        tx_hash=None,
        asset=None,
    ) -> Trade:
        if size_tokens is None:
            size_tokens = notional_usdc / price if price > 0 else notional_usdc
        if outcome is None:
            outcome = "YES" if outcome_index == 1 else "NO"
        if tx_hash is None:
            tx_hash = f"tx-{wallet}-{timestamp_ms}-{outcome_index}-{side}"
        if asset is None:
            asset = f"asset-{outcome_index}"
        return Trade(
            wallet=wallet,
            condition_id=condition_id,
            market_slug=market_slug,
            side=side,
            outcome=outcome,
            outcome_index=outcome_index,
            size_tokens=float(size_tokens),
            price=float(price),
            notional_usdc=float(notional_usdc),
            timestamp_ms=int(timestamp_ms),
            tx_hash=tx_hash,
            asset=asset,
        )

    return _make_trade


@pytest.fixture
def make_batch():
    def _make_batch(
        *,
        wallets=("0xa", "0xb", "0xa", "0xc"),
        sides=("BUY", "SELL", "BUY", "SELL"),
        outcome_index=(0, 1, 0, 1),
        size_tokens=(100.0, 200.0, 300.0, 400.0),
        price=(0.5, 0.4, 0.25, 0.8),
        notional_usdc=(50.0, 80.0, 75.0, 320.0),
        timestamp_ms=(3000, 1000, 2000, 4000),
        tx_hashes=("tx1", "tx2", "tx3", "tx4"),
        condition_id="condition-1",
        market_slug="market-1",
        outcomes=("NO", "YES"),
        assets=("asset-no", "asset-yes"),
    ) -> TradeBatch:
        return TradeBatch.from_columns(
            wallets=wallets,
            sides=sides,
            outcome_index=outcome_index,
            size_tokens=size_tokens,
            price=price,
            notional_usdc=notional_usdc,
            timestamp_ms=timestamp_ms,
            tx_hashes=tx_hashes,
            condition_id=condition_id,
            market_slug=market_slug,
            outcomes=outcomes,
            assets=assets,
        )

    return _make_batch


@pytest.fixture
def make_orderbook():
    def _make_orderbook(
        *,
        asset_id="asset-0",
        bids=((0.49, 100.0), (0.48, 200.0)),
        asks=((0.51, 100.0), (0.52, 200.0)),
        timestamp_ms=0,
    ) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            asset_id=asset_id,
            timestamp_ms=timestamp_ms,
            bids=list(bids),
            asks=list(asks),
        )

    return _make_orderbook


@pytest.fixture
def tiny_config():
    return {
        "alert_threshold": 0.5,
        "detectors": {
            "volume_anomaly": {
                "lookback_window_hours": 24,
                "min_trades_for_baseline": 3,
                "z_score_threshold": 99.0,
                "min_absolute_notional": 1_000_000.0,
                "max_confidence": 0.0,
            },
            "probability_impact": {
                "min_delta_prob": 0.05,
                "min_delta_log_odds": 99.0,
                "min_notional": 0.0,
                "max_confidence": 0.6,
            },
            "accumulation_detector": {
                "min_accumulation_usdc": 1_000_000.0,
                "min_directional_ratio": 0.99,
                "max_confidence": 0.0,
                "min_outcome_concentration": 0.99,
            },
            "extreme_position": {
                "tail_threshold": 0.01,
                "min_notional": 1_000_000.0,
                "max_confidence": 0.0,
            },
            "contra_outcome_silence": {
                "min_gap_samples": 99,
                "silence_threshold": 99.0,
                "min_notional": 1_000_000.0,
                "max_contra_age_minutes": 1.0,
                "max_confidence": 0.0,
            },
            "recidivism_detector": {
                "min_prior_flags": 1,
                "midpoint": 1,
                "k": 1.0,
                "max_confidence": 0.4,
            },
        },
    }


class FakeWalletCache:
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def get_wallet_info(self, wallet):
        return self.mapping.get(wallet)


@pytest.fixture
def fake_wallet_cache():
    return FakeWalletCache


@pytest.fixture
def sqlite_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_markets_csv(data_dir)
    _write_sqlite_trades(data_dir / "trades.db")
    return data_dir


@pytest.fixture
def csv_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    _write_markets_csv(data_dir)
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "market_id": 1,
                "maker": "0xa",
                "taker": "0xt",
                "nonusdc_side": "token1",
                "maker_direction": "BUY",
                "taker_direction": "SELL",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transactionHash": "tx1",
            },
            {
                "timestamp": "2024-01-01T00:01:00Z",
                "market_id": 1,
                "maker": "0xb",
                "taker": "0xt",
                "nonusdc_side": "token2",
                "maker_direction": "SELL",
                "taker_direction": "BUY",
                "price": 0.4,
                "usd_amount": 50.0,
                "token_amount": 125.0,
                "transactionHash": "tx2",
            },
        ]
    ).to_csv(processed / "trades.csv", index=False)
    return data_dir


def _write_markets_csv(data_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "id": 1,
                "condition_id": "condition-1",
                "market_slug": "market-1",
                "token1": "asset-no",
                "token2": "asset-yes",
                "answer1": "NO",
                "answer2": "YES",
                "volume": 20_000.0,
                "closedTime": "2024-01-02T00:00:00Z",
            },
            {
                "id": 2,
                "condition_id": "condition-2",
                "market_slug": "market-2",
                "token1": "asset-2-no",
                "token2": "asset-2-yes",
                "answer1": "NO",
                "answer2": "YES",
                "volume": 10_000.0,
                "closedTime": "2024-01-03T00:00:00Z",
            },
        ]
    ).to_csv(data_dir / "markets.csv", index=False)


def _write_sqlite_trades(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
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
            (
                "2024-01-01T00:00:00Z",
                1,
                "0xa",
                "0xt",
                "token1",
                "BUY",
                "SELL",
                0.5,
                100.0,
                200.0,
                "tx1",
            ),
            (
                "2024-01-01T00:01:00Z",
                1,
                "0xb",
                "0xt",
                "token2",
                "SELL",
                "BUY",
                0.4,
                50.0,
                125.0,
                "tx2",
            ),
            (
                "2024-01-01T00:02:00Z",
                2,
                "0xc",
                "0xt",
                "token1",
                "BUY",
                "SELL",
                0.25,
                200.0,
                800.0,
                "tx3",
            ),
        ],
    )
    conn.commit()
    conn.close()
