import pytest

from backtesting.data_loader import HistoricalDataLoader
from models import TradeBatch


def test_timestamp_to_ms_normalizes_seconds_milliseconds_and_iso_strings():
    assert HistoricalDataLoader._timestamp_to_ms(1_704_067_200) == 1_704_067_200_000
    assert HistoricalDataLoader._timestamp_to_ms(1_704_067_200_123) == 1_704_067_200_123
    assert HistoricalDataLoader._timestamp_to_ms(1_704_067_200.0) == 1_704_067_200_000
    assert HistoricalDataLoader._timestamp_to_ms("2024-01-01T00:00:00Z") == 1_704_067_200_000
    assert HistoricalDataLoader._timestamp_to_ms("2024-01-01T01:00:00+01:00") == 1_704_067_200_000


def test_sqlite_loader_reads_metadata_counts_filters_and_batches(sqlite_data_dir):
    loader = HistoricalDataLoader(str(sqlite_data_dir), cache_size=2)
    loader.load_data()

    assert loader.use_sqlite is True
    assert loader.get_market_metadata(1)["condition_id"] == "condition-1"
    assert loader.get_markets_by_volume(min_volume=15_000) == [1]
    assert loader.get_trade_count(1) == 2
    assert loader.get_trade_count(1, min_usd_amount=75.0) == 1

    trades = loader.get_trades_for_market(1, min_usd_amount=75.0)

    assert isinstance(trades, TradeBatch)
    assert len(trades) == 1
    assert trades[0].wallet == "0xa"
    assert trades[0].outcome_index == 0
    assert trades[0].asset == "asset-no"
    assert trades[0].timestamp_ms == 1_704_067_200_000
    loader.close()


def test_sqlite_loader_lru_cache_eviction(sqlite_data_dir):
    loader = HistoricalDataLoader(str(sqlite_data_dir), cache_size=1)
    loader.load_data()

    first = loader.get_trades_for_market(1)
    assert loader.get_trades_for_market(1) is first
    loader.get_trades_for_market(2)

    assert list(loader._trade_cache.keys()) == [(2, None, None, None, False)]
    loader.close()


def test_sqlite_loader_filters_by_trade_time_window(sqlite_data_dir):
    loader = HistoricalDataLoader(
        str(sqlite_data_dir),
        cache_size=2,
        trade_start_time="2024-01-01T00:01:00Z",
        trade_end_time="2024-01-01T00:01:00Z",
    )
    loader.load_data()

    assert loader.get_trade_count(1) == 1

    trades = loader.get_trades_for_market(1)

    assert len(trades) == 1
    assert trades[0].wallet == "0xb"
    assert trades[0].timestamp_ms == 1_704_067_260_000
    assert list(loader._trade_cache.keys()) == [
        (1, None, 1_704_067_260_000, 1_704_067_260_000, False)
    ]
    loader.close()


def test_csv_loader_fallback_reads_sorted_trade_batches(csv_data_dir):
    loader = HistoricalDataLoader(str(csv_data_dir), cache_size=0)
    loader.load_data()

    assert loader.use_sqlite is False
    trades = loader.get_trades_for_market(1)

    assert [t.tx_hash for t in trades] == ["tx1", "tx2"]
    assert [t.timestamp_ms for t in trades] == [1_704_067_200_000, 1_704_067_260_000]
    assert [t.outcome for t in trades] == ["NO", "YES"]


def test_csv_loader_fallback_filters_by_trade_time_window(csv_data_dir):
    loader = HistoricalDataLoader(
        str(csv_data_dir),
        cache_size=0,
        trade_start_time="2024-01-01T00:01:00Z",
        trade_end_time="2024-01-01T00:01:00Z",
    )
    loader.load_data()

    assert loader.get_trade_count(1) == 1

    trades = loader.get_trades_for_market(1)

    assert len(trades) == 1
    assert trades[0].tx_hash == "tx2"


def test_loader_returns_empty_batch_when_market_metadata_is_missing(sqlite_data_dir):
    loader = HistoricalDataLoader(str(sqlite_data_dir), cache_size=0)
    loader.load_data()

    trades = loader.get_trades_for_market(999)

    assert isinstance(trades, TradeBatch)
    assert len(trades) == 0
    assert trades.condition_id == ""
    loader.close()


def test_loader_requires_metadata_before_market_queries(sqlite_data_dir):
    loader = HistoricalDataLoader(str(sqlite_data_dir), cache_size=0)

    with pytest.raises(RuntimeError, match="Call load_data"):
        loader.get_markets_by_volume()
