import numpy as np
import pytest

from detectors.base import DetectionContext, RollingStats
from models import OrderBookSnapshot, Trade, TradeBatch, WalletProfile, filter_trades_by_notional


def test_rolling_stats_evicts_old_values_and_computes_sample_std():
    stats = RollingStats(window_hours=1)
    stats.add_trade(0, 10.0)
    stats.add_trade(30 * 60_000, 20.0)

    assert stats.get_count() == 2
    assert stats.get_mean() == pytest.approx(15.0)
    assert stats.get_std() == pytest.approx(7.0710678119)

    stats.add_trade(2 * 60 * 60_000, 40.0)

    assert stats.get_count() == 1
    assert stats.get_mean() == pytest.approx(40.0)
    assert stats.get_std() is None


def test_detection_context_updates_market_wallet_orderbook_and_outcome_gap_state(make_trade, make_orderbook):
    context = DetectionContext()
    context.market_stats_window_hours["condition-1"] = 1
    context.register_market_assets("condition-1", ["asset-no", "asset-yes"])
    context.update_orderbook("asset-yes", make_orderbook(asset_id="asset-yes"))

    first = make_trade(wallet="0xa", outcome_index=1, price=0.55, timestamp_ms=1_000)
    second = make_trade(wallet="0xa", outcome_index=1, price=0.60, timestamp_ms=61_000)
    context.add_trade(first)
    context.add_trade(second)

    assert context.last_prices[("condition-1", 1)] == 0.60
    assert context.wallet_profiles["0xa"].total_trades == 2
    assert len(context.wallet_activity[("0xa", "condition-1")]) == 2
    assert context.market_stats["condition-1"].window_ms == 3_600_000
    assert context.get_orderbook_for_outcome("condition-1", 1).asset_id == "asset-yes"
    assert list(context.outcome_trade_gaps[("condition-1", 1)]) == [60_000]


def test_wallet_profile_outcome_concentration_directionality_and_hedging(make_trade):
    profile = WalletProfile(wallet_address="0xwallet")
    profile.update(make_trade(wallet="0xwallet", outcome_index=0, side="BUY", notional_usdc=10_000, price=0.5))
    profile.update(make_trade(wallet="0xwallet", outcome_index=0, side="SELL", notional_usdc=4_000, price=0.5))
    profile.update(make_trade(wallet="0xwallet", outcome_index=1, side="BUY", notional_usdc=6_000, price=0.5))

    assert profile.get_outcome_concentration("condition-1", 0) == pytest.approx(0.7)
    assert profile.get_outcome_position("condition-1", 0) == pytest.approx(12_000)
    assert profile.get_outcome_directional_ratio("condition-1", 0) == pytest.approx(6_000 / 14_000)
    assert profile.is_hedged("condition-1", threshold=0.3)


def test_trade_parsing_batch_operations_and_filter_helper(make_batch):
    raw = {
        "proxyWallet": "0xapi",
        "conditionId": "condition-api",
        "slug": "api-market",
        "side": "BUY",
        "outcome": "YES",
        "outcomeIndex": "1",
        "size": "10",
        "price": "0.42",
        "timestamp": "1704067200",
        "transactionHash": "tx-api",
        "asset": "asset-yes",
    }
    trade = Trade.from_api_response(raw)
    assert trade is not None
    assert trade.notional_usdc == pytest.approx(4.2)
    assert trade.timestamp_ms == 1_704_067_200_000

    batch = make_batch()
    assert batch[-1].wallet == "0xc"
    assert batch[1].asset == "asset-yes"
    assert [t.tx_hash for t in batch[1:3]] == ["tx2", "tx3"]
    assert [t.wallet for t in batch.take([3, 0])] == ["0xc", "0xa"]
    assert [t.timestamp_ms for t in batch.sort_by_timestamp()] == [1000, 2000, 3000, 4000]
    assert [t.tx_hash for t in filter_trades_by_notional(batch, 100.0)] == ["tx4"]
    assert [t.tx_hash for t in filter_trades_by_notional(batch.to_trade_list(), 100.0)] == ["tx4"]

    empty = batch.filter_mask(np.zeros(len(batch), dtype=bool))
    assert not empty
    assert TradeBatch.concat([batch[:2], batch[2:]]).to_trade_list()[2].tx_hash == "tx3"


def test_orderbook_snapshot_depth_imbalance_and_spread():
    book = OrderBookSnapshot(
        asset_id="asset",
        timestamp_ms=0,
        bids=[(0.49, 100.0), (0.48, 200.0)],
        asks=[(0.51, 50.0), (0.52, 100.0)],
    )

    assert book.get_bid_depth() == 300.0
    assert book.get_bid_depth(price_threshold=0.485) == 100.0
    assert book.get_ask_depth() == 150.0
    assert book.get_imbalance_ratio() == pytest.approx(2.0)
    assert book.get_spread_bps() == pytest.approx(400.0)
