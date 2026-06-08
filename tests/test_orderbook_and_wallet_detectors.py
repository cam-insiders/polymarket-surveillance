from detectors.base import DetectionContext
from detectors.orderbook_detectors import (
    OrderbookConsumptionDetector,
    OrderbookImbalanceDetector,
    ThinLiquidityExploitDetector,
)
from detectors.wallet_detectors import NewWalletDetector, RecidivismDetector


def _context_with_book(make_orderbook, book):
    context = DetectionContext()
    context.register_market_assets("condition-1", ["asset-0", "asset-1"])
    context.update_orderbook("asset-0", book)
    return context


def test_orderbook_consumption_detector_flags_multi_level_fills(make_trade, make_orderbook):
    book = make_orderbook(asks=[(0.50, 10.0), (0.51, 10.0), (0.52, 10.0)])
    context = _context_with_book(make_orderbook, book)
    detector = OrderbookConsumptionDetector(
        {"min_levels_consumed": 3, "min_notional": 0.0, "max_slippage_bps": 50.0, "max_confidence": 1.0}
    )

    signal = detector.analyze(make_trade(side="BUY", outcome_index=0, size_tokens=25, notional_usdc=2_000), context)

    assert signal is not None
    assert signal.metadata["levels_consumed"] == 3
    assert signal.metadata["slippage_bps"] > 0


def test_orderbook_imbalance_detector_flags_trading_against_heavy_side(make_trade, make_orderbook):
    book = make_orderbook(bids=[(0.49, 1_000.0)], asks=[(0.51, 100.0)])
    context = _context_with_book(make_orderbook, book)
    detector = OrderbookImbalanceDetector({"min_imbalance_ratio": 3.0, "min_notional": 0.0, "max_confidence": 1.0})

    signal = detector.analyze(make_trade(side="SELL", outcome_index=0, notional_usdc=1_000), context)

    assert signal is not None
    assert signal.metadata["imbalance_ratio"] == 10.0
    assert detector.analyze(make_trade(side="BUY", outcome_index=0, notional_usdc=1_000), context) is None


def test_thin_liquidity_detector_flags_large_trade_relative_to_depth(make_trade, make_orderbook):
    book = make_orderbook(bids=[(0.49, 100.0)], asks=[(0.51, 100.0)])
    context = _context_with_book(make_orderbook, book)
    detector = ThinLiquidityExploitDetector(
        {"min_depth_ratio": 0.3, "max_total_depth": 500.0, "min_notional": 0.0, "max_confidence": 1.0}
    )

    signal = detector.analyze(make_trade(side="BUY", outcome_index=0, size_tokens=60, notional_usdc=1_000), context)

    assert signal is not None
    assert signal.metadata["depth_ratio"] == 0.6
    assert signal.metadata["total_depth"] == 200.0


def test_recidivism_detector_scores_prior_wallet_flags(make_trade):
    context = DetectionContext()
    context.wallet_flag_counts["0xrepeat"] = 3
    detector = RecidivismDetector({"min_prior_flags": 2, "midpoint": 3, "k": 1.0, "max_confidence": 0.8})

    signal = detector.analyze(make_trade(wallet="0xrepeat"), context)

    assert signal is not None
    assert signal.metadata["flag_count"] == 3
    assert signal.confidence_score == 0.4
    assert detector.analyze(make_trade(wallet="0xquiet"), context) is None


def test_new_wallet_detector_uses_cache_and_notional_gate(make_trade, fake_wallet_cache):
    cache = fake_wallet_cache({"0xnew": {"is_new": True}, "0xold": {"is_new": False}})
    context = DetectionContext(wallet_cache=cache)
    detector = NewWalletDetector({"min_notional": 1_000.0, "max_confidence": 0.4})

    signal = detector.analyze(make_trade(wallet="0xnew", notional_usdc=2_000), context)

    assert signal is not None
    assert signal.metadata["is_new_to_polymarket"] is True
    assert signal.confidence_score == 0.4
    assert detector.analyze(make_trade(wallet="0xold", notional_usdc=2_000), context) is None
    assert detector.analyze(make_trade(wallet="0xnew", notional_usdc=999), context) is None
