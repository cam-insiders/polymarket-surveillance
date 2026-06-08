#!/usr/bin/env python3
"""
Measure TradeBatch storage for one market.

Example:
    PYTHONPATH=. python3 scripts/measure_trade_batch_memory.py --market-id 253591
"""

from __future__ import annotations

import argparse
import resource
import sys

from backtesting.data_loader import HistoricalDataLoader


def _format_bytes(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    return f"{size:,.2f} TB"


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TradeBatch memory for a market.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--market-id", type=int, default=None)
    parser.add_argument("--min-usd-amount", type=float, default=None)
    args = parser.parse_args()

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()
    try:
        market_id = args.market_id
        if market_id is None:
            market_id = loader.get_markets_by_volume(min_volume=0, limit=1)[0]

        trades = loader.get_trades_for_market(
            market_id,
            min_usd_amount=args.min_usd_amount,
            use_cache=False,
        )
        batch_bytes = getattr(trades, "nbytes", 0)
        print(f"market_id={market_id}")
        print(f"trades={len(trades):,}")
        print(f"batch_storage={_format_bytes(batch_bytes)}")
        print(f"peak_process_rss={_format_bytes(_peak_rss_bytes())}")
        if len(trades) > 0:
            print(f"bytes_per_trade={batch_bytes / len(trades):,.1f}")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
