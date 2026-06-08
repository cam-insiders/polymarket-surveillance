#!/usr/bin/env python3
"""
Convert trades.csv in a HistoricalDataLoader data directory to trades.db.

Usage:
    python3 scripts/csv_dir_to_sqlite.py --data-dir data/curated_fromvm
    python3 scripts/csv_dir_to_sqlite.py --data-dir data/curated_fromvm --overwrite

    python3 -m experiments.curated_reported_insider_recall config.json \\
        --data-dir data/curated_fromvm
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd

DEFAULT_CHUNK_SIZE = 100_000


def convert_csv_dir_to_sqlite(
    *,
    data_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
) -> int:
    csv_path = data_dir / "trades.csv"
    db_path = data_dir / "trades.db"
    markets_path = data_dir / "markets.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing trades CSV: {csv_path}")
    if not markets_path.exists():
        raise FileNotFoundError(
            f"Missing markets.csv (required by HistoricalDataLoader): {markets_path}"
        )

    if db_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{db_path} already exists; pass --overwrite to replace it"
            )
        db_path.unlink()
        logging.info("Removed existing %s", db_path)

    start_time = time.time()
    conn = sqlite3.connect(db_path)
    total_rows = 0

    try:
        logging.info("Reading %s", csv_path)
        for chunk_count, chunk in enumerate(
            pd.read_csv(csv_path, chunksize=chunk_size),
            start=1,
        ):
            chunk.to_sql("trades", conn, if_exists="append", index=False)
            total_rows += len(chunk)
            if chunk_count % 10 == 0:
                elapsed = time.time() - start_time
                logging.info("Inserted %s rows (%.0fs)", f"{total_rows:,}", elapsed)

        logging.info("Creating SQLite indexes")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_id ON trades(market_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_timestamp ON trades(market_id, timestamp)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_maker ON trades(maker)")
        conn.commit()
    except Exception:
        conn.close()
        if db_path.exists():
            db_path.unlink()
        raise
    finally:
        conn.close()

    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time
    logging.info(
        "Done: %s rows -> %s (%.1f MB, %.1fs)",
        f"{total_rows:,}",
        db_path,
        db_size_mb,
        elapsed,
    )
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build trades.db from trades.csv in a data directory."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/curated_fromvm"),
        help="Directory containing trades.csv and markets.csv",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    data_dir = args.data_dir.resolve()
    total_rows = convert_csv_dir_to_sqlite(
        data_dir=data_dir,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    print(f"\nReady: {data_dir}/trades.db ({total_rows:,} rows)")
    print(
        "Run curated recall with:\n"
        f"  python3 -m experiments.curated_reported_insider_recall path/to/config.json "
        f"--data-dir {data_dir}"
    )


if __name__ == "__main__":
    main()
