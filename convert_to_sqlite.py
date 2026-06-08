"""
One-time conversion of trades.csv to SQLite database.
Run once, then all future backtests are 100x faster.

Expected time: 30-60 minutes for 30GB file
"""

import pandas as pd
import sqlite3
import logging
import time
from pathlib import Path

def convert_trades_to_sqlite():
    """Convert trades.csv to SQLite with indices"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    csv_path = "data/processed/trades.csv"
    db_path = "data/trades.db"
    
    # Check if already exists
    if Path(db_path).exists():
        print(f"\n⚠️  {db_path} already exists!")
        response = input("Delete and recreate? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
        Path(db_path).unlink()
        print("Deleted existing database.")
    
    print("\n" + "="*80)
    print("CONVERTING TRADES.CSV TO SQLITE")
    print("="*80)
    print(f"\nInput:  {csv_path}")
    print(f"Output: {db_path}")
    print(f"\nThis will take 30-60 minutes but only runs once.")
    print("You can continue working in another terminal while this runs.\n")
    
    start_time = time.time()
    
    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    
    # Read CSV in chunks and insert
    chunk_size = 100000
    chunk_count = 0
    total_rows = 0
    
    logging.info("Reading and inserting data in chunks...")
    
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            # Insert chunk
            chunk.to_sql('trades', conn, if_exists='append', index=False)
            
            chunk_count += 1
            total_rows += len(chunk)
            
            # Progress update every 1M rows
            if chunk_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed
                print(f"  Processed {total_rows:,} rows in {elapsed:.0f}s "
                      f"({rate:,.0f} rows/sec)")
        
        elapsed_insert = time.time() - start_time
        logging.info(f"✓ Inserted {total_rows:,} rows in {elapsed_insert:.1f}s")
        
        # Create indices (THIS IS THE KEY SPEEDUP)
        logging.info("\nCreating indices (this makes queries 1000x faster)...")
        
        index_start = time.time()
        
        # Index on market_id (most important - used for filtering)
        logging.info("  Creating index on market_id...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_id ON trades(market_id)")
        
        # Index on timestamp (useful for time-range queries)
        logging.info("  Creating index on timestamp...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)")
        
        # Composite index for common query pattern
        logging.info("  Creating composite index on (market_id, timestamp)...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_timestamp ON trades(market_id, timestamp)")
        
        conn.commit()
        
        elapsed_index = time.time() - index_start
        logging.info(f"✓ Indices created in {elapsed_index:.1f}s")
        
        # Get database size
        db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
        
        # Final stats
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("✓ CONVERSION COMPLETE!")
        print("="*80)
        print(f"\nStatistics:")
        print(f"  Total rows:     {total_rows:,}")
        print(f"  Database size:  {db_size_mb:,.1f} MB")
        print(f"  Total time:     {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Insert rate:    {total_rows/elapsed_insert:,.0f} rows/sec")
        print(f"\nNext step:")
        print(f"  The data_loader.py has been updated to use SQLite automatically.")
        print(f"  Re-run your tests: python -m pytest tests/test_data_loader.py")
        print(f"  Market loading should now take <10 seconds instead of 26 minutes!")
        
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return
    
    conn.close()


if __name__ == "__main__":
    convert_trades_to_sqlite()
