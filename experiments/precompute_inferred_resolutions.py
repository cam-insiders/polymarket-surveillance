"""
Precompute inferred market resolutions and store them in SQLite.

This is intended as a one-time/background job so experiments can reuse inferred
resolutions instead of repeatedly scanning trades.

Usage:
    python -m experiments.precompute_inferred_resolutions
    python -m experiments.precompute_inferred_resolutions --output-db inferred_resolutions.db
    python -m experiments.precompute_inferred_resolutions --resolution-threshold 0.99 --min-trades 10
    python -m experiments.precompute_inferred_resolutions --output-csv experiments/results/inferred_resolutions_latest.csv
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False

from backtesting.data_loader import HistoricalDataLoader
from backtesting.trade_event_study import infer_market_winning_outcome_from_last_prices


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at_utc TEXT NOT NULL,
            data_dir TEXT NOT NULL,
            resolution_threshold REAL NOT NULL,
            min_trades INTEGER NOT NULL,
            min_usd_amount REAL,
            total_markets INTEGER NOT NULL,
            resolved_markets INTEGER NOT NULL,
            unresolved_markets INTEGER NOT NULL,
            too_few_trades_markets INTEGER NOT NULL,
            no_trades_markets INTEGER NOT NULL,
            error_markets INTEGER NOT NULL,
            min_closed_time_utc TEXT,
            max_closed_time_utc TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inferred_market_resolutions (
            job_id INTEGER NOT NULL,
            market_id INTEGER NOT NULL,
            market_slug TEXT,
            closed_time_utc TEXT,
            volume REAL,
            n_trades INTEGER NOT NULL,
            inferred_winning_outcome INTEGER,
            inference_status TEXT NOT NULL,
            latest_trade_ts_ms INTEGER,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (job_id, market_id),
            FOREIGN KEY (job_id) REFERENCES inference_jobs(job_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_imr_market_id ON inferred_market_resolutions(market_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_imr_closed_time ON inferred_market_resolutions(closed_time_utc)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_imr_status ON inferred_market_resolutions(inference_status)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_resolution_cache (
            market_id INTEGER PRIMARY KEY,
            market_slug TEXT,
            closed_time_utc TEXT,
            volume REAL,
            n_trades INTEGER NOT NULL,
            inferred_winning_outcome INTEGER,
            inference_status TEXT NOT NULL,
            latest_trade_ts_ms INTEGER,
            resolution_threshold REAL,
            updated_at_utc TEXT NOT NULL,
            source TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mrc_status ON market_resolution_cache(inference_status)"
    )
    conn.commit()


def _as_iso_utc(value: object) -> Optional[str]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _iter_markets(df: pd.DataFrame, limit: Optional[int]):
    rows = df[["id", "market_slug", "closedTime", "volume"]].itertuples(index=False, name=None)
    if limit is None:
        return list(rows)
    out = []
    for idx, row in enumerate(rows):
        if idx >= int(limit):
            break
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute inferred market resolutions and save to SQLite."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-db", type=str, default="inferred_resolutions.db")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional CSV export of latest job results")
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--min-usd-amount", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional market limit for test runs")
    parser.add_argument("--commit-every", type=int, default=500,
                        help="SQLite commit frequency")
    parser.add_argument("--resume-job-id", type=int, default=None,
                        help="Resume an existing inference job id")
    parser.add_argument("--resume-latest", action="store_true",
                        help="Resume latest inference job in the output DB")
    args = parser.parse_args()

    if args.resume_job_id is not None and args.resume_latest:
        raise ValueError("Use either --resume-job-id or --resume-latest, not both")

    Path(args.output_db).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    if loader.markets_df is None:
        loader.close()
        raise RuntimeError("HistoricalDataLoader.markets_df is not loaded")

    markets_df = loader.markets_df.copy()
    if "id" not in markets_df.columns:
        loader.close()
        raise RuntimeError("markets_df is missing required column: id")

    n_rows_before = len(markets_df)
    markets_df = markets_df.drop_duplicates(subset=["id"], keep="first")
    n_rows_after = len(markets_df)
    if n_rows_after < n_rows_before:
        logging.info(
            "Deduplicated market metadata rows by id: "
            f"{n_rows_before:,} -> {n_rows_after:,}"
        )
    markets = _iter_markets(markets_df, args.limit)
    total_markets = len(markets)

    logging.info(
        "Starting resolution inference: "
        f"markets={total_markets:,}, threshold={args.resolution_threshold}, "
        f"min_trades={args.min_trades}, min_usd_amount={args.min_usd_amount}"
    )

    conn = sqlite3.connect(args.output_db)
    try:
        _init_db(conn)

        resume_job_id: Optional[int] = args.resume_job_id
        if args.resume_latest:
            row = conn.execute("SELECT MAX(job_id) FROM inference_jobs").fetchone()
            resume_job_id = int(row[0]) if row and row[0] is not None else None

        if resume_job_id is not None:
            job_row = conn.execute(
                """
                SELECT
                    job_id,
                    data_dir,
                    resolution_threshold,
                    min_trades,
                    min_usd_amount,
                    total_markets
                FROM inference_jobs
                WHERE job_id = ?
                """,
                (int(resume_job_id),),
            ).fetchone()
            if job_row is None:
                raise ValueError(f"Job id {resume_job_id} not found in {args.output_db}")

            job_id = int(job_row[0])
            prev_data_dir = str(job_row[1])
            prev_threshold = float(job_row[2])
            prev_min_trades = int(job_row[3])
            prev_min_usd = job_row[4]
            prev_total_markets = int(job_row[5])

            if prev_data_dir != args.data_dir:
                raise ValueError(f"resume job data_dir mismatch: {prev_data_dir} != {args.data_dir}")
            if abs(prev_threshold - float(args.resolution_threshold)) > 1e-12:
                raise ValueError(
                    f"resume job resolution_threshold mismatch: {prev_threshold} != {args.resolution_threshold}"
                )
            if prev_min_trades != int(args.min_trades):
                raise ValueError(f"resume job min_trades mismatch: {prev_min_trades} != {args.min_trades}")

            prev_min_usd_f = float(prev_min_usd) if prev_min_usd is not None else None
            arg_min_usd_f = float(args.min_usd_amount) if args.min_usd_amount is not None else None
            if prev_min_usd_f != arg_min_usd_f:
                raise ValueError(
                    "resume job min_usd_amount mismatch: "
                    f"{prev_min_usd_f} != {arg_min_usd_f}"
                )
            if prev_total_markets != int(total_markets):
                logging.warning(
                    "resume job total_markets mismatch: "
                    f"{prev_total_markets} != {total_markets}. "
                    "Proceeding with market-id based resume."
                )

            existing = pd.read_sql_query(
                "SELECT market_id, inference_status, closed_time_utc FROM inferred_market_resolutions WHERE job_id = ?",
                conn,
                params=(job_id,),
            )
            existing_ids = set(existing["market_id"].astype(int).tolist()) if not existing.empty else set()

            counts = existing["inference_status"].value_counts().to_dict() if not existing.empty else {}
            resolved = int(counts.get("resolved", 0))
            unresolved = int(counts.get("unresolved", 0))
            too_few = int(counts.get("too_few_trades", 0))
            no_trades = int(counts.get("no_trades", 0))
            errors = int(counts.get("error", 0))

            closed_times = [x for x in existing.get("closed_time_utc", pd.Series(dtype=str)).dropna().tolist()]

            current_market_ids = {int(m[0]) for m in markets}
            existing_not_in_current = len(existing_ids - current_market_ids)
            if existing_not_in_current > 0:
                logging.warning(
                    f"Resume note: {existing_not_in_current:,} already-processed market ids "
                    "are not present in current metadata snapshot."
                )

            markets = [m for m in markets if int(m[0]) not in existing_ids]
            total_remaining = len(markets)
            logging.info(
                f"Resuming job_id={job_id}: already processed={len(existing_ids):,}, remaining={total_remaining:,}"
            )
        else:
            created_at_utc = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO inference_jobs (
                    created_at_utc,
                    data_dir,
                    resolution_threshold,
                    min_trades,
                    min_usd_amount,
                    total_markets,
                    resolved_markets,
                    unresolved_markets,
                    too_few_trades_markets,
                    no_trades_markets,
                    error_markets,
                    min_closed_time_utc,
                    max_closed_time_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, NULL, NULL)
                """,
                (
                    created_at_utc,
                    args.data_dir,
                    float(args.resolution_threshold),
                    int(args.min_trades),
                    args.min_usd_amount,
                    int(total_markets),
                ),
            )
            job_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

            resolved = 0
            unresolved = 0
            too_few = 0
            no_trades = 0
            errors = 0
            closed_times = []
        batch: List[Tuple] = []

        iterable = markets
        if _TQDM_AVAILABLE:
            iterable = tqdm(markets, desc="Inferring resolutions", unit="market")

        for idx, (market_id_raw, market_slug, closed_time_raw, volume_raw) in enumerate(iterable, start=1):
            market_id = int(market_id_raw)
            closed_time_utc = _as_iso_utc(closed_time_raw)
            if closed_time_utc is not None:
                closed_times.append(closed_time_utc)
            updated_at_utc = datetime.now(timezone.utc).isoformat()

            n_trades = 0
            inferred = None
            status = "unresolved"
            latest_ts_ms = None

            try:
                try:
                    trades = loader.get_trades_for_market(
                        market_id=market_id,
                        min_usd_amount=None,
                        use_cache=False,
                    )
                except TypeError:
                    trades = loader.get_trades_for_market(market_id)

                n_trades = len(trades)

                if n_trades == 0:
                    status = "no_trades"
                    no_trades += 1
                else:
                    latest_ts_ms = int(max(t.timestamp_ms for t in trades))
                    inferred = infer_market_winning_outcome_from_last_prices(
                        trades=trades,
                        threshold=float(args.resolution_threshold),
                    )
                    if inferred is None:
                        status = "unresolved"
                        unresolved += 1
                    else:
                        status = "resolved"
                        inferred = int(inferred)
                        resolved += 1

            except Exception:
                status = "error"
                errors += 1

            batch.append(
                (
                    job_id,
                    market_id,
                    str(market_slug) if market_slug is not None else None,
                    closed_time_utc,
                    float(volume_raw) if volume_raw is not None else None,
                    int(n_trades),
                    inferred,
                    status,
                    latest_ts_ms,
                    updated_at_utc,
                )
            )

            if len(batch) >= int(args.commit_every):
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO inferred_market_resolutions (
                        job_id,
                        market_id,
                        market_slug,
                        closed_time_utc,
                        volume,
                        n_trades,
                        inferred_winning_outcome,
                        inference_status,
                        latest_trade_ts_ms,
                        updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO market_resolution_cache (
                        market_id,
                        market_slug,
                        closed_time_utc,
                        volume,
                        n_trades,
                        inferred_winning_outcome,
                        inference_status,
                        latest_trade_ts_ms,
                        resolution_threshold,
                        updated_at_utc,
                        source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row[1],  # market_id
                            row[2],  # market_slug
                            row[3],  # closed_time_utc
                            row[4],  # volume
                            row[5],  # n_trades
                            row[6],  # inferred_winning_outcome
                            row[7],  # inference_status
                            row[8],  # latest_trade_ts_ms
                            float(args.resolution_threshold),
                            row[9],  # updated_at_utc
                            "precompute",
                        )
                        for row in batch
                    ],
                )
                conn.commit()
                batch = []

            if (not _TQDM_AVAILABLE) and (idx % 250 == 0 or idx == total_markets):
                logging.info(
                    f"Progress: {idx:,}/{total_markets:,} | resolved={resolved:,} "
                    f"unresolved={unresolved:,} too_few={too_few:,} "
                    f"no_trades={no_trades:,} errors={errors:,}"
                )

        if batch:
            conn.executemany(
                """
                INSERT OR REPLACE INTO inferred_market_resolutions (
                    job_id,
                    market_id,
                    market_slug,
                    closed_time_utc,
                    volume,
                    n_trades,
                    inferred_winning_outcome,
                    inference_status,
                    latest_trade_ts_ms,
                    updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO market_resolution_cache (
                    market_id,
                    market_slug,
                    closed_time_utc,
                    volume,
                    n_trades,
                    inferred_winning_outcome,
                    inference_status,
                    latest_trade_ts_ms,
                    resolution_threshold,
                    updated_at_utc,
                    source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        float(args.resolution_threshold),
                        row[9],
                        "precompute",
                    )
                    for row in batch
                ],
            )
            conn.commit()

        min_closed = min(closed_times) if closed_times else None
        max_closed = max(closed_times) if closed_times else None

        conn.execute(
            """
            UPDATE inference_jobs
            SET resolved_markets = ?,
                unresolved_markets = ?,
                too_few_trades_markets = ?,
                no_trades_markets = ?,
                error_markets = ?,
                min_closed_time_utc = ?,
                max_closed_time_utc = ?
            WHERE job_id = ?
            """,
            (
                int(resolved),
                int(unresolved),
                int(too_few),
                int(no_trades),
                int(errors),
                min_closed,
                max_closed,
                int(job_id),
            ),
        )
        conn.commit()

        logging.info(
            "Inference complete: "
            f"job_id={job_id}, resolved={resolved:,}, unresolved={unresolved:,}, "
            f"too_few={too_few:,}, no_trades={no_trades:,}, errors={errors:,}"
        )
        logging.info(f"Saved SQLite cache: {args.output_db}")

        if args.output_csv:
            query = """
                SELECT
                    r.market_id,
                    r.market_slug,
                    r.closed_time_utc,
                    r.volume,
                    r.n_trades,
                    r.inferred_winning_outcome,
                    r.inference_status,
                    r.latest_trade_ts_ms
                FROM inferred_market_resolutions r
                WHERE r.job_id = ?
                ORDER BY r.market_id
            """
            df = pd.read_sql_query(query, conn, params=(job_id,))
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False)
            logging.info(f"Saved CSV export: {args.output_csv}")

    finally:
        conn.close()
        loader.close()


if __name__ == "__main__":
    main()
