#!/usr/bin/env python3
"""Count unique wallets across notional cutoffs from trades.db."""

from __future__ import annotations

import argparse
import sqlite3
import time
from typing import List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count unique wallets in trades.db overall and under filters "
            "(date range + min trade size)."
        )
    )
    parser.add_argument("--trades-db", default="data/trades.db", help="Path to trades.db")
    parser.add_argument(
        "--wallet-source",
        choices=["maker", "both"],
        default="maker",
        help="Wallet source columns from trades table",
    )
    parser.add_argument(
        "--min-notional",
        type=float,
        default=None,
        help="Optional single-threshold summary in addition to cutoff grid",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Lower bound for trade timestamp (inclusive), e.g. 2025-01-01 or ISO timestamp",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Upper bound for trade timestamp (inclusive), e.g. 2025-12-31 or ISO timestamp",
    )
    parser.add_argument(
        "--cutoff-start",
        type=int,
        default=0,
        help="Start of cutoff grid (inclusive)",
    )
    parser.add_argument(
        "--cutoff-end",
        type=int,
        default=1000,
        help="End of cutoff grid (inclusive)",
    )
    parser.add_argument(
        "--cutoff-step",
        type=int,
        default=100,
        help="Cutoff step size",
    )
    return parser.parse_args()


def wallet_union_sql(wallet_source: str) -> str:
    if wallet_source == "both":
        return """
            SELECT LOWER(maker) AS wallet, timestamp, usd_amount
            FROM trades
            WHERE maker IS NOT NULL AND TRIM(maker) != ''
            UNION ALL
            SELECT LOWER(taker) AS wallet, timestamp, usd_amount
            FROM trades
            WHERE taker IS NOT NULL AND TRIM(taker) != ''
        """

    return """
        SELECT LOWER(maker) AS wallet, timestamp, usd_amount
        FROM trades
        WHERE maker IS NOT NULL AND TRIM(maker) != ''
    """


def build_filtered_predicates(date_from: str | None, date_to: str | None) -> Tuple[str, List[object]]:
    clauses: List[str] = []
    params: List[object] = []

    if date_from:
        clauses.append("timestamp >= ?")
        params.append(date_from)

    if date_to:
        clauses.append("timestamp <= ?")
        params.append(date_to)

    if not clauses:
        return "", params

    return "WHERE " + " AND ".join(clauses), params


def build_cutoffs(start: int, end: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("--cutoff-step must be > 0")
    if start > end:
        raise ValueError("--cutoff-start must be <= --cutoff-end")
    return list(range(start, end + 1, step))


def count_wallets_by_cutoffs(
    conn: sqlite3.Connection,
    wallet_source: str,
    where_sql: str,
    where_params: List[object],
    cutoffs: Sequence[float],
) -> Tuple[int, List[Tuple[int, int]]]:
    if not cutoffs:
        return 0, []

    agg_columns = [
        f"SUM(CASE WHEN max_usd_amount >= ? THEN 1 ELSE 0 END) AS c{i}"
        for i, _ in enumerate(cutoffs)
    ]

    sql = f"""
        WITH wallet_trades AS (
            {wallet_union_sql(wallet_source)}
        ),
        filtered_wallet_trades AS (
            SELECT wallet, usd_amount
            FROM wallet_trades
            {where_sql}
        ),
        wallet_max AS (
            SELECT wallet, MAX(COALESCE(usd_amount, 0)) AS max_usd_amount
            FROM filtered_wallet_trades
            GROUP BY wallet
        )
        SELECT
            COUNT(*) AS total_wallets,
            {', '.join(agg_columns)}
        FROM wallet_max
    """

    params: List[object] = []
    params.extend(where_params)
    params.extend(float(c) for c in cutoffs)

    row = conn.execute(sql, tuple(params)).fetchone()
    if row is None:
        return 0, [(c, 0) for c in cutoffs]

    total_wallets = int(row[0] or 0)
    counts: List[Tuple[int, int]] = []
    for idx, cutoff in enumerate(cutoffs):
        counts.append((cutoff, int(row[idx + 1] or 0)))
    return total_wallets, counts


def main() -> None:
    args = parse_args()

    if args.min_notional is not None and args.min_notional < 0:
        raise SystemExit("--min-notional must be >= 0")

    if args.date_from and args.date_to and args.date_from > args.date_to:
        raise SystemExit("--date-from must be <= --date-to")

    cutoffs = build_cutoffs(args.cutoff_start, args.cutoff_end, args.cutoff_step)

    t0 = time.time()
    conn = sqlite3.connect(args.trades_db)
    try:
        filtered_where, filtered_params = build_filtered_predicates(
            date_from=args.date_from,
            date_to=args.date_to,
        )

        total_wallets, cutoff_counts = count_wallets_by_cutoffs(
            conn,
            args.wallet_source,
            filtered_where,
            filtered_params,
            cutoffs,
        )
    finally:
        conn.close()

    elapsed = time.time() - t0

    print(f"Trades DB: {args.trades_db}")
    print(f"Wallet source: {args.wallet_source}")
    print(f"Total unique wallets (after date filters): {total_wallets:,}")
    print("Filters:")
    print(f"  date_from: {args.date_from or 'None'}")
    print(f"  date_to: {args.date_to or 'None'}")
    print("\nWallet counts by max single-trade notional cutoff:")
    print("  cutoff_usd   wallets       coverage")
    print("  ----------   ----------    --------")
    for cutoff, count in cutoff_counts:
        pct = (100.0 * count / total_wallets) if total_wallets > 0 else 0.0
        print(f"  {cutoff:>10,.0f}   {count:>10,}    {pct:>7.2f}%")

    if args.min_notional is not None:
        min_cutoff = float(args.min_notional)
        # derive from grid if exact hit exists, otherwise run single extra query
        matched = next((count for cutoff, count in cutoff_counts if float(cutoff) == min_cutoff), None)
        if matched is None:
            conn2 = sqlite3.connect(args.trades_db)
            try:
                _, single_counts = count_wallets_by_cutoffs(
                    conn2,
                    args.wallet_source,
                    filtered_where,
                    filtered_params,
                    [min_cutoff],
                )
            finally:
                conn2.close()
            matched = single_counts[0][1]
        pct = (100.0 * matched / total_wallets) if total_wallets > 0 else 0.0
        print(f"\nRequested cutoff >= {min_cutoff:,.2f}: {matched:,} wallets ({pct:.2f}%)")

    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
