#!/usr/bin/env python3
"""
Query wallet-level insider trading classifications from the live DB.

Usage:
    python query_wallets.py suspects                          # all suspects across markets
    python query_wallets.py suspects --market fed-decision    # one market (substring match)
    python query_wallets.py suspects --threshold 0.15         # custom threshold
    python query_wallets.py wallet 0xABC123                   # full profile for one wallet
    python query_wallets.py market fed-decision               # all wallets in a market
    python query_wallets.py summary                           # overview stats
    python query_wallets.py evaluate fed-decision 0           # PnL eval if outcome 0 wins
"""

import argparse
import json
import sqlite3
import sys
from typing import Optional


DEFAULT_DB = "data/insider_trading_v1.db"
CLUSTERING_DB = "data/clustering_v1.db"
DEFAULT_THRESHOLD = 0.25
MIN_TRADES = 3


def get_connections(db_path: str, clustering_db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Attach clustering DB for boost lookups
    conn.execute(f"ATTACH DATABASE ? AS clustering", (clustering_db_path,))
    return conn


def cmd_suspects(conn, market_slug: Optional[str], threshold: float):
    """List all wallets classified as suspected insiders."""

    market_filter = ""
    params = [MIN_TRADES, threshold]

    if market_slug:
        market_filter = "AND w.market_slug LIKE ?"
        params.append(f"%{market_slug}%")

    rows = conn.execute(f"""
        SELECT
            w.wallet,
            w.condition_id,
            w.market_slug,
            w.trade_count,
            w.flag_count,
            CAST(w.flag_count AS REAL) / w.trade_count AS flag_rate,
            COALESCE(cm.density, 0.0) AS cluster_density,
            COALESCE(cm.size, 0) AS cluster_size,
            COALESCE(cm.has_common_ownership, 0) AS common_ownership,
            ca.cluster_id,
            w.total_buy_notional,
            w.total_sell_notional,
            w.first_trade_ts,
            w.last_trade_ts,
            w.last_flag_ts
        FROM wallet_market_stats w
        LEFT JOIN clustering.cluster_assignments ca ON w.wallet = ca.wallet
        LEFT JOIN clustering.cluster_metadata cm ON ca.cluster_id = cm.cluster_id
        WHERE w.trade_count >= ?
          AND (CAST(w.flag_count AS REAL) / w.trade_count) >= ?
          {market_filter}
        ORDER BY flag_rate DESC, w.flag_count DESC
    """, params).fetchall()

    if not rows:
        print("No suspects found.")
        return

    print(f"\n{'SUSPECTED INSIDERS':=^90}")
    print(f"Threshold: flag_rate >= {threshold:.2f}, min_trades >= {MIN_TRADES}")
    print(f"{'─' * 90}")
    print(f"{'Wallet':<15} {'Market':<30} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'Cluster':>8} {'Buy$':>10}")
    print(f"{'─' * 90}")

    for r in rows:
        wallet_short = r["wallet"][:12] + "..."
        slug = (r["market_slug"] or "")[:28]
        cluster_info = f"C{r['cluster_id']}" if r["cluster_id"] is not None else "—"
        if r["common_ownership"]:
            cluster_info += "★"

        print(
            f"{wallet_short:<15} {slug:<30} {r['trade_count']:>6} "
            f"{r['flag_count']:>5} {r['flag_rate']:>5.1%} {cluster_info:>8} "
            f"${r['total_buy_notional']:>9,.0f}"
        )

    print(f"{'─' * 90}")
    print(f"Total: {len(rows)} suspect wallet×market pairs\n")


def cmd_wallet(conn, wallet_prefix: str):
    """Show full profile for a wallet (prefix match)."""

    rows = conn.execute("""
        SELECT
            w.wallet,
            w.condition_id,
            w.market_slug,
            w.trade_count,
            w.flag_count,
            CAST(w.flag_count AS REAL) / w.trade_count AS flag_rate,
            w.total_buy_notional,
            w.total_sell_notional,
            w.first_trade_ts,
            w.last_trade_ts,
            ca.cluster_id
        FROM wallet_market_stats w
        LEFT JOIN clustering.cluster_assignments ca ON w.wallet = ca.wallet
        WHERE w.wallet LIKE ?
        ORDER BY w.flag_count DESC, w.trade_count DESC
    """, (f"{wallet_prefix}%",)).fetchall()

    if not rows:
        print(f"No data for wallet starting with '{wallet_prefix}'")
        return

    wallet_full = rows[0]["wallet"]
    cluster_id = rows[0]["cluster_id"]

    print(f"\n{'WALLET PROFILE':=^70}")
    print(f"Address:  {wallet_full}")
    print(f"Cluster:  {cluster_id if cluster_id is not None else 'None'}")
    print(f"Markets:  {len(rows)}")
    print(f"{'─' * 70}")

    total_flags = sum(r["flag_count"] for r in rows)
    total_trades = sum(r["trade_count"] for r in rows)
    total_buy = sum(r["total_buy_notional"] for r in rows)

    print(f"Global:   {total_trades} trades, {total_flags} flags, ${total_buy:,.0f} bought")
    print(f"{'─' * 70}")
    print(f"{'Market':<35} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'Buy$':>10}")
    print(f"{'─' * 70}")

    for r in rows:
        slug = (r["market_slug"] or "")[:33]
        print(
            f"{slug:<35} {r['trade_count']:>6} {r['flag_count']:>5} "
            f"{r['flag_rate']:>5.1%} ${r['total_buy_notional']:>9,.0f}"
        )

    # Show positions
    positions = conn.execute("""
        SELECT condition_id, outcome_index, net_shares,
               buy_notional, sell_notional, buy_shares, sell_shares
        FROM wallet_outcome_positions
        WHERE wallet = ?
        ORDER BY condition_id, outcome_index
    """, (wallet_full,)).fetchall()

    if positions:
        print(f"\n{'POSITIONS':─^70}")
        print(f"{'Market':<35} {'Outcome':>7} {'Shares':>10} {'Buy$':>10} {'Sell$':>10}")
        print(f"{'─' * 70}")

        for p in positions:
            cid_short = p["condition_id"][:33]
            print(
                f"{cid_short:<35} {p['outcome_index']:>7} "
                f"{p['net_shares']:>10,.1f} ${p['buy_notional']:>9,.0f} "
                f"${p['sell_notional']:>9,.0f}"
            )

    print()


def cmd_market(conn, market_slug: str):
    """Show all wallets in a market, sorted by flag rate."""

    rows = conn.execute("""
        SELECT
            w.wallet,
            w.trade_count,
            w.flag_count,
            CAST(w.flag_count AS REAL) / w.trade_count AS flag_rate,
            w.total_buy_notional,
            w.total_sell_notional,
            ca.cluster_id
        FROM wallet_market_stats w
        LEFT JOIN clustering.cluster_assignments ca ON w.wallet = ca.wallet
        WHERE w.market_slug LIKE ?
        ORDER BY flag_rate DESC, w.total_buy_notional DESC
    """, (f"%{market_slug}%",)).fetchall()

    if not rows:
        print(f"No data for market matching '{market_slug}'")
        return

    flagged = [r for r in rows if r["flag_count"] > 0]
    total_buy = sum(r["total_buy_notional"] for r in rows)

    print(f"\n{'MARKET WALLETS':=^80}")
    print(f"Slug filter: {market_slug}")
    print(f"Wallets: {len(rows)} total, {len(flagged)} flagged")
    print(f"Total buy volume: ${total_buy:,.0f}")
    print(f"{'─' * 80}")
    print(f"{'Wallet':<15} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'Cluster':>8} {'Buy$':>10} {'Sell$':>10}")
    print(f"{'─' * 80}")

    for r in rows[:50]:  # Top 50
        wallet_short = r["wallet"][:12] + "..."
        cluster = f"C{r['cluster_id']}" if r["cluster_id"] is not None else "—"
        print(
            f"{wallet_short:<15} {r['trade_count']:>6} {r['flag_count']:>5} "
            f"{r['flag_rate']:>5.1%} {cluster:>8} "
            f"${r['total_buy_notional']:>9,.0f} ${r['total_sell_notional']:>9,.0f}"
        )

    if len(rows) > 50:
        print(f"  ... and {len(rows) - 50} more wallets")
    print()


def cmd_summary(conn):
    """High-level system stats."""

    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT wallet) AS unique_wallets,
            COUNT(*) AS wallet_market_pairs,
            SUM(trade_count) AS total_trades,
            SUM(flag_count) AS total_flags,
            COUNT(DISTINCT condition_id) AS markets
        FROM wallet_market_stats
    """).fetchone()

    flagged_wallets = conn.execute("""
        SELECT COUNT(DISTINCT wallet) FROM wallet_market_stats WHERE flag_count > 0
    """).fetchone()[0]

    suspect_pairs = conn.execute("""
        SELECT COUNT(*) FROM wallet_market_stats
        WHERE trade_count >= ? AND CAST(flag_count AS REAL) / trade_count >= ?
    """, (MIN_TRADES, DEFAULT_THRESHOLD)).fetchone()[0]

    print(f"\n{'SYSTEM SUMMARY':=^60}")
    print(f"Markets monitored:    {stats['markets']}")
    print(f"Unique wallets:       {stats['unique_wallets']:,}")
    print(f"Wallet×market pairs:  {stats['wallet_market_pairs']:,}")
    print(f"Total trades tracked: {stats['total_trades']:,}")
    print(f"Total flags:          {stats['total_flags']:,}")
    print(f"Wallets with flags:   {flagged_wallets:,}")
    print(f"Suspect pairs (rate>={DEFAULT_THRESHOLD:.0%}, trades>={MIN_TRADES}): {suspect_pairs}")
    print()


def cmd_evaluate(conn, market_slug: str, winning_outcome: int):
    """
    Post-resolution evaluation: compute PnL for all wallets in a market
    given which outcome won. Compares flagged vs unflagged performance.
    """

    # Get all wallets with positions in this market
    rows = conn.execute("""
        SELECT
            w.wallet,
            w.trade_count,
            w.flag_count,
            CAST(w.flag_count AS REAL) / w.trade_count AS flag_rate,
            w.total_buy_notional,
            w.total_sell_notional
        FROM wallet_market_stats w
        WHERE w.market_slug LIKE ?
          AND w.trade_count >= ?
    """, (f"%{market_slug}%", MIN_TRADES)).fetchall()

    if not rows:
        print(f"No data for market '{market_slug}'")
        return

    results = []
    for r in rows:
        # Get positions per outcome
        positions = conn.execute("""
            SELECT outcome_index, net_shares, buy_notional, sell_notional
            FROM wallet_outcome_positions
            WHERE wallet = ? AND condition_id IN (
                SELECT DISTINCT condition_id FROM wallet_market_stats
                WHERE market_slug LIKE ?
            )
        """, (r["wallet"], f"%{market_slug}%")).fetchall()

        total_cost = 0.0
        total_payout = 0.0
        for p in positions:
            total_cost += p["buy_notional"] - p["sell_notional"]
            if p["outcome_index"] == winning_outcome:
                total_payout = p["net_shares"]  # shares of winning outcome = payout

        net_pnl = total_payout - total_cost
        ret = (net_pnl / r["total_buy_notional"]) if r["total_buy_notional"] > 0 else 0.0

        results.append({
            "wallet": r["wallet"],
            "trade_count": r["trade_count"],
            "flag_count": r["flag_count"],
            "flag_rate": r["flag_rate"],
            "is_suspect": r["flag_rate"] >= DEFAULT_THRESHOLD,
            "total_cost": total_cost,
            "payout": total_payout,
            "net_pnl": net_pnl,
            "return": ret,
            "buy_notional": r["total_buy_notional"],
        })

    results.sort(key=lambda x: x["net_pnl"], reverse=True)
    suspects = [r for r in results if r["is_suspect"]]
    non_suspects = [r for r in results if not r["is_suspect"]]

    def _median(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    suspect_pnls = [r["net_pnl"] for r in suspects]
    suspect_returns = [r["return"] for r in suspects]
    other_pnls = [r["net_pnl"] for r in non_suspects]
    other_returns = [r["return"] for r in non_suspects]

    print(f"\n{'POST-RESOLUTION EVALUATION':=^80}")
    print(f"Market: {market_slug}  |  Winning outcome: {winning_outcome}")
    print(f"Wallets evaluated: {len(results)}  |  Suspects: {len(suspects)}")
    print(f"{'─' * 80}")
    print(f"{'Group':<12} {'Count':>6} {'Med PnL':>12} {'Med Return':>12} {'Total PnL':>12}")
    print(f"{'─' * 80}")
    print(
        f"{'Suspects':<12} {len(suspects):>6} "
        f"${_median(suspect_pnls):>11,.0f} {_median(suspect_returns):>11.1%} "
        f"${sum(suspect_pnls):>11,.0f}"
    )
    print(
        f"{'Others':<12} {len(non_suspects):>6} "
        f"${_median(other_pnls):>11,.0f} {_median(other_returns):>11.1%} "
        f"${sum(other_pnls):>11,.0f}"
    )
    print(f"{'─' * 80}")

    # Top 20 by PnL
    print(f"\n{'TOP 20 BY PNL':─^80}")
    print(f"{'Wallet':<15} {'Suspect':>7} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'PnL':>12} {'Return':>8}")
    print(f"{'─' * 80}")

    for r in results[:20]:
        tag = "  ★ YES" if r["is_suspect"] else "     no"
        print(
            f"{r['wallet'][:12] + '...':<15} {tag:>7} {r['trade_count']:>6} "
            f"{r['flag_count']:>5} {r['flag_rate']:>5.1%} "
            f"${r['net_pnl']:>11,.0f} {r['return']:>7.0%}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(description="Query wallet insider trading classifications")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to main DB")
    parser.add_argument("--clustering-db", default=CLUSTERING_DB, help="Path to clustering DB")

    sub = parser.add_subparsers(dest="command")

    p_suspects = sub.add_parser("suspects", help="List suspected insiders")
    p_suspects.add_argument("--market", default=None, help="Filter by market slug (substring)")
    p_suspects.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    p_wallet = sub.add_parser("wallet", help="Full profile for a wallet")
    p_wallet.add_argument("address", help="Wallet address (prefix match)")

    p_market = sub.add_parser("market", help="All wallets in a market")
    p_market.add_argument("slug", help="Market slug (substring match)")

    sub.add_parser("summary", help="System overview stats")

    p_eval = sub.add_parser("evaluate", help="Post-resolution PnL evaluation")
    p_eval.add_argument("slug", help="Market slug")
    p_eval.add_argument("winning_outcome", type=int, help="Winning outcome index (0 or 1)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    conn = get_connections(args.db, args.clustering_db)

    try:
        if args.command == "suspects":
            cmd_suspects(conn, args.market, args.threshold)
        elif args.command == "wallet":
            cmd_wallet(conn, args.address)
        elif args.command == "market":
            cmd_market(conn, args.slug)
        elif args.command == "summary":
            cmd_summary(conn)
        elif args.command == "evaluate":
            cmd_evaluate(conn, args.slug, args.winning_outcome)
    finally:
        conn.close()


if __name__ == "__main__":
    main()