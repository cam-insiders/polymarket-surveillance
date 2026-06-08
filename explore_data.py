#!/usr/bin/env python3
"""
Polymarket Insider Trading Detection — Data Explorer v2

Command-based REPL for querying alerts, clusters, wallet classifications,
and post-resolution evaluation.

Usage:
    python explore_data.py [alert_db] [clustering_db]

    Defaults:
        alert_db     = data/insider_trading_v1.db
        clustering_db = data/clustering_v1.db

Commands (type 'help' for full list):
    suspects          — wallets classified as suspected insiders
    wallet <addr>     — deep-dive on a single wallet
    market <slug>     — all wallets in a market
    evaluate <slug> <outcome>  — post-resolution PnL analysis
    clusters          — list detected clusters
    cluster <id>      — inspect a cluster
    summary           — system overview
"""

import cmd
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

class C:
    """ANSI colour shortcuts."""
    R  = '\033[91m'   # red
    G  = '\033[92m'   # green
    Y  = '\033[93m'   # yellow
    B  = '\033[94m'   # blue
    M  = '\033[95m'   # magenta
    CY = '\033[96m'   # cyan
    W  = '\033[97m'   # white
    BD = '\033[1m'    # bold
    UL = '\033[4m'    # underline
    E  = '\033[0m'    # end / reset


def _ts(ms: Optional[int]) -> str:
    """Format a millisecond timestamp, or '—' if None/0."""
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d %H:%M')


def _ts_s(epoch: Optional[int]) -> str:
    """Format a second-precision timestamp."""
    if not epoch:
        return "—"
    return datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M')


def _median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _pct(n: float) -> str:
    """Format as percentage string."""
    return f"{n:.1%}"


def _ruler(width: int = 90):
    print("─" * width)

class DB:
    """Thin wrapper holding both DB connections."""

    def __init__(self, alert_path: str, clustering_path: str):
        self.conn = sqlite3.connect(alert_path)
        self.conn.row_factory = sqlite3.Row
        # Attach clustering DB so we can join in single queries
        self.conn.execute("ATTACH DATABASE ? AS cl", (clustering_path,))
        print(f"{C.G}Connected{C.E}  alerts={alert_path}  clustering={clustering_path}\n")

    def q(self, sql: str, params=()) -> List[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def q1(self, sql: str, params=()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def close(self):
        self.conn.close()

class Explorer(cmd.Cmd):
    intro = (
        f"\n{C.BD}{C.CY}{'═' * 70}\n"
        f"  Polymarket Insider Trading — Data Explorer v2\n"
        f"{'═' * 70}{C.E}\n"
        f"  Type {C.BD}help{C.E} for commands, {C.BD}quit{C.E} to exit.\n"
    )
    prompt = f"{C.BD}explore>{C.E} "

    def __init__(self, db: DB):
        super().__init__()
        self.db = db

    def _parse(self, line: str) -> List[str]:
        return line.strip().split()

    def _resolve_wallet(self, prefix: str) -> Optional[str]:
        """Resolve a wallet prefix to a full address."""
        # Try wallet_market_stats first (has all wallets), fall back to alerts
        row = self.db.q1(
            "SELECT wallet FROM wallet_market_stats WHERE wallet LIKE ? LIMIT 1",
            (f"{prefix}%",),
        )
        if row:
            return row["wallet"]
        row = self.db.q1(
            "SELECT wallet FROM alerts WHERE wallet LIKE ? LIMIT 1",
            (f"{prefix}%",),
        )
        return row["wallet"] if row else None

    def do_summary(self, _line):
        """System overview: wallets tracked, alerts, clusters."""
        wms = self.db.q1("""
            SELECT
                COUNT(DISTINCT wallet) AS wallets,
                COUNT(*)               AS pairs,
                SUM(trade_count)       AS trades,
                SUM(flag_count)        AS flags,
                COUNT(DISTINCT condition_id) AS markets
            FROM wallet_market_stats
        """)

        alert_stats = self.db.q1("""
            SELECT COUNT(*) AS total, COUNT(DISTINCT wallet) AS wallets,
                   SUM(notional_usdc) AS volume
            FROM alerts
        """)

        cl = self.db.q1("""
            SELECT COUNT(*) AS clusters,
                   SUM(size) AS clustered_wallets,
                   SUM(has_common_ownership) AS owned
            FROM cl.cluster_metadata
        """)

        print(f"\n{C.BD}{C.CY}{'═' * 60}")
        print(f"  SYSTEM SUMMARY")
        print(f"{'═' * 60}{C.E}")

        if wms and wms["wallets"]:
            print(f"\n  Markets monitored      {wms['markets']}")
            print(f"  Unique wallets         {wms['wallets']:,}")
            print(f"  Wallet×market pairs    {wms['pairs']:,}")
            print(f"  Trades tracked         {wms['trades']:,}")
            print(f"  Flags (alert trades)   {wms['flags']:,}")
        else:
            print(f"\n  {C.Y}No wallet_market_stats data yet (system just started?){C.E}")

        if alert_stats and alert_stats["total"]:
            print(f"\n  Alerts in DB           {alert_stats['total']:,}")
            print(f"  Wallets with alerts    {alert_stats['wallets']:,}")
            print(f"  Total flagged volume   ${alert_stats['volume']:,.0f}")

        if cl:
            print(f"\n  Clusters               {cl['clusters']}")
            print(f"  Clustered wallets      {cl['clustered_wallets'] or 0}")
            print(f"  Common ownership       {cl['owned'] or 0}")

        # Detector breakdown from alerts
        rows = self.db.q("SELECT signals FROM alerts")
        if rows:
            counts = defaultdict(int)
            for r in rows:
                for part in (r["signals"] or "").split(" | "):
                    name = part.split("(")[0].strip()
                    if name:
                        counts[name] += 1
            print(f"\n  {C.BD}Detector breakdown:{C.E}")
            for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"    {name:<35} {cnt:>6}")

        print()

    def do_suspects(self, line):
        """
        List wallets classified as suspected insiders.

        Usage:  suspects [--market <slug>] [--threshold <0.25>] [--min-trades <3>]
        """
        args = self._parse(line)
        market_filter = None
        threshold = 0.25
        min_trades = 3

        i = 0
        while i < len(args):
            if args[i] == "--market" and i + 1 < len(args):
                market_filter = args[i + 1]; i += 2
            elif args[i] == "--threshold" and i + 1 < len(args):
                threshold = float(args[i + 1]); i += 2
            elif args[i] == "--min-trades" and i + 1 < len(args):
                min_trades = int(args[i + 1]); i += 2
            else:
                i += 1

        where = "WHERE w.trade_count >= ? AND CAST(w.flag_count AS REAL) / w.trade_count >= ? AND w.total_buy_notional > 0"
        params: list = [min_trades, threshold]

        if market_filter:
            where += " AND w.market_slug LIKE ?"
            params.append(f"%{market_filter}%")

        rows = self.db.q(f"""
            SELECT
                w.wallet, w.condition_id, w.market_slug,
                w.trade_count, w.flag_count,
                CAST(w.flag_count AS REAL) / w.trade_count AS flag_rate,
                w.total_buy_notional,
                ca.cluster_id,
                COALESCE(cm.has_common_ownership, 0) AS common_ownership
            FROM wallet_market_stats w
            LEFT JOIN cl.cluster_assignments ca ON w.wallet = ca.wallet
            LEFT JOIN cl.cluster_metadata cm ON ca.cluster_id = cm.cluster_id
            {where}
            ORDER BY flag_rate DESC, w.flag_count DESC
        """, params)

        if not rows:
            print(f"  {C.Y}No suspects (threshold={_pct(threshold)}, min_trades={min_trades}){C.E}\n")
            return

        print(f"\n{C.BD}  SUSPECTED INSIDERS  (rate≥{_pct(threshold)}, trades≥{min_trades}){C.E}")
        _ruler()
        print(f"  {'Wallet':<44} {'Market':<32} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'Clust':>6} {'Buy$':>10}")
        _ruler()

        for r in rows:
            w = r["wallet"]
            slug = (r["market_slug"] or "")[:30]
            cl_str = f"C{r['cluster_id']}" if r["cluster_id"] is not None else "  —"
            if r["common_ownership"]:
                cl_str += "★"

            color = C.R if r["flag_rate"] >= 0.5 else C.Y if r["flag_rate"] >= threshold else C.W
            print(
                f"  {color}{w:<44} {slug:<32} {r['trade_count']:>6} "
                f"{r['flag_count']:>5} {_pct(r['flag_rate']):>6} "
                f"{cl_str:>6} ${r['total_buy_notional']:>9,.0f}{C.E}"
            )

        _ruler()
        print(f"  {len(rows)} suspect wallet×market pairs\n")

    def do_wallet(self, line):
        """
        Deep-dive on a single wallet.

        Usage:  wallet <address_or_prefix>
        """
        prefix = line.strip()
        if not prefix:
            print(f"  {C.Y}Usage: wallet <address_or_prefix>{C.E}"); return

        wallet = self._resolve_wallet(prefix)
        if not wallet:
            print(f"  {C.R}No wallet found starting with '{prefix}'{C.E}"); return

        # ── header ──
        print(f"\n{C.BD}{C.CY}{'═' * 70}")
        print(f"  WALLET  {wallet}")
        print(f"{'═' * 70}{C.E}")

        # cluster
        cl_row = self.db.q1(
            "SELECT cluster_id FROM cl.cluster_assignments WHERE wallet = ?", (wallet,)
        )
        if cl_row and cl_row["cluster_id"] is not None:
            cid = cl_row["cluster_id"]
            cm = self.db.q1("SELECT size, density, has_common_ownership FROM cl.cluster_metadata WHERE cluster_id = ?", (cid,))
            own = f"  {C.G}★ common ownership{C.E}" if cm and cm["has_common_ownership"] else ""
            print(f"\n  Cluster {C.R}C{cid}{C.E}  (size={cm['size']}, density={cm['density']:.2f}){own}")
        else:
            print(f"\n  Cluster: none")

        # per-market stats
        mkt_rows = self.db.q("""
            SELECT condition_id, market_slug, trade_count, flag_count,
                   CAST(flag_count AS REAL) / trade_count AS flag_rate,
                   total_buy_notional, total_sell_notional,
                   first_trade_ts, last_trade_ts, last_flag_ts
            FROM wallet_market_stats
            WHERE wallet = ?
            ORDER BY flag_count DESC, trade_count DESC
        """, (wallet,))

        if mkt_rows:
            total_trades = sum(r["trade_count"] for r in mkt_rows)
            total_flags = sum(r["flag_count"] for r in mkt_rows)
            total_buy = sum(r["total_buy_notional"] for r in mkt_rows)

            print(f"\n  {C.BD}Across {len(mkt_rows)} markets:{C.E}  "
                  f"{total_trades} trades, {total_flags} flags, ${total_buy:,.0f} bought")

            print(f"\n  {'Market':<34} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'Buy$':>10} {'Sell$':>10}")
            _ruler()
            for r in mkt_rows:
                slug = (r["market_slug"] or "")[:32]
                color = C.R if r["flag_rate"] >= 0.5 else C.Y if r["flag_count"] > 0 else C.W
                print(
                    f"  {color}{slug:<34} {r['trade_count']:>6} {r['flag_count']:>5} "
                    f"{_pct(r['flag_rate']):>6} ${r['total_buy_notional']:>9,.0f} "
                    f"${r['total_sell_notional']:>9,.0f}{C.E}"
                )
        else:
            print(f"\n  {C.Y}No wallet_market_stats (wallet may only appear in alerts){C.E}")

        # positions
        pos_rows = self.db.q("""
            SELECT condition_id, outcome_index, net_shares,
                   buy_notional, sell_notional, buy_shares, sell_shares
            FROM wallet_outcome_positions WHERE wallet = ?
            ORDER BY condition_id, outcome_index
        """, (wallet,))

        if pos_rows:
            print(f"\n  {C.BD}Positions:{C.E}")
            print(f"  {'Condition':<34} {'Out':>3} {'Shares':>10} {'Buy$':>10} {'Sell$':>10}")
            _ruler()
            for p in pos_rows:
                cid_short = p["condition_id"][:32]
                color = C.G if p["net_shares"] > 0 else C.R if p["net_shares"] < 0 else C.W
                print(
                    f"  {cid_short:<34} {p['outcome_index']:>3} "
                    f"{color}{p['net_shares']:>10,.1f}{C.E} "
                    f"${p['buy_notional']:>9,.0f} ${p['sell_notional']:>9,.0f}"
                )

        # top alerts
        alert_rows = self.db.q("""
            SELECT market_slug, side, notional_usdc, total_score, signals, timestamp_ms
            FROM alerts WHERE wallet = ?
            ORDER BY total_score DESC LIMIT 5
        """, (wallet,))

        if alert_rows:
            print(f"\n  {C.BD}Top alerts:{C.E}")
            print(f"  {'Market':<34} {'Side':<5} {'Amount':>10} {'Score':>6} {'When'}")
            _ruler()
            for a in alert_rows:
                sc = C.R if a["total_score"] >= 0.7 else C.Y
                detectors = [d.split("(")[0] for d in (a["signals"] or "").split(" | ")]
                print(
                    f"  {(a['market_slug'] or '')[:32]:<34} {a['side']:<5} "
                    f"${a['notional_usdc']:>9,.0f} {sc}{a['total_score']:>5.2f}{C.E}  "
                    f"{_ts(a['timestamp_ms'])}"
                )
                print(f"    {C.CY}{', '.join(detectors)}{C.E}")

        # attribution transfers
        out_rows = self.db.q("""
            SELECT to_wallet, total_amount, tx_count FROM cl.attribution_edges
            WHERE from_wallet = ? ORDER BY total_amount DESC LIMIT 5
        """, (wallet,))
        in_rows = self.db.q("""
            SELECT from_wallet, total_amount, tx_count FROM cl.attribution_edges
            WHERE to_wallet = ? ORDER BY total_amount DESC LIMIT 5
        """, (wallet,))

        if out_rows or in_rows:
            print(f"\n  {C.BD}USDC transfers (top 5 each direction):{C.E}")
            for r in out_rows:
                print(f"    {C.R}→{C.E} {r['to_wallet'][:14]}…  ${r['total_amount']:>10,.0f}  ({r['tx_count']} txs)")
            for r in in_rows:
                print(f"    {C.G}←{C.E} {r['from_wallet'][:14]}…  ${r['total_amount']:>10,.0f}  ({r['tx_count']} txs)")

        print()

    def do_market(self, line):
        """
        Show all wallets in a market, sorted by flag rate.

        Usage:  market <slug_substring>
        """
        slug = line.strip()
        if not slug:
            print(f"  {C.Y}Usage: market <slug_substring>{C.E}"); return

        rows = self.db.q("""
            SELECT
                w.wallet, w.trade_count, w.flag_count,
                CAST(w.flag_count AS REAL) / w.trade_count AS flag_rate,
                w.total_buy_notional, w.total_sell_notional,
                ca.cluster_id
            FROM wallet_market_stats w
            LEFT JOIN cl.cluster_assignments ca ON w.wallet = ca.wallet
            WHERE w.market_slug LIKE ? AND w.total_buy_notional > 0
            ORDER BY flag_rate DESC, w.total_buy_notional DESC
        """, (f"%{slug}%",))

        if not rows:
            print(f"  {C.Y}No data for '{slug}'{C.E}\n"); return

        flagged = [r for r in rows if r["flag_count"] > 0]
        total_buy = sum(r["total_buy_notional"] for r in rows)

        print(f"\n{C.BD}  MARKET: *{slug}*{C.E}")
        print(f"  {len(rows)} wallets, {len(flagged)} flagged, ${total_buy:,.0f} total buy volume")
        _ruler()
        print(f"  {'Wallet':<44} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'Clust':>6} {'Buy$':>10} {'Sell$':>10}")
        _ruler()

        for r in rows[:40]:
            w = r["wallet"]
            cl_str = f"C{r['cluster_id']}" if r["cluster_id"] is not None else "  —"
            color = C.R if r["flag_rate"] >= 0.5 else C.Y if r["flag_count"] > 0 else C.W
            print(
                f"  {color}{w:<14} {r['trade_count']:>6} {r['flag_count']:>5} "
                f"{_pct(r['flag_rate']):>6} {cl_str:>6} "
                f"${r['total_buy_notional']:>9,.0f} ${r['total_sell_notional']:>9,.0f}{C.E}"
            )

        if len(rows) > 40:
            print(f"  … and {len(rows) - 40} more wallets")
        print()

    def do_evaluate(self, line):
        """
        Post-resolution PnL evaluation.

        Usage:  evaluate <slug_substring> <winning_outcome_index>

        Example: evaluate fed-decision 0
        """
        args = self._parse(line)
        if len(args) < 2:
            print(f"  {C.Y}Usage: evaluate <slug> <winning_outcome>{C.E}"); return

        slug, winning = args[0], int(args[1])
        threshold = 0.25
        min_trades = 3

        # Get wallets with stats in this market
        wallets = self.db.q("""
            SELECT wallet, trade_count, flag_count,
                   CAST(flag_count AS REAL) / trade_count AS flag_rate,
                   total_buy_notional, total_sell_notional
            FROM wallet_market_stats
            WHERE market_slug LIKE ? AND trade_count >= ? AND total_buy_notional > 0
        """, (f"%{slug}%", min_trades))

        if not wallets:
            print(f"  {C.Y}No data for '{slug}'{C.E}\n"); return

        # Get the condition_id(s) for this slug
        cid_rows = self.db.q(
            "SELECT DISTINCT condition_id FROM wallet_market_stats WHERE market_slug LIKE ?",
            (f"%{slug}%",)
        )
        cids = [r["condition_id"] for r in cid_rows]
        cid_placeholders = ",".join("?" * len(cids))

        results = []
        for w in wallets:
            positions = self.db.q(f"""
                SELECT outcome_index, net_shares, buy_notional, sell_notional
                FROM wallet_outcome_positions
                WHERE wallet = ? AND condition_id IN ({cid_placeholders})
            """, [w["wallet"]] + cids)

            total_cost = 0.0
            payout = 0.0
            for p in positions:
                total_cost += p["buy_notional"] - p["sell_notional"]
                if p["outcome_index"] == winning:
                    payout = p["net_shares"]

            net_pnl = payout - total_cost
            ret = (net_pnl / w["total_buy_notional"]) if w["total_buy_notional"] > 0 else 0.0

            results.append({
                "wallet": w["wallet"],
                "trades": w["trade_count"],
                "flags": w["flag_count"],
                "rate": w["flag_rate"],
                "suspect": w["flag_rate"] >= threshold,
                "cost": total_cost,
                "payout": payout,
                "pnl": net_pnl,
                "return": ret,
                "buy": w["total_buy_notional"],
            })

        results.sort(key=lambda x: x["pnl"], reverse=True)
        suspects = [r for r in results if r["suspect"]]
        others = [r for r in results if not r["suspect"]]

        s_pnls = [r["pnl"] for r in suspects]
        s_rets = [r["return"] for r in suspects]
        o_pnls = [r["pnl"] for r in others]
        o_rets = [r["return"] for r in others]

        print(f"\n{C.BD}{C.CY}{'═' * 80}")
        print(f"  POST-RESOLUTION EVALUATION")
        print(f"{'═' * 80}{C.E}")
        print(f"  Market: *{slug}*  |  Winning outcome: {winning}")
        print(f"  Wallets: {len(results)}  |  Suspects (rate≥{_pct(threshold)}): {len(suspects)}")
        _ruler(80)
        print(f"  {'Group':<12} {'Count':>6} {'Med PnL':>12} {'Med Return':>10} {'Total PnL':>12}")
        _ruler(80)
        print(
            f"  {C.R}{'Suspects':<12}{C.E} {len(suspects):>6} "
            f"${_median(s_pnls):>11,.0f} {_pct(_median(s_rets)):>10} "
            f"${sum(s_pnls):>11,.0f}"
        )
        print(
            f"  {'Others':<12} {len(others):>6} "
            f"${_median(o_pnls):>11,.0f} {_pct(_median(o_rets)):>10} "
            f"${sum(o_pnls):>11,.0f}"
        )

        # Top 20
        print(f"\n  {C.BD}Top 20 by PnL:{C.E}")
        print(f"  {'Wallet':<44} {'Suspect':>7} {'Trades':>6} {'Flags':>5} {'Rate':>6} {'PnL':>12} {'Return':>8}")
        _ruler(80)
        for r in results[:20]:
            tag = f"{C.R}    ★{C.E}" if r["suspect"] else "     "
            print(
                f"  {r['wallet']:<44} {tag:>7} {r['trades']:>6} "
                f"{r['flags']:>5} {_pct(r['rate']):>6} "
                f"${r['pnl']:>11,.0f} {_pct(r['return']):>8}"
            )

        print()

    def do_clusters(self, line):
        """
        List all detected clusters.

        Usage:  clusters [--min-size <N>] [--min-density <D>]
        """
        args = self._parse(line)
        min_size, min_density = 0, 0.0
        i = 0
        while i < len(args):
            if args[i] == "--min-size" and i + 1 < len(args):
                min_size = int(args[i + 1]); i += 2
            elif args[i] == "--min-density" and i + 1 < len(args):
                min_density = float(args[i + 1]); i += 2
            else:
                i += 1

        rows = self.db.q("""
            SELECT cluster_id, size, density, total_edge_weight,
                   has_common_ownership, attribution_enriched, computed_at
            FROM cl.cluster_metadata
            WHERE size >= ? AND density >= ?
            ORDER BY size DESC, density DESC
        """, (min_size, min_density))

        if not rows:
            print(f"  {C.Y}No clusters found{C.E}\n"); return

        print(f"\n{C.BD}  CLUSTERS ({len(rows)} total){C.E}")
        _ruler()
        print(f"  {'ID':<5} {'Size':>5} {'Density':>8} {'Weight':>10} {'Owner':>6} {'Computed'}")
        _ruler()

        for r in rows:
            color = C.R if r["size"] >= 10 and r["density"] >= 0.7 else C.Y if r["size"] >= 5 else C.W
            own = f"{C.G}YES{C.E}" if r["has_common_ownership"] else " no"
            print(
                f"  {color}{r['cluster_id']:<5} {r['size']:>5} {r['density']:>8.3f} "
                f"{r['total_edge_weight']:>10.1f} {own:>6}  "
                f"{_ts_s(r['computed_at'])}{C.E}"
            )
        print()

    def do_cluster(self, line):
        """
        Inspect a specific cluster: members, alerts, attribution.

        Usage:  cluster <id>
        """
        cid_str = line.strip()
        if not cid_str:
            print(f"  {C.Y}Usage: cluster <id>{C.E}"); return
        cid = int(cid_str)

        meta = self.db.q1("SELECT * FROM cl.cluster_metadata WHERE cluster_id = ?", (cid,))
        if not meta:
            print(f"  {C.R}Cluster {cid} not found{C.E}\n"); return

        wallets_rows = self.db.q("SELECT wallet FROM cl.cluster_assignments WHERE cluster_id = ?", (cid,))
        wallets = [r["wallet"] for r in wallets_rows]

        print(f"\n{C.BD}{C.CY}{'═' * 70}")
        print(f"  CLUSTER {cid}")
        print(f"{'═' * 70}{C.E}")
        own = f"{C.G}YES{C.E}" if meta["has_common_ownership"] else f"{C.R}NO{C.E}"
        print(f"  Size: {meta['size']}  Density: {meta['density']:.3f}  "
              f"Weight: {meta['total_edge_weight']:.1f}  Ownership: {own}")

        # Member table with alert stats
        print(f"\n  {C.BD}Members:{C.E}")
        print(f"  {'Wallet':<44} {'Alerts':>6} {'Buy$':>10} {'Markets':>8}")
        _ruler(70)

        for w in wallets:
            a = self.db.q1(
                "SELECT COUNT(*) AS n, COALESCE(SUM(notional_usdc),0) AS vol, "
                "COALESCE(SUM(CASE WHEN side='BUY' THEN notional_usdc ELSE 0 END),0) AS buy_vol "
                "FROM alerts WHERE wallet = ?",
                (w,)
            )
            m = self.db.q1(
                "SELECT COUNT(DISTINCT condition_id) AS n FROM wallet_market_stats WHERE wallet = ?",
                (w,)
            )
            mkt_count = m["n"] if m else 0
            color = C.R if a["n"] >= 5 else C.Y if a["n"] >= 1 else C.W
            buy_label = f"${a['buy_vol']:>9,.0f}" if a["buy_vol"] > 0 else f"{C.Y}$        0{C.E}"
            print(f"  {color}{w:<44} {a['n']:>6} {buy_label} {mkt_count:>8}{C.E}")

        # Attribution edges
        if len(wallets) >= 2:
            ph = ",".join("?" * len(wallets))
            edges = self.db.q(f"""
                SELECT from_wallet, to_wallet, total_amount, tx_count
                FROM cl.attribution_edges
                WHERE from_wallet IN ({ph}) AND to_wallet IN ({ph})
                ORDER BY total_amount DESC LIMIT 10
            """, wallets + wallets)

            if edges:
                print(f"\n  {C.BD}USDC transfers (intra-cluster):{C.E}")
                for e in edges:
                    print(f"    {e['from_wallet'][:11]}… → {e['to_wallet'][:11]}…  "
                          f"${e['total_amount']:>10,.0f}  ({e['tx_count']} txs)")
            else:
                print(f"\n  {C.Y}No USDC transfers between members{C.E}")

        print()

    def do_markets(self, line):
        """
        Show markets ranked by alert count.

        Usage:  markets [<limit=10>]
        """
        limit = int(line.strip()) if line.strip() else 10

        rows = self.db.q("""
            SELECT condition_id, market_slug,
                   COUNT(*) AS alerts,
                   COUNT(DISTINCT wallet) AS wallets,
                   SUM(notional_usdc) AS volume,
                   AVG(total_score) AS avg_score
            FROM alerts
            GROUP BY condition_id
            ORDER BY alerts DESC
            LIMIT ?
        """, (limit,))

        if not rows:
            print(f"  {C.Y}No alerts yet{C.E}\n"); return

        print(f"\n{C.BD}  TOP {limit} MARKETS BY ALERT COUNT{C.E}")
        _ruler()
        print(f"  {'Market':<40} {'Alerts':>7} {'Wallets':>8} {'Volume':>12} {'AvgScore':>8}")
        _ruler()

        for r in rows:
            color = C.R if r["alerts"] >= 50 else C.Y if r["alerts"] >= 20 else C.W
            print(
                f"  {color}{(r['market_slug'] or '')[:38]:<40} {r['alerts']:>7} "
                f"{r['wallets']:>8} ${r['volume']:>11,.0f} {r['avg_score']:>8.3f}{C.E}"
            )
        print()

    def do_attribution(self, line):
        """
        Attribution (Layer 2) summary or wallet-level transfers.

        Usage:
            attribution              — overall stats
            attribution <wallet>     — transfers for a specific wallet
        """
        target = line.strip()

        if target:
            self._wallet_transfers(target)
            return

        # Overall summary
        cache = self.db.q1("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status='queried' THEN 1 ELSE 0 END) AS ok,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS fail
            FROM cl.attribution_cache
        """)

        edges = self.db.q1("""
            SELECT COUNT(*) AS n,
                   COUNT(DISTINCT from_wallet) AS senders,
                   COUNT(DISTINCT to_wallet) AS receivers,
                   COALESCE(SUM(total_amount), 0) AS total_usdc,
                   COALESCE(SUM(tx_count), 0) AS txs
            FROM cl.attribution_edges
        """)

        print(f"\n{C.BD}  ATTRIBUTION SUMMARY (Layer 2){C.E}")
        _ruler()
        if cache:
            print(f"  Wallets queried: {cache['total']}  ({C.G}{cache['ok']} ok{C.E}, {C.R}{cache['fail']} failed{C.E})")
        if edges:
            print(f"  Transfer edges:  {edges['n']}  ({edges['senders']} senders → {edges['receivers']} receivers)")
            print(f"  Total USDC:      ${edges['total_usdc']:,.0f}  ({edges['txs']} transactions)")
        print()

    def _wallet_transfers(self, prefix: str):
        wallet = self._resolve_wallet(prefix)
        if not wallet:
            print(f"  {C.R}No wallet found for '{prefix}'{C.E}\n"); return

        out = self.db.q("""
            SELECT to_wallet, total_amount, tx_count, first_tx, last_tx
            FROM cl.attribution_edges WHERE from_wallet = ?
            ORDER BY total_amount DESC LIMIT 10
        """, (wallet,))

        inc = self.db.q("""
            SELECT from_wallet, total_amount, tx_count, first_tx, last_tx
            FROM cl.attribution_edges WHERE to_wallet = ?
            ORDER BY total_amount DESC LIMIT 10
        """, (wallet,))

        print(f"\n{C.BD}  TRANSFERS: {wallet[:20]}…{C.E}")

        if out:
            print(f"\n  {C.BD}Outgoing:{C.E}")
            for r in out:
                print(f"    → {r['to_wallet'][:14]}…  ${r['total_amount']:>10,.0f}  "
                      f"({r['tx_count']} txs, {_ts_s(r['first_tx'])} — {_ts_s(r['last_tx'])})")

        if inc:
            print(f"\n  {C.BD}Incoming:{C.E}")
            for r in inc:
                print(f"    ← {r['from_wallet'][:14]}…  ${r['total_amount']:>10,.0f}  "
                      f"({r['tx_count']} txs, {_ts_s(r['first_tx'])} — {_ts_s(r['last_tx'])})")

        if not out and not inc:
            print(f"  {C.Y}No transfers found{C.E}")
        print()

    def do_top(self, line):
        """
        Top wallets by alert count.

        Usage:  top [<limit=10>]
        """
        limit = int(line.strip()) if line.strip() else 10

        rows = self.db.q("""
            SELECT wallet, COUNT(*) AS alerts,
                   SUM(notional_usdc) AS volume,
                   AVG(total_score) AS avg_score,
                   MAX(total_score) AS max_score,
                   SUM(CASE WHEN side = 'BUY' THEN notional_usdc ELSE 0 END) AS buy_volume
            FROM alerts
            GROUP BY wallet
            HAVING buy_volume > 0
            ORDER BY alerts DESC
            LIMIT ?
        """, (limit,))

        if not rows:
            print(f"  {C.Y}No alerts yet{C.E}\n"); return

        print(f"\n{C.BD}  TOP {limit} WALLETS BY ALERT COUNT{C.E}")
        _ruler()
        print(f"  {'Wallet':<44} {'Alerts':>7} {'Volume':>12} {'Avg':>6} {'Max':>6}")
        _ruler()

        for r in rows:
            color = C.R if r["alerts"] >= 10 else C.Y if r["alerts"] >= 3 else C.W
            print(
                f"  {color}{r['wallet']:<44} {r['alerts']:>7} "
                f"${r['volume']:>11,.0f} {r['avg_score']:>6.3f} {r['max_score']:>6.3f}{C.E}"
            )
        print()

    def do_quit(self, _line):
        """Exit the explorer."""
        print(f"  {C.G}Goodbye{C.E}\n")
        return True

    do_exit = do_quit
    do_q = do_quit

    def do_EOF(self, _line):
        print()
        return True

    def emptyline(self):
        pass

    def default(self, line):
        print(f"  {C.Y}Unknown command: '{line}'. Type 'help' for commands.{C.E}")


def main():
    alert_path = sys.argv[1] if len(sys.argv) > 1 else "insider_trading_v1.db"
    cluster_path = sys.argv[2] if len(sys.argv) > 2 else "clustering_v1.db"

    try:
        db = DB(alert_path, cluster_path)
    except Exception as e:
        print(f"{C.R}Failed to connect: {e}{C.E}")
        sys.exit(1)

    try:
        Explorer(db).cmdloop()
    except KeyboardInterrupt:
        print(f"\n  {C.G}Goodbye{C.E}\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()