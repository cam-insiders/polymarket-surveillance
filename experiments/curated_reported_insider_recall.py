"""
Experiment: reported-insider wallet recall on curated suspicious markets.

Runs the current official backtest path on a hardcoded set of markets that were
reported publicly as suspicious, then checks whether reported insider wallets
were classified by the system and whether they were ever alerted at least once.

Reported-wallet presence uses maker-or-taker SQLite rows in the active data
directory. Classification metrics (flag_rate) use maker-only backtest trade
counts so they match the detector path; participation counts are exported
separately for reference.

Usage:
    python3 -m experiments.curated_reported_insider_recall path/to/config.json

    # Also run SOTA baselines (train window uses data/ by default; eval uses --data-dir):
    python3 -m experiments.curated_reported_insider_recall path/to/config.json --compare-sota \\
        --data-dir data/curated_fromvm --train-start 2025-02-01 --train-end 2025-02-14

    # Or run the comparison-focused entry point directly:
    python3 -m experiments.curated_sota_common path/to/config.json \\
        --train-start 2024-01-01 --train-end 2025-05-01
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import (
    DEFAULT_CLUSTERING_CONFIG,
    evaluate_config,
    predict_wallet_positive,
)
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode
from experiments.timeframe_market_common import infer_resolutions


DEFAULT_OUTPUT_DIR = "experiments/results/curated_reported_insider_recall"
FULL_WALLET_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


# NOTE:
# Keep this list conservative. Only full wallet addresses in reported_wallets
# are counted in denominators; handles/partial addresses stay in notes until
# independently verified.
CURATED_CASES: List[Dict[str, Any]] = [
    {
        "market_id": 1419011,
        "market_slug": "will-axiom-be-accused-of-insider-trading",
        "external_market_ref": (
            "polymarket.com/event/which-crypto-company-will-zachxbt-expose-for-insider-trading/"
            "will-axiom-be-accused-of-insider-trading"
        ),
        "reported_wallets": [
            "0x1d9af60c679cd0b577c3c4ccb4b1a4be4174426d",
            "0xe56526b27b96f009b31ddb46558a134047bfce48",
            "0x054ec2f0ccfdae941886a3ed306635068c716639",
            "0x6d6affce1ed04a0e9611484daf1cef5cbcf3fb40",
            "0x581f34349babaf03b2d3c8f5f60cf44ffbe19a3a",
            "0x5e524f43357198fa815e6766f02fe686b444b064",
            "0x572c8005aa033237175f16de725969b044cd0383",
            "0xaab29084bcc42daff9e11b4a5a4cc55cda3eb306",
            "0x98a96619e482700e83e8486e4f3727dba17f5381",
            "0xeeff2d748ad5efcfbbb3c8858f608d6b6321a398",
            "0xd9eab53eaba81333045da5bd84ce6c833f721e89",
            "0xff55beaf369387d7748a31213699a51f1ca8b877",
        ],
        "source_urls": [
            "https://coincentral.com/polymarket-bets-hint-at-insider-edge-in-axiom-probe/",
            "https://www.kucoin.com/news/flash/12-suspected-insider-wallets-earn-over-1m-on-polymarket-predicting-zachxbt-s-leak",
            "https://polygonscan.com/address/0x1d9af60c679cd0b577c3c4ccb4b1a4be4174426d",
        ],
        "label_confidence": "reported_full_address_not_local_yet",
        "notes": (
            "Not in local dataset yet. Lookonchain/CoinDesk-linked reporting "
            "identifies predictorxyz as the largest Axiom holder in the "
            "ZachXBT investigation market; source reports list this full "
            "wallet address."
        ),
    },
    {
        "market_id": 1198423,
        "market_slug": "us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761",
        "external_market_ref": "Polymarket: US strikes Iran by February 28, 2026?",
        "reported_wallets": [
            "0x1caa6a7ad0c6916aef7b67946de2e57ad24846a0",
        ],
        "source_urls": [
            "https://polyspotter.com/wallet/0x1caa6a7ad0c6916aef7b67946de2e57ad24846a0",
            "https://www.frenflow.com/traders/0x1caa6a7ad0c6916aef7b67946de2e57ad24846a0",
            "https://polymarket.com/es/profile/%400x1caA6a7ad0c6916aeF7b67946De2e57Ad24846a0-1772054568088",
        ],
        "label_confidence": "reported_full_address_not_local_yet",
        "notes": (
            "Not in local dataset yet. Public trader/profile mirrors show this "
            "wallet's large winning YES position on US strikes Iran by "
            "February 28, 2026; Perplexity lead maps it to the Bubblemaps "
            "fresh-wallet cluster."
        ),
    },
    {
        "market_id": 916440,
        "market_slug": "maduro-out-by-january-31-2026-318",
        "external_market_ref": "Polymarket: Maduro out by January 31, 2026?",
        "reported_wallets": [
            "0xa72db1749e9ac2379d49a3c12708325ed17febd4",
            "0x6baf05d193692bb208d616709e27442c910a94c5",
        ],
        "source_urls": [
            "https://www.kucoin.com/news/flash/three-wallets-on-polymarket-profit-over-630-000-by-betting-maduro-s-downfall-before-arrest",
            "https://www.scanwhale.com/traders/0x6baf05d193692bb208d616709e27442c910a94c5",
            "https://fortune.com/2026/01/12/polymarket-kalshi-insider-trading-prediction-markets-cftc-torres-titus-venezuela/",
        ],
        "label_confidence": "reported_full_address_not_local_yet",
        "notes": (
            "Not in local dataset yet. KuCoin/Lookonchain reporting identifies "
            "0xa72D as one of the Maduro insider wallets; ScanWhale links "
            "SBet365 to 0x6baf...94c5 and shows Maduro out by January 31, "
            "2026 positions. Burdensome-Mix is not added because public "
            "sources found here expose only handle/redacted address details."
        ),
    },
    {
        "market_id": 635246,
        "market_slug": "will-d4vd-be-the-1-searched-person-on-google-this-year",
        "external_market_ref": (
            "polymarket.com/event/1-searched-person-on-google-this-year/"
            "will-d4vd-be-the-1-searched-person-on-google-this-year"
        ),
        "reported_wallets": [
            "0xee50a31c3f5a7c77824b12a941a54388a2827ed6",
        ],
        "source_urls": [
            "https://polymarket.com/profile/0xee50a31c3f5a7c77824b12a941a54388a2827ed6",
            "https://polymarket.com/event/1-searched-person-on-google-this-year/will-d4vd-be-the-1-searched-person-on-google-this-year",
            "https://thedefiant.io/news/defi/polymarket-users-suspect-insider-trading-after-google-trend-markets-crown-surprise-winner",
            "https://www.vegasslotsonline.com/news/2025/12/04/trader-hits-22-of-23-google-bets-as-insider-allegations-rock-polymarket/",
            "https://finance.yahoo.com/news/polymarket-trader-makes-1-million-090001027.html",
            "https://www.frenflow.com/traders/%400xafEe",
        ],
        "label_confidence": "reported_full_address_not_local_yet",
        "notes": (
            "Not in local dataset yet. Public reports identify AlphaRacoon, "
            "later renamed 0xafEe, as the suspicious Google Year in Search "
            "trader who won 22/23 Google search markets and made over $1M; "
            "Polymarket/FrenFlow profile pages map 0xafEe to this full wallet."
        ),
    },
    {
        "market_id": 560868,
        "market_slug": "will-mara-corina-machado-win-the-nobel-peace-prize-in-2025",
        "reported_wallets": [
            "0xa430506774f9efaf39903ee7e0db1351f66891ca",
        ],
        "source_urls": [
            "https://polymarket.com/profile/0xa430506774f9efaf39903ee7e0db1351f66891ca",
        ],
    },
    {
        "market_id": 573811,
        "market_slug": "will-gemini-3pt0-be-released-by-october-31-881-568",
        "reported_wallets": [
            "0xee50a31c3f5a7c77824b12a941a54388a2827ed6",
        ],
        "source_urls": [
            "https://polymarket.com/profile/0xee50a31c3f5a7c77824b12a941a54388a2827ed6",
            "https://polymarket.com/event/1-searched-person-on-google-this-year/will-d4vd-be-the-1-searched-person-on-google-this-year",
            "https://thedefiant.io/news/defi/polymarket-users-suspect-insider-trading-after-google-trend-markets-crown-surprise-winner",
            "https://www.vegasslotsonline.com/news/2025/12/04/trader-hits-22-of-23-google-bets-as-insider-allegations-rock-polymarket/",
            "https://finance.yahoo.com/news/polymarket-trader-makes-1-million-090001027.html",
            "https://www.frenflow.com/traders/%400xafEe",
        ],
        "label_confidence": "reported_full_address_not_local_yet",
    },
    {
        "market_id": 551651,
        "market_slug": "israel-military-action-against-iran-by-friday-477",
        "winning_outcome": 0,
        "reported_wallets": [
            "0x0afc7ce56285bde1fbe3a75efaffdfc86d6530b2",
        ],
        "source_urls": [
            "https://www.gamblingnerd.com/blog/polymarket-iran-insider-trading/",
            "https://www.bitget.com/news/detail/12560605241825",
            "https://dyutam.com/news/israel-polymarket-insider-trading-military-secrets/",
            "https://www.jpost.com/israel-news/crime-in-israel/article-884318",
            "https://polygonscan.com/address/0x0Afc7CE56285BdE1fBE3A75eFAffdFC86d6530B2",
        ],
        "label_confidence": "reported_full_address",
        "notes": (
            "Reports link the Polymarket account ricosuave666 to this full "
            "wallet address and describe profitable June 2025 Israel/Iran "
            "military-action bets, including this Friday market."
        ),
    },
    {
        "market_id": 532742,
        "market_slug": "israel-military-action-against-iran-before-july",
        "winning_outcome": 0,
        "reported_wallets": [
            "0x0afc7ce56285bde1fbe3a75efaffdfc86d6530b2",
        ],
        "source_urls": [
            "https://www.gamblingnerd.com/blog/polymarket-iran-insider-trading/",
            "https://dyutam.com/news/israel-polymarket-insider-trading-military-secrets/",
            "https://www.jpost.com/israel-news/crime-in-israel/article-884318",
            "https://polygonscan.com/address/0x0Afc7CE56285BdE1fBE3A75eFAffdFC86d6530B2",
        ],
        "label_confidence": "reported_full_address",
        "notes": (
            "JPost reports the same anonymous Polymarket user correctly "
            "predicted Israel attacking Iran before July; Dyutam links "
            "ricosuave666 to this full wallet address."
        ),
    },
    {
        "market_id": 554000,
        "market_slug": "israel-strike-on-iran-on-june-24",
        "reported_wallets": [
            "0x0afc7ce56285bde1fbe3a75efaffdfc86d6530b2",
        ],
        "source_urls": [
            "https://comments.cftc.gov/Handlers/PdfHandler.ashx?id=35914",
            "https://dyutam.com/news/israel-polymarket-insider-trading-military-secrets/",
            "https://polygonscan.com/address/0x0Afc7CE56285BdE1fBE3A75eFAffdFC86d6530B2",
        ],
        "label_confidence": "reported_full_address",
        "notes": (
            "CFTC comment by Mitts/Ofir identifies the ricosuave666 wallet "
            "as associated with suspicious trading in the Israel strike on "
            "Iran on June 24 market; Dyutam links ricosuave666 to this full "
            "wallet address."
        ),
    },
    {
        "market_id": 552658,
        "market_slug": "israel-announces-end-of-military-operations-against-iran-before-july",
        "winning_outcome": 0,
        "reported_wallets": [
            "0x0afc7ce56285bde1fbe3a75efaffdfc86d6530b2",
        ],
        "source_urls": [
            "https://www.gamblingnerd.com/blog/polymarket-iran-insider-trading/",
            "https://dyutam.com/news/israel-polymarket-insider-trading-military-secrets/",
            "https://www.jpost.com/israel-news/crime-in-israel/article-884318",
            "https://polygonscan.com/address/0x0Afc7CE56285BdE1fBE3A75eFAffdFC86d6530B2",
        ],
        "label_confidence": "reported_full_address",
        "notes": (
            "JPost reports the same anonymous Polymarket user correctly "
            "predicted Israel announcing the end of Iran operations before "
            "July; Dyutam links ricosuave666 to this full wallet address."
        ),
    },
]


def _normalize_wallet(wallet: str) -> str:
    return str(wallet).strip().lower()


def is_full_wallet_address(wallet: str) -> bool:
    return bool(FULL_WALLET_RE.match(str(wallet).strip()))


def get_counted_reported_wallets(case: Dict[str, Any]) -> List[str]:
    """Return unique, normalized full wallet addresses for a curated case."""
    seen = set()
    wallets: List[str] = []
    for raw_wallet in case.get("reported_wallets", []) or []:
        wallet = _normalize_wallet(raw_wallet)
        if not is_full_wallet_address(wallet) or wallet in seen:
            continue
        seen.add(wallet)
        wallets.append(wallet)
    return wallets


def is_case_in_local_dataset(case: Dict[str, Any]) -> bool:
    return str(case.get("dataset_status", "local")) != "not_in_local_dataset_yet"


def is_case_available_in_loader(loader: HistoricalDataLoader, case: Dict[str, Any]) -> bool:
    """Return whether the active data directory has metadata for this case."""
    return loader.get_market_metadata(int(case["market_id"])) is not None


def _wallet_address_variants(wallet: str) -> List[str]:
    """Return lowercase address variants stored in trades.db (with/without 0x)."""
    wallet = _normalize_wallet(wallet)
    variants = {wallet}
    if wallet.startswith("0x"):
        variants.add(wallet[2:])
    else:
        variants.add(f"0x{wallet}")
    return sorted(variants)


def build_reported_wallet_participation_counts(
    loader: HistoricalDataLoader,
    curated_cases: Iterable[Dict[str, Any]],
) -> Dict[Tuple[int, str], int]:
    """
    Count SQLite trade rows where a reported wallet is maker or taker.

    Used only by this experiment so recall denominators match on-chain
    participation without changing HistoricalDataLoader (maker-only backtests).
    """
    if not loader.use_sqlite or not Path(loader.db_path).exists():
        return {}

    pairs: List[Tuple[int, str]] = []
    for case in curated_cases:
        market_id = int(case["market_id"])
        for wallet in get_counted_reported_wallets(case):
            pairs.append((market_id, _normalize_wallet(wallet)))

    if not pairs:
        return {}

    by_market: Dict[int, List[str]] = defaultdict(list)
    for market_id, wallet in pairs:
        if wallet not in by_market[market_id]:
            by_market[market_id].append(wallet)

    counts: Dict[Tuple[int, str], int] = {}
    conn = sqlite3.connect(loader.db_path)
    try:
        for market_id, wallets in by_market.items():
            for wallet in wallets:
                variants = _wallet_address_variants(wallet)
                placeholders = ",".join("?" * len(variants))
                query = (
                    f"SELECT COUNT(*) FROM trades WHERE market_id = ? AND ("
                    f"lower(maker) IN ({placeholders}) OR lower(taker) IN ({placeholders})"
                    f")"
                )
                params: List[Any] = [market_id, *variants, *variants]
                row = conn.execute(query, params).fetchone()
                counts[(market_id, wallet)] = int(row[0]) if row else 0
    finally:
        conn.close()
    return counts


def _wallet_eval_from_backtest(
    backtest_result: Optional[Any],
    wallet: str,
    *,
    participation_trade_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """Build the minimal wallet-evaluation shape used by predict_wallet_positive."""
    wallet = _normalize_wallet(wallet)
    if backtest_result is None:
        trade_counts: Dict[str, int] = {}
        flags_by_wallet: Dict[str, List[Any]] = {}
        suspicion_by_wallet = {}
        cluster_boost_by_wallet = {}
        common_ownership_by_wallet = {}
    else:
        trade_counts = {
            _normalize_wallet(w): int(v)
            for w, v in getattr(backtest_result, "wallet_trade_counts", {}).items()
        }
        flags_by_wallet = {
            _normalize_wallet(w): list(v)
            for w, v in getattr(backtest_result, "wallet_flags", {}).items()
        }
        suspicion_by_wallet = {
            _normalize_wallet(w): float(v)
            for w, v in getattr(backtest_result, "wallet_suspicion", {}).items()
        }
        cluster_boost_by_wallet = {
            _normalize_wallet(w): float(v)
            for w, v in getattr(backtest_result, "wallet_cluster_boost", {}).items()
        }
        common_ownership_by_wallet = {
            _normalize_wallet(w): bool(v)
            for w, v in getattr(backtest_result, "wallet_has_common_ownership", {}).items()
        }

    in_backtest = wallet in trade_counts
    if participation_trade_count <= 0 and not in_backtest:
        return None

    flags = flags_by_wallet.get(wallet, [])
    backtest_trade_count = trade_counts.get(wallet, 0)
    # Flag-rate classification must use trades the backtest actually processed
    # (maker legs). Participation can include taker-only rows the detector never sees.
    if backtest_trade_count > 0:
        trade_count = backtest_trade_count
    else:
        trade_count = participation_trade_count
    return {
        "wallet": wallet,
        "trade_count": trade_count,
        "participation_trade_count": participation_trade_count,
        "backtest_trade_count": backtest_trade_count,
        "num_flags": len(flags),
        "has_alert": len(flags) > 0,
        "suspicion_score": suspicion_by_wallet.get(wallet, 0.0),
        "cluster_boost": cluster_boost_by_wallet.get(wallet, 1.0),
        "has_common_ownership": common_ownership_by_wallet.get(wallet, False),
    }


def build_maker_trade_counts_by_market(
    loader: HistoricalDataLoader,
    market_ids: Iterable[int],
    *,
    min_usd_amount: Optional[float] = None,
) -> Dict[int, Dict[str, int]]:
    """Count maker-leg trades per wallet (matches backtest / SOTA trade paths)."""
    counts: Dict[int, Dict[str, int]] = {}
    for market_id in market_ids:
        try:
            trades = loader.get_trades_for_market(
                market_id=int(market_id),
                min_usd_amount=min_usd_amount,
                use_cache=False,
            )
        except TypeError:
            trades = loader.get_trades_for_market(int(market_id))
        market_counts: Dict[str, int] = defaultdict(int)
        for trade in trades:
            market_counts[_normalize_wallet(trade.wallet)] += 1
        counts[int(market_id)] = dict(market_counts)
    return counts


def build_reported_wallet_rows_from_sota_flags(
    *,
    curated_cases: Iterable[Dict[str, Any]],
    participation_counts: Optional[Dict[Tuple[int, str], int]] = None,
    flagged_wallets_by_market: Dict[int, Set[str]],
    wallet_flag_counts_by_market: Dict[int, Dict[str, int]],
    maker_trade_counts_by_market: Dict[int, Dict[str, int]],
    method: str,
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
    wallet_level_positive: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build reported-wallet rows from SOTA per-market flag state.
    """
    participation_counts = participation_counts or {}
    rows: List[Dict[str, Any]] = []
    for case in curated_cases:
        market_id = int(case["market_id"])
        flagged_wallets = {
            _normalize_wallet(w)
            for w in flagged_wallets_by_market.get(market_id, set())
        }
        flag_counts = {
            _normalize_wallet(w): int(c)
            for w, c in wallet_flag_counts_by_market.get(market_id, {}).items()
        }
        trade_counts = maker_trade_counts_by_market.get(market_id, {})

        for wallet in get_counted_reported_wallets(case):
            participation = int(participation_counts.get((market_id, wallet), 0))
            backtest_trade_count = int(trade_counts.get(wallet, 0))
            present = participation > 0 or backtest_trade_count > 0
            num_flags = int(flag_counts.get(wallet, 0))
            if wallet in flagged_wallets and num_flags <= 0:
                num_flags = 1
            trade_count = backtest_trade_count if backtest_trade_count > 0 else participation
            flag_rate = (num_flags / trade_count) if trade_count > 0 else 0.0
            has_alert = num_flags > 0
            wallet_eval = {
                "wallet": wallet,
                "trade_count": trade_count,
                "participation_trade_count": participation,
                "backtest_trade_count": backtest_trade_count,
                "num_flags": num_flags,
                "has_alert": has_alert,
                "suspicion_score": 0.0,
                "cluster_boost": 1.0,
                "has_common_ownership": False,
            }
            if wallet_level_positive:
                classified = has_alert
            elif present:
                classified = predict_wallet_positive(
                    wallet_eval,
                    prediction_mode,
                    suspicion_threshold,
                    flag_rate_threshold,
                )
            else:
                classified = False
            rows.append(
                {
                    "method": method,
                    "market_id": market_id,
                    "market_slug": str(case.get("market_slug", market_id)),
                    "wallet": wallet,
                    "present_in_trade_data": bool(present),
                    "participation_trade_count": participation,
                    "backtest_trade_count": backtest_trade_count,
                    "trade_count": trade_count,
                    "num_flags": num_flags,
                    "flag_rate": float(flag_rate),
                    "has_alert": bool(has_alert),
                    "classified_positive": bool(classified),
                    "wallet_level_positive": bool(wallet_level_positive),
                    "suspicion_score": 0.0,
                    "cluster_boost": 1.0,
                    "has_common_ownership": False,
                    "label_confidence": str(case.get("label_confidence", "")),
                    "source_urls": ";".join(str(x) for x in case.get("source_urls", []) or []),
                    "notes": str(case.get("notes", "")),
                }
            )
    return rows


def build_reported_wallet_rows(
    *,
    curated_cases: Iterable[Dict[str, Any]],
    backtest_results: Dict[int, Any],
    participation_counts: Optional[Dict[Tuple[int, str], int]] = None,
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
) -> List[Dict[str, Any]]:
    """Build one output row per counted reported wallet."""
    participation_counts = participation_counts or {}
    rows: List[Dict[str, Any]] = []
    for case in curated_cases:
        market_id = int(case["market_id"])
        backtest_result = backtest_results.get(market_id)
        for wallet in get_counted_reported_wallets(case):
            participation = int(participation_counts.get((market_id, wallet), 0))
            wallet_eval = _wallet_eval_from_backtest(
                backtest_result,
                wallet,
                participation_trade_count=participation,
            )
            present = wallet_eval is not None
            participation_trade_count = (
                int(wallet_eval.get("participation_trade_count", 0)) if present else 0
            )
            backtest_trade_count = int(wallet_eval.get("backtest_trade_count", 0)) if present else 0
            trade_count = int(wallet_eval.get("trade_count", 0)) if present else 0
            num_flags = int(wallet_eval.get("num_flags", 0)) if present else 0
            flag_rate = (num_flags / trade_count) if trade_count > 0 else 0.0
            classified = (
                predict_wallet_positive(
                    wallet_eval,
                    prediction_mode,
                    suspicion_threshold,
                    flag_rate_threshold,
                )
                if present
                else False
            )
            rows.append(
                {
                    "method": "full_system",
                    "market_id": market_id,
                    "market_slug": str(case.get("market_slug", market_id)),
                    "wallet": wallet,
                    "present_in_trade_data": bool(present),
                    "participation_trade_count": participation_trade_count,
                    "backtest_trade_count": backtest_trade_count,
                    "trade_count": trade_count,
                    "num_flags": num_flags,
                    "flag_rate": float(flag_rate),
                    "has_alert": bool(num_flags > 0),
                    "classified_positive": bool(classified),
                    "suspicion_score": float(wallet_eval.get("suspicion_score", 0.0)) if present else 0.0,
                    "cluster_boost": float(wallet_eval.get("cluster_boost", 1.0)) if present else 1.0,
                    "has_common_ownership": (
                        bool(wallet_eval.get("has_common_ownership", False)) if present else False
                    ),
                    "label_confidence": str(case.get("label_confidence", "")),
                    "source_urls": ";".join(str(x) for x in case.get("source_urls", []) or []),
                    "notes": str(case.get("notes", "")),
                }
            )
    return rows


def summarize_method_rows(
    curated_cases: Iterable[Dict[str, Any]],
    wallet_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One aggregate summary row per detection method."""
    methods = sorted({str(row.get("method", "full_system")) for row in wallet_rows})
    summaries: List[Dict[str, Any]] = []
    for method in methods:
        method_wallet_rows = [row for row in wallet_rows if row.get("method") == method]
        method_market_rows = summarize_market_rows(
            curated_cases,
            method_wallet_rows,
            method=method,
        )
        summaries.append(
            {
                "method": method,
                **summarize_aggregate(method_wallet_rows, method_market_rows),
            }
        )
    return summaries


def summarize_all_method_market_rows(
    curated_cases: Iterable[Dict[str, Any]],
    wallet_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One market-level summary row per detection method and curated market."""
    methods = sorted({str(row.get("method", "full_system")) for row in wallet_rows})
    rows: List[Dict[str, Any]] = []
    for method in methods:
        method_wallet_rows = [row for row in wallet_rows if row.get("method") == method]
        rows.extend(
            summarize_market_rows(
                curated_cases,
                method_wallet_rows,
                method=method,
            )
        )
    return rows


def summarize_market_rows(
    curated_cases: Iterable[Dict[str, Any]],
    wallet_rows: List[Dict[str, Any]],
    *,
    method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows_by_market: Dict[int, List[Dict[str, Any]]] = {}
    for row in wallet_rows:
        rows_by_market.setdefault(int(row["market_id"]), []).append(row)

    market_rows: List[Dict[str, Any]] = []
    for case in curated_cases:
        market_id = int(case["market_id"])
        target_wallets = get_counted_reported_wallets(case)
        rows = rows_by_market.get(market_id, [])
        total = len(target_wallets)
        present = sum(1 for row in rows if bool(row["present_in_trade_data"]))
        classified = sum(1 for row in rows if bool(row["classified_positive"]))
        ever_flagged = sum(1 for row in rows if bool(row["has_alert"]))
        present_flag_rates = [
            float(row.get("flag_rate", 0.0))
            for row in rows
            if bool(row.get("present_in_trade_data", False))
        ]
        classified_rate_present = classified / present if present > 0 else 0.0
        classified_rate_total = classified / total if total > 0 else 0.0
        market_rows.append(
            {
                "method": (
                    method
                    if method is not None
                    else str(rows[0].get("method", "full_system")) if rows else "full_system"
                ),
                "market_id": market_id,
                "market_slug": str(case.get("market_slug", market_id)),
                "reported_wallets_total": total,
                "reported_wallets_present": present,
                "classified_reported_wallets": classified,
                "ever_flagged_reported_wallets": ever_flagged,
                "classification_recall_present": classified_rate_present,
                "classification_rate_present": classified_rate_present,
                "ever_flagged_recall_present": ever_flagged / present if present > 0 else 0.0,
                "classification_recall_total": classified_rate_total,
                "classification_rate_total": classified_rate_total,
                "ever_flagged_recall_total": ever_flagged / total if total > 0 else 0.0,
                "mean_reported_wallet_flag_rate_present": (
                    sum(present_flag_rates) / len(present_flag_rates)
                    if present_flag_rates
                    else 0.0
                ),
                "max_reported_wallet_flag_rate_present": (
                    max(present_flag_rates) if present_flag_rates else 0.0
                ),
                "source_urls": ";".join(str(x) for x in case.get("source_urls", []) or []),
                "notes": str(case.get("notes", "")),
            }
        )
    return market_rows


def summarize_aggregate(
    wallet_rows: List[Dict[str, Any]],
    market_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total = sum(int(row["reported_wallets_total"]) for row in market_rows)
    present = sum(int(row["reported_wallets_present"]) for row in market_rows)
    classified = sum(int(row["classified_reported_wallets"]) for row in market_rows)
    ever_flagged = sum(int(row["ever_flagged_reported_wallets"]) for row in market_rows)
    missing_wallets = [
        {
            "market_id": int(row["market_id"]),
            "market_slug": row["market_slug"],
            "wallet": row["wallet"],
        }
        for row in wallet_rows
        if not bool(row["present_in_trade_data"])
    ]
    unique_wallets: Dict[str, Dict[str, bool]] = {}
    for row in wallet_rows:
        wallet = _normalize_wallet(str(row.get("wallet", "")))
        if not wallet:
            continue
        state = unique_wallets.setdefault(
            wallet,
            {
                "present": False,
                "classified": False,
                "ever_flagged": False,
            },
        )
        state["present"] = state["present"] or bool(row.get("present_in_trade_data", False))
        state["classified"] = state["classified"] or bool(row.get("classified_positive", False))
        state["ever_flagged"] = state["ever_flagged"] or bool(row.get("has_alert", False))

    unique_total = len(unique_wallets)
    unique_present = sum(1 for state in unique_wallets.values() if state["present"])
    unique_classified = sum(1 for state in unique_wallets.values() if state["classified"])
    unique_ever_flagged = sum(1 for state in unique_wallets.values() if state["ever_flagged"])
    return {
        "reported_wallets_total": total,
        "reported_wallets_present": present,
        "classified_reported_wallets": classified,
        "ever_flagged_reported_wallets": ever_flagged,
        "classification_recall_present": classified / present if present > 0 else 0.0,
        "ever_flagged_recall_present": ever_flagged / present if present > 0 else 0.0,
        "classification_recall_total": classified / total if total > 0 else 0.0,
        "ever_flagged_recall_total": ever_flagged / total if total > 0 else 0.0,
        "unique_reported_wallets_total": unique_total,
        "unique_reported_wallets_present": unique_present,
        "unique_classified_reported_wallets": unique_classified,
        "unique_ever_flagged_reported_wallets": unique_ever_flagged,
        "unique_classification_recall_present": (
            unique_classified / unique_present if unique_present > 0 else 0.0
        ),
        "unique_ever_flagged_recall_present": (
            unique_ever_flagged / unique_present if unique_present > 0 else 0.0
        ),
        "unique_classification_recall_total": (
            unique_classified / unique_total if unique_total > 0 else 0.0
        ),
        "unique_ever_flagged_recall_total": (
            unique_ever_flagged / unique_total if unique_total > 0 else 0.0
        ),
        "missing_wallets": missing_wallets,
    }


def _validate_curated_cases(loader: HistoricalDataLoader, cases: List[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    seen_market_ids = set()
    for case in cases:
        market_id = int(case["market_id"])
        if market_id in seen_market_ids:
            warnings.append(f"Duplicate curated market_id={market_id}")
        seen_market_ids.add(market_id)

        meta = loader.get_market_metadata(market_id)
        if meta is None:
            if is_case_in_local_dataset(case):
                warnings.append(f"Curated market_id={market_id} not found in local markets.csv")
            continue
        else:
            expected_slug = str(case.get("market_slug", ""))
            actual_slug = str(meta.get("market_slug", ""))
            if expected_slug and expected_slug != actual_slug:
                warnings.append(
                    f"Curated market_id={market_id} slug mismatch: "
                    f"expected={expected_slug!r} actual={actual_slug!r}"
                )

        invalid_wallets = [
            str(wallet)
            for wallet in case.get("reported_wallets", []) or []
            if not is_full_wallet_address(str(wallet))
        ]
        if invalid_wallets:
            warnings.append(
                f"Curated market_id={market_id} has non-counted non-full wallet entries: "
                + ", ".join(invalid_wallets)
            )
    return warnings


def _build_winning_outcomes(
    *,
    loader: HistoricalDataLoader,
    cases: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    explicit: Dict[int, int] = {
        int(case["market_id"]): int(case["winning_outcome"])
        for case in cases
        if (
            is_case_available_in_loader(loader, case)
            and get_counted_reported_wallets(case)
            and case.get("winning_outcome") is not None
        )
    }
    missing = [
        int(case["market_id"])
        for case in cases
        if (
            is_case_available_in_loader(loader, case)
            and get_counted_reported_wallets(case)
            and case.get("winning_outcome") is None
        )
    ]
    inferred: Dict[int, int] = {}
    stats: Dict[str, Any] = {
        "explicit": len(explicit),
        "needs_inference": len(missing),
        "inferred": 0,
        "resolution_stats": {},
    }
    if missing:
        inferred, resolution_stats = infer_resolutions(
            loader=loader,
            market_ids=missing,
            resolution_threshold=args.resolution_threshold,
            min_trades=args.min_trades,
            min_usd_amount=args.min_usd_amount,
            inferred_resolutions_db=args.inferred_resolutions_db,
            save_cache=True,
        )
        stats["inferred"] = len(inferred)
        stats["resolution_stats"] = resolution_stats

    winning_outcomes = dict(explicit)
    winning_outcomes.update(inferred)
    return winning_outcomes, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest a config on curated reported-insider markets and measure wallet recall."
    )
    parser.add_argument("config_path", type=str, help="Path to detector config JSON")
    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--flag-rate-threshold", type=float, default=0.2)
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--min-usd-amount", type=float, default=None)
    parser.add_argument("--include-recidivism", action="store_true", default=False)
    parser.add_argument("--clustering-min-trade-size", type=float, default=5000.0)
    parser.add_argument("--no-clustering", action="store_true", default=False)
    parser.add_argument("--no-jump-anticipation", action="store_true", default=False)
    parser.add_argument("--enable-layer2-attribution", action="store_true", default=True)
    parser.add_argument("--usdc-cache", type=str, default="data/usdc_transfers.db")
    parser.add_argument("--polygonscan-api-key", type=str, default=None)
    parser.add_argument("--copytrade-fixed-size", type=float, default=100.0)
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--inferred-resolutions-db", type=str, default="inferred_resolutions.db")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose-output", action="store_true", default=False)
    parser.add_argument(
        "--compare-sota",
        action="store_true",
        help="Also run SOTA baselines on curated markets and report recall per method.",
    )
    from experiments.curated_sota_common import add_compare_sota_args

    add_compare_sota_args(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    set_experiment_backtest_log_quiet_mode(enabled=not args.verbose_output)

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    cases = deepcopy(CURATED_CASES)
    validation_warnings = _validate_curated_cases(loader, cases)
    for warning in validation_warnings:
        logging.warning(warning)

    winning_outcomes, resolution_meta = _build_winning_outcomes(
        loader=loader,
        cases=cases,
        args=args,
    )
    cases_with_targets = [case for case in cases if get_counted_reported_wallets(case)]
    targetless_markets = [
        int(case["market_id"])
        for case in cases
        if not get_counted_reported_wallets(case)
    ]
    not_local_yet_markets = [
        int(case["market_id"])
        for case in cases_with_targets
        if not is_case_available_in_loader(loader, case)
    ]
    market_ids = [
        int(case["market_id"])
        for case in cases_with_targets
        if (
            is_case_available_in_loader(loader, case)
            and int(case["market_id"]) in winning_outcomes
            and loader.get_market_metadata(int(case["market_id"]))
        )
    ]
    skipped_markets = [
        int(case["market_id"])
        for case in cases_with_targets
        if is_case_available_in_loader(loader, case) and int(case["market_id"]) not in market_ids
    ]
    if not market_ids:
        loader.close()
        raise RuntimeError("No curated markets had local metadata and a known/inferred resolution.")

    clustering_config = None if args.no_clustering else config.get("clustering_config", DEFAULT_CLUSTERING_CONFIG)
    jump_anticipation_config = None if args.no_jump_anticipation else config.get("jump_anticipation_config")

    logging.info(
        "Evaluating %d curated market(s) with prediction_mode=%s flag_rate_threshold=%.4f",
        len(market_ids),
        args.prediction_mode,
        args.flag_rate_threshold,
    )
    result = evaluate_config(
        config=config,
        loader=loader,
        market_ids=market_ids,
        prediction_mode=args.prediction_mode,
        flag_rate_threshold=args.flag_rate_threshold,
        suspicion_threshold=args.suspicion_threshold,
        z_score_threshold=args.z_score_threshold,
        min_wallet_notional=args.min_wallet_notional,
        min_usd_amount=args.min_usd_amount,
        include_recidivism=args.include_recidivism,
        clustering_config=clustering_config,
        clustering_min_trade_size=args.clustering_min_trade_size,
        jump_anticipation_config=jump_anticipation_config,
        copytrade_fixed_size=args.copytrade_fixed_size,
        measure_memory=False,
        winning_outcomes_override=winning_outcomes,
        enable_layer2_attribution=args.enable_layer2_attribution,
        usdc_cache_db=args.usdc_cache,
        polygonscan_api_key=args.polygonscan_api_key,
    )

    participation_counts = build_reported_wallet_participation_counts(loader, cases)
    wallet_rows = build_reported_wallet_rows(
        curated_cases=cases,
        backtest_results=result.backtest_results,
        participation_counts=participation_counts,
        prediction_mode=args.prediction_mode,
        suspicion_threshold=args.suspicion_threshold,
        flag_rate_threshold=args.flag_rate_threshold,
    )
    taker_only_present = [
        row
        for row in wallet_rows
        if (
            bool(row["present_in_trade_data"])
            and int(row["participation_trade_count"]) > 0
            and int(row["backtest_trade_count"]) == 0
        )
    ]
    if taker_only_present:
        examples = ", ".join(
            f"{row['market_id']}/{row['wallet'][:10]}..."
            for row in taker_only_present[:5]
        )
        logging.warning(
            "%d reported wallet(s) have SQLite fills but 0 maker legs in the "
            "backtest (detector only processes maker). They cannot be flagged or "
            "classified until those rows are maker in the active DB. Examples: %s",
            len(taker_only_present),
            examples,
        )

    market_rows = summarize_market_rows(cases, wallet_rows)
    aggregate = summarize_aggregate(wallet_rows, market_rows)

    method_summaries: List[Dict[str, Any]] = []
    method_market_rows: List[Dict[str, Any]] = []
    if args.compare_sota:
        from experiments.curated_sota_common import (
            collect_sota_wallet_rows,
            get_train_loader_for_sota,
            resolve_train_markets_for_sota,
        )
        from experiments.curated_recall_zscore_metrics import enrich_method_summaries_with_zscore_metrics
        from experiments.sota_algorithms.curated_recall_flags import PerMarketFlagState

        maker_trade_counts = build_maker_trade_counts_by_market(
            loader,
            market_ids,
            min_usd_amount=args.min_usd_amount,
        )
        match_flag_rate: Optional[float] = None
        flagged_buy = sum(int(row.get("num_flags", 0)) for row in wallet_rows if row.get("has_alert"))
        buy_trades = sum(int(row.get("trade_count", 0)) for row in wallet_rows)
        if buy_trades > 0:
            match_flag_rate = flagged_buy / buy_trades
        flag_states: Dict[str, PerMarketFlagState] = {}
        train_loader, close_train_loader = get_train_loader_for_sota(args, loader)
        try:
            train_market_ids, train_winning_outcomes = resolve_train_markets_for_sota(
                train_loader=train_loader,
                args=args,
            )
            sota_wallet_rows, flag_states = collect_sota_wallet_rows(
                eval_loader=loader,
                train_loader=train_loader,
                curated_cases=cases,
                market_ids=market_ids,
                winning_outcomes=winning_outcomes,
                participation_counts=participation_counts,
                maker_trade_counts=maker_trade_counts,
                args=args,
                train_market_ids=train_market_ids,
                train_winning_outcomes=train_winning_outcomes,
                prediction_mode=args.prediction_mode,
                suspicion_threshold=args.suspicion_threshold,
                flag_rate_threshold=args.flag_rate_threshold,
                match_flag_rate=match_flag_rate,
            )
        finally:
            if close_train_loader:
                train_loader.close()
        wallet_rows = wallet_rows + sota_wallet_rows
        method_summaries = summarize_method_rows(cases, wallet_rows)
        method_market_rows = summarize_all_method_market_rows(cases, wallet_rows)
        method_summaries = enrich_method_summaries_with_zscore_metrics(
            method_summaries,
            eval_loader=loader,
            market_ids=market_ids,
            winning_outcomes=winning_outcomes,
            flag_states=flag_states,
            backtest_results=result.backtest_results,
            maker_trade_counts_by_market=maker_trade_counts,
            z_score_threshold=args.z_score_threshold,
            min_wallet_notional=args.min_wallet_notional,
            min_usd_amount=args.min_usd_amount,
            flag_rate_threshold=args.flag_rate_threshold,
            prediction_mode=args.prediction_mode,
            suspicion_threshold=args.suspicion_threshold,
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wallet_path = f"{args.output_dir}/curated_reported_wallets_{ts}.csv"
    market_path = f"{args.output_dir}/curated_reported_markets_{ts}.csv"
    method_market_path = f"{args.output_dir}/curated_reported_method_markets_{ts}.csv"
    summary_path = f"{args.output_dir}/curated_reported_summary_{ts}.json"
    method_path = f"{args.output_dir}/curated_reported_methods_{ts}.csv"

    pd.DataFrame(wallet_rows).to_csv(wallet_path, index=False)
    pd.DataFrame(market_rows).to_csv(market_path, index=False)
    if method_summaries:
        pd.DataFrame(method_summaries).to_csv(method_path, index=False)
    if method_market_rows:
        pd.DataFrame(method_market_rows).to_csv(method_market_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": str(config_path.resolve()),
                "prediction_mode": args.prediction_mode,
                "flag_rate_threshold": args.flag_rate_threshold,
                "suspicion_threshold": args.suspicion_threshold,
                "min_usd_amount": args.min_usd_amount,
                "include_recidivism": bool(args.include_recidivism),
                "clustering_enabled": clustering_config is not None,
                "jump_anticipation_enabled": jump_anticipation_config is not None,
                "enable_layer2_attribution": bool(args.enable_layer2_attribution),
                "market_ids_evaluated": market_ids,
                "skipped_markets": skipped_markets,
                "targetless_markets": targetless_markets,
                "not_local_yet_markets": not_local_yet_markets,
                "validation_warnings": validation_warnings,
                "resolution_meta": resolution_meta,
                "aggregate": aggregate,
                "method_summaries": method_summaries,
                "method_market_summaries": method_market_rows,
                "curated_cases": cases,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 88}")
    print("CURATED REPORTED-INSIDER RECALL")
    print(f"{'=' * 88}")
    print(f"Markets evaluated:             {len(market_ids):,}")
    print(f"Reported wallets total:        {aggregate['reported_wallets_total']:,}")
    print(f"Reported wallets present:      {aggregate['reported_wallets_present']:,}")
    print(
        "Classified recall (present):   "
        f"{aggregate['classified_reported_wallets']}/{aggregate['reported_wallets_present']} "
        f"({aggregate['classification_recall_present']:.2%})"
    )
    print(
        "Ever-flagged recall (present): "
        f"{aggregate['ever_flagged_reported_wallets']}/{aggregate['reported_wallets_present']} "
        f"({aggregate['ever_flagged_recall_present']:.2%})"
    )
    print(
        "Unique classified (present):   "
        f"{aggregate['unique_classified_reported_wallets']}/"
        f"{aggregate['unique_reported_wallets_present']} "
        f"({aggregate['unique_classification_recall_present']:.2%})"
    )
    print(
        "Unique ever-flagged (present): "
        f"{aggregate['unique_ever_flagged_reported_wallets']}/"
        f"{aggregate['unique_reported_wallets_present']} "
        f"({aggregate['unique_ever_flagged_recall_present']:.2%})"
    )
    if aggregate["reported_wallets_total"]:
        print(
            "Classified recall (total):     "
            f"{aggregate['classification_recall_total']:.2%}"
        )
        print(
            "Ever-flagged recall (total):   "
            f"{aggregate['ever_flagged_recall_total']:.2%}"
        )

    print("\nPer-market:")
    for row in market_rows:
        print(
            f"  {row['market_id']} {str(row['market_slug'])[:52]:52s} "
            f"classified={row['classified_reported_wallets']}/{row['reported_wallets_present']} "
            f"ever_flagged={row['ever_flagged_reported_wallets']}/{row['reported_wallets_present']} "
            f"targets_total={row['reported_wallets_total']}"
        )

    if aggregate["missing_wallets"]:
        print("\nMissing reported wallets:")
        for row in aggregate["missing_wallets"]:
            print(f"  {row['market_id']} {row['wallet']}")
    if skipped_markets:
        print(f"\nSkipped markets: {skipped_markets}")
    if targetless_markets:
        print(f"\nTargetless scaffold markets not backtested: {targetless_markets}")
    if not_local_yet_markets:
        print(f"\nNot in active data directory yet: {not_local_yet_markets}")
    if aggregate["reported_wallets_total"] == 0:
        print(
            "\nNo full reported wallet addresses are currently counted. "
            "Populate CURATED_CASES.reported_wallets with verified 0x addresses to score recall."
        )

    if method_summaries:
        from experiments.curated_sota_common import (
            _print_method_market_sections,
            _print_method_table,
        )

        _print_method_table(method_summaries)
        _print_method_market_sections(method_market_rows)

    print("\nSaved:")
    print(f"  - {wallet_path}")
    print(f"  - {market_path}")
    if method_summaries:
        print(f"  - {method_path}")
    if method_market_rows:
        print(f"  - {method_market_path}")
    print(f"  - {summary_path}")
    loader.close()


if __name__ == "__main__":
    main()
