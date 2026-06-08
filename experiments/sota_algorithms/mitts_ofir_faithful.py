"""
Faithful reconstruction of the Mitts & Ofir (2026) insider screen.

Mitts & Ofir, "From Iran to Taylor Swift" (SSRN 6426778). Their screen operates
at the (wallet, market) PAIR level, combining FIVE standardised signals into a
continuous composite anomaly score and flagging the upper tail. Buy-side only;
positions below $500 are excluded. The screen is retrospective — profitability
is only known at resolution.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats

from backtesting.data_loader import HistoricalDataLoader
from backtesting.trade_event_study import _compute_resolution_return, _cohens_d
from experiments.sota_algorithms.common import (
    build_wallet_insider_labels,
    copytrade_trade_summary,
    get_market_resolution_timestamp_ms,
    summarize_pooled_returns,
    wallet_classification_metrics,
    wallet_flagged_pnl_from_wallet_data,
)
from models import Trade

class _Welford:
    """Running mean/variance for causal (no-lookahead) standardisation."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def zscore(self, x: float) -> float:
        """z-score of x against the values seen so far (strictly prior)."""
        if self.n < 2:
            return 0.0
        std = math.sqrt(self.m2 / (self.n - 1))
        if std < 1e-12:
            return 0.0
        return (x - self.mean) / std

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)


def _zscore_array(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _largest_price_jump_ts(trades: List[Trade], winning: int) -> Optional[int]:
    """
    Timestamp of the largest interval increase in the winning outcome's traded
    price across the market's full trade history (retrospective S4 anchor).
    """
    pts = sorted(
        (
            (int(t.timestamp_ms), float(t.price))
            for t in trades
            if int(t.outcome_index) == int(winning)
        ),
        key=lambda x: x[0],
    )
    if len(pts) < 2:
        return None
    best_inc: Optional[float] = None
    best_ts: Optional[int] = None
    for (_, p0), (t1, p1) in zip(pts, pts[1:]):
        inc = p1 - p0
        if best_inc is None or inc > best_inc:
            best_inc = inc
            best_ts = t1
    return best_ts

def _build_pairs_all_markets(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes: Dict[int, Optional[int]],
    min_usd_amount: Optional[float],
    min_wallet_notional: float,
) -> Tuple[List[Dict], Dict[int, List[Trade]], Dict[int, Optional[int]], int]:
    """
    Aggregate each (wallet, market) pair over the wallet's BUY trades in that market
    """
    pairs: List[Dict] = []
    market_trades: Dict[int, List[Trade]] = {}
    resolution_ts_by_market: Dict[int, Optional[int]] = {}
    total_buy_eval = 0

    for market_id in market_ids:
        winning = winning_outcomes.get(market_id)
        try:
            trades = loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=min_usd_amount,
                use_cache=False,
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)
        market_trades[market_id] = trades
        resolution_ts_by_market[market_id] = get_market_resolution_timestamp_ms(
            loader, market_id
        )
        if winning is None:
            continue

        buys_by_wallet: Dict[str, List[Trade]] = defaultdict(list)
        for t in trades:
            if t.side.upper() != "BUY":
                continue
            total_buy_eval += 1
            buys_by_wallet[t.wallet].append(t)

        for wallet, buys in buys_by_wallet.items():
            buy_notional = sum(float(t.notional_usdc) for t in buys)
            if buy_notional < min_wallet_notional:
                continue
            outcome_notional: Dict[int, float] = defaultdict(float)
            for t in buys:
                outcome_notional[int(t.outcome_index)] += float(t.notional_usdc)
            pairs.append({
                "market_id": int(market_id),
                "wallet": wallet,
                "buy_notional": buy_notional,
                "outcome_notional": dict(outcome_notional),
                "buy_trades": sorted(buys, key=lambda t: int(t.timestamp_ms)),
                "winning": int(winning),
                "n_buy_trades": len(buys),
            })

    return pairs, market_trades, resolution_ts_by_market, total_buy_eval

def _score_pairs_retrospective(
    pairs: List[Dict],
    market_trades: Dict[int, List[Trade]],
) -> None:
    """Equal-weight sum of full-population z-scores over all five signals."""
    if not pairs:
        return

    winning_by_market = {p["market_id"]: p["winning"] for p in pairs}
    jump_ts = {
        mid: _largest_price_jump_ts(market_trades.get(mid, []), w)
        for mid, w in winning_by_market.items()
    }

    s1: List[float] = []  # cross-sectional bet size
    s3: List[float] = []  # profitability
    s4: List[float] = []  # pre-event timing (resolved price jump)
    s5: List[float] = []  # directional concentration

    for p in pairs:
        bn = p["buy_notional"]
        w = p["winning"]
        s1.append(math.log1p(bn))

        win_tokens = sum(
            float(t.size_tokens)
            for t in p["buy_trades"]
            if int(t.outcome_index) == w
        )
        s3.append((win_tokens - bn) / bn if bn > 1e-12 else 0.0)

        tj = jump_ts.get(p["market_id"])
        if tj is None:
            s4.append(0.0)
        else:
            before = sum(
                float(t.notional_usdc)
                for t in p["buy_trades"]
                if int(t.timestamp_ms) < tj
            )
            s4.append(before / bn if bn > 1e-12 else 0.0)

        s5.append(max(p["outcome_notional"].values()) / bn if bn > 1e-12 else 0.0)

    # S2: within-trader bet size, standardised against THIS wallet's distribution
    # of per-market buy notionals (0 if the wallet has fewer than 2 markets).
    s2 = [0.0] * len(pairs)
    idxs_by_wallet: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        idxs_by_wallet[p["wallet"]].append(i)
    for idxs in idxs_by_wallet.values():
        if len(idxs) < 2:
            continue
        vals = np.array([pairs[i]["buy_notional"] for i in idxs])
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        if std < 1e-12:
            continue
        for i in idxs:
            s2[i] = (pairs[i]["buy_notional"] - mean) / std

    composite = (
        _zscore_array(s1)
        + _zscore_array(s2)
        + _zscore_array(s3)
        + _zscore_array(s4)
        + _zscore_array(s5)
    )
    for i, p in enumerate(pairs):
        p["composite"] = float(composite[i])


def _score_pairs_causal(
    pairs: List[Dict],
    resolution_ts_by_market: Dict[int, Optional[int]],
    low_price_threshold: float,
    min_wallet_notional: float,
) -> List[Dict]:
    """
    Four-signal composite (no profitability) using only data observable at each
    pair's scoring cutoff — the wallet's last BUY in that market before
    resolution. S1/S2/S4/S5 are running z-scores over strictly-prior pairs.

    Returns the subset of pairs that have at least one pre-resolution BUY and a
    pre-resolution buy notional above the $500 floor.
    """
    scored: List[Dict] = []
    for p in pairs:
        res_ts = resolution_ts_by_market.get(p["market_id"])
        if res_ts is not None:
            pre = [t for t in p["buy_trades"] if int(t.timestamp_ms) < res_ts]
        else:
            pre = list(p["buy_trades"])
        if not pre:
            continue
        bn = sum(float(t.notional_usdc) for t in pre)
        if bn < min_wallet_notional:
            continue

        cutoff = max(int(t.timestamp_ms) for t in pre)

        # S4 causal proxy: cheap accumulation that is also early
        low_notional = sum(
            float(t.notional_usdc) for t in pre
            if float(t.price) <= low_price_threshold
        )
        low_share = low_notional / bn
        if res_ts is not None:
            hours = sum(
                ((res_ts - int(t.timestamp_ms)) / 3.6e6) * float(t.notional_usdc)
                for t in pre
            ) / bn
            hours = max(hours, 0.0)
        else:
            hours = 0.0
        s4_raw = low_share * hours

        oc: Dict[int, float] = defaultdict(float)
        for t in pre:
            oc[int(t.outcome_index)] += float(t.notional_usdc)
        s5_raw = max(oc.values()) / bn

        p["_causal"] = {
            "cutoff": cutoff,
            "bn": bn,
            "s4_raw": s4_raw,
            "s5_raw": s5_raw,
        }
        scored.append(p)

    scored.sort(key=lambda p: p["_causal"]["cutoff"])

    w_s1 = _Welford()
    w_s4 = _Welford()
    w_s5 = _Welford()
    w_s2: Dict[str, _Welford] = defaultdict(_Welford)

    for p in scored:
        c = p["_causal"]
        bn = c["bn"]
        wallet = p["wallet"]
        log_bn = math.log1p(bn)

        z1 = w_s1.zscore(log_bn)
        z2 = w_s2[wallet].zscore(bn)
        z4 = w_s4.zscore(c["s4_raw"])
        z5 = w_s5.zscore(c["s5_raw"])
        p["composite"] = float(z1 + z2 + z4 + z5)

        w_s1.update(log_bn)
        w_s2[wallet].update(bn)
        w_s4.update(c["s4_raw"])
        w_s5.update(c["s5_raw"])

    return scored


def _select_flagged_pairs(
    pairs: List[Dict],
    flag_percentile: float,
    match_flag_rate: Optional[float],
    total_buy_eval: int,
) -> Tuple[List[Dict], str]:
    """
    (a) natural fixed percentile: flag the top flag_percentile% of pairs by
        composite score; or
    (b) flag-rate matched: flag the highest-composite pairs until their BUY
        trades reach match_flag_rate * total_buy_eval.
    """
    if not pairs:
        return [], "empty"

    ordered = sorted(pairs, key=lambda p: p["composite"], reverse=True)

    if match_flag_rate is not None:
        target = float(match_flag_rate) * float(total_buy_eval)
        flagged: List[Dict] = []
        cum = 0
        for p in ordered:
            if cum >= target:
                break
            flagged.append(p)
            cum += int(p["n_buy_trades"])
        return flagged, "flag_rate_matched"

    n_flag = max(1, int(round(flag_percentile / 100.0 * len(ordered))))
    return ordered[:n_flag], "natural_percentile"

def _event_study_and_metrics(
    market_ids: List[int],
    market_trades: Dict[int, List[Trade]],
    winning_outcomes: Dict[int, Optional[int]],
    flagged_wallets_by_market: Dict[int, Set[str]],
    all_entries: List[Tuple[Trade, int]],
    z_score_threshold: float,
    min_wallet_notional: float,
) -> Tuple[Dict, float]:
    flagged_returns_all: List[float] = []
    flagged_notionals_all: List[float] = []
    unflagged_returns_all: List[float] = []
    per_market_flagged: Dict[int, List[float]] = defaultdict(list)
    per_market_unflagged: Dict[int, List[float]] = defaultdict(list)
    n_buy_eval = 0
    n_flagged_buy = 0
    n_unflagged_buy = 0

    for market_id in market_ids:
        winning = winning_outcomes.get(market_id)
        if winning is None:
            continue
        flagged_wallets = flagged_wallets_by_market.get(market_id, set())
        for trade in market_trades.get(market_id, []):
            if trade.side.upper() != "BUY":
                continue
            n_buy_eval += 1
            ret = _compute_resolution_return(trade, winning)
            if trade.wallet in flagged_wallets:
                flagged_returns_all.append(ret)
                flagged_notionals_all.append(float(trade.notional_usdc))
                per_market_flagged[market_id].append(ret)
                n_flagged_buy += 1
            else:
                unflagged_returns_all.append(ret)
                per_market_unflagged[market_id].append(ret)
                n_unflagged_buy += 1

    pooled_stats = summarize_pooled_returns(flagged_returns_all, unflagged_returns_all)

    per_market_d: List[float] = []
    sig_welch = 0
    n_markets_eval = 0
    for market_id in market_ids:
        fr = np.array(per_market_flagged.get(market_id, []))
        ur = np.array(per_market_unflagged.get(market_id, []))
        if len(fr) < 2 or len(ur) < 2:
            continue
        per_market_d.append(_cohens_d(fr, ur))
        n_markets_eval += 1
        try:
            _, p_val = scipy_stats.ttest_ind(fr, ur, equal_var=False)
            if p_val < 0.05:
                sig_welch += 1
        except Exception:
            pass

    mean_cohens_d = float(np.mean(per_market_d)) if per_market_d else 0.0
    actual_flag_rate = n_flagged_buy / max(n_buy_eval, 1)

    wallet_data = build_wallet_insider_labels(
        all_entries, winning_outcomes, z_score_threshold, min_wallet_notional
    )
    wallet_pnl_stats = wallet_flagged_pnl_from_wallet_data(
        wallet_data, flagged_wallets_by_market
    )
    copytrade_stats = copytrade_trade_summary(flagged_returns_all, flagged_notionals_all)
    wallet_metrics = wallet_classification_metrics(wallet_data, flagged_wallets_by_market)

    common = {
        "flagged_trades": n_flagged_buy,
        "unflagged_trades": n_unflagged_buy,
        "flagged_mean_return": pooled_stats["flagged_mean_return"],
        "unflagged_mean_return": pooled_stats["unflagged_mean_return"],
        "mean_return_diff": pooled_stats["mean_return_diff"],
        "mean_cohens_d": mean_cohens_d,
        "sig_welch_p05": sig_welch,
        "n_markets": n_markets_eval,
        "tp": wallet_metrics["tp"],
        "fp": wallet_metrics["fp"],
        "fn": wallet_metrics["fn"],
        "flagged_avg_return": wallet_metrics["flagged_avg_return"],
        "tp_avg_return": wallet_metrics["tp_avg_return"],
        "fp_avg_return": wallet_metrics["fp_avg_return"],
        "trades_per_second": 0.0,
        "det_p95_us": 0.0,
        "wall_clock_s": 0.0,
        "flagged_wallet_mean_net_pnl": wallet_pnl_stats["flagged_wallet_mean_net_pnl"],
        "flagged_wallet_median_net_pnl": wallet_pnl_stats["flagged_wallet_median_net_pnl"],
        "copytrade_total_flagged_buys": copytrade_stats["copytrade_total_flagged_buys"],
        "copytrade_total_capital_deployed": copytrade_stats["copytrade_total_capital_deployed"],
        "copytrade_total_pnl": copytrade_stats["copytrade_total_pnl"],
        "copytrade_portfolio_roi": copytrade_stats["copytrade_portfolio_roi"],
        "copytrade_win_rate": copytrade_stats["copytrade_win_rate"],
        "copytrade_mean_trade_return": copytrade_stats["copytrade_mean_trade_return"],
        "copytrade_median_trade_return": copytrade_stats["copytrade_median_trade_return"],
    }
    return common, actual_flag_rate


def _flagged_wallets_from_pairs(
    market_ids: List[int],
    flagged_pairs: List[Dict],
) -> Dict[int, Set[str]]:
    flagged_wallets_by_market: Dict[int, Set[str]] = defaultdict(set)
    for market_id in market_ids:
        flagged_wallets_by_market[market_id] = set()
    for p in flagged_pairs:
        flagged_wallets_by_market[p["market_id"]].add(p["wallet"])
    return flagged_wallets_by_market


_RETRO_SIGNALS = [
    "S1_cross_sectional_size",
    "S2_within_trader_size",
    "S3_profitability",
    "S4_pre_event_timing",
    "S5_directional_concentration",
]
_CAUSAL_SIGNALS = [
    "S1_cross_sectional_size",
    "S2_within_trader_size",
    "S4_pre_event_timing",
    "S5_directional_concentration",
]


def run_mitts_ofir_faithful_retrospective(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    all_entries: List[Tuple[Trade, int]],
    winning_outcomes: Dict[int, Optional[int]],
    *,
    flag_percentile: float = 5.0,
    match_flag_rate: Optional[float] = None,
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
    min_usd_amount: Optional[float] = None,
) -> Dict:
    start = time.time()

    pairs, market_trades, _resolution_ts, total_buy_eval = _build_pairs_all_markets(
        loader, market_ids, winning_outcomes, min_usd_amount, min_wallet_notional
    )
    _score_pairs_retrospective(pairs, market_trades)
    flagged_pairs, flag_mode = _select_flagged_pairs(
        pairs, flag_percentile, match_flag_rate, total_buy_eval
    )
    flagged_wallets_by_market = _flagged_wallets_from_pairs(market_ids, flagged_pairs)

    common, actual_flag_rate = _event_study_and_metrics(
        market_ids, market_trades, winning_outcomes, flagged_wallets_by_market,
        all_entries, z_score_threshold, min_wallet_notional,
    )
    elapsed = time.time() - start

    logging.info(
        f"mitts_ofir_retrospective: {common['flagged_trades']:,} flagged BUY "
        f"trades ({actual_flag_rate:.2%}), {len(flagged_pairs):,}/{len(pairs):,} pairs, "
        f"TP={common['tp']}, FP={common['fp']}, FN={common['fn']}, wall={elapsed:.1f}s"
    )

    return {
        "baseline": "mitts_ofir_retrospective",
        **common,
        "num_flags": int(len(flagged_pairs)),
        "wall_clock_s": elapsed,
        "min_usd_amount": min_usd_amount,
        "match_flag_rate": match_flag_rate,
        "mo_flag_mode": flag_mode,
        "mo_flag_percentile": float(flag_percentile),
        "mo_actual_flag_rate": actual_flag_rate,
        "mo_n_pairs_scored": len(pairs),
        "mo_n_flagged_pairs": len(flagged_pairs),
        "mo_variant": "retrospective",
        "mo_signals_used": _RETRO_SIGNALS,
        "mo_profitability_included": True,
        "mo_stat_scope": "full_population",
        "mo_timing_anchor": "resolved_price_jump",
        "mo_composite_rule": "equal_weight_standardised_sum_top_percentile",
        "mo_definition_match": "faithful_five_signal_pair_level_composite (weights not public)",
        "deployable_live": False,
    }


def run_mitts_ofir_faithful_causal(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    all_entries: List[Tuple[Trade, int]],
    winning_outcomes: Dict[int, Optional[int]],
    *,
    flag_percentile: float = 5.0,
    match_flag_rate: Optional[float] = None,
    low_price_threshold: float = 0.15,
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
    min_usd_amount: Optional[float] = None,
) -> Dict:
    start = time.time()

    pairs, market_trades, resolution_ts_by_market, total_buy_eval = _build_pairs_all_markets(
        loader, market_ids, winning_outcomes, min_usd_amount, min_wallet_notional
    )
    scored_pairs = _score_pairs_causal(
        pairs, resolution_ts_by_market, low_price_threshold, min_wallet_notional
    )
    flagged_pairs, flag_mode = _select_flagged_pairs(
        scored_pairs, flag_percentile, match_flag_rate, total_buy_eval
    )
    flagged_wallets_by_market = _flagged_wallets_from_pairs(market_ids, flagged_pairs)

    common, actual_flag_rate = _event_study_and_metrics(
        market_ids, market_trades, winning_outcomes, flagged_wallets_by_market,
        all_entries, z_score_threshold, min_wallet_notional,
    )
    elapsed = time.time() - start

    logging.info(
        f"mitts_ofir_causal: {common['flagged_trades']:,} flagged BUY "
        f"trades ({actual_flag_rate:.2%}), {len(flagged_pairs):,}/{len(scored_pairs):,} pairs, "
        f"TP={common['tp']}, FP={common['fp']}, FN={common['fn']}, wall={elapsed:.1f}s"
    )

    return {
        "baseline": "mitts_ofir_causal",
        **common,
        "num_flags": int(len(flagged_pairs)),
        "wall_clock_s": elapsed,
        "min_usd_amount": min_usd_amount,
        "match_flag_rate": match_flag_rate,
        "mo_flag_mode": flag_mode,
        "mo_flag_percentile": float(flag_percentile),
        "mo_actual_flag_rate": actual_flag_rate,
        "mo_low_price_threshold": float(low_price_threshold),
        "mo_n_pairs_scored": len(scored_pairs),
        "mo_n_flagged_pairs": len(flagged_pairs),
        "mo_variant": "causal",
        "mo_signals_used": _CAUSAL_SIGNALS,
        "mo_profitability_included": False,
        "mo_stat_scope": "running_causal",
        "mo_timing_anchor": "causal_entry_price_and_horizon",
        "mo_composite_rule": "equal_weight_standardised_sum_top_percentile",
        "mo_definition_match": "faithful_five_signal_pair_level_composite (weights not public)",
        "deployable_live": True,
    }
