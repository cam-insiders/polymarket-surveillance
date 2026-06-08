"""
Faithful, NON-CAUSAL CONSOB / Ravagnani et al. (2024) reconstruction screen.

Ravagnani, Lillo, Deriu, Mazzarisi, Medda, Russo, "Dimensionality reduction
techniques to support insider trading detection," CONSOB Fintech Series No. 12 /
arXiv:2403.00707.

This reproduces the paper's single-asset / single-PSE detector exactly, processing
ONE MARKET at a time. It is RETROSPECTIVE BY DESIGN (deployable_live = False): it
uses the resolved winning outcome and the full windowed price path. The method is
event-anchored — its core conditions are defined relative to the price-sensitive
event (PSE) — so there is no causal variant, and none is provided.

Only the LINEAR (PCA) case of the paper's method is implemented; the paper's
autoencoder variant is out of scope.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

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

MS_PER_HOUR = 3_600_000

NComponents = Union[int, float]


def _utc_bucket_index(timestamp_ms: int, bucket_hours: int) -> int:
    """Fixed-grid bucket index (6h by default). Distinct from consob_pca's daily index."""
    return int(timestamp_ms) // (int(bucket_hours) * MS_PER_HOUR)


def parse_n_components(value: Union[str, int, float]) -> NComponents:
    """A value in (0, 1) is a target explained-variance ratio; otherwise an int K."""
    f = float(value)
    if 0.0 < f < 1.0:
        return f
    return int(round(f))


# ---------------------------------------------------------------------------
# Trajectory matrix (paper Eqs. 1-2)
# ---------------------------------------------------------------------------

def _build_consob_faithful_matrix(
    trades: List[Trade],
    winning_outcome: int,
    bucket_hours: int,
    min_wallet_notional: float,
) -> Optional[Dict]:
    """
    Build L-infinity-normalised cumulative winning-outcome position trajectories
    over the market's own contiguous 6h bucket grid.

    Returns None if the market has no trades, fewer than 2 buckets, or fewer than
    2 eligible non-constant wallets.
    """
    if not trades:
        return None

    bucket_ms = int(bucket_hours) * MS_PER_HOUR

    total_notional: Dict[str, float] = defaultdict(float)
    all_ts: List[int] = []
    for t in trades:
        total_notional[t.wallet] += float(t.notional_usdc)
        all_ts.append(int(t.timestamp_ms))

    min_ts, max_ts = min(all_ts), max(all_ts)
    min_bucket = min_ts // bucket_ms
    max_bucket = max_ts // bucket_ms
    grid = list(range(min_bucket, max_bucket + 1))
    T = len(grid)
    if T < 2:
        return None
    col = {b: j for j, b in enumerate(grid)}

    wallet_delta: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(T, dtype=float))
    for t in trades:
        if int(t.outcome_index) != int(winning_outcome):
            continue
        j = col.get(int(t.timestamp_ms) // bucket_ms)
        if j is None:
            continue
        size = float(t.size_tokens)
        if t.side.upper() == "BUY":
            wallet_delta[t.wallet][j] += size
        else:
            wallet_delta[t.wallet][j] -= size

    rows: List[np.ndarray] = []
    wallet_ids: List[str] = []
    d_active: List[int] = []
    for wallet in sorted(wallet_delta.keys()):
        if total_notional[wallet] < float(min_wallet_notional):
            continue
        delta = wallet_delta[wallet]
        cum = np.cumsum(delta)
        linf = float(np.max(np.abs(cum)))
        if linf < 1e-12:
            # constant position: non-trader or strict round-tripper (Eq. 2 discard)
            continue
        rows.append(cum / linf)
        wallet_ids.append(wallet)
        d_active.append(int(np.sum(np.abs(delta) > 1e-12)))

    if len(wallet_ids) < 2:
        return None

    return {
        "X": np.stack(rows, axis=0),
        "wallet_ids": wallet_ids,
        "grid": grid,
        "col": col,
        "T": T,
        "d_active": np.asarray(d_active, dtype=int),
        "min_ts": min_ts,
        "max_ts": max_ts,
    }


def _largest_price_jump_bucket(
    trades: List[Trade],
    winning_outcome: int,
    bucket_hours: int,
) -> Optional[int]:
    """Bucket containing the largest single-interval increase in the winning
    outcome's traded price across the windowed history."""
    bucket_ms = int(bucket_hours) * MS_PER_HOUR
    pts = sorted(
        (
            (int(t.timestamp_ms), float(t.price))
            for t in trades
            if int(t.outcome_index) == int(winning_outcome)
        ),
        key=lambda x: x[0],
    )
    if len(pts) < 2:
        return None
    best_inc = 0.0
    best_ts: Optional[int] = None
    for (_, p0), (t1, p1) in zip(pts, pts[1:]):
        inc = p1 - p0
        if inc > best_inc:
            best_inc = inc
            best_ts = t1
    if best_ts is None:
        # no positive price increase: no price-sensitive jump, use resolution
        return None
    return best_ts // bucket_ms


# ---------------------------------------------------------------------------
# Threshold rules
# ---------------------------------------------------------------------------

def _estimate_eps_theta(
    s_star: np.ndarray,
    min_wallets_for_kde: int,
    percentile_fallback: float,
) -> Tuple[float, str]:
    """
    eps_theta = local minimum (trough) between the two largest modes of the s*
    distribution, estimated per market via a Gaussian KDE. Falls back to a
    percentile of s* for small-N or non-bimodal/degenerate distributions.
    """
    vals = np.asarray(s_star, dtype=float)
    fallback = (float(np.percentile(vals, float(percentile_fallback))), "percentile_fallback")

    if vals.size < int(min_wallets_for_kde):
        return fallback
    if float(np.ptp(vals)) < 1e-12:
        return fallback

    try:
        kde = scipy_stats.gaussian_kde(vals)
    except (np.linalg.LinAlgError, ValueError):
        return fallback

    eval_grid = np.linspace(float(vals.min()), float(vals.max()), 256)
    density = kde(eval_grid)
    peaks, _ = find_peaks(density)
    if peaks.size < 2:
        return fallback

    top2 = peaks[np.argsort(density[peaks])[-2:]]
    lo, hi = int(min(top2)), int(max(top2))
    if hi <= lo:
        return fallback
    trough_offset = int(np.argmin(density[lo : hi + 1]))
    return float(eval_grid[lo + trough_offset]), "bimodal_trough"


# ---------------------------------------------------------------------------
# Per-market core (paper Eq. 3)
# ---------------------------------------------------------------------------

def run_consob_faithful_market(
    trades: List[Trade],
    winning_outcome: int,
    resolution_ts: Optional[int],
    *,
    bucket_hours: int = 6,
    investigation_hours: int = 24,
    d_theta: int = 3,
    n_components: NComponents = 3,
    min_wallet_notional: float = 500.0,
    min_wallets_for_kde: int = 8,
    percentile_fallback: float = 90.0,
) -> Tuple[Set[str], Dict]:
    """
    Score one market self-contained and return (flagged_wallets, diagnostics).

    Wallets are returned with their raw ids (as on the trades), matching
    build_wallet_insider_labels for the baseline event study.
    """
    diagnostics: Dict = {
        "n_eligible": 0,
        "T": 0,
        "K_eff": 0,
        "eps_theta": 0.0,
        "n_theta": 0.0,
        "threshold_rule": None,
        "pse_anchor": None,
        "short_market": False,
        "n_flagged": 0,
        "skipped": None,
        "ranking": [],
    }

    built = _build_consob_faithful_matrix(
        trades, winning_outcome, bucket_hours, min_wallet_notional
    )
    if built is None:
        diagnostics["skipped"] = "no_matrix"
        return set(), diagnostics

    X = built["X"]
    wallet_ids = built["wallet_ids"]
    col = built["col"]
    grid = built["grid"]
    T = built["T"]
    d_active = built["d_active"]
    N = X.shape[0]
    diagnostics["n_eligible"] = N
    diagnostics["T"] = T

    # --- PSE anchor + investigation window --------------------------------
    bucket_ms = int(bucket_hours) * MS_PER_HOUR
    pse_bucket = _largest_price_jump_bucket(trades, winning_outcome, bucket_hours)
    pse_anchor = "largest_price_jump"
    if pse_bucket is None:
        if resolution_ts is None:
            diagnostics["skipped"] = "no_pse_anchor"
            return set(), diagnostics
        pse_bucket = int(resolution_ts) // bucket_ms
        pse_anchor = "resolution"
    diagnostics["pse_anchor"] = pse_anchor

    pse_col = col.get(pse_bucket)
    if pse_col is None:
        # resolution can land outside the trade grid; clamp into range
        pse_col = T - 1 if pse_bucket > grid[-1] else 0
    pse_col = int(min(max(pse_col, 0), T - 1))

    span_hours = (built["max_ts"] - built["min_ts"]) / MS_PER_HOUR
    if span_hours < 48.0:
        inv_len = max(1, int(np.ceil(T * 0.5)))
        diagnostics["short_market"] = True
    else:
        inv_len = max(1, int(round(float(investigation_hours) / float(bucket_hours))))

    delta_start = max(0, pse_col - inv_len + 1)
    n_ref = delta_start
    n_inv = pse_col - delta_start + 1
    if n_ref < 2 or n_inv < 1:
        # Delta would swallow the trajectory, leaving condition (b) vacuous.
        diagnostics["skipped"] = "insufficient_reference_or_investigation_buckets"
        return set(), diagnostics

    # --- PCA reconstruction (paper Eq. PCA case) --------------------------
    max_k = min(N - 1, T - 1)
    if max_k < 1:
        diagnostics["skipped"] = "K_eff_lt_1"
        return set(), diagnostics

    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        pca = PCA(n_components=float(n_components), random_state=42)
    else:
        K_eff = min(int(n_components), max_k)
        if K_eff < 1:
            diagnostics["skipped"] = "K_eff_lt_1"
            return set(), diagnostics
        pca = PCA(n_components=K_eff, random_state=42)

    Z = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)
    diagnostics["K_eff"] = int(pca.n_components_)

    # --- Per-bucket error + anomaly score (paper: max per-bucket abs) -----
    eps = np.abs(X - X_hat)
    b_star = np.argmax(eps, axis=1)
    s_star = eps[np.arange(N), b_star]

    n_b = np.zeros(T, dtype=int)
    for b in b_star:
        n_b[int(b)] += 1

    eps_theta, threshold_rule = _estimate_eps_theta(
        s_star, min_wallets_for_kde, percentile_fallback
    )
    n_theta = float(np.percentile(n_b, 90.0))
    diagnostics["eps_theta"] = float(eps_theta)
    diagnostics["n_theta"] = n_theta
    diagnostics["threshold_rule"] = threshold_rule

    # --- Four-condition criterion (paper Eq. 3) ---------------------------
    flagged: Set[str] = set()
    flagged_idx: List[int] = []
    for i in range(N):
        bi = int(b_star[i])
        cond_a = s_star[i] >= eps_theta
        cond_b = delta_start <= bi <= pse_col
        cond_c = (d_active[i] <= int(d_theta)) or (n_b[bi] < n_theta)
        cond_d = (float(X[i, pse_col]) - float(X[i, 0])) > 0.5
        if cond_a and cond_b and cond_c and cond_d:
            flagged.add(wallet_ids[i])
            flagged_idx.append(i)

    diagnostics["n_flagged"] = len(flagged)

    # --- Ranking (faithful diagnostic; does NOT change the flag set) ------
    if flagged_idx:
        s_sub = s_star[flagged_idx]
        nbar = np.array(
            [n_b[int(b_star[i])] * (1.0 if d_active[i] > int(d_theta) else 0.0)
             for i in flagged_idx],
            dtype=float,
        )

        def _unit(arr: np.ndarray) -> np.ndarray:
            rng = float(np.ptp(arr))
            if rng < 1e-12:
                return np.zeros_like(arr)
            return (arr - float(arr.min())) / rng

        s_scaled = _unit(s_sub)
        n_scaled = _unit(nbar)
        dist = np.sqrt((s_scaled - 1.0) ** 2 + (n_scaled - 0.0) ** 2)
        order = np.argsort(dist)
        diagnostics["ranking"] = [wallet_ids[flagged_idx[j]] for j in order]

    return flagged, diagnostics


# ---------------------------------------------------------------------------
# Event study + metrics (mirrors the other baselines)
# ---------------------------------------------------------------------------

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
            if int(trade.outcome_index) != int(winning):
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


def _load_market_trades(
    loader: HistoricalDataLoader,
    market_id: int,
    min_usd_amount: Optional[float],
) -> List[Trade]:
    try:
        return list(
            loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=min_usd_amount,
                use_cache=False,
            )
        )
    except TypeError:
        return list(loader.get_trades_for_market(market_id))


# ---------------------------------------------------------------------------
# Public baseline + curated-recall collector
# ---------------------------------------------------------------------------

def run_consob_pca_faithful_baseline(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    all_entries: List[Tuple[Trade, int]],
    winning_outcomes: Dict[int, Optional[int]],
    *,
    bucket_hours: int = 6,
    investigation_hours: int = 24,
    d_theta: int = 3,
    n_components: NComponents = 3,
    min_wallets_for_kde: int = 8,
    percentile_fallback: float = 90.0,
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
    min_usd_amount: Optional[float] = None,
) -> Dict:
    """Faithful CONSOB baseline, scored per-market and self-contained."""
    start = time.time()

    market_trades: Dict[int, List[Trade]] = {}
    flagged_wallets_by_market: Dict[int, Set[str]] = {}
    K_eff_values: List[int] = []
    T_values: List[int] = []
    threshold_rule_counts: Dict[str, int] = defaultdict(int)
    pse_anchor_counts: Dict[str, int] = defaultdict(int)
    markets_scored = 0

    for market_id in market_ids:
        trades = _load_market_trades(loader, market_id, min_usd_amount)
        market_trades[market_id] = trades
        winning = winning_outcomes.get(market_id)
        if winning is None:
            flagged_wallets_by_market[market_id] = set()
            continue
        resolution_ts = get_market_resolution_timestamp_ms(loader, market_id)
        flagged, diag = run_consob_faithful_market(
            trades,
            int(winning),
            resolution_ts,
            bucket_hours=bucket_hours,
            investigation_hours=investigation_hours,
            d_theta=d_theta,
            n_components=n_components,
            min_wallet_notional=min_wallet_notional,
            min_wallets_for_kde=min_wallets_for_kde,
            percentile_fallback=percentile_fallback,
        )
        flagged_wallets_by_market[market_id] = flagged
        if diag["skipped"] is None:
            markets_scored += 1
            T_values.append(int(diag["T"]))
            K_eff_values.append(int(diag["K_eff"]))
            if diag["threshold_rule"]:
                threshold_rule_counts[diag["threshold_rule"]] += 1
            if diag["pse_anchor"]:
                pse_anchor_counts[diag["pse_anchor"]] += 1

    common, actual_flag_rate = _event_study_and_metrics(
        market_ids, market_trades, winning_outcomes, flagged_wallets_by_market,
        all_entries, z_score_threshold, min_wallet_notional,
    )
    elapsed = time.time() - start

    def _majority(counts: Dict[str, int], default: str) -> str:
        return max(counts, key=counts.get) if counts else default

    n_rule = sum(threshold_rule_counts.values()) or 1
    n_anchor = sum(pse_anchor_counts.values()) or 1
    bimodal_frac = threshold_rule_counts.get("bimodal_trough", 0) / n_rule
    percentile_frac = threshold_rule_counts.get("percentile_fallback", 0) / n_rule
    n_flagged_wallets_total = int(sum(len(s) for s in flagged_wallets_by_market.values()))

    is_variance_ratio = isinstance(n_components, float) and 0.0 < n_components < 1.0
    k_selection = (
        f"variance_ratio_{n_components}" if is_variance_ratio else str(int(n_components))
    )

    logging.info(
        "consob_pca: %s flagged BUY trades (%.2f%%), TP=%s, FP=%s, FN=%s, "
        "markets_scored=%s, bimodal=%.0f%%/percentile=%.0f%%, wall=%.1fs",
        f"{common['flagged_trades']:,}", actual_flag_rate * 100,
        common["tp"], common["fp"], common["fn"], markets_scored,
        bimodal_frac * 100, percentile_frac * 100, elapsed,
    )

    return {
        "baseline": "consob_pca",
        **common,
        "num_flags": n_flagged_wallets_total,
        "wall_clock_s": elapsed,
        "min_usd_amount": min_usd_amount,
        "consob_flagged_wallets": n_flagged_wallets_total,
        "consob_flag_rate_buys": actual_flag_rate,
        "consob_markets_scored": markets_scored,
        "consob_mean_T_buckets": float(np.mean(T_values)) if T_values else 0.0,
        "consob_mean_K_eff": float(np.mean(K_eff_values)) if K_eff_values else 0.0,
        "consob_threshold_rule_bimodal_frac": bimodal_frac,
        "consob_threshold_rule_percentile_frac": percentile_frac,
        "consob_pse_anchor_resolution_frac": pse_anchor_counts.get("resolution", 0) / n_anchor,
        # ---- provenance (per the spec) ----
        "consob_normalisation": "linf",
        "consob_error_stat": "max_per_bucket_abs",
        "consob_conditions_implemented": ["a", "b", "c", "d"],
        "consob_threshold_rule": _majority(threshold_rule_counts, "percentile_fallback"),
        "consob_pse_anchor": _majority(pse_anchor_counts, "largest_price_jump"),
        "consob_bucket_hours": int(bucket_hours),
        "consob_investigation_hours": int(investigation_hours),
        "consob_short_market_fallback": "final_50pct_of_span_if_span_under_48h",
        "consob_d_theta": int(d_theta),
        "consob_n_theta_rule": "top_decile",
        "consob_k_selection": k_selection,
        "consob_temporal_unit_note": (
            "rescaled daily->6h for prediction-market horizon; fixed a priori, not label-tuned"
        ),
        "uses_resolved_winning_outcome": True,
        "deployable_live": False,
        "consob_definition_match": (
            "faithful_four_condition_reconstruction_screen "
            "(PCA only; PSE and investigation window adapted for prediction markets)"
        ),
    }


def collect_consob_pca_faithful_flags(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes: Dict[int, Optional[int]],
    *,
    bucket_hours: int = 6,
    investigation_hours: int = 24,
    d_theta: int = 3,
    n_components: NComponents = 3,
    min_wallets_for_kde: int = 8,
    percentile_fallback: float = 90.0,
    min_wallet_notional: float = 500.0,
    min_usd_amount: Optional[float] = None,
) -> Tuple[Dict[int, Set[str]], Dict[int, Dict[str, int]]]:
    """
    Per-market eval-fit flagged-wallet extraction for the curated reported-insider
    recall comparison (the PRIMARY metric). Wallets are normalised; counts are 1
    per flagged wallet (this is a wallet-level method).
    """
    from experiments.sota_algorithms.curated_recall_flags import _normalize_wallet

    flagged_by_market: Dict[int, Set[str]] = {}
    counts_by_market: Dict[int, Dict[str, int]] = {}
    for market_id in market_ids:
        winning = winning_outcomes.get(market_id)
        if winning is None:
            flagged_by_market[market_id] = set()
            counts_by_market[market_id] = {}
            continue
        trades = _load_market_trades(loader, market_id, min_usd_amount)
        resolution_ts = get_market_resolution_timestamp_ms(loader, market_id)
        flagged, _diag = run_consob_faithful_market(
            trades,
            int(winning),
            resolution_ts,
            bucket_hours=bucket_hours,
            investigation_hours=investigation_hours,
            d_theta=d_theta,
            n_components=n_components,
            min_wallet_notional=min_wallet_notional,
            min_wallets_for_kde=min_wallets_for_kde,
            percentile_fallback=percentile_fallback,
        )
        normalized = {_normalize_wallet(w) for w in flagged}
        flagged_by_market[market_id] = normalized
        counts_by_market[market_id] = {w: 1 for w in normalized}
    return flagged_by_market, counts_by_market
