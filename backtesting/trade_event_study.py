"""Trade-level event study evaluator for flagged BUY trades."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

from backtesting.backtest_runner import BacktestResult
from backtesting.market_resolutions import get_winning_outcome
from models import Trade


@dataclass
class TradeEventStudyResult:
    """Results from a BUY-trade event study on a single market."""
    market_id: int
    market_slug: str
    winning_outcome: int

    # Counts
    total_buy_trades: int
    flagged_buy_trades: int
    unflagged_buy_trades: int

    # Notionals
    flagged_total_notional: float
    unflagged_total_notional: float

    # Return distributions
    flagged_mean_return: float
    flagged_median_return: float
    flagged_std_return: float
    unflagged_mean_return: float
    unflagged_median_return: float
    unflagged_std_return: float

    # Difference
    mean_return_diff: float  # flagged - unflagged

    # Effect size
    cohens_d: float

    # Statistical tests
    welch_t_stat: float
    welch_p_value: float
    mann_whitney_u_stat: float
    mann_whitney_p_value: float

    # Notional-weighted returns
    flagged_weighted_return: float
    unflagged_weighted_return: float

    # Win rates (fraction of BUY trades on the winning outcome)
    flagged_win_rate: float
    unflagged_win_rate: float

    # Stratified analysis (optional, populated if trade features available)
    by_score_band: Dict[str, Dict] = field(default_factory=dict)
    by_detector: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialise for JSON aggregation across markets."""
        d = {
            "market_id": self.market_id,
            "market_slug": self.market_slug,
            "winning_outcome": self.winning_outcome,
            "total_buy_trades": self.total_buy_trades,
            "flagged_buy_trades": self.flagged_buy_trades,
            "unflagged_buy_trades": self.unflagged_buy_trades,
            "flagged_total_notional": self.flagged_total_notional,
            "unflagged_total_notional": self.unflagged_total_notional,
            "flagged_mean_return": self.flagged_mean_return,
            "flagged_median_return": self.flagged_median_return,
            "flagged_std_return": self.flagged_std_return,
            "unflagged_mean_return": self.unflagged_mean_return,
            "unflagged_median_return": self.unflagged_median_return,
            "unflagged_std_return": self.unflagged_std_return,
            "mean_return_diff": self.mean_return_diff,
            "cohens_d": self.cohens_d,
            "welch_t_stat": self.welch_t_stat,
            "welch_p_value": self.welch_p_value,
            "mann_whitney_u_stat": self.mann_whitney_u_stat,
            "mann_whitney_p_value": self.mann_whitney_p_value,
            "flagged_weighted_return": self.flagged_weighted_return,
            "unflagged_weighted_return": self.unflagged_weighted_return,
            "flagged_win_rate": self.flagged_win_rate,
            "unflagged_win_rate": self.unflagged_win_rate,
            "by_score_band": self.by_score_band,
            "by_detector": self.by_detector,
        }
        return d

    def summary(self) -> str:
        """Human-readable report for logging."""
        lines = [
            f"=== Trade Event Study: {self.market_slug} (id={self.market_id}) ===",
            f"Winning outcome: {self.winning_outcome}",
            f"",
            f"BUY trades: {self.total_buy_trades:,} total "
            f"({self.flagged_buy_trades:,} flagged, {self.unflagged_buy_trades:,} unflagged)",
            f"",
            f"  Flagged   — mean={self.flagged_mean_return:+.4f}  "
            f"median={self.flagged_median_return:+.4f}  "
            f"std={self.flagged_std_return:.4f}  "
            f"notional=${self.flagged_total_notional:,.0f}  "
            f"win_rate={self.flagged_win_rate:.2%}",
            f"  Unflagged — mean={self.unflagged_mean_return:+.4f}  "
            f"median={self.unflagged_median_return:+.4f}  "
            f"std={self.unflagged_std_return:.4f}  "
            f"notional=${self.unflagged_total_notional:,.0f}  "
            f"win_rate={self.unflagged_win_rate:.2%}",
            f"",
            f"  Mean diff (flagged - unflagged): {self.mean_return_diff:+.4f}",
            f"  Cohen's d: {self.cohens_d:.3f}",
            f"  Weighted returns — flagged={self.flagged_weighted_return:+.4f}  "
            f"unflagged={self.unflagged_weighted_return:+.4f}",
            f"",
            f"  Welch's t-test:   t={self.welch_t_stat:.3f}  p={self.welch_p_value:.4e}",
            f"  Mann-Whitney U:   U={self.mann_whitney_u_stat:.1f}  p={self.mann_whitney_p_value:.4e}",
        ]

        if self.by_score_band:
            lines.append("")
            lines.append("  Score-band stratification:")
            for band, info in sorted(self.by_score_band.items()):
                lines.append(
                    f"    {band}: n={info['count']}  "
                    f"mean_ret={info['mean_return']:+.4f}  "
                    f"win_rate={info['win_rate']:.2%}"
                )

        if self.by_detector:
            lines.append("")
            lines.append("  Per-detector stratification:")
            for det, info in sorted(self.by_detector.items()):
                lines.append(
                    f"    {det}: n={info['count']}  "
                    f"mean_ret={info['mean_return']:+.4f}  "
                    f"win_rate={info['win_rate']:.2%}"
                )

        return "\n".join(lines)

def _compute_resolution_return(trade: Trade, winning_outcome: int) -> float:
    """
    Resolution return for a BUY trade.
    """
    if trade.outcome_index == winning_outcome:
        return (1.0 - trade.price) / trade.price if trade.price > 1e-9 else 0.0
    else:
        return -1.0


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Notional-weighted mean return."""
    w_sum = weights.sum()
    if w_sum < 1e-9:
        return 0.0
    return float(np.sum(values * weights) / w_sum)


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Cohen's d: (mean_a - mean_b) / pooled_std.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var)

    if pooled_std < 1e-12:
        return 0.0

    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def run_trade_event_study(
    trades: List[Trade],
    backtest_result: BacktestResult,
    market_metadata: Dict,
    winning_outcome: Optional[int] = None,
) -> Optional[TradeEventStudyResult]:
    """Run a trade-level event study on one market."""
    market_id = int(market_metadata.get("id", -1))
    market_slug = str(market_metadata.get("market_slug", ""))

    if winning_outcome is None:
        winning_outcome = get_winning_outcome(market_id)
    if winning_outcome is None:
        return None

    flagged_keys: Set[Tuple[str, int]] = set()
    flag_metadata: Dict[Tuple[str, int], Dict] = {}

    for wallet, flags in backtest_result.wallet_flags.items():
        for flag_entry in flags:
            ts = int(flag_entry.get("timestamp_ms", 0))
            key = (wallet, ts)
            flagged_keys.add(key)
            flag_metadata[key] = {
                "score": float(flag_entry.get("score", 0.0)),
                "detectors": list(flag_entry.get("detectors", [])),
            }

    flagged_returns: List[float] = []
    flagged_notionals: List[float] = []
    flagged_wins: int = 0
    flagged_scores: List[float] = []
    flagged_detector_lists: List[List[str]] = []

    unflagged_returns: List[float] = []
    unflagged_notionals: List[float] = []
    unflagged_wins: int = 0

    for trade in trades:
        if trade.side.upper() != "BUY":
            continue

        ret = _compute_resolution_return(trade, winning_outcome)
        is_win = trade.outcome_index == winning_outcome
        key = (trade.wallet, trade.timestamp_ms)
        is_flagged = key in flagged_keys

        if is_flagged:
            flagged_returns.append(ret)
            flagged_notionals.append(trade.notional_usdc)
            flagged_wins += int(is_win)
            meta = flag_metadata.get(key, {})
            flagged_scores.append(meta.get("score", 0.0))
            flagged_detector_lists.append(meta.get("detectors", []))
        else:
            unflagged_returns.append(ret)
            unflagged_notionals.append(trade.notional_usdc)
            unflagged_wins += int(is_win)

    if len(flagged_returns) < 1 or len(unflagged_returns) < 1:
        return None

    f_ret = np.array(flagged_returns, dtype=np.float64)
    u_ret = np.array(unflagged_returns, dtype=np.float64)
    f_not = np.array(flagged_notionals, dtype=np.float64)
    u_not = np.array(unflagged_notionals, dtype=np.float64)

    f_mean = float(np.mean(f_ret))
    f_median = float(np.median(f_ret))
    f_std = float(np.std(f_ret, ddof=1)) if len(f_ret) > 1 else 0.0
    u_mean = float(np.mean(u_ret))
    u_median = float(np.median(u_ret))
    u_std = float(np.std(u_ret, ddof=1)) if len(u_ret) > 1 else 0.0

    mean_diff = f_mean - u_mean
    d = _cohens_d(f_ret, u_ret)

    if len(f_ret) >= 2 and len(u_ret) >= 2:
        t_stat, t_p = stats.ttest_ind(f_ret, u_ret, equal_var=False)
    else:
        t_stat, t_p = 0.0, 1.0

    # Test whether flagged trades are stochastically greater than unflagged.
    if len(f_ret) >= 1 and len(u_ret) >= 1:
        u_stat, mw_p = stats.mannwhitneyu(
            f_ret, u_ret, alternative="greater"
        )
    else:
        u_stat, mw_p = 0.0, 1.0

    f_weighted = _weighted_mean(f_ret, f_not)
    u_weighted = _weighted_mean(u_ret, u_not)

    n_f = len(flagged_returns)
    n_u = len(unflagged_returns)
    f_win_rate = flagged_wins / n_f if n_f > 0 else 0.0
    u_win_rate = unflagged_wins / n_u if n_u > 0 else 0.0

    by_score_band: Dict[str, Dict] = {}
    if flagged_scores:
        score_arr = np.array(flagged_scores)
        bands = [
            ("0.5-0.6", 0.5, 0.6),
            ("0.6-0.7", 0.6, 0.7),
            ("0.7-0.8", 0.7, 0.8),
            ("0.8-0.9", 0.8, 0.9),
            ("0.9-1.0", 0.9, 1.01),
        ]
        for label, lo, hi in bands:
            mask = (score_arr >= lo) & (score_arr < hi)
            if mask.sum() == 0:
                continue
            band_ret = f_ret[mask]
            band_not = f_not[mask]
            band_wins = 0
            for idx in np.where(mask)[0]:
                # Recover whether this trade was on the winning outcome
                # from the return: if ret > -1.0, it was a winning-side buy
                # (since losing-side is exactly -1.0)
                if flagged_returns[idx] > -0.999:
                    band_wins += 1
            by_score_band[label] = {
                "count": int(mask.sum()),
                "mean_return": float(np.mean(band_ret)),
                "median_return": float(np.median(band_ret)),
                "weighted_return": _weighted_mean(band_ret, band_not),
                "win_rate": band_wins / int(mask.sum()) if mask.sum() > 0 else 0.0,
            }

    by_detector: Dict[str, Dict] = {}
    if flagged_detector_lists:
        all_detectors: Set[str] = set()
        for dl in flagged_detector_lists:
            all_detectors.update(dl)

        for det_name in sorted(all_detectors):
            mask = np.array(
                [det_name in dl for dl in flagged_detector_lists],
                dtype=bool,
            )
            if mask.sum() == 0:
                continue
            det_ret = f_ret[mask]
            det_not = f_not[mask]
            det_wins = sum(
                1 for idx in np.where(mask)[0]
                if flagged_returns[idx] > -0.999
            )
            by_detector[det_name] = {
                "count": int(mask.sum()),
                "mean_return": float(np.mean(det_ret)),
                "median_return": float(np.median(det_ret)),
                "weighted_return": _weighted_mean(det_ret, det_not),
                "win_rate": det_wins / int(mask.sum()) if mask.sum() > 0 else 0.0,
            }

    return TradeEventStudyResult(
        market_id=market_id,
        market_slug=market_slug,
        winning_outcome=winning_outcome,
        total_buy_trades=n_f + n_u,
        flagged_buy_trades=n_f,
        unflagged_buy_trades=n_u,
        flagged_total_notional=float(f_not.sum()),
        unflagged_total_notional=float(u_not.sum()),
        flagged_mean_return=f_mean,
        flagged_median_return=f_median,
        flagged_std_return=f_std,
        unflagged_mean_return=u_mean,
        unflagged_median_return=u_median,
        unflagged_std_return=u_std,
        mean_return_diff=mean_diff,
        cohens_d=d,
        welch_t_stat=float(t_stat),
        welch_p_value=float(t_p),
        mann_whitney_u_stat=float(u_stat),
        mann_whitney_p_value=float(mw_p),
        flagged_weighted_return=f_weighted,
        unflagged_weighted_return=u_weighted,
        flagged_win_rate=f_win_rate,
        unflagged_win_rate=u_win_rate,
        by_score_band=by_score_band,
        by_detector=by_detector,
    )

def run_trade_event_study_multi(
    results_per_market: List[TradeEventStudyResult],
) -> Dict:
    """Aggregate trade-level event study results across markets."""
    if not results_per_market:
        return {"pooled": {}, "per_market": []}

    total_flagged = sum(r.flagged_buy_trades for r in results_per_market)
    total_unflagged = sum(r.unflagged_buy_trades for r in results_per_market)
    total_flagged_notional = sum(r.flagged_total_notional for r in results_per_market)
    total_unflagged_notional = sum(r.unflagged_total_notional for r in results_per_market)

    # Trade-count-weighted mean returns across markets
    if total_flagged > 0:
        pooled_flagged_mean = sum(
            r.flagged_mean_return * r.flagged_buy_trades
            for r in results_per_market
        ) / total_flagged
    else:
        pooled_flagged_mean = 0.0

    if total_unflagged > 0:
        pooled_unflagged_mean = sum(
            r.unflagged_mean_return * r.unflagged_buy_trades
            for r in results_per_market
        ) / total_unflagged
    else:
        pooled_unflagged_mean = 0.0

    # Notional-weighted returns across markets
    if total_flagged_notional > 0:
        pooled_flagged_weighted = sum(
            r.flagged_weighted_return * r.flagged_total_notional
            for r in results_per_market
        ) / total_flagged_notional
    else:
        pooled_flagged_weighted = 0.0

    if total_unflagged_notional > 0:
        pooled_unflagged_weighted = sum(
            r.unflagged_weighted_return * r.unflagged_total_notional
            for r in results_per_market
        ) / total_unflagged_notional
    else:
        pooled_unflagged_weighted = 0.0

    # Count how many markets have significant results
    sig_welch = sum(1 for r in results_per_market if r.welch_p_value < 0.05)
    sig_mw = sum(1 for r in results_per_market if r.mann_whitney_p_value < 0.05)

    # Mean Cohen's d (inverse-variance weighting would be better for a
    # proper meta-analysis, but simple mean is fine for now)
    mean_d = float(np.mean([r.cohens_d for r in results_per_market]))

    pooled = {
        "n_markets": len(results_per_market),
        "total_flagged_trades": total_flagged,
        "total_unflagged_trades": total_unflagged,
        "total_flagged_notional": total_flagged_notional,
        "total_unflagged_notional": total_unflagged_notional,
        "pooled_flagged_mean_return": pooled_flagged_mean,
        "pooled_unflagged_mean_return": pooled_unflagged_mean,
        "pooled_mean_return_diff": pooled_flagged_mean - pooled_unflagged_mean,
        "pooled_flagged_weighted_return": pooled_flagged_weighted,
        "pooled_unflagged_weighted_return": pooled_unflagged_weighted,
        "mean_cohens_d": mean_d,
        "markets_significant_welch_p05": sig_welch,
        "markets_significant_mw_p05": sig_mw,
    }

    per_market = [r.to_dict() for r in results_per_market]

    return {"pooled": pooled, "per_market": per_market}

@dataclass
class CopytradeResult:
    """Simulated P&L from copytrading every flagged BUY trade."""
    # Notional-matched view (always populated)
    total_flagged_buys: int
    total_capital_deployed: float
    total_payout: float
    total_pnl: float
    portfolio_roi: float             # notional-weighted

    winning_trades: int
    losing_trades: int
    win_rate: float

    mean_trade_return: float         # equal-weighted mean (same formula as fixed-size roi)
    median_trade_return: float
    std_trade_return: float
    weighted_return: float           # same as portfolio_roi; kept for back-compat

    # Fixed-size view (populated when fixed_trade_size is not None)
    fixed_trade_size: Optional[float] = None
    fixed_capital_deployed: Optional[float] = None   # n_trades * fixed_trade_size
    fixed_total_pnl: Optional[float] = None
    fixed_roi: Optional[float] = None                # = mean_trade_return (equal-weighted)
    fixed_median_return: Optional[float] = None

    # Per-market breakdown
    per_market: Dict[int, Dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Copytrade Simulation ===",
            f"Flagged BUY trades copied: {self.total_flagged_buys:,}",
            "",
            "  [Notional-matched]",
            f"    Capital deployed: ${self.total_capital_deployed:,.2f}",
            f"    Net P&L:          ${self.total_pnl:+,.2f}",
            f"    Portfolio ROI:    {self.portfolio_roi:+.2%}  (notional-weighted)",
            f"    Win rate:         {self.win_rate:.2%} ({self.winning_trades}/{self.total_flagged_buys})",
            f"    Mean return:      {self.mean_trade_return:+.4f}",
            f"    Median return:    {self.median_trade_return:+.4f}",
        ]
        if self.fixed_trade_size is not None:
            lines += [
                "",
                f"  [Fixed-size @ ${self.fixed_trade_size:,.0f}/trade]",
                f"    Capital deployed: ${self.fixed_capital_deployed:,.2f}",
                f"    Net P&L:          ${self.fixed_total_pnl:+,.2f}",
                f"    ROI:              {self.fixed_roi:+.2%}  (equal-weighted)",
                f"    Median return:    {self.fixed_median_return:+.4f}",
            ]
        if self.per_market:
            lines.append("")
            lines.append("Per-market breakdown:")
            for mid, m in sorted(self.per_market.items()):
                line = (
                    f"  market={mid}: "
                    f"trades={m['n_trades']} "
                    f"capital=${m['capital']:,.0f} "
                    f"pnl=${m['pnl']:+,.0f} "
                    f"roi={m['roi']:+.2%} "
                    f"win_rate={m['win_rate']:.2%}"
                )
                if self.fixed_trade_size is not None:
                    line += (
                        f" | fixed_pnl=${m.get('fixed_pnl', 0.0):+,.0f}"
                        f" fixed_roi={m.get('fixed_roi', 0.0):+.2%}"
                    )
                lines.append(line)
        return "\n".join(lines)


def run_copytrade_simulation(
    trades: List[Trade],
    backtest_result: BacktestResult,
    market_metadata: Dict,
    winning_outcome: Optional[int] = None,
    fixed_trade_size: Optional[float] = None,
) -> Optional[CopytradeResult]:
    """Simulate copytrading every flagged BUY trade in one market."""
    market_id = int(market_metadata.get("id", -1))

    if winning_outcome is None:
        winning_outcome = get_winning_outcome(market_id)
    if winning_outcome is None:
        return None

    # Build flagged keys set
    flagged_keys: Set[Tuple[str, int]] = set()
    for wallet, flags in backtest_result.wallet_flags.items():
        for flag_entry in flags:
            ts = int(flag_entry.get("timestamp_ms", 0))
            flagged_keys.add((wallet, ts))

    if not flagged_keys:
        return None

    returns_list: List[float] = []
    notionals_list: List[float] = []
    total_payout_notional = 0.0
    total_capital_notional = 0.0
    winning_count = 0

    for trade in trades:
        if trade.side.upper() != "BUY":
            continue

        key = (trade.wallet, trade.timestamp_ms)
        if key not in flagged_keys:
            continue

        capital = trade.notional_usdc
        total_capital_notional += capital

        is_win = trade.outcome_index == winning_outcome
        if is_win:
            payout = trade.size_tokens   # each token resolves to $1
            winning_count += 1
        else:
            payout = 0.0

        total_payout_notional += payout
        trade_pnl = payout - capital
        trade_return = trade_pnl / capital if capital > 1e-9 else 0.0

        returns_list.append(trade_return)
        notionals_list.append(capital)

    if not returns_list:
        return None

    returns_arr = np.array(returns_list, dtype=np.float64)
    notionals_arr = np.array(notionals_list, dtype=np.float64)
    n = len(returns_list)

    # Notional-matched portfolio
    total_pnl_notional = total_payout_notional - total_capital_notional
    portfolio_roi = total_pnl_notional / total_capital_notional if total_capital_notional > 1e-9 else 0.0

    w_sum = notionals_arr.sum()
    weighted_return = float(np.sum(returns_arr * notionals_arr) / w_sum) if w_sum > 1e-9 else 0.0

    # Fixed-size portfolio (optional)
    fixed_capital: Optional[float] = None
    fixed_pnl: Optional[float] = None
    fixed_roi_val: Optional[float] = None
    fixed_median: Optional[float] = None

    if fixed_trade_size is not None and fixed_trade_size > 0:
        fixed_capital = fixed_trade_size * n
        fixed_roi_val = float(np.mean(returns_arr))
        fixed_pnl = fixed_roi_val * fixed_capital
        fixed_median = float(np.median(returns_arr))

    per_market_entry: Dict = {
        "n_trades": n,
        "capital": total_capital_notional,
        "payout": total_payout_notional,
        "pnl": total_pnl_notional,
        "roi": portfolio_roi,
        "win_rate": winning_count / n if n > 0 else 0.0,
    }
    if fixed_trade_size is not None:
        per_market_entry["fixed_pnl"] = fixed_pnl
        per_market_entry["fixed_roi"] = fixed_roi_val

    return CopytradeResult(
        total_flagged_buys=n,
        total_capital_deployed=total_capital_notional,
        total_payout=total_payout_notional,
        total_pnl=total_pnl_notional,
        portfolio_roi=portfolio_roi,
        winning_trades=winning_count,
        losing_trades=n - winning_count,
        win_rate=winning_count / n if n > 0 else 0.0,
        mean_trade_return=float(np.mean(returns_arr)),
        median_trade_return=float(np.median(returns_arr)),
        std_trade_return=float(np.std(returns_arr, ddof=1)) if n > 1 else 0.0,
        weighted_return=weighted_return,
        fixed_trade_size=fixed_trade_size,
        fixed_capital_deployed=fixed_capital,
        fixed_total_pnl=fixed_pnl,
        fixed_roi=fixed_roi_val,
        fixed_median_return=fixed_median,
        per_market={market_id: per_market_entry},
    )


def run_copytrade_simulation_multi(
    results: List[CopytradeResult],
) -> Optional[CopytradeResult]:
    """
    Aggregate copytrade results across multiple markets into a single
    portfolio-level summary. Preserves both notional-matched and
    fixed-size views if the constituent results include fixed-size data.
    """
    if not results:
        return None

    total_capital = sum(r.total_capital_deployed for r in results)
    total_payout = sum(r.total_payout for r in results)
    total_pnl = total_payout - total_capital
    n_total = sum(r.total_flagged_buys for r in results)
    winning_total = sum(r.winning_trades for r in results)

    all_per_market: Dict[int, Dict] = {}
    for r in results:
        all_per_market.update(r.per_market)

    weighted_return = total_pnl / total_capital if total_capital > 1e-9 else 0.0

    # Trade-count-weighted mean return (= fixed-size ROI pooled across markets)
    if n_total > 0:
        mean_return = sum(r.mean_trade_return * r.total_flagged_buys for r in results) / n_total
        median_return = float(np.median([r.median_trade_return for r in results]))  # approx
    else:
        mean_return = 0.0
        median_return = 0.0

    # Fixed-size aggregation
    fixed_trade_size = results[0].fixed_trade_size
    fixed_capital: Optional[float] = None
    fixed_pnl: Optional[float] = None
    fixed_roi_val: Optional[float] = None
    fixed_median: Optional[float] = None

    if fixed_trade_size is not None and all(r.fixed_trade_size == fixed_trade_size for r in results):
        fixed_capital = fixed_trade_size * n_total
        fixed_roi_val = mean_return   # identical by construction
        fixed_pnl = fixed_roi_val * fixed_capital
        fixed_median = median_return  # approx from per-market medians

    return CopytradeResult(
        total_flagged_buys=n_total,
        total_capital_deployed=total_capital,
        total_payout=total_payout,
        total_pnl=total_pnl,
        portfolio_roi=weighted_return,
        winning_trades=winning_total,
        losing_trades=n_total - winning_total,
        win_rate=winning_total / n_total if n_total > 0 else 0.0,
        mean_trade_return=mean_return,
        median_trade_return=median_return,
        std_trade_return=0.0,   # can't pool std exactly from summaries
        weighted_return=weighted_return,
        fixed_trade_size=fixed_trade_size,
        fixed_capital_deployed=fixed_capital,
        fixed_total_pnl=fixed_pnl,
        fixed_roi=fixed_roi_val,
        fixed_median_return=fixed_median,
        per_market=all_per_market,
    )


def infer_market_winning_outcome_from_last_prices(
    trades: List[Trade],
    threshold: float = 0.99,
) -> Optional[int]:
    """
    Infer market resolution from the last traded price of each outcome.

    Returns the outcome index whose latest observed trade price is >= threshold.
    If no outcome qualifies, returns None.
    """
    if not trades:
        return None

    latest_by_outcome: Dict[int, Tuple[int, float]] = {}
    for trade in trades:
        idx = int(trade.outcome_index)
        ts = int(trade.timestamp_ms)
        price = float(trade.price)
        prev = latest_by_outcome.get(idx)
        if prev is None or ts >= prev[0]:
            latest_by_outcome[idx] = (ts, price)

    qualifying = [
        (outcome_idx, ts_price[1])
        for outcome_idx, ts_price in latest_by_outcome.items()
        if ts_price[1] >= float(threshold)
    ]
    if not qualifying:
        return None

    qualifying.sort(key=lambda x: x[1], reverse=True)
    return int(qualifying[0][0])
