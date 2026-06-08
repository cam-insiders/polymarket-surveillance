"""Cached wallet and trade-return evaluation helpers."""

from __future__ import annotations

import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backtesting.backtest_runner import BacktestResult
from backtesting.logging_utils import experiment_backtest_logs_quiet
from backtesting.market_resolutions import get_winning_outcome
from backtesting.trade_level_metrics import (
    compute_pooled_trade_metrics,
    trade_level_zero_defaults,
)
from models import Trade

logger = logging.getLogger(__name__)

def _is_experiments_run() -> bool:
    """True when the entry script lives under the experiments directory."""
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0:
        return False
    normalized = os.path.normpath(argv0).lower()
    parts = normalized.split(os.sep)
    return "experiments" in parts


def _warn(msg: str) -> None:
    """Suppress warning noise during experiments sweeps."""
    if _is_experiments_run():
        return
    logger.warning(msg)


@dataclass
class WalletGroundTruth:
    """Config-independent financial data for one wallet in one market."""
    wallet: str
    net_pnl: float
    gross_buy_notional: float
    total_notional: float
    wallet_return: float       # net_pnl / gross_buy_notional
    informed_score: float      # return * log2(1 + gross_buy / 1000)
    label_value: float         # the value used for z-score (return or pnl)
    label_mean: float          # population mean for this market
    label_std: float           # population std for this market
    z_score: float
    is_insider: bool
    trade_count: int           # also config-independent: all trades are unconditionally counted


@dataclass
class PrecomputedMarketGroundTruth:
    """Config-independent ground truth for a single market."""
    market_id: int
    market_slug: str
    winning_outcome: int
    label_metric: str
    z_score_threshold: float
    wallet_truths: Dict[str, WalletGroundTruth] = field(default_factory=dict)

    @property
    def insider_count(self) -> int:
        return sum(1 for gt in self.wallet_truths.values() if gt.is_insider)

    @property
    def wallet_count(self) -> int:
        return len(self.wallet_truths)

    def __repr__(self) -> str:
        return (
            f"PrecomputedMarketGroundTruth("
            f"market_id={self.market_id}, "
            f"slug={self.market_slug!r}, "
            f"wallets={self.wallet_count}, "
            f"insiders={self.insider_count})"
        )
    

def precompute_ground_truth(
    trades: List[Trade],
    market_metadata: Dict,
    label_metric: str = "return",
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
    winning_outcome_override: Optional[int] = None,
) -> Optional[PrecomputedMarketGroundTruth]:
    """Compute config-independent wallet financials for one market."""
    if label_metric not in {"return", "pnl"}:
        raise ValueError(f"label_metric must be 'return' or 'pnl', got {label_metric!r}")

    market_id = int(market_metadata.get("id", -1))
    market_slug = str(market_metadata.get("market_slug", ""))
    winning_outcome = (
        int(winning_outcome_override)
        if winning_outcome_override is not None
        else get_winning_outcome(market_id)
    )

    if winning_outcome is None:
        _warn(
            f"precompute_ground_truth: market {market_slug!r} ({market_id}) "
            f"not in resolution dataset — skipping."
        )
        return None

    wallet_positions: Dict[str, Dict[int, float]] = {}   # wallet -> {outcome_idx -> shares}
    wallet_costs: Dict[str, Dict[int, float]] = {}        # wallet -> {outcome_idx -> usdc_cost}
    wallet_notional: Dict[str, float] = {}
    wallet_gross_buy: Dict[str, float] = {}
    wallet_trade_counts: Dict[str, int] = {}

    for trade in trades:
        w = trade.wallet

        if w not in wallet_positions:
            wallet_positions[w] = {}
            wallet_costs[w] = {}
            wallet_notional[w] = 0.0
            wallet_gross_buy[w] = 0.0
            wallet_trade_counts[w] = 0

        wallet_trade_counts[w] += 1
        wallet_notional[w] += trade.notional_usdc

        oi = trade.outcome_index
        if oi not in wallet_positions[w]:
            wallet_positions[w][oi] = 0.0
            wallet_costs[w][oi] = 0.0

        if trade.side.upper() == "BUY":
            wallet_positions[w][oi] += trade.size_tokens
            wallet_costs[w][oi] += trade.notional_usdc
            wallet_gross_buy[w] += trade.notional_usdc
        else:
            # SELL: reduce position and cost basis
            wallet_positions[w][oi] -= trade.size_tokens
            wallet_costs[w][oi] -= trade.notional_usdc

    eligible_rows = []

    for w, positions in wallet_positions.items():
        if not positions:
            continue

        total_notional = wallet_notional.get(w, 0.0)
        if total_notional < min_wallet_notional:
            continue

        gross_buy = wallet_gross_buy.get(w, 0.0)
        costs = wallet_costs.get(w, {})

        # Compute payout: only winning-outcome shares are worth anything at resolution
        total_payout = 0.0
        total_cost = 0.0
        for oi, shares in positions.items():
            if oi == winning_outcome:
                total_payout += shares
            total_cost += costs.get(oi, 0.0)

        net_pnl = total_payout - total_cost
        wallet_return = (net_pnl / gross_buy) if gross_buy > 1e-9 else 0.0

        # Informed score: same formula as WalletEvaluator
        informed_score = (
            wallet_return * math.log2(1.0 + gross_buy / 1000.0)
            if gross_buy > 0 else 0.0
        )

        label_value = wallet_return if label_metric == "return" else net_pnl

        eligible_rows.append({
            "wallet": w,
            "net_pnl": net_pnl,
            "gross_buy_notional": gross_buy,
            "total_notional": total_notional,
            "wallet_return": wallet_return,
            "informed_score": informed_score,
            "label_value": label_value,
            "trade_count": wallet_trade_counts.get(w, 0),
        })

    if len(eligible_rows) < 3:
        _warn(
            f"precompute_ground_truth: only {len(eligible_rows)} eligible wallets "
            f"for market {market_slug!r} — skipping (need >= 3 for z-score)."
        )
        return None

    label_values = np.array([r["label_value"] for r in eligible_rows], dtype=float)
    mean_v = float(np.mean(label_values))
    std_v = float(np.std(label_values, ddof=1)) if len(label_values) > 1 else 0.0

    wallet_truths: Dict[str, WalletGroundTruth] = {}
    insider_count = 0

    for row in eligible_rows:
        z = (row["label_value"] - mean_v) / std_v if std_v > 1e-12 else 0.0
        is_insider = bool(z > z_score_threshold)
        insider_count += int(is_insider)

        wallet_truths[row["wallet"]] = WalletGroundTruth(
            wallet=row["wallet"],
            net_pnl=row["net_pnl"],
            gross_buy_notional=row["gross_buy_notional"],
            total_notional=row["total_notional"],
            wallet_return=row["wallet_return"],
            informed_score=row["informed_score"],
            label_value=row["label_value"],
            label_mean=mean_v,
            label_std=std_v,
            z_score=z,
            is_insider=is_insider,
            trade_count=row["trade_count"],
        )

    result = PrecomputedMarketGroundTruth(
        market_id=market_id,
        market_slug=market_slug,
        winning_outcome=winning_outcome,
        label_metric=label_metric,
        z_score_threshold=z_score_threshold,
        wallet_truths=wallet_truths,
    )

    if not experiment_backtest_logs_quiet():
        logger.info(
            f"precompute_ground_truth: market={market_slug!r} "
            f"| wallets={len(wallet_truths)} "
            f"| insiders={insider_count} ({insider_count/len(wallet_truths):.2%}) "
            f"| metric={label_metric} mean={mean_v:.4f} std={std_v:.4f}"
        )

    return result

@dataclass
class PrecomputedTradeReturns:
    """Config-independent BUY-trade resolution returns for one market."""
    market_id: int
    winning_outcome: int
    n_buy_trades: int

    # Parallel arrays, length = n_buy_trades
    wallets: np.ndarray
    timestamps_ms: np.ndarray
    wallet_codes: np.ndarray
    wallet_code_map: Dict[str, int]
    trade_keys: np.ndarray
    returns: np.ndarray
    notionals: np.ndarray

def precompute_trade_returns(
        trades: List[Trade],
        market_metadata: Dict,
        winning_outcome: Optional[int] = None,
) -> Optional[PrecomputedTradeReturns]:
    """Precompute resolution returns for all BUY trades in a market."""
    market_id = int(market_metadata.get("id", -1))

    if winning_outcome is None:
        winning_outcome = get_winning_outcome(market_id)
    if winning_outcome is None:
        _warn(
            f"precompute_trade_returns: market {market_id} not in resolution dataset - skipping."
        )
        return None

    wallets: List[str] = []
    timestamps_ms: List[int] = []
    returns: List[float] = []
    notionals: List[float] = []

    for trade in trades:
        if trade.side.upper() != "BUY":
            continue

        wallets.append(trade.wallet)
        timestamps_ms.append(trade.timestamp_ms)
        notionals.append(trade.notional_usdc)

        if trade.outcome_index == winning_outcome:
            ret = (1.0 - trade.price) / trade.price if trade.price > 1e-9 else 0.0
        else:
            ret = -1.0  # total loss
        
        returns.append(ret)

    if not returns:
        _warn(
            f"precompute_trade_returns: no BUY trades for market {market_id} - skipping."
        )
        return None
    
    wallets_arr = np.array(wallets, dtype=object)
    timestamps_arr = np.array(timestamps_ms, dtype=np.int64)
    wallet_code_map: Dict[str, int] = {}
    wallet_codes = np.empty(len(wallets_arr), dtype=np.int64)
    next_code = 0
    for i, wallet in enumerate(wallets_arr):
        code = wallet_code_map.get(wallet)
        if code is None:
            code = next_code
            wallet_code_map[wallet] = code
            next_code += 1
        wallet_codes[i] = code

    trade_keys = np.empty(len(wallets_arr), dtype=[("w", np.int64), ("t", np.int64)])
    trade_keys["w"] = wallet_codes
    trade_keys["t"] = timestamps_arr

    return PrecomputedTradeReturns(
        market_id=market_id,
        winning_outcome=winning_outcome,
        n_buy_trades=len(returns),
        wallets=wallets_arr,
        timestamps_ms=timestamps_arr,
        wallet_codes=wallet_codes,
        wallet_code_map=wallet_code_map,
        trade_keys=trade_keys,
        returns=np.array(returns, dtype=np.float64),
        notionals=np.array(notionals, dtype=np.float64)
    )


def build_trade_flag_mask(
    precomputed_tr: PrecomputedTradeReturns,
    backtest_result: BacktestResult,
) -> np.ndarray:
    """
    Build a vectorized BUY-trade flag mask using structured key matching.
    """
    flagged_wallet_codes: List[int] = []
    flagged_timestamps: List[int] = []

    for wallet, flags in backtest_result.wallet_flags.items():
        wallet_code = precomputed_tr.wallet_code_map.get(wallet)
        if wallet_code is None:
            continue
        for entry in flags:
            flagged_wallet_codes.append(wallet_code)
            flagged_timestamps.append(int(entry.get("timestamp_ms", 0)))

    if not flagged_wallet_codes:
        return np.zeros(precomputed_tr.n_buy_trades, dtype=bool)

    flag_keys = np.empty(len(flagged_wallet_codes), dtype=[("w", np.int64), ("t", np.int64)])
    flag_keys["w"] = np.asarray(flagged_wallet_codes, dtype=np.int64)
    flag_keys["t"] = np.asarray(flagged_timestamps, dtype=np.int64)
    return np.isin(precomputed_tr.trade_keys, flag_keys, assume_unique=False)

def fast_trade_level_metrics(
    precomputed_tr: PrecomputedTradeReturns,
    backtest_result: BacktestResult,
) -> Dict:
    """Compute trade-level metrics from cached returns and result flags."""
    is_flagged = build_trade_flag_mask(precomputed_tr, backtest_result)

    if not np.any(is_flagged):
        out = trade_level_zero_defaults()
        out.update({
            "flagged_trade_count": 0,
            "flagged_trade_return_sum": 0.0,
            "flagged_trade_notional_sum": 0.0,
            "unflagged_trade_count": precomputed_tr.n_buy_trades,
            "unflagged_trade_return_sum": float(np.sum(precomputed_tr.returns)),
            "unflagged_trade_notional_sum": float(np.sum(precomputed_tr.notionals)),
        })
        return out

    flagged_returns = precomputed_tr.returns[is_flagged]
    unflagged_returns = precomputed_tr.returns[~is_flagged]
    flagged_notionals = precomputed_tr.notionals[is_flagged]
    unflagged_notionals = precomputed_tr.notionals[~is_flagged]

    n_f = len(flagged_returns)
    n_u = len(unflagged_returns)

    if n_f == 0 or n_u == 0:
        _warn(
            f"fast_trade_level_metrics: all trades are {'flagged' if n_f > 0 else 'unflagged'} "
            f"for market {precomputed_tr.market_id} - metrics may be skewed."
        )
        out = compute_pooled_trade_metrics(
            flagged_returns if n_f > 0 else np.array([], dtype=np.float64),
            unflagged_returns if n_u > 0 else np.array([], dtype=np.float64),
            flagged_notionals if n_f > 0 else None,
            unflagged_notionals if n_u > 0 else None,
        )
        return out

    return compute_pooled_trade_metrics(
        flagged_returns,
        unflagged_returns,
        flagged_notionals,
        unflagged_notionals,
    )


def precompute_all_trade_returns(
    loader,
    market_ids: List[int],
) -> Dict[int, PrecomputedTradeReturns]:
    """Precompute unfiltered trade-level returns for all markets."""
    result: Dict[int, PrecomputedTradeReturns] = {}

    for market_id in market_ids:
        # Trade returns must cover the full market.
        try:
            trades = loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=None,
                use_cache=False,
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)

        metadata = dict(loader.get_market_metadata(market_id) or {})
        metadata["id"] = market_id

        ptr = precompute_trade_returns(trades=trades, market_metadata=metadata)
        if ptr is not None:
            result[market_id] = ptr

    if not experiment_backtest_logs_quiet():
        logger.info(
            f"precompute_all_trade_returns: precomputed {len(result)}/{len(market_ids)} markets "
            f"[unfiltered trade returns]"
        )
    return result


def fast_evaluate_wallets(
    precomputed: PrecomputedMarketGroundTruth,
    backtest_result: BacktestResult,
) -> List[Dict]:
    """Merge cached ground truth with config-dependent backtest fields."""
    evaluations: List[Dict] = []

    for wallet, gt in precomputed.wallet_truths.items():
        suspicion_score = float(backtest_result.wallet_suspicion.get(wallet, 0.0))
        flags = backtest_result.wallet_flags.get(wallet, [])
        num_flags = len(flags)
        cluster_boost = float(backtest_result.wallet_cluster_boost.get(wallet, 1.0))
        has_common_ownership = bool(
            backtest_result.wallet_has_common_ownership.get(wallet, False)
        )

        # trade_count is needed for flag_rate prediction mode.
        trade_count = gt.trade_count

        evaluations.append({
            "suspicion_score": suspicion_score,
            "num_flags": num_flags,
            "has_alert": num_flags > 0,
            "cluster_boost": cluster_boost,
            "has_common_ownership": has_common_ownership,

            "wallet": wallet,
            "trade_count": trade_count,
            "total_notional": gt.total_notional,
            "gross_buy_notional": gt.gross_buy_notional,
            "net_pnl": gt.net_pnl,
            "return": gt.wallet_return,
            "informed_score": gt.informed_score,
            "label_value": gt.label_value,
            "label_metric": precomputed.label_metric,
            "label_mean": gt.label_mean,
            "label_std": gt.label_std,
            "z_score": gt.z_score,
            "is_insider": gt.is_insider,
            "winning_outcome": precomputed.winning_outcome,
            "market_slug": precomputed.market_slug,
        })

    return evaluations


def precompute_all_markets(
    loader,
    market_ids: List[int],
    label_metric: str = "return",
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
) -> Dict[int, PrecomputedMarketGroundTruth]:
    """Precompute unfiltered wallet ground truth for all markets."""
    result: Dict[int, PrecomputedMarketGroundTruth] = {}

    for market_id in market_ids:
        # Ground truth must see all trades.
        try:
            trades = loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=None,
                use_cache=False,
                ignore_trade_time_bounds=True,
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)

        metadata = dict(loader.get_market_metadata(market_id) or {})
        metadata["id"] = market_id

        precomputed = precompute_ground_truth(
            trades=trades,
            market_metadata=metadata,
            label_metric=label_metric,
            z_score_threshold=z_score_threshold,
            min_wallet_notional=min_wallet_notional,
        )

        if precomputed is not None:
            result[market_id] = precomputed

    if not experiment_backtest_logs_quiet():
        logger.info(
            f"precompute_all_markets: precomputed {len(result)}/{len(market_ids)} markets "
            f"({len(market_ids) - len(result)} skipped — no resolution or too few wallets) "
            f"[unfiltered ground truth]"
        )

    return result
