"""
Z-score wallet labels and classification metrics for curated recall vs SOTA.

Ground truth: per-market wallet return z-score > threshold (same as compare_sota).
Predictions: wallet-level flags from each method on curated eval markets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import predict_wallet_positive
from experiments.curated_reported_insider_recall import _wallet_eval_from_backtest
from experiments.sota_algorithms.common import build_wallet_insider_labels
from experiments.sota_algorithms.curated_recall_flags import PerMarketFlagState
from models import Trade


def normalize_wallet(wallet: str) -> str:
    return str(wallet).strip().lower()


def build_eval_trade_entries(
    loader: HistoricalDataLoader,
    market_ids: Iterable[int],
    *,
    min_usd_amount: Optional[float] = None,
) -> List[Tuple[Trade, int]]:
    entries: List[Tuple[Trade, int]] = []
    for market_id in market_ids:
        try:
            trades = loader.get_trades_for_market(
                market_id=int(market_id),
                min_usd_amount=min_usd_amount,
                use_cache=False,
            )
        except TypeError:
            trades = loader.get_trades_for_market(int(market_id))
        for trade in trades:
            entries.append((trade, int(market_id)))
    entries.sort(key=lambda x: x[0].timestamp_ms)
    return entries


def build_zscore_wallet_labels(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes: Dict[int, int],
    *,
    z_score_threshold: float,
    min_wallet_notional: float,
    min_usd_amount: Optional[float] = None,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    entries = build_eval_trade_entries(loader, market_ids, min_usd_amount=min_usd_amount)
    winning: Dict[int, Optional[int]] = {
        int(mid): int(win) for mid, win in winning_outcomes.items()
    }
    return build_wallet_insider_labels(
        entries,
        winning,
        z_score_threshold=z_score_threshold,
        min_wallet_notional=min_wallet_notional,
    )


def count_tp_fp_fn(
    wallet_data: Dict[int, Dict[str, Dict[str, Any]]],
    market_ids: List[int],
    predicted_positive_by_market: Dict[int, Set[str]],
) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for market_id in market_ids:
        w_evals = wallet_data.get(market_id, {})
        predicted = {
            normalize_wallet(wallet)
            for wallet in predicted_positive_by_market.get(market_id, set())
        }
        for wallet, data in w_evals.items():
            is_insider = bool(data.get("is_insider", False))
            is_predicted = normalize_wallet(wallet) in predicted
            if is_predicted and is_insider:
                tp += 1
            elif is_predicted and not is_insider:
                fp += 1
            elif not is_predicted and is_insider:
                fn += 1
    return tp, fp, fn


def metrics_from_tp_fp_fn(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision_den = tp + fp
    recall_den = tp + fn
    precision = (tp / precision_den) if precision_den > 0 else 0.0
    recall = (tp / recall_den) if recall_den > 0 else 0.0
    f1_den = precision + recall
    f1 = (2.0 * precision * recall / f1_den) if f1_den > 0 else 0.0
    f05_den = 0.25 * precision + recall
    f0_5 = (1.25 * precision * recall / f05_den) if f05_den > 0 else 0.0
    return {
        "zscore_tp": int(tp),
        "zscore_fp": int(fp),
        "zscore_fn": int(fn),
        "zscore_labeled_wallets": int(tp + fn),
        "zscore_predicted_positive_wallets": int(tp + fp),
        "zscore_precision": float(precision),
        "zscore_recall": float(recall),
        "zscore_f1": float(f1),
        "zscore_f0_5": float(f0_5),
    }


def flag_volume_from_flag_state(
    flagged_by_market: Dict[int, Set[str]],
    wallet_flag_counts_by_market: Dict[int, Dict[str, int]],
) -> Dict[str, int]:
    flagged_trades = int(
        sum(
            int(count)
            for market_counts in wallet_flag_counts_by_market.values()
            for count in market_counts.values()
        )
    )
    flagged_wallet_market_pairs = int(
        sum(len(wallets) for wallets in flagged_by_market.values())
    )
    unique_wallets = len(
        {
            normalize_wallet(wallet)
            for wallets in flagged_by_market.values()
            for wallet in wallets
        }
    )
    return {
        "num_flags": flagged_trades,
        "flagged_trades": flagged_trades,
        "flagged_wallet_market_pairs": flagged_wallet_market_pairs,
        "flagged_wallets_unique": unique_wallets,
    }


def classified_wallets_from_flag_counts(
    flagged_by_market: Dict[int, Set[str]],
    wallet_flag_counts_by_market: Dict[int, Dict[str, int]],
    maker_trade_counts_by_market: Dict[int, Dict[str, int]],
    *,
    flag_rate_threshold: float,
    wallet_level_positive: bool,
) -> Dict[int, Set[str]]:
    result: Dict[int, Set[str]] = defaultdict(set)
    if wallet_level_positive:
        for market_id, wallets in flagged_by_market.items():
            result[int(market_id)] = {normalize_wallet(w) for w in wallets}
        return dict(result)

    for market_id, flag_counts in wallet_flag_counts_by_market.items():
        trade_counts = maker_trade_counts_by_market.get(int(market_id), {})
        for wallet, num_flags in flag_counts.items():
            wallet_n = normalize_wallet(wallet)
            trade_count = int(trade_counts.get(wallet_n, 0))
            if trade_count <= 0:
                continue
            wallet_eval = {
                "wallet": wallet_n,
                "trade_count": trade_count,
                "num_flags": int(num_flags),
                "has_alert": int(num_flags) > 0,
                "suspicion_score": 0.0,
                "cluster_boost": 1.0,
                "has_common_ownership": False,
            }
            if predict_wallet_positive(
                wallet_eval,
                "flag_rate",
                suspicion_threshold=2.0,
                flag_rate_threshold=flag_rate_threshold,
            ):
                result[int(market_id)].add(wallet_n)
    return dict(result)


def full_system_predicted_by_market(
    backtest_results: Dict[int, Any],
    market_ids: List[int],
    *,
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
    use_classified: bool,
) -> Dict[int, Set[str]]:
    predicted: Dict[int, Set[str]] = defaultdict(set)
    for market_id in market_ids:
        backtest_result = backtest_results.get(market_id)
        if backtest_result is None:
            continue
        trade_counts = {
            normalize_wallet(w): int(v)
            for w, v in getattr(backtest_result, "wallet_trade_counts", {}).items()
        }
        flags_by_wallet = {
            normalize_wallet(w): list(v)
            for w, v in getattr(backtest_result, "wallet_flags", {}).items()
        }
        wallets = set(trade_counts) | set(flags_by_wallet)
        for wallet in wallets:
            wallet_eval = _wallet_eval_from_backtest(
                backtest_result,
                wallet,
                participation_trade_count=trade_counts.get(wallet, 0),
            )
            if wallet_eval is None:
                continue
            if use_classified:
                positive = predict_wallet_positive(
                    wallet_eval,
                    prediction_mode,
                    suspicion_threshold,
                    flag_rate_threshold,
                )
            else:
                positive = bool(wallet_eval.get("has_alert", False))
            if positive:
                predicted[int(market_id)].add(wallet)
    return dict(predicted)


def flag_volume_from_backtest(
    backtest_results: Dict[int, Any],
    market_ids: List[int],
) -> Dict[str, int]:
    flagged_trades = 0
    flagged_wallet_market_pairs = 0
    unique_wallets: Set[str] = set()
    for market_id in market_ids:
        backtest_result = backtest_results.get(market_id)
        if backtest_result is None:
            continue
        flags_by_wallet = getattr(backtest_result, "wallet_flags", {}) or {}
        for wallet, flags in flags_by_wallet.items():
            n = len(flags)
            if n <= 0:
                continue
            wallet_n = normalize_wallet(wallet)
            flagged_trades += n
            flagged_wallet_market_pairs += 1
            unique_wallets.add(wallet_n)
    return {
        "num_flags": int(flagged_trades),
        "flagged_trades": int(flagged_trades),
        "flagged_wallet_market_pairs": int(flagged_wallet_market_pairs),
        "flagged_wallets_unique": len(unique_wallets),
    }


def build_method_zscore_and_flag_metrics(
    *,
    method: str,
    market_ids: List[int],
    wallet_data: Dict[int, Dict[str, Dict[str, Any]]],
    flag_states: Dict[str, PerMarketFlagState],
    backtest_results: Optional[Dict[int, Any]],
    maker_trade_counts_by_market: Dict[int, Dict[str, int]],
    flag_rate_threshold: float,
    prediction_mode: str,
    suspicion_threshold: float,
) -> Dict[str, Any]:
    if method == "full_system":
        if backtest_results is None:
            raise ValueError("backtest_results required for full_system metrics")
        any_flag_map = full_system_predicted_by_market(
            backtest_results,
            market_ids,
            prediction_mode=prediction_mode,
            suspicion_threshold=suspicion_threshold,
            flag_rate_threshold=flag_rate_threshold,
            use_classified=False,
        )
        classified_map = full_system_predicted_by_market(
            backtest_results,
            market_ids,
            prediction_mode=prediction_mode,
            suspicion_threshold=suspicion_threshold,
            flag_rate_threshold=flag_rate_threshold,
            use_classified=True,
        )
        flag_volume = flag_volume_from_backtest(backtest_results, market_ids)
    else:
        flagged_by_market, counts_by_market = flag_states[method]
        wallet_level = method.startswith("consob_pca") or method.startswith("mitts_ofir")
        any_flag_map = {
            int(mid): {normalize_wallet(w) for w in wallets}
            for mid, wallets in flagged_by_market.items()
        }
        classified_map = classified_wallets_from_flag_counts(
            flagged_by_market,
            counts_by_market,
            maker_trade_counts_by_market,
            flag_rate_threshold=flag_rate_threshold,
            wallet_level_positive=wallet_level,
        )
        flag_volume = flag_volume_from_flag_state(flagged_by_market, counts_by_market)

    tp_c, fp_c, fn_c = count_tp_fp_fn(wallet_data, market_ids, classified_map)
    tp_a, fp_a, fn_a = count_tp_fp_fn(wallet_data, market_ids, any_flag_map)

    out: Dict[str, Any] = dict(flag_volume)
    out.update(metrics_from_tp_fp_fn(tp_c, fp_c, fn_c))
    any_metrics = metrics_from_tp_fp_fn(tp_a, fp_a, fn_a)
    for key, value in any_metrics.items():
        if key.startswith("zscore_") and key not in ("zscore_labeled_wallets",):
            out[f"{key}_any_flag"] = value
        elif key == "zscore_labeled_wallets":
            out["zscore_labeled_wallets"] = value
    return out


def enrich_method_summaries_with_zscore_metrics(
    method_summaries: List[Dict[str, Any]],
    *,
    eval_loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes: Dict[int, int],
    flag_states: Dict[str, PerMarketFlagState],
    backtest_results: Optional[Dict[int, Any]],
    maker_trade_counts_by_market: Dict[int, Dict[str, int]],
    z_score_threshold: float,
    min_wallet_notional: float,
    min_usd_amount: Optional[float],
    flag_rate_threshold: float,
    prediction_mode: str,
    suspicion_threshold: float,
) -> List[Dict[str, Any]]:
    wallet_data = build_zscore_wallet_labels(
        eval_loader,
        market_ids,
        winning_outcomes,
        z_score_threshold=z_score_threshold,
        min_wallet_notional=min_wallet_notional,
        min_usd_amount=min_usd_amount,
    )
    enriched: List[Dict[str, Any]] = []
    for row in method_summaries:
        method = str(row["method"])
        extra = build_method_zscore_and_flag_metrics(
            method=method,
            market_ids=market_ids,
            wallet_data=wallet_data,
            flag_states=flag_states,
            backtest_results=backtest_results,
            maker_trade_counts_by_market=maker_trade_counts_by_market,
            flag_rate_threshold=flag_rate_threshold,
            prediction_mode=prediction_mode,
            suspicion_threshold=suspicion_threshold,
        )
        enriched.append({**row, **extra})
    return enriched
