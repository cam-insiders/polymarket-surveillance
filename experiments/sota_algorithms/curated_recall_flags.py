"""
Per-market flagged-wallet extraction for curated reported-insider recall vs SOTA.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from sklearn.ensemble import IsolationForest

from backtesting.data_loader import HistoricalDataLoader
from backtesting.market_resolutions import get_winning_outcome
from experiments.sota_algorithms.common import (
    build_wallet_prior_trade_counts_by_market,
    get_market_resolution_timestamp_ms,
)

from experiments.sota_algorithms.consob_pca_faithful import (
    collect_consob_pca_faithful_flags,
    parse_n_components,
)
from experiments.sota_algorithms.isolation_forest import (
    _select_matched_buy_indices,
    extract_isolation_forest_features,
)
from experiments.sota_algorithms.mitts_ofir_faithful import (
    _build_pairs_all_markets,
    _score_pairs_causal,
    _score_pairs_retrospective,
    _select_flagged_pairs,
)
from experiments.sota_algorithms.timing_heuristic import run_timing_heuristic_market_baseline
from models import Trade

PerMarketFlagState = Tuple[Dict[int, Set[str]], Dict[int, Dict[str, int]]]


def _normalize_wallet(wallet: str) -> str:
    return str(wallet).strip().lower()


def _merge_trade_flags(
    flagged_indices: Set[int],
    trades: List[Trade],
) -> Tuple[Set[str], Dict[str, int]]:
    flagged_wallets: Set[str] = set()
    wallet_flag_counts: Dict[str, int] = defaultdict(int)
    for idx in flagged_indices:
        if idx < 0 or idx >= len(trades):
            continue
        wallet = _normalize_wallet(trades[idx].wallet)
        flagged_wallets.add(wallet)
        wallet_flag_counts[wallet] += 1
    return flagged_wallets, dict(wallet_flag_counts)


def _per_market_state_from_trade_flags(
    market_ids: List[int],
    per_market_indices: Dict[int, Set[int]],
    trades_by_market: Dict[int, List[Trade]],
) -> PerMarketFlagState:
    flagged_by_market: Dict[int, Set[str]] = {}
    counts_by_market: Dict[int, Dict[str, int]] = {}
    for market_id in market_ids:
        trades = trades_by_market.get(market_id, [])
        flagged, counts = _merge_trade_flags(per_market_indices.get(market_id, set()), trades)
        flagged_by_market[market_id] = flagged
        counts_by_market[market_id] = counts
    return flagged_by_market, counts_by_market


def _load_trades_by_market(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    *,
    min_usd_amount: Optional[float],
) -> Dict[int, List[Trade]]:
    trades_by_market: Dict[int, List[Trade]] = {}
    for market_id in market_ids:
        try:
            trades = loader.get_trades_for_market(
                market_id=market_id,
                min_usd_amount=min_usd_amount,
                use_cache=False,
            )
        except TypeError:
            trades = loader.get_trades_for_market(market_id)
        trades_by_market[market_id] = list(trades)
    return trades_by_market


def _all_entries_from_trades_by_market(
    trades_by_market: Dict[int, List[Trade]],
) -> List[Tuple[Trade, int]]:
    entries: List[Tuple[Trade, int]] = []
    for market_id, trades in trades_by_market.items():
        for trade in trades:
            entries.append((trade, int(market_id)))
    entries.sort(key=lambda x: int(x[0].timestamp_ms))
    return entries


def collect_timing_heuristic_flags(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes: Dict[int, int],
    *,
    max_prior_trades: int,
    min_notional: float,
    max_hours: float,
    min_usd_amount: Optional[float],
) -> PerMarketFlagState:
    trades_by_market = _load_trades_by_market(loader, market_ids, min_usd_amount=min_usd_amount)
    prior_counts_by_market = build_wallet_prior_trade_counts_by_market(
        _all_entries_from_trades_by_market(trades_by_market)
    )
    params = {
        "max_prior_trades": int(max_prior_trades),
        "min_notional": float(min_notional),
        "max_hours": float(max_hours),
    }
    per_market_indices: Dict[int, Set[int]] = {}
    for market_id in market_ids:
        trades = trades_by_market[market_id]
        resolution_ts = get_market_resolution_timestamp_ms(loader, market_id)
        flagged_indices, _wallet_counts = run_timing_heuristic_market_baseline(
            trades=trades,
            resolution_timestamp_ms=resolution_ts,
            params=params,
            initial_wallet_trade_counts=prior_counts_by_market.get(market_id, {}),
        )
        per_market_indices[market_id] = flagged_indices
    return _per_market_state_from_trade_flags(market_ids, per_market_indices, trades_by_market)


def collect_mitts_ofir_faithful_flags(
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes: Dict[int, int],
    *,
    variant: str,
    flag_percentile: float,
    match_flag_rate: Optional[float],
    low_price_threshold: float,
    min_usd_amount: Optional[float],
    min_wallet_notional: float,
) -> PerMarketFlagState:
    """
    Pair-level flagging from the faithful Mitts & Ofir five-signal screen.

    A flagged (wallet, market) pair makes that wallet a wallet-level positive in
    the market (counts carry the pair's BUY-trade count for reference).
    """
    pairs, market_trades, resolution_ts_by_market, total_buy_eval = _build_pairs_all_markets(
        loader, market_ids, winning_outcomes, min_usd_amount, min_wallet_notional
    )
    if variant == "retrospective":
        _score_pairs_retrospective(pairs, market_trades)
        scored = pairs
    elif variant == "causal":
        scored = _score_pairs_causal(
            pairs, resolution_ts_by_market, low_price_threshold, min_wallet_notional
        )
    else:
        raise ValueError(f"Unknown faithful Mitts-Ofir variant: {variant!r}")

    flagged_pairs, _mode = _select_flagged_pairs(
        scored, flag_percentile, match_flag_rate, total_buy_eval
    )

    flagged_by_market: Dict[int, Set[str]] = {int(mid): set() for mid in market_ids}
    counts_by_market: Dict[int, Dict[str, int]] = {int(mid): {} for mid in market_ids}
    for p in flagged_pairs:
        mid = int(p["market_id"])
        wallet = _normalize_wallet(p["wallet"])
        flagged_by_market[mid].add(wallet)
        counts_by_market[mid][wallet] = (
            counts_by_market[mid].get(wallet, 0) + int(p["n_buy_trades"])
        )
    return flagged_by_market, counts_by_market


def collect_isolation_forest_flags(
    eval_loader: HistoricalDataLoader,
    eval_market_ids: List[int],
    eval_winning_outcomes: Dict[int, int],
    *,
    train_loader: HistoricalDataLoader,
    train_market_ids: Optional[List[int]],
    train_winning_outcomes: Optional[Dict[int, int]],
    n_estimators: int,
    contamination: str,
    random_state: int,
    match_flag_rate: Optional[float],
    min_usd_amount: Optional[float],
) -> PerMarketFlagState:
    eval_overrides = {int(mid): int(win) for mid, win in eval_winning_outcomes.items()}
    feature_matrix, trade_info = extract_isolation_forest_features(
        eval_loader,
        eval_market_ids,
        min_usd_amount=min_usd_amount,
        winning_outcome_overrides=eval_overrides,
    )

    model: Optional[IsolationForest] = None
    if train_market_ids and train_winning_outcomes:
        train_overrides = {int(mid): int(win) for mid, win in train_winning_outcomes.items()}
        train_matrix, train_info = extract_isolation_forest_features(
            train_loader,
            train_market_ids,
            min_usd_amount=min_usd_amount,
            winning_outcome_overrides=train_overrides,
        )
        if train_matrix.shape[0] > 0:
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(train_matrix)
            logging.info(
                "IF curated recall: fitted on %s train trades, scoring %s eval trades.",
                f"{train_matrix.shape[0]:,}",
                f"{feature_matrix.shape[0]:,}",
            )

    if feature_matrix.shape[0] == 0:
        empty = {mid: set() for mid in eval_market_ids}
        return empty, {mid: {} for mid in eval_market_ids}

    if model is None:
        if train_market_ids:
            logging.warning(
                "IF: train window produced no usable train trades; "
                "fitting on %d curated eval market(s) only.",
                len(eval_market_ids),
            )
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(feature_matrix)

    scores = model.decision_function(feature_matrix)
    buy_eval_indices = [
        i
        for i, info in enumerate(trade_info)
        if info["trade"].side.upper() == "BUY" and info["winning_outcome"] is not None
    ]
    n_buy_eval = len(buy_eval_indices)
    natural_predictions = model.predict(feature_matrix)
    flagged_indices: Set[int] = set()

    if match_flag_rate is not None and n_buy_eval > 0:
        calibration_scores = None
        calibration_buy_indices = None
        if model is not None and train_market_ids and train_winning_outcomes:
            train_overrides = {int(mid): int(win) for mid, win in train_winning_outcomes.items()}
            train_matrix, train_info = extract_isolation_forest_features(
                train_loader,
                train_market_ids,
                min_usd_amount=min_usd_amount,
                winning_outcome_overrides=train_overrides,
            )
            if train_matrix.shape[0] > 0:
                calibration_scores = model.decision_function(train_matrix)
                calibration_buy_indices = [
                    i
                    for i, info in enumerate(train_info)
                    if info["trade"].side.upper() == "BUY"
                    and info["winning_outcome"] is not None
                ]
        flagged_indices, _threshold_source = _select_matched_buy_indices(
            scores,
            buy_eval_indices,
            match_flag_rate,
            calibration_scores=calibration_scores,
            calibration_buy_indices=calibration_buy_indices,
        )
    else:
        flagged_indices = {i for i, pred in enumerate(natural_predictions) if pred == -1}

    flagged_by_market: Dict[int, Set[str]] = defaultdict(set)
    counts_by_market: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, info in enumerate(trade_info):
        if i not in flagged_indices:
            continue
        market_id = int(info["market_id"])
        wallet = _normalize_wallet(info["trade"].wallet)
        flagged_by_market[market_id].add(wallet)
        counts_by_market[market_id][wallet] += 1

    for market_id in eval_market_ids:
        flagged_by_market.setdefault(market_id, set())
        counts_by_market.setdefault(market_id, {})

    return (
        {mid: set(flagged_by_market.get(mid, set())) for mid in eval_market_ids},
        {mid: dict(counts_by_market.get(mid, {})) for mid in eval_market_ids},
    )


def collect_all_curated_sota_flags(
    eval_loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes: Dict[int, int],
    args: Any,
    *,
    train_loader: HistoricalDataLoader,
    train_market_ids: List[int],
    train_winning_outcomes: Dict[int, int],
    match_flag_rate: Optional[float],
) -> Dict[str, PerMarketFlagState]:
    """Return baseline_name -> (flagged_wallets_by_market, wallet_flag_counts_by_market)."""
    results: Dict[str, PerMarketFlagState] = {}

    results["timing_heuristic"] = collect_timing_heuristic_flags(
        eval_loader,
        market_ids,
        winning_outcomes,
        max_prior_trades=args.timing_max_prior_trades,
        min_notional=args.timing_min_notional,
        max_hours=args.timing_max_hours,
        min_usd_amount=args.min_usd_amount,
    )

    results["mitts_ofir_retrospective"] = collect_mitts_ofir_faithful_flags(
        eval_loader,
        market_ids,
        winning_outcomes,
        variant="retrospective",
        flag_percentile=args.mo_faithful_flag_percentile,
        match_flag_rate=match_flag_rate,
        low_price_threshold=args.mo_faithful_low_price_threshold,
        min_usd_amount=args.min_usd_amount,
        min_wallet_notional=args.min_wallet_notional,
    )
    results["mitts_ofir_causal"] = collect_mitts_ofir_faithful_flags(
        eval_loader,
        market_ids,
        winning_outcomes,
        variant="causal",
        flag_percentile=args.mo_faithful_flag_percentile,
        match_flag_rate=match_flag_rate,
        low_price_threshold=args.mo_faithful_low_price_threshold,
        min_usd_amount=args.min_usd_amount,
        min_wallet_notional=args.min_wallet_notional,
    )

    # CONSOB / Ravagnani is event-anchored and fits PCA per market on
    # that market's own eligible wallet trajectories (no train window). It is
    # always routed through the per-market eval-fit path.
    logging.info(
        "CONSOB PCA: per-market eval-fit on %d curated eval market(s).",
        len(market_ids),
    )
    results["consob_pca"] = collect_consob_pca_faithful_flags(
        eval_loader,
        market_ids,
        winning_outcomes,
        bucket_hours=getattr(args, "consob_bucket_hours", 6),
        investigation_hours=getattr(args, "consob_investigation_hours", 24),
        d_theta=getattr(args, "consob_d_theta", 3),
        n_components=parse_n_components(getattr(args, "consob_n_components", 3)),
        min_wallets_for_kde=getattr(args, "consob_min_wallets_for_kde", 8),
        percentile_fallback=getattr(args, "consob_percentile_fallback", 90.0),
        min_wallet_notional=args.min_wallet_notional,
        min_usd_amount=args.min_usd_amount,
    )

    results["isolation_forest"] = collect_isolation_forest_flags(
        eval_loader,
        market_ids,
        winning_outcomes,
        train_loader=train_loader,
        train_market_ids=train_market_ids or None,
        train_winning_outcomes=train_winning_outcomes or None,
        n_estimators=args.if_n_estimators,
        contamination=args.if_contamination,
        random_state=args.if_random_state,
        match_flag_rate=None,
        min_usd_amount=args.min_usd_amount,
    )

    return results
