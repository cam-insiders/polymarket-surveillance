"""
Experiment: reported-insider recall on curated markets vs full system and SOTA baselines.

Usage:
    python3 -m experiments.curated_sota_common path/to/config.json

    python3 -m experiments.curated_sota_common path/to/config.json \\
        --data-dir data/curated_fromvm \\
        --train-start 2025-02-01 --train-end 2025-02-14

    # Train-window selection/fit uses ``data/`` by default; override with --train-data-dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import DEFAULT_CLUSTERING_CONFIG, evaluate_config
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode
from backtesting.market_resolutions import get_winning_outcome
from experiments.timeframe_market_common import infer_resolutions, select_market_ids_in_timeframe
from experiments.curated_reported_insider_recall import (
    CURATED_CASES,
    DEFAULT_OUTPUT_DIR,
    _build_arg_parser as _build_base_arg_parser,
    _build_winning_outcomes,
    _validate_curated_cases,
    build_maker_trade_counts_by_market,
    build_reported_wallet_participation_counts,
    build_reported_wallet_rows,
    build_reported_wallet_rows_from_sota_flags,
    get_counted_reported_wallets,
    is_case_available_in_loader,
    summarize_all_method_market_rows,
    summarize_aggregate,
    summarize_market_rows,
    summarize_method_rows,
)
from experiments.curated_recall_zscore_metrics import enrich_method_summaries_with_zscore_metrics
from experiments.sota_algorithms.curated_recall_flags import (
    PerMarketFlagState,
    collect_all_curated_sota_flags,
)

DEFAULT_COMPARE_OUTPUT_DIR = "experiments/results/curated_reported_insider_recall_compare_sota"
DEFAULT_TRAIN_DATA_DIR = "data"


def prepare_sota_train_markets(
    train_market_ids: List[int],
    train_winning_outcomes: Dict[int, int],
    eval_market_ids: List[int],
) -> Tuple[List[int], Dict[int, int]]:
    """
    Drop curated eval markets from the train fit set so IF never trains on test markets.
    """
    eval_set = {int(mid) for mid in eval_market_ids}
    overlap = sorted(int(mid) for mid in train_market_ids if int(mid) in eval_set)
    if overlap:
        logging.warning(
            "Excluding %d curated eval market(s) from SOTA train fit (leakage guard): %s",
            len(overlap),
            overlap,
        )
    filtered_ids = [int(mid) for mid in train_market_ids if int(mid) not in eval_set]
    filtered_winners = {
        int(mid): int(win)
        for mid, win in train_winning_outcomes.items()
        if int(mid) in filtered_ids
    }
    return filtered_ids, filtered_winners


def add_compare_sota_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=DEFAULT_TRAIN_DATA_DIR,
        help=(
            "Data directory for SOTA train-window market selection and IF fitting "
            f"(default: {DEFAULT_TRAIN_DATA_DIR}). Curated eval/backtests use --data-dir."
        ),
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="Inclusive train-window start for IF (ISO date).",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Inclusive train-window end for IF (ISO date).",
    )
    parser.add_argument("--min-market-volume", type=float, default=0.0)
    parser.add_argument("--if-n-estimators", type=int, default=200)
    parser.add_argument("--if-contamination", type=str, default="auto")
    parser.add_argument("--if-random-state", type=int, default=42)
    parser.add_argument("--timing-max-prior-trades", type=int, default=5)
    parser.add_argument("--timing-min-notional", type=float, default=5000.0)
    parser.add_argument("--timing-max-hours", type=float, default=48.0)
    parser.add_argument("--mo-faithful-flag-percentile", type=float, default=5.0)
    parser.add_argument("--mo-faithful-low-price-threshold", type=float, default=0.15)
    parser.add_argument("--consob-n-components", type=str, default="3")
    parser.add_argument("--consob-bucket-hours", type=int, default=6)
    parser.add_argument("--consob-investigation-hours", type=int, default=24)
    parser.add_argument("--consob-d-theta", type=int, default=3)
    parser.add_argument("--consob-min-wallets-for-kde", type=int, default=8)
    parser.add_argument("--consob-percentile-fallback", type=float, default=90.0)


def get_train_loader_for_sota(
    args: argparse.Namespace,
    eval_loader: HistoricalDataLoader,
) -> Tuple[HistoricalDataLoader, bool]:
    """
    Return (train_loader, must_close).
    """
    if not args.train_start or not args.train_end:
        return eval_loader, False

    train_dir = str(getattr(args, "train_data_dir", DEFAULT_TRAIN_DATA_DIR) or DEFAULT_TRAIN_DATA_DIR)
    eval_dir = str(args.data_dir)
    if Path(train_dir).resolve() == Path(eval_dir).resolve():
        logging.info("SOTA train data same as --data-dir: %s", train_dir)
        return eval_loader, False

    train_loader = HistoricalDataLoader(data_dir=train_dir, cache_size=0)
    train_loader.load_data()
    logging.info("SOTA train data: %s | curated eval data: %s", train_dir, eval_dir)
    return train_loader, True


def resolve_train_markets_for_sota(
    train_loader: HistoricalDataLoader,
    args: argparse.Namespace,
) -> Tuple[List[int], Dict[int, int]]:
    if not args.train_start or not args.train_end:
        return [], {}

    train_market_ids = select_market_ids_in_timeframe(
        train_loader,
        args.train_start,
        args.train_end,
        args.min_market_volume,
    )
    if not train_market_ids:
        logging.warning(
            "Train window %s..%s selected 0 markets in train data dir %s; "
            "IF will fit on curated eval markets only.",
            args.train_start,
            args.train_end,
            getattr(args, "train_data_dir", DEFAULT_TRAIN_DATA_DIR),
        )
        return [], {}

    inferred, stats = infer_resolutions(
        loader=train_loader,
        market_ids=train_market_ids,
        resolution_threshold=args.resolution_threshold,
        min_trades=args.min_trades,
        min_usd_amount=getattr(args, "min_usd_amount", None),
        inferred_resolutions_db=args.inferred_resolutions_db,
        save_cache=True,
    )
    logging.info(
        "SOTA train markets in %s: %s selected, %s resolved (inference stats: %s)",
        getattr(args, "train_data_dir", DEFAULT_TRAIN_DATA_DIR),
        len(train_market_ids),
        len(inferred),
        stats,
    )
    return train_market_ids, inferred


def collect_sota_wallet_rows(
    *,
    eval_loader: HistoricalDataLoader,
    train_loader: HistoricalDataLoader,
    curated_cases: List[Dict[str, Any]],
    market_ids: List[int],
    winning_outcomes: Dict[int, int],
    participation_counts: Dict[Tuple[int, str], int],
    maker_trade_counts: Dict[int, Dict[str, int]],
    args: argparse.Namespace,
    train_market_ids: List[int],
    train_winning_outcomes: Dict[int, int],
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
    match_flag_rate: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, PerMarketFlagState]]:
    train_winning: Dict[int, int] = dict(train_winning_outcomes)
    for mid in train_market_ids:
        if mid not in train_winning:
            static = get_winning_outcome(mid)
            if static is not None:
                train_winning[mid] = int(static)

    baseline_train_ids, train_winning = prepare_sota_train_markets(
        train_market_ids,
        train_winning,
        market_ids,
    )
    baseline_train_ids = sorted(baseline_train_ids)
    if args.train_start and args.train_end:
        if not train_market_ids:
            logging.warning(
                "Train window %s..%s matched 0 markets in train data dir %s; "
                "IF will fit on curated eval markets only.",
                args.train_start,
                args.train_end,
                getattr(args, "train_data_dir", DEFAULT_TRAIN_DATA_DIR),
            )
        elif not baseline_train_ids:
            logging.warning(
                "After excluding curated eval markets, 0 train markets remain; "
                "IF will fit on curated eval markets only.",
            )
        else:
            logging.info(
                "SOTA train fit: %d market(s) (eval curated markets excluded from train).",
                len(baseline_train_ids),
            )

    flag_states = collect_all_curated_sota_flags(
        eval_loader,
        market_ids,
        winning_outcomes,
        args,
        train_loader=train_loader,
        train_market_ids=baseline_train_ids,
        train_winning_outcomes=train_winning,
        match_flag_rate=match_flag_rate,
    )

    rows: List[Dict[str, Any]] = []
    for method, (flagged_by_market, counts_by_market) in flag_states.items():
        wallet_level = method.startswith("consob_pca") or method.startswith("mitts_ofir")
        method_rows = build_reported_wallet_rows_from_sota_flags(
            curated_cases=curated_cases,
            participation_counts=participation_counts,
            flagged_wallets_by_market=flagged_by_market,
            wallet_flag_counts_by_market=counts_by_market,
            maker_trade_counts_by_market=maker_trade_counts,
            method=method,
            prediction_mode=prediction_mode,
            suspicion_threshold=suspicion_threshold,
            flag_rate_threshold=flag_rate_threshold,
            wallet_level_positive=wallet_level,
        )
        logging.info("SOTA recall rows for %s: %d wallet-market pairs", method, len(method_rows))
        rows.extend(method_rows)
    return rows, flag_states


def _build_arg_parser() -> argparse.ArgumentParser:
    # Base parser already registers SOTA/train-window flags via add_compare_sota_args.
    parser = _build_base_arg_parser()
    parser.description = (
        "Curated reported-insider recall for full system and SOTA baselines."
    )
    parser.set_defaults(output_dir=DEFAULT_COMPARE_OUTPUT_DIR)
    return parser


def _print_method_table(method_summaries: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 88}")
    print("RECALL BY METHOD (reported insiders present in trade data)")
    print(f"{'=' * 88}")
    for row in method_summaries:
        print(
            f"  {row['method']:40s} "
            f"classified={row['classified_reported_wallets']}/{row['reported_wallets_present']} "
            f"({row['classification_recall_present']:.1%})  "
            f"ever_flagged={row['ever_flagged_reported_wallets']}/{row['reported_wallets_present']} "
            f"({row['ever_flagged_recall_present']:.1%})"
        )
        print(
            f"  {'unique wallets':40s} "
            f"classified={row.get('unique_classified_reported_wallets', 0)}/"
            f"{row.get('unique_reported_wallets_present', 0)} "
            f"({row.get('unique_classification_recall_present', 0.0):.1%})  "
            f"ever_flagged={row.get('unique_ever_flagged_reported_wallets', 0)}/"
            f"{row.get('unique_reported_wallets_present', 0)} "
            f"({row.get('unique_ever_flagged_recall_present', 0.0):.1%})"
        )

    print(f"\n{'=' * 88}")
    print("Z-SCORE WALLET LABELS (all wallets with min notional on curated markets)")
    print(f"{'=' * 88}")
    for row in method_summaries:
        print(
            f"  {row['method']:40s} "
            f"P={row.get('zscore_precision', 0.0):.3f} R={row.get('zscore_recall', 0.0):.3f} "
            f"F1={row.get('zscore_f1', 0.0):.3f} F0.5={row.get('zscore_f0_5', 0.0):.3f} "
            f"(TP={row.get('zscore_tp', 0)} FP={row.get('zscore_fp', 0)} FN={row.get('zscore_fn', 0)}) "
            f"flags={row.get('num_flags', row.get('flagged_trades', 0)):,} "
            f"wallets={row.get('flagged_wallet_market_pairs', 0):,} "
            f"unique_wallets={row.get('flagged_wallets_unique', 0):,}"
        )
    print(
        "\n  zscore_* uses classified-positive wallets; "
        "zscore_*_any_flag uses any flagged trade (compare_sota style)."
    )


def _print_method_market_sections(method_market_rows: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 88}")
    print("PER-MARKET CLASSIFICATION BY METHOD")
    print(f"{'=' * 88}")

    methods = sorted({str(row.get("method", "full_system")) for row in method_market_rows})
    for method in methods:
        rows = [row for row in method_market_rows if str(row.get("method", "full_system")) == method]
        rows = sorted(
            rows,
            key=lambda r: (
                float(r.get("classification_rate_present", 0.0)),
                int(r.get("reported_wallets_present", 0)),
                int(r.get("market_id", 0)),
            ),
        )
        print(f"\n{method}")
        for row in rows:
            present = int(row.get("reported_wallets_present", 0))
            classified = int(row.get("classified_reported_wallets", 0))
            ever_flagged = int(row.get("ever_flagged_reported_wallets", 0))
            rate = float(row.get("classification_rate_present", 0.0))
            flag_rate = float(row.get("mean_reported_wallet_flag_rate_present", 0.0))
            slug = str(row.get("market_slug", row.get("market_id", "")))[:50]
            print(
                f"  {int(row['market_id'])} {slug:50s} "
                f"classified={classified}/{present} ({rate:.1%})  "
                f"ever_flagged={ever_flagged}/{present}  "
                f"mean_flag_rate={flag_rate:.1%}"
            )


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
    for warning in _validate_curated_cases(loader, cases):
        logging.warning(warning)

    winning_outcomes, resolution_meta = _build_winning_outcomes(
        loader=loader,
        cases=cases,
        args=args,
    )
    cases_with_targets = [case for case in cases if get_counted_reported_wallets(case)]
    market_ids = [
        int(case["market_id"])
        for case in cases_with_targets
        if (
            is_case_available_in_loader(loader, case)
            and int(case["market_id"]) in winning_outcomes
            and loader.get_market_metadata(int(case["market_id"]))
        )
    ]
    if not market_ids:
        loader.close()
        raise RuntimeError("No curated markets had local metadata and a known/inferred resolution.")

    clustering_config = None if args.no_clustering else config.get("clustering_config", DEFAULT_CLUSTERING_CONFIG)
    jump_anticipation_config = None if args.no_jump_anticipation else config.get("jump_anticipation_config")

    logging.info(
        "Full system on %d curated market(s), prediction_mode=%s",
        len(market_ids),
        args.prediction_mode,
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
    maker_trade_counts = build_maker_trade_counts_by_market(
        loader,
        market_ids,
        min_usd_amount=args.min_usd_amount,
    )

    wallet_rows = build_reported_wallet_rows(
        curated_cases=cases,
        backtest_results=result.backtest_results,
        participation_counts=participation_counts,
        prediction_mode=args.prediction_mode,
        suspicion_threshold=args.suspicion_threshold,
        flag_rate_threshold=args.flag_rate_threshold,
    )

    match_flag_rate: Optional[float] = None
    flagged_buy = sum(int(row.get("num_flags", 0)) for row in wallet_rows if row.get("has_alert"))
    buy_trades = sum(int(row.get("trade_count", 0)) for row in wallet_rows)
    if buy_trades > 0:
        match_flag_rate = flagged_buy / buy_trades
        logging.info("IF matched flag rate from full system reported wallets: %.4f", match_flag_rate)

    train_market_ids: List[int] = []
    train_winning_outcomes: Dict[int, int] = {}
    flag_states: Dict[str, PerMarketFlagState] = {}
    train_loader, close_train_loader = get_train_loader_for_sota(args, loader)
    try:
        train_market_ids, train_winning_outcomes = resolve_train_markets_for_sota(
            train_loader=train_loader,
            args=args,
        )
        sota_rows, flag_states = collect_sota_wallet_rows(
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
        wallet_rows.extend(sota_rows)
    finally:
        if close_train_loader:
            train_loader.close()

    method_summaries = summarize_method_rows(cases, wallet_rows)
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
    full_system_summary = next(
        (row for row in method_summaries if row["method"] == "full_system"),
        summarize_aggregate(wallet_rows, summarize_market_rows(cases, wallet_rows)),
    )
    method_market_rows = summarize_all_method_market_rows(cases, wallet_rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wallet_path = f"{args.output_dir}/curated_recall_compare_wallets_{ts}.csv"
    market_path = f"{args.output_dir}/curated_recall_compare_method_markets_{ts}.csv"
    method_path = f"{args.output_dir}/curated_recall_compare_methods_{ts}.csv"
    summary_path = f"{args.output_dir}/curated_recall_compare_summary_{ts}.json"

    pd.DataFrame(wallet_rows).to_csv(wallet_path, index=False)
    pd.DataFrame(method_market_rows).to_csv(market_path, index=False)
    pd.DataFrame(method_summaries).to_csv(method_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": str(config_path.resolve()),
                "prediction_mode": args.prediction_mode,
                "flag_rate_threshold": args.flag_rate_threshold,
                "data_dir": args.data_dir,
                "train_data_dir": getattr(args, "train_data_dir", DEFAULT_TRAIN_DATA_DIR),
                "train_start": args.train_start,
                "train_end": args.train_end,
                "train_markets": len(train_market_ids),
                "market_ids_evaluated": market_ids,
                "match_flag_rate": match_flag_rate,
                "z_score_threshold": args.z_score_threshold,
                "min_wallet_notional": args.min_wallet_notional,
                "resolution_meta": resolution_meta,
                "full_system_aggregate": full_system_summary,
                "method_summaries": method_summaries,
                "method_market_summaries": method_market_rows,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 88}")
    print("CURATED REPORTED-INSIDER RECALL (FULL SYSTEM + SOTA)")
    print(f"{'=' * 88}")
    print(f"Markets evaluated: {len(market_ids):,}")
    _print_method_table(method_summaries)
    _print_method_market_sections(method_market_rows)

    print("\nSaved:")
    print(f"  - {wallet_path}")
    print(f"  - {market_path}")
    print(f"  - {method_path}")
    print(f"  - {summary_path}")
    loader.close()


if __name__ == "__main__":
    main()
