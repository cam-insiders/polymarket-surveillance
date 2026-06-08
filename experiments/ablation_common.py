"""
Experiment: component ablation for detector + clustering + jump anticipation.

Given a config JSON, this script evaluates:
  - full_system
  - without_<detector> for every core detector (excluding recidivism)
  - without_clustering (if clustering is enabled in the base run)
  - without_jump_anticipation (if JA is enabled in the base run)

Usage:
    python -m experiments.ablation_common path/to/config.json \
        --start-date 2025-02-01 --end-date 2025-02-14
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from backtesting.trade_event_study import _compute_resolution_return
from backtesting.causal_boost_replay import build_live_parity_boost_schedule
from backtesting.data_loader import HistoricalDataLoader
from backtesting.evaluation import evaluate_config
from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode
from models import filter_trades_by_notional
from experiments.timeframe_market_common import (
    _normalize_category_list,
    infer_resolutions,
    select_market_ids_in_timeframe,
)


DETECTOR_NAMES: List[str] = [
    "volume_anomaly",
    "probability_impact",
    "accumulation_detector",
    "extreme_position",
    "contra_outcome_silence",
]
JA_DIAGNOSTIC_NAME = "jump_anticipation"

DETECTOR_CLASS_BY_CONFIG_NAME: Dict[str, str] = {
    "volume_anomaly": "VolumeAnomalyDetector",
    "probability_impact": "ProbabilityImpactDetector",
    "accumulation_detector": "AccumulationDetector",
    "extreme_position": "ExtremePositionDetector",
    "contra_outcome_silence": "ContraOutcomeSilenceDetector",
}

DETECTOR_CONFIG_BY_CLASS_NAME: Dict[str, str] = {
    class_name: config_name
    for config_name, class_name in DETECTOR_CLASS_BY_CONFIG_NAME.items()
}

COMPARISON_METRICS: List[str] = [
    "wallet_f1",
    "wallet_flagged_mean_net_pnl",
    "wallet_flagged_mean_return",
    "trade_copytrade_notional_portfolio_roi",
    "trade_copytrade_fixed_100_roi",
]


@dataclass
class AblationVariant:
    name: str
    component_type: str
    component_name: str
    config: Dict[str, Any]
    include_recidivism: bool
    clustering_config: Optional[Dict[str, Any]]
    jump_anticipation_config: Optional[Dict[str, Any]]


def _detector_map(config: Dict[str, Any]) -> Dict[str, Any]:
    if "detectors" in config and isinstance(config["detectors"], dict):
        return config["detectors"]
    return config


def _build_leave_one_out_config(base_config: Dict[str, Any], detector_name: str) -> Dict[str, Any]:
    """Disable one detector by setting max_confidence=0.0."""
    c = deepcopy(base_config)
    dcfg = _detector_map(c)
    detector_cfg = dcfg.setdefault(detector_name, {})
    if not isinstance(detector_cfg, dict):
        detector_cfg = {}
        dcfg[detector_name] = detector_cfg
    detector_cfg["max_confidence"] = 0.0
    return c


def _build_variants(
    base_config: Dict[str, Any],
    *,
    include_recidivism: bool,
    clustering_config: Optional[Dict[str, Any]],
    jump_anticipation_config: Optional[Dict[str, Any]],
) -> List[AblationVariant]:
    variants: List[AblationVariant] = [
        AblationVariant(
            name="full_system",
            component_type="baseline",
            component_name="full_system",
            config=deepcopy(base_config),
            include_recidivism=bool(include_recidivism),
            clustering_config=deepcopy(clustering_config),
            jump_anticipation_config=deepcopy(jump_anticipation_config),
        )
    ]

    for detector_name in DETECTOR_NAMES:
        variants.append(
            AblationVariant(
                name=f"without_{detector_name}",
                component_type="detector",
                component_name=detector_name,
                config=_build_leave_one_out_config(base_config, detector_name),
                include_recidivism=bool(include_recidivism),
                clustering_config=deepcopy(clustering_config),
                jump_anticipation_config=deepcopy(jump_anticipation_config),
            )
        )

    if clustering_config is not None:
        variants.append(
            AblationVariant(
                name="without_clustering",
                component_type="stack_component",
                component_name="clustering",
                config=deepcopy(base_config),
                include_recidivism=bool(include_recidivism),
                clustering_config=None,
                jump_anticipation_config=deepcopy(jump_anticipation_config),
            )
        )

    if jump_anticipation_config is not None:
        variants.append(
            AblationVariant(
                name="without_jump_anticipation",
                component_type="stack_component",
                component_name="jump_anticipation",
                config=deepcopy(base_config),
                include_recidivism=bool(include_recidivism),
                clustering_config=deepcopy(clustering_config),
                jump_anticipation_config=None,
            )
        )

    return variants


def _evaluate_variant(
    *,
    variant: AblationVariant,
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes_override: Dict[int, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    start = time.time()
    fixed_copytrade_size = 100.0
    if float(args.copytrade_fixed_size) != fixed_copytrade_size:
        logging.warning(
            "Ignoring --copytrade-fixed-size=%s for ablation comparison; "
            "using fixed $100 per trade for the fixed-size ROI metric.",
            args.copytrade_fixed_size,
        )

    result = evaluate_config(
        config=variant.config,
        loader=loader,
        market_ids=market_ids,
        prediction_mode=args.prediction_mode,
        flag_rate_threshold=args.flag_rate_threshold,
        suspicion_threshold=args.suspicion_threshold,
        z_score_threshold=args.z_score_threshold,
        min_wallet_notional=args.min_wallet_notional,
        min_usd_amount=args.min_usd_amount,
        include_recidivism=variant.include_recidivism,
        clustering_config=variant.clustering_config,
        clustering_min_trade_size=args.clustering_min_trade_size,
        jump_anticipation_config=variant.jump_anticipation_config,
        copytrade_fixed_size=fixed_copytrade_size,
        measure_memory=False,
        winning_outcomes_override=winning_outcomes_override,
        enable_layer2_attribution=(
            bool(args.enable_layer2_attribution) and variant.clustering_config is not None
        ),
        usdc_cache_db=args.usdc_cache,
        polygonscan_api_key=args.polygonscan_api_key,
    )
    elapsed = time.time() - start

    cs = result.copytrade_summary
    ct = result.copytrade_result
    flagged_wallet = cs.get("flagged", {})
    row = {
        "variant": variant.name,
        "component_type": variant.component_type,
        "component_name": variant.component_name,
        "include_recidivism": bool(variant.include_recidivism),
        "clustering_enabled": bool(variant.clustering_config is not None),
        "jump_anticipation_enabled": bool(variant.jump_anticipation_config is not None),
        "wallet_f1": float(cs.get("f1", 0.0)),
        "wallet_flagged_mean_net_pnl": float(flagged_wallet.get("avg_net_pnl", 0.0)),
        "wallet_flagged_mean_return": float(flagged_wallet.get("avg_return", 0.0)),
        "trade_copytrade_notional_portfolio_roi": (
            float(ct.portfolio_roi) if ct is not None else 0.0
        ),
        "trade_copytrade_fixed_100_roi": (
            float(ct.fixed_roi) if (ct is not None and ct.fixed_roi is not None) else 0.0
        ),
        "wall_clock_s": elapsed,
    }
    row.update(_flatten_detector_diagnostics(
        _build_detector_diagnostics(
            result,
            winning_outcomes_override,
            loader=loader,
            market_ids=market_ids,
            min_usd_amount=args.min_usd_amount,
            jump_anticipation_config=variant.jump_anticipation_config,
        )
    ))
    return row


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    w_sum = float(sum(weights))
    if w_sum <= 1e-9:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights)) / w_sum)


def _build_detector_diagnostics(
    result,
    winning_outcomes_override: Dict[int, int],
    *,
    loader: Optional[HistoricalDataLoader] = None,
    market_ids: Optional[List[int]] = None,
    min_usd_amount: Optional[float] = None,
    jump_anticipation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    diagnostic_names = list(DETECTOR_NAMES)
    if jump_anticipation_config is not None:
        diagnostic_names.append(JA_DIAGNOSTIC_NAME)

    diagnostics: Dict[str, Dict[str, Any]] = {
        name: {
            "signal_fires": 0,
            "alert_trades": 0,
            "alert_trades_single_detector": 0,
            "alert_trades_multi_detector": 0,
            "alert_buy_trades": 0,
            "alert_buy_total_notional": 0.0,
            "alert_buy_mean_net_pnl": 0.0,
            "alert_buy_total_net_pnl": 0.0,
            "alert_buy_mean_return": 0.0,
            "alert_buy_weighted_return": 0.0,
            "wallets_once_flagged": 0,
            "wallet_mean_net_pnl": 0.0,
            "wallet_total_net_pnl": 0.0,
            "wallet_mean_return": 0.0,
            "wallet_total_gross_buy_notional": 0.0,
        }
        for name in diagnostic_names
    }
    trade_returns: Dict[str, List[float]] = {name: [] for name in diagnostic_names}
    trade_notionals: Dict[str, List[float]] = {name: [] for name in diagnostic_names}
    detector_wallets: Dict[str, Set[Tuple[str, str]]] = {
        name: set() for name in diagnostic_names
    }

    market_slug_by_id = {
        int(market_id): str(br.market_slug or market_id)
        for market_id, br in result.backtest_results.items()
    }

    for market_id, br in result.backtest_results.items():
        mid = int(market_id)
        market_slug = market_slug_by_id.get(mid, str(mid))
        winning_outcome = winning_outcomes_override.get(mid)

        for class_name, count in br.detector_stats.items():
            config_name = DETECTOR_CONFIG_BY_CLASS_NAME.get(str(class_name))
            if config_name in diagnostics:
                diagnostics[config_name]["signal_fires"] += int(count)

        for wallet, flags in br.wallet_flags.items():
            for flag_entry in flags:
                detectors = {
                    DETECTOR_CONFIG_BY_CLASS_NAME.get(str(det), str(det))
                    for det in flag_entry.get("detectors", [])
                }
                for config_name in detectors:
                    if config_name in detector_wallets:
                        detector_wallets[config_name].add((market_slug, str(wallet)))

        for alert in br.alerts:
            detector_names = [
                DETECTOR_CONFIG_BY_CLASS_NAME.get(str(sig.detector_name), str(sig.detector_name))
                for sig in alert.signals
            ]
            detector_set = {name for name in detector_names if name in diagnostics}
            if not detector_set:
                continue

            is_single_detector_alert = len(detector_set) == 1
            for config_name in detector_set:
                diagnostics[config_name]["alert_trades"] += 1
                if is_single_detector_alert:
                    diagnostics[config_name]["alert_trades_single_detector"] += 1
                else:
                    diagnostics[config_name]["alert_trades_multi_detector"] += 1

            trade = alert.trade
            if trade.side.upper() != "BUY" or winning_outcome is None:
                continue

            ret = _compute_resolution_return(trade, int(winning_outcome))
            notional = float(trade.notional_usdc)
            for config_name in detector_set:
                diagnostics[config_name]["alert_buy_trades"] += 1
                diagnostics[config_name]["alert_buy_total_notional"] += notional
                trade_returns[config_name].append(ret)
                trade_notionals[config_name].append(notional)

    if (
        jump_anticipation_config is not None
        and loader is not None
        and market_ids is not None
    ):
        _add_jump_anticipation_diagnostics(
            diagnostics=diagnostics,
            trade_returns=trade_returns,
            trade_notionals=trade_notionals,
            detector_wallets=detector_wallets,
            result=result,
            loader=loader,
            market_ids=market_ids,
            winning_outcomes_override=winning_outcomes_override,
            min_usd_amount=min_usd_amount,
            jump_anticipation_config=jump_anticipation_config,
        )

    wallet_eval_by_key = {
        (str(e.get("market_slug", "")), str(e.get("wallet", ""))): e
        for e in result.wallet_evaluations
    }
    for config_name, wallet_keys in detector_wallets.items():
        wallet_evals = [wallet_eval_by_key[key] for key in wallet_keys if key in wallet_eval_by_key]
        wallet_pnls = [float(e.get("net_pnl", 0.0) or 0.0) for e in wallet_evals]
        wallet_returns = [float(e.get("return", 0.0) or 0.0) for e in wallet_evals]
        wallet_gross = [float(e.get("gross_buy_notional", 0.0) or 0.0) for e in wallet_evals]
        diagnostics[config_name]["wallets_once_flagged"] = len(wallet_evals)
        diagnostics[config_name]["wallet_mean_net_pnl"] = _mean(wallet_pnls)
        diagnostics[config_name]["wallet_total_net_pnl"] = float(sum(wallet_pnls))
        diagnostics[config_name]["wallet_mean_return"] = _mean(wallet_returns)
        diagnostics[config_name]["wallet_total_gross_buy_notional"] = float(sum(wallet_gross))

    for config_name in diagnostic_names:
        returns = trade_returns[config_name]
        notionals = trade_notionals[config_name]
        pnls = [r * n for r, n in zip(returns, notionals)]
        diagnostics[config_name]["alert_buy_mean_net_pnl"] = _mean(pnls)
        diagnostics[config_name]["alert_buy_total_net_pnl"] = float(sum(pnls))
        diagnostics[config_name]["alert_buy_mean_return"] = _mean(returns)
        diagnostics[config_name]["alert_buy_weighted_return"] = _weighted_mean(returns, notionals)
        alert_count = int(diagnostics[config_name]["alert_trades"])
        multi_count = int(diagnostics[config_name]["alert_trades_multi_detector"])
        diagnostics[config_name]["alert_trades_multi_detector_share"] = (
            float(multi_count / alert_count) if alert_count > 0 else 0.0
        )

    return diagnostics


def _build_flagged_keys_by_market(result) -> Dict[int, Set[Tuple[str, int]]]:
    flagged: Dict[int, Set[Tuple[str, int]]] = {}
    for market_id, br in result.backtest_results.items():
        keys: Set[Tuple[str, int]] = set()
        for wallet, flags in br.wallet_flags.items():
            w = str(wallet)
            for flag_entry in flags or []:
                keys.add((w, int(flag_entry.get("timestamp_ms", 0) or 0)))
        flagged[int(market_id)] = keys
    return flagged


def _add_jump_anticipation_diagnostics(
    *,
    diagnostics: Dict[str, Dict[str, Any]],
    trade_returns: Dict[str, List[float]],
    trade_notionals: Dict[str, List[float]],
    detector_wallets: Dict[str, Set[Tuple[str, str]]],
    result,
    loader: HistoricalDataLoader,
    market_ids: List[int],
    winning_outcomes_override: Dict[int, int],
    min_usd_amount: Optional[float],
    jump_anticipation_config: Dict[str, Any],
) -> None:
    name = JA_DIAGNOSTIC_NAME
    flagged_by_market = _build_flagged_keys_by_market(result)

    for market_id in market_ids:
        mid = int(market_id)
        br = result.backtest_results.get(mid)
        if br is None:
            continue

        try:
            all_trades = loader.get_trades_for_market(
                market_id=mid,
                min_usd_amount=None,
                use_cache=False,
            )
        except TypeError:
            all_trades = loader.get_trades_for_market(mid)

        detector_trades = (
            filter_trades_by_notional(all_trades, min_usd_amount)
            if min_usd_amount is not None
            else all_trades
        )
        if not detector_trades:
            continue

        schedule = build_live_parity_boost_schedule(
            detector_trades=detector_trades,
            market_id=str(mid),
            clustering_config=None,
            jump_anticipation_config=jump_anticipation_config,
        )
        flagged_keys = flagged_by_market.get(mid, set())
        market_slug = str(br.market_slug or mid)
        winning_outcome = winning_outcomes_override.get(mid)

        for idx, trade in enumerate(detector_trades):
            ja_multiplier = float(schedule.ja_multiplier_by_trade_idx[idx])
            if ja_multiplier <= 1.0 + 1e-6:
                continue

            diagnostics[name]["signal_fires"] += 1
            key = (str(trade.wallet), int(trade.timestamp_ms))
            if key not in flagged_keys:
                continue

            diagnostics[name]["alert_trades"] += 1
            detector_wallets[name].add((market_slug, str(trade.wallet)))

            if str(trade.side).upper() != "BUY" or winning_outcome is None:
                continue

            ret = _compute_resolution_return(trade, int(winning_outcome))
            notional = float(trade.notional_usdc)
            diagnostics[name]["alert_buy_trades"] += 1
            diagnostics[name]["alert_buy_total_notional"] += notional
            trade_returns[name].append(ret)
            trade_notionals[name].append(notional)


def _flatten_detector_diagnostics(diagnostics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for detector_name, stats in diagnostics.items():
        for metric, value in stats.items():
            row[f"det_{detector_name}_{metric}"] = value
    return row


def _add_change_after_removing_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "full_system" not in set(df["variant"].astype(str)):
        return df

    full = df.loc[df["variant"] == "full_system"].iloc[0]
    for metric in COMPARISON_METRICS:
        if metric in df.columns:
            df[f"{metric}_change_after_removing"] = df[metric] - float(full[metric])
    return df


def _add_drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return _add_change_after_removing_columns(df)


def _fmt_money(v: float) -> str:
    return f"${float(v):+,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{float(v):+.2%}"


def _print_compact_terminal_summary(df: pd.DataFrame) -> None:
    """Print a compact multi-line summary per variant for narrow terminals."""
    for row in df.to_dict(orient="records"):
        variant = str(row.get("variant", ""))
        component_type = str(row.get("component_type", ""))
        print(f"\n- {variant} [{component_type}]")
        print(
            "  wallet: "
            f"f1={float(row.get('wallet_f1', 0.0)):.4f} "
            f"(change={float(row.get('wallet_f1_change_after_removing', 0.0)):+.4f})"
        )
        print(
            "          "
            f"mean_net_pnl={_fmt_money(float(row.get('wallet_flagged_mean_net_pnl', 0.0)))} "
            f"(change={_fmt_money(float(row.get('wallet_flagged_mean_net_pnl_change_after_removing', 0.0)))})"
        )
        print(
            "          "
            f"mean_return={_fmt_pct(float(row.get('wallet_flagged_mean_return', 0.0)))} "
            f"(change={_fmt_pct(float(row.get('wallet_flagged_mean_return_change_after_removing', 0.0)))})"
        )
        print(
            "  trade : "
            f"roi_notional={_fmt_pct(float(row.get('trade_copytrade_notional_portfolio_roi', 0.0)))} "
            f"(change={_fmt_pct(float(row.get('trade_copytrade_notional_portfolio_roi_change_after_removing', 0.0)))})"
        )
        print(
            "          "
            f"roi_fixed_100={_fmt_pct(float(row.get('trade_copytrade_fixed_100_roi', 0.0)))} "
            f"(change={_fmt_pct(float(row.get('trade_copytrade_fixed_100_roi_change_after_removing', 0.0)))})"
        )


def _print_full_system_detector_diagnostics(df: pd.DataFrame) -> None:
    if df.empty or "variant" not in df.columns:
        return
    full_rows = df.loc[df["variant"].astype(str) == "full_system"]
    if full_rows.empty:
        return

    row = full_rows.iloc[0]
    print("\nFull-system detector diagnostics:")
    print(
        "  detector                     signals  alerts  multi%  buy_alerts  "
        "trade_pnl  trade_ret  wallets  wallet_pnl  wallet_ret"
    )
    display_names = list(DETECTOR_NAMES)
    if any(str(col).startswith(f"det_{JA_DIAGNOSTIC_NAME}_") for col in df.columns):
        display_names.append(JA_DIAGNOSTIC_NAME)

    for detector_name in display_names:
        prefix = f"det_{detector_name}_"
        signals = int(row.get(prefix + "signal_fires", 0) or 0)
        alerts = int(row.get(prefix + "alert_trades", 0) or 0)
        multi_share = float(row.get(prefix + "alert_trades_multi_detector_share", 0.0) or 0.0)
        buy_alerts = int(row.get(prefix + "alert_buy_trades", 0) or 0)
        trade_pnl = float(row.get(prefix + "alert_buy_mean_net_pnl", 0.0) or 0.0)
        trade_ret = float(row.get(prefix + "alert_buy_mean_return", 0.0) or 0.0)
        wallets = int(row.get(prefix + "wallets_once_flagged", 0) or 0)
        wallet_pnl = float(row.get(prefix + "wallet_mean_net_pnl", 0.0) or 0.0)
        wallet_ret = float(row.get(prefix + "wallet_mean_return", 0.0) or 0.0)
        print(
            f"  {detector_name:28s} "
            f"{signals:8,d} {alerts:7,d} {multi_share:6.1%} {buy_alerts:10,d} "
            f"{_fmt_money(trade_pnl):>10s} {_fmt_pct(trade_ret):>10s} "
            f"{wallets:7,d} {_fmt_money(wallet_pnl):>11s} {_fmt_pct(wallet_ret):>10s}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Timeframe component ablation on a fixed config: all detectors + "
            "clustering + jump anticipation."
        )
    )
    parser.add_argument("config_path", type=str)
    parser.add_argument("--output-dir", type=str, default="experiments/results/ablation")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Inclusive ISO start date (market closedTime).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Inclusive ISO end date (market closedTime).",
    )
    parser.add_argument("--resolution-threshold", type=float, default=0.99)
    parser.add_argument("--min-market-volume", type=float, default=0.0)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--inferred-resolutions-db", type=str, default="inferred_resolutions.db")
    parser.add_argument(
        "--insider-plausible-only",
        action="store_true",
        help="Filter to markets classified as insider-plausible.",
    )
    parser.add_argument(
        "--non-insider-plausible-only",
        action="store_true",
        help="Filter to markets classified as non-insider-plausible.",
    )
    parser.add_argument(
        "--market-categories",
        type=str,
        nargs="+",
        default=None,
        help="Include only these categories.",
    )
    parser.add_argument(
        "--exclude-categories",
        type=str,
        nargs="+",
        default=None,
        help="Exclude these categories.",
    )
    parser.add_argument(
        "--classifications-path",
        type=str,
        default="data/market_classifications.json",
        help="Path to market classifications JSON.",
    )

    parser.add_argument("--prediction-mode", type=str, default="flag_rate")
    parser.add_argument("--flag-rate-threshold", type=float, default=0.2)
    parser.add_argument("--suspicion-threshold", type=float, default=2.0)
    parser.add_argument("--z-score-threshold", type=float, default=2.0)
    parser.add_argument("--min-wallet-notional", type=float, default=500.0)
    parser.add_argument("--min-usd-amount", type=float, default=None)
    parser.add_argument("--clustering-min-trade-size", type=float, default=5000.0)
    parser.add_argument(
        "--copytrade-fixed-size",
        type=float,
        default=100.0,
        help="Kept for CLI compatibility; ablation comparison always uses fixed $100.",
    )

    parser.set_defaults(include_recidivism=False)
    parser.add_argument(
        "--include-recidivism",
        dest="include_recidivism",
        action="store_true",
        help="Include RecidivismDetector in the baseline/full-system run (default: off).",
    )
    parser.add_argument(
        "--exclude-recidivism",
        dest="include_recidivism",
        action="store_false",
        help="Disable RecidivismDetector in the baseline/full-system run.",
    )

    parser.add_argument(
        "--disable-clustering",
        action="store_true",
        default=False,
        help="Ignore clustering_config from the config and skip clustering ablation.",
    )
    parser.add_argument(
        "--disable-jump-anticipation",
        action="store_true",
        default=False,
        help="Ignore jump_anticipation_config from the config and skip JA ablation.",
    )

    parser.add_argument("--enable-layer2-attribution", action="store_true", default=False)
    parser.add_argument("--usdc-cache", type=str, default="data/usdc_transfers.db")
    parser.add_argument("--polygonscan-api-key", type=str, default=None)
    parser.add_argument("--verbose-output", action="store_true", default=False)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.insider_plausible_only and args.non_insider_plausible_only:
        raise SystemExit(
            "--insider-plausible-only and --non-insider-plausible-only are mutually exclusive"
        )
    args.market_categories = _normalize_category_list(args.market_categories)
    args.exclude_categories = _normalize_category_list(args.exclude_categories)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    set_experiment_backtest_log_quiet_mode(enabled=not args.verbose_output)

    with open(args.config_path, encoding="utf-8") as f:
        base_config = json.load(f)

    base_clustering_config = None if args.disable_clustering else deepcopy(base_config.get("clustering_config"))
    base_ja_config = None if args.disable_jump_anticipation else deepcopy(
        base_config.get("jump_anticipation_config")
    )

    variants = _build_variants(
        base_config=base_config,
        include_recidivism=bool(args.include_recidivism),
        clustering_config=base_clustering_config,
        jump_anticipation_config=base_ja_config,
    )

    logging.info(
        "Prepared %d ablation variants (detectors=%d, clustering=%s, jump_anticipation=%s)",
        len(variants),
        sum(1 for v in variants if v.component_type == "detector"),
        bool(base_clustering_config is not None),
        bool(base_ja_config is not None),
    )

    loader = HistoricalDataLoader(data_dir=args.data_dir, cache_size=0)
    loader.load_data()

    candidate_market_ids = select_market_ids_in_timeframe(
        loader=loader,
        start_date=args.start_date,
        end_date=args.end_date,
        min_volume=args.min_market_volume,
        classifications_path=args.classifications_path,
        insider_plausible_only=args.insider_plausible_only,
        non_insider_plausible_only=args.non_insider_plausible_only,
        market_categories=args.market_categories,
        exclude_categories=args.exclude_categories,
    )
    logging.info("Candidate markets in timeframe: %d", len(candidate_market_ids))

    winning_overrides, resolution_stats = infer_resolutions(
        loader=loader,
        market_ids=candidate_market_ids,
        resolution_threshold=args.resolution_threshold,
        min_trades=args.min_trades,
        min_usd_amount=args.min_usd_amount,
        inferred_resolutions_db=args.inferred_resolutions_db,
        save_cache=True,
    )
    market_ids = sorted(winning_overrides.keys())
    if not market_ids:
        loader.close()
        raise RuntimeError("No markets resolved in selected timeframe; nothing to ablate.")

    rows: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant in variants:
        logging.info("\n%s\nABLATION: %s\n%s", "=" * 72, variant.name, "=" * 72)
        row = _evaluate_variant(
            variant=variant,
            loader=loader,
            market_ids=market_ids,
            winning_outcomes_override=winning_overrides,
            args=args,
        )
        rows.append(row)
        logging.info(
            "  -> wallet_f1=%.4f wallet_mean_net_pnl=%+.2f wallet_mean_return=%+.4f "
            "roi_notional=%+.4f roi_fixed100=%+.4f elapsed=%.1fs",
            float(row["wallet_f1"]),
            float(row["wallet_flagged_mean_net_pnl"]),
            float(row["wallet_flagged_mean_return"]),
            float(row["trade_copytrade_notional_portfolio_roi"]),
            float(row["trade_copytrade_fixed_100_roi"]),
            float(row["wall_clock_s"]),
        )

    df = pd.DataFrame(rows)
    df = _add_change_after_removing_columns(df)

    csv_path = f"{args.output_dir}/component_ablation_{timestamp}.csv"
    meta_path = f"{args.output_dir}/component_ablation_{timestamp}_meta.json"
    df.to_csv(csv_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": str(Path(args.config_path).resolve()),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "candidate_markets": len(candidate_market_ids),
                "markets_evaluated": len(market_ids),
                "resolution_stats": resolution_stats,
                "prediction_mode": args.prediction_mode,
                "flag_rate_threshold": args.flag_rate_threshold,
                "include_recidivism": bool(args.include_recidivism),
                "clustering_enabled": bool(base_clustering_config is not None),
                "jump_anticipation_enabled": bool(base_ja_config is not None),
                "enable_layer2_attribution": bool(args.enable_layer2_attribution),
                "comparison_metrics": COMPARISON_METRICS,
                "classification_filters": {
                    "insider_plausible_only": bool(args.insider_plausible_only),
                    "non_insider_plausible_only": bool(args.non_insider_plausible_only),
                    "market_categories": args.market_categories,
                    "exclude_categories": args.exclude_categories,
                    "classifications_path": args.classifications_path,
                },
                "variants": [v.name for v in variants],
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 88}")
    print("COMPONENT ABLATION RESULTS")
    print(f"{'=' * 88}")
    print(f"Timeframe: {args.start_date} .. {args.end_date}")
    print(f"Markets:   {len(market_ids):,} resolved / {len(candidate_market_ids):,} candidates")
    _print_compact_terminal_summary(df)
    _print_full_system_detector_diagnostics(df)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {meta_path}")

    loader.close()


if __name__ == "__main__":
    main()
