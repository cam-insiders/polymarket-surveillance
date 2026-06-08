"""Single-config backtest evaluation and metric aggregation."""

from __future__ import annotations

import json
import logging
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.backtest_runner import BacktestResult, BacktestRunner
from backtesting.bucket_clustering_backtest_runner import BucketClusteringBacktestRunner
from backtesting.causal_boost_replay import build_live_parity_boost_schedule
from models import filter_trades_by_notional
from backtesting.evaluation_support import (
    build_attribution_provider,
    evaluate_wallets_with_ground_truth,
    load_all_trades_for_market,
)
from backtesting.data_loader import HistoricalDataLoader
from backtesting.market_resolutions import get_market_info, get_resolved_market_ids, get_winning_outcome
from backtesting.trade_event_study import (
    TradeEventStudyResult,
    CopytradeResult,
    run_copytrade_simulation,
    run_copytrade_simulation_multi,
    run_trade_event_study,
    run_trade_event_study_multi,
)
from backtesting.wallet_evaluator import WalletEvaluator
from jump_anticipation.core import run_jump_anticipation_boost

logger = logging.getLogger(__name__)


@dataclass
class MarketPerformance:
    """Per-market timing breakdown. One per market in the evaluation."""
    market_id: int
    market_slug: str
    n_trades: int
    n_wallets: int
    n_alerts: int
    wall_clock_seconds: float
    trades_per_second: float

    # Detection-only latency (detectors + Noisy-OR, excludes wallet bookkeeping)
    detection_latency_mean_us: float
    detection_latency_p50_us: float
    detection_latency_p95_us: float
    detection_latency_p99_us: float

    # Full per-trade latency (detection + wallet bookkeeping + flagging)
    total_latency_mean_us: float
    total_latency_p50_us: float
    total_latency_p95_us: float
    total_latency_p99_us: float


@dataclass
class AggregatePerformance:
    """Pooled performance stats across all markets."""
    total_trades: int
    total_markets: int
    total_wall_clock_seconds: float
    overall_trades_per_second: float
    peak_memory_mb: float              # 0.0 if memory measurement disabled

    # Pooled latency percentiles (concatenated across all markets)
    detection_latency_mean_us: float
    detection_latency_p50_us: float
    detection_latency_p95_us: float
    detection_latency_p99_us: float
    total_latency_mean_us: float
    total_latency_p50_us: float
    total_latency_p95_us: float
    total_latency_p99_us: float


def _latency_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute mean + percentile stats from a latency array (microseconds)."""
    if len(arr) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _build_market_performance(
    market_id: int,
    market_slug: str,
    backtest_result: BacktestResult,
) -> MarketPerformance:
    """Extract performance metrics from a BacktestResult."""
    n = backtest_result.total_trades
    wc = backtest_result.wall_clock_seconds

    det = _latency_stats(backtest_result.detection_latencies_us
                         if backtest_result.detection_latencies_us is not None
                         else np.array([]))
    tot = _latency_stats(backtest_result.total_latencies_us
                         if backtest_result.total_latencies_us is not None
                         else np.array([]))

    return MarketPerformance(
        market_id=market_id,
        market_slug=market_slug,
        n_trades=n,
        n_wallets=len(backtest_result.wallet_suspicion),
        n_alerts=backtest_result.alerts_generated,
        wall_clock_seconds=wc,
        trades_per_second=(n / wc) if wc > 1e-9 else 0.0,
        detection_latency_mean_us=det["mean"],
        detection_latency_p50_us=det["p50"],
        detection_latency_p95_us=det["p95"],
        detection_latency_p99_us=det["p99"],
        total_latency_mean_us=tot["mean"],
        total_latency_p50_us=tot["p50"],
        total_latency_p95_us=tot["p95"],
        total_latency_p99_us=tot["p99"],
    )


def _build_aggregate_performance(
    market_perfs: List[MarketPerformance],
    backtest_results: Dict[int, BacktestResult],
    peak_memory_mb: float,
) -> AggregatePerformance:
    """Pool performance stats across all markets."""
    total_trades = sum(mp.n_trades for mp in market_perfs)
    total_wc = sum(mp.wall_clock_seconds for mp in market_perfs)

    # Concatenate all latency arrays for pooled percentiles
    all_det = []
    all_tot = []
    for br in backtest_results.values():
        if br.detection_latencies_us is not None:
            all_det.append(br.detection_latencies_us)
        if br.total_latencies_us is not None:
            all_tot.append(br.total_latencies_us)

    pooled_det = np.concatenate(all_det) if all_det else np.array([])
    pooled_tot = np.concatenate(all_tot) if all_tot else np.array([])

    det = _latency_stats(pooled_det)
    tot = _latency_stats(pooled_tot)

    return AggregatePerformance(
        total_trades=total_trades,
        total_markets=len(market_perfs),
        total_wall_clock_seconds=total_wc,
        overall_trades_per_second=(total_trades / total_wc) if total_wc > 1e-9 else 0.0,
        peak_memory_mb=peak_memory_mb,
        detection_latency_mean_us=det["mean"],
        detection_latency_p50_us=det["p50"],
        detection_latency_p95_us=det["p95"],
        detection_latency_p99_us=det["p99"],
        total_latency_mean_us=tot["mean"],
        total_latency_p50_us=tot["p50"],
        total_latency_p95_us=tot["p95"],
        total_latency_p99_us=tot["p99"],
    )


# ---------------------------------------------------------------------------
# Wallet prediction helper
# ---------------------------------------------------------------------------

def predict_wallet_positive(
    e: Dict,
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
) -> bool:
    """Apply prediction thresholding to a single wallet evaluation dict."""
    if prediction_mode == "has_alert":
        return bool(e.get("has_alert", False))
    if prediction_mode == "suspicion_threshold":
        return float(e.get("suspicion_score", 0.0)) >= float(suspicion_threshold)
    if prediction_mode == "flag_rate":
        tc = int(e.get("trade_count", 0) or 0)
        nf = int(e.get("num_flags", 0) or 0)
        rate = (nf / tc) if tc > 0 else 0.0
        return rate >= float(flag_rate_threshold)
    if prediction_mode == "boosted_flag_rate":
        # Backward-compatible alias: in causal trade-level mode, boosts are
        # already reflected in per-trade alert decisions.
        tc = int(e.get("trade_count", 0) or 0)
        nf = int(e.get("num_flags", 0) or 0)
        rate = (nf / tc) if tc > 0 else 0.0
        return rate >= float(flag_rate_threshold)
    raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")

def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    m = n // 2
    return float(s[m]) if n % 2 == 1 else float((s[m - 1] + s[m]) / 2.0)


def _cohort_stats(rows: List[Dict]) -> Dict:
    if not rows:
        return {"count": 0, "avg_return": 0.0, "median_return": 0.0,
                "weighted_return": 0.0, "avg_net_pnl": 0.0, "median_net_pnl": 0.0,
                "total_gross_buy": 0.0}

    returns = [float(r.get("return", 0.0)) for r in rows]
    pnls = [float(r.get("net_pnl", 0.0)) for r in rows]
    weights = [max(0.0, float(r.get("gross_buy_notional", 0.0))) for r in rows]
    w_sum = sum(weights)

    return {
        "count": len(rows),
        "avg_return": sum(returns) / len(returns),
        "median_return": _median(returns),
        "weighted_return": (sum(r * w for r, w in zip(returns, weights)) / w_sum) if w_sum > 0 else 0.0,
        "avg_net_pnl": sum(pnls) / len(pnls),
        "median_net_pnl": _median(pnls),
        "total_gross_buy": float(w_sum),
    }


def build_copytrade_report(
    wallet_evaluations: List[Dict],
    prediction_mode: str,
    suspicion_threshold: float,
    flag_rate_threshold: float,
) -> Tuple[Dict, pd.DataFrame]:
    """Build wallet-level copytrade economic report. Returns (summary_dict, per_market_df)."""
    cohorts = {"flagged": [], "tp": [], "fp": [], "fn": []}
    market_buckets: Dict[str, Dict[str, List[Dict]]] = {}

    for e in wallet_evaluations:
        slug = str(e.get("market_slug", "unknown"))
        b = market_buckets.setdefault(
            slug, {"flagged": [], "tp": [], "fp": [], "fn": [], "insiders": []})

        is_insider = bool(e.get("is_insider", False))
        pred_pos = predict_wallet_positive(e, prediction_mode, suspicion_threshold, flag_rate_threshold)

        if is_insider:
            b["insiders"].append(e)
        if pred_pos:
            cohorts["flagged"].append(e)
            b["flagged"].append(e)
            if is_insider:
                cohorts["tp"].append(e)
                b["tp"].append(e)
            else:
                cohorts["fp"].append(e)
                b["fp"].append(e)
        elif is_insider:
            cohorts["fn"].append(e)
            b["fn"].append(e)

    summary = {name: _cohort_stats(rows) for name, rows in cohorts.items()}

    market_rows = []
    for slug, b in market_buckets.items():
        fs = _cohort_stats(b["flagged"])
        ts = _cohort_stats(b["tp"])
        fps = _cohort_stats(b["fp"])
        fns = _cohort_stats(b["fn"])
        fn_count = len(b["flagged"])
        in_count = len(b["insiders"])
        tp_count = len(b["tp"])

        market_rows.append({
            "market_slug": slug,
            "flagged_wallets": fn_count, "insider_wallets": in_count,
            "tp": tp_count, "fp": len(b["fp"]), "fn": len(b["fn"]),
            "precision": (tp_count / fn_count) if fn_count > 0 else 0.0,
            "recall": (tp_count / in_count) if in_count > 0 else 0.0,
            "flagged_avg_return": fs["avg_return"],
            "flagged_weighted_return": fs["weighted_return"],
            "tp_avg_return": ts["avg_return"],
            "fp_avg_return": fps["avg_return"],
            "fn_avg_return": fns["avg_return"],
            "flagged_avg_net_pnl": fs["avg_net_pnl"],
        })

    per_market_df = pd.DataFrame(market_rows)
    if not per_market_df.empty:
        per_market_df = per_market_df.sort_values("flagged_weighted_return", ascending=False)

    return summary, per_market_df

@dataclass
class EvaluationResult:
    """
    Complete evaluation output from a single config run across markets.
    """
    # Config metadata
    config: Dict
    prediction_mode: str
    flag_rate_threshold: float
    suspicion_threshold: float
    min_usd_amount: Optional[float]
    market_ids: List[int]
    timestamp: str                     # ISO timestamp of evaluation run

    # Wallet-level results
    wallet_evaluations: List[Dict]
    copytrade_summary: Dict
    copytrade_per_market: pd.DataFrame

    # Trade-level results
    event_study_results: List[TradeEventStudyResult]
    event_study_pooled: Dict

    # Copytrade simulation
    copytrade_result: Optional[CopytradeResult]

    # Performance metrics
    per_market_performance: List[MarketPerformance]
    aggregate_performance: AggregatePerformance

    # Raw backtest results (for post-hoc analysis like flag_rate sweeps)
    backtest_results: Dict[int, BacktestResult] = field(default_factory=dict)

    def save(self, output_dir: str, tag: str = "") -> Dict[str, str]:
        """
        Save all results to structured files. Returns dict of {label: filepath}.

        File naming: eval_{tag}_{timestamp}.{ext}
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        prefix = f"{output_dir}/eval{suffix}_{self.timestamp}"
        saved = {}

        # 1. Config + parameters
        meta = {
            "config": self.config,
            "prediction_mode": self.prediction_mode,
            "flag_rate_threshold": self.flag_rate_threshold,
            "suspicion_threshold": self.suspicion_threshold,
            "min_usd_amount": self.min_usd_amount,
            "market_ids": self.market_ids,
            "timestamp": self.timestamp,
        }
        p = f"{prefix}_meta.json"
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)
        saved["meta"] = p

        # 2. Copytrade summary
        p = f"{prefix}_copytrade_summary.json"
        with open(p, "w") as f:
            json.dump({"summary": self.copytrade_summary}, f, indent=2)
        saved["copytrade_summary"] = p

        # 3. Per-market copytrade
        p = f"{prefix}_copytrade_markets.csv"
        self.copytrade_per_market.to_csv(p, index=False)
        saved["copytrade_markets"] = p

        # 4. Trade-level event study
        p = f"{prefix}_event_study.json"
        with open(p, "w") as f:
            json.dump(self.event_study_pooled, f, indent=2, default=str)
        saved["event_study"] = p

        # 5. Copytrade simulation
        if self.copytrade_result is not None:
            p = f"{prefix}_copytrade_sim.json"
            with open(p, "w") as f:
                # CopytradeResult doesn't have .to_dict(), serialise manually
                json.dump({
                    "total_flagged_buys": self.copytrade_result.total_flagged_buys,
                    "total_capital_deployed": self.copytrade_result.total_capital_deployed,
                    "total_pnl": self.copytrade_result.total_pnl,
                    "portfolio_roi": self.copytrade_result.portfolio_roi,
                    "win_rate": self.copytrade_result.win_rate,
                    "mean_trade_return": self.copytrade_result.mean_trade_return,
                    "median_trade_return": self.copytrade_result.median_trade_return,
                    "fixed_trade_size": self.copytrade_result.fixed_trade_size,
                    "fixed_roi": self.copytrade_result.fixed_roi,
                }, f, indent=2)
            saved["copytrade_sim"] = p

        # 6. Performance metrics
        p = f"{prefix}_performance.json"
        perf_data = {
            "aggregate": {
                "total_trades": self.aggregate_performance.total_trades,
                "total_markets": self.aggregate_performance.total_markets,
                "total_wall_clock_seconds": self.aggregate_performance.total_wall_clock_seconds,
                "overall_trades_per_second": self.aggregate_performance.overall_trades_per_second,
                "peak_memory_mb": self.aggregate_performance.peak_memory_mb,
                "detection_latency_mean_us": self.aggregate_performance.detection_latency_mean_us,
                "detection_latency_p50_us": self.aggregate_performance.detection_latency_p50_us,
                "detection_latency_p95_us": self.aggregate_performance.detection_latency_p95_us,
                "detection_latency_p99_us": self.aggregate_performance.detection_latency_p99_us,
                "total_latency_mean_us": self.aggregate_performance.total_latency_mean_us,
                "total_latency_p50_us": self.aggregate_performance.total_latency_p50_us,
                "total_latency_p95_us": self.aggregate_performance.total_latency_p95_us,
                "total_latency_p99_us": self.aggregate_performance.total_latency_p99_us,
            },
            "per_market": [
                {
                    "market_id": mp.market_id,
                    "market_slug": mp.market_slug,
                    "n_trades": mp.n_trades,
                    "n_wallets": mp.n_wallets,
                    "n_alerts": mp.n_alerts,
                    "wall_clock_seconds": mp.wall_clock_seconds,
                    "trades_per_second": mp.trades_per_second,
                    "detection_latency_mean_us": mp.detection_latency_mean_us,
                    "detection_latency_p50_us": mp.detection_latency_p50_us,
                    "detection_latency_p95_us": mp.detection_latency_p95_us,
                    "detection_latency_p99_us": mp.detection_latency_p99_us,
                    "total_latency_mean_us": mp.total_latency_mean_us,
                    "total_latency_p50_us": mp.total_latency_p50_us,
                    "total_latency_p95_us": mp.total_latency_p95_us,
                    "total_latency_p99_us": mp.total_latency_p99_us,
                }
                for mp in self.per_market_performance
            ],
        }
        with open(p, "w") as f:
            json.dump(perf_data, f, indent=2)
        saved["performance"] = p

        # 7. Performance CSV (for easy plotting)
        p = f"{prefix}_performance_markets.csv"
        pd.DataFrame(perf_data["per_market"]).to_csv(p, index=False)
        saved["performance_csv"] = p

        logger.info(f"Saved {len(saved)} result files to {output_dir}/")
        return saved

# Default clustering config. Experiments can import and override individual fields.
DEFAULT_CLUSTERING_CONFIG = {
    "bucket_size": 300,
    "same_direction_mult": 2.0,
    "size_normalizer": 10000,
    "max_size_mult": 5.0,
    "cross_outcome_penalty": 0.1,
    "k_core": 2,
    "min_edge_weight": 0.5,
    "boost": {
        "max_boost_factor": 2.0,
        "size_weight": 0.5,
        "density_weight": 0.25,
        "ownership_boost": 0.4,
        "size_normalizer": 50.0,
    },
}

def evaluate_config(
    config: Dict,
    loader: HistoricalDataLoader,
    market_ids: List[int],
    *,
    prediction_mode: str = "flag_rate",
    flag_rate_threshold: float = 0.2,
    suspicion_threshold: float = 2.0,
    z_score_threshold: float = 2.0,
    min_wallet_notional: float = 500.0,
    min_usd_amount: Optional[float] = None,
    include_recidivism: bool = False,
    clustering_config: Optional[Dict] = DEFAULT_CLUSTERING_CONFIG,
    clustering_min_trade_size: float = 5000.0,
    jump_anticipation_config: Optional[Dict] = None,
    copytrade_fixed_size: float = 100.0,
    measure_memory: bool = True,
    winning_outcomes_override: Optional[Dict[int, int]] = None,
    enable_layer2_attribution: bool = False,
    usdc_cache_db: str = "data/usdc_transfers.db",
    polygonscan_api_key: Optional[str] = None,
    use_causal_trade_level_boosts: bool = True,
    poll_interval_seconds: float = 5.0,
    quiet_per_market: bool = False,
) -> EvaluationResult:
    """Evaluate one detector config across resolved markets."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if measure_memory:
        tracemalloc.start()

    runner = BacktestRunner(config=config, include_recidivism=include_recidivism)
    evaluator = WalletEvaluator(
        z_score_threshold=z_score_threshold,
        min_wallet_notional=min_wallet_notional,
        label_metric="return",
    )
    attribution_provider = build_attribution_provider(
        enable_layer2_attribution=enable_layer2_attribution,
        usdc_cache_db=usdc_cache_db,
        polygonscan_api_key=polygonscan_api_key,
    )

    all_wallet_evals: List[Dict] = []
    event_study_results: List[TradeEventStudyResult] = []
    copytrade_results: List[CopytradeResult] = []
    backtest_results: Dict[int, BacktestResult] = {}
    market_perfs: List[MarketPerformance] = []

    for market_id in market_ids:
        info = get_market_info(market_id)
        slug = info.get("market_slug", str(market_id)) if info else str(market_id)

        # Load the replay trade list. If the caller installed a timeframe
        # trade filter, this tape is intentionally window-scoped.
        try:
            replay_trades = loader.get_trades_for_market(
                market_id=market_id, min_usd_amount=None, use_cache=False)
        except TypeError:
            replay_trades = loader.get_trades_for_market(market_id)

        # Ground-truth wallet financials must use complete market history even
        # when detector replay is timeframe-scoped.
        ground_truth_trades = load_all_trades_for_market(loader, market_id)

        # Filter for detector processing
        if min_usd_amount is not None:
            detector_trades = filter_trades_by_notional(replay_trades, min_usd_amount)
        else:
            detector_trades = replay_trades

        metadata = dict(loader.get_market_metadata(market_id) or {})
        metadata["id"] = market_id
        winning_outcome = None
        if winning_outcomes_override is not None and market_id in winning_outcomes_override:
            winning_outcome = int(winning_outcomes_override[market_id])

        score_multipliers = None
        score_cap = 0.95
        wallet_cluster_boost_diag: Optional[Dict[str, float]] = None
        wallet_has_common_ownership_diag: Optional[Dict[str, bool]] = None

        if use_causal_trade_level_boosts and (
            clustering_config is not None or jump_anticipation_config is not None
        ):
            schedule = build_live_parity_boost_schedule(
                detector_trades=detector_trades,
                market_id=str(market_id),
                clustering_config=clustering_config,
                clustering_min_trade_size=clustering_min_trade_size,
                jump_anticipation_config=jump_anticipation_config,
                poll_interval_seconds=poll_interval_seconds,
                attribution_provider=attribution_provider,
                fetch_if_missing=enable_layer2_attribution,
            )
            score_multipliers = schedule.score_multiplier_by_trade_idx
            score_cap = float(schedule.score_cap)
            wallet_cluster_boost_diag = schedule.final_wallet_cluster_boost
            wallet_has_common_ownership_diag = schedule.final_wallet_has_common_ownership
        # Stage 1: Run detector backtest with optional causal per-trade multipliers
        backtest_result = runner.run_backtest(
            trades=detector_trades,
            market_metadata=metadata,
            capture_alerts=True,
            capture_trade_features=False,
            progress_every=0,
            score_multipliers=score_multipliers,
            score_cap=score_cap,
            wallet_cluster_boost=wallet_cluster_boost_diag,
            wallet_has_common_ownership=wallet_has_common_ownership_diag,
        )

        # Legacy fallback path (kept for A/B validation).
        if not use_causal_trade_level_boosts:
            if clustering_config is not None:
                clustering_runner = BucketClusteringBacktestRunner(
                    detector_config={},
                    clustering_config=clustering_config,
                    attribution_provider=attribution_provider,
                )
                graph_trades = filter_trades_by_notional(
                    detector_trades,
                    clustering_min_trade_size,
                )
                backtest_result = clustering_runner.run_boost_only(
                    base_result=backtest_result,
                    graph_trades=graph_trades,
                    market_id=str(market_id),
                )

            if jump_anticipation_config is not None:
                run_jump_anticipation_boost(
                    result=backtest_result,
                    all_trades=replay_trades,
                    config=jump_anticipation_config,
                    scoring_trades=detector_trades,
                )

        backtest_results[market_id] = backtest_result

        wallet_evals = evaluate_wallets_with_ground_truth(
            all_trades=ground_truth_trades,
            backtest_result=backtest_result,
            market_metadata=metadata,
            evaluator=evaluator,
            winning_outcome_override=winning_outcome,
        )
        all_wallet_evals.extend(wallet_evals)

        es = run_trade_event_study(
            trades=detector_trades,
            backtest_result=backtest_result,
            market_metadata=metadata,
            winning_outcome=winning_outcome,
        )
        if es is not None:
            event_study_results.append(es)

        ct = run_copytrade_simulation(
            trades=detector_trades,
            backtest_result=backtest_result,
            market_metadata=metadata,
            winning_outcome=winning_outcome,
            fixed_trade_size=copytrade_fixed_size,
        )
        if ct is not None:
            copytrade_results.append(ct)

        mp = _build_market_performance(market_id, slug, backtest_result)
        market_perfs.append(mp)

        if not quiet_per_market:
            logger.info(
                f"  {slug[:55]:55s} | trades={len(detector_trades):>7,} | "
                f"alerts={backtest_result.alerts_generated:>5,} | "
                f"wallets={len(wallet_evals):>5,} | "
                f"{mp.wall_clock_seconds:.2f}s | "
                f"{mp.trades_per_second:,.0f} trades/s | "
                f"det_p95={mp.detection_latency_p95_us:.0f}μs"
            )

    peak_memory_mb = 0.0
    if measure_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_bytes / (1024 * 1024)

    agg_perf = _build_aggregate_performance(market_perfs, backtest_results, peak_memory_mb)

    copytrade_summary, copytrade_per_market = build_copytrade_report(
        wallet_evaluations=all_wallet_evals,
        prediction_mode=prediction_mode,
        suspicion_threshold=suspicion_threshold,
        flag_rate_threshold=flag_rate_threshold,
    )

    event_study_pooled = run_trade_event_study_multi(event_study_results) if event_study_results else {}
    copytrade_combined = run_copytrade_simulation_multi(copytrade_results) if copytrade_results else None

    if attribution_provider is not None:
        logger.info(
            "Attribution stats: "
            f"cache_hits={attribution_provider.get_stats().get('cache_hits', 0):,}, "
            f"cache_misses={attribution_provider.get_stats().get('cache_misses', 0):,}, "
            f"api_fetches={attribution_provider.get_stats().get('api_fetches', 0):,}, "
            f"api_failures={attribution_provider.get_stats().get('api_failures', 0):,}"
        )
        attribution_provider.close()

    return EvaluationResult(
        config=config,
        prediction_mode=prediction_mode,
        flag_rate_threshold=flag_rate_threshold,
        suspicion_threshold=suspicion_threshold,
        min_usd_amount=min_usd_amount,
        market_ids=market_ids,
        timestamp=ts,
        wallet_evaluations=all_wallet_evals,
        copytrade_summary=copytrade_summary,
        copytrade_per_market=copytrade_per_market,
        event_study_results=event_study_results,
        event_study_pooled=event_study_pooled,
        copytrade_result=copytrade_combined,
        per_market_performance=market_perfs,
        aggregate_performance=agg_perf,
        backtest_results=backtest_results,
    )

def print_performance_summary(result: EvaluationResult):
    """Print human-readable performance summary to stdout."""
    agg = result.aggregate_performance
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"  Total trades processed: {agg.total_trades:,}")
    print(f"  Total markets:          {agg.total_markets}")
    print(f"  Wall clock time:        {agg.total_wall_clock_seconds:.2f}s")
    print(f"  Throughput:             {agg.overall_trades_per_second:,.0f} trades/sec")
    if agg.peak_memory_mb > 0:
        print(f"  Peak memory:            {agg.peak_memory_mb:.1f} MB")
    print()
    print(f"  Detection latency (pooled across all trades):")
    print(f"    mean={agg.detection_latency_mean_us:.1f}μs  "
          f"p50={agg.detection_latency_p50_us:.1f}μs  "
          f"p95={agg.detection_latency_p95_us:.1f}μs  "
          f"p99={agg.detection_latency_p99_us:.1f}μs")
    print(f"  Full per-trade latency:")
    print(f"    mean={agg.total_latency_mean_us:.1f}μs  "
          f"p50={agg.total_latency_p50_us:.1f}μs  "
          f"p95={agg.total_latency_p95_us:.1f}μs  "
          f"p99={agg.total_latency_p99_us:.1f}μs")

    print(f"\n  Per-market breakdown:")
    for mp in result.per_market_performance:
        print(f"    {mp.market_slug[:50]:50s} | "
              f"{mp.n_trades:>7,} trades | "
              f"{mp.trades_per_second:>8,.0f} t/s | "
              f"det_p95={mp.detection_latency_p95_us:>6.0f}μs")


def print_copytrade_summary(title: str, summary: Dict):
    """Print wallet-level copytrade summary."""
    print(f"\n{title}")
    labels = [
        ("flagged", "Flagged wallets"),
        ("tp", "Flagged + insider (TP)"),
        ("fp", "Flagged + not insider (FP)"),
        ("fn", "Not flagged + insider (FN)"),
    ]
    for key, label in labels:
        s = summary.get(key, {})
        print(f"  {label}: n={int(s.get('count', 0)):,} | "
              f"avg_return={float(s.get('avg_return', 0.0)):.2%} | "
              f"weighted_return={float(s.get('weighted_return', 0.0)):.2%} | "
              f"median_return={float(s.get('median_return', 0.0)):.2%} | "
              f"mean_net_pnl=${float(s.get('avg_net_pnl', 0.0)):,.2f} | "
              f"median_net_pnl=${float(s.get('median_net_pnl', 0.0)):,.2f}")


def print_event_study_summary(result: EvaluationResult):
    """Print trade-level event study summary."""
    for es in result.event_study_results:
        print(es.summary())
        print()

    pooled = result.event_study_pooled.get("pooled", {})
    if pooled:
        print(f"--- Pooled across {pooled.get('n_markets', 0)} markets ---")
        print(f"  Flagged trades: {pooled.get('total_flagged_trades', 0):,}")
        print(f"  Unflagged trades: {pooled.get('total_unflagged_trades', 0):,}")
        print(f"  Flagged mean return: {pooled.get('pooled_flagged_mean_return', 0):+.4f}")
        print(f"  Unflagged mean return: {pooled.get('pooled_unflagged_mean_return', 0):+.4f}")
        print(f"  Mean diff: {pooled.get('pooled_mean_return_diff', 0):+.4f}")
        print(f"  Mean Cohen's d: {pooled.get('mean_cohens_d', 0):.3f}")
        print(f"  Markets with Welch p<0.05: "
              f"{pooled.get('markets_significant_welch_p05', 0)}/{pooled.get('n_markets', 0)}")
        print(f"  Markets with MW p<0.05: "
              f"{pooled.get('markets_significant_mw_p05', 0)}/{pooled.get('n_markets', 0)}")
