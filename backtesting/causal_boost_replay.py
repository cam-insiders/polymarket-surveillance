"""
Causal boost replay helpers for live-parity backtesting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from clustering.bucket_graph_builder import build_graph_from_trades_bucketed
from clustering.cluster_computer import ClusterComputer
from clustering.models import ClusterInfo
from clustering.ownership_analyser import CommonOwnershipAnalyser
from jump_anticipation.core import find_jumps, score_wallets_jump_anticipation
from models import Trade


@dataclass
class BoostSchedule:
    """
    Per-trade causal boost schedule for one market.
    """

    score_multiplier_by_trade_idx: np.ndarray
    cluster_multiplier_by_trade_idx: np.ndarray
    ja_multiplier_by_trade_idx: np.ndarray
    score_cap: float = 0.95

    # Final wallet-level diagnostics
    final_wallet_cluster_boost: Dict[str, float] = field(default_factory=dict)
    final_wallet_has_common_ownership: Dict[str, bool] = field(default_factory=dict)

    # Diagnostics
    cluster_update_trade_indices: List[int] = field(default_factory=list)
    ja_score_batch_indices: List[int] = field(default_factory=list)
    batch_ranges: List[Tuple[int, int]] = field(default_factory=list)


def build_trade_batches(
    trades: List[Trade],
    poll_interval_seconds: float = 5.0,
) -> List[Tuple[int, int]]:
    """
    Partition trades into polling-style batches using event timestamps.
    """
    n = len(trades)
    if n == 0:
        return []

    poll_ms = max(1, int(float(poll_interval_seconds) * 1000))
    batches: List[Tuple[int, int]] = []

    start_idx = 0
    window_start_ms = trades[0].timestamp_ms
    window_end_ms = window_start_ms + poll_ms

    for i, trade in enumerate(trades):
        ts = int(trade.timestamp_ms)
        if ts >= window_end_ms:
            batches.append((start_idx, i))
            start_idx = i
            window_start_ms = ts
            window_end_ms = window_start_ms + poll_ms

    if start_idx < n:
        batches.append((start_idx, n))

    return batches


def _compute_cluster_boost(cluster: ClusterInfo, boost_config: Dict) -> float:
    max_boost = float(boost_config.get("max_boost_factor", 2.0))
    size_weight = float(boost_config.get("size_weight", 0.4))
    density_weight = float(boost_config.get("density_weight", 0.2))
    size_normalizer = float(boost_config.get("size_normalizer", 50.0))

    boost = 1.0
    boost += min(cluster.size / size_normalizer, size_weight)
    boost += cluster.density * density_weight
    if bool(cluster.has_common_ownership):
        boost += float(boost_config.get("ownership_boost", 0.4))
    return min(boost, max_boost)


def _should_recluster(
    *,
    current_time_s: int,
    last_cluster_time_s: int,
    trades_since_last_cluster: int,
    clustering_cfg: Dict,
) -> bool:
    """
    Event-time equivalent of ClusteringManager._should_recluster().
    """
    time_since_last = int(current_time_s) - int(last_cluster_time_s)
    min_interval = int(clustering_cfg.get("min_cluster_interval", 300))
    if time_since_last < min_interval:
        return False

    max_interval = int(clustering_cfg.get("max_cluster_interval", 3600))
    if time_since_last >= max_interval:
        return True

    threshold = int(clustering_cfg.get("significant_change_threshold", 20))
    if trades_since_last_cluster >= threshold:
        return True

    return False


def _enrich_clusters_with_attribution(
    cluster_metadata: Dict[int, ClusterInfo],
    *,
    attribution_provider,
    attribution_config: Dict,
    reference_timestamp_s: Optional[int],
    fetch_if_missing: bool,
) -> None:
    """
    Enrich cluster metadata with ownership flags using cache/API provider.
    """
    if attribution_provider is None or not cluster_metadata:
        return

    min_size = int(attribution_config.get("min_size_for_attribution", 2))
    min_density = float(attribution_config.get("min_density_for_attribution", 0.0))
    analyser = CommonOwnershipAnalyser(attribution_config)

    for cluster_info in cluster_metadata.values():
        if cluster_info.size < min_size:
            continue
        if cluster_info.density < min_density:
            continue

        wallets: Set[str] = set(cluster_info.wallets)
        attribution_provider.ensure_wallets_cached(
            wallets,
            fetch_if_missing=fetch_if_missing,
        )
        edges = attribution_provider.get_intra_cluster_edges(
            wallets,
            before_timestamp=reference_timestamp_s,
        )
        analysis = analyser.analyse_cluster(cluster_info, edges)
        cluster_info.has_common_ownership = bool(analysis["has_ownership"])
        cluster_info.attribution_enriched = True


def build_live_parity_boost_schedule(
    *,
    detector_trades: List[Trade],
    market_id: str,
    clustering_config: Optional[Dict],
    clustering_min_trade_size: float = 5000.0,
    jump_anticipation_config: Optional[Dict] = None,
    poll_interval_seconds: float = 5.0,
    attribution_provider=None,
    fetch_if_missing: bool = True,
) -> BoostSchedule:
    """
    Build per-trade multipliers with live-parity causal timing.
    """
    n = len(detector_trades)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return BoostSchedule(
            score_multiplier_by_trade_idx=empty,
            cluster_multiplier_by_trade_idx=empty,
            ja_multiplier_by_trade_idx=empty,
            score_cap=0.95,
        )

    cluster_mult = np.ones(n, dtype=np.float32)
    ja_mult = np.ones(n, dtype=np.float32)

    batches = build_trade_batches(detector_trades, poll_interval_seconds=poll_interval_seconds)

    cluster_cfg = dict(clustering_config or {})
    cluster_algo_cfg = {
        "k_core": cluster_cfg.get("k_core", 2),
        "min_edge_weight": cluster_cfg.get("min_edge_weight", 0.5),
    }
    cluster_boost_cfg = dict(cluster_cfg.get("boost", {}))
    attribution_cfg = dict(cluster_cfg.get("attribution", {}))
    score_cap = float(cluster_boost_cfg.get("max_final_score", 0.95))

    cluster_computer = ClusterComputer(cluster_algo_cfg)
    graph_trades_prefix: List[Trade] = []
    last_cluster_time_s = int(detector_trades[0].timestamp_ms // 1000)
    trades_since_last_cluster = 0
    wallet_to_cluster: Dict[str, int] = {}
    cluster_metadata: Dict[int, ClusterInfo] = {}
    cluster_update_trade_indices: List[int] = []

    ja_cfg = dict(jump_anticipation_config or {})
    ja_enabled = jump_anticipation_config is not None
    ja_wallet_boosts: Dict[str, float] = {}
    ja_buffer: Deque[Trade] = deque()
    ja_score_batch_indices: List[int] = []
    ja_last_scored_at_s = 0.0
    ja_interval_s = float(ja_cfg.get("scoring_interval_minutes", 15)) * 60.0
    ja_buffer_max_age_ms = int(float(ja_cfg.get("buffer_hours", 24)) * 3600 * 1000)

    for batch_idx, (start_idx, end_idx) in enumerate(batches):
        # Per-trade scoring: cluster + (stale-until-batch-end) JA
        for i in range(start_idx, end_idx):
            trade = detector_trades[i]
            ts_s = int(trade.timestamp_ms // 1000)

            if clustering_config is not None and float(trade.notional_usdc) >= float(clustering_min_trade_size):
                graph_trades_prefix.append(trade)
                trades_since_last_cluster += 1

                if _should_recluster(
                    current_time_s=ts_s,
                    last_cluster_time_s=last_cluster_time_s,
                    trades_since_last_cluster=trades_since_last_cluster,
                    clustering_cfg=cluster_cfg,
                ):
                    graph = build_graph_from_trades_bucketed(
                        trades=graph_trades_prefix,
                        config=cluster_cfg,
                        market_id=market_id,
                    )
                    wallet_to_cluster, cluster_metadata = cluster_computer.compute_clusters(graph)
                    _enrich_clusters_with_attribution(
                        cluster_metadata=cluster_metadata,
                        attribution_provider=attribution_provider,
                        attribution_config=attribution_cfg,
                        reference_timestamp_s=ts_s,
                        fetch_if_missing=fetch_if_missing,
                    )
                    last_cluster_time_s = ts_s
                    trades_since_last_cluster = 0
                    cluster_update_trade_indices.append(i)

            c_boost = 1.0
            c_idx = wallet_to_cluster.get(trade.wallet)
            if c_idx is not None:
                info = cluster_metadata.get(c_idx)
                if info is not None:
                    c_boost = _compute_cluster_boost(info, cluster_boost_cfg)
            cluster_mult[i] = np.float32(c_boost)
            ja_mult[i] = np.float32(ja_wallet_boosts.get(trade.wallet, 1.0))

        # Batch-end JA update
        if ja_enabled:
            for i in range(start_idx, end_idx):
                t = detector_trades[i]
                ja_buffer.append(t)
                cutoff_ms = int(t.timestamp_ms) - ja_buffer_max_age_ms
                while ja_buffer and int(ja_buffer[0].timestamp_ms) < cutoff_ms:
                    ja_buffer.popleft()

            batch_end_s = float(detector_trades[end_idx - 1].timestamp_ms) / 1000.0
            if (batch_end_s - ja_last_scored_at_s) >= ja_interval_s:
                buffered = list(ja_buffer)
                if len(buffered) >= 10:
                    jumps = find_jumps(buffered, ja_cfg)
                    if jumps:
                        scores = score_wallets_jump_anticipation(buffered, jumps, ja_cfg)
                        ja_wallet_boosts = {w: b for w, b in scores.items() if b > 1.001}
                    else:
                        ja_wallet_boosts = {}
                    ja_score_batch_indices.append(batch_idx)
                ja_last_scored_at_s = batch_end_s

    score_mult = (cluster_mult * ja_mult).astype(np.float32)

    # Final wallet-level diagnostics only (not used in causal decisioning).
    final_wallet_cluster_boost: Dict[str, float] = {}
    final_wallet_has_common_ownership: Dict[str, bool] = {}
    all_wallets = {t.wallet for t in detector_trades}
    for wallet in all_wallets:
        c_boost = 1.0
        has_ownership = False
        c_idx = wallet_to_cluster.get(wallet)
        if c_idx is not None:
            info = cluster_metadata.get(c_idx)
            if info is not None:
                c_boost = _compute_cluster_boost(info, cluster_boost_cfg)
                has_ownership = bool(info.has_common_ownership)
        total_boost = float(c_boost * float(ja_wallet_boosts.get(wallet, 1.0)))
        final_wallet_cluster_boost[wallet] = total_boost
        final_wallet_has_common_ownership[wallet] = has_ownership

    return BoostSchedule(
        score_multiplier_by_trade_idx=score_mult,
        cluster_multiplier_by_trade_idx=cluster_mult,
        ja_multiplier_by_trade_idx=ja_mult,
        score_cap=score_cap,
        final_wallet_cluster_boost=final_wallet_cluster_boost,
        final_wallet_has_common_ownership=final_wallet_has_common_ownership,
        cluster_update_trade_indices=cluster_update_trade_indices,
        ja_score_batch_indices=ja_score_batch_indices,
        batch_ranges=batches,
    )

