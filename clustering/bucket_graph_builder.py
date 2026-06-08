"""Bucket-projection graph builder for co-activity analysis."""

import math
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx

from clustering.models import ClusteringState, EntryRecord
from models import Trade


def build_graph_from_trades_bucketed(
    trades: List[Trade],
    config: Dict,
    market_id: str,
) -> nx.Graph:
    """Build a co-activity wallet graph using directional time buckets."""
    bucket_size = config.get("bucket_size", 300)
    size_normalizer = config.get("size_normalizer", 10000)
    max_size_mult = config.get("max_size_mult", 5.0)

    # Bucket key includes direction + outcome, so only wallets with the
    # same directional bet can ever share a bucket.
    buckets: Dict[Tuple, List[Tuple[str, float]]] = defaultdict(list)

    for trade in trades:
        ts_seconds = trade.timestamp_ms // 1000
        bucket_key = (
            market_id,
            ts_seconds // bucket_size,
            trade.outcome_index,
            trade.side,
        )
        buckets[bucket_key].append((trade.wallet, trade.notional_usdc))

    # A wallet may appear multiple times in one bucket (multiple trades
    # in the same window), so aggregate before projecting edges.
    edge_accumulator: Dict[Tuple[str, str], float] = defaultdict(float)
    buckets_with_edges = 0

    for bucket_key, participants in buckets.items():
        wallet_sizes: Dict[str, float] = defaultdict(float)
        for wallet, size in participants:
            wallet_sizes[wallet] += size

        wallets = list(wallet_sizes.keys())
        if len(wallets) < 2:
            continue

        buckets_with_edges += 1

        for i in range(len(wallets)):
            for j in range(i + 1, len(wallets)):
                w_i, w_j = wallets[i], wallets[j]
                edge_key = (min(w_i, w_j), max(w_i, w_j))

                size_mult = min(
                    math.sqrt(wallet_sizes[w_i] * wallet_sizes[w_j]) / size_normalizer,
                    max_size_mult,
                )

                edge_accumulator[edge_key] += size_mult

    graph = nx.Graph()
    for (w_a, w_b), weight in edge_accumulator.items():
        graph.add_edge(w_a, w_b, weight=weight)

    logging.debug(
        f"BucketGraphBuilder: {len(trades):,} trades -> "
        f"{len(buckets):,} buckets ({buckets_with_edges:,} multi-wallet) -> "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    return graph


def _compute_bucket_pair_weight(
    entry_i: EntryRecord,
    entry_j: EntryRecord,
    same_direction_mult: float,
    size_normalizer: float,
    max_size_mult: float,
    cross_outcome_penalty: float,
) -> float:
    """Compute one bucketed edge-weight contribution."""
    direction_mult = (
        same_direction_mult
        if entry_i.direction == entry_j.direction
        else 1.0
    )

    # Size factor: geometric mean, capped — same logic as original
    size_mult = min(
        math.sqrt(entry_i.size * entry_j.size) / size_normalizer,
        max_size_mult,
    )

    # Outcome awareness: same logic as original
    if entry_i.outcome_index is not None and entry_j.outcome_index is not None:
        if entry_i.outcome_index != entry_j.outcome_index:
            return cross_outcome_penalty * size_mult * direction_mult

    return direction_mult * size_mult
