"""Common Ownership Analyser for Attribution Analysis"""

import logging
from typing import Dict, List, Set, Tuple, Optional

from backtesting.logging_utils import experiment_backtest_logs_quiet
from clustering.models import ClusterInfo


logger = logging.getLogger(__name__)


class CommonOwnershipAnalyser:
    """Analyses USDC transfer patterns within clusters to detect common ownership."""

    def __init__(self, config: Dict):
        self.ownership_threshold = config.get("ownership_flag_threshold", 1)
        if not experiment_backtest_logs_quiet():
            logger.info(f"Ownership flag threshold set to {self.ownership_threshold}")

    def analyse_cluster(
        self, 
        cluster: ClusterInfo, 
        attribution_edges: List[Tuple[str, str, float, int]]
    ) -> Dict[str, any]:
        """Analyse attribution edges to detect common ownership"""
        if not attribution_edges:
            return {
                "has_ownership": False,
                "intra_cluster_edges": 0,
                "bidirectional_pairs": 0,
                "total_transfer_amount": 0.0,
                "pattern": "No USDC transfers found",
            }

        cluster_wallets = cluster.wallets

        intra_cluster_edges = [
            (from_w, to_w, amount, tx_count)
            for from_w, to_w, amount, tx_count in attribution_edges
            if from_w in cluster_wallets and to_w in cluster_wallets
        ]

        if not intra_cluster_edges:
            return {
                "has_ownership": False,
                "intra_cluster_edges": 0,
                "bidirectional_pairs": 0,
                "total_transfer_amount": 0.0,
                "pattern": "No intra-cluster USDC transfers",
            }

        # has any transfer between cluster members = ownership (due to temporal coactivity)
        has_ownership = len(intra_cluster_edges) >= self.ownership_threshold

        # detect bidirectional pairs
        bidirectional_pairs = self._detect_bidirectional(intra_cluster_edges)

        total_amount = sum(amount for _, _, amount, _ in intra_cluster_edges)
        total_tx_count = sum(tx_count for _, _, _, tx_count in intra_cluster_edges)

        if bidirectional_pairs > 0:
            pattern = f"Bidirectional transfers ({bidirectional_pairs} pairs)"
        else:
            pattern = f"Unidirectional transfers ({len(intra_cluster_edges)} edges)"
        
        if not experiment_backtest_logs_quiet():
            logger.info(
                f"Cluster {cluster.cluster_id} ownership analysis: "
                f"{len(intra_cluster_edges)} edges, "
                f"${total_amount:.2f} USDC, "
                f"{total_tx_count} transactions, "
                f"bidirectional: {bidirectional_pairs}, "
                f"has_ownership: {has_ownership}"
            )
        
        return {
            "has_ownership": has_ownership,
            "intra_cluster_edges": len(intra_cluster_edges),
            "bidirectional_pairs": bidirectional_pairs,
            "total_transfer_amount": total_amount,
            "pattern": pattern,
        }
    
    def _detect_bidirectional(
        self,
        edges: List[Tuple[str, str, float, int]]
    ) -> int:
        """Detect bidirectional transfer pairs in the edges"""
        edge_set = {(from_w, to_w) for from_w, to_w, _, _ in edges}
        bidirectional_pairs = set()

        for from_w, to_w in edge_set:
            if (to_w, from_w) in edge_set:
                # normalise the pair to avoid duplicates
                pair = tuple(sorted((from_w, to_w)))
                bidirectional_pairs.add(pair)
        
        return len(bidirectional_pairs)
    
    def analyse_all_clusters(
        self,
        clusters: Dict[int, ClusterInfo],
        get_edges_fn
    ) -> Dict[int, Dict]:
        """Analyse ownership for all clusters"""
        results = {}
        
        for cluster_id, cluster_info in clusters.items():
            edges = get_edges_fn(cluster_info.wallets)
            results[cluster_id] = self.analyse_cluster(cluster_info, edges)
        
        return results
    