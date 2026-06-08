from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import networkx as nx


@dataclass
class EntryRecord:
    timestamp: int
    direction: str
    size: float
    outcome_index: int


@dataclass
class ClusterInfo:
    """Metadata about a detected cluster of coordinated wallets"""
    cluster_id: int
    size: int                           # Number of wallets
    density: float                      # 0.0 to 1.0 (edge_count / max_possible)
    total_edge_weight: float            # Sum of all edge weights
    wallets: Set[str]                   # Members
    
    # Attribution enrichment (Layer 2, populated later)
    has_common_ownership: bool = False
    attribution_enriched: bool = False

    def __repr__(self):
        return (
            f"Cluster(id={self.cluster_id}, size={self.size}, "
            f"density={self.density:.2f}, common_ownership={self.has_common_ownership})"
        )


class AttributionStatus(Enum):
    """Status of Polygonscan attribution query for a wallet"""
    NOT_QUERIED = "not_queried"
    QUERIED = "queried"
    PENDING = "pending"
    FAILED = "failed"


@dataclass
class ClusteringState:
    """
    In-memory state of the clustering process.
    """

    # The co-activity graph (undirected, weighted).
    # Built by bucket projection in ClusteringManager._build_graph().
    coactivity_graph: nx.Graph = field(default_factory=nx.Graph)

    last_cluster_time: int = 0

    # Maps each wallet to its assigned cluster (or None if unassigned)
    wallet_to_cluster: Dict[str, Optional[int]] = field(default_factory=dict)
    
    # Metadata for each cluster (cluster_id -> ClusterInfo)
    cluster_metadata: Dict[int, ClusterInfo] = field(default_factory=dict)

    def get_cluster_for_wallet(self, wallet: str) -> Optional[ClusterInfo]:
        cluster_id = self.wallet_to_cluster.get(wallet)
        if cluster_id is None:
            return None
        return self.cluster_metadata.get(cluster_id)

    def get_wallets_in_cluster(self, cluster_id: int) -> Set[str]:
        cluster = self.cluster_metadata.get(cluster_id)
        return cluster.wallets if cluster else set()