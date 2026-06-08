import logging
from typing import Dict, Set, Tuple
from collections import defaultdict

import networkx as nx

from backtesting.logging_utils import experiment_backtest_logs_quiet

try:
    import community as community_louvain
except ImportError:
    logging.error(
        "python-louvain not installed. "
        "Install with: pip install python-louvain"
    )
    community_louvain = None

from clustering.models import ClusterInfo


logger = logging.getLogger(__name__)


class ClusterComputer:
    """Computes clusters from the co-activity graph using Louvain method"""

    def __init__(self, config: Dict):
        self.config = config
        
        self.k_core = config.get("k_core", 2)
        
        self.min_edge_weight = config.get("min_edge_weight", 0.5)

        if community_louvain is None:
            raise ImportError(
                "python-louvain is required for ClusterComputer"
            )
        if not experiment_backtest_logs_quiet():
            logger.info(
                f"ClusterComputer initialized "
                f"(k_core={self.k_core}, min_edge_weight={self.min_edge_weight})"
            )

    def compute_clusters(self, graph: nx.Graph) -> Tuple[Dict[str, int], Dict[int, ClusterInfo]]:
        """Run clustering on the co-activity graph and return cluster assignments and metadata."""

        filtered_graph = self._filter_graph(graph)
        if filtered_graph.number_of_nodes() == 0:
            if not experiment_backtest_logs_quiet():
                logger.info("Filtered graph is empty, no clusters to compute")
            return {}, {}

        k_core_graph = self._compute_k_core(filtered_graph)
        if k_core_graph.number_of_nodes() == 0:
            if not experiment_backtest_logs_quiet():
                logger.info("K-core graph is empty, no clusters to compute")
            return {}, {}
        
        wallet_to_cluster = self._compute_communities(k_core_graph)
        if not wallet_to_cluster:
            if not experiment_backtest_logs_quiet():
                logger.info("No communities detected in the graph")
            return {}, {}
        
        cluster_metadata = self._compute_metadata(k_core_graph, wallet_to_cluster)

        if not experiment_backtest_logs_quiet():
            logger.info(
                f"Computed {len(cluster_metadata)} clusters "
                f"from {k_core_graph.number_of_nodes()} wallets"
            )

        return wallet_to_cluster, cluster_metadata

    def _filter_graph(self, graph: nx.Graph) -> nx.Graph:
        """Filter graph to keep only edges above weight threshold"""

        filtered = nx.Graph()

        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 0.0)
            if weight >= self.min_edge_weight:
                filtered.add_edge(u, v, weight=weight)

        logging.debug(
            f"Filtered graph: "
            f"{graph.number_of_nodes()} -> {filtered.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} -> {filtered.number_of_edges()} edges"
        )

        return filtered
    
    def _compute_k_core(self, graph: nx.Graph) -> nx.Graph:
        """Compute k-core of the graph to remove low-degree nodes"""

        try:
            k_core = nx.k_core(graph, k=self.k_core)

            logging.debug(
                f"K-core graph: "
                f"{graph.number_of_nodes()} -> {k_core.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} -> {k_core.number_of_edges()} edges"
            )

            return k_core
        except nx.NetworkXError as e:
            logging.error(f"Error computing k-core: {e}")
            return nx.Graph()
        
    def _compute_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Use Louvain method to compute communities in the graph"""

        if graph.number_of_nodes() == 0:
            return {}
        try:
            partition = community_louvain.best_partition(graph, weight='weight', random_state=42)
            
            logging.debug(
                f"Detected {len(set(partition.values()))} communities"
            )

            return partition
        except Exception as e:
            logging.error(f"Error computing communities: {e}")
            return {}
    
    def _compute_metadata(self, graph: nx.Graph, wallet_to_cluster: Dict[str, int]) -> Dict[int, ClusterInfo]:
        """Compute metadata for each cluster"""

        cluster_wallets = defaultdict(set)
        for wallet, cluster_id in wallet_to_cluster.items():
            cluster_wallets[cluster_id].add(wallet)

        metadata = {}

        for cluster_id, wallets in cluster_wallets.items():
            # Create subgraph for this cluster
            subgraph = graph.subgraph(wallets)
            
            # Count edges and total weight
            edge_count = subgraph.number_of_edges()
            total_weight = sum(
                data.get('weight', 0.0) 
                for _, _, data in subgraph.edges(data=True)
            )
            
            # Compute density
            size = len(wallets)
            max_edges = size * (size - 1) / 2
            density = edge_count / max_edges if max_edges > 0 else 0.0
            
            metadata[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                size=size,
                density=density,
                total_edge_weight=total_weight,
                wallets=wallets,
                has_common_ownership=False,
                attribution_enriched=False
            )

        return metadata
    