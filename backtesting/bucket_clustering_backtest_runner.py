"""Backtest runner for bucket-projected clustering boosts."""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

from models import Trade
from backtesting.backtest_runner import BacktestRunner, BacktestResult
from clustering.models import ClusteringState, ClusterInfo
from clustering.cluster_computer import ClusterComputer
from clustering.bucket_graph_builder import build_graph_from_trades_bucketed
from clustering.ownership_analyser import CommonOwnershipAnalyser


class BucketClusteringBacktestRunner:
    """Run backtests with bucket-clustering and optional attribution."""

    def __init__(
        self,
        detector_config: Dict,
        clustering_config: Dict,
        include_recidivism: bool = False,
        attribution_provider=None,
    ):
        """Initialize the clustering-aware backtest runner."""
        self.detector_config = detector_config
        self.detector_runner = BacktestRunner(
            config=detector_config,
            include_recidivism=include_recidivism,
        )

        self.graph_config = clustering_config
        self.cluster_config = {
            "k_core": clustering_config.get("k_core", 2),
            "min_edge_weight": clustering_config.get("min_edge_weight", 0.5),
        }
        self.boost_config = clustering_config.get("boost", {})
        
        self.attribution_config = clustering_config.get("attribution", {})
        self.attribution_provider = attribution_provider
        
        # Initialize ownership analyser if we have attribution capability
        if self.attribution_provider is not None:
            self.ownership_analyser = CommonOwnershipAnalyser(self.attribution_config)
        else:
            self.ownership_analyser = None

        logging.debug(
            f"BucketClusteringBacktestRunner initialized: "
            f"bucket_size={clustering_config.get('bucket_size', 300)}, "
            f"k_core={self.cluster_config['k_core']}, "
            f"min_edge_weight={self.cluster_config['min_edge_weight']}, "
            f"max_boost={self.boost_config.get('max_boost_factor', 2.0)}, "
            f"attribution={'enabled' if self.attribution_provider else 'disabled'}"
        )

    def run_backtest(
        self,
        trades: List[Trade],
        market_metadata: Dict,
    ) -> BacktestResult:
        """
        Run complete backtest with clustering (end-to-end).
        """
        base_result = self.detector_runner.run_backtest(
            trades=trades,
            market_metadata=market_metadata,
            capture_alerts=False,
            capture_trade_features=False,
            progress_every=0,
        )

        market_id = str(market_metadata.get("id", "unknown"))
        
        # Compute earliest trade timestamp for lookahead avoidance
        earliest_timestamp = min(t.timestamp_ms // 1000 for t in trades) if trades else None
        
        self._apply_clustering_to_result(
            base_result, trades, market_id,
            reference_timestamp=earliest_timestamp
        )

        return base_result

    def run_boost_only(
        self,
        base_result: BacktestResult,
        graph_trades: List[Trade],
        market_id: str,
        reference_timestamp: Optional[int] = None,
    ) -> BacktestResult:
        """Apply clustering boost to precomputed detector results."""
        boosted_result = copy.copy(base_result)
        # Shallow-copy the result so we don't mutate the cached original.
        # wallet_suspicion is the only field we modify, so deep-copy just that.
        boosted_result.wallet_suspicion = dict(base_result.wallet_suspicion)
        boosted_result.wallet_cluster_boost = dict(base_result.wallet_cluster_boost) if base_result.wallet_cluster_boost else {}
        boosted_result.wallet_has_common_ownership = (
            dict(base_result.wallet_has_common_ownership)
            if base_result.wallet_has_common_ownership
            else {}
        )

        wallet_boosts, wallet_ownership = self.compute_wallet_cluster_attribution(
            graph_trades=graph_trades,
            market_id=market_id,
            reference_timestamp=reference_timestamp,
        )
        self.apply_precomputed_wallet_boosts(
            boosted_result, wallet_boosts, wallet_ownership=wallet_ownership
        )

        return boosted_result

    def _apply_clustering_to_result(
        self,
        result: BacktestResult,
        graph_trades: List[Trade],
        market_id: str,
        reference_timestamp: Optional[int] = None,
    ):
        """
        Build graph, compute clusters, enrich with attribution, apply boost IN-PLACE.
        """
        wallet_boosts, wallet_ownership = self.compute_wallet_cluster_attribution(
            graph_trades=graph_trades,
            market_id=market_id,
            reference_timestamp=reference_timestamp,
        )
        self.apply_precomputed_wallet_boosts(
            result, wallet_boosts, wallet_ownership=wallet_ownership
        )

    @staticmethod
    def apply_precomputed_wallet_boosts(
        result: BacktestResult,
        wallet_boosts: Dict[str, float],
        wallet_ownership: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Apply a precomputed wallet -> boost mapping (and optional ownership
        mapping) to a backtest result.
        """
        result.wallet_cluster_boost = {
            wallet: float(wallet_boosts.get(wallet, 1.0))
            for wallet in result.wallet_suspicion
        }
        if wallet_ownership is None:
            wallet_ownership = {}
        result.wallet_has_common_ownership = {
            wallet: bool(wallet_ownership.get(wallet, False))
            for wallet in result.wallet_suspicion
        }

    def compute_wallet_cluster_boosts(
        self,
        graph_trades: List[Trade],
        market_id: str,
        reference_timestamp: Optional[int] = None,
        fetch_if_missing: bool = True,
    ) -> Dict[str, float]:
        """
        Build clusters once and return the per-wallet boost map.
        """
        boosts, _ownership = self.compute_wallet_cluster_attribution(
            graph_trades=graph_trades,
            market_id=market_id,
            reference_timestamp=reference_timestamp,
            fetch_if_missing=fetch_if_missing,
        )
        return boosts

    def compute_wallet_cluster_attribution(
        self,
        graph_trades: List[Trade],
        market_id: str,
        reference_timestamp: Optional[int] = None,
        fetch_if_missing: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Build clusters once and return (wallet -> boost, wallet -> has_common_ownership).
        """
        clustering_state = self._build_clustering_state(
            graph_trades=graph_trades,
            market_id=market_id,
            reference_timestamp=reference_timestamp,
            fetch_if_missing=fetch_if_missing,
        )
        wallet_boosts: Dict[str, float] = {}
        wallet_ownership: Dict[str, bool] = {}
        for cluster in clustering_state.cluster_metadata.values():
            boost = self._compute_cluster_boost(cluster)
            has_ownership = bool(cluster.has_common_ownership)
            for wallet in cluster.wallets:
                wallet_boosts[wallet] = boost
                wallet_ownership[wallet] = has_ownership
        return wallet_boosts, wallet_ownership

    def _build_clustering_state(
        self,
        graph_trades: List[Trade],
        market_id: str,
        reference_timestamp: Optional[int] = None,
        fetch_if_missing: bool = True,
    ) -> ClusteringState:
        """
        Build graph, clusters, and optional attribution state for a market.
        """
        # Build graph via bucket projection
        graph = build_graph_from_trades_bucketed(
            trades=graph_trades,
            config=self.graph_config,
            market_id=market_id,
        )

        # Compute clusters (unchanged — same ClusterComputer)
        cluster_computer = ClusterComputer(self.cluster_config)
        wallet_to_cluster, cluster_metadata = cluster_computer.compute_clusters(graph)

        logging.debug(
            f"Bucket clustering: {len(graph_trades):,} graph trades, "
            f"{graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges, "
            f"{len(cluster_metadata)} clusters"
        )

        # Attribution enrichment (Layer 2) - if provider is available
        if self.attribution_provider is not None and cluster_metadata:
            self._enrich_clusters_with_attribution(
                cluster_metadata,
                reference_timestamp=reference_timestamp,
                fetch_if_missing=fetch_if_missing,
            )

        # Package into ClusteringState for boost lookup
        clustering_state = ClusteringState()
        clustering_state.wallet_to_cluster = wallet_to_cluster
        clustering_state.cluster_metadata = cluster_metadata

        return clustering_state

    def _enrich_clusters_with_attribution(
        self,
        cluster_metadata: Dict[int, ClusterInfo],
        reference_timestamp: Optional[int] = None,
        fetch_if_missing: bool = True,
    ):
        """Set common-ownership flags from intra-cluster USDC transfers."""
        min_size = self.attribution_config.get("min_size_for_attribution", 2)
        min_density = self.attribution_config.get("min_density_for_attribution", 0.0)
        
        clusters_enriched = 0
        clusters_with_ownership = 0
        
        for cluster_id, cluster_info in cluster_metadata.items():
            # Skip small or sparse clusters
            if cluster_info.size < min_size:
                continue
            if cluster_info.density < min_density:
                continue
            
            wallets = cluster_info.wallets
            
            # Ensure all wallets in cluster have their transfers cached
            self.attribution_provider.ensure_wallets_cached(
                wallets,
                fetch_if_missing=fetch_if_missing,
            )
            
            # Get intra-cluster transfer edges
            edges = self.attribution_provider.get_intra_cluster_edges(
                wallets,
                before_timestamp=reference_timestamp
            )
            
            # Analyze ownership
            analysis = self.ownership_analyser.analyse_cluster(cluster_info, edges)
            
            cluster_info.has_common_ownership = analysis["has_ownership"]
            cluster_info.attribution_enriched = True
            clusters_enriched += 1
            
            if analysis["has_ownership"]:
                clusters_with_ownership += 1
                logging.debug(
                    f"Cluster {cluster_id} has common ownership: "
                    f"{analysis['intra_cluster_edges']} edges, "
                    f"${analysis['total_transfer_amount']:.2f} USDC"
                )
        
        if clusters_enriched > 0:
            logging.debug(
                f"Attribution enrichment: {clusters_enriched} clusters checked, "
                f"{clusters_with_ownership} have common ownership"
            )

    def _apply_cluster_boost(
        self,
        result: BacktestResult,
        clustering_state: ClusteringState,
    ):
        """Compute and store cluster boost factors per wallet."""
        # Initialize all wallets to 1.0 (no boost)
        result.wallet_cluster_boost = {
            wallet: 1.0 for wallet in result.wallet_suspicion
        }

        for wallet in result.wallet_cluster_boost:
            cluster = clustering_state.get_cluster_for_wallet(wallet)
            if cluster is not None:
                result.wallet_cluster_boost[wallet] = self._compute_cluster_boost(cluster)

    def _compute_cluster_boost(self, cluster: ClusterInfo) -> float:
        """
        Compute boost factor from cluster properties.
        """
        max_boost = float(self.boost_config.get("max_boost_factor", 2.0))
        size_weight = float(self.boost_config.get("size_weight", 0.4))
        density_weight = float(self.boost_config.get("density_weight", 0.2))
        size_normalizer = float(self.boost_config.get("size_normalizer", 50.0))

        boost = 1.0

        size_boost = min(cluster.size / size_normalizer, size_weight)
        boost += size_boost

        density_boost = cluster.density * density_weight
        boost += density_boost

        if cluster.has_common_ownership:
            ownership_boost = float(self.boost_config.get("ownership_boost", 0.4))
            boost += ownership_boost

        return min(boost, max_boost)
