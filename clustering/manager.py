"""Live clustering coordinator using bucket-projection graph rebuilds."""

from collections import defaultdict
from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Optional, Set

from clustering.ownership_analyser import CommonOwnershipAnalyser
from clustering.polygonscan_client import PolygonscanClient
from clustering.cluster_computer import ClusterComputer
from clustering.database import ClusteringDatabase
from clustering.bucket_graph_builder import build_graph_from_trades_bucketed
from clustering.models import ClusterInfo, ClusteringState, EntryRecord
from models import Trade
import networkx as nx


@dataclass
class _GraphTrade:
    """Trade-shaped record reconstructed from persisted clustering entries."""
    wallet: str
    condition_id: str
    timestamp_ms: int
    outcome_index: int
    side: str
    notional_usdc: float


class ClusteringManager:
    """Orchestrates the clustering detection system"""

    def __init__(self, config: Dict, db_path: str):
        self.full_config = config
        self.config = config.get("clustering", {})
        self.db = ClusteringDatabase(db_path)
        self.state = ClusteringState()

        self.cluster_computer = ClusterComputer(self.config)

        # Accumulated trades for graph rebuilds, keyed by market.
        self.graph_trades: Dict[str, List] = defaultdict(list)

        self.trades_since_last_cluster = 0

        attribution_config = self.full_config.get("attribution", {})
        if attribution_config:
            self.polygonscan_client = PolygonscanClient(attribution_config)
            self.ownership_analyser = CommonOwnershipAnalyser(attribution_config)
        else:
            self.polygonscan_client = None
            self.ownership_analyser = None

        self._load_persisted_state()

        logging.info("ClusteringManager initialised (bucket projection mode)")

    def on_trade(self, trade: Trade):
        """Process a clustering-eligible trade."""
        self.graph_trades[trade.condition_id].append(trade)
        self.trades_since_last_cluster += 1

        entry = EntryRecord(
            timestamp=trade.timestamp_ms // 1000,
            direction=trade.side,
            size=trade.notional_usdc,
            outcome_index=trade.outcome_index,
        )
        self.db.write_entry(trade.condition_id, trade.wallet, entry)

        current_time = int(time.time())
        if self._should_recluster(current_time):
            self._compute_clusters(current_time)

    def _should_recluster(self, current_time: int) -> bool:
        """Decide whether to trigger clustering recomputation."""
        time_since_last = current_time - self.state.last_cluster_time
        min_interval = self.config.get("min_cluster_interval", 300)
        if time_since_last < min_interval:
            return False

        max_interval = self.config.get("max_cluster_interval", 3600)
        if time_since_last >= max_interval:
            logging.info("Max cluster interval exceeded; forcing recluster")
            return True

        threshold = self.config.get("significant_change_threshold", 20)
        if self.trades_since_last_cluster >= threshold:
            logging.info(
                f"Trade count threshold reached "
                f"({self.trades_since_last_cluster} trades); triggering recluster"
            )
            return True

        return False

    def _build_graph(self) -> nx.Graph:
        """
        Build co-activity graph from accumulated trades using bucket projection.
        """
        combined = nx.Graph()

        for market_id, trades in self.graph_trades.items():
            if len(trades) < 2:
                continue

            market_graph = build_graph_from_trades_bucketed(
                trades=trades,
                config=self.config,
                market_id=market_id,
            )

            # Merge into combined graph (sum weights for cross-market edges)
            for u, v, data in market_graph.edges(data=True):
                w = data.get("weight", 0.0)
                if combined.has_edge(u, v):
                    combined[u][v]["weight"] += w
                else:
                    combined.add_edge(u, v, weight=w)

        logging.info(
            f"Graph built: {sum(len(t) for t in self.graph_trades.values()):,} trades "
            f"across {len(self.graph_trades)} markets -> "
            f"{combined.number_of_nodes()} nodes, {combined.number_of_edges()} edges"
        )

        return combined

    def _compute_clusters(self, current_time: int):
        """Recompute cluster assignments using bucket graph + k-core + Louvain."""
        logging.info("Starting clustering recomputation")

        graph = self._build_graph()
        self.state.coactivity_graph = graph

        wallet_to_cluster, cluster_metadata = self.cluster_computer.compute_clusters(graph)

        self.state.wallet_to_cluster = wallet_to_cluster
        self.state.cluster_metadata = cluster_metadata

        self.trades_since_last_cluster = 0
        self.state.last_cluster_time = current_time

        # Attribution enrichment (Layer 2)
        if "attribution" in self.full_config:
            suspicious_clusters = self._identify_suspicious_clusters(cluster_metadata)

            if suspicious_clusters:
                logging.info(
                    f"Identified {len(suspicious_clusters)} suspicious clusters "
                    f"for attribution enrichment"
                )
                self._enrich_clusters_with_attribution(suspicious_clusters)
            else:
                logging.info("No suspicious clusters meet attribution enrichment criteria")
        else:
            logging.info("Attribution analysis disabled (no config section)")

        self.db.write_cluster_state(
            wallet_to_cluster,
            cluster_metadata,
            current_time,
        )

        if cluster_metadata:
            sizes = [info.size for info in cluster_metadata.values()]
            densities = [info.density for info in cluster_metadata.values()]

            logging.info(
                f"Clustering complete: {len(cluster_metadata)} clusters, "
                f"{len(wallet_to_cluster)} wallets. "
                f"Sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}. "
                f"Densities: min={min(densities):.2f}, max={max(densities):.2f}, "
                f"avg={sum(densities)/len(densities):.2f}"
            )
        else:
            logging.info("Clustering complete: no clusters formed")

    def _identify_suspicious_clusters(self, cluster_metadata: Dict[int, ClusterInfo]) -> List[int]:
        """Identify clusters worth enriching with attribution data."""
        if not cluster_metadata:
            return []

        attribution_config = self.full_config.get("attribution", {})
        min_size = attribution_config.get("min_size_for_attribution", 3)
        min_density = attribution_config.get("min_density_for_attribution", 0.7)

        suspicious_clusters = []
        for cluster_id, info in cluster_metadata.items():
            if info.attribution_enriched:
                continue

            if info.size >= min_size and info.density >= min_density:
                suspicious_clusters.append(cluster_id)
                logging.info(
                    f"Cluster {cluster_id} flagged for attribution enrichment "
                    f"(size={info.size}, density={info.density:.2f})"
                )

        return suspicious_clusters

    def _enrich_clusters_with_attribution(self, cluster_ids: List[int]):
        """Query Polygonscan for wallets in clusters and update ownership flags."""
        if not cluster_ids:
            return

        if not self.polygonscan_client:
            logging.warning("Polygonscan client not configured; skipping attribution enrichment")
            return

        logging.info(f"Starting attribution enrichment for {len(cluster_ids)} clusters")

        all_wallets = set()
        for cluster_id in cluster_ids:
            cluster_info = self.state.cluster_metadata.get(cluster_id)
            if cluster_info:
                all_wallets.update(cluster_info.wallets)
                logging.debug(f"  Cluster {cluster_id}: {len(cluster_info.wallets)} wallets")

        logging.info(
            f"Collected {len(all_wallets)} unique wallets from {len(cluster_ids)} clusters"
        )

        unqueried_wallets = self.db.get_unqueried_wallets(list(all_wallets))

        if not unqueried_wallets:
            logging.info("All wallets have been previously queried; skipping Polygonscan API calls")

            for cluster_id in cluster_ids:
                if cluster_id in self.state.cluster_metadata:
                    self.state.cluster_metadata[cluster_id].attribution_enriched = True
            return

        logging.info(f"Querying Polygonscan for {len(unqueried_wallets)} unqueried wallets")

        total_edges = 0
        success_count = 0
        failed_wallets = []

        for i, wallet in enumerate(unqueried_wallets, 1):
            logging.debug(f"Querying wallet {i}/{len(unqueried_wallets)}: {wallet[:10]}...")

            transfers = self.polygonscan_client.get_usdc_transfers(wallet)

            if transfers is None:
                logging.warning(f"Polygonscan query failed for wallet {wallet[:10]}...")
                self.db.update_attribution_cache(wallet, "failed")
                failed_wallets.append(wallet)
                continue

            success_count += 1
            self.db.update_attribution_cache(wallet, "queried")

            if not transfers:
                continue

            edges = self.polygonscan_client.parse_transfers_to_edges(wallet, transfers)

            if edges:
                self.db.write_attribution_edges(edges)
                total_edges += len(edges)

        logging.info(
            f"Attribution enrichment complete: "
            f"{success_count}/{len(unqueried_wallets)} wallets queried successfully, "
            f"{total_edges} attribution edges persisted"
        )

        if failed_wallets:
            logging.warning(f"Failed to query {len(failed_wallets)} wallets from Polygonscan")

        logging.info(f"Analyzing ownership patterns for {len(cluster_ids)} clusters")

        ownership_detected = 0
        for cluster_id in cluster_ids:
            cluster_info = self.state.cluster_metadata.get(cluster_id)
            if not cluster_info:
                continue

            attribution_edges = self.db.get_attribution_edges_for_cluster(cluster_info.wallets)
            analysis = self.ownership_analyser.analyse_cluster(cluster_info, attribution_edges)

            cluster_info.has_common_ownership = analysis["has_ownership"]
            cluster_info.attribution_enriched = True

            if analysis["has_ownership"]:
                ownership_detected += 1
                logging.info(
                    f"Cluster {cluster_id} flagged for common ownership: "
                    f"{analysis['pattern']}, "
                    f"{analysis['intra_cluster_edges']} intra-cluster edges, "
                    f"${analysis['total_transfer_amount']:.2f} total USDC transferred"
                )

        logging.info(
            f"Ownership analysis complete: {ownership_detected}/{len(cluster_ids)} clusters "
            f"flagged for common ownership"
        )

    def _load_persisted_state(self):
        """Load persisted clustering entries from the database."""
        logging.info("Loading persisted clustering state from database")

        raw_entries = self.db.load_all_entries()

        # Reconstruct graph_trades from persisted entries
        trade_count = 0
        for market_id, wallets in raw_entries.items():
            for wallet, entries in wallets.items():
                for entry in entries:
                    self.graph_trades[market_id].append(
                        _GraphTrade(
                            wallet=wallet,
                            condition_id=market_id,
                            timestamp_ms=entry.timestamp * 1000,
                            outcome_index=entry.outcome_index,
                            side=entry.direction,
                            notional_usdc=entry.size,
                        )
                    )
                    trade_count += 1

        if trade_count == 0:
            logging.info("No persisted entries found; starting fresh")
            self.state.last_cluster_time = int(time.time())
            return

        logging.info(f"Reconstructed {trade_count:,} trades from persisted entries")

        # Build initial graph from all historical trades
        graph = self._build_graph()
        self.state.coactivity_graph = graph

        # Load cluster assignments and metadata (from last run)
        self.state.wallet_to_cluster = self.db.load_cluster_assignments()

        metadata_dicts = self.db.load_cluster_metadata()
        for cluster_id, data in metadata_dicts.items():
            self.state.cluster_metadata[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                size=data["size"],
                density=data["density"],
                total_edge_weight=data["total_edge_weight"],
                wallets=set(data["wallets"]),
                has_common_ownership=data["has_common_ownership"],
                attribution_enriched=data["attribution_enriched"],
            )

        self.state.last_cluster_time = int(time.time())

        cluster_count = len(self.state.cluster_metadata)
        clustered_wallet_count = len(self.state.wallet_to_cluster)

        logging.info(
            f"Loaded {trade_count:,} entries, "
            f"{clustered_wallet_count} wallet assignments, "
            f"{cluster_count} clusters"
        )

    def get_cluster_for_wallet(self, wallet: str) -> Optional[ClusterInfo]:
        """Query the cluster info for a given wallet."""
        return self.state.get_cluster_for_wallet(wallet)

    def is_wallet_clustered(self, wallet: str) -> bool:
        """Check if wallet is in any cluster."""
        return wallet in self.state.wallet_to_cluster

    def get_cluster_wallets(self, cluster_id: int) -> Set[str]:
        """Get all wallets in a given cluster."""
        return self.state.get_wallets_in_cluster(cluster_id)

    def close(self):
        """Close database."""
        self.db.close()
        logging.info("ClusteringManager closed")
