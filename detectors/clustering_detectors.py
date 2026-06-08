"""Clustering detectors module"""

import logging
from typing import Optional

from detectors.base import Detector, DetectionContext
from models import Trade, DetectionSignal

class ClusterCoordinationDetector(Detector):
    """Detects when a wallet is part of a suspiciously coordinated cluster"""

    REQUIRES_DIRECTIONAL_POSITION = False
    
    def __init__(self, config: dict):
        super().__init__(config)

        self.base_confidence = config.get("base_confidence", 0.2)
        self.size_threshold = config.get("size_threshold", 5)
        self.size_bonus = config.get("size_bonus", 0.2)
        self.density_threshold = config.get("density_threshold", 0.8)
        self.density_bonus = config.get("density_bonus", 0.2)
        self.ownership_bonus = config.get("ownership_bonus", 0.3)
        self.max_confidence = config.get("max_confidence", 0.85)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        """Check if wallet is in a suspicious cluster"""

        if context.clustering_state is None:
            return None
        
        cluster = context.clustering_state.get_cluster_for_wallet(trade.wallet)
        if cluster is None:
            return None
        
        logging.debug(
            f"Wallet {trade.wallet[:10]}... in cluster {cluster.cluster_id} "
            f"(size={cluster.size}, density={cluster.density:.2f})"
        )
        
        confidence = self._score_cluster(cluster)
        
        # metadata for alert
        metadata = {
            "cluster_id": cluster.cluster_id,
            "cluster_size": cluster.size,
            "cluster_density": round(cluster.density, 3),
            "total_edge_weight": round(cluster.total_edge_weight, 2),
            "has_common_ownership": cluster.has_common_ownership,
        }

        reasons = []
        if cluster.size >= self.size_threshold:
            reasons.append(f"large cluster (size={cluster.size})")
        if cluster.density >= self.density_threshold:
            reasons.append(f"high density (density={cluster.density:.2f})")
        if cluster.has_common_ownership:
            reasons.append("common ownership detected")

        reason = f"Member of coordinate cluster: " + ", ".join(reasons)

        return DetectionSignal(
            detector_name=self.name,
            confidence_score=confidence,
            reason=reason,
            metadata=metadata
        )
    
    def update_state(self, trade: Trade) -> None:
        pass

    def _score_cluster(self, cluster) -> float:
        """Calculate confidence score based on cluster properties"""
        confidence = self.base_confidence

        if cluster.size >= self.size_threshold:
            confidence += self.size_bonus
        
        if cluster.density >= self.density_threshold:
            confidence += self.density_bonus
        
        if cluster.has_common_ownership:
            confidence += self.ownership_bonus
        
        return min(confidence, self.max_confidence)