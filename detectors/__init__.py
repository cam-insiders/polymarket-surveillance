"""Detector modules for insider trading detection"""

from .base import Detector, DetectionContext, RollingStats
from .trade_detectors import (
    VolumeAnomalyDetector,
    ProbabilityImpactDetector,
    AccumulationDetector,
    ExtremePositionDetector,
    ContraOutcomeSilenceDetector,
)
from .orderbook_detectors import (
    OrderbookConsumptionDetector,
    OrderbookImbalanceDetector,
    ThinLiquidityExploitDetector,
)
from .wallet_detectors import (
    RecidivismDetector,
    NewWalletDetector,
)
from .clustering_detectors import (
    ClusterCoordinationDetector,
)

__all__ = [
    "Detector",
    "DetectionContext",
    "RollingStats",
    "VolumeAnomalyDetector",
    "ProbabilityImpactDetector",
    "AccumulationDetector",
    "ExtremePositionDetector",
    "ContraOutcomeSilenceDetector",
    "OrderbookConsumptionDetector",
    "OrderbookImbalanceDetector",
    "ThinLiquidityExploitDetector",
    "RecidivismDetector",
    "NewWalletDetector",
    "ClusterCoordinationDetector",
]