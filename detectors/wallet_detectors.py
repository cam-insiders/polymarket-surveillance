"""Wallet/ID-based detectors"""

import logging
import math
from typing import Dict, Optional, Set, Tuple
from collections import defaultdict
from .base import Detector, DetectionContext
from models import Trade, DetectionSignal

class RecidivismDetector(Detector):
    """
    Detects wallets with repeated flags across multiple markets.
    High recidivism indicates a pattern of suspicious behavior.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_prior_flags = config.get("min_prior_flags", 1)
        # Sigmoid parameters: midpoint is where confidence is 0.5, 
        # k controls the steepness of the curve.
        self.midpoint = config.get("midpoint", 4)
        self.k = config.get("k", 0.7)
        self.max_confidence = config.get("max_confidence", 1.0)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        # NOTE: This currently counts individual detector signals as flags.
        # Future improvement: Only count 'combined' alerts (where total_score >= 0.5)
        # to reduce noise and focus on high-confidence recidivism.
        flag_count = context.wallet_flag_counts[trade.wallet]

        if flag_count >= self.min_prior_flags:
            # Sigmoid function for non-linear scaling: 1 / (1 + e^(-k * (x - midpoint)))
            # This ensures low weight for few flags and higher weight as they accumulate.
            confidence = 1.0 / (1.0 + math.exp(-self.k * (flag_count - self.midpoint)))
            confidence *= self.max_confidence

            return DetectionSignal(
                detector_name=self.name,
                confidence_score=round(confidence, 4),
                reason=f"Repeat offender: wallet has {flag_count} prior flags",
                metadata={
                    "flag_count": flag_count,
                }
            )      
          
        return None
    
    def update_state(self, trade: Trade) -> None:
        pass

class NewWalletDetector(Detector):
    """
    Flags trades from wallets that are new to the platform.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_notional = config.get("min_notional", 1000.0) 
        self.max_confidence = config.get("max_confidence", 0.7)

    def analyze(self, trade: Trade, context: DetectionContext) -> Optional[DetectionSignal]:
        if trade.notional_usdc < self.min_notional:
            return None
        
        if not context.wallet_cache:
            logging.warning(f"[{self.name}] Wallet cache not available, skipping")
            return None
        
        wallet_info = context.wallet_cache.get_wallet_info(trade.wallet)

        if wallet_info is None:
            logging.debug(f"[{self.name}] Could not fetch wallet info for {trade.wallet[:10]}, skipping")
            return None

        if wallet_info["is_new"]:
            # This is a genuinely new wallet making their first Polymarket trade
            confidence = self.max_confidence
            
            return DetectionSignal(
                detector_name=self.name,
                confidence_score=round(confidence, 4),
                reason=f"NEW WALLET: First trade on Polymarket, large size ${trade.notional_usdc:.0f}",
                metadata={
                    "notional": trade.notional_usdc,
                    "is_new_to_polymarket": True,
                    "historical_trades": 0,
                }
            )
        
        return None
    
    def update_state(self, trade: Trade) -> None:
        pass