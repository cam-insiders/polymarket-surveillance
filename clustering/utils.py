"""Reference utilities for the incremental clustering implementation."""

import math
from typing import Dict, Optional
from clustering.models import EntryRecord
from clustering.usdc_transfer_provider import UsdcTransferProvider

def compute_edge_weight(
    entry_i: EntryRecord,
    entry_j: EntryRecord,
    config: Dict
) -> float:
    """Compute co-activity edge weight between two entry records."""
    max_time_window = config.get("max_time_window", 3600)
    same_direction_mult = config.get("same_direction_mult", 2.0)
    size_normalizer = config.get("size_normalizer", 10000)
    max_size_mult = config.get("max_size_mult", 5.0)
    cross_outcome_penalty = config.get("cross_outcome_penalty", 0.1)
    time_diff = abs(entry_i.timestamp - entry_j.timestamp)
    
    if time_diff > max_time_window:
        return 0.0
    
    time_weight = 1.0 - (time_diff / max_time_window)
    
    direction_mult = (
        same_direction_mult 
        if entry_i.direction == entry_j.direction 
        else 1.0
    )
    
    size_mult = min(
        math.sqrt(entry_i.size * entry_j.size) / size_normalizer,
        max_size_mult
    )
    
    if entry_i.outcome_index is not None and entry_j.outcome_index is not None:
        if entry_i.outcome_index != entry_j.outcome_index:
            return time_weight * cross_outcome_penalty * size_mult * direction_mult

    return time_weight * direction_mult * size_mult

def create_attribution_provider(
    cache_db_path: str = "data/usdc_transfers.db",
    enable_api: bool = True,
    api_key: Optional[str] = None,
) -> Optional[UsdcTransferProvider]:
    """Create a USDC transfer provider when cache/API settings allow it."""
    import os
    from clustering.usdc_transfer_provider import UsdcTransferProvider
    
    polygonscan_config = None
    
    if enable_api:
        key = api_key or os.environ.get("POLYGONSCAN_API_KEY")
        if key:
            polygonscan_config = {
                "api_key": key,
                "api_url": "https://api.polygonscan.com/api",
                "usdc_contract": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                "max_requests_per_second": 4.5,
            }
    
    return UsdcTransferProvider(
        cache_db_path=cache_db_path,
        polygonscan_config=polygonscan_config,
    )
