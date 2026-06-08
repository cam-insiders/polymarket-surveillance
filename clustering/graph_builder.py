"""Reference implementation of incremental co-activity graph updates."""

import logging
from typing import Dict, List, Tuple
from collections import defaultdict

from clustering.models import ClusteringState, EntryRecord
from clustering.utils import compute_edge_weight
from models import Trade

class CoactivityGraphBuilder:
    """Incrementally builds and maintains a co-activity graph."""

    def __init__(self, config: Dict, state: ClusteringState):
        self.config = config
        self.state = state
        
        self.min_edge_weight = config.get("min_edge_weight", 0.1)

        self.significant_change_delta = config.get("significant_change_delta", self.min_edge_weight * 0.5)

        logging.info(
            f"CoactivityGraphBuilder initialized "
            f"(min_edge_weight={self.min_edge_weight}, "
            f"significant_delta={self.significant_change_delta})"
        )
    def on_trade(self, trade: Trade) -> List[Tuple[str, str, float, int]]:
        """Process a trade and update the co-activity graph accordingly."""

        wallet = trade.wallet
        market_id = trade.condition_id
        
        new_entry = EntryRecord(
            timestamp=trade.timestamp_ms // 1000,
            direction=trade.side,
            size=trade.notional_usdc,
            outcome_index=trade.outcome_index
        )

        updated_edges = []

        # Compare against other wallets in this market.
        if market_id in self.state.market_entries:
            market_wallets = self.state.market_entries[market_id]

            for other_wallet, existing_entries in market_wallets.items():
                if other_wallet == wallet:
                    continue

                weight_delta = 0.0

                for existing_entry in existing_entries:
                    pair_weight = compute_edge_weight(
                        new_entry, existing_entry, self.config
                    )
                    weight_delta += pair_weight

                if weight_delta > 0:
                    edge_key = self._normalise_edge_key(wallet, other_wallet)
                    current_weight = self._get_edge_weight(edge_key)

                    new_weight = current_weight + weight_delta

                    self.state.coactivity_graph.add_edge(
                        edge_key[0], edge_key[1], weight=new_weight
                    )

                    if weight_delta >= self.significant_change_delta:
                        self.state.significant_changes += 1
                    
                    updated_edges.append((
                        edge_key[0], edge_key[1], new_weight, new_entry.timestamp
                    ))
        
        if market_id not in self.state.market_entries:
            self.state.market_entries[market_id] = defaultdict(list)
        if wallet not in self.state.market_entries[market_id]:
            self.state.market_entries[market_id][wallet] = []
        self.state.market_entries[market_id][wallet].append(new_entry)

        return updated_edges

    def _normalise_edge_key(self, wallet_a: str, wallet_b: str) -> Tuple[str, str]:
        """Normalise edge key to ensure consistent ordering"""
        return tuple(sorted([wallet_a, wallet_b]))

    def _get_edge_weight(self, edge_key: Tuple[str, str]) -> float:
        """Retrieve the current weight of an edge, defaulting to 0.0 if not present"""
        edge_data = self.state.coactivity_graph.get_edge_data(edge_key[0], edge_key[1], default={})
        return edge_data.get("weight", 0.0)
