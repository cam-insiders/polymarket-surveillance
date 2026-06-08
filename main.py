#!/usr/bin/env python3
"""
Polymarket Insider Trading Detection System v1
Main orchestrator module.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading
from control_api import start_control_api

from clustering.manager import ClusteringManager
from clustering.models import ClusterInfo
from config import CONFIG
from detectors.clustering_detectors import ClusterCoordinationDetector
from models import Trade, Alert
from detectors import (
    DetectionContext,
    VolumeAnomalyDetector,
    ProbabilityImpactDetector,
    AccumulationDetector,
    ExtremePositionDetector,
    ContraOutcomeSilenceDetector,
    OrderbookConsumptionDetector,
    OrderbookImbalanceDetector,
    ThinLiquidityExploitDetector,
    RecidivismDetector,
    NewWalletDetector,
)
from market_manager import MarketMetadataManager
from data_fetcher import PolymarketDataFetcher, PolymarketOrderbookStream
from alert_manager import AlertManager
from wallet_cache import WalletCacheManager
from jump_anticipation.manager import JumpAnticipationManager

class InsiderTradingDetector:
    """Main system orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._market_lock = threading.Lock()
        self._start_time = time.time()
        
        logging.info("Fetching market metadata...")
        self.metadata_manager = MarketMetadataManager(config["gamma_api_url"])
        markets = self.metadata_manager.fetch_markets(config["market_slugs"])
        
        if not markets:
            raise RuntimeError("No markets loaded! Check your slugs.")
        
        logging.info(f"Loaded {len(markets)} markets")
        
        wallet_cache = WalletCacheManager(
            db_path=config["wallet_cache_db_path"],
            api_url=config["data_api_url"],
            new_wallet_threshold_minutes=config.get("new_wallet_threshold_minutes", 60)
        )

        self.context = DetectionContext(wallet_cache=wallet_cache)
        self.metadata_manager.register_with_context(self.context)

        self.clustering_manager = ClusteringManager(
            config=config,
            db_path=config.get("clustering_db_path", "clustering.db")
        )

        self.context.clustering_state = self.clustering_manager.state
        
        enable_trade_prefilter = config.get("enable_trade_prefilter", False)
        trade_prefilter_min_notional = config.get("trade_prefilter_min_notional", 500.0)

        self.fetcher = PolymarketDataFetcher(
            api_url=config["data_api_url"],
            min_notional=config["min_notional_filter"],
            max_trades=config["max_trades_per_request"],
            enable_prefilter=enable_trade_prefilter,
        )
        self.fetcher.set_condition_ids(self.metadata_manager.get_condition_ids())
        
        if enable_trade_prefilter:
            logging.info(
                f"Trade prefilter enabled: min notional = ${trade_prefilter_min_notional:.0f}"
            )
        else:
            logging.info("Trade prefilter disabled, all trades will be processed")

        self.alert_manager = AlertManager(config["db_path"], self.context)
        
        self.detectors = [
            VolumeAnomalyDetector(config["volume_anomaly"]),
            ProbabilityImpactDetector(config["probability_impact"]),
            AccumulationDetector(config["accumulation_detector"]),
            ExtremePositionDetector(config["extreme_position"]),
            ContraOutcomeSilenceDetector(config["contra_outcome_silence"]),
            OrderbookConsumptionDetector(config["orderbook_consumption"]),
            OrderbookImbalanceDetector(config["orderbook_imbalance"]),
            ThinLiquidityExploitDetector(config["thin_liquidity"]),

            # Disabled for now
            # RecidivismDetector(config["recidivism_detector"]),
            NewWalletDetector(config["new_wallet_detector"]),

            # Disabled for now, looking to add "SuspiciousCluster" detector instead
            # ClusterCoordinationDetector(config["cluster_coordination_detector"]),
        ]

        ja_config = config.get("jump_anticipation_config")
        if ja_config:
            self.jump_anticipation_manager = JumpAnticipationManager(ja_config)
        else:
            self.jump_anticipation_manager = None
            logging.info("Jump anticipation disabled (no jump_anticipation_config in CONFIG)")
        
        self.alert_threshold = config.get("alert_threshold", 0.5)
        
        self.orderbook_stream = None
        if config.get("enable_orderbook_stream", True):
            self._init_orderbook_stream()
    
    def _init_orderbook_stream(self):
        """Initialize websocket for orderbook data"""
        asset_ids = self.metadata_manager.get_all_asset_ids()
        
        if not asset_ids:
            logging.warning("No asset IDs available for orderbook stream")
            return
        
        self.orderbook_stream = PolymarketOrderbookStream(asset_ids, self.context)
        self.orderbook_stream.start()
        logging.info(f"Orderbook stream started for {len(asset_ids)} assets")
    
    def add_market(self, slug: str) -> dict:
        """
        Add a market slug at runtime.
        
        Fetches metadata from Gamma API, registers with context,
        updates the trade fetcher, and restarts the orderbook stream.
        """
        with self._market_lock:
            # Check if already monitored
            existing_slugs = set(self.metadata_manager.condition_to_slug.values())
            if slug in existing_slugs:
                return {"ok": False, "error": f"Slug '{slug}' is already being monitored"}

            # Fetch metadata
            new_markets = self.metadata_manager.fetch_markets([slug])
            if not new_markets:
                return {"ok": False, "error": f"No markets found for slug '{slug}'"}

            # Register new markets with detection context
            for market in new_markets:
                self.context.register_market_assets(market.condition_id, market.clob_token_ids)

            # Update trade fetcher
            self.fetcher.set_condition_ids(self.metadata_manager.get_condition_ids())

            # Restart orderbook stream with new asset list
            self._restart_orderbook_stream()

            market_names = [m.question[:60] for m in new_markets]
            logging.info(f"Added slug '{slug}': {len(new_markets)} markets")

            return {
                "ok": True,
                "slug": slug,
                "markets_added": len(new_markets),
                "market_names": market_names,
                "total_markets": len(self.metadata_manager.markets),
            }

    def remove_market(self, slug: str) -> dict:
        """Remove a market slug at runtime."""
        with self._market_lock:
            # Find all markets belonging to this slug
            keys_to_remove = []
            condition_ids_to_remove = []
            for key, market in self.metadata_manager.markets.items():
                market_slug = self.metadata_manager.condition_to_slug.get(market.condition_id)
                if market_slug == slug:
                    keys_to_remove.append(key)
                    condition_ids_to_remove.append(market.condition_id)

            if not keys_to_remove:
                return {"ok": False, "error": f"Slug '{slug}' is not currently monitored"}

            # Remove from metadata manager
            for key in keys_to_remove:
                del self.metadata_manager.markets[key]
            for cid in condition_ids_to_remove:
                self.metadata_manager.condition_to_slug.pop(cid, None)

            # Update trade fetcher
            self.fetcher.set_condition_ids(self.metadata_manager.get_condition_ids())

            # Restart orderbook stream
            self._restart_orderbook_stream()

            logging.info(
                f"Removed slug '{slug}': {len(keys_to_remove)} markets removed, "
                f"{len(self.metadata_manager.markets)} markets remaining"
            )

            return {
                "ok": True,
                "slug": slug,
                "markets_removed": len(keys_to_remove),
                "total_markets": len(self.metadata_manager.markets),
            }

    def _restart_orderbook_stream(self):
        """Stop and restart the orderbook websocket with current asset list."""
        if self.orderbook_stream:
            self.orderbook_stream.stop()
            logging.info("Orderbook stream stopped for reconfiguration")

        if self.config.get("enable_orderbook_stream", True):
            self._init_orderbook_stream()

    def get_status(self) -> dict:
        """Return system status for the control API."""
        uptime_s = time.time() - self._start_time
        hours, remainder = divmod(int(uptime_s), 3600)
        minutes, seconds = divmod(remainder, 60)

        slugs = sorted(set(self.metadata_manager.condition_to_slug.values()))
        condition_ids = self.metadata_manager.get_condition_ids()

        cluster_count = len(self.clustering_manager.state.cluster_metadata)
        clustered_wallets = len(self.clustering_manager.state.wallet_to_cluster)
        graph = self.clustering_manager.state.coactivity_graph
        trades_pending = self.clustering_manager.trades_since_last_cluster

        return {
            "uptime": f"{hours}h {minutes}m {seconds}s",
            "markets": {
                "slugs": slugs,
                "slug_count": len(slugs),
                "condition_id_count": len(condition_ids),
            },
            "clustering": {
                "clusters": cluster_count,
                "clustered_wallets": clustered_wallets,
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "trades_since_last_cluster": trades_pending,
            },
            "detectors": [d.name for d in self.detectors],
            "dedup_set_size": len(self.fetcher.seen_trades),
        }
    
    def process_trade(self, trade: Trade) -> Optional[Alert]:
        """Run all detectors on a trade"""
        
        # Only feed large trades to clustering (matches backtest validation).
        # All trades still go through individual detectors below.
        min_clustering_size = self.config.get("min_clustering_trade_size", 5000.0)
        if trade.notional_usdc >= min_clustering_size:
            self.clustering_manager.on_trade(trade)
        
        signals = []
        
        for detector in self.detectors:
            signal = detector.analyze(trade, self.context)
            if signal:
                signals.append(signal)
            detector.update_state(trade)
        
        self.context.add_trade(trade)

        # Persist wallet×market stats for classification & post-resolution eval
        self.alert_manager.record_trade(trade)
        
        # Noisy-OR
        if signals:
            # Compute P(not insider) = product of (1 - p_i)
            p_not_insider = 1.0
            for signal in signals:
                p_not_insider *= (1.0 - signal.confidence_score)
            
            # P(insider) = 1 - P(not insider)
            total_score = 1.0 - p_not_insider

            # ClusterBoost
            cluster = None
            if self.context.clustering_state:
                cluster = self.context.clustering_state.get_cluster_for_wallet(trade.wallet)

            if cluster:
                boost_factor = self._compute_cluster_boost(cluster)
                original_score = total_score
                total_score *= boost_factor
                total_score = min(total_score, 0.95)
                logging.debug(
                    f"Cluster boost: {original_score:.3f} -> {total_score:.3f} "
                    f"(x{boost_factor:.2f}, cluster {cluster.cluster_id})"
                )

            # Stage 3: Jump anticipation boost
            if self.jump_anticipation_manager is not None:
                ja_boost = self.jump_anticipation_manager.get_wallet_boost(trade.wallet)
                if ja_boost > 1.001:
                    total_score = min(total_score * ja_boost, 0.95)
                    logging.debug(
                        f"JA boost: x{ja_boost:.3f} -> score={total_score:.3f} "
                        f"| wallet={trade.wallet[:10]}..."
                    )

            if total_score >= self.alert_threshold:
                return Alert(
                    trade=trade,
                    signals=signals,
                    total_score=total_score,
                    timestamp=datetime.now()
                )
        
        return None
    
    def _compute_cluster_boost(self, cluster: ClusterInfo) -> float:
        """Calculate confidence boost based on cluster properties."""
        boost_config = self.config.get("cluster_boost", {})
        max_boost = float(boost_config.get("max_boost_factor", 2.0))
        size_weight = float(boost_config.get("size_weight", 0.4))
        density_weight = float(boost_config.get("density_weight", 0.2))
        size_normalizer = float(boost_config.get("size_normalizer", 50.0))

        boost = 1.0
        size_boost = min(cluster.size / size_normalizer, size_weight)
        boost += size_boost
        boost += cluster.density * density_weight

        if cluster.has_common_ownership:
            ownership_boost = float(boost_config.get("ownership_boost", 0.4))
            boost += ownership_boost

        return min(boost, max_boost)
    
    def process_batch(self) -> int:
        """Process one batch of trades"""
        trades = self.fetcher.fetch_recent_trades()
        alert_count = 0
        
        if trades:
            logging.info(f"Fetched {len(trades)} new trades")

        for trade in trades:
            alert = self.process_trade(trade)
            
            if alert:
                success = self.alert_manager.save_alert(alert)
                if success:
                    self.alert_manager.record_flag(
                        alert.trade.wallet,
                        alert.trade.condition_id,
                        alert.trade.timestamp_ms,
                    )
                    alert_count += 1
                    
                    slug = self.metadata_manager.condition_to_slug.get(
                        trade.condition_id, "unknown"
                    )
                    
                    logging.info(
                        f"ALERT | {alert.trade.wallet[:10]}... | "
                        f"{slug[:40]} | ${alert.trade.notional_usdc:.0f} | "
                        f"Score: {alert.total_score:.2f} | "
                        f"Detectors: {', '.join([s.detector_name for s in alert.signals])}"
                    )
        
        # Periodic jump anticipation re-scoring
        if self.jump_anticipation_manager is not None:
            for trade in trades:
                self.jump_anticipation_manager.on_trade(trade)
            self.jump_anticipation_manager.maybe_score()

        return alert_count
    
    def run(self):
        """Main event loop"""
        control_port = self.config.get("control_api_port", 8585)
        self._control_server = start_control_api(self, port=control_port)
        logging.info("=" * 80)
        logging.info("POLYMARKET INSIDER TRADING DETECTION SYSTEM v1")
        logging.info("=" * 80)
        logging.info(f"Monitoring {len(self.metadata_manager.markets)} markets:")
        for market in self.metadata_manager.markets.values():
            logging.info(f"   • {market.question[:70]}")
        logging.info(f"Active detectors: {len(self.detectors)}")
        for detector in self.detectors:
            logging.info(f"   • {detector.name}")
        logging.info("=" * 80)
        
        try:
            while True:
                alert_count = self.process_batch()
                if alert_count > 0:
                    logging.info(f"Batch complete: {alert_count} new alerts")
                time.sleep(self.config["poll_interval_seconds"])
        except KeyboardInterrupt:
            logging.info("\nShutting down...")
        finally:
            if self.orderbook_stream:
                self.orderbook_stream.stop()
            self.alert_manager.close()

            self.clustering_manager.close()

            if self.context.wallet_cache:
                self.context.wallet_cache.close()

            logging.info("Shutdown complete")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if not CONFIG.get("market_slugs"):
        raise SystemExit(
            "ERROR: No market slugs configured!\n"
            "Add market slugs to CONFIG['market_slugs'].\n"
            "Example: 'will-trump-win-the-2024-us-presidential-election'"
        )
    
    detector = InsiderTradingDetector(CONFIG)
    detector.run()

if __name__ == "__main__":
    main()
