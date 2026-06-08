"""Market metadata management for Polymarket"""

import json
import logging
from typing import Dict, List, Optional

import requests

from models import MarketMetadata
from detectors.base import DetectionContext


class MarketMetadataManager:
    """
    Fetches and caches market metadata from slugs.
    Handles both events (multi-market) and individual markets.
    """
    
    def __init__(self, gamma_api_url: str):
        self.gamma_api_url = gamma_api_url
        self.markets: Dict[str, MarketMetadata] = {}
        self.condition_to_slug: Dict[str, str] = {}
        
    def fetch_markets(self, slugs: List[str]) -> List[MarketMetadata]:
        """Fetch metadata for all provided slugs"""
        fetched = []
        
        for slug in slugs:
            try:
                event_markets = self._fetch_event_markets(slug)
                if event_markets:
                    fetched.extend(event_markets)
                    logging.info(f"✓ Loaded event '{slug}': {len(event_markets)} markets")
                    continue
                
                market = self._fetch_single_market(slug)
                if market:
                    self.markets[slug] = market
                    self.condition_to_slug[market.condition_id] = slug
                    fetched.append(market)
                    logging.info(f"✓ Loaded market: {market.question[:60]}")
                else:
                    logging.warning(f"✗ No event or market found for slug: {slug}")
                    
            except Exception as e:
                logging.error(f"✗ Failed to fetch '{slug}': {e}")
        
        return fetched
    
    def _fetch_event_markets(self, event_slug: str) -> List[MarketMetadata]:
        """Fetch all markets within an event."""
        url = f"{self.gamma_api_url}/events/slug/{event_slug}"
        
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            event = resp.json()
            
            if not isinstance(event, dict):
                return []
            
            markets_data = event.get("markets", [])
            if not markets_data:
                logging.debug(f"Event {event_slug} has no markets")
                return []
            
            event_title = event.get("title", event_slug)
            parsed_markets = []
            
            for market_data in markets_data:
                market = self._parse_market_data(market_data, event_slug, event_title)
                if market:
                    market_key = f"{event_slug}:{market.condition_id}"
                    self.markets[market_key] = market
                    self.condition_to_slug[market.condition_id] = event_slug
                    parsed_markets.append(market)
            
            return parsed_markets
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return []
            raise
    
    def _fetch_single_market(self, market_slug: str) -> Optional[MarketMetadata]:
        """Fetch a single standalone market"""
        url = f"{self.gamma_api_url}/markets"
        params = {"slug": market_slug}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not isinstance(data, list) or len(data) == 0:
                return None
            
            return self._parse_market_data(data[0], market_slug)
            
        except requests.HTTPError:
            return None
    
    def _parse_market_data(self, market_data: dict, slug: str, 
                          event_title: Optional[str] = None) -> Optional[MarketMetadata]:
        """Parse raw market JSON into MarketMetadata"""
        try:
            condition_id = market_data.get("conditionId")
            if not condition_id:
                logging.warning(f"Market in {slug} has no conditionId")
                return None
            
            clob_token_ids_raw = market_data.get("clobTokenIds", "")
            if isinstance(clob_token_ids_raw, str):
                clob_token_ids_raw = clob_token_ids_raw.strip()
                if clob_token_ids_raw.startswith('['):
                    clob_token_ids = json.loads(clob_token_ids_raw)
                else:
                    clob_token_ids = [tid.strip() for tid in clob_token_ids_raw.split(',') if tid.strip()]
            elif isinstance(clob_token_ids_raw, list):
                clob_token_ids = clob_token_ids_raw
            else:
                clob_token_ids = []
            
            if not clob_token_ids:
                logging.warning(f"Market {condition_id} has no clobTokenIds")
                return None
            
            question = market_data.get("question", "Unknown")
            
            if event_title and event_title.lower() not in question.lower():
                question = f"{event_title}: {question}"
            
            outcomes_raw = market_data.get("outcomes", "[]")
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            elif isinstance(outcomes_raw, list):
                outcomes = outcomes_raw
            else:
                outcomes = ["Yes", "No"]
            
            end_date = market_data.get("endDate")
            
            return MarketMetadata(
                slug=slug,
                condition_id=condition_id,
                clob_token_ids=clob_token_ids,
                question=question,
                outcomes=outcomes,
                end_date_iso=end_date,
            )
            
        except (KeyError, ValueError, TypeError) as e:
            logging.error(f"Failed to parse market data: {e}")
            return None
    
    def get_condition_ids(self) -> List[str]:
        """Get all condition IDs for trade API"""
        return list(self.condition_to_slug.keys())
    
    def get_all_asset_ids(self) -> List[str]:
        """Get all asset IDs for websocket"""
        asset_ids = []
        for market in self.markets.values():
            asset_ids.extend(market.clob_token_ids)
        return asset_ids
    
    def register_with_context(self, context: DetectionContext):
        """Register all markets with detection context"""
        for market in self.markets.values():
            context.register_market_assets(market.condition_id, market.clob_token_ids)