"""
Market resolution data for backtesting validation.
Manually curated list of resolved markets with known outcomes.
Not used anymore
"""

import json
import os
from typing import Dict

# Market resolutions: market_id -> winning outcome index (0 or 1)
RESOLVED_MARKETS = {
    # Trump Election - Won (Yes = outcome 0)
    # 253591: {
    #     'market_slug': 'will-donald-trump-win-the-2024-us-presidential-election',
    #     'winning_outcome': 0,  # "Yes" won
    #     'resolution_date': '2024-11-06',
    #     'notes': 'Trump won 2024 election'
    # },

    # 253597: {
    #     'market_slug': 'will-kamala-harris-win-the-2024-us-presidential-election',
    #     'winning_outcome': 1,  # "No" won
    #     'resolution_date': '2024-11-06',
    #     'notes': 'Harris lost 2024 election'
    # },

    551651: {
        'market_slug': 'israel-military-action-against-iran-by-friday-477',
        'winning_outcome': 0,  # "Yes" won
        'resolution_date': '2025-06-13',
        'notes': 'Israel took military action against Iran in 2025'
    },

    552658: {
        'market_slug': 'israel-announces-end-of-military-operations-against-iran-before-july',
        'winning_outcome': 0,  # "Yes" won
        'resolution_date': '2025-06-24',
        'notes': 'Israel announced end of operations against Iran in 2025'
    },

    # 533744: {
    #     'market_slug': 'will-trump-lower-tarrifs-on-china-in-april',
    #     'winning_outcome': 0,  # "Yes" won
    #     'resolution_date': '2025-04-16',
    #     'notes': 'Trump lowered tariffs on China in April 2025'
    # },

    # 514049: {
    #     'market_slug': 'will-trump-impose-25-tarriff-on-mexicocanada',
    #     'winning_outcome': 1,  # "Yes" won
    #     'resolution_date': '2025-02-04',
    #     'notes': 'Trump did not impose 25% tariff on Mexico/Canada before February 2025'
    # },

    # 504494: {
    #     'market_slug': 'fed-increases-interest-rates-by-25-bps-after-november-2024-meeting',
    #     'winning_outcome': 1,  # "Yes" won
    #     'resolution_date': '2024-11-07',
    #     'notes': 'Fed decreased interest rates by 25 bps after November 2024 meeting'
    # },

    # 528974: {
    #     'market_slug': 'no-change-in-fed-interest-rates-after-july-2025-meeting',
    #     'winning_outcome': 0,  # "No" won
    #     'resolution_date': '2025-07-30',
    #     'notes': 'No change in Fed interest rates after July 2025 meeting'
    # },

    # 525015: {
    #     'market_slug': 'will-robert-francis-prevost-be-the-next-pope',
    #     'winning_outcome': 0,  # "Yes" won
    #     'resolution_date': '2025-05-08',
    #     'notes': 'Robert Francis Prevost was elected pope in 2025 conclave'
    # }
}

_RUNTIME_OVERRIDES_LOADED = False


def _load_runtime_overrides_if_needed() -> None:
    """
    Optionally merge runtime resolution overrides from JSON file.

    Set env var: POLYMARKET_RESOLUTIONS_OVERRIDE_PATH=/path/to/file.json

    Accepted JSON formats:
      1) {"123": 0, "456": 1}
      2) {"123": {"winning_outcome": 0, ...}, ...}
    """
    global _RUNTIME_OVERRIDES_LOADED
    if _RUNTIME_OVERRIDES_LOADED:
        return
    _RUNTIME_OVERRIDES_LOADED = True

    path = os.getenv("POLYMARKET_RESOLUTIONS_OVERRIDE_PATH", "").strip()
    if not path:
        return
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return

    if not isinstance(payload, dict):
        return

    for raw_market_id, raw_value in payload.items():
        try:
            market_id = int(raw_market_id)
        except (TypeError, ValueError):
            continue

        if isinstance(raw_value, dict):
            if "winning_outcome" not in raw_value:
                continue
            try:
                winning_outcome = int(raw_value["winning_outcome"])
            except (TypeError, ValueError):
                continue
            entry: Dict = dict(raw_value)
            entry["winning_outcome"] = winning_outcome
            entry.setdefault("market_slug", f"market-{market_id}")
            entry.setdefault("resolution_date", "runtime_override")
            entry.setdefault("notes", "Loaded from runtime override file")
            RESOLVED_MARKETS[market_id] = entry
            continue

        try:
            winning_outcome = int(raw_value)
        except (TypeError, ValueError):
            continue

        RESOLVED_MARKETS[market_id] = {
            "market_slug": f"market-{market_id}",
            "winning_outcome": winning_outcome,
            "resolution_date": "runtime_override",
            "notes": "Loaded from runtime override file",
        }


def get_winning_outcome(market_id: int):
    """Get winning outcome for a market, or None if not in dataset"""
    _load_runtime_overrides_if_needed()
    if market_id in RESOLVED_MARKETS:
        return RESOLVED_MARKETS[market_id]['winning_outcome']
    return None


def get_resolved_market_ids():
    """Get list of all market IDs we have resolutions for"""
    _load_runtime_overrides_if_needed()
    return list(RESOLVED_MARKETS.keys())


def is_market_resolved(market_id: int):
    """Check if we have resolution data for this market"""
    _load_runtime_overrides_if_needed()
    return market_id in RESOLVED_MARKETS


def get_market_info(market_id: int):
    """Get full info about a resolved market"""
    _load_runtime_overrides_if_needed()
    return RESOLVED_MARKETS.get(market_id)
