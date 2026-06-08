"""Configuration for Polymarket Insider Trading Detection System"""

import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # API settings
    "data_api_url": "https://data-api.polymarket.com/trades",
    "gamma_api_url": "https://gamma-api.polymarket.com",
    "poll_interval_seconds": 5,
    "max_trades_per_request": 500,

    # Control API
    "control_api_port": 8585,
    
    "market_slugs": [
        "fed-decision-in-march",
        "will-trump-visit-china-by",
        "where-will-zelenskyy-and-putin-meet-next",
        "us-x-iran-military-engagement-by",
        # Add more slugs from Polymarket URLs
        # "nvda-quarterly-earnings-nongaap-eps",
        # "lead-bank-in-spacexs-ipo",
        "kraken-ipo-in-2025",
        "ipos-before-2027",
        "clarity-act-signed-into-law-in-2026",
        "bitcoin-all-time-high-by",
        "ethereum-all-time-high-by",
        "bitcoin-vs-gold-vs-sp-500-in-2026",
        "fed-rate-cut-by",
        "gc-hit-jun-2026",
    ],

    # Orderbook stream
    "enable_orderbook_stream": True,

    "alert_threshold": 0.5,  # Overall score threshold to trigger alert
    
    # Database
    "db_path": "insider_trading_v1.db",
    "wallet_cache_db_path": "wallet_cache_v1.db",
    "clustering_db_path": "clustering_v1.db",

    "new_wallet_threshold_minutes": 60,  # Consider wallets with only recent trades as "new"
    
    # Trade ingestion / filtering
    # Default: DO NOT pre-filter. Small trades should still update wallet history,
    # rolling stats, exits, and other context. Detector-level min_notional checks
    # decide whether a trade is eligible to fire a signal.
    "enable_trade_prefilter": False,
    "min_notional_filter": 500.0,  # only used if enable_trade_prefilter=True
    # Detection thresholds
    "min_clustering_trade_size": 5000.0,  # Only trades >= $5k are used for clustering graph

    "clustering": {
        # Bucket graph builder parameters (optimized via Stage 2 backtest)
        "bucket_size": 300,               # 5-minute time buckets
        "same_direction_mult": 2.0,       # Same direction = 2x weight (encoded in bucket key)
        "size_normalizer": 12000,         # $12k = 1.0 size multiplier
        "max_size_mult": 5.0,             # Cap on size effect
        "cross_outcome_penalty": 0.1,     # Penalty for opposite outcomes

        # Clustering algorithm
        "k_core": 2,                      # Minimum degree for k-core
        "min_edge_weight": 20,            # Edge weight threshold (high = fewer, denser clusters)

        # Clustering trigger
        "min_cluster_interval": 300,      # Don't recluster more often than every 5 min
        "max_cluster_interval": 3600,     # Force recluster at least every hour
        "significant_change_threshold": 20,  # Recluster after 20 new clustering-eligible trades
    },

    "cluster_boost": {
        "max_boost_factor": 2.0,        # Maximum multiplier on suspicion score
        "size_weight": 0.3,              # How much cluster size matters
        "density_weight": 0.2,           # How much density matters
        "ownership_boost": 0.4,          # Boost for confirmed common ownership (Layer 2)
        "size_normalizer": 50.0,         # Cluster size denominator
        "max_final_score": 0.95,         # Cap boosted scores
    },

    # Jump anticipation scorer
    # Parameters are optimised offline by the timeframe optimizer pipeline
    # and persisted into best_config.json. These values serve as live defaults
    # and are overridden at startup if a config file is loaded.
    "jump_anticipation_config": {
        "jump_threshold": 0.05,           # min |Delta p| to qualify as a jump
        "jump_window_minutes": 30,        # time window over which to measure price change
        "pre_jump_lookback_minutes": 60,  # how far before a jump to look for suspicious trades
        "min_pre_jump_trades": 2,         # min pre-jump trades required to score a wallet
        "max_boost_factor": 2.0,          # maximum boost multiplier on Noisy-OR score
        "min_trade_notional": 0.0,        # min trade size to count (0 = all trades)
        "scoring_interval_minutes": 15,   # how often the live scorer re-runs
        "buffer_hours": 24,               # how far back the rolling trade buffer extends
    },

    # Layer 2: Attribution Analysis (USDC transfer detection)
    "attribution": {
        # Polygonscan API settings
        "api_key": os.getenv("POLYGONSCAN_API_KEY", ""),
        "api_url": "https://api.etherscan.io/v2/api?chainid=137", # Polygonscan base URL (uses Etherscan)
        "usdc_contract": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # Polygon USDC
        # Rate limiting
        "max_requests_per_second": 5,        # Polygonscan free tier limit
        "retry_max_attempts": 3,              # Retry on failures
        "retry_backoff_base": 2,              # Exponential backoff: 2s, 4s, 8s
        # Query parameters
        "transfer_lookback_days": None,       # None = all-time, or set to 30/90 for recent
        # Enrichment trigger (lazy strategy)
        "min_size_for_attribution": 5,        # Only enrich clusters with ≥5 wallets
        "min_density_for_attribution": 0.4,   # Only enrich high-density clusters
        # Ownership detection threshold
        "ownership_flag_threshold": 1,        # 1 = any transfer flags ownership
    },

    # Clustering based detectors

    "cluster_coordination_detector": {
        "base_confidence": 0.2,        # Being in any cluster
        "size_threshold": 5,            # Large cluster threshold
        "size_bonus": 0.2,              # Confidence boost for large
        "density_threshold": 0.8,       # High density threshold
        "density_bonus": 0.2,           # Confidence boost for density
        "ownership_bonus": 0.3,         # Confidence boost for ownership
        "max_confidence": 0.85,         # Confidence cap
    },

    # CORE DETECTORS
    "volume_anomaly": {
        "lookback_window_hours": 24,
        "min_trades_for_baseline": 10,
        "z_score_threshold": 3.0,
        "min_absolute_notional": 1000.0,
        "max_confidence": 0.6,  # Even extreme outliers shouldn't be >60% certain
    },
    
    "probability_impact": {
        "min_delta_prob": 0.03,
        "min_delta_log_odds": 0.4,
        "min_notional": 500.0,
        "max_confidence": 0.7,  # Price impact is strong signal
    },
    
    "accumulation_detector": {
        "min_accumulation_usdc": 5000.0, # Flag after they build a $5k position
        "min_directional_ratio": 0.8,   # Must be mostly buying (80% directional)
        "max_confidence": 0.8,  # Strong signal if they meet criteria
        "min_outcome_concentration": 0.9,  # 90% on one outcome
    },
    
    "extreme_position": {
        "tail_threshold": 0.20,
        "min_notional": 1000.0,
        "max_confidence": 0.4,  # Tail bets can be legitimate speculation
    },

    "contra_outcome_silence": {
        "min_gap_samples": 10,
        "silence_threshold": 5.0,
        "min_notional": 1000.0,
        "max_contra_age_minutes": 120.0,
        "max_confidence": 0.3,  # Market-maker withdrawal is a strong corroborating signal
    },

    # Orderbook detectors
    "orderbook_consumption": {
        "min_levels_consumed": 3,
        "min_notional": 1000.0,
        "max_slippage_bps": 50,
        "max_confidence": 0.5,  # Urgency indicator but not conclusive
    },
    
    "orderbook_imbalance": {
        "min_imbalance_ratio": 3.0,
        "min_notional": 800.0,
        "max_confidence": 0.45,  # Trading against flow is suspicious but common
    },
    
    "thin_liquidity": {
        "min_depth_ratio": 0.3,
        "max_total_depth": 5000.0,
        "min_notional": 500.0,
        "max_confidence": 0.55,  # High impact in thin markets is notable
    },

    # Wallet-based detectors
    "recidivism_detector": {
        "min_prior_flags": 1,
        "midpoint": 5,
        "k": 0.7,
        "max_confidence": 0.8,  # Repeat offenders are highly suspicious
    },

    "new_wallet_detector": {
        "min_notional": 1000.0,
        "max_confidence": 0.4,  # New wallet alone is weak signal
    },
}
