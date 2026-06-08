"""
Helpers shared by optimizers and evaluation scripts.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from backtesting.causal_boost_replay import BoostSchedule, build_live_parity_boost_schedule
from backtesting.cached_evaluator import fast_evaluate_wallets, precompute_ground_truth
from models import filter_trades_by_notional


logger = logging.getLogger(__name__)


def get_backtest_worker_count(
    max_workers: int,
    candidate_count: int,
    *,
    enable_layer2_attribution: bool = False,
    clustering_enabled: bool = False,
    live_layer2_fetches: bool = True,
) -> int:
    """
    Cap worker count for backtests that may perform live Layer 2 attribution.
    """
    if candidate_count <= 0:
        return 0

    worker_count = max(1, min(int(max_workers), int(candidate_count)))
    if (
        worker_count > 1
        and enable_layer2_attribution
        and clustering_enabled
        and live_layer2_fetches
    ):
        logger.info(
            "Layer 2 attribution with clustering enabled; forcing a single "
            "backtest worker to avoid API rate limits."
        )
        return 1
    return worker_count


def load_all_trades_for_market(loader, market_id: int):
    """
    Load the complete market trade history without applying detector notional
    or active timeframe replay filters.
    """
    try:
        return loader.get_trades_for_market(
            market_id,
            min_usd_amount=None,
            use_cache=False,
            ignore_trade_time_bounds=True,
        )
    except TypeError:
        return loader.get_trades_for_market(market_id)


def build_attribution_provider(
    enable_layer2_attribution: bool = False,
    usdc_cache_db: str = "data/usdc_transfers.db",
    polygonscan_api_key: Optional[str] = None,
):
    """
    Create a cache-first attribution provider when Layer 2 is enabled.

    When no API key is available the provider still runs in cache-only mode.
    """
    if not enable_layer2_attribution:
        return None

    try:
        from clustering.usdc_transfer_provider import UsdcTransferProvider

        polygonscan_config = None
        api_key = polygonscan_api_key or os.environ.get("POLYGONSCAN_API_KEY")
        if api_key:
            polygonscan_config = {
                "api_key": api_key,
                "api_url": "https://api.etherscan.io/v2/api",
                "chain_id": 137,
                "usdc_contract": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                "max_requests_per_second": 4.5,
            }

        return UsdcTransferProvider(
            cache_db_path=usdc_cache_db,
            polygonscan_config=polygonscan_config,
        )
    except Exception as exc:
        logger.warning(f"Failed to initialize Layer 2 attribution provider: {exc}")
        return None


def precompute_wallet_cluster_boosts(
    *,
    loader,
    market_ids: List[int],
    clustering_config: Optional[Dict],
    clustering_min_trade_size: float = 5000.0,
    min_usd_amount: Optional[float] = None,
    enable_layer2_attribution: bool = False,
    usdc_cache_db: str = "data/usdc_transfers.db",
    polygonscan_api_key: Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Precompute detector-stage wallet boost maps once for a fixed clustering config.
    """
    if clustering_config is None or not market_ids:
        return {}

    from backtesting.bucket_clustering_backtest_runner import BucketClusteringBacktestRunner

    provider = build_attribution_provider(
        enable_layer2_attribution=enable_layer2_attribution,
        usdc_cache_db=usdc_cache_db,
        polygonscan_api_key=polygonscan_api_key,
    )
    boost_maps: Dict[int, Dict[str, float]] = {}

    logger.info(
        "Precomputing frozen clustering boosts for %d market(s)%s.",
        len(market_ids),
        " with Layer 2 attribution" if enable_layer2_attribution else "",
    )

    try:
        runner = BucketClusteringBacktestRunner(
            detector_config={},
            clustering_config=clustering_config,
            attribution_provider=provider,
        )

        for market_id in market_ids:
            try:
                all_trades = loader.get_trades_for_market(
                    market_id,
                    min_usd_amount=None,
                    use_cache=False,
                )
            except TypeError:
                all_trades = loader.get_trades_for_market(market_id)
            if min_usd_amount is not None:
                detector_trades = filter_trades_by_notional(all_trades, min_usd_amount)
            else:
                detector_trades = all_trades

            graph_trades = filter_trades_by_notional(
                detector_trades,
                clustering_min_trade_size,
            )
            boost_maps[market_id] = runner.compute_wallet_cluster_boosts(
                graph_trades=graph_trades,
                market_id=str(market_id),
                fetch_if_missing=True,
            )
    finally:
        if provider is not None:
            provider.close()

    return boost_maps


def precompute_causal_boost_schedules(
    *,
    loader,
    market_ids: List[int],
    clustering_config: Optional[Dict],
    jump_anticipation_config: Optional[Dict],
    min_usd_amount: Optional[float] = None,
    clustering_min_trade_size: float = 5000.0,
    poll_interval_seconds: float = 5.0,
    enable_layer2_attribution: bool = False,
    usdc_cache_db: str = "data/usdc_transfers.db",
    polygonscan_api_key: Optional[str] = None,
) -> Dict[int, BoostSchedule]:
    """
    Precompute causal per-trade boost schedules for a fixed boost configuration.
    """
    if (clustering_config is None and jump_anticipation_config is None) or not market_ids:
        return {}

    provider = build_attribution_provider(
        enable_layer2_attribution=enable_layer2_attribution,
        usdc_cache_db=usdc_cache_db,
        polygonscan_api_key=polygonscan_api_key,
    )
    schedules: Dict[int, BoostSchedule] = {}

    logger.info(
        "Precomputing causal boost schedules for %d market(s)%s.",
        len(market_ids),
        " with Layer 2 attribution" if enable_layer2_attribution else "",
    )

    try:
        for market_id in market_ids:
            try:
                all_trades = loader.get_trades_for_market(
                    market_id,
                    min_usd_amount=None,
                    use_cache=False,
                )
            except TypeError:
                all_trades = loader.get_trades_for_market(market_id)
            if min_usd_amount is not None:
                detector_trades = filter_trades_by_notional(all_trades, min_usd_amount)
            else:
                detector_trades = all_trades

            schedules[market_id] = build_live_parity_boost_schedule(
                detector_trades=detector_trades,
                market_id=str(market_id),
                clustering_config=clustering_config,
                clustering_min_trade_size=clustering_min_trade_size,
                jump_anticipation_config=jump_anticipation_config,
                poll_interval_seconds=poll_interval_seconds,
                attribution_provider=provider,
                fetch_if_missing=enable_layer2_attribution,
            )
    finally:
        if provider is not None:
            provider.close()

    return schedules


def evaluate_wallets_with_ground_truth(
    *,
    all_trades,
    backtest_result,
    market_metadata: Dict,
    evaluator,
    winning_outcome_override: Optional[int] = None,
) -> List[Dict]:
    """
    Evaluate wallets against ground truth computed from the unfiltered trade set.
    """
    gt = precompute_ground_truth(
        trades=all_trades,
        market_metadata=market_metadata,
        label_metric=evaluator.label_metric,
        z_score_threshold=evaluator.z_score_threshold,
        min_wallet_notional=evaluator.min_wallet_notional,
        winning_outcome_override=winning_outcome_override,
    )
    if gt is not None:
        return fast_evaluate_wallets(gt, backtest_result)

    return evaluator.evaluate_wallets(
        backtest_result,
        market_metadata,
        winning_outcome_override=winning_outcome_override,
    )
