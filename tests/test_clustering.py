import networkx as nx
import pytest

from backtesting.backtest_runner import BacktestResult
from backtesting.bucket_clustering_backtest_runner import BucketClusteringBacktestRunner
from clustering.bucket_graph_builder import build_graph_from_trades_bucketed
from clustering.cluster_computer import ClusterComputer
from clustering.models import ClusterInfo, ClusteringState
from detectors.clustering_detectors import ClusterCoordinationDetector


def test_bucket_graph_aggregates_duplicate_wallets_before_projecting_edges(make_trade):
    trades = [
        make_trade(wallet="0xa", timestamp_ms=0, notional_usdc=1_000),
        make_trade(wallet="0xa", timestamp_ms=10_000, notional_usdc=3_000),
        make_trade(wallet="0xb", timestamp_ms=20_000, notional_usdc=4_000),
    ]

    graph = build_graph_from_trades_bucketed(
        trades=trades,
        config={"bucket_size": 300, "size_normalizer": 1_000, "max_size_mult": 10.0},
        market_id="1",
    )

    assert graph.number_of_edges() == 1
    assert graph["0xa"]["0xb"]["weight"] == pytest.approx(4.0)


def test_bucket_graph_hard_excludes_cross_side_and_cross_outcome_pairs(make_trade):
    trades = [
        make_trade(wallet="0xa", timestamp_ms=0, side="BUY", outcome_index=0),
        make_trade(wallet="0xb", timestamp_ms=10_000, side="SELL", outcome_index=0),
        make_trade(wallet="0xc", timestamp_ms=20_000, side="BUY", outcome_index=1),
    ]

    graph = build_graph_from_trades_bucketed(
        trades=trades,
        config={"bucket_size": 300, "size_normalizer": 1_000, "max_size_mult": 10.0},
        market_id="1",
    )

    assert graph.number_of_edges() == 0


def test_cluster_computer_returns_dense_cluster_metadata():
    graph = nx.Graph()
    graph.add_edge("0xa", "0xb", weight=1.0)
    graph.add_edge("0xb", "0xc", weight=1.0)
    graph.add_edge("0xa", "0xc", weight=1.0)
    graph.add_edge("0xa", "0xd", weight=0.1)

    wallet_to_cluster, metadata = ClusterComputer({"k_core": 2, "min_edge_weight": 0.5}).compute_clusters(graph)

    assert set(wallet_to_cluster) == {"0xa", "0xb", "0xc"}
    assert len(metadata) == 1
    cluster = next(iter(metadata.values()))
    assert cluster.size == 3
    assert cluster.density == pytest.approx(1.0)
    assert cluster.total_edge_weight == pytest.approx(3.0)


def test_cluster_coordination_detector_scores_cluster_properties(make_trade):
    state = ClusteringState()
    state.wallet_to_cluster = {"0xa": 7}
    state.cluster_metadata = {
        7: ClusterInfo(
            cluster_id=7,
            size=5,
            density=0.9,
            total_edge_weight=12.0,
            wallets={"0xa", "0xb", "0xc", "0xd", "0xe"},
            has_common_ownership=True,
        )
    }
    context = type("Context", (), {"clustering_state": state})()
    detector = ClusterCoordinationDetector(
        {
            "base_confidence": 0.2,
            "size_threshold": 5,
            "size_bonus": 0.2,
            "density_threshold": 0.8,
            "density_bonus": 0.2,
            "ownership_bonus": 0.3,
            "max_confidence": 0.85,
        }
    )

    signal = detector.analyze(make_trade(wallet="0xa"), context)

    assert signal is not None
    assert signal.confidence_score == 0.85
    assert signal.metadata["cluster_id"] == 7
    assert detector.analyze(make_trade(wallet="0xoutside"), context) is None


def test_bucket_clustering_runner_boost_formula_and_application():
    runner = BucketClusteringBacktestRunner(
        detector_config={},
        clustering_config={
            "boost": {
                "max_boost_factor": 2.0,
                "size_weight": 0.4,
                "density_weight": 0.2,
                "ownership_boost": 0.4,
                "size_normalizer": 10.0,
            }
        },
    )
    cluster = ClusterInfo(1, 3, 0.5, 3.0, {"0xa", "0xb"}, has_common_ownership=True)

    assert runner._compute_cluster_boost(cluster) == pytest.approx(1.8)

    result = BacktestResult(
        total_trades=0,
        alerts_generated=0,
        alerts=[],
        detector_stats={},
        all_trade_features=[],
        wallet_suspicion={"0xa": 1.0, "0xoutside": 1.0},
    )
    runner.apply_precomputed_wallet_boosts(result, {"0xa": 1.8}, {"0xa": True})

    assert result.wallet_cluster_boost == {"0xa": 1.8, "0xoutside": 1.0}
    assert result.wallet_has_common_ownership == {"0xa": True, "0xoutside": False}
