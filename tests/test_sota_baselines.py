import pytest

from experiments.sota_algorithms.common import (
    build_wallet_prior_trade_counts_by_market,
    wallet_classification_metrics,
)
from experiments.sota_algorithms.isolation_forest import _select_matched_buy_indices
from experiments.sota_algorithms.timing_heuristic import run_timing_heuristic_market_baseline


def test_sota_algorithms_public_exports_use_current_faithful_modules():
    import experiments.sota_algorithms as sota_algorithms

    assert hasattr(sota_algorithms, "run_consob_pca_faithful_baseline")
    assert hasattr(sota_algorithms, "run_mitts_ofir_faithful_causal")
    assert hasattr(sota_algorithms, "run_mitts_ofir_faithful_retrospective")


def test_wallet_prior_trade_counts_use_other_markets_before_market_start(make_trade):
    entries = [
        (make_trade(wallet="0xseasoned", condition_id="m0", timestamp_ms=100), 10),
        (make_trade(wallet="0xseasoned", condition_id="m1", timestamp_ms=200), 1),
        (make_trade(wallet="0xlate", condition_id="m2", timestamp_ms=300), 2),
    ]

    counts = build_wallet_prior_trade_counts_by_market(entries)

    assert counts[1]["0xseasoned"] == 1
    assert "0xlate" not in counts[1]


def test_timing_heuristic_honors_seeded_platform_wallet_history(make_trade):
    trade = make_trade(wallet="0xseasoned", notional_usdc=5_000, timestamp_ms=100)

    flagged, counts = run_timing_heuristic_market_baseline(
        trades=[trade],
        resolution_timestamp_ms=200,
        params={
            "max_prior_trades": 0,
            "min_notional": 1_000.0,
            "max_hours": 1.0,
        },
        initial_wallet_trade_counts={"0xseasoned": 1},
    )

    assert flagged == set()
    assert counts == {}


def test_isolation_forest_matched_mode_flags_only_eval_buys():
    flagged, source = _select_matched_buy_indices(
        scores=[-0.5, -10.0, 0.1],
        buy_eval_indices=[0, 2],
        match_flag_rate=0.5,
    )

    assert source == "eval_buy_scores_retrospective"
    assert flagged == {0}


def test_wallet_classification_metrics_counts_random_baseline_style_flags():
    wallet_data = {
        1: {
            "0xtp": {"is_insider": True, "return": 1.0},
            "0xfp": {"is_insider": False, "return": -1.0},
            "0xfn": {"is_insider": True, "return": 0.5},
        }
    }

    metrics = wallet_classification_metrics(
        wallet_data,
        {1: {"0xtp", "0xfp"}},
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp_avg_return"] == pytest.approx(1.0)
    assert metrics["fp_avg_return"] == pytest.approx(-1.0)
