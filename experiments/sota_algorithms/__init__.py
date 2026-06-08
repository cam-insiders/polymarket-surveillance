"""
SOTA anomaly-detection baselines used by experiments.compare_sota.

Modules:
  common            — shared helpers (wallet labels, copytrade summary, resolution times)
  isolation_forest  — Liu et al. (2008)
  timing_heuristic  — practitioner timing rule
  consob_pca_faithful — faithful four-condition CONSOB screen (non-causal, per-market eval-fit)
  mitts_ofir_faithful — faithful five-signal pair-level screen (retrospective + causal)
  random_baseline   — random null
"""

from experiments.sota_algorithms.common import (
    build_wallet_insider_labels,
    copytrade_trade_summary,
    get_market_resolution_timestamp_ms,
    wallet_flagged_pnl_from_evaluations,
    wallet_flagged_pnl_from_wallet_data,
)
from experiments.sota_algorithms.consob_pca_faithful import (
    run_consob_pca_faithful_baseline,
)
from experiments.sota_algorithms.isolation_forest import (
    IF_FEATURE_NAMES,
    extract_isolation_forest_features,
    run_isolation_forest_baseline,
)
from experiments.sota_algorithms.mitts_ofir_faithful import (
    run_mitts_ofir_faithful_causal,
    run_mitts_ofir_faithful_retrospective,
)
from experiments.sota_algorithms.random_baseline import run_random_baseline
from experiments.sota_algorithms.timing_heuristic import run_timing_heuristic_baseline

__all__ = [
    "IF_FEATURE_NAMES",
    "build_wallet_insider_labels",
    "copytrade_trade_summary",
    "extract_isolation_forest_features",
    "get_market_resolution_timestamp_ms",
    "run_isolation_forest_baseline",
    "run_mitts_ofir_faithful_causal",
    "run_mitts_ofir_faithful_retrospective",
    "run_consob_pca_faithful_baseline",
    "run_random_baseline",
    "run_timing_heuristic_baseline",
    "wallet_flagged_pnl_from_evaluations",
    "wallet_flagged_pnl_from_wallet_data",
]
