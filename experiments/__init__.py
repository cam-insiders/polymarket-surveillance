"""
Experiment scripts for dissertation evaluation.

Each script uses backtesting.evaluation.evaluate_config() as the core
evaluation function and produces structured results for dissertation tables.

Usage:
    python -m experiments.sweep_min_usd path/to/config.json --start-date YYYY-MM-DD --end-date YYYY-MM-DD
    python -m experiments.compare_sota backtest_results/best_config_xxx.json
    python -m experiments.compare_sota_timeframe --train-start ... --train-end ... --test-start ... --test-end ...
    python -m experiments.curated_reported_insider_recall config.json --compare-sota
    python -m experiments.curated_sota_common config.json --train-start ... --train-end ...
    python -m experiments.clustering_effectiveness_timeframe --train-start ... --test-end ...
    python -m experiments.ablation_detectors_timeframe config.json --test-start ... --test-end ...
    python -m experiments.sweep_flag_rate path/to/config.json --start-date YYYY-MM-DD --end-date YYYY-MM-DD
    python -m experiments.timeframe_trade_window_train_backtest --train-start ... --train-end ... --test-start ... --test-end ...
"""

from backtesting.logging_utils import set_experiment_backtest_log_quiet_mode


# Default experiment behavior: keep high-level progress logs, suppress noisy
# per-backtest/per-cluster INFO logs unless a script explicitly re-enables them.
set_experiment_backtest_log_quiet_mode(True)
