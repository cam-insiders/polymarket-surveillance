"""
Utilities for controlling noisy log output during experiment runs.
"""

from __future__ import annotations

import logging
import os


QUIET_EXPERIMENT_BACKTEST_LOGS_ENV_VAR = "POLYMARKET_QUIET_EXPERIMENT_BACKTEST_LOGS"

_NOISY_EXPERIMENT_LOGGERS = (
    "backtesting.backtest_runner",
    "backtesting.cached_evaluator",
    "backtesting.wallet_evaluator",
    "clustering.cluster_computer",
    "clustering.ownership_analyser",
    "clustering.polygonscan_client",
    "clustering.usdc_transfer_provider",
)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def experiment_backtest_logs_quiet() -> bool:
    """
    Return whether low-level backtest/clustering INFO logs should be suppressed.
    """
    return os.environ.get(QUIET_EXPERIMENT_BACKTEST_LOGS_ENV_VAR, "").strip().lower() in _TRUE_VALUES


def set_experiment_backtest_log_quiet_mode(enabled: bool = True) -> None:
    """
    Enable or disable suppression for noisy low-level experiment logs.

    This uses an environment variable so the setting propagates to worker
    processes spawned by experiment scripts.
    """
    if enabled:
        os.environ[QUIET_EXPERIMENT_BACKTEST_LOGS_ENV_VAR] = "1"
    else:
        os.environ.pop(QUIET_EXPERIMENT_BACKTEST_LOGS_ENV_VAR, None)

    logger_level = logging.WARNING if enabled else logging.NOTSET
    for logger_name in _NOISY_EXPERIMENT_LOGGERS:
        logging.getLogger(logger_name).setLevel(logger_level)
