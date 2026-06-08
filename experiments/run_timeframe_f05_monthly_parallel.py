"""Launch Jan-Apr 2025 monthly trade-window runs with an F0.5 default."""

from __future__ import annotations

from pathlib import Path

from experiments.common.monthly_launcher import parse_and_run


DEFAULT_OUTPUT_ROOT = Path(
    "experiments/results/timeframe_trade_window_train_backtest/monthly_f05_2025"
)


def main() -> None:
    parse_and_run(
        description="Run Jan-Apr 2025 trade-window experiments in parallel.",
        default_objective="f0_5",
        default_output_root=DEFAULT_OUTPUT_ROOT,
        output_help="Parent directory; each month writes to <root>/<YYYY-MM>/.",
    )


if __name__ == "__main__":
    main()
