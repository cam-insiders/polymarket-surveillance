"""Launch Jan-Apr 2025 monthly trade-window runs with trade-objective defaults."""

from __future__ import annotations

from pathlib import Path

from experiments.common.monthly_launcher import parse_and_run


_OBJECTIVE_ALIASES = {
    "cohens_d": "trade_cohens_d",
    "cohens_d_lcb": "trade_cohens_d_lcb",
    "weighted_cohens_d": "trade_weighted_cohens_d",
    "flagged_win_rate": "trade_flagged_win_rate",
    "flagged_mean_return": "trade_flagged_mean_return",
    "mean_return_diff": "trade_mean_return_diff",
    "t_stat": "trade_t_stat",
}


def normalize_trade_objective(name: str) -> str:
    key = name.strip()
    if not key:
        raise ValueError("objective must be non-empty")
    return _OBJECTIVE_ALIASES.get(key, key)


def _default_output_root(args) -> Path:
    return args.output_root or Path(
        "experiments/results/timeframe_trade_window_train_backtest/"
        f"monthly_trade_{args.objective}_2025"
    )


def main() -> None:
    parse_and_run(
        description=(
            "Run Jan-Apr 2025 trade-window experiments in parallel using a "
            "trade-level optimization objective."
        ),
        default_objective="cohens_d",
        default_output_root=None,
        output_help=(
            "Parent directory; each month writes to <root>/<YYYY-MM>/. "
            "Default: experiments/results/.../monthly_trade_<objective>_2025"
        ),
        objective_normalizer=normalize_trade_objective,
        output_root_factory=_default_output_root,
    )


if __name__ == "__main__":
    main()
