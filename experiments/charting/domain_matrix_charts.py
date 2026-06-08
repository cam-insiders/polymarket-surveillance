#!/usr/bin/env python3
"""
Generate **grouped bar charts** from a *timeframe domain matrix* run
(``experiments/timeframe_domain_matrix.py``).

What this visualizes
----------------------
Each run trains three optimizers (on **all** markets, **insider-plausible** only, and
**non-insider-plausible** only), then evaluates each saved config on each of three **test /
evaluation** slices. That yields nine numeric results per metric (3×3).

For **each metric**, we draw one chart where:

- The **horizontal axis** is the **evaluation (test) domain**: all markets, insider-plausible
  only, or non-insider-plausible only.
- At each evaluation domain we plot **three bars**: the value when the model was trained
  on all / insider / non-insider respectively.

So every chart shows **all nine** train×eval cells at once. The natural comparison is
**within a fixed evaluation slice**: which training pool produced the best bar among the
three.

Metrics covered (separate subplot per metric)
---------------------------------------------
Wallet classification: F1, F0.5, precision, recall. **Flagged-wallet returns:** median
return and average net PnL (from the copytrade summary’s flagged-wallet bucket).
**Copy-trade simulation:** portfolio ROI and median per-trade return.

Inputs
------
- ``domain_matrix_summary.json`` (preferred), or
- ``domain_matrix_metrics.csv`` in the same run folder.

Outputs (written under ``<run_dir>/charts/`` by default)
-------------------------------------------------------
- ``domain_matrix_barcharts.png`` — multi-panel figure with the metrics above.

Usage
-----
List recent runs under the default results root::

    python -m experiments.charting.domain_matrix_charts --list

Chart a run by folder name (under ``experiments/results/timeframe_domain_matrix``)::

    python -m experiments.charting.domain_matrix_charts --run domain_matrix_20260414_180208

Point directly at a summary file::

    python -m experiments.charting.domain_matrix_charts \\
        --summary path/to/domain_matrix_summary.json

Interactive pick from the list (stdin)::

    python -m experiments.charting.domain_matrix_charts --interactive

Dependencies: ``pandas`` and ``matplotlib``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Canonical axis order (must match ``TRAIN_TEST_DOMAINS`` in ``timeframe_domain_matrix.py``).
DOMAIN_KEYS: Tuple[str, ...] = ("all", "insider", "non_insider")

DOMAIN_LABELS: Dict[str, str] = {
    "all": "All markets",
    "insider": "Insider-plausible",
    "non_insider": "Non-insider",
}

# One color per *training* domain — reused in every subplot so the legend is readable.
TRAIN_BAR_COLORS: Dict[str, str] = {
    "all": "#3274A1",
    "insider": "#E1812C",
    "non_insider": "#3A923A",
}


def _discover_runs(results_root: Path) -> List[Path]:
    """
    Return ``domain_matrix_*`` directories that contain ``domain_matrix_summary.json``,
    newest first (by filesystem mtime).
    """
    if not results_root.is_dir():
        return []
    candidates: List[Tuple[float, Path]] = []
    for p in results_root.glob("domain_matrix_*"):
        if not p.is_dir():
            continue
        if (p / "domain_matrix_summary.json").is_file():
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((mtime, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def _load_summary_or_csv(run_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Load metadata and a tidy dataframe of matrix rows.

    Prefer JSON summary (includes train/test windows and optimizer metadata); fall back
    to CSV if present.
    """
    summary_path = run_dir / "domain_matrix_summary.json"
    csv_path = run_dir / "domain_matrix_metrics.csv"

    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            meta = json.load(f)
        rows = meta.get("rows", [])
        df = _rows_to_dataframe(rows)
        return meta, df

    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        meta = {
            "train_start_date": None,
            "train_end_date": None,
            "test_start_date": None,
            "test_end_date": None,
            "optimizer_mode": None,
            "objective_metric": None,
            "prediction_mode": None,
            "source": str(csv_path),
        }
        return meta, df

    raise FileNotFoundError(
        f"No domain_matrix_summary.json or domain_matrix_metrics.csv under {run_dir}"
    )


def _rows_to_dataframe(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Drop errored backtest rows and build a ``DataFrame``."""
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise ValueError("No successful matrix rows to plot (all cells failed?).")
    return pd.DataFrame(ok)


def _pivot(
    df: pd.DataFrame,
    metric: str,
    index_order: Sequence[str] = DOMAIN_KEYS,
    column_order: Sequence[str] = DOMAIN_KEYS,
) -> pd.DataFrame:
    """
    Pivot ``metric`` with train_domain as rows and test_domain as columns.

    Missing combinations (should not happen for a complete run) appear as NaN.
    """
    sub = df[["train_domain", "test_domain", metric]].copy()
    pt = sub.pivot(index="train_domain", columns="test_domain", values=metric)
    # Reindex to enforce order and stable labels
    pt = pt.reindex(index=list(index_order), columns=list(column_order))
    return pt


def _fig_title(meta: Dict[str, Any], run_dir: Path) -> str:
    """One-line description of the run for the figure suptitle."""
    parts: List[str] = [run_dir.name]
    ts = meta.get("test_start_date") or meta.get("train_start_date")
    te = meta.get("test_end_date") or meta.get("train_end_date")
    if ts and te:
        parts.append(f"eval {ts} → {te}")
    obj = meta.get("objective_metric")
    if obj:
        parts.append(f"objective={obj}")
    opt = meta.get("optimizer_mode")
    if opt:
        parts.append(str(opt))
    return "  |  ".join(parts)


def _grouped_bars_for_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    title: str,
    *,
    ylabel: str,
    y_axis_mode: str = "auto",
) -> None:
    """
    Grouped bars: x = evaluation (test) domain; three bars per group = training domains.

    ``y_axis_mode``: ``"zero_baseline"`` draws a horizontal line at 0 (for ROI / returns
    that may be negative). ``"auto"`` uses matplotlib's default scaling.
    """
    if column not in df.columns:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(missing column {column!r})")
        return

    pt = _pivot(df, column)
    x = np.arange(len(DOMAIN_KEYS), dtype=float)
    n_train = len(DOMAIN_KEYS)
    bar_width = 0.22
    offsets = (np.arange(n_train) - (n_train - 1) / 2.0) * bar_width

    for bi, train in enumerate(DOMAIN_KEYS):
        heights: List[float] = []
        for td in DOMAIN_KEYS:
            v = pt.loc[train, td]
            heights.append(float(v) if pd.notna(v) else float("nan"))
        ax.bar(
            x + offsets[bi],
            heights,
            bar_width,
            label=f"Train: {DOMAIN_LABELS[train]}",
            color=TRAIN_BAR_COLORS[train],
            edgecolor="0.35",
            linewidth=0.45,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Eval:\n{DOMAIN_LABELS[t]}" for t in DOMAIN_KEYS], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    if y_axis_mode == "zero_baseline":
        ax.axhline(0.0, color="0.45", linewidth=0.7, zorder=1)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)


def _format_usd_axis(ax: plt.Axes) -> None:
    """Compact tick labels for large dollar amounts on the vertical axis."""

    def _fmt(v: float, _pos: int) -> str:
        if not np.isfinite(v):
            return ""
        av = abs(v)
        if av >= 1e9:
            return f"{v/1e9:.1f}B"
        if av >= 1e6:
            return f"{v/1e6:.1f}M"
        if av >= 1e3:
            return f"{v/1e3:.1f}k"
        return f"{v:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))


def plot_domain_matrix_charts(
    meta: Dict[str, Any],
    df: pd.DataFrame,
    run_dir: Path,
    out_dir: Path,
    dpi: int = 140,
) -> Path:
    """
    Write ``domain_matrix_barcharts.png`` under ``out_dir``.

    Returns the path to the PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "domain_matrix_barcharts.png"

    # (column, subplot title, y-axis label, y_axis_mode)
    panels: List[Tuple[str, str, str, str]] = [
        ("wallet_f1", "Wallet F1", "F1 (0–1)", "auto"),
        ("wallet_f0_5", "Wallet F0.5", "F0.5 (0–1)", "auto"),
        ("wallet_precision", "Wallet precision", "Precision (0–1)", "auto"),
        ("wallet_recall", "Wallet recall", "Recall (0–1)", "auto"),
        ("flagged_mean_return", "Flagged wallets: mean return", "Mean return", "zero_baseline"),
        ("flagged_avg_net_pnl", "Flagged wallets: avg net PnL", "Avg net PnL (USDC)", "zero_baseline"),
        ("copytrade_portfolio_roi", "Copy-trade: portfolio ROI", "Portfolio ROI", "zero_baseline"),
        ("copytrade_median_trade_return", "Copy-trade: median trade return", "Median trade return", "zero_baseline"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12.0, 16.0))
    fig.suptitle(
        _fig_title(meta, run_dir) + "\n(x = evaluation domain; bars = model trained on …)",
        fontsize=11,
        y=0.995,
    )

    for ax, (col, title, ylab, ymode) in zip(np.ravel(axes), panels):
        _grouped_bars_for_metric(ax, df, col, title, ylabel=ylab, y_axis_mode=ymode)
        if col == "flagged_avg_net_pnl":
            _format_usd_axis(ax)
        elif col == "copytrade_portfolio_roi":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.3f}"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def resolve_run_directory(
    results_root: Path,
    run: Optional[str],
    summary_path: Optional[Path],
) -> Path:
    """
    Resolve which directory holds the matrix artifacts.

    ``run`` may be a folder name, a path to the run directory, or a path to
    ``domain_matrix_summary.json`` inside the run.
    """
    if summary_path is not None:
        p = summary_path.expanduser().resolve()
        if p.is_file():
            return p.parent
        raise FileNotFoundError(f"Summary file not found: {p}")

    if not run:
        raise ValueError("Either --run or --summary is required (unless using --list).")

    rp = Path(run).expanduser()
    if rp.is_file() and rp.name == "domain_matrix_summary.json":
        return rp.parent.resolve()

    # Absolute or relative path to run folder
    if rp.is_dir():
        return rp.resolve()

    # Short name under results_root
    candidate = (results_root / run).resolve()
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve run {run!r}. Try --list or pass a path to the run folder "
        f"or to domain_matrix_summary.json."
    )


def _print_run_list(runs: Sequence[Path], results_root: Path) -> None:
    print(f"Runs under {results_root} (newest first):\n")
    for i, p in enumerate(runs, start=1):
        try:
            rel = p.relative_to(results_root)
        except ValueError:
            rel = p
        print(f"  [{i}]  {rel}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot grouped bar charts from a timeframe domain matrix experiment run."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="experiments/results/timeframe_domain_matrix",
        help="Directory containing domain_matrix_<timestamp> run folders.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run folder name (e.g. domain_matrix_20260414_180208), path to run dir, "
        "or path to domain_matrix_summary.json.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Explicit path to domain_matrix_summary.json (overrides --run).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNG output (default: <run_dir>/charts).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered runs and exit.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After --list, read a number from stdin to select a run.",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI for PNG output.")

    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path.cwd()
    results_root = (repo_root / args.results_root).resolve()
    summary_path = Path(args.summary).resolve() if args.summary else None

    runs = _discover_runs(results_root)
    run_dir: Optional[Path] = None

    if args.list:
        if not runs:
            print(f"No runs found under {results_root} (expected domain_matrix_*/domain_matrix_summary.json).")
            return 1
        _print_run_list(runs, results_root)
        if not args.interactive:
            return 0
        try:
            choice = input("\nEnter index (1-based) to chart, or Enter to skip: ").strip()
        except EOFError:
            return 0
        if not choice:
            return 0
        try:
            idx = int(choice)
        except ValueError:
            print(f"Invalid index: {choice!r}", file=sys.stderr)
            return 1
        if idx < 1 or idx > len(runs):
            print(f"Invalid index: {idx}", file=sys.stderr)
            return 1
        run_dir = runs[idx - 1]
    elif args.interactive and not args.run and not summary_path:
        if not runs:
            print(f"No runs under {results_root}.", file=sys.stderr)
            return 1
        _print_run_list(runs, results_root)
        try:
            choice = input("\nEnter index (1-based): ").strip()
        except EOFError:
            return 1
        try:
            idx = int(choice)
        except ValueError:
            print(f"Invalid index: {choice!r}", file=sys.stderr)
            return 1
        if idx < 1 or idx > len(runs):
            print(f"Invalid index: {idx}", file=sys.stderr)
            return 1
        run_dir = runs[idx - 1]
    else:
        if not args.run and not summary_path:
            print(
                "Specify --run or --summary, or use --list [--interactive] to pick a run.",
                file=sys.stderr,
            )
            return 1
        try:
            run_dir = resolve_run_directory(results_root, args.run, summary_path)
        except (FileNotFoundError, ValueError) as e:
            print(e, file=sys.stderr)
            return 1

    assert run_dir is not None
    run_dir = run_dir.resolve()

    try:
        meta, df = _load_summary_or_csv(run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "charts")

    out_png = plot_domain_matrix_charts(meta, df, run_dir, out_dir, dpi=args.dpi)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
