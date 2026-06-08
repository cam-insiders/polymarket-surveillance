#!/usr/bin/env python3
"""
Bar charts for the *timeframe optimize + category eval* experiment
(``experiments/timeframe_optimize_category_eval.py``).

What this visualizes
--------------------
The experiment trains an optimizer on **all train-window markets by default** (no
insider-plausible filter). With ``--train-both-domains``, it also trains on
insider-plausible-only markets; each saved config is evaluated on the same **eval
slices**: all markets, insider-plausible only, and each classifier category
(``category_<CAT>``) separately.

When a run contains multiple ``train_domain`` rows, this script writes four PNGs per
train pool (same stem suffix as before); otherwise four files without a train-domain stem.

Slices on the x-axis:

- Always **``all``** and **``insider_plausible``** when present, then
- **Every** ``category_*`` slice in the data, ordered by **``resolved_markets``**
  descending (ties: alphabetical). Use **``--top-categories N``** (``N`` > 0) only if
  you want to cap how many category bars appear.

Metrics are split across four figures (two related metrics per figure); see Outputs below.

Outputs (under ``<run_dir>/charts/`` by default)
----------------------------------------------
Four figures (each 1×2 panels for separation):

- ``category_eval_wallet_f1_f05.png`` — F1 and F0.5
- ``category_eval_precision_recall.png`` — precision and recall
- ``category_eval_flagged_mean_return_avg_pnl.png`` — mean return and avg net PnL
- ``category_eval_copytrade.png`` — copy-trade portfolio ROI and median trade return

With multiple train pools, each filename gets ``_<train_domain>.png`` before ``.png``.

Usage
-----
::

    python -m experiments.charting.category_eval_charts --list
    python -m experiments.charting.category_eval_charts --run category_eval_20260414_213516
    python -m experiments.charting.category_eval_charts --summary path/to/category_eval_summary.json

    # Optional: show only the N busiest categories (default is all categories)
    python -m experiments.charting.category_eval_charts --run category_eval_... --top-categories 15

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
from matplotlib.patches import Patch

# Baseline slices (fixed order when present).
BASELINE_SLICES: Tuple[str, ...] = ("all", "insider_plausible")

BASELINE_COLORS: Dict[str, str] = {
    "all": "#3274A1",
    "insider_plausible": "#E1812C",
}
CATEGORY_BAR_COLOR = "#7A7A7A"

# (output stem suffix, subtitle line for this figure, two metric panels)
_CATEGORY_EVAL_FIGURE_GROUPS: List[Tuple[str, str, List[Tuple[str, str, str, str]]]] = [
    (
        "wallet_f1_f05",
        "Wallet F1 and F0.5",
        [
            ("wallet_f1", "Wallet F1", "F1 (0–1)", "auto"),
            ("wallet_f0_5", "Wallet F0.5", "F0.5 (0–1)", "auto"),
        ],
    ),
    (
        "precision_recall",
        "Wallet precision and recall",
        [
            ("wallet_precision", "Wallet precision", "Precision (0–1)", "auto"),
            ("wallet_recall", "Wallet recall", "Recall (0–1)", "auto"),
        ],
    ),
    (
        "flagged_mean_return_avg_pnl",
        "Flagged wallets: mean return and average net PnL",
        [
            ("flagged_mean_return", "Flagged wallets: mean return", "Mean return", "zero_baseline"),
            ("flagged_avg_net_pnl", "Flagged wallets: avg net PnL", "Avg net PnL (USDC)", "zero_baseline"),
        ],
    ),
    (
        "copytrade",
        "Copy-trade: portfolio ROI and median trade return",
        [
            ("copytrade_portfolio_roi", "Copy-trade: portfolio ROI", "Portfolio ROI", "zero_baseline"),
            ("copytrade_median_trade_return", "Copy-trade: median trade return", "Median trade return", "zero_baseline"),
        ],
    ),
]


def _discover_runs(results_root: Path) -> List[Path]:
    """``category_eval_*`` dirs containing ``category_eval_summary.json``, newest first."""
    if not results_root.is_dir():
        return []
    candidates: List[Tuple[float, Path]] = []
    for p in results_root.glob("category_eval_*"):
        if not p.is_dir():
            continue
        if (p / "category_eval_summary.json").is_file():
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((mtime, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def _load_summary_or_csv(run_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load ``category_eval_summary.json`` or fall back to ``category_eval_metrics.csv``."""
    summary_path = run_dir / "category_eval_summary.json"
    csv_path = run_dir / "category_eval_metrics.csv"

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
            "train_mode": None,
            "optimizer_mode": None,
            "objective_metric": None,
            "prediction_mode": None,
            "source": str(csv_path),
        }
        return meta, df

    raise FileNotFoundError(
        f"No category_eval_summary.json or category_eval_metrics.csv under {run_dir}"
    )


def _rows_to_dataframe(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        raise ValueError("No successful eval rows to plot (all slices failed?).")
    return pd.DataFrame(ok)


def _pretty_slice_label(test_domain: str) -> str:
    """Short tick label for the x-axis."""
    if test_domain.startswith("category_"):
        return test_domain[len("category_") :]
    if test_domain == "insider_plausible":
        return "insider"
    return test_domain


def _slice_bar_color(test_domain: str) -> str:
    return BASELINE_COLORS.get(test_domain, CATEGORY_BAR_COLOR)


def _select_slice_order(
    df: pd.DataFrame,
    *,
    max_categories: int,
) -> List[str]:
    """
    Build eval-slice order: baselines (if present), then category slices by market count.

    Category rows are those with ``test_domain`` starting with ``category_``. Ties on
    ``resolved_markets`` fall back to alphabetical slice name for stability.

    ``max_categories`` <= 0 means include every category slice present in ``df``;
    if > 0, keep only that many category bars after baselines (still in market-count order).
    """
    have = set(df["test_domain"].astype(str).tolist())
    ordered: List[str] = []
    for key in BASELINE_SLICES:
        if key in have:
            ordered.append(key)

    cat_rows = df[df["test_domain"].astype(str).str.startswith("category_")].copy()
    if cat_rows.empty:
        return ordered

    if "resolved_markets" not in cat_rows.columns:
        cat_sorted = cat_rows.sort_values("test_domain")
    else:
        cat_sorted = cat_rows.sort_values(
            ["resolved_markets", "test_domain"],
            ascending=[False, True],
        )

    names = cat_sorted["test_domain"].astype(str).tolist()
    # De-duplicate while preserving sort
    seen = set()
    uniq: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    if max_categories > 0:
        uniq = uniq[:max_categories]
    ordered.extend(uniq)
    return ordered


def _category_fig_title(meta: Dict[str, Any], run_dir: Path) -> str:
    """Suptitle line: run id, windows, objective, optimizer, train pool."""
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
    tm = meta.get("train_mode")
    if tm:
        parts.append(f"train={tm}")
    return "  |  ".join(parts)


def _format_usd_axis(ax: plt.Axes) -> None:
    """Compact tick labels for large dollar amounts on the y-axis."""

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


def _vertical_bars_for_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    slice_order: Sequence[str],
    column: str,
    title: str,
    *,
    ylabel: str,
    y_axis_mode: str = "auto",
) -> None:
    """One bar per eval slice (same model), colored by baseline vs category slice."""
    if column not in df.columns:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(missing column {column!r})")
        return

    lookup = df.set_index("test_domain")[column]
    x = np.arange(len(slice_order), dtype=float)
    heights: List[float] = []
    colors: List[str] = []
    for sl in slice_order:
        v = lookup.get(sl)
        heights.append(float(v) if pd.notna(v) else float("nan"))
        colors.append(_slice_bar_color(sl))

    ax.bar(
        x,
        heights,
        color=colors,
        edgecolor="0.35",
        linewidth=0.45,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_slice_label(s) for s in slice_order], rotation=65, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    if y_axis_mode == "zero_baseline":
        ax.axhline(0.0, color="0.45", linewidth=0.7, zorder=1)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)


def _category_bars_caption(max_categories: int) -> str:
    if max_categories <= 0:
        return (
            "all category slices ordered by resolved market count (highest first; "
            "ties alphabetical)"
        )
    return f"top {max_categories} categories by resolved market count after baselines"


def _write_category_eval_two_panel_png(
    meta: Dict[str, Any],
    df: pd.DataFrame,
    run_dir: Path,
    out_path: Path,
    slice_order: Sequence[str],
    *,
    fig_w: float,
    group_subtitle: str,
    panels: Sequence[Tuple[str, str, str, str]],
    max_categories: int,
    dpi: int,
) -> Path:
    """Write one 1×2 bar-chart PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.2))
    base = _category_fig_title(meta, run_dir)
    fig.suptitle(
        f"{base}\n{group_subtitle}\n"
        f"(one bar per eval slice; {_category_bars_caption(max_categories)})",
        fontsize=10,
        y=1.02,
    )

    for ax, (col, title, ylab, ymode) in zip(np.ravel(axes), panels):
        _vertical_bars_for_metric(
            ax,
            df,
            slice_order,
            col,
            title,
            ylabel=ylab,
            y_axis_mode=ymode,
        )
        if col == "flagged_avg_net_pnl":
            _format_usd_axis(ax)
        elif col == "copytrade_portfolio_roi":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.3f}"))

    legend_handles = [
        Patch(facecolor=BASELINE_COLORS["all"], edgecolor="0.35", label="Eval: all markets"),
        Patch(
            facecolor=BASELINE_COLORS["insider_plausible"],
            edgecolor="0.35",
            label="Eval: insider-plausible",
        ),
        Patch(facecolor=CATEGORY_BAR_COLOR, edgecolor="0.35", label="Eval: category slice"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.88])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_category_eval_chart_pngs(
    meta: Dict[str, Any],
    df: pd.DataFrame,
    run_dir: Path,
    out_dir: Path,
    *,
    max_categories: int,
    dpi: int,
    filename_suffix: str,
) -> List[Path]:
    """Write four 1×2 bar-chart PNGs; ``df`` must have unique ``test_domain`` rows."""
    slice_order = _select_slice_order(df, max_categories=max_categories)
    if not slice_order:
        raise ValueError("No eval slices to plot.")

    n = len(slice_order)
    fig_w = min(28.0, max(12.0, 0.42 * n + 6.0))

    paths: List[Path] = []
    for stem, group_subtitle, panels in _CATEGORY_EVAL_FIGURE_GROUPS:
        name = f"category_eval_{stem}{filename_suffix}.png"
        paths.append(
            _write_category_eval_two_panel_png(
                meta,
                df,
                run_dir,
                out_dir / name,
                slice_order,
                fig_w=fig_w,
                group_subtitle=group_subtitle,
                panels=panels,
                max_categories=max_categories,
                dpi=dpi,
            )
        )
    return paths


def plot_category_eval_charts(
    meta: Dict[str, Any],
    df: pd.DataFrame,
    run_dir: Path,
    out_dir: Path,
    *,
    max_categories: int = 0,
    dpi: int = 140,
) -> List[Path]:
    """
    Write four ``category_eval_*.png`` files under ``out_dir`` (see module docstring),
    or four files per ``train_domain`` when the dataframe mixes multiple training pools.

    Returns paths to each PNG written (four per train pool).
    """
    if "train_domain" in df.columns and df["train_domain"].nunique() > 1:
        paths: List[Path] = []
        for tm in sorted(df["train_domain"].dropna().astype(str).unique()):
            sub = df[df["train_domain"].astype(str) == tm].copy()
            meta_sub = dict(meta)
            meta_sub["train_mode"] = tm
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tm)
            paths.extend(
                _write_category_eval_chart_pngs(
                    meta_sub,
                    sub,
                    run_dir,
                    out_dir,
                    max_categories=max_categories,
                    dpi=dpi,
                    filename_suffix=f"_{safe}",
                )
            )
        return paths

    return _write_category_eval_chart_pngs(
        meta,
        df,
        run_dir,
        out_dir,
        max_categories=max_categories,
        dpi=dpi,
        filename_suffix="",
    )


def resolve_run_directory(
    results_root: Path,
    run: Optional[str],
    summary_path: Optional[Path],
) -> Path:
    """Resolve the run directory holding ``category_eval_summary.json``."""
    if summary_path is not None:
        p = summary_path.expanduser().resolve()
        if p.is_file():
            return p.parent
        raise FileNotFoundError(f"Summary file not found: {p}")

    if not run:
        raise ValueError("Either --run or --summary is required (unless using --list).")

    rp = Path(run).expanduser()
    if rp.is_file() and rp.name == "category_eval_summary.json":
        return rp.parent.resolve()

    if rp.is_dir():
        return rp.resolve()

    candidate = (results_root / run).resolve()
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve run {run!r}. Try --list or pass a path to the run folder "
        f"or to category_eval_summary.json."
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
        description="Plot bar charts from a timeframe optimize + category eval run."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="experiments/results/timeframe_optimize_category_eval",
        help="Directory containing category_eval_<timestamp> run folders.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run folder name, path to run dir, or path to category_eval_summary.json.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Explicit path to category_eval_summary.json (overrides --run).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNG output (default: <run_dir>/charts).",
    )
    parser.add_argument(
        "--top-categories",
        type=int,
        default=0,
        metavar="N",
        help="Cap category_* bars after baselines: 0 (default) = all categories, ordered by "
        "resolved_markets; N>0 = at most N (still in market-count order).",
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
            print(
                f"No runs found under {results_root} (expected category_eval_*/category_eval_summary.json).",
            )
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

    out_pngs = plot_category_eval_charts(
        meta,
        df,
        run_dir,
        out_dir,
        max_categories=args.top_categories,
        dpi=args.dpi,
    )
    for p in out_pngs:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
