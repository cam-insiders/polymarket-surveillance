#!/usr/bin/env python3
"""
Line chart for the *min USD trade prefilter sweep* experiment
(``experiments/sweep_min_usd.py``).

What this visualizes
--------------------
Each row is a full detector run at a different ``min_usd_amount`` (``0`` = no
prefilter). This script plots **mean return difference** (flagged vs unflagged
trades, from the pooled event study) and **mean Cohen's d** vs the threshold
on one figure (twin y-axes), so you can see whether larger minimum trade sizes
strengthen or wash out the trade-level signal.

Input
-----
A comparison CSV written by ``sweep_min_usd`` (filename pattern
``sweep_min_usd_comparison_*.csv``), with columns including ``min_usd_amount``,
``mean_return_diff``, and ``mean_cohens_d``.

Output
------
One PNG per invocation (default name derived from the input CSV stem).

Usage
-----
::

    python -m experiments.charting.sweep_min_usd_charts --list
    python -m experiments.charting.sweep_min_usd_charts \\
        --csv experiments/results/sweep_min_usd/sweep_min_usd_comparison_20260117_120000.csv

    python -m experiments.charting.sweep_min_usd_charts --interactive

Dependencies: ``pandas`` and ``matplotlib``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOR_MEAN_DIFF = "#3274A1"
COLOR_COHENS_D = "#E1812C"

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "min_usd_amount",
    "mean_return_diff",
    "mean_cohens_d",
)


def _discover_csvs(results_root: Path) -> List[Path]:
    """``sweep_min_usd_comparison_*.csv`` under ``results_root``, newest first."""
    if not results_root.is_dir():
        return []
    candidates: List[Tuple[float, Path]] = []
    for p in results_root.glob("sweep_min_usd_comparison_*.csv"):
        if not p.is_file():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def load_sweep_min_usd_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate a sweep comparison CSV."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {missing}. "
            f"Expected at least: {list(REQUIRED_COLUMNS)}."
        )
    return df.sort_values("min_usd_amount", kind="mergesort").reset_index(drop=True)


def _suptitle_line(csv_path: Path) -> str:
    return csv_path.stem.replace("_", " ")


def _format_min_usd_xtick(v: float) -> str:
    if v == 0:
        return "0 (no filter)"
    v = float(v)
    if v >= 1000:
        return f"{v:g}"
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def plot_sweep_min_usd_chart(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """
    Write one figure: ``mean_return_diff`` and ``mean_cohens_d`` vs ``min_usd_amount``.

    Returns the path to the PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = df["min_usd_amount"].to_numpy(dtype=float)
    mean_diff = df["mean_return_diff"].to_numpy(dtype=float)
    cohens = df["mean_cohens_d"].to_numpy(dtype=float)

    fig, ax_left = plt.subplots(figsize=(10.0, 5.5))
    ax_left.plot(
        x,
        mean_diff,
        label="Mean return diff (flagged vs unflagged)",
        color=COLOR_MEAN_DIFF,
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        zorder=3,
    )
    ax_left.axhline(0.0, color="0.45", linewidth=0.9, linestyle="--", zorder=1)
    ax_left.set_xlabel("min_usd_amount (trade prefilter)", fontsize=10)
    ax_left.set_ylabel("Mean return difference", fontsize=10, color=COLOR_MEAN_DIFF)
    ax_left.tick_params(axis="y", labelcolor=COLOR_MEAN_DIFF)
    ax_left.grid(axis="y", alpha=0.35, zorder=0)
    ax_left.grid(axis="x", alpha=0.2, zorder=0)

    ax_right = ax_left.twinx()
    ax_right.plot(
        x,
        cohens,
        label="Mean Cohen's d",
        color=COLOR_COHENS_D,
        marker="s",
        markersize=4.0,
        linewidth=1.5,
        zorder=3,
    )
    ax_right.axhline(0.0, color="0.65", linewidth=0.7, linestyle=":", zorder=1)
    ax_right.set_ylabel("Mean Cohen's d", fontsize=10, color=COLOR_COHENS_D)
    ax_right.tick_params(axis="y", labelcolor=COLOR_COHENS_D)

    tick_labels = [_format_min_usd_xtick(float(v)) for v in x]
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)

    ax_left.set_title(
        "Trade-level signal vs min trade size (USDC)\n"
        "(each point is a full evaluation at that prefilter)",
        fontsize=11,
    )

    lines_l, lab_l = ax_left.get_legend_handles_labels()
    lines_r, lab_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, lab_l + lab_r, loc="best", frameon=True, fontsize=8)

    fig.suptitle(_suptitle_line(csv_path), fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def resolve_sweep_csv(
    results_root: Path,
    csv_arg: Optional[str],
    csv_path_override: Optional[Path],
) -> Path:
    if csv_path_override is not None:
        p = csv_path_override.expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"CSV not found: {p}")

    if not csv_arg:
        raise ValueError("Either --csv or a discovered file is required (unless using --list).")

    rp = Path(csv_arg).expanduser()
    if rp.is_file():
        return rp.resolve()

    candidate = (results_root / csv_arg).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve CSV {csv_arg!r}. Try --list or pass a path to sweep_min_usd_comparison_*.csv."
    )


def _print_csv_list(paths: Sequence[Path], results_root: Path) -> None:
    print(f"Min-USD sweep CSVs under {results_root} (newest first):\n")
    for i, p in enumerate(paths, start=1):
        try:
            rel = p.relative_to(results_root)
        except ValueError:
            rel = p
        print(f"  [{i}]  {rel}")


def default_output_path(csv_path: Path, out_dir: Optional[Path]) -> Path:
    name = f"{csv_path.stem}_chart.png"
    if out_dir is not None:
        return (out_dir / name).resolve()
    return (csv_path.parent / name).resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot mean return diff and Cohen's d vs min_usd_amount from a sweep_min_usd CSV."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="experiments/results/sweep_min_usd",
        help="Directory containing sweep_min_usd_comparison_*.csv (for --list / short --csv names).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to sweep_min_usd_comparison_*.csv, or basename under --results-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the PNG (default: same folder as the input CSV).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output PNG path (overrides --output-dir).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered comparison CSVs under --results-root and exit.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After --list, read a number from stdin to select a CSV to chart.",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI for PNG output.")

    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path.cwd()
    results_root = (repo_root / args.results_root).resolve()

    csvs = _discover_csvs(results_root)
    csv_path: Optional[Path] = None

    if args.list:
        if not csvs:
            print(
                f"No CSVs found under {results_root} (expected sweep_min_usd_comparison_*.csv).",
                file=sys.stderr,
            )
            return 1
        _print_csv_list(csvs, results_root)
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
        if idx < 1 or idx > len(csvs):
            print(f"Invalid index: {idx}", file=sys.stderr)
            return 1
        csv_path = csvs[idx - 1]
    elif args.interactive and not args.csv:
        if not csvs:
            print(f"No CSVs under {results_root}.", file=sys.stderr)
            return 1
        _print_csv_list(csvs, results_root)
        try:
            choice = input("\nEnter index (1-based): ").strip()
        except EOFError:
            return 1
        try:
            idx = int(choice)
        except ValueError:
            print(f"Invalid index: {choice!r}", file=sys.stderr)
            return 1
        if idx < 1 or idx > len(csvs):
            print(f"Invalid index: {idx}", file=sys.stderr)
            return 1
        csv_path = csvs[idx - 1]
    else:
        explicit: Optional[Path] = None
        if args.csv:
            cand = Path(args.csv).expanduser()
            if cand.is_file():
                explicit = cand.resolve()
        try:
            csv_path = resolve_sweep_csv(results_root, args.csv, explicit)
        except (FileNotFoundError, ValueError) as e:
            print(e, file=sys.stderr)
            return 1

    assert csv_path is not None

    try:
        df = load_sweep_min_usd_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_dir = Path(args.output_dir).resolve() if args.output_dir else None
        out_path = default_output_path(csv_path, out_dir)

    try:
        written = plot_sweep_min_usd_chart(df, csv_path=csv_path, out_path=out_path, dpi=args.dpi)
    except Exception as e:
        print(f"Plot failed: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
