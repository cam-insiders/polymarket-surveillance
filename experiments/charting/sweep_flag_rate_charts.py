#!/usr/bin/env python3
"""
Charts for the *flag rate threshold sweep* experiment
(``experiments/sweep_flag_rate.py``).

What this visualizes
--------------------
The sweep evaluates detectors **once**, then recomputes wallet metrics at many
``flag_rate_threshold`` values. Precision/recall/F1 alone are often **flat and
unreadable** on a shared 0–1 axis when the interesting regime change is a
narrow cliff (e.g.~0.55) while **flagged wallet economics** and **flag volume**
move a lot.

This module writes **several PNGs** per run:

1. **Dashboard** (2×2): F1 & F0.5; precision & recall; **flagged_avg_return** and
   **flagged_avg_pnl** (twin y-axes); **n_flagged**.
2. **Precision–recall path**: ``precision`` vs ``recall`` with points and a
   path ordered by threshold; **color = flag_rate_threshold** (shows where the
   cliff sits in PR space).
3. **F1 vs volume**: ``n_flagged`` vs ``F1`` (log-scaled x when span is large),
   color = threshold — useful for seeing the precision–volume tradeoff.

Input
-----
A CSV written by ``sweep_flag_rate`` (``flag_rate_sweep_*.csv``), including
wallet counts: ``n_flagged``, ``tp`` (true positives), etc.

Output
------
Multiple PNGs per invocation (see ``plot_all_sweep_flag_rate_charts``).

Usage
-----
::

    python -m experiments.charting.sweep_flag_rate_charts --list
    python -m experiments.charting.sweep_flag_rate_charts \\
        --csv experiments/results/sweep_flag_rate/flag_rate_sweep_20260117_120000.csv

    python -m experiments.charting.sweep_flag_rate_charts --interactive

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
import numpy as np
import pandas as pd

# Match ``domain_matrix_charts`` / ``category_eval_charts`` palette.
COLOR_F1 = "#3274A1"
COLOR_F05 = "#3A923A"
COLOR_PRECISION = "#3274A1"
COLOR_RECALL = "#E1812C"
COLOR_RETURN = "#3274A1"
COLOR_PNL = "#C44E52"
COLOR_NFLAG = "#6A4C93"
COLOR_TP = "#2E8B57"

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "flag_rate_threshold",
    "precision",
    "recall",
    "f1",
    "f0_5",
    "flagged_avg_return",
    "flagged_avg_pnl",
    "n_flagged",
    "tp",
)


def _discover_csvs(results_root: Path) -> List[Path]:
    """``flag_rate_sweep_*.csv`` files directly under ``results_root``, newest first."""
    if not results_root.is_dir():
        return []
    candidates: List[Tuple[float, Path]] = []
    for p in results_root.glob("flag_rate_sweep_*.csv"):
        if not p.is_file():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def load_sweep_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate a sweep CSV."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {missing}. "
            f"Expected at least: {list(REQUIRED_COLUMNS)}."
        )
    return df.sort_values("flag_rate_threshold", kind="mergesort").reset_index(drop=True)


def _suptitle_line(csv_path: Path) -> str:
    return csv_path.stem.replace("_", " ")


def plot_sweep_flag_rate_dashboard(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """
    2×2 panel figure: classification metrics, economics (twin y), and flagged count vs TP.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    thr = df["flag_rate_threshold"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), sharex=True)
    fig.suptitle(_suptitle_line(csv_path), fontsize=11, y=1.0)

    # --- F1 / F0.5 ---
    ax = axes[0, 0]
    ax.plot(thr, df["f1"], label="F1", color=COLOR_F1, marker="o", markersize=3.5, lw=1.3)
    ax.plot(thr, df["f0_5"], label="F0.5", color=COLOR_F05, marker="o", markersize=3.5, lw=1.3)
    ax.set_ylabel("Score (0–1)", fontsize=9)
    ax.set_title("Wallet F1 and F0.5", fontsize=10)
    ax.grid(alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    top_f = max(float(np.nanmax(df["f1"])), float(np.nanmax(df["f0_5"]))) * 1.15
    ax.set_ylim(0.0, min(1.05, max(0.02, top_f)))

    # --- Precision / Recall ---
    ax = axes[0, 1]
    ax.plot(thr, df["precision"], label="Precision", color=COLOR_PRECISION, marker="o", markersize=3.5, lw=1.3)
    ax.plot(thr, df["recall"], label="Recall", color=COLOR_RECALL, marker="o", markersize=3.5, lw=1.3)
    ax.set_ylabel("Metric (0–1)", fontsize=9)
    ax.set_title("Precision and recall", fontsize=10)
    ax.grid(alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    # --- Flagged economics: return + PnL (twin y) ---
    ax_l = axes[1, 0]
    ax_r = ax_l.twinx()
    ax_l.plot(
        thr,
        df["flagged_avg_return"],
        label="Flagged avg return",
        color=COLOR_RETURN,
        marker="o",
        markersize=3.5,
        lw=1.4,
    )
    ax_r.plot(
        thr,
        df["flagged_avg_pnl"],
        label="Flagged avg net PnL (USDC)",
        color=COLOR_PNL,
        marker="s",
        markersize=3.5,
        lw=1.4,
    )
    ax_l.set_xlabel("flag_rate_threshold", fontsize=9)
    ax_l.set_ylabel("Avg return", fontsize=9, color=COLOR_RETURN)
    ax_l.tick_params(axis="y", labelcolor=COLOR_RETURN)
    ax_r.set_ylabel("Avg net PnL (USDC)", fontsize=9, color=COLOR_PNL)
    ax_r.tick_params(axis="y", labelcolor=COLOR_PNL)
    ax_l.set_title("Flagged wallets: avg return and avg net PnL", fontsize=10)
    ax_l.grid(alpha=0.3)
    lines_l, lab_l = ax_l.get_legend_handles_labels()
    lines_r, lab_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, lab_l + lab_r, loc="best", fontsize=7)

    # --- n_flagged + true positives (twin y; very different scales) ---
    ax_nf = axes[1, 1]
    ax_tp = ax_nf.twinx()
    ax_nf.plot(
        thr,
        df["n_flagged"],
        label="n_flagged",
        color=COLOR_NFLAG,
        marker="o",
        markersize=3.5,
        lw=1.4,
    )
    ax_tp.plot(
        thr,
        df["tp"],
        label="True positives (TP)",
        color=COLOR_TP,
        marker="s",
        markersize=3.5,
        lw=1.4,
    )
    ax_nf.set_xlabel("flag_rate_threshold", fontsize=9)
    ax_nf.set_ylabel("n_flagged", fontsize=9, color=COLOR_NFLAG)
    ax_nf.tick_params(axis="y", labelcolor=COLOR_NFLAG)
    ax_tp.set_ylabel("True positives (TP)", fontsize=9, color=COLOR_TP)
    ax_tp.tick_params(axis="y", labelcolor=COLOR_TP)
    ax_nf.set_title("Flagged volume vs true positives", fontsize=10)
    ax_nf.grid(alpha=0.35)
    lines_nf, lab_nf = ax_nf.get_legend_handles_labels()
    lines_tp, lab_tp = ax_tp.get_legend_handles_labels()
    ax_nf.legend(lines_nf + lines_tp, lab_nf + lab_tp, loc="best", fontsize=7)
    try:
        nmax = float(np.nanmax(df["n_flagged"]))
        if nmax > 0:
            ax_nf.set_ylim(0.0, nmax * 1.05)
    except (TypeError, ValueError):
        pass
    try:
        tpmax = float(np.nanmax(df["tp"]))
        if tpmax > 0:
            ax_tp.set_ylim(0.0, tpmax * 1.05)
    except (TypeError, ValueError):
        pass

    for ax in axes.ravel():
        ax.set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_precision_recall_threshold_path(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """
    Precision vs recall in the plane, path ordered by ``flag_rate_threshold``;
    scatter color encodes threshold (highlights cliff regions).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = df.sort_values("flag_rate_threshold", kind="mergesort")
    rec = d["recall"].to_numpy(dtype=float)
    prec = d["precision"].to_numpy(dtype=float)
    thr = d["flag_rate_threshold"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.plot(rec, prec, "-", color="0.55", lw=1.0, alpha=0.85, zorder=1)
    sc = ax.scatter(
        rec,
        prec,
        c=thr,
        cmap="viridis",
        s=45,
        edgecolors="0.25",
        linewidths=0.4,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("flag_rate_threshold", fontsize=9)

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(
        "Precision vs recall (parametric in threshold)\n"
        f"{_suptitle_line(csv_path)}",
        fontsize=10,
    )
    ax.grid(alpha=0.35)
    ax.set_xlim(0.0, min(1.05, max(0.02, float(np.nanmax(rec)) * 1.1)))
    ax.set_ylim(0.0, min(1.05, max(0.02, float(np.nanmax(prec)) * 1.1)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_f1_vs_n_flagged(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    out_path: Path,
    dpi: int = 140,
) -> Path:
    """
    Operating-point view: F1 vs number of flagged wallets, color = threshold.
    Log-scaled x when the span of ``n_flagged`` is large.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nf = df["n_flagged"].to_numpy(dtype=float)
    f1 = df["f1"].to_numpy(dtype=float)
    thr = df["flag_rate_threshold"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    sc = ax.scatter(
        nf,
        f1,
        c=thr,
        cmap="plasma",
        s=55,
        edgecolors="0.25",
        linewidths=0.45,
        zorder=3,
    )
    # Light path in threshold order (may cross in x–y space)
    order = np.argsort(thr)
    ax.plot(nf[order], f1[order], "-", color="0.5", lw=0.9, alpha=0.65, zorder=1)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("flag_rate_threshold", fontsize=9)

    ax.set_xlabel("n_flagged (log scale if wide span)", fontsize=10)
    ax.set_ylabel("F1", fontsize=10)
    ax.set_title(
        "F1 vs flagged volume (operating points)\n"
        f"{_suptitle_line(csv_path)}",
        fontsize=10,
    )
    ax.grid(alpha=0.35)
    if np.nanmax(nf) / max(np.nanmin(nf), 1.0) > 25:
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_all_sweep_flag_rate_charts(
    df: pd.DataFrame,
    *,
    csv_path: Path,
    out_dir: Path,
    dpi: int = 140,
    dashboard_path: Optional[Path] = None,
) -> List[Path]:
    """
    Write dashboard + PR-path + F1-vs-volume charts.

    Filenames: ``<stem>_dashboard.png``, ``<stem>_pr_threshold_path.png``,
    ``<stem>_f1_vs_n_flagged.png`` unless ``dashboard_path`` is set (dashboard
    only is written there; companions use ``csv_path.stem`` in the same directory).
    """
    stem = csv_path.stem
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dash = dashboard_path if dashboard_path is not None else out_dir / f"{stem}_dashboard.png"
    paths: List[Path] = []
    paths.append(plot_sweep_flag_rate_dashboard(df, csv_path=csv_path, out_path=dash, dpi=dpi))

    companion_dir = dash.parent
    paths.append(
        plot_precision_recall_threshold_path(
            df,
            csv_path=csv_path,
            out_path=companion_dir / f"{stem}_pr_threshold_path.png",
            dpi=dpi,
        )
    )
    paths.append(
        plot_f1_vs_n_flagged(
            df,
            csv_path=csv_path,
            out_path=companion_dir / f"{stem}_f1_vs_n_flagged.png",
            dpi=dpi,
        )
    )
    return paths


def resolve_sweep_csv(
    results_root: Path,
    csv_arg: Optional[str],
    csv_path_override: Optional[Path],
) -> Path:
    """
    Resolve which CSV to chart.

    ``csv_path_override`` is used when ``--csv`` points at an existing file path.
    ``csv_arg`` may be a basename under ``results_root`` or any path that exists.
    """
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
        f"Could not resolve CSV {csv_arg!r}. Try --list or pass a path to flag_rate_sweep_*.csv."
    )


def _print_csv_list(paths: Sequence[Path], results_root: Path) -> None:
    print(f"Sweep CSVs under {results_root} (newest first):\n")
    for i, p in enumerate(paths, start=1):
        try:
            rel = p.relative_to(results_root)
        except ValueError:
            rel = p
        print(f"  [{i}]  {rel}")


def default_output_dir(csv_path: Path, out_dir: Optional[Path]) -> Path:
    """Directory for PNGs (default: same folder as the CSV)."""
    if out_dir is not None:
        return out_dir.resolve()
    return csv_path.parent.resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot flag-rate sweep dashboards and auxiliary charts from a sweep_flag_rate CSV."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="experiments/results/sweep_flag_rate",
        help="Directory containing flag_rate_sweep_*.csv files (for --list / short --csv names).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to flag_rate_sweep_*.csv, or basename under --results-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNG outputs (default: same folder as the input CSV).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit path for the dashboard PNG only; companion charts use the same directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered sweep CSVs under --results-root and exit.",
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
                f"No CSVs found under {results_root} (expected flag_rate_sweep_*.csv).",
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
        df = load_sweep_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    out_dir = default_output_dir(csv_path, Path(args.output_dir) if args.output_dir else None)
    dash_override = Path(args.output).expanduser().resolve() if args.output else None

    try:
        written = plot_all_sweep_flag_rate_charts(
            df,
            csv_path=csv_path,
            out_dir=out_dir,
            dpi=args.dpi,
            dashboard_path=dash_override,
        )
    except Exception as e:
        print(f"Plot failed: {e}", file=sys.stderr)
        return 1

    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
