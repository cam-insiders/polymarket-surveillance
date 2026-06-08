#!/usr/bin/env python3
"""
Charts for curated reported-insider recall vs SOTA baselines.

Input is the method-level CSV written by ``experiments.curated_sota_common``, for
example ``curated_recall_compare_methods_20260531_125728.csv``. A summary JSON
from the same run also works.

Outputs:
- ``<csv_stem>_dashboard.png``: reported-insider coverage plus z-score
  precision / F1 / F0.5.
- ``<csv_stem>_errors.png``: z-score TP/FP volume and lower-is-better error rates.

Usage:
    python -m experiments.charting.curated_compare_sota_charts --list
    python -m experiments.charting.curated_compare_sota_charts --csv path/to/curated_recall_compare_methods_....csv
    python -m experiments.charting.curated_compare_sota_charts --summary path/to/curated_recall_compare_summary_....json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The execution sandbox can have a read-only home directory, which makes
# Matplotlib's default cache path noisy. Use a writable cache unless the caller
# deliberately supplied one.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

DEFAULT_RESULTS_ROOT = "experiments/results/curated_reported_insider_recall_compare_sota"

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "method",
    "classification_recall_present",
    "ever_flagged_recall_present",
    "zscore_precision",
    "zscore_recall",
    "zscore_f1",
    "zscore_f0_5",
    "zscore_tp",
    "zscore_fp",
    "zscore_fn",
)

METHOD_ORDER: Tuple[str, ...] = (
    "full_system",
    "mitts_ofir_causal",
    "timing_heuristic",
    "isolation_forest",
    "consob_pca",
    "mitts_ofir_retrospective",
)

METHOD_COLORS: Dict[str, str] = {
    "full_system": "#3274A1",
    "mitts_ofir_causal": "#3A923A",
    "timing_heuristic": "#E1812C",
    "isolation_forest": "#6A4C93",
    "consob_pca": "#7A7A7A",
    "mitts_ofir_retrospective": "#C44E52",
}

NON_CAUSAL_METHODS = {
    "consob_pca",
    "consob_pca_faithful",
    "mitts_ofir_retrospective",
}

METRIC_COLORS: Dict[str, str] = {
    "classification_recall_present": "#3274A1",
    "ever_flagged_recall_present": "#E1812C",
    "zscore_precision": "#3A923A",
    "zscore_f1": "#6A4C93",
    "zscore_f0_5": "#C44E52",
}


def _discover_method_csvs(results_root: Path) -> List[Path]:
    if not results_root.is_dir():
        return []
    candidates: List[Tuple[float, Path]] = []
    for p in results_root.glob("curated_recall_compare_methods_*.csv"):
        if not p.is_file():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def _method_label(method: str) -> str:
    labels = {
        "full_system": "Full system",
        "isolation_forest": "Isolation forest",
        "consob_pca": "CONSOB PCA",
        "consob_pca_faithful": "CONSOB PCA",
        "mitts_ofir_causal": "Mitts-Ofir causal",
        "mitts_ofir_retrospective": "Mitts-Ofir retro",
        "timing_heuristic": "Timing heuristic",
    }
    return labels.get(method, method.replace("_", " ").title())


def _ordered_methods(df: pd.DataFrame) -> List[str]:
    present = [str(m) for m in df["method"].tolist()]
    known = [m for m in METHOD_ORDER if m in present]
    rest = sorted(m for m in present if m not in set(known))
    return known + rest


def _load_methods_from_summary(summary_path: Path) -> Tuple[pd.DataFrame, str]:
    with open(summary_path, encoding="utf-8") as f:
        meta = json.load(f)
    rows = meta.get("method_summaries", [])
    if not rows:
        raise ValueError(f"{summary_path} does not contain method_summaries.")
    return pd.DataFrame(rows), summary_path.stem.replace("_summary_", "_methods_")


def load_method_metrics(path: Path, *, from_summary: bool = False) -> Tuple[pd.DataFrame, str]:
    if from_summary or path.suffix.lower() == ".json":
        df, stem = _load_methods_from_summary(path)
    else:
        df = pd.read_csv(path)
        stem = path.stem

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. Expected the "
            "method-level curated compare CSV, not wallets or method-markets."
        )

    df = df.copy()
    df["method"] = df["method"].astype(str)
    for col in REQUIRED_COLUMNS:
        if col != "method":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    order = _ordered_methods(df)
    df["_order"] = df["method"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)
    return df, stem


def _bar_alpha(method: str) -> float:
    return 0.42 if method in NON_CAUSAL_METHODS else 0.92


def _bar_hatch(method: str) -> str:
    return "///" if method in NON_CAUSAL_METHODS else ""


def _format_percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))


def _set_percent_ylim(ax: plt.Axes, values: Sequence[float]) -> None:
    finite = [float(v) for v in values if np.isfinite(v)]
    top = max(finite) if finite else 0.0
    ax.set_ylim(0.0, min(1.05, max(0.05, top * 1.18)))


def _annotate_percent_bars(ax: plt.Axes, bars: Sequence[Any]) -> None:
    for bar in bars:
        h = float(bar.get_height())
        if not np.isfinite(h) or h <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h:.0%}" if h >= 0.095 else f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def _draw_grouped_metric_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics: Sequence[Tuple[str, str]],
    *,
    title: str,
    ylabel: str,
) -> None:
    methods = df["method"].astype(str).tolist()
    x = np.arange(len(methods), dtype=float)
    n = len(metrics)
    bar_width = min(0.24, 0.72 / max(1, n))
    offsets = (np.arange(n) - (n - 1) / 2.0) * bar_width
    all_values: List[float] = []

    for i, (col, label) in enumerate(metrics):
        vals = df[col].to_numpy(dtype=float)
        all_values.extend(vals.tolist())
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=bar_width,
            label=label,
            color=METRIC_COLORS[col],
            edgecolor="0.25",
            linewidth=0.45,
            zorder=2,
        )
        for method, bar in zip(methods, bars):
            bar.set_alpha(_bar_alpha(method))
            bar.set_hatch(_bar_hatch(method))
        _annotate_percent_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    _format_percent_axis(ax)
    _set_percent_ylim(ax, all_values)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8, frameon=True)


def _legend_handles() -> List[Patch]:
    return [
        Patch(facecolor="0.45", edgecolor="0.25", alpha=0.92, label="Causal / deployable"),
        Patch(facecolor="0.45", edgecolor="0.25", alpha=0.42, hatch="///", label="Non-causal / retrospective"),
    ]


def plot_curated_compare_sota_dashboard(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    dpi: int = 140,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    fig.suptitle(
        f"{title}\nReported-insider coverage vs z-score wallet-label quality",
        fontsize=11,
        y=1.03,
    )

    _draw_grouped_metric_bars(
        axes[0],
        df,
        [
            ("classification_recall_present", "Classified reported wallets"),
            ("ever_flagged_recall_present", "Ever flagged reported wallets"),
        ],
        title="Reported-insider coverage",
        ylabel="Share of reported wallets",
    )
    _draw_grouped_metric_bars(
        axes[1],
        df,
        [
            ("zscore_precision", "Precision"),
            ("zscore_f1", "F1"),
            ("zscore_f0_5", "F0.5"),
        ],
        title="Z-score precision / F-scores",
        ylabel="Score",
    )

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_curated_compare_sota_errors(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    dpi: int = 140,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = df["method"].astype(str).tolist()
    labels = [_method_label(m) for m in methods]
    x = np.arange(len(methods), dtype=float)
    colors = [METHOD_COLORS.get(m, "#7A7A7A") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    fig.suptitle(
        f"{title}\nFalse positives and missed z-score positives",
        fontsize=11,
        y=1.03,
    )

    ax = axes[0]
    tp = df["zscore_tp"].to_numpy(dtype=float)
    fp = df["zscore_fp"].to_numpy(dtype=float)
    bars_tp = ax.bar(
        x,
        tp,
        width=0.62,
        color="#3A923A",
        edgecolor="0.25",
        linewidth=0.45,
        label="TP",
        zorder=2,
    )
    bars_fp = ax.bar(
        x,
        fp,
        bottom=tp,
        width=0.62,
        color="#C44E52",
        edgecolor="0.25",
        linewidth=0.45,
        label="FP",
        zorder=2,
    )
    for method, bars in ((m, (b1, b2)) for m, b1, b2 in zip(methods, bars_tp, bars_fp)):
        for bar in bars:
            bar.set_alpha(_bar_alpha(method))
            bar.set_hatch(_bar_hatch(method))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Wallet-market pairs", fontsize=9)
    ax.set_title("Predicted positives: TP vs FP", fontsize=10)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    ax = axes[1]
    false_discovery = (1.0 - df["zscore_precision"].to_numpy(dtype=float)).clip(0.0, 1.0)
    missed_positive = (1.0 - df["zscore_recall"].to_numpy(dtype=float)).clip(0.0, 1.0)
    width = 0.28
    bars_fdr = ax.bar(
        x - width / 2.0,
        false_discovery,
        width=width,
        color="#C44E52",
        edgecolor="0.25",
        linewidth=0.45,
        label="False discovery rate",
        zorder=2,
    )
    bars_miss = ax.bar(
        x + width / 2.0,
        missed_positive,
        width=width,
        color="#E1812C",
        edgecolor="0.25",
        linewidth=0.45,
        label="Missed-positive rate",
        zorder=2,
    )
    for method, bar1, bar2 in zip(methods, bars_fdr, bars_miss):
        for bar in (bar1, bar2):
            bar.set_alpha(_bar_alpha(method))
            bar.set_hatch(_bar_hatch(method))
    _annotate_percent_bars(ax, bars_fdr)
    _annotate_percent_bars(ax, bars_miss)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Rate (lower is better)", fontsize=9)
    ax.set_title("Error rates (lower is better)", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    _format_percent_axis(ax)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.91])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def plot_curated_compare_sota_charts(
    df: pd.DataFrame,
    *,
    stem: str,
    out_dir: Path,
    dpi: int = 140,
) -> List[Path]:
    title = stem.replace("_", " ")
    return [
        plot_curated_compare_sota_dashboard(
            df,
            title=title,
            out_path=out_dir / f"{stem}_dashboard.png",
            dpi=dpi,
        ),
        plot_curated_compare_sota_errors(
            df,
            title=title,
            out_path=out_dir / f"{stem}_errors.png",
            dpi=dpi,
        ),
    ]


def _print_csv_list(csvs: Sequence[Path], results_root: Path) -> None:
    print(f"Method CSVs under {results_root} (newest first):\n")
    for i, p in enumerate(csvs, start=1):
        try:
            rel = p.relative_to(results_root)
        except ValueError:
            rel = p
        print(f"  [{i}]  {rel}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot curated reported-insider recall vs SOTA method charts."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing curated_recall_compare_methods_*.csv files.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to a method-level compare CSV.")
    parser.add_argument("--summary", type=str, default=None, help="Path to a compare summary JSON.")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: <csv-or-summary-dir>/charts.")
    parser.add_argument("--list", action="store_true", help="List discovered method CSVs and exit.")
    parser.add_argument("--interactive", action="store_true", help="Choose from --list by number.")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI for PNG output.")

    args = parser.parse_args(list(argv) if argv is not None else None)

    results_root = (Path.cwd() / args.results_root).resolve()
    csvs = _discover_method_csvs(results_root)
    input_path: Optional[Path] = None
    from_summary = False

    if args.list:
        if not csvs:
            print(f"No curated_recall_compare_methods_*.csv files found under {results_root}.")
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
        input_path = csvs[idx - 1]
    elif args.summary:
        input_path = Path(args.summary).expanduser().resolve()
        from_summary = True
    elif args.csv:
        input_path = Path(args.csv).expanduser().resolve()
    elif args.interactive:
        if not csvs:
            print(f"No CSVs under {results_root}.", file=sys.stderr)
            return 1
        _print_csv_list(csvs, results_root)
        try:
            idx = int(input("\nEnter index (1-based): ").strip())
        except (EOFError, ValueError):
            print("Invalid index.", file=sys.stderr)
            return 1
        if idx < 1 or idx > len(csvs):
            print(f"Invalid index: {idx}", file=sys.stderr)
            return 1
        input_path = csvs[idx - 1]
    else:
        print("Specify --csv or --summary, or use --list [--interactive].", file=sys.stderr)
        return 1

    assert input_path is not None
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    try:
        df, stem = load_method_metrics(input_path, from_summary=from_summary)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(e, file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (input_path.parent / "charts")
    paths = plot_curated_compare_sota_charts(df, stem=stem, out_dir=out_dir, dpi=args.dpi)
    for p in paths:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
