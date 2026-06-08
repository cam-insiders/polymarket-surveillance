"""Shared launcher for monthly timeframe train/backtest batches."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence


@dataclass(frozen=True)
class MonthWindow:
    label: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


MONTH_WINDOWS: Sequence[MonthWindow] = (
    MonthWindow("2025-01", "2025-01-01", "2025-01-14", "2025-01-15", "2025-01-28"),
    MonthWindow("2025-02", "2025-02-01", "2025-02-14", "2025-02-15", "2025-02-28"),
    MonthWindow("2025-03", "2025-03-01", "2025-03-14", "2025-03-15", "2025-03-28"),
    MonthWindow("2025-04", "2025-04-01", "2025-04-14", "2025-04-15", "2025-04-28"),
)


def add_monthly_launcher_args(
    parser: argparse.ArgumentParser,
    *,
    default_objective: str,
    default_output_root: Optional[Path],
    output_help: str,
) -> None:
    parser.add_argument("--objective", type=str, default=default_objective)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help=output_help,
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=("coordinate_descent", "alternating_det_clust"),
        default="alternating_det_clust",
    )
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--n-passes", type=int, default=2)
    parser.add_argument("--min-usd-amount", type=float, default=500.0)
    parser.add_argument(
        "--disable-trade-prefilter",
        dest="enable_trade_prefilter",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--disable-layer2-attribution",
        dest="enable_layer2_attribution",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--disable-jump-anticipation",
        dest="enable_jump_anticipation",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--disable-ja-optimization",
        dest="enable_ja_optimization",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Workers per child run. Default: CPU count divided by the number "
            "of concurrent monthly runs."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run months one after another instead of all selected months at once.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and exit without running.",
    )
    parser.add_argument(
        "--extra-child-arg",
        action="append",
        default=[],
        help=(
            "Extra argument forwarded to each child. Repeat for multiple tokens, "
            "for example --extra-child-arg=--min-window-trades --extra-child-arg=2."
        ),
    )


def default_workers_per_child(num_jobs: int, *, sequential: bool) -> int:
    cpu_count = os.cpu_count() or 1
    concurrent_jobs = 1 if sequential else max(1, int(num_jobs))
    return max(1, cpu_count // concurrent_jobs)


def build_monthly_command(
    *,
    window: MonthWindow,
    output_dir: Path,
    objective: str,
    args: argparse.Namespace,
    max_workers: int,
    child_module: str = "experiments.timeframe_trade_window_train_backtest",
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        child_module,
        "--train-start",
        window.train_start,
        "--train-end",
        window.train_end,
        "--test-start",
        window.test_start,
        "--test-end",
        window.test_end,
        "--optimizer-mode",
        args.optimizer_mode,
        "--objective",
        objective,
        "--n-starts",
        str(args.n_starts),
        "--n-passes",
        str(args.n_passes),
        "--min-usd-amount",
        str(args.min_usd_amount),
        "--output-dir",
        str(output_dir),
        "--max-workers",
        str(max_workers),
    ]
    if args.enable_trade_prefilter:
        cmd.append("--enable-trade-prefilter")
    if args.enable_layer2_attribution:
        cmd.append("--enable-layer2-attribution")
    if args.enable_jump_anticipation:
        cmd.append("--enable-jump-anticipation")
    if args.enable_ja_optimization:
        cmd.append("--enable-ja-optimization")
    cmd.extend(str(x) for x in args.extra_child_arg)
    return cmd


def run_monthly_launcher(
    *,
    args: argparse.Namespace,
    output_root: Path,
    objective: str,
    child_module: str = "experiments.timeframe_trade_window_train_backtest",
    banner: Optional[str] = None,
) -> None:
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = default_workers_per_child(
            len(MONTH_WINDOWS),
            sequential=bool(args.sequential),
        )

    jobs: List[tuple[MonthWindow, List[str], Path]] = []
    launch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for window in MONTH_WINDOWS:
        out_dir = output_root / window.label
        cmd = build_monthly_command(
            window=window,
            output_dir=out_dir,
            objective=objective,
            args=args,
            max_workers=int(max_workers),
            child_module=child_module,
        )
        log_path = out_dir / f"launcher_{window.label}_{launch_ts}.log"
        jobs.append((window, cmd, log_path))

    if args.dry_run:
        if banner:
            print(banner)
        print(f"Output root: {output_root}")
        print(f"Workers per child: {max_workers}")
        for _window, cmd, log_path in jobs:
            print(" ".join(cmd))
            print(f"  -> {log_path}")
        return

    exit_code = 0
    if args.sequential:
        for window, cmd, log_path in jobs:
            proc = _run_one(cmd, log_path)
            code = proc.wait()
            if code != 0:
                print(f"FAILED {window.label} (exit {code})", file=sys.stderr)
                exit_code = code
            else:
                print(f"OK {window.label}")
    else:
        procs: List[tuple[MonthWindow, subprocess.Popen[bytes], Path]] = []
        for window, cmd, log_path in jobs:
            procs.append((window, _run_one(cmd, log_path), log_path))
        for window, proc, log_path in procs:
            code = proc.wait()
            if code != 0:
                print(f"FAILED {window.label} (exit {code}); see {log_path}", file=sys.stderr)
                exit_code = code
            else:
                print(f"OK {window.label}")

    if exit_code != 0:
        raise SystemExit(exit_code)


def _run_one(cmd: List[str], log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    print(f"Starting: {' '.join(cmd)}")
    print(f"  log: {log_path}")
    return subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=Path.cwd(),
    )


def parse_and_run(
    *,
    description: str,
    default_objective: str,
    default_output_root: Optional[Path],
    output_help: str,
    child_module: str = "experiments.timeframe_trade_window_train_backtest",
    objective_normalizer: Optional[Callable[[str], str]] = None,
    output_root_factory: Optional[Callable[[argparse.Namespace], Path]] = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    add_monthly_launcher_args(
        parser,
        default_objective=default_objective,
        default_output_root=default_output_root,
        output_help=output_help,
    )
    args = parser.parse_args()
    objective = (
        objective_normalizer(args.objective)
        if objective_normalizer is not None
        else args.objective
    )
    output_root = (
        output_root_factory(args)
        if output_root_factory is not None
        else args.output_root
    )
    if output_root is None:
        raise ValueError("output_root must be provided or derived")
    banner = None
    if objective != args.objective:
        banner = f"Objective: {args.objective} -> {objective}"
    run_monthly_launcher(
        args=args,
        output_root=output_root,
        objective=objective,
        child_module=child_module,
        banner=banner,
    )
