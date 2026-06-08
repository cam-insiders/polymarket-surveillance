"""Recover completed alternating multi-start trajectories from a log file."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as a script from repo root (``python -m experiments.recover_...``).
from backtesting.clustering_parameter_grid import ClusteringParameterGrid
from backtesting.multi_start_optimizer import MultiStartCoordinateDescentOptimizer
from backtesting.parameter_grid import ParameterGrid
from experiments.timeframe_optimizers import (
    CLUSTERING_OPTIMIZE_ORDER,
    DETECTOR_OPTIMIZE_ORDER,
)


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_LOG_PREFIX = re.compile(r"^\S+\s+\S+\s+\[\w+\]\s+")

_RE_HEADER_INIT = re.compile(
    r"MULTI-START ALTERNATING DETECTOR\+CLUSTERING \| n_starts=(?P<n_starts>\d+) "
    r"\| strategy=(?P<strategy>\S+) "
    r"\| train_markets=(?P<n_train>\d+) \| val_markets=(?P<n_val>\d+)"
)
_RE_RUN_HEADER = re.compile(
    r"Alternating multi-start run (?P<idx1>\d+)/(?P<total>\d+) \| "
    r"(?P<label>\S+) \(strategy=(?P<strategy>\S+)\)"
)
_RE_DETECTOR_ORDER = re.compile(r"detector_order: (?P<order>\[.*\])")
_RE_DONE = re.compile(
    r"Done (?P<group>\S+): baseline_\S+=(?P<baseline>[-\d.eE+]+), "
    r"best_\S+=(?P<best>[-\d.eE+]+), improvement=(?P<improvement>[-+\d.]+)%"
)
_RE_BEST_PARAMS = re.compile(r"Best params:\s+(?P<dict>\{.*\})")
_RE_TRAJ_SUMMARY_VAL = re.compile(
    r"->\s+(?P<label>\S+): train_obj=(?P<train>[-\d.]+)\s+val_obj=(?P<val>[-\d.]+)\s+"
    r"gap=(?P<gap>[-+\d.]+)\s+\((?P<train_el>[\d.]+)s train \+ (?P<val_el>[\d.]+)s val\)"
)
_RE_TRAJ_SUMMARY_NOVAL = re.compile(
    r"->\s+(?P<label>\S+): final_obj=(?P<train>[-\d.]+)\s+\((?P<train_el>[\d.]+)s\)"
)

def _apply_detector_params(
    detector_config: Dict[str, Any],
    group_name: str,
    best_params: Dict[str, Any],
) -> None:
    """Apply a ``Best params:`` dict from the cached CD optimizer to the
    running detector config in place.  Mirrors
    :func:`ParameterGrid.generate_configs_for_detector`: alert-threshold updates
    the top-level scalar; everything else mutates
    ``config['detectors'][group_name]``.
    """
    if group_name == "alert_threshold":
        if "alert_threshold" not in best_params:
            raise ValueError(
                f"alert_threshold Best params missing 'alert_threshold' key: {best_params}"
            )
        detector_config["alert_threshold"] = best_params["alert_threshold"]
        return

    detectors = detector_config.setdefault("detectors", {})
    det = detectors.setdefault(group_name, {})
    for key, value in best_params.items():
        det[key] = value


def _apply_clustering_params(
    clustering_config: Dict[str, Any],
    best_params: Dict[str, Any],
) -> None:
    """Apply a ``Best params:`` dict from the clustering CD optimizer in
    place.  Mirrors
    :func:`ClusteringParameterGrid.generate_configs_for_param_group`: keys
    prefixed ``boost.`` land in ``config['boost'][<rest>]``, everything else
    lands at the top level.
    """
    boost = clustering_config.setdefault("boost", {})
    for key, value in best_params.items():
        if key.startswith("boost."):
            sub_key = key.split(".", 1)[1]
            boost[sub_key] = value
        else:
            clustering_config[key] = value

def _strip_prefix(line: str) -> str:
    return _LOG_PREFIX.sub("", line, count=1).rstrip("\n")


def _iter_messages(log_path: Path):
    """Yield ``(line_no, message)`` for every log line, stripped of the
    timestamp/level prefix."""
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            yield line_no, _strip_prefix(raw)

def _parse_header_meta(messages: List[Tuple[int, str]]) -> Dict[str, Any]:
    """Extract ``n_starts``, ``start_strategy``, train/val market counts from
    the ``MULTI-START ALTERNATING DETECTOR+CLUSTERING`` banner.  Raises if the
    banner is missing."""
    for _, msg in messages:
        m = _RE_HEADER_INIT.search(msg)
        if m:
            return {
                "n_starts": int(m.group("n_starts")),
                "start_strategy": m.group("strategy"),
                "n_train_markets_logged": int(m.group("n_train")),
                "n_val_markets_logged": int(m.group("n_val")),
            }
    raise RuntimeError(
        "Could not find 'MULTI-START ALTERNATING DETECTOR+CLUSTERING' banner in log"
    )


def _split_into_run_blocks(
    messages: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    """Split the log messages into a list of per-run blocks.  Each block is a
    dict ``{idx0, label, strategy, detector_order, body, summary_msg,
    completed}``.  The last block may be incomplete (no summary line)."""
    blocks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line_no, msg in messages:
        m_run = _RE_RUN_HEADER.search(msg)
        if m_run:
            if current is not None:
                blocks.append(current)
            current = {
                "idx0": int(m_run.group("idx1")) - 1,
                "label": m_run.group("label"),
                "strategy": m_run.group("strategy"),
                "detector_order": None,
                "body": [],
                "summary_msg": None,
                "completed": False,
                "first_line_no": line_no,
            }
            continue

        if current is None:
            continue

        if current["detector_order"] is None:
            m_order = _RE_DETECTOR_ORDER.search(msg)
            if m_order:
                current["detector_order"] = list(ast.literal_eval(m_order.group("order")))
                continue

        # Trajectory summary line — matches both val and no-val flavours.
        m_sum_val = _RE_TRAJ_SUMMARY_VAL.search(msg)
        m_sum_noval = None if m_sum_val else _RE_TRAJ_SUMMARY_NOVAL.search(msg)
        if m_sum_val or m_sum_noval:
            if (m_sum_val or m_sum_noval).group("label") == current["label"]:
                current["summary_msg"] = msg
                current["completed"] = True
                continue

        current["body"].append(msg)

    if current is not None:
        blocks.append(current)

    return blocks


def _apply_block(
    block: Dict[str, Any],
    detector_config: Dict[str, Any],
    clustering_config: Dict[str, Any],
) -> Tuple[int, int]:
    """Walk a completed run block's body, applying every ``Best params:`` to
    the in-memory configs in log order.  Returns ``(detector_updates,
    clustering_updates)`` for diagnostics."""
    det_updates = 0
    clust_updates = 0
    pending_group: Optional[str] = None  # name of the last ``Done <group>:``

    for msg in block["body"]:
        m_done = _RE_DONE.search(msg)
        if m_done:
            pending_group = m_done.group("group")
            continue

        m_params = _RE_BEST_PARAMS.search(msg)
        if m_params and pending_group is not None:
            try:
                params = ast.literal_eval(m_params.group("dict"))
            except (ValueError, SyntaxError) as exc:
                raise RuntimeError(
                    f"Could not parse Best params dict in run '{block['label']}': {msg!r}"
                ) from exc

            if pending_group in DETECTOR_OPTIMIZE_ORDER:
                _apply_detector_params(detector_config, pending_group, params)
                det_updates += 1
            elif pending_group in CLUSTERING_OPTIMIZE_ORDER:
                _apply_clustering_params(clustering_config, params)
                clust_updates += 1
            else:
                raise RuntimeError(
                    f"Unknown param group '{pending_group}' in run '{block['label']}'"
                )
            pending_group = None

    return det_updates, clust_updates


def _parse_summary(block: Dict[str, Any]) -> Dict[str, float]:
    msg = block["summary_msg"]
    if msg is None:
        return {}
    m = _RE_TRAJ_SUMMARY_VAL.search(msg)
    if m:
        return {
            "train_final_objective": float(m.group("train")),
            "val_final_objective": float(m.group("val")),
            "train_val_gap": float(m.group("gap")),
            "train_elapsed_seconds": float(m.group("train_el")),
            "val_elapsed_seconds": float(m.group("val_el")),
            "used_validation": True,
        }
    m = _RE_TRAJ_SUMMARY_NOVAL.search(msg)
    if m:
        return {
            "train_final_objective": float(m.group("train")),
            "val_final_objective": float("nan"),
            "train_val_gap": float("nan"),
            "train_elapsed_seconds": float(m.group("train_el")),
            "val_elapsed_seconds": 0.0,
            "used_validation": False,
        }
    return {}


def build_reconstruction(
    *,
    log_path: Path,
    random_seed: int,
    perturb_prob: float,
    include_baseline_start: bool,
    shuffle_order: bool,
) -> Dict[str, Any]:
    """Parse a log and return a JSON-serialisable reconstruction."""
    messages = list(_iter_messages(log_path))
    header_meta = _parse_header_meta(messages)
    blocks = _split_into_run_blocks(messages)

    if not blocks:
        raise RuntimeError("No 'Alternating multi-start run K/N' headers found in log")

    n_starts = header_meta["n_starts"]
    logged_strategy = header_meta["start_strategy"]

    # Reproduce the deterministic start-generation and per-start shuffles.
    baseline_detector_config = ParameterGrid.get_baseline_config()
    rng = random.Random(random_seed)
    msopt = MultiStartCoordinateDescentOptimizer(
        n_starts=n_starts,
        start_strategy=logged_strategy,
        perturb_prob=perturb_prob,
        include_baseline_start=include_baseline_start,
        shuffle_order=shuffle_order,
        random_seed=random_seed,
    )
    starts = msopt._generate_starts(baseline_detector_config, rng)

    # Seed each reconstruction from a fresh copy of the clustering baseline;
    # CONFIG exposes no override for that key, so this matches the runtime.
    clustering_baseline = ClusteringParameterGrid.get_baseline_config()

    per_start: List[Dict[str, Any]] = []
    completed_indices: List[int] = []

    for start_idx, (label, strategy, initial_detector_config) in enumerate(starts):
        # Reproduce the in-loop shuffle exactly as the runner does.
        expected_order = list(DETECTOR_OPTIMIZE_ORDER)
        if (
            shuffle_order
            and start_idx > 0
            and not (include_baseline_start and start_idx == 0)
        ):
            rng.shuffle(expected_order)

        block = next((b for b in blocks if b["idx0"] == start_idx), None)
        if block is None:
            logger.info(
                "Start %d (%s) not present in log — treating as not-yet-run",
                start_idx,
                label,
            )
            continue

        # Cross-checks: label + strategy + detector_order must match.
        if block["label"] != label:
            raise RuntimeError(
                f"Start {start_idx}: log label '{block['label']}' does not match "
                f"replayed label '{label}'"
            )
        if block["strategy"] != strategy:
            raise RuntimeError(
                f"Start {start_idx}: log strategy '{block['strategy']}' does not match "
                f"replayed strategy '{strategy}'"
            )

        logged_order = block["detector_order"]
        if logged_order is None:
            raise RuntimeError(
                f"Start {start_idx}: no 'detector_order: [...]' line found in block "
                f"(first line {block['first_line_no']})"
            )
        if logged_order != expected_order:
            raise RuntimeError(
                f"Start {start_idx} ({label}): detector_order mismatch.\n"
                f"  logged:   {logged_order}\n"
                f"  replayed: {expected_order}\n"
                "RNG state is not reproducing — refuse to continue."
            )

        if not block["completed"]:
            logger.info(
                "Start %d (%s) was interrupted mid-run — skipping (partial block at line %d)",
                start_idx,
                label,
                block["first_line_no"],
            )
            continue

        detector_config = deepcopy(initial_detector_config)
        clustering_config = deepcopy(clustering_baseline)
        det_updates, clust_updates = _apply_block(block, detector_config, clustering_config)

        summary = _parse_summary(block)
        if not summary:
            raise RuntimeError(
                f"Start {start_idx} ({label}): marked completed but could not parse "
                "trajectory summary line"
            )

        per_start.append(
            {
                "start_idx": start_idx,
                "label": label,
                "strategy": strategy,
                "detector_order_logged": logged_order,
                "detector_order_replayed": expected_order,
                "initial_detector_config": initial_detector_config,
                "final_detector_config": detector_config,
                "final_clustering_config": clustering_config,
                "detector_updates_applied": det_updates,
                "clustering_updates_applied": clust_updates,
                **summary,
            }
        )
        completed_indices.append(start_idx)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_log": str(log_path),
        "random_seed": random_seed,
        "start_strategy": logged_strategy,
        "perturb_prob": perturb_prob,
        "include_baseline_start": include_baseline_start,
        "shuffle_order": shuffle_order,
        "n_starts": n_starts,
        "n_train_markets_logged": header_meta["n_train_markets_logged"],
        "n_val_markets_logged": header_meta["n_val_markets_logged"],
        "completed_starts": completed_indices,
        "per_start": per_start,
    }

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct completed trajectories of an interrupted "
            "run_multi_start_alternating_timeframe run from its log."
        )
    )
    parser.add_argument("--log", required=True, type=Path, help="Path to the .log file.")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Where to write the reconstruction JSON (consumed by --resume-from-json).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="Must match the --start-seed used by the original run (default 42).",
    )
    parser.add_argument(
        "--perturb-prob",
        type=float,
        default=0.3,
        help="Must match the --perturb-prob used by the original run (default 0.3).",
    )
    parser.add_argument(
        "--no-shuffle-order",
        dest="shuffle_order",
        action="store_false",
        default=True,
        help="Use only if the original run disabled per-start order shuffling.",
    )
    parser.add_argument(
        "--exclude-baseline-start",
        dest="include_baseline_start",
        action="store_false",
        default=True,
        help="Use only if the original run was launched with --exclude-baseline-start.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.log.exists():
        logger.error("Log file does not exist: %s", args.log)
        return 2

    reconstruction = build_reconstruction(
        log_path=args.log,
        random_seed=args.start_seed,
        perturb_prob=args.perturb_prob,
        include_baseline_start=args.include_baseline_start,
        shuffle_order=args.shuffle_order,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(reconstruction, f, indent=2, default=str)

    logger.info(
        "Reconstructed %d/%d starts from %s -> %s",
        len(reconstruction["completed_starts"]),
        reconstruction["n_starts"],
        args.log,
        args.out,
    )
    for entry in reconstruction["per_start"]:
        logger.info(
            "  [%2d] %-26s %-8s  train=%.4f  val=%.4f  det_updates=%d  clust_updates=%d",
            entry["start_idx"],
            entry["label"],
            entry["strategy"],
            entry["train_final_objective"],
            entry["val_final_objective"],
            entry["detector_updates_applied"],
            entry["clustering_updates_applied"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
