"""
Jump anticipation module for insider trading detection.

Components:
    core.py       — Jump detection, wallet scoring, boost application
    manager.py    — Live system periodic scorer (rolling buffer)
    optimizer.py  — Stage 3 parameter optimisation (coordinate descent)
"""

from jump_anticipation.core import (
    JumpEvent,
    find_jumps,
    score_wallets_jump_anticipation,
    apply_jump_boost,
    run_jump_anticipation_boost,
)
from jump_anticipation.manager import JumpAnticipationManager

__all__ = [
    "JumpEvent",
    "find_jumps",
    "score_wallets_jump_anticipation",
    "apply_jump_boost",
    "run_jump_anticipation_boost",
    "JumpAnticipationManager",
]