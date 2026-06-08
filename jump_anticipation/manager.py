"""
jump_anticipation/manager.py

Live system counterpart to the backtesting jump anticipation scorer.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict

from jump_anticipation.core import find_jumps, score_wallets_jump_anticipation
from models import Trade

logger = logging.getLogger(__name__)


class JumpAnticipationManager:
    """Periodic background scorer for jump-anticipation behaviour in the live system."""

    def __init__(self, config: Dict):
        self.config = config
        self.scoring_interval_s = float(config.get("scoring_interval_minutes", 15)) * 60
        self.buffer_max_age_ms = int(float(config.get("buffer_hours", 24)) * 3600 * 1000)

        self._buffer: Deque[Trade] = deque()
        self._wallet_boosts: Dict[str, float] = {}
        self._last_scored_at: float = 0.0

        # Validate buffer is large enough to cover the full detection window.
        # Minimum needed: jump_window + pre_jump_lookback, with 50% headroom.
        jump_window_h = float(config.get("jump_window_minutes", 30)) / 60.0
        lookback_h = float(config.get("pre_jump_lookback_minutes", 60)) / 60.0
        min_required_h = (jump_window_h + lookback_h) * 1.5
        buffer_h = float(config.get("buffer_hours", 24))

        if buffer_h < min_required_h:
            logger.warning(
                f"JumpAnticipationManager: buffer_hours={buffer_h:.1f}h may be too short. "
                f"jump_window={config.get('jump_window_minutes', 30)}min + "
                f"pre_jump_lookback={config.get('pre_jump_lookback_minutes', 60)}min "
                f"requires at least {min_required_h:.1f}h (with 1.5x headroom). "
                f"Trades at the start of the buffer may have no pre-jump context. "
                f"Consider increasing buffer_hours in CONFIG['jump_anticipation_config']."
            )

        logger.info(
            f"JumpAnticipationManager initialised | "
            f"interval={config.get('scoring_interval_minutes', 15)}min | "
            f"buffer={buffer_h:.0f}h | "
            f"jump_threshold={config.get('jump_threshold', 0.05)} | "
            f"max_boost={config.get('max_boost_factor', 2.0)}"
        )

    def on_trade(self, trade: Trade) -> None:
        """
        Append trade to rolling buffer.
        """
        self._buffer.append(trade)
        cutoff_ms = trade.timestamp_ms - self.buffer_max_age_ms
        while self._buffer and self._buffer[0].timestamp_ms < cutoff_ms:
            self._buffer.popleft()

    def maybe_score(self) -> bool:
        """
        Run scoring if the scoring interval has elapsed.
        """
        if time.time() - self._last_scored_at < self.scoring_interval_s:
            return False
        self._run_scoring()
        self._last_scored_at = time.time()
        return True

    def get_wallet_boost(self, wallet: str) -> float:
        """Return cached jump-anticipation boost for this wallet (1.0 if unscored)."""
        return self._wallet_boosts.get(wallet, 1.0)

    def _run_scoring(self) -> None:
        """Full pipeline: find_jumps -> score_wallets -> update boost cache.
        """
        trades = list(self._buffer)
        n = len(trades)

        if n < 10:
            logger.debug(
                f"JumpAnticipationManager: only {n} trades in buffer — skipping"
            )
            return

        t0 = time.time()
        jumps = find_jumps(trades, self.config)

        if not jumps:
            # No detectable jumps — reset all boosts to 1.0 (implicitly via empty dict)
            self._wallet_boosts = {}
            logger.debug("JumpAnticipationManager: no jumps in buffer — boosts cleared")
            return

        scores = score_wallets_jump_anticipation(trades, jumps, self.config)

        # Keep only wallets with meaningful boost (saves memory, speeds up lookup)
        self._wallet_boosts = {w: b for w, b in scores.items() if b > 1.001}

        logger.info(
            f"JumpAnticipationManager: {n:,} trades | "
            f"{len(jumps):,} jumps | "
            f"{len(self._wallet_boosts)} wallets boosted | "
            f"{time.time() - t0:.2f}s"
        )