"""Parameter search grids for detector coordinate descent."""

import random
from itertools import product
from copy import deepcopy
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _cat(*seqs, decimals: int = 10) -> List:
    """Merge, deduplicate, and sort multiple sequences."""
    merged = []
    for seq in seqs:
        for v in seq:
            merged.append(round(float(v), decimals) if isinstance(v, float) else v)
    seen = set()
    unique = []
    for v in merged:
        key = round(v, 8) if isinstance(v, float) else v
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return sorted(unique)


class ParameterGrid:
    """Parameter search space for coordinate descent optimisation."""

    @staticmethod
    def get_detector_groups_new() -> Dict[str, Dict[str, List]]:
        """Very wide profile for broad regime search."""
        return {
            "volume_anomaly": {
                "lookback_window_hours":    np.round(np.geomspace(1, 24, 4)).astype(int).tolist(),
                "min_trades_for_baseline":  [1, 2, 5, 10], # np.arange(1, 12, 3).tolist(),   # [1, 4, 7, 10]
                "z_score_threshold":        np.round(np.linspace(1.5, 4, 3), 1).tolist(), # [1.5, 2.5, 3.5]
                "min_absolute_notional":    [20, 100, 500, 1000], # np.geomspace(20, 1000, 5).round(0).astype(int).tolist(),  # [20, 50, 100, 250, 1000]
                "max_confidence":           np.round(np.linspace(0.10, 1, 5), 2).tolist(), # [0.10, 0.30, 0.50, 0.70, 0.90]
            },
            "probability_impact": {
                "min_delta_prob":       np.round(np.linspace(0.005, 0.070, 9), 3).tolist(),
                "min_delta_log_odds":   np.round(np.linspace(0.10,  1.25,  9), 2).tolist(),
                "min_notional":         np.geomspace(250, 4000, 8).round(0).astype(int).tolist(),
                "max_confidence":       np.round(np.linspace(0.10, 1, 10), 2).tolist(),
            },
            # 9*9*8*8 = 5,184

            "accumulation_detector": {
                "min_accumulation_usdc":    np.geomspace(100, 100000, 5).round(0).astype(int).tolist(),
                "min_directional_ratio":    np.round(np.linspace(0.70, 0.99, 8), 2).tolist(),
                "max_confidence":           np.round(np.linspace(0.10, 1, 10), 2).tolist(),
                "min_outcome_concentration": np.round(np.linspace(0.70, 0.99, 8), 2).tolist(),
            },
            # 9*8*8*8 = 4,608

            "extreme_position": {
                "tail_threshold":   np.round(np.linspace(0.05, 0.30, 9), 2).tolist(),
                "min_notional":     np.geomspace(100, 5000, 8).round(0).astype(int).tolist(),
                "max_confidence":   np.round(np.linspace(0.10, 1, 10), 2).tolist(),
            },
            # 9*8*8 = 576

            "contra_outcome_silence": {
                "min_gap_samples":         np.arange(5, 31, 5).tolist(),
                "silence_threshold":       np.round(np.linspace(2.0, 10.0, 9), 1).tolist(),
                "min_notional":            np.geomspace(100, 5000, 7).round(0).astype(int).tolist(),
                "max_contra_age_minutes":  np.round(np.geomspace(15, 240, 6), 0).astype(int).tolist(),
                "max_confidence":          np.round(np.linspace(0.10, 1, 10), 2).tolist(),
            },
            # 6*9*7*6*10 = 22,680

            "alert_threshold": {
                "value": np.round(np.linspace(0.05, 0.95, 19), 2).tolist(),
            },
            # 19
        }

    @staticmethod
    def _materialize_values(values: Any) -> List[Any]:
        if isinstance(values, list):
            return values
        if isinstance(values, (str, bytes)):
            return [values]
        if isinstance(values, Iterable):
            return list(values)
        return [values]

    @staticmethod
    def _materialize_detector_groups(
        detector_groups: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        return {
            detector_name: {
                param_name: ParameterGrid._materialize_values(values)
                for param_name, values in param_grid.items()
            }
            for detector_name, param_grid in detector_groups.items()
        }

    @staticmethod
    def get_detector_groups_old() -> Dict[str, Dict[str, List]]:
        """Broad active grid for optimisation runs."""
        return {
            "volume_anomaly": {
                "lookback_window_hours":    [1, 4, 12, 24],
                "min_trades_for_baseline":  [1, 5, 10],
                "z_score_threshold":        [2.0, 2.6, 3.2, 4.0, 5.0],
                "min_absolute_notional":    [20, 100, 500, 1000],
                "max_confidence":           [0.05, 0.15, 0.35, 0.70],
            },
            # 4*3*5*4*4 = 960

            "probability_impact": {
                "min_delta_prob":       [0.005, 0.015, 0.030, 0.050, 0.070],
                "min_delta_log_odds":   [0.10, 0.25, 0.45, 0.75, 1.25],
                "min_notional":         [250, 500, 1000, 2000, 4000],
                "max_confidence":       [0.10, 0.25, 0.45, 0.70],
            },
            # 5*5*5*4 = 500

            "accumulation_detector": {
                "min_accumulation_usdc":        [1000, 3000, 5000, 10000, 20000, 40000],
                "min_directional_ratio":        [0.70, 0.80, 0.90, 0.96, 0.99],
                "max_confidence":               [0.05, 0.15, 0.35, 0.60, 0.85],
                "min_outcome_concentration":    [0.70, 0.80, 0.90, 0.96, 0.99],
            },
            # 6*5*5*5 = 750

            "extreme_position": {
                "tail_threshold":   [0.05, 0.08, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30],
                "min_notional":     [100, 250, 400, 600, 900, 1300, 2000, 3000],
                "max_confidence":   [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00],
            },
            # 8*8*8 = 512

            "contra_outcome_silence": {
                "min_gap_samples":         [5, 10, 20, 30],
                "silence_threshold":       [2.0, 4.0, 6.5, 10.0],
                "min_notional":            [100, 500, 1000, 2500],
                "max_contra_age_minutes":  [15, 45, 120, 240],
                "max_confidence":          [0.05, 0.20, 0.40],
            },
            # 4*4*4*4*3 = 768

            "alert_threshold": {
                "value": [
                    0.05, 0.10, 0.15, 0.20, 0.25,
                    0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75,
                    0.80, 0.85, 0.90, 0.95,
                ],
            },
            # 19
        }

    @staticmethod
    def get_detector_groups_og() -> Dict[str, Dict[str, List]]:
        """Focused grid retained for historical comparison."""
        return {
            "volume_anomaly": {
                "lookback_window_hours":    [1, 2],
                "min_trades_for_baseline":  [1, 2, 3],
                "z_score_threshold":        np.round(np.arange(3.4, 4.3, 0.2), 1).tolist(),   # [3.4, 3.6, 3.8, 4.0, 4.2]
                "min_absolute_notional":    [20, 35, 50, 75, 100],
                "max_confidence":           [0.02, 0.03, 0.05, 0.07, 0.10],
            },
            # 2*3*5*5*5 = 750

            "probability_impact": {
                "min_delta_prob":       np.round(np.arange(0.015, 0.041, 0.005), 3).tolist(),  # [0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
                "min_delta_log_odds":   np.round(np.arange(0.30, 0.51, 0.05), 2).tolist(),     # [0.30, 0.35, 0.40, 0.45, 0.50]
                "min_notional":         [750, 900, 1000, 1250],
                "max_confidence":       np.round(np.arange(0.10, 0.41, 0.05), 2).tolist(),     # [0.10, 0.15, ..., 0.40]
            },
            # 6*5*4*7 = 840 (slight expansion from original 400 — added 0.035/0.040 delta_prob)

            "accumulation_detector": {
                # Reduced around historically selected basins while staying broad:
                # min_accumulation tends to settle around 4k / 10k / 20k,
                # directional ratio is usually near 0.90 with occasional ~0.94-0.98,
                # concentration is most often near 0.90.
                "min_accumulation_usdc":        [2000, 4000, 6000, 8000, 10000, 15000, 20000],
                "min_directional_ratio":        [0.90, 0.92, 0.94, 0.96, 0.98],
                "max_confidence":               [0.02, 0.05, 0.08, 0.1,],
                "min_outcome_concentration":    [0.90, 0.93, 0.95, 0.97],
            },
            # 7*5*4*4 = 560

            "extreme_position": {
                "tail_threshold":   np.round(np.arange(0.16, 0.25, 0.02), 2).tolist(),   # [0.16, 0.18, 0.20, 0.22, 0.24]
                "min_notional":     [350, 450, 550, 700, 1000, 2000],
                "max_confidence":   np.round(np.arange(0.75, 0.96, 0.05), 2).tolist(),   # [0.75, 0.80, 0.85, 0.90, 0.95]
            },

            "contra_outcome_silence": {
                # Reduced around frequent winners: gap=20, silence=3.0, notional=500/1100,
                # and max_contra_age=45 are common; keep wider tails for robustness.
                "min_gap_samples":         [10, 16, 20],
                "silence_threshold":       [3.0, 4.5, 6.0, 7.5],
                "min_notional":            [500, 1100, 2500],
                "max_contra_age_minutes":  [45, 90, 120, 240],
                "max_confidence":          [0.05, 0.10, 0.25, 0.35, 0.40],
            },
            # 3*4*3*4*5 = 720

            "alert_threshold": {
                # Fine sweep around [0.44, 0.62] plus tail coverage
                "value": _cat(
                    np.round(np.arange(0.40, 0.71, 0.02), 2),   # [0.40, 0.42, ..., 0.70]
                    [0.44, 0.46, 0.48, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61],             # extra probe points near best
                ),
            },
        }

    @staticmethod
    def get_detector_groups_experimental() -> Dict[str, Dict[str, List]]:
        """Experimental grid retained for follow-up runs."""
        return {
            "volume_anomaly": {
                # 1h catches immediate bursts; 2h is a short session; 4h is a
                # half-day liquidity regime; 12h covers daily drift without
                # giving very old activity equal weight.
                "lookback_window_hours":    [1, 2, 4, 12],
                # One trade is meaningful in thin markets; 2 confirms a small
                # local baseline; 5 asks for a stable enough sample.
                "min_trades_for_baseline":  [1, 2, 5],
                # 3.0 is a classic anomaly cutoff; 3.4 preserves the old
                # winner; 4.0+ asks for increasingly exceptional flow.
                "z_score_threshold":        [2.6, 3.0, 3.4, 4.0, 4.8],
                # Dust / small visible trade / normal retail / serious order.
                "min_absolute_notional":    [20, 50, 100, 500],
                # Volume alone is noisy, so keep mostly weak corroborating caps.
                "max_confidence":           [0.02, 0.05, 0.10, 0.20],
            },
            # 4*3*5*4*4 = 960

            "probability_impact": {
                # 1-2 points can matter in efficient markets; 3-5 points are
                # visibly market-moving.
                "min_delta_prob":       [0.010, 0.015, 0.020, 0.030, 0.050],
                # Log-odds keeps the same idea away from the 50/50 center; 0.5
                # is retained as the old winner's meaningful impact threshold.
                "min_delta_log_odds":   [0.25, 0.35, 0.50, 0.75, 1.00],
                # Retail-plus through institutional-sized individual trades.
                "min_notional":         [500, 750, 1000, 2000, 4000],
                # Impact can be more informative than volume, but should still
                # need corroboration before dominating the total score.
                "max_confidence":       [0.10, 0.25, 0.35, 0.50],
            },
            # 5*5*5*4 = 500

            "accumulation_detector": {
                # Position-building bands: small exploratory, meaningful, large,
                # and very large accumulated exposure.
                "min_accumulation_usdc":        [1000, 2000, 5000, 10000, 20000, 40000],
                # 0.80 allows mostly-one-sided flow; 0.98/0.99 require near-pure
                # one-way accumulation.
                "min_directional_ratio":        [0.80, 0.90, 0.95, 0.98, 0.99],
                # Accumulation can be a broad behaviour, so include tiny caps
                # for weak context and moderate caps when it is very clean.
                "max_confidence":               [0.02, 0.05, 0.10, 0.20, 0.40],
                # Outcome concentration separates hedged activity from an
                # explicit directional bet.
                "min_outcome_concentration":    [0.80, 0.90, 0.95, 0.98, 0.99],
            },
            # 6*5*5*5 = 750

            "extreme_position": {
                # Tail exposure is naturally described by percentile bands;
                # 0.20 is retained as the old winner and a readable "top fifth"
                # threshold.
                "tail_threshold":   [0.08, 0.10, 0.14, 0.18, 0.20, 0.22, 0.26, 0.30],
                # From small conviction trades through large capital-at-risk.
                "min_notional":     [100, 250, 500, 750, 1000, 1500, 2500, 4000],
                # Tail bets range from weak evidence to a possible dominant
                # signal, depending on how extreme and well-sized they are.
                "max_confidence":   [0.05, 0.15, 0.30, 0.40, 0.60, 0.75, 0.90, 1.00],
            },
            # 8*8*8 = 512

            "contra_outcome_silence": {
                # A silence gap of 5 can catch fast withdrawals; 10 is a clean
                # short gap; 20 asks for a more persistent absence.
                "min_gap_samples":         [5, 10, 20],
                # 3.0 is moderate silence; 4.5 retains the old winner; 6.0/8.0
                # require increasingly unusual lack of contra-side activity.
                "silence_threshold":       [3.0, 4.5, 6.0, 8.0],
                # Ignore tiny contra-side gaps; keep the old 2500 serious-trade
                # threshold and one larger institutional probe.
                "min_notional":            [500, 1000, 2500, 5000],
                # Immediate, short-lived, two-hour, and half-day silence regimes.
                "max_contra_age_minutes":  [15, 45, 120, 240],
                # Silence is corroborating evidence, not usually enough alone.
                "max_confidence":          [0.05, 0.10, 0.20, 0.40],
            },
            # 3*4*4*4*4 = 768

            "alert_threshold": {
                # Focus on plausible deployable thresholds, with extra probes
                # around the old 0.64 basin and wider coverage up to high
                # precision / low recall settings.
                "value": [
                    0.35, 0.40, 0.45, 0.50, 0.55,
                    0.60, 0.62, 0.64, 0.66, 0.68,
                    0.70, 0.75, 0.80, 0.85, 0.90,
                ],
            },
            # 15
        }

    @staticmethod
    def get_detector_groups() -> Dict[str, Dict[str, List]]:
        """Active wide-coverage grid."""
        return {
            "volume_anomaly": {
                "lookback_window_hours":    [1, 2, 4, 8, 24, 48],
                # keep 1 to allow early-market detection, and
                # probe the 3 and 8 bands as natural round numbers.
                "min_trades_for_baseline":  [1, 3, 8],
                # Extend lower bound to 2.5 — correct windowing produces tighter
                # baselines that make the z-distribution noisier, so moderate
                # thresholds may now be worth exploring.
                "z_score_threshold":        [2.5, 3.0, 3.5, 4.0, 5.0],
                # Compress to three representative bands (dust, visible, serious)
                # to offset the extra lookback values.
                "min_absolute_notional":    [50, 200, 500],
                # Unchanged — volume anomaly should remain a weak corroborator.
                "max_confidence":           [0.02, 0.05, 0.10],
            },

            "probability_impact": {
                # Add a very sensitive lower bound (0.005 ≈ half a cent); price
                # impact is now correctly time-anchored and may be meaningful at
                # finer granularity than before.
                "min_delta_prob":       [0.005, 0.010, 0.020, 0.035, 0.055],
                # Widen both tails of the log-odds range; 0.20 captures near-50/50
                # moves, 0.90 captures strongly directional shifts.
                "min_delta_log_odds":   [0.20, 0.35, 0.50, 0.70, 0.90],
                # Add a smaller threshold (250) and remove two intermediate values
                # to keep count similar; extreme notionals below 250 are dust.
                "min_notional":         [250, 600, 1500, 3000],
                # Extend ceiling to 0.55 — probability impact can plausibly be a
                # moderate primary signal rather than pure corroboration.
                "max_confidence":       [0.10, 0.25, 0.40, 0.55],
            },

            "accumulation_detector": {
                # Extend floor to 500 USDC; position-building by an informed actor
                # in a low-volume market may start here. Extend ceiling to 20 000.
                "min_accumulation_usdc":     [500, 1500, 3500, 8000, 20000],
                # Add 0.83 as a lower bound; strict one-directionality (≥0.98)
                # risks missing wallets that hedge a small fraction.
                "min_directional_ratio":     [0.83, 0.88, 0.92, 0.95, 0.97, 0.99],
                # Unchanged shape; accumulation should remain primarily
                # corroborating at lower levels.
                "max_confidence":            [0.02, 0.05, 0.10, 0.18],
                # Widen floor to 0.78; a wallet can be somewhat diversified and
                # still be building a meaningful directional position.
                "min_outcome_concentration": [0.78, 0.85, 0.91, 0.96, 0.99],
            },

            "extreme_position": {
                # Widen both tails: 0.08 explores very tight tail bets; 0.33 is
                # a softer definition that captures moderate-probability trades.
                "tail_threshold":   [0.08, 0.12, 0.15, 0.18, 0.22, 0.27, 0.33],
                # Add a lower floor (150) and a higher ceiling (6000) to expand
                # the range of capital commitment explored.
                "min_notional":     [150, 400, 800, 1500, 3000, 6000],
                # Extend ceiling to 0.95; tail bets are one of the most
                # theory-grounded signals and can carry high confidence alone.
                "max_confidence":   [0.20, 0.40, 0.60, 0.80, 0.95],
            },

            "contra_outcome_silence": {
                # Keep three representative gap sample thresholds; too few risks
                # a spurious baseline, too many requires a deeply liquid market.
                "min_gap_samples":        [3, 8, 20],
                # Lower bound moved to 2.0; correct timestamps mean silence ratios
                # now reflect real elapsed time, so moderate thresholds are
                # genuinely different from high ones.
                "silence_threshold":      [2.0, 3.5, 5.5, 8.0],
                # Compress to three notional bands to pay for the time expansion.
                "min_notional":           [500, 1500, 4000],
                # This is the key temporal axis. Add short (10m) and long (480m)
                # extremes; both were previously unmapped because the underlying
                # timestamps were in the wrong unit.
                "max_contra_age_minutes": [10, 30, 90, 240, 480],
                # Unchanged; silence is corroborating evidence.
                "max_confidence":         [0.05, 0.12, 0.25, 0.40],
            },

            "alert_threshold": {
                # Add a lower bound (0.45) to probe whether looser thresholds
                # improve recall, and extend ceiling to 0.90 for precision runs.
                "value": [
                    0.45, 0.50, 0.54, 0.57,
                    0.60, 0.62, 0.64, 0.66,
                    0.68, 0.70, 0.73, 0.76,
                    0.80, 0.85, 0.90,
                ],
            },
        }

    @staticmethod
    def get_baseline_config() -> Dict:
        """Returns current production config as baseline."""
        from config import CONFIG

        return {
            "detectors": {
                "volume_anomaly":       CONFIG.get("volume_anomaly", {}).copy(),
                "probability_impact":   CONFIG.get("probability_impact", {}).copy(),
                "accumulation_detector": CONFIG.get("accumulation_detector", {}).copy(),
                "extreme_position":     CONFIG.get("extreme_position", {}).copy(),
                "contra_outcome_silence": CONFIG.get("contra_outcome_silence", {}).copy(),
                "recidivism_detector":  CONFIG.get("recidivism_detector", {}).copy(),
            },
            "alert_threshold": float(CONFIG.get("alert_threshold", 0.5)),
        }

    @staticmethod
    def generate_configs_for_detector(
        detector_name: str,
        base_config: Dict,
    ) -> List[Tuple[Dict, Dict]]:
        detector_groups = ParameterGrid.get_detector_groups()

        if detector_name not in detector_groups:
            raise ValueError(f"Unknown detector: {detector_name}")

        param_grid = detector_groups[detector_name]

        if detector_name == "alert_threshold":
            configs = []
            for value in param_grid["value"]:
                config = deepcopy(base_config)
                config["alert_threshold"] = value
                configs.append((config, {"alert_threshold": value}))
            return configs

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        configs = []
        for combo in product(*param_values):
            config = deepcopy(base_config)
            if detector_name not in config["detectors"]:
                config["detectors"][detector_name] = {}

            param_dict = {}
            for param_name, value in zip(param_names, combo):
                config["detectors"][detector_name][param_name] = value
                param_dict[param_name] = value

            configs.append((config, param_dict))

        return configs

    @staticmethod
    def count_configs_per_detector() -> Dict[str, int]:
        detector_groups = ParameterGrid._materialize_detector_groups(
            ParameterGrid.get_detector_groups()
        )
        counts = {}

        for detector_name, param_grid in detector_groups.items():
            if detector_name == "alert_threshold":
                counts[detector_name] = len(param_grid["value"])
            else:
                count = 1
                for values in param_grid.values():
                    count *= len(values)
                counts[detector_name] = count

        return counts

    @staticmethod
    def print_search_space_summary():
        detector_groups = ParameterGrid._materialize_detector_groups(
            ParameterGrid.get_detector_groups()
        )
        counts = ParameterGrid.count_configs_per_detector()

        print("\n" + "=" * 80)
        print("COORDINATE DESCENT SEARCH SPACE")
        print("=" * 80)

        total_configs = 0
        for detector_name in sorted(detector_groups.keys()):
            param_grid = detector_groups[detector_name]
            n_configs = counts[detector_name]
            print(f"\n{detector_name}:")
            print(f"  Parameters:     {len(param_grid)}")
            print(f"  Configurations: {n_configs:,}")
            for param, values in param_grid.items():
                lo, hi = min(values), max(values)
                print(f"    • {param}: {len(values)} values  [{lo} … {hi}]")
            total_configs += n_configs

        full_grid = ParameterGrid._calculate_full_grid_size()
        reduction = full_grid / total_configs if total_configs else 0.0

        print("\n" + "=" * 80)
        print("TOTALS:")
        print(f"  Total detectors:                  {len(detector_groups)}")
        print(f"  Total configs (coordinate descent): {total_configs:,}")
        print(f"  Total configs (full joint grid):    {full_grid:,}")
        print(f"  Reduction factor:                   {reduction:,.0f}x")
        print("=" * 80)

    @staticmethod
    def _calculate_full_grid_size() -> int:
        detector_groups = ParameterGrid._materialize_detector_groups(
            ParameterGrid.get_detector_groups()
        )
        size = 1
        for detector_name, param_grid in detector_groups.items():
            if detector_name == "alert_threshold":
                size *= len(param_grid["value"])
            else:
                for values in param_grid.values():
                    size *= len(values)
        return size

    @staticmethod
    def sample_random_config(rng: Optional[random.Random] = None) -> Dict:
        """
        Sample a single config uniformly at random from the full joint parameter space.
        """
        if rng is None:
            rng = random

        detector_groups = ParameterGrid._materialize_detector_groups(
            ParameterGrid.get_detector_groups()
        )
        config = deepcopy(ParameterGrid.get_baseline_config())

        for group_name, param_grid in detector_groups.items():
            if group_name == "alert_threshold":
                config["alert_threshold"] = rng.choice(param_grid["value"])
            else:
                if group_name not in config["detectors"]:
                    config["detectors"][group_name] = {}
                for param_name, values in param_grid.items():
                    config["detectors"][group_name][param_name] = rng.choice(values)

        return config

    @staticmethod
    def perturb_baseline(
        rng: Optional[random.Random] = None,
        perturb_prob: float = 0.3,
        base_config: Optional[Dict] = None,
    ) -> Dict:
        """Perturb a baseline config for multi-start optimisation."""
        if rng is None:
            rng = random
        if not (0.0 <= float(perturb_prob) <= 1.0):
            raise ValueError(f"perturb_prob must be in [0, 1]; got {perturb_prob}")

        base = deepcopy(base_config) if base_config is not None else ParameterGrid.get_baseline_config()
        if "detectors" not in base:
            base["detectors"] = {}

        detector_groups = ParameterGrid._materialize_detector_groups(
            ParameterGrid.get_detector_groups()
        )

        for group_name, param_grid in detector_groups.items():
            if group_name == "alert_threshold":
                if rng.random() < perturb_prob:
                    base["alert_threshold"] = rng.choice(param_grid["value"])
                continue
            if group_name not in base["detectors"]:
                base["detectors"][group_name] = {}
            for param_name, values in param_grid.items():
                if rng.random() < perturb_prob:
                    base["detectors"][group_name][param_name] = rng.choice(values)

        return base

    @staticmethod
    def print_expanded_detector_groups() -> None:
        detector_groups = ParameterGrid._materialize_detector_groups(
            ParameterGrid.get_detector_groups()
        )

        print("\n" + "=" * 80)
        print("CURRENT DETECTOR GROUPS (MATERIALIZED)")
        print("=" * 80)

        for detector_name, param_grid in detector_groups.items():
            print(f"\n{detector_name}:")
            for param_name, values in param_grid.items():
                print(f"  {param_name} ({len(values)}): {values}")

if __name__ == "__main__":
    ParameterGrid.print_expanded_detector_groups()
    ParameterGrid.print_search_space_summary()

    print("\n" + "=" * 80)
    print("SAMPLE: VolumeAnomaly configs (first 3)")
    print("=" * 80)
    try:
        baseline = ParameterGrid.get_baseline_config()
        configs = ParameterGrid.generate_configs_for_detector("volume_anomaly", baseline)
        for i, (_, params) in enumerate(configs[:3]):
            print(f"\nConfig {i + 1}: {params}")
    except ModuleNotFoundError as exc:
        print(f"\nSkipping sample config preview: {exc}")
