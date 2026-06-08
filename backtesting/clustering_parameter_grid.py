"""
Parameter grid for clustering optimization (bucket-projection version).

Defines baseline config and variations for coordinate descent.
Uses bucket_size instead of max_time_window for graph construction.
"""

import copy
from typing import Dict, List, Tuple


class ClusteringParameterGrid:
    """
    Parameter grid for clustering optimization.
    Similar to ParameterGrid but for clustering + boost parameters.
    """

    @staticmethod
    def get_baseline_config() -> Dict:
        """
        Return baseline clustering configuration.
        """
        return {
            # Bucket graph construction parameters
            "bucket_size": 300,             # 5 minutes — trades within same bucket get edges
            "same_direction_mult": 2.0,     # 2x weight for same direction
            "size_normalizer": 10000,       # $10k = 1.0 size multiplier
            "max_size_mult": 5.0,           # Cap size effect at 5x
            "cross_outcome_penalty": 0.1,   # Reduce weight for different outcomes

            # Clustering algorithm parameters (unchanged)
            "k_core": 2,                    # K-core decomposition threshold
            "min_edge_weight": 0.5,         # Minimum edge weight to keep in graph

            # Boost parameters (unchanged)
            "boost": {
                "max_boost_factor": 2.0,
                "size_weight": 0.4,
                "density_weight": 0.2,
                "ownership_boost": 0.4,
                "size_normalizer": 50.0,
            },
        }

    @staticmethod
    def get_no_clustering_config() -> Dict:
        """
        Return config that effectively disables clustering boost.
        Used as baseline comparison.
        """
        config = ClusteringParameterGrid.get_baseline_config()
        config["boost"]["max_boost_factor"] = 1.0
        config["boost"]["size_weight"] = 0.0
        config["boost"]["density_weight"] = 0.0
        config["boost"]["ownership_boost"] = 0.0
        return config

    @staticmethod
    def get_parameter_groups() -> Dict[str, List[str]]:
        """
        Group parameters for coordinate descent optimization.
        """
        return {
            # Fastest: just changes boost calculation (no graph rebuild)
            "boost_magnitude": ["boost.max_boost_factor"],
            "boost_weights": ["boost.size_weight", "boost.density_weight", "boost.ownership_boost"],
            "boost_normalizer": ["boost.size_normalizer"],

            # Medium: requires recomputing clusters (cheap)
            "clustering": ["k_core", "min_edge_weight"],

            # Graph rebuild required — but now fast with buckets
            "time_window": ["bucket_size"],
            "size": ["size_normalizer", "max_size_mult"],
        }

    @staticmethod
    def get_variations(param_name: str) -> List:
        variations = {
            # Bucket size - finer resolution around 60-600, extended
            "bucket_size": [180, 240, 300, 360, 480],

            # Size normalizer - wider range
            "size_normalizer": [5000, 8000, 10000, 15000],

            "max_size_mult": [3.0, 5.0, 7.0],

            # Clustering — EXTENDED min_edge_weight past 2.0
            "k_core": [2, 3],

            "min_edge_weight": [0.3, 0.6, 1.0, 1.5, 2.5, 4.0],

            # Boost magnitude — finer at low end
            "boost.max_boost_factor": [1.5, 2.0, 2.5],

            # Boost weights — finer + extended
            "boost.size_weight": [0.3, 0.4, 0.5],

            "boost.density_weight": [0.15, 0.20, 0.25],
            "boost.ownership_boost": [0.1, 0.2, 0.3, 0.4, 0.6],
            # Boost normalizer — wider range
            "boost.size_normalizer": [30.0, 50.0, 70.0],

            # Direction (unused in bucket but kept for completeness)
            "same_direction_mult": [
                1.0, 1.5, 2.0, 2.5, 3.0,
            ],
            "cross_outcome_penalty": [
                0.0, 0.05, 0.1, 0.2, 0.5,
            ],
        }
        return variations.get(param_name, [])

    @staticmethod
    def get_variations_sparse_return(param_name: str) -> List:
        """
        Sparse return-oriented clustering profile.
        """
        variations = {
            # Round five-minute bucket plus one shorter and one longer cadence.
            "bucket_size": [180, 300, 480],

            # Keep normalizer near the existing baseline; this avoids making
            # cluster score mostly a trade-size proxy.
            "size_normalizer": [8000, 10000, 12000],
            "max_size_mult": [3.0, 5.0, 7.0],

            # k=2 captures small coordinated groups; k=3 requires a stricter
            # core without making clusters vanish.
            "k_core": [2, 3],
            "min_edge_weight": [0.5, 0.8, 1.0, 1.2, 2.0],

            # Boosts should corroborate detector evidence, not dominate it.
            "boost.max_boost_factor": [1.5, 2.0],
            "boost.size_weight": [0.3, 0.4, 0.5],
            "boost.density_weight": [0.15, 0.20, 0.25],
            "boost.ownership_boost": [0.2, 0.4, 0.6],
            "boost.size_normalizer": [30.0, 50.0, 70.0],

            # Direction/outcome are unused by the bucket graph groups above but
            # retained for callers that inspect the full profile.
            "same_direction_mult": [1.5, 2.0, 2.5],
            "cross_outcome_penalty": [0.05, 0.1, 0.2],
        }
        return variations.get(param_name, [])

    @staticmethod
    def generate_configs_for_param_group_sparse_return(
        param_group_name: str,
        base_config: Dict,
    ) -> List[Tuple[Dict, Dict]]:
        """
        Generate clustering configs using get_variations_sparse_return.
        """
        groups = ClusteringParameterGrid.get_parameter_groups()

        if param_group_name not in groups:
            raise ValueError(f"Unknown parameter group: {param_group_name}")

        param_names = groups[param_group_name]
        configs = []

        for param_name in param_names:
            variations = ClusteringParameterGrid.get_variations_sparse_return(param_name)

            for value in variations:
                config = copy.deepcopy(base_config)

                if param_name.startswith("boost."):
                    boost_param = param_name.split(".")[1]
                    config["boost"][boost_param] = value
                else:
                    config[param_name] = value

                param_dict = {param_name: value}
                configs.append((config, param_dict))

        return configs

    @staticmethod
    def generate_configs_for_param_group(
        param_group_name: str,
        base_config: Dict,
    ) -> List[Tuple[Dict, Dict]]:
        """
        Generate configs by varying all parameters in a group.
        """
        groups = ClusteringParameterGrid.get_parameter_groups()

        if param_group_name not in groups:
            raise ValueError(f"Unknown parameter group: {param_group_name}")

        param_names = groups[param_group_name]
        configs = []

        for param_name in param_names:
            variations = ClusteringParameterGrid.get_variations(param_name)

            for value in variations:
                config = copy.deepcopy(base_config)

                if param_name.startswith("boost."):
                    boost_param = param_name.split(".")[1]
                    config["boost"][boost_param] = value
                else:
                    config[param_name] = value

                param_dict = {param_name: value}
                configs.append((config, param_dict))

        return configs

    @staticmethod
    def print_search_space_summary():
        """Print summary of search space size for all parameter groups."""
        groups = ClusteringParameterGrid.get_parameter_groups()

        print("\nClustering Parameter Search Space (Bucket Projection):")
        print("-" * 60)

        total_configs = 0

        for group_name, param_names in groups.items():
            group_size = 0
            for param_name in param_names:
                variations = ClusteringParameterGrid.get_variations(param_name)
                group_size += len(variations)

            total_configs += group_size
            print(f"{group_name:20s}: {len(param_names):2d} params × variations = {group_size:3d} configs")

        print("-" * 60)
        print(f"{'Total':20s}:                              {total_configs:3d} configs")
        print(f"\nNote: With 2 passes, this is ~{total_configs * 2} configs total")
        print("Note: Graph rebuild groups are now fast (bucket projection)")
