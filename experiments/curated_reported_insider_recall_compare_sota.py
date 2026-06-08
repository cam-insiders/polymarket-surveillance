"""Compatibility wrapper for curated reported-insider recall plus SOTA baselines.

New code should import from :mod:`experiments.curated_sota_common`.
"""

from experiments.curated_sota_common import *  # noqa: F401,F403
from experiments.curated_sota_common import main


if __name__ == "__main__":
    main()
