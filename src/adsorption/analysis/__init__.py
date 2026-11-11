"""Pre-registered Tier-0 measurements of the placement layer.

The package holds the geometry-only diagnostics that the evaluation protocol
asks for before any energy engine is named. :mod:`adsorption.analysis.light`
measures the resolution of the direction and orientation sets, the share of
the fixed budget that survives the geometric floor, the unique-placement
count against the hash settings, the bijection of the candidate index and the
throughput of the generation tier.
"""

from __future__ import annotations

from adsorption.analysis.light import (
    budget_report,
    covering_radius,
    dedup_report,
    direction_grid_report,
    height_uniformity,
    index_report,
    load_config,
    main,
    run_suite,
    so3_covering_radius,
    so3_quaternions,
    throughput_report,
)

__all__ = [
    "budget_report",
    "covering_radius",
    "dedup_report",
    "direction_grid_report",
    "height_uniformity",
    "index_report",
    "load_config",
    "main",
    "run_suite",
    "so3_covering_radius",
    "so3_quaternions",
    "throughput_report",
]
