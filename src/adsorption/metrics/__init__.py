"""Reusable metrics for the pre-registered evaluation protocol.

The functions are pure and calculator-free. They are consumed by the campaign
drivers under ``bench/``; the package does not ship a benchmark orchestrator.
"""

from __future__ import annotations

from .cascade import (
    CascadeRatios,
    cascade_ratios,
    false_rejection,
    max_false_rejection,
)
from .geometry import (
    accessible_solid_angle,
    admissible_fraction,
    discretization_residual,
    high_resolution_fraction,
)
from .ranking import kendall, regret, spearman, top_k_ids, top_k_recall
from .rotation import (
    matrices_to_quaternions,
    quaternion_geodesic,
    rotation_covering_radius,
    rotation_nearest_neighbor_angles,
)
from .spherical import (
    cap_fraction,
    covering_radius,
    nearest_neighbor_angles,
    pole_bias,
)

__all__ = [
    "CascadeRatios",
    "accessible_solid_angle",
    "admissible_fraction",
    "cap_fraction",
    "cascade_ratios",
    "covering_radius",
    "discretization_residual",
    "false_rejection",
    "high_resolution_fraction",
    "kendall",
    "matrices_to_quaternions",
    "max_false_rejection",
    "nearest_neighbor_angles",
    "pole_bias",
    "quaternion_geodesic",
    "regret",
    "rotation_covering_radius",
    "rotation_nearest_neighbor_angles",
    "spearman",
    "top_k_ids",
    "top_k_recall",
]
