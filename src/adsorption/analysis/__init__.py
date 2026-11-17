"""Analysis of placements and of relaxed structures.

Nothing in this package evaluates an energy.

:mod:`adsorption.analysis.light`
    Geometry-only diagnostics of the placement layer: resolution of the
    direction and orientation sets, the share of a budget that survives the
    geometric floor, the unique-placement count against the hash settings,
    the bijection of the candidate index and the generation throughput.
:mod:`adsorption.analysis.labels`
    The label of a relaxed structure: intact, dissociated, desorbed or
    reconstructed.
:mod:`adsorption.analysis.basins`
    Which relaxed structures are the same minimum (tolerance-based distance
    fingerprint).
:mod:`adsorption.analysis.criteria`
    Geometric criteria that identify a published configuration (contacts,
    site kind, azimuth).
:mod:`adsorption.analysis.recall`
    Basin recall of relaxation runs and recovery of published
    configurations.
"""

from __future__ import annotations

from adsorption.analysis.basins import cluster_minima, fingerprint
from adsorption.analysis.criteria import (
    adsorbate_azimuth,
    adsorbate_tilt,
    matches_reference,
    nearest_site,
)
from adsorption.analysis.labels import (
    LabelThresholds,
    adsorbate_bonds,
    label_structure,
)
from adsorption.analysis.light import (
    budget_report,
    covering_radius,
    dedup_report,
    direction_grid_report,
    height_uniformity,
    index_report,
    largest_neighbor_angle,
    load_config,
    main,
    run_suite,
    so3_covering_radius,
    so3_quaternions,
    throughput_report,
)
from adsorption.analysis.recall import analyse_runs, load_run, write_report

__all__ = [
    "LabelThresholds",
    "adsorbate_azimuth",
    "adsorbate_tilt",
    "adsorbate_bonds",
    "analyse_runs",
    "budget_report",
    "cluster_minima",
    "covering_radius",
    "dedup_report",
    "direction_grid_report",
    "fingerprint",
    "height_uniformity",
    "index_report",
    "label_structure",
    "largest_neighbor_angle",
    "load_config",
    "load_run",
    "main",
    "matches_reference",
    "nearest_site",
    "run_suite",
    "so3_covering_radius",
    "so3_quaternions",
    "throughput_report",
    "write_report",
]
