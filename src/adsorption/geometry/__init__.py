"""The calculator-free geometric layer of the adsorption construction.

The subpackage implements the four factors of an adsorption event: the site
``O`` (core set, centre, neighbour shell, direction sphere), the approach
direction ``OA`` (a direction grid plus the steric filter), the molecular
orientation ``AB`` (a rotation about the anchor), and the anchor distance
``d``. Nothing here imports an ASE calculator, so the generated candidates are
independent of the energy engine.

Example:
    >>> import numpy as np
    >>> from ase.build import fcc111
    >>> from adsorption.geometry import admissible_directions, build_site
    >>> slab = fcc111("Cu", size=(2, 2, 2), vacuum=8, orthogonal=True)
    >>> site = build_site(slab, core=[0])
    >>> grid = admissible_directions(site, n=10)
    >>> grid.n_oa <= 10
    True
"""

from __future__ import annotations

from .admissibility import AdmissibleSet, admissible_directions, filter_grid
from .conformers import (
    generate_conformers,
    near_congruent,
    prune_by_energy,
)
from .constants import (
    DELTA_NBR,
    GOLDEN_RATIO,
    MAX_CORE_SIZE,
    SCALE_MULTI,
    SCALE_SINGLE,
    SKIN_MULTI,
    SKIN_SINGLE,
    default_scale,
    default_skin,
)
from .distance import distance_array
from .grids import (
    DirectionGrid,
    FibonacciGrid,
    LatLonGrid,
    SeededRandomGrid,
)
from .mic import displacements_mic, distances_mic, is_periodic
from .placement import Anchor, ResolvedAnchor, place, resolve_anchor
from .rotation import apply_rotation, shortest_arc_rotation, unit_vector
from .rotations import (
    DirectionRollSet,
    FixedRotation,
    RotationSet,
    SO3Set,
    ShortestArcSet,
)
from .shell import first_neighbor_shell, mean_mic, outward_normal
from .site import SiteGeometry, build_site

__all__ = [
    "DELTA_NBR",
    "GOLDEN_RATIO",
    "MAX_CORE_SIZE",
    "SCALE_MULTI",
    "SCALE_SINGLE",
    "SKIN_MULTI",
    "SKIN_SINGLE",
    "AdmissibleSet",
    "Anchor",
    "DirectionGrid",
    "DirectionRollSet",
    "FibonacciGrid",
    "FixedRotation",
    "LatLonGrid",
    "ResolvedAnchor",
    "RotationSet",
    "SO3Set",
    "SeededRandomGrid",
    "SiteGeometry",
    "admissible_directions",
    "apply_rotation",
    "build_site",
    "default_scale",
    "default_skin",
    "displacements_mic",
    "distance_array",
    "distances_mic",
    "filter_grid",
    "generate_conformers",
    "first_neighbor_shell",
    "is_periodic",
    "mean_mic",
    "near_congruent",
    "outward_normal",
    "place",
    "prune_by_energy",
    "resolve_anchor",
    "ShortestArcSet",
    "shortest_arc_rotation",
    "unit_vector",
]
