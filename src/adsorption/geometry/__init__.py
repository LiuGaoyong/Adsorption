"""The calculator-free geometric layer of the adsorption construction.

The subpackage implements the geometric primitives of an adsorption event:
the site ``O`` (core set, centre, neighbour shell), the approach direction
``OA`` (a direction grid), the molecular orientation (a rotation set, including
the Super-Fibonacci sequence on ``SO(3)``), and the anchor distance ``d``.
Nothing here imports an ASE calculator, so the generated candidates are
independent of the energy engine.

Example:
    >>> from ase.build import fcc111
    >>> from adsorption.geometry import build_site
    >>> slab = fcc111("Cu", size=(2, 2, 2), vacuum=8, orthogonal=True)
    >>> site = build_site(slab, core=[0])
    >>> tuple(site.center.shape)
    (3,)
"""

from __future__ import annotations

from adsorption.utils.conformers import (
    generate_conformers,
    near_congruent,
    prune_by_energy,
)
from adsorption.geometry.constants import (
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
from adsorption.geometry.distance import distance_array
from adsorption.geometry.grids import (
    DirectionGrid,
    FibonacciGrid,
    LatLonGrid,
    SeededRandomGrid,
)
from adsorption.geometry.mic import (
    displacements_mic,
    distances_mic,
    is_periodic,
)
from adsorption.geometry.placement import (
    Anchor,
    ResolvedAnchor,
    place,
    resolve_anchor,
)
from adsorption.geometry.rotation import (
    apply_rotation,
    matrix_to_quaternion,
    quaternion_to_rotation,
    shortest_arc_rotation,
    unit_vector,
)
from adsorption.geometry.rotations import (
    DirectionRollSet,
    FixedRotation,
    RotationSet,
    SO3Set,
    ShortestArcSet,
)
from adsorption.geometry.shell import (
    first_neighbor_shell,
    mean_mic,
    outward_normal,
)
from adsorption.geometry.superfibonacci import SuperFibonacciSO3
from adsorption.geometry.site import SiteGeometry, build_site

__all__ = [
    "DELTA_NBR",
    "GOLDEN_RATIO",
    "MAX_CORE_SIZE",
    "SCALE_MULTI",
    "SCALE_SINGLE",
    "SKIN_MULTI",
    "SKIN_SINGLE",
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
    "apply_rotation",
    "build_site",
    "default_scale",
    "default_skin",
    "displacements_mic",
    "distance_array",
    "distances_mic",
    "generate_conformers",
    "first_neighbor_shell",
    "is_periodic",
    "matrix_to_quaternion",
    "mean_mic",
    "near_congruent",
    "outward_normal",
    "place",
    "prune_by_energy",
    "quaternion_to_rotation",
    "resolve_anchor",
    "ShortestArcSet",
    "SuperFibonacciSO3",
    "shortest_arc_rotation",
    "unit_vector",
]
