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

from adsorption.geometry.constants import (
    DELTA_NBR,
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
from adsorption.geometry.mic import displacements_mic, is_periodic
from adsorption.geometry.placement import (
    Anchor,
    ResolvedAnchor,
    place,
    resolve_anchor,
)
from adsorption.geometry.rotation import (
    apply_quaternion,
    axis_angle_quaternion,
    quaternion_from_scipy,
    scipy_from_quaternion,
    shortest_arc_quaternion,
    unit_vector,
)
from adsorption.geometry.rotations import (
    DirectionRollSet,
    FixedRotation,
    RotationSet,
    ShortestArcSet,
)
from adsorption.geometry.shell import first_neighbor_shell, mean_mic
from adsorption.geometry.superfibonacci import SuperFibonacciSO3
from adsorption.geometry.site import SiteGeometry, build_site

__all__ = [
    "Anchor",
    "DELTA_NBR",
    "DirectionGrid",
    "DirectionRollSet",
    "FibonacciGrid",
    "FixedRotation",
    "LatLonGrid",
    "MAX_CORE_SIZE",
    "ResolvedAnchor",
    "RotationSet",
    "SCALE_MULTI",
    "SCALE_SINGLE",
    "SKIN_MULTI",
    "SKIN_SINGLE",
    "SeededRandomGrid",
    "ShortestArcSet",
    "SiteGeometry",
    "SuperFibonacciSO3",
    "apply_quaternion",
    "axis_angle_quaternion",
    "build_site",
    "default_scale",
    "default_skin",
    "displacements_mic",
    "distance_array",
    "first_neighbor_shell",
    "is_periodic",
    "mean_mic",
    "place",
    "quaternion_from_scipy",
    "resolve_anchor",
    "scipy_from_quaternion",
    "shortest_arc_quaternion",
    "unit_vector",
]
