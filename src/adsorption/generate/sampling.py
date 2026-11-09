"""Sampling protocol: approach directions, orientations and distances.

A :class:`Sampling` object is the three-grid protocol of the construction:
``n_oa`` directions on the unit sphere, ``n_rot`` orientations on the rotation
set, and ``n_d`` anchor-centre distances. :meth:`Sampling.bind` resolves the
protocol against one rigid adsorbate and returns a :class:`BoundAdsorbate`,
which is what the generator places.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from adsorption.generate.orientation import OrientationShape, build_orientation
from adsorption.geometry.distance import distance_array
from adsorption.geometry.grids import DirectionGrid, FibonacciGrid
from adsorption.geometry.rotations import RotationSet

if TYPE_CHECKING:
    from adsorption.generate.adsorbate import RigidAdsorbate

__all__ = [
    "BoundAdsorbate",
    "DEFAULT_D_MAX",
    "DEFAULT_D_MAX_RADIUS_SCALE",
    "DEFAULT_D_MIN",
    "DEFAULT_D_MIN_RADIUS_SCALE",
    "DEFAULT_N_D",
    "DEFAULT_N_OA",
    "DEFAULT_N_ROT",
    "Sampling",
]

DEFAULT_N_OA = 200
"""Default number of approach directions (the protocol value of ``N``)."""

DEFAULT_N_ROT = 200
"""Default number of orientations (the protocol value of ``N_AB``)."""

DEFAULT_N_D = 12
"""Default length of the anchor-distance array (the protocol value of ``M``)."""

DEFAULT_D_MIN = 0.8
"""Default lower bound of the anchor-distance array, in Angstrom."""

DEFAULT_D_MAX = 3.0
"""Default upper bound of the anchor-distance array, in Angstrom."""

DEFAULT_D_MIN_RADIUS_SCALE = 0.0
"""Default radius coefficient added to the lower distance bound."""

DEFAULT_D_MAX_RADIUS_SCALE = 0.0
"""Default radius coefficient added to the upper distance bound."""


@dataclass(frozen=True)
class BoundAdsorbate:
    """A rigid adsorbate bound to the three sampling grids.

    Attributes:
        rigid: The rigid adsorbate.
        directions: Unit approach directions, shape ``(n_oa, 3)``.
        orientation: The rotation set.
        distances: Anchor-centre distances, shape ``(n_d,)``.
    """

    rigid: RigidAdsorbate
    directions: np.ndarray
    orientation: RotationSet
    distances: np.ndarray

    def __post_init__(self) -> None:
        """Validate the grid shapes and sizes."""
        directions = np.asarray(self.directions, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(
                f"directions must be (n, 3), got {directions.shape}."
            )
        if len(directions) == 0:
            raise ValueError("directions must not be empty.")
        if not np.all(np.isfinite(directions)):
            raise ValueError("directions must be finite.")
        distances = np.asarray(self.distances, dtype=float).ravel()
        if len(distances) == 0:
            raise ValueError("distances must not be empty.")
        if not np.all(np.isfinite(distances)):
            raise ValueError("distances must be finite.")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "distances", distances)

    @property
    def n_oa(self) -> int:
        """Return the number of realised approach directions."""
        return int(len(self.directions))

    @property
    def n_rot(self) -> int:
        """Return the number of realised orientations."""
        return int(self.orientation.count())

    @property
    def n_d(self) -> int:
        """Return the number of realised distances."""
        return int(len(self.distances))


@dataclass(frozen=True)
class Sampling:
    """The three-grid sampling protocol.

    Attributes:
        n_oa: Requested number of approach directions.
        n_rot: Requested number of orientations; ignored when ``orientation``
            is given.
        n_d: Requested number of distances; ignored when ``distances`` is
            given.
        d_min: Lower bound of the distance array, in Angstrom.
        d_max: Upper bound of the distance array, in Angstrom.
        d_min_radius_scale: Coefficient of the adsorbate radius added to
            ``d_min``.
        d_max_radius_scale: Coefficient of the adsorbate radius added to
            ``d_max``.
        distances: Explicit distance array; overrides the bounds.
        orientation: Explicit rotation set; overrides ``orientation_kind``
            and ``n_rot``.
        orientation_kind: How to build the rotation set.
        grid: Direction grid; the spherical Fibonacci lattice when ``None``.
    """

    n_oa: int = DEFAULT_N_OA
    n_rot: int = DEFAULT_N_ROT
    n_d: int = DEFAULT_N_D
    d_min: float = DEFAULT_D_MIN
    d_max: float = DEFAULT_D_MAX
    d_min_radius_scale: float = DEFAULT_D_MIN_RADIUS_SCALE
    d_max_radius_scale: float = DEFAULT_D_MAX_RADIUS_SCALE
    distances: Sequence[float] | np.ndarray | None = None
    orientation: RotationSet | None = None
    orientation_kind: OrientationShape = "auto"
    grid: DirectionGrid | None = None

    def __post_init__(self) -> None:
        """Validate the sizes and the scalar parameters."""
        if int(self.n_oa) < 1:
            raise ValueError(f"n_oa must be positive, got {self.n_oa}.")
        if self.orientation is None and int(self.n_rot) < 1:
            raise ValueError(f"n_rot must be positive, got {self.n_rot}.")
        if self.orientation is not None and int(self.orientation.count()) < 1:
            raise ValueError("orientation must contain at least one rotation.")
        for name in ("d_min_radius_scale", "d_max_radius_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
        if self.distances is not None:
            values = np.asarray(self.distances, dtype=float).ravel()
            if len(values) == 0:
                raise ValueError("distances must not be empty.")
            if not np.all(np.isfinite(values)):
                raise ValueError("distances must be finite.")
            object.__setattr__(self, "distances", values)

    def bind(self, rigid: RigidAdsorbate) -> BoundAdsorbate:
        """Bind the protocol to one rigid adsorbate.

        Args:
            rigid: The rigid adsorbate.

        Returns:
            BoundAdsorbate: The adsorbate with its three grids.

        Raises:
            ValueError: If the rotation set or the distances are invalid.
        """
        grid: DirectionGrid = (
            FibonacciGrid() if self.grid is None else self.grid
        )
        directions = np.asarray(grid.points(int(self.n_oa)), dtype=float)
        if self.orientation is None:
            _, orientation = build_orientation(
                self.orientation_kind,
                rigid.atoms,
                n=int(self.n_rot),
            )
        else:
            orientation = self.orientation
        if self.distances is None:
            d_min = float(self.d_min)
            d_max = float(self.d_max)
            if (
                float(self.d_min_radius_scale) != 0.0
                or float(self.d_max_radius_scale) != 0.0
            ):
                center = np.asarray(
                    rigid.atoms.get_center_of_mass(), dtype=float
                )
                offsets = (
                    np.asarray(rigid.atoms.positions, dtype=float) - center
                )
                radius = float(np.linalg.norm(offsets, axis=1).max())
                d_min += float(self.d_min_radius_scale) * radius
                d_max += float(self.d_max_radius_scale) * radius
            distances = distance_array(d_min, d_max, n=int(self.n_d))
        else:
            distances = np.asarray(self.distances, dtype=float)
        return BoundAdsorbate(
            rigid=rigid,
            directions=directions,
            orientation=orientation,
            distances=distances,
        )

    def with_n_oa(self, n_oa: int) -> Sampling:
        """Return a copy with a different number of approach directions.

        Args:
            n_oa: The new number of directions.

        Returns:
            Sampling: The copy.
        """
        return replace(self, n_oa=int(n_oa))

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the protocol."""
        return {
            "n_oa": int(self.n_oa),
            "n_rot": (
                int(self.n_rot)
                if self.orientation is None
                else int(self.orientation.count())
            ),
            "n_d": (
                int(self.n_d)
                if self.distances is None
                else int(len(np.asarray(self.distances, dtype=float).ravel()))
            ),
            "d_min": float(self.d_min),
            "d_max": float(self.d_max),
            "d_min_radius_scale": float(self.d_min_radius_scale),
            "d_max_radius_scale": float(self.d_max_radius_scale),
            "orientation_kind": self.orientation_kind,
            "grid": getattr(self.grid, "name", "fibonacci"),
        }
