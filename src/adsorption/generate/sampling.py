"""Sampling protocol: approach directions, orientations and distances.

A :class:`Sampling` object is the three-grid protocol of the construction:
``n_oa`` directions on the unit sphere, ``n_rot`` orientations on the rotation
set, and ``n_d`` anchor-centre distances. :meth:`Sampling.bind` resolves the
protocol against one rigid adsorbate on one site and returns a
:class:`BoundAdsorbate`, which is what the generator places.

Two parts of the protocol depend on the site:

* The lower distance bound. With ``d_min=None`` (the default) it is the
  hard-sphere contact distance of :func:`adsorption.geometry.contact_distance`,
  i.e. the anchor-centre distance at which a probe carrying the smallest
  adsorbate Lennard-Jones floor touches the core atoms. A fixed ``d_min`` in
  Angstrom restores the absolute array.
* The direction set. With ``hemisphere=True`` only the grid points on the
  outward side of the site (``u . n > 0`` with ``n`` the site normal) enter the
  enumeration; the realised ``n_oa`` is then about half the request. The
  default keeps the full sphere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from adsorption.generate.orientation import OrientationShape, build_orientation
from adsorption.geometry.contact import contact_distance
from adsorption.geometry.distance import distance_array
from adsorption.geometry.grids import DirectionGrid, FibonacciGrid
from adsorption.geometry.rotations import RotationSet
from adsorption.geometry.site import SiteGeometry

if TYPE_CHECKING:
    from adsorption.generate.adsorbate import RigidAdsorbate

__all__ = [
    "BoundAdsorbate",
    "DEFAULT_D_CONTACT_SCALE",
    "DEFAULT_D_MAX",
    "DEFAULT_D_MAX_RADIUS_SCALE",
    "DEFAULT_D_MIN",
    "DEFAULT_D_MIN_RADIUS_SCALE",
    "DEFAULT_HEMISPHERE",
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

DEFAULT_D_MIN: float | None = None
"""Default lower bound of the anchor-distance array.

``None`` selects the hard-sphere contact distance of the site and adsorbate.
"""

DEFAULT_D_MAX = 3.0
"""Default upper bound of the anchor-distance array, in Angstrom."""

DEFAULT_D_MIN_RADIUS_SCALE = 0.0
"""Default radius coefficient added to the lower distance bound."""

DEFAULT_D_MAX_RADIUS_SCALE = 0.0
"""Default radius coefficient added to the upper distance bound."""

DEFAULT_D_CONTACT_SCALE = 1.0
"""Default coefficient of the pairwise floor used for the contact distance."""

DEFAULT_HEMISPHERE = False
"""Whether the direction set is restricted to the outward hemisphere."""


@dataclass(frozen=True)
class BoundAdsorbate:
    """A rigid adsorbate bound to the three sampling grids.

    Attributes:
        rigid: The rigid adsorbate.
        directions: Unit approach directions, shape ``(n_oa, 3)``.
        orientation: The rotation set.
        distances: Anchor-centre distances, shape ``(n_d,)``.
        direction_indices: Index of every direction in the full grid, shape
            ``(n_oa,)``; the identity when the grid is unfiltered.
    """

    rigid: RigidAdsorbate
    directions: np.ndarray
    orientation: RotationSet
    distances: np.ndarray
    direction_indices: np.ndarray | None = None

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
        if self.direction_indices is None:
            indices = np.arange(len(directions), dtype=int)
        else:
            indices = np.asarray(self.direction_indices, dtype=int).ravel()
            if len(indices) != len(directions):
                raise ValueError(
                    "direction_indices must match the directions, got "
                    f"{len(indices)} and {len(directions)}."
                )
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "direction_indices", indices)

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

    def signature(self) -> Mapping[str, Any]:
        """Return the realised grid sizes of this binding.

        Unlike :meth:`Sampling.signature`, which records the *requested*
        protocol, these are the sizes that actually enter the candidate index
        (a single atom realises ``n_rot = 1`` whatever was requested, and a
        hemisphere binding realises about half the requested directions).

        Returns:
            Mapping[str, Any]: ``n_oa``, ``n_rot``, ``n_d``, the rotation set
            name and the realised distance bounds.
        """
        return {
            "n_oa": self.n_oa,
            "n_rot": self.n_rot,
            "n_d": self.n_d,
            "orientation": getattr(self.orientation, "name", ""),
            "d_min": float(self.distances.min()),
            "d_max": float(self.distances.max()),
        }


@dataclass(frozen=True)
class Sampling:
    """The three-grid sampling protocol.

    Attributes:
        n_oa: Requested number of approach directions (grid size ``N``).
        n_rot: Requested number of orientations; ignored when ``orientation``
            is given.
        n_d: Requested number of distances; ignored when ``distances`` is
            given.
        d_min: Lower bound of the distance array in Angstrom, or ``None`` for
            the hard-sphere contact distance of the site and adsorbate.
        d_max: Upper bound of the distance array, in Angstrom.
        d_min_radius_scale: Coefficient of the adsorbate radius added to
            the lower bound. With a centre-of-mass anchor the lowest adsorbate
            atom lies up to one radius below the anchor, which this term
            compensates.
        d_max_radius_scale: Coefficient of the adsorbate radius added to
            ``d_max``.
        d_contact_scale: Coefficient of the pairwise Lennard-Jones floor used
            by the contact distance; match it to the ``scale`` of the
            geometric screen.
        hemisphere: Whether to keep only the directions on the outward side
            of the site.
        distances: Explicit distance array; overrides the bounds.
        orientation: Explicit rotation set; overrides ``orientation_kind``
            and ``n_rot``.
        orientation_kind: How to build the rotation set.
        grid: Direction grid; the spherical Fibonacci lattice when ``None``.
    """

    n_oa: int = DEFAULT_N_OA
    n_rot: int = DEFAULT_N_ROT
    n_d: int = DEFAULT_N_D
    d_min: float | None = DEFAULT_D_MIN
    d_max: float = DEFAULT_D_MAX
    d_min_radius_scale: float = DEFAULT_D_MIN_RADIUS_SCALE
    d_max_radius_scale: float = DEFAULT_D_MAX_RADIUS_SCALE
    d_contact_scale: float = DEFAULT_D_CONTACT_SCALE
    hemisphere: bool = DEFAULT_HEMISPHERE
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
        for name in ("d_min_radius_scale", "d_max_radius_scale", "d_max"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
        if self.d_min is not None:
            d_min = float(self.d_min)
            if not np.isfinite(d_min) or d_min < 0.0:
                raise ValueError(
                    f"d_min must be non-negative or None, got {self.d_min}."
                )
            if self.distances is None and float(self.d_max) < d_min:
                raise ValueError(
                    "d_max must be at least d_min, got "
                    f"{self.d_max} < {self.d_min}."
                )
        scale = float(self.d_contact_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"d_contact_scale must be positive, got {self.d_contact_scale}."
            )
        if self.distances is not None:
            values = np.asarray(self.distances, dtype=float).ravel()
            if len(values) == 0:
                raise ValueError("distances must not be empty.")
            if not np.all(np.isfinite(values)):
                raise ValueError("distances must be finite.")
            object.__setattr__(self, "distances", values)

    @property
    def site_dependent(self) -> bool:
        """Return whether a binding depends on the site geometry.

        The contact-based lower bound and the hemisphere restriction both read
        the site, so a binding cannot be shared across sites when either is
        active.
        """
        return bool(self.hemisphere) or (
            self.distances is None and self.d_min is None
        )

    def _directions(
        self,
        site: SiteGeometry | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the direction set and the grid indices it keeps."""
        grid: DirectionGrid = (
            FibonacciGrid() if self.grid is None else self.grid
        )
        points = np.asarray(grid.points(int(self.n_oa)), dtype=float)
        indices = np.arange(len(points), dtype=int)
        if not self.hemisphere:
            return points, indices
        if site is None:
            raise ValueError(
                "hemisphere=True needs a site geometry to define the normal."
            )
        keep = points @ np.asarray(site.normal, dtype=float) > 0.0
        if not np.any(keep):
            raise ValueError(
                "No direction of the grid lies in the outward hemisphere."
            )
        return points[keep], indices[keep]

    def _distances(
        self,
        rigid: RigidAdsorbate,
        site: SiteGeometry | None,
    ) -> np.ndarray:
        """Return the anchor-distance array of one binding."""
        if self.distances is not None:
            return np.asarray(self.distances, dtype=float)
        if self.d_min is None:
            if site is None:
                raise ValueError(
                    "d_min=None needs a site geometry to define the contact "
                    "distance; pass d_min explicitly for a site-free binding."
                )
            d_min = contact_distance(
                site,
                rigid.atoms.numbers,
                scale=float(self.d_contact_scale),
            )
        else:
            d_min = float(self.d_min)
        d_max = float(self.d_max)
        if (
            float(self.d_min_radius_scale) != 0.0
            or float(self.d_max_radius_scale) != 0.0
        ):
            center = np.asarray(rigid.atoms.get_center_of_mass(), dtype=float)
            offsets = np.asarray(rigid.atoms.positions, dtype=float) - center
            radius = float(np.linalg.norm(offsets, axis=1).max())
            d_min += float(self.d_min_radius_scale) * radius
            d_max += float(self.d_max_radius_scale) * radius
        if d_max < d_min:
            raise ValueError(
                f"The distance array is empty: the lower bound {d_min:.3f} A "
                f"exceeds d_max={d_max:.3f} A. Raise d_max or set d_min."
            )
        return distance_array(d_min, d_max, n=int(self.n_d))

    def bind(
        self,
        rigid: RigidAdsorbate,
        site: SiteGeometry | None = None,
    ) -> BoundAdsorbate:
        """Bind the protocol to one rigid adsorbate on one site.

        Args:
            rigid: The rigid adsorbate.
            site: The site geometry. Required when the protocol is site
                dependent (see :attr:`site_dependent`).

        Returns:
            BoundAdsorbate: The adsorbate with its three grids.

        Raises:
            ValueError: If the rotation set or the distances are invalid, or a
                site-dependent protocol is bound without a site.
        """
        directions, indices = self._directions(site)
        if self.orientation is None:
            _, orientation = build_orientation(
                self.orientation_kind,
                rigid.atoms,
                n=int(self.n_rot),
            )
        else:
            orientation = self.orientation
        return BoundAdsorbate(
            rigid=rigid,
            directions=directions,
            orientation=orientation,
            distances=self._distances(rigid, site),
            direction_indices=indices,
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
        """Return a JSON-serialisable description of the requested protocol.

        ``n_rot`` is the requested orientation count; the realised count
        depends on the adsorbate (see :meth:`BoundAdsorbate.signature`).
        ``d_min`` is ``"contact"`` when the lower bound is the contact
        distance, and ``n_oa`` is the grid size, of which the hemisphere
        binding keeps a site-dependent subset.
        """
        return {
            "n_oa": int(self.n_oa),
            "hemisphere": bool(self.hemisphere),
            "n_rot": (
                int(self.n_rot)
                if self.orientation is None
                else int(self.orientation.count())
            ),
            "n_rot_is_requested": self.orientation is None,
            "n_d": (
                int(self.n_d)
                if self.distances is None
                else int(len(np.asarray(self.distances, dtype=float).ravel()))
            ),
            "d_min": "contact" if self.d_min is None else float(self.d_min),
            "d_max": float(self.d_max),
            "d_min_radius_scale": float(self.d_min_radius_scale),
            "d_max_radius_scale": float(self.d_max_radius_scale),
            "d_contact_scale": float(self.d_contact_scale),
            "orientation_kind": self.orientation_kind,
            "grid": getattr(self.grid, "name", "fibonacci"),
        }
