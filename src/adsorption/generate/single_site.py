"""Process A for one adsorption site: enumerate ``(OA, AB, d)``.

``SingleSiteGenerator`` owns one adsorption site and one reference adsorbate.
It enumerates the approach direction on a near-uniform spherical point set
(the spherical Fibonacci lattice by default), the molecular orientation on a
rotation set, and the anchor distance on a one-dimensional array. The
candidate index follows the protocol order: direction outermost, orientation
next, distance fastest.

The generator is built from three inputs -- the substrate, the index (or
indices) of the adsorption centre, and the adsorbate molecule. Every remaining
choice is a named parameter with a protocol default.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from ase import Atoms

from adsorption.base import AdsorptionConfigurationGeneratorABC, Trial
from adsorption.generate.orientation import Shape, build_orientation
from adsorption.geometry import (
    Anchor,
    DirectionGrid,
    FibonacciGrid,
    ResolvedAnchor,
    RotationSet,
    build_site,
    distance_array,
    place,
    resolve_anchor,
)

__all__ = ["SingleSiteGenerator"]

DEFAULT_N_OA = 200
"""Default number of approach directions (the protocol value of $N$)."""

DEFAULT_N_ROT = 200
"""Default number of orientations (the protocol value of $N_{AB}$)."""

DEFAULT_N_D = 12
"""Default length of the anchor-distance array (the protocol value of $M$)."""

DEFAULT_D_MIN = 0.8
"""Default lower bound of the anchor-distance array, in Angstrom."""

DEFAULT_D_MAX = 3.0
"""Default upper bound of the anchor-distance array, in Angstrom."""

DEFAULT_D_MIN_RADIUS_SCALE = 0.0
"""Default radius coefficient added to the lower distance bound."""

DEFAULT_D_MAX_RADIUS_SCALE = 0.0
"""Default radius coefficient added to the upper distance bound."""


class SingleSiteGenerator(AdsorptionConfigurationGeneratorABC):
    """Enumerate the trial configurations of one fixed site.

    The site is derived from the adsorption-centre index: the core atom set is
    the given index (one atom for a top site, a pair for a bridge, a triple
    for a hollow), the site centre is its minimum-image mean, and the
    first-neighbour shell is detected from the covalent radii.

    Attributes:
        substrate: The substrate structure.
        core: The core atom indices of the adsorption centre.
        site: The site geometry built from ``core``.
        reference: The reference adsorbate.
        anchor: The resolved anchor in the reference frame.
        orientation: The rotation set.
        distances: The anchor-centre distances, shape ``(n_d,)``.
        n_oa: Number of approach directions realised by the grid.
        n_rot: Number of orientations.
        n_d: Number of distances.
    """

    def __init__(
        self,
        substrate: Atoms,
        core: int | Sequence[int] | np.ndarray,
        reference: Atoms,
        *,
        n_oa: int = DEFAULT_N_OA,
        n_rot: int = DEFAULT_N_ROT,
        n_d: int = DEFAULT_N_D,
        d_min: float = DEFAULT_D_MIN,
        d_max: float = DEFAULT_D_MAX,
        d_min_radius_scale: float = DEFAULT_D_MIN_RADIUS_SCALE,
        d_max_radius_scale: float = DEFAULT_D_MAX_RADIUS_SCALE,
        distances: np.ndarray | Sequence[float] | None = None,
        anchor: Anchor | ResolvedAnchor = "com",
        orientation: RotationSet | None = None,
        orientation_kind: Shape = "auto",
        grid: DirectionGrid | None = None,
        scale: float | None = None,
        skin: float | None = None,
        site_index: int = 0,
        conformer: int = 0,
    ) -> None:
        """Build the site, the adsorbate and the three grids.

        Args:
            substrate: The substrate structure.
            core: Index (or indices) of the adsorption-centre core atoms.
            reference: The reference adsorbate molecule.
            n_oa: Number of approach directions.
            n_rot: Number of molecular orientations.
            n_d: Number of anchor distances.
            d_min: Lower bound of the anchor-distance array, in Angstrom.
            d_max: Upper bound of the anchor-distance array, in Angstrom.
            d_min_radius_scale: Coefficient of the adsorbate radius
                ``R = max_i ||r_i - r_com||`` added to ``d_min``.
            d_max_radius_scale: Coefficient of the adsorbate radius added to
                ``d_max``.
            distances: Explicit anchor-distance array; overrides ``n_d``,
                ``d_min``, ``d_max`` and both radius coefficients.
            anchor: The anchor declaration, or an already resolved anchor.
            orientation: An explicit rotation set; overrides
                ``orientation_kind`` and ``n_rot``.
            orientation_kind: How to build the rotation set when
                ``orientation`` is not given: ``"auto"``, ``"fixed"``,
                ``"s2"`` or ``"so3"``.
            grid: Direction grid. Defaults to the spherical Fibonacci lattice.
            scale: Direction-sphere scale factor of the site.
            skin: Steric margin of the site, in Angstrom.
            site_index: Zero-based index of the site in the parent generator.
            conformer: Zero-based index of the conformer.

        Raises:
            TypeError: If ``substrate`` or ``reference`` is not ``Atoms``.
            ValueError: If a grid, a distance or an orientation is invalid.
        """
        if not isinstance(substrate, Atoms):
            raise TypeError(
                f"substrate must be an ase.Atoms, got {type(substrate)}."
            )
        if not isinstance(reference, Atoms):
            raise TypeError(
                f"reference must be an ase.Atoms, got {type(reference)}."
            )
        if int(n_oa) < 1:
            raise ValueError(f"n_oa must be positive, got {n_oa}.")
        for name, value in (
            ("d_min_radius_scale", d_min_radius_scale),
            ("d_max_radius_scale", d_max_radius_scale),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")

        self.substrate = substrate
        self.reference = reference
        self.site = build_site(substrate, core, scale=scale, skin=skin)
        self.core = np.asarray(self.site.core_indices, dtype=int)
        self.anchor = (
            anchor
            if isinstance(anchor, ResolvedAnchor)
            else resolve_anchor(reference, anchor)
        )
        if orientation is None:
            if int(n_rot) < 1:
                raise ValueError(f"n_rot must be positive, got {n_rot}.")
            _, orientation = build_orientation(
                orientation_kind, reference, n=int(n_rot)
            )
        self.orientation = orientation
        if int(self.orientation.count()) < 1:
            raise ValueError("orientation must contain at least one rotation.")
        if distances is None:
            if d_min_radius_scale != 0.0 or d_max_radius_scale != 0.0:
                com = np.asarray(reference.get_center_of_mass(), dtype=float)
                v = np.asarray(reference.positions, dtype=float) - com
                radius = float(np.linalg.norm(v, axis=1).max())
                d_min = float(d_min) + float(d_min_radius_scale) * radius
                d_max = float(d_max) + float(d_max_radius_scale) * radius
            distances = distance_array(d_min, d_max, n=n_d)
        values = np.asarray(distances, dtype=float).ravel()
        if values.size == 0:
            raise ValueError("distances must not be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError("distances must be finite.")

        self.distances = values
        self.site_index = int(site_index)
        self.conformer = int(conformer)
        active_grid: DirectionGrid = FibonacciGrid() if grid is None else grid
        self.unit = np.asarray(active_grid.points(int(n_oa)), dtype=float)
        self.n_oa = int(len(self.unit))
        self.n_rot = int(self.orientation.count())
        self.n_d = int(values.size)

    @property
    def n_candidates(self) -> int:
        """Return ``n_oa * n_rot * n_d``."""
        return self.n_oa * self.n_rot * self.n_d

    def _decode(self, kappa: int) -> tuple[int, int, int]:
        """Decode a one-based index into ``(i, j, k)``, one-based.

        Args:
            kappa: One-based candidate index.

        Returns:
            tuple[int, int, int]: Direction, orientation and distance indices.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, n_candidates]``.
        """
        index = int(kappa)
        if index < 1 or index > self.n_candidates:
            raise IndexError(
                f"kappa must be in [1, {self.n_candidates}], got {kappa}."
            )
        local = index - 1
        i = local // (self.n_rot * self.n_d) + 1
        rest = local % (self.n_rot * self.n_d)
        j = rest // self.n_d + 1
        k = rest % self.n_d + 1
        return i, j, k

    def trial(self, kappa: int) -> Trial:
        """Build the trial of one candidate index.

        Args:
            kappa: One-based candidate index.

        Returns:
            Trial: The placed configuration and its provenance.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, n_candidates]``.
        """
        i, j, k = self._decode(kappa)
        direction = np.asarray(self.unit[i - 1], dtype=float)
        distance = float(self.distances[k - 1])
        target = (
            np.asarray(self.site.center, dtype=float) + distance * direction
        )
        quaternion = self.orientation.quaternion(j - 1)
        placed = place(
            self.reference,
            anchor=self.anchor,
            target=target,
            quaternion=quaternion,
        )
        structure = self.substrate.copy()
        structure.extend(placed)
        meta: dict[str, Any] = {
            "kind": "single_site",
            "site": self.site_index,
            "conformer": self.conformer,
            "i": i,
            "j": j,
            "k": k,
            "distance": distance,
            "n_substrate": len(self.substrate),
            "n_oa": self.n_oa,
            "n_rot": self.n_rot,
            "n_d": self.n_d,
        }
        return Trial(
            atoms=structure,
            index=int(kappa),
            n_total=self.n_candidates,
            meta=meta,
        )

    def generate(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Trial]:
        """Yield the trials of a one-based index interval.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end when ``None``.

        Yields:
            Trial: The trials in index order.

        Raises:
            IndexError: If the interval is outside ``[1, n_candidates]``.
            ValueError: If ``start`` is greater than ``stop``.
        """
        first = int(start)
        last = self.n_candidates if stop is None else int(stop)
        if first < 1 or last > self.n_candidates:
            raise IndexError(
                f"interval must be in [1, {self.n_candidates}], got "
                f"[{first}, {last}]."
            )
        if first > last:
            raise ValueError(f"start must not exceed stop, got {start}>{stop}.")
        for kappa in range(first, last + 1):
            yield self.trial(kappa)

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this generator.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        digest = hashlib.sha1(
            np.asarray(self.reference.positions, dtype=float).tobytes()
        ).hexdigest()
        return {
            "kind": "single_site",
            "site_index": self.site_index,
            "conformer": self.conformer,
            "core": [int(index) for index in self.core],
            "center": [float(value) for value in self.site.center],
            "anchor": {
                "kind": self.anchor.kind,
                "index": self.anchor.index,
                "position": [float(value) for value in self.anchor.position],
            },
            "orientation": {
                "name": getattr(self.orientation, "name", "unknown"),
                "count": self.n_rot,
            },
            "n_oa": self.n_oa,
            "n_d": self.n_d,
            "distances": [float(value) for value in self.distances],
            "reference": digest,
        }
