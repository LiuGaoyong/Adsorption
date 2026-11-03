"""Process A for one adsorption site: enumerate ``(OA, SO(3), d)``.

``SingleSiteGen`` owns a single :class:`~adsorption.geometry.SiteGeometry` and
a single reference adsorbate. It enumerates the approach direction on a
Fibonacci grid, the molecular orientation on a rotation set (by default the
Super-Fibonacci sequence on ``SO(3)``), and the anchor distance on a one-
dimensional array. The candidate index follows the protocol order: direction
outermost, orientation next, distance fastest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from ase import Atoms

from ..geometry import (
    DirectionGrid,
    FibonacciGrid,
    ResolvedAnchor,
    RotationSet,
    SiteGeometry,
    place,
)
from ..process import AdsConfGenABC, Trial

__all__ = ["SingleSiteGen"]


class SingleSiteGen(AdsConfGenABC):
    """Enumerate the trial configurations of one fixed site.

    Attributes:
        substrate: The substrate structure.
        site: The site geometry.
        reference: The reference adsorbate.
        anchor: The resolved anchor in the reference frame.
        orientation: The rotation set.
        distances: The anchor-centre distances, shape ``(n_d,)``.
        n_oa: Number of approach directions.
        n_rot: Number of orientations.
        n_d: Number of distances.
    """

    def __init__(
        self,
        substrate: Atoms,
        site: SiteGeometry,
        reference: Atoms,
        *,
        anchor: ResolvedAnchor,
        orientation: RotationSet,
        distances: np.ndarray,
        grid: DirectionGrid | None = None,
        n_grid: int = 200,
        site_index: int = 0,
        conformer: int = 0,
    ) -> None:
        """Store the site, the adsorbate and the three grids.

        Args:
            substrate: The substrate structure.
            site: The geometry of the fixed site.
            reference: The reference adsorbate.
            anchor: The resolved anchor of the reference.
            orientation: The rotation set.
            distances: The anchor-centre distances in Angstrom.
            grid: Direction grid. Defaults to the Fibonacci lattice.
            n_grid: Requested number of directions.
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
        values = np.asarray(distances, dtype=float).ravel()
        if values.size == 0:
            raise ValueError("distances must not be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError("distances must be finite.")
        if int(n_grid) < 1:
            raise ValueError(f"n_grid must be positive, got {n_grid}.")
        if int(orientation.count()) < 1:
            raise ValueError("orientation must contain at least one rotation.")

        self.substrate = substrate
        self.site = site
        self.reference = reference
        self.anchor = anchor
        self.orientation = orientation
        self.distances = values
        self.site_index = int(site_index)
        self.conformer = int(conformer)
        active_grid: DirectionGrid = FibonacciGrid() if grid is None else grid
        self.unit = np.asarray(active_grid.points(int(n_grid)), dtype=float)
        self.n_oa = int(len(self.unit))
        self.n_rot = int(orientation.count())
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
            np.asarray(self.site.center, dtype=float)
            + distance * direction
        )
        rotation = np.asarray(self.orientation.rotation(j - 1), dtype=float)
        placed = place(
            self.reference,
            anchor=self.anchor,
            target=target,
            rotation=rotation,
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
            "core": [int(index) for index in self.site.core_indices],
            "center": [float(value) for value in self.site.center],
            "anchor": {
                "kind": self.anchor.kind,
                "index": self.anchor.index,
                "position": [
                    float(value) for value in self.anchor.position
                ],
            },
            "orientation": {
                "name": getattr(self.orientation, "name", "unknown"),
                "count": self.n_rot,
            },
            "n_grid": self.n_oa,
            "n_d": self.n_d,
            "distances": [float(value) for value in self.distances],
            "reference": digest,
        }
