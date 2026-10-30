"""Orientation sets: rigid-body rotations sampled about a declared axis.

The direction-only construction of Section 2.4 samples the molecular
reference axis on the sphere and applies the shortest-arc rotation. The
``RotationSet`` interface hides that choice from the protocol layer: a set has
a deterministic count and returns the ``index``-th rotation matrix, so the
budget index keeps a single orientation radix whatever construction is used.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .grids import DirectionGrid, FibonacciGrid
from .rotation import shortest_arc_rotation, unit_vector

__all__ = ["FixedRotation", "RotationSet", "ShortestArcSet"]


@runtime_checkable
class RotationSet(Protocol):
    """A deterministic, indexed set of rotations about an anchor."""

    name: str
    """Short name used in protocol records."""

    def count(self) -> int:
        """Return the number of rotations in the set."""
        ...

    def rotation(self, index: int) -> np.ndarray:
        """Return the ``(3, 3)`` rotation matrix at ``index``."""
        ...


class FixedRotation:
    """The identity rotation, used when orientation does not matter.

    A single-atom adsorbate has no orientation degree of freedom; this set
    keeps the orientation radix at one instead of wasting candidates.
    """

    name = "fixed"

    def count(self) -> int:
        """Return ``1``."""
        return 1

    def rotation(self, index: int) -> np.ndarray:
        """Return the identity for ``index = 0``."""
        if int(index) != 0:
            raise IndexError(
                f"The fixed rotation set has one element, got {index}."
            )
        return np.eye(3)


class ShortestArcSet:
    """The direction-only orientation set of Eqs. (9)-(10).

    A body-frame reference axis is mapped onto every direction of a sphere
    grid with the minimal rotation. The roll about the axis is not sampled.

    Attributes:
        axis: The body-frame reference axis, unit length.
        directions: The sampled unit directions, shape ``(N_AB, 3)``.
    """

    name = "shortest_arc"

    def __init__(
        self,
        axis: np.ndarray,
        n: int,
        *,
        grid: DirectionGrid | None = None,
    ) -> None:
        """Precompute the rotations.

        Args:
            axis: Body-frame reference axis, normalized internally.
            n: Number of sampled directions.
            grid: Direction grid. Defaults to the Fibonacci lattice.
        """
        self.axis = unit_vector(np.asarray(axis, dtype=float))
        active_grid: DirectionGrid = FibonacciGrid() if grid is None else grid
        self.directions = np.asarray(active_grid.points(n), dtype=float)
        self._rotations = tuple(
            shortest_arc_rotation(self.axis, direction)
            for direction in self.directions
        )

    def count(self) -> int:
        """Return the number of sampled orientations."""
        return len(self._rotations)

    def rotation(self, index: int) -> np.ndarray:
        """Return the rotation that maps the axis onto direction ``index``."""
        return np.asarray(self._rotations[int(index)], dtype=float)

    def image(self, index: int) -> np.ndarray:
        """Return the placed reference axis at ``index``."""
        return np.asarray(self.directions[int(index)], dtype=float)
