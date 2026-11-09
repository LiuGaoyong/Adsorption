"""Orientation sets: rigid-body rotations sampled about a declared axis.

The direction-only construction of Section 2.4 samples the molecular
reference axis on the sphere and applies the shortest-arc rotation. The
``RotationSet`` interface hides that choice from the protocol layer: a set has
a deterministic count and returns the ``index``-th unit quaternion
``(w, x, y, z)``, so the budget index keeps a single orientation radix
whatever construction is used. Quaternion algebra goes through
:mod:`adsorption.geometry._quat`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from adsorption.geometry._quat import (
    quaternion_raw_multiply,
    standardize_quaternion,
)
from adsorption.geometry.grids import DirectionGrid, FibonacciGrid
from adsorption.geometry.rotation import (
    axis_angle_quaternion,
    shortest_arc_quaternion,
    unit_vector,
)

__all__ = [
    "DirectionRollSet",
    "FixedRotation",
    "RotationSet",
    "ShortestArcSet",
]


class RotationSet(ABC):
    """A deterministic, indexed set of rotations about an anchor."""

    name: str = ""
    """Short name used in protocol records."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of rotations in the set."""
        raise NotImplementedError

    @abstractmethod
    def quaternion(self, index: int) -> np.ndarray:
        """Return the unit quaternion ``(w, x, y, z)`` at ``index``."""
        raise NotImplementedError


class FixedRotation(RotationSet):
    """The identity rotation, used when orientation does not matter.

    A single-atom adsorbate has no orientation degree of freedom; this set
    keeps the orientation radix at one instead of wasting candidates.
    """

    name = "fixed"

    def count(self) -> int:
        """Return ``1``."""
        return 1

    def quaternion(self, index: int) -> np.ndarray:
        """Return the identity quaternion for ``index = 0``."""
        if int(index) != 0:
            raise IndexError(
                f"The fixed rotation set has one element, got {index}."
            )
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


class ShortestArcSet(RotationSet):
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
        """Precompute the quaternions.

        Args:
            axis: Body-frame reference axis, normalized internally.
            n: Number of sampled directions.
            grid: Direction grid. Defaults to the Fibonacci lattice.
        """
        self.axis = unit_vector(np.asarray(axis, dtype=float))
        active_grid: DirectionGrid = FibonacciGrid() if grid is None else grid
        self.directions = np.asarray(active_grid.points(n), dtype=float)
        self._quaternions = tuple(
            shortest_arc_quaternion(self.axis, direction)
            for direction in self.directions
        )

    def count(self) -> int:
        """Return the number of sampled orientations."""
        return len(self._quaternions)

    def quaternion(self, index: int) -> np.ndarray:
        """Return the shortest-arc rotation onto direction ``index``."""
        return np.asarray(self._quaternions[int(index)], dtype=float)

    def image(self, index: int) -> np.ndarray:
        """Return the placed reference axis at ``index``."""
        return np.asarray(self.directions[int(index)], dtype=float)


class DirectionRollSet(RotationSet):
    """The direction-plus-roll orientation set on ``S^2 x S^1``.

    The reference axis is mapped onto every direction of a grid with the
    shortest-arc rotation, and a roll about the placed axis is composed on
    top: ``q(j, gamma) = q_roll(v_j, gamma) q_arc(v_j)``.

    The roll grid is equally spaced. The index runs over the directions first
    and the roll fastest, i.e. ``j = index // n_roll`` and
    ``roll = index % n_roll``.

    Attributes:
        axis: Body-frame reference axis, unit length.
        directions: The sampled unit directions, shape ``(n_dir, 3)``.
        n_roll: Number of roll angles per direction.
    """

    name = "direction_roll"

    def __init__(
        self,
        axis: np.ndarray,
        n_dir: int,
        *,
        n_roll: int = 1,
        grid: DirectionGrid | None = None,
    ) -> None:
        """Precompute the quaternions.

        Args:
            axis: Body-frame reference axis, normalized internally.
            n_dir: Number of sampled directions.
            n_roll: Number of roll angles per direction.
            grid: Direction grid. Defaults to the Fibonacci lattice.

        Raises:
            ValueError: If ``n_dir`` or ``n_roll`` is not positive.
        """
        if int(n_dir) < 1 or int(n_roll) < 1:
            raise ValueError(
                f"n_dir and n_roll must be positive, got {n_dir} and {n_roll}."
            )
        self.axis = unit_vector(np.asarray(axis, dtype=float))
        self.n_roll = int(n_roll)
        self._arcs = ShortestArcSet(self.axis, int(n_dir), grid=grid)
        self.directions = self._arcs.directions
        rolls = 2.0 * np.pi * np.arange(self.n_roll) / self.n_roll
        quaternions: list[np.ndarray] = []
        for j, direction in enumerate(self.directions):
            base = self._arcs.quaternion(j)
            for gamma in rolls:
                roll = axis_angle_quaternion(direction, float(gamma))
                quaternions.append(
                    standardize_quaternion(quaternion_raw_multiply(roll, base))
                )
        self._quaternions = tuple(quaternions)

    def count(self) -> int:
        """Return the number of sampled orientations."""
        return len(self._quaternions)

    def quaternion(self, index: int) -> np.ndarray:
        """Return the rotation at ``index``."""
        return np.asarray(self._quaternions[int(index)], dtype=float)

    def image(self, index: int) -> np.ndarray:
        """Return the placed reference axis at ``index`` (roll invariant)."""
        return np.asarray(
            self.directions[int(index) // self.n_roll], dtype=float
        )
