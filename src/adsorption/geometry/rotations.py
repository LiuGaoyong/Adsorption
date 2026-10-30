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
from .rotation import (
    axis_angle_rotation,
    quaternion_to_rotation,
    shortest_arc_rotation,
    unit_vector,
)

__all__ = [
    "DirectionRollSet",
    "FixedRotation",
    "RotationSet",
    "SO3Set",
    "ShortestArcSet",
]


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


class DirectionRollSet:
    """The direction-plus-roll orientation set on ``S^2 x S^1``.

    The reference axis is mapped onto every direction of a grid with the
    shortest-arc rotation, and a roll about the placed axis is applied on top:

    ``R(j, gamma) = T(v_j, gamma) R_j``.

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
        """Precompute the rotations.

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
        rotations: list[np.ndarray] = []
        for j, direction in enumerate(self.directions):
            base = self._arcs.rotation(j)
            for gamma in rolls:
                rotations.append(
                    axis_angle_rotation(direction, float(gamma)) @ base
                )
        self._rotations = tuple(rotations)

    def count(self) -> int:
        """Return the number of sampled orientations."""
        return len(self._rotations)

    def rotation(self, index: int) -> np.ndarray:
        """Return the rotation at ``index``."""
        return np.asarray(self._rotations[int(index)], dtype=float)

    def image(self, index: int) -> np.ndarray:
        """Return the placed reference axis at ``index`` (roll invariant)."""
        return np.asarray(
            self.directions[int(index) // self.n_roll], dtype=float
        )


def _generalized_golden_ratio(dimension: int) -> float:
    """Return the root above one of ``x ** (dimension + 1) = x + 1``."""
    def residual(x: float) -> float:
        return x ** (dimension + 1) - x - 1.0

    lower, upper = 1.0, 2.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


class SO3Set:
    """A deterministic low-discrepancy point set on ``SO(3)``.

    Three Kronecker sequences built from the generalised golden ratio of
    dimension three are mapped through Shoemake's uniform-quaternion
    parameterisation:

    ``q = (sqrt(1-u1) sin 2*pi*u2, sqrt(1-u1) cos 2*pi*u2,
           sqrt(u1) sin 2*pi*u3, sqrt(u1) cos 2*pi*u3)``

    with ``u`` the Kronecker point. The construction is closed-form and
    reproducible. It samples the roll as well as the axis direction, so for a
    linear adsorbate it over-samples the physically irrelevant roll; use
    ``ShortestArcSet`` there.

    Attributes:
        quaternions: The unit quaternions ``(w, x, y, z)``, shape ``(n, 4)``.
    """

    name = "so3"

    def __init__(self, n: int) -> None:
        """Precompute the quaternions and rotation matrices.

        Args:
            n: Number of rotations.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        if int(n) < 1:
            raise ValueError(f"n must be positive, got {n}.")
        dimension = 3
        golden = _generalized_golden_ratio(dimension)
        alpha = np.array(
            [1.0 / golden**k for k in range(1, dimension + 1)]
        )
        points = (
            0.5 + np.arange(int(n), dtype=float)[:, None] * alpha[None, :]
        ) % 1.0
        u1, u2, u3 = points[:, 0], points[:, 1], points[:, 2]
        qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
        qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
        self.quaternions = np.column_stack((qw, qx, qy, qz))
        self._rotations = tuple(
            quaternion_to_rotation(q) for q in self.quaternions
        )

    def count(self) -> int:
        """Return the number of sampled rotations."""
        return len(self._rotations)

    def rotation(self, index: int) -> np.ndarray:
        """Return the rotation at ``index``."""
        return np.asarray(self._rotations[int(index)], dtype=float)
