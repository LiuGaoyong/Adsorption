"""Direction grids on the unit sphere.

A grid is a deterministic point set: for a given lattice size ``n`` it returns
the same points in the same order. The protocol layer uses ``size(n)`` as the
radix of the direction dimension, and the admissibility filter keeps the
original grid indices so that a candidate can be regenerated from its index.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from graphatoms.geometry.sample import fibonacci_lattice

__all__ = [
    "DirectionGrid",
    "FibonacciGrid",
    "LatLonGrid",
    "SeededRandomGrid",
]


@runtime_checkable
class DirectionGrid(Protocol):
    """A deterministic point set on the unit sphere."""

    name: str
    """Short name used in protocol records."""

    def size(self, n: int) -> int:
        """Return the number of points generated for the request ``n``."""
        ...

    def points(self, n: int) -> np.ndarray:
        """Return unit vectors of shape ``(size(n), 3)``."""
        ...


def _check_n(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError(f"The grid size must be positive, got {n}.")
    return n


def _normalize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-15):
        raise ValueError("The grid contains a zero-length vector.")
    return points / norms[:, None]


class FibonacciGrid:
    """The spherical Fibonacci lattice of Eqs. (5)-(6).

    The point ordering is the lattice index order: ``z_i = 1 - (2i+1)/n``
    decreasing. The construction is closed-form, has no pole points, and the
    height coordinate is uniformly spaced, i.e. the points own equal-area
    bands.
    """

    name = "fibonacci"

    def size(self, n: int) -> int:
        """Return ``n``."""
        return _check_n(n)

    def points(self, n: int) -> np.ndarray:
        """Return the ``n`` unit Fibonacci vectors in index order."""
        n = _check_n(n)
        return np.asarray(fibonacci_lattice(n), dtype=float)


class LatLonGrid:
    """An equal-angle latitude-longitude grid.

    ``n`` is only a request: the grid is a product of a polar and an azimuthal
    count, so ``size(n)`` is the realised number of points (at most a few
    percent away from ``n``). Equal-angle spacing over-samples the poles, which
    is exactly the bias the Fibonacci grid avoids.
    """

    name = "latlon"

    @staticmethod
    def _shape(n: int) -> tuple[int, int]:
        n = _check_n(n)
        n_theta = max(2, int(round(np.sqrt(n / 2.0))))
        n_phi = max(2, int(round(n / n_theta)))
        return n_theta, n_phi

    def size(self, n: int) -> int:
        """Return the realised number of points."""
        n_theta, n_phi = self._shape(n)
        return n_theta * n_phi

    def points(self, n: int) -> np.ndarray:
        """Return the latitude-longitude unit vectors."""
        n_theta, n_phi = self._shape(n)
        theta = np.pi * (np.arange(n_theta) + 0.5) / n_theta
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        grid_theta, grid_phi = np.meshgrid(theta, phi, indexing="ij")
        points = np.column_stack(
            (
                (np.sin(grid_theta) * np.cos(grid_phi)).ravel(),
                (np.sin(grid_theta) * np.sin(grid_phi)).ravel(),
                np.cos(grid_theta).ravel(),
            )
        )
        return _normalize(points)


class SeededRandomGrid:
    """A uniform random point set with an explicit seed.

    This is a baseline, never the default. The seed is part of the protocol
    record so that a run can be repeated.
    """

    name = "random"

    def __init__(self, seed: int) -> None:
        """Store the seed.

        Args:
            seed: Integer seed of the ``numpy.random.Generator``.
        """
        self.seed = int(seed)

    def size(self, n: int) -> int:
        """Return ``n``."""
        return _check_n(n)

    def points(self, n: int) -> np.ndarray:
        """Return ``n`` random unit vectors, reproducible from the seed."""
        n = _check_n(n)
        rng = np.random.default_rng(self.seed)
        return _normalize(rng.normal(size=(n, 3)))
