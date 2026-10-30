"""Coverage metrics on the unit sphere."""

from __future__ import annotations

import numpy as np

from ..geometry import FibonacciGrid

__all__ = [
    "cap_fraction",
    "covering_radius",
    "nearest_neighbor_angles",
    "pole_bias",
]


def _unit(points: object) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must be (n, 3), got {array.shape}.")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("The point set contains a zero-length vector.")
    return array / norms[:, None]


def _reference(reference: object | None, n: int = 4096) -> np.ndarray:
    if reference is None:
        return FibonacciGrid().points(n)
    return _unit(reference)


def _angles(dots: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(dots, -1.0, 1.0))


def nearest_neighbor_angles(
    points: object,
    *,
    chunk: int = 2048,
) -> np.ndarray:
    """Return the nearest-neighbour angular spacing of every point.

    Args:
        points: Unit vectors of shape ``(n, 3)``.
        chunk: Row block size used to bound memory.

    Returns:
        np.ndarray: Angular spacing in radians, shape ``(n,)``.

    Raises:
        ValueError: If fewer than two points are given.
    """
    unit = _unit(points)
    count = len(unit)
    if count < 2:
        raise ValueError("At least two points are required.")
    output = np.empty(count, dtype=float)
    for start in range(0, count, max(1, int(chunk))):
        stop = min(count, start + max(1, int(chunk)))
        dots = unit[start:stop] @ unit.T
        rows = np.arange(stop - start)
        dots[rows, np.arange(start, stop)] = -1.0
        output[start:stop] = _angles(dots.max(axis=1))
    return output


def covering_radius(
    points: object,
    reference: object | None = None,
    *,
    chunk: int = 512,
) -> float:
    """Return the covering radius of a point set on the sphere.

    Every reference point is mapped to its closest member of ``points``; the
    largest such angle is the covering radius. The default reference is the
    deterministic Fibonacci set of 4096 directions.

    Args:
        points: Unit vectors of shape ``(n, 3)``.
        reference: Probe directions of shape ``(m, 3)``.
        chunk: Row block size used to bound memory.

    Returns:
        float: The covering radius in radians.
    """
    unit = _unit(points)
    probes = _reference(reference)
    radius = 0.0
    for start in range(0, len(probes), max(1, int(chunk))):
        stop = min(len(probes), start + max(1, int(chunk)))
        dots = probes[start:stop] @ unit.T
        radius = max(radius, float(_angles(dots.max(axis=1)).max()))
    return radius


def pole_bias(points: object, *, bins: int = 18) -> float:
    """Return the maximum relative excess of a polar bin over uniform.

    The height coordinate ``z`` of an equal-area set is uniform on
    ``[-1, 1]``; a latitude-longitude grid piles up at the poles. A value of
    zero means no bin exceeds the uniform expectation.

    Args:
        points: Unit vectors of shape ``(n, 3)``.
        bins: Number of bins over ``z``.

    Returns:
        float: ``max(|observed - expected| / expected)``.
    """
    if int(bins) < 1:
        raise ValueError(f"bins must be positive, got {bins}.")
    unit = _unit(points)
    counts, _ = np.histogram(unit[:, 2], bins=int(bins), range=(-1.0, 1.0))
    expected = len(unit) / float(bins)
    return float(np.max(np.abs(counts - expected) / expected))


def cap_fraction(
    points: object,
    direction: np.ndarray,
    half_angle: float,
) -> float:
    """Return the fraction of points inside a spherical cap.

    Args:
        points: Unit vectors of shape ``(n, 3)``.
        direction: Cap axis of shape ``(3,)``.
        half_angle: Cap half-angle in radians, in ``[0, pi]``.

    Returns:
        float: The fraction of points inside the cap.
    """
    unit = _unit(points)
    axis = np.asarray(direction, dtype=float)
    if axis.shape != (3,) or np.linalg.norm(axis) < 1e-12:
        raise ValueError("direction must be a non-zero 3-vector.")
    axis = axis / np.linalg.norm(axis)
    if not 0.0 <= float(half_angle) <= np.pi:
        raise ValueError(
            f"half_angle must be in [0, pi], got {half_angle}."
        )
    cosine = np.cos(float(half_angle))
    return float(np.mean(unit @ axis >= cosine))
