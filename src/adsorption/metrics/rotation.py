"""Coverage metrics on SO(3)."""

from __future__ import annotations

import numpy as np

from ..geometry import SO3Set, matrix_to_quaternion

__all__ = [
    "quaternion_geodesic",
    "rotation_covering_radius",
    "rotation_nearest_neighbor_angles",
]


def _unit_quaternions(quaternions: object) -> np.ndarray:
    array = np.asarray(quaternions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"quaternions must be (n, 4), got {array.shape}.")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("The quaternion set contains a zero quaternion.")
    return array / norms[:, None]


def quaternion_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    """Return the geodesic angle between two rotations.

    A unit quaternion and its negative describe the same rotation, so the
    angle is ``2 * arccos(|q1 . q2|)`` in ``[0, pi]``.

    Args:
        left: Quaternion ``(w, x, y, z)``.
        right: Quaternion ``(w, x, y, z)``.

    Returns:
        float: The rotation angle in radians.

    Raises:
        ValueError: If a quaternion is not a 4-vector.
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("Quaternions must be 4-vectors.")
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * np.arccos(min(1.0, abs(float(a @ b)))))


def rotation_nearest_neighbor_angles(
    quaternions: object,
    *,
    chunk: int = 2048,
) -> np.ndarray:
    """Return the nearest-neighbour geodesic angle of every rotation.

    Args:
        quaternions: Unit quaternions ``(w, x, y, z)`` of shape ``(n, 4)``.
        chunk: Row block size used to bound memory.

    Returns:
        np.ndarray: Angles in radians, shape ``(n,)``.

    Raises:
        ValueError: If fewer than two rotations are given.
    """
    unit = _unit_quaternions(quaternions)
    count = len(unit)
    if count < 2:
        raise ValueError("At least two rotations are required.")
    output = np.empty(count, dtype=float)
    for start in range(0, count, max(1, int(chunk))):
        stop = min(count, start + max(1, int(chunk)))
        dots = np.abs(unit[start:stop] @ unit.T)
        dots[np.arange(stop - start), np.arange(start, stop)] = 0.0
        output[start:stop] = 2.0 * np.arccos(np.clip(dots.max(axis=1), 0, 1))
    return output


def rotation_covering_radius(
    quaternions: object,
    reference: object | None = None,
    *,
    chunk: int = 256,
) -> float:
    """Return the covering radius of a rotation set.

    Args:
        quaternions: Unit quaternions of shape ``(n, 4)``.
        reference: Probe quaternions; defaults to the deterministic
            ``SO3Set(2048)`` sequence.
        chunk: Row block size used to bound memory.

    Returns:
        float: The covering radius in radians.
    """
    unit = _unit_quaternions(quaternions)
    probes = (
        SO3Set(2048).quaternions
        if reference is None
        else _unit_quaternions(reference)
    )
    radius = 0.0
    for start in range(0, len(probes), max(1, int(chunk))):
        stop = min(len(probes), start + max(1, int(chunk)))
        dots = np.abs(probes[start:stop] @ unit.T)
        angles = 2.0 * np.arccos(np.clip(dots.max(axis=1), 0, 1))
        radius = max(radius, float(angles.max()))
    return radius


def matrices_to_quaternions(matrices: object) -> np.ndarray:
    """Convert a stack of rotation matrices to quaternions.

    Args:
        matrices: Rotation matrices of shape ``(n, 3, 3)``.

    Returns:
        np.ndarray: Quaternions of shape ``(n, 4)``.
    """
    array = np.asarray(matrices, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"matrices must be (n, 3, 3), got {array.shape}.")
    return np.stack([matrix_to_quaternion(matrix) for matrix in array])
