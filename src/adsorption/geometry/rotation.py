"""Rigid-body rotations used to orient an adsorbate.

The orientation factor is represented by a rotation matrix ``R`` acting on the
body frame. The shortest-arc construction of Eq. (9) maps a declared body axis
onto a grid direction with the smallest possible rotation angle; the roll about
that axis is not sampled by this construction.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_rotation",
    "axis_angle_rotation",
    "quaternion_to_rotation",
    "shortest_arc_rotation",
    "unit_vector",
]


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return the normalised vector.

    Args:
        vector: A vector of shape ``(3,)``.

    Returns:
        np.ndarray: The unit vector.

    Raises:
        ValueError: If the vector is not three-dimensional or is zero.
    """
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"The vector must be (3,), got {v.shape}.")
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        raise ValueError("Cannot normalise a zero-length vector.")
    return v / length


def shortest_arc_rotation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the minimal rotation that maps ``u`` onto ``v``.

    The rotation is ``R = I + sin(beta) [k]_x + (1 - cos(beta)) [k]_x^2`` with
    ``k = u x v / |u x v|`` and ``beta = arccos(u . v)``. The degenerate cases
    are handled explicitly: ``beta = 0`` is the identity and ``beta = pi`` is a
    half turn about an axis perpendicular to ``u``.

    Args:
        u: Source unit vector (normalised internally), shape ``(3,)``.
        v: Target unit vector (normalised internally), shape ``(3,)``.

    Returns:
        np.ndarray: A ``(3, 3)`` rotation matrix with ``R @ u == v``.
    """
    a = unit_vector(u)
    b = unit_vector(v)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    axis = np.cross(a, b)
    sine = float(np.linalg.norm(axis))
    if sine < 1e-12:
        # beta = pi: any axis perpendicular to a is a valid half turn.
        helper = (
            np.array([1.0, 0.0, 0.0])
            if abs(a[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        k = unit_vector(np.cross(a, helper))
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = axis / sine
    skew = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the rotation matrix of a rotation about an axis.

    Args:
        axis: Rotation axis of shape ``(3,)`` (normalised internally).
        angle: Rotation angle in radians.

    Returns:
        np.ndarray: A ``(3, 3)`` rotation matrix.
    """
    k = unit_vector(axis)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    skew = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion (real part first) to a rotation matrix.

    Args:
        quaternion: Quaternion ``(w, x, y, z)``, normalised internally.

    Returns:
        np.ndarray: The ``(3, 3)`` rotation matrix.
    """
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must be (4,), got {q.shape}.")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Cannot normalise a zero-length quaternion.")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
             2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
             1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def apply_rotation(
    positions: np.ndarray,
    rotation: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a rotation matrix to Cartesian positions.

    Args:
        positions: Positions of shape ``(n, 3)``.
        rotation: A ``(3, 3)`` rotation matrix.
        center: Optional rotation centre of shape ``(3,)``.

    Returns:
        np.ndarray: The rotated positions, shape ``(n, 3)``.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {pos.shape}.")
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {matrix.shape}.")
    if center is None:
        return pos @ matrix.T
    origin = np.asarray(center, dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"center must be (3,), got {origin.shape}.")
    return (pos - origin) @ matrix.T + origin
