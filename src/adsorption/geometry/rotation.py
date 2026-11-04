"""Quaternion helpers for the rigid-body rotations of an adsorbate.

A rotation is represented by a unit quaternion ``(w, x, y, z)``. Composition,
inversion and application go through the Array-API helpers of
:mod:`adsorption.geometry._quat`; scipy's
:class:`scipy.spatial.transform.Rotation` supplies the closed-form
constructions and the conversion between the two quaternion layouts. Rotation
matrices are not part of the interface.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from adsorption.geometry._quat import (
    quaternion_apply,
    quaternion_invert,
    quaternion_raw_multiply,
    standardize_quaternion,
)

__all__ = [
    "apply_quaternion",
    "axis_angle_quaternion",
    "compose_quaternions",
    "quaternion_from_scipy",
    "scipy_from_quaternion",
    "shortest_arc_quaternion",
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


def quaternion_from_scipy(rotation: Rotation) -> np.ndarray:
    """Return the ``(w, x, y, z)`` quaternion of a scipy rotation.

    Args:
        rotation: A scipy ``Rotation``.

    Returns:
        np.ndarray: The unit quaternion, real part first, shape ``(4,)``.
    """
    x, y, z, w = np.asarray(rotation.as_quat(), dtype=float)
    return standardize_quaternion(np.array([w, x, y, z], dtype=float))


def scipy_from_quaternion(quaternion: np.ndarray) -> Rotation:
    """Return the scipy rotation of a ``(w, x, y, z)`` unit quaternion.

    Args:
        quaternion: Quaternion, real part first, shape ``(4,)``.

    Returns:
        Rotation: The equivalent scipy rotation.

    Raises:
        ValueError: If the quaternion has the wrong shape, a non-finite
            entry or a zero norm.
    """
    q = np.asarray(quaternion, dtype=float).ravel()
    if q.shape != (4,):
        raise ValueError(f"quaternion must be (4,), got {q.shape}.")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"quaternion must be finite, got {q}.")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Cannot normalise a zero-length quaternion.")
    w, x, y, z = q / norm
    return Rotation.from_quat([x, y, z, w])


def axis_angle_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the unit quaternion of a rotation about ``axis``.

    Args:
        axis: Rotation axis, shape ``(3,)``, normalised internally.
        angle: Rotation angle in radians.

    Returns:
        np.ndarray: The unit quaternion ``(w, x, y, z)``.
    """
    return quaternion_from_scipy(
        Rotation.from_rotvec(unit_vector(axis) * float(angle))
    )


def shortest_arc_quaternion(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the minimal rotation that maps ``u`` onto ``v``.

    The construction is the closed-form shortest arc; scipy's
    :meth:`Rotation.align_vectors` solves it and returns the quaternion.

    Args:
        u: Source unit vector (normalised internally), shape ``(3,)``.
        v: Target unit vector (normalised internally), shape ``(3,)``.

    Returns:
        np.ndarray: The unit quaternion ``(w, x, y, z)`` with ``R(u) = v``.
    """
    rotation, _ = Rotation.align_vectors(
        np.asarray([v], dtype=float),
        np.asarray([u], dtype=float),
    )
    return quaternion_from_scipy(rotation)


def apply_quaternion(
    quaternion: np.ndarray,
    positions: np.ndarray,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Apply a unit quaternion to Cartesian positions.

    Args:
        quaternion: Unit quaternion ``(w, x, y, z)``, shape ``(4,)``.
        positions: Positions of shape ``(n, 3)``.
        inverse: Apply the inverse rotation when ``True``.

    Returns:
        np.ndarray: The rotated positions, shape ``(n, 3)``.

    Raises:
        ValueError: If the quaternion or the positions have a wrong shape.
    """
    q = np.asarray(quaternion, dtype=float).ravel()
    if q.shape != (4,):
        raise ValueError(f"quaternion must be (4,), got {q.shape}.")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"quaternion must be finite, got {q}.")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Cannot normalise a zero-length quaternion.")
    q = q / norm
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {pos.shape}.")
    if inverse:
        q = quaternion_invert(q)
    return np.asarray(quaternion_apply(q, pos), dtype=float)


def compose_quaternions(*quaternions: np.ndarray) -> np.ndarray:
    """Compose unit quaternions left to right.

    Args:
        *quaternions: Unit quaternions ``(w, x, y, z)``.

    Returns:
        np.ndarray: The product, real part first, shape ``(4,)``.

    Raises:
        ValueError: If no quaternion is given.
    """
    if not quaternions:
        raise ValueError("At least one quaternion is required.")
    result = np.asarray(quaternions[0], dtype=float).ravel()
    for quaternion in quaternions[1:]:
        result = np.asarray(
            quaternion_raw_multiply(result, quaternion), dtype=float
        ).ravel()
    return standardize_quaternion(result)
