"""Minimum-image helpers shared by the geometric construction.

The functions here are the only place where periodic boundary conditions enter
the geometry. A structure is treated as periodic only when its cell is
non-singular and at least one axis is periodic; the periodic axes are then
handed to :func:`ase.geometry.find_mic`.
"""

from __future__ import annotations

import numpy as np
from ase.geometry import find_mic


def is_periodic(cell: np.ndarray | None, pbc: np.ndarray | None = None) -> bool:
    """Return whether a cell supports minimum-image distances.

    Args:
        cell: A ``(3, 3)`` cell matrix or ``None``.
        pbc: Optional ``(3,)`` boolean periodic-axis mask.

    Returns:
        bool: True when the cell is non-singular and some axis is periodic.
    """
    if cell is None:
        return False
    c = np.asarray(cell, dtype=float)
    if c.shape != (3, 3):
        raise ValueError(f"The cell must be (3, 3), got {c.shape}.")
    if abs(np.linalg.det(c)) < 1e-12:
        return False
    if pbc is None:
        return True
    return bool(np.any(pbc))


def _axis_mask(pbc: np.ndarray | None) -> np.ndarray | bool:
    if pbc is None:
        return True
    mask = np.asarray(pbc, dtype=bool)
    if mask.shape != (3,):
        raise ValueError(f"pbc must be (3,), got {mask.shape}.")
    return mask


def displacements_mic(
    vectors: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> np.ndarray:
    """Return the minimum-image displacement of Cartesian vectors.

    Args:
        vectors: Displacements of shape ``(..., 3)``.
        cell: Optional ``(3, 3)`` cell matrix.
        pbc: Optional ``(3,)`` periodic-axis mask. ``None`` means all axes
            periodic; it is only consulted when the cell is periodic.

    Returns:
        np.ndarray: Displacements of the same shape as ``vectors``.
    """
    v = np.asarray(vectors, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"The last axis must be 3, got {v.shape}.")
    if not is_periodic(cell, pbc):
        return v.copy()
    flat = v.reshape(-1, 3)
    out, _ = find_mic(flat, np.asarray(cell, dtype=float), _axis_mask(pbc))
    return np.asarray(out, dtype=float).reshape(v.shape)


def distances_mic(
    points_a: np.ndarray,
    points_b: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> np.ndarray:
    """Return the pairwise minimum-image distances between two point sets.

    Args:
        points_a: Points of shape ``(n, 3)``.
        points_b: Points of shape ``(m, 3)``.
        cell: Optional ``(3, 3)`` cell matrix.
        pbc: Optional ``(3,)`` periodic-axis mask.

    Returns:
        np.ndarray: Distances of shape ``(n, m)``.
    """
    a = np.atleast_2d(np.asarray(points_a, dtype=float))
    b = np.atleast_2d(np.asarray(points_b, dtype=float))
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError(
            f"Points must be (..., 3), got {a.shape} and {b.shape}."
        )
    diff = b[None, :, :] - a[:, None, :]
    diff = displacements_mic(diff, cell, pbc)
    return np.linalg.norm(diff, axis=-1)
