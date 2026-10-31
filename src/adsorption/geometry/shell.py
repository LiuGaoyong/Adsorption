"""Core-mean and first-neighbour-shell helpers."""

from __future__ import annotations

from functools import reduce

import numpy as np
from ase import Atoms
from ase.data import covalent_radii as COV_R

from .constants import DELTA_NBR
from .mic import displacements_mic, is_periodic

__all__ = ["first_neighbor_shell", "mean_mic", "outward_normal"]


def mean_mic(
    positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> np.ndarray:
    """Return the minimum-image mean of a set of positions.

    The mean is taken after bringing every position into the image closest to
    the first one, so a core set that straddles a cell boundary still has a
    well-defined centre. For a non-periodic structure this is the arithmetic
    mean.

    Args:
        positions: Positions of shape ``(n, 3)`` with ``n >= 1``.
        cell: Optional ``(3, 3)`` cell matrix.
        pbc: Optional ``(3,)`` periodic-axis mask.

    Returns:
        np.ndarray: The mean position, shape ``(3,)``.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {pos.shape}.")
    if pos.shape[0] == 0:
        raise ValueError("Cannot take the mean of an empty position set.")
    if not is_periodic(cell, pbc):
        return pos.mean(axis=0)
    ref = pos[0]
    displacement = displacements_mic(pos - ref, cell, pbc)
    return ref + displacement.mean(axis=0)


def outward_normal(
    center: np.ndarray,
    neighbor_positions: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> np.ndarray:
    """Return the outward reference direction of Eq. (4).

    The vector runs from the minimum-image centroid of the first-neighbour
    shell to the site centre and is normalised. It is bookkeeping and
    visualisation only; it does not constrain the direction grid.

    Args:
        center: Site centre, shape ``(3,)``.
        neighbor_positions: Shell positions of shape ``(m, 3)``.
        cell: Optional ``(3, 3)`` cell matrix.
        pbc: Optional ``(3,)`` periodic-axis mask.

    Returns:
        np.ndarray: The unit outward direction, shape ``(3,)``.

    Raises:
        ValueError: If the shell centroid coincides with the centre.
    """
    c = np.asarray(center, dtype=float)
    if c.shape != (3,):
        raise ValueError(f"center must be (3,), got {c.shape}.")
    centroid = mean_mic(neighbor_positions, cell, pbc)
    vector = c - centroid
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError(
            "The outward direction is undefined: the shell centroid "
            "coincides with the site centre."
        )
    return vector / length


def first_neighbor_shell(
    atoms: Atoms,
    core: np.ndarray | list[int] | int,
    *,
    delta_nbr: float = DELTA_NBR,
    origin: object | None = None,
) -> np.ndarray:
    """Detect the first-neighbour shell ``J(s)`` of a core set.

    An atom ``j`` belongs to the shell when its minimum-image distance to a
    core atom ``k`` is smaller than ``R_cov(j) + R_cov(k) + delta_nbr``. The
    core atoms themselves are excluded. When ``origin`` (a ``graphatoms``
    ``System``/``Cluster``) is given, its neighbour lookup is used instead.

    Args:
        atoms: Substrate structure.
        core: Core atom indices.
        delta_nbr: Neighbour-detection tolerance in Angstrom.
        origin: Optional ``graphatoms`` structure used for the neighbour
            lookup.

    Returns:
        np.ndarray: Sorted unique shell indices.
    """
    core_array = np.unique(np.asarray(core, dtype=int))
    if core_array.size == 0:
        raise ValueError("The core set must not be empty.")
    if not np.isfinite(delta_nbr) or delta_nbr < 0.0:
        raise ValueError(f"delta_nbr must be non-negative, got {delta_nbr}.")

    if origin is not None:
        get_neighbors = getattr(origin, "get_neighbors", None)
        if get_neighbors is None:
            raise TypeError(
                "origin must provide a get_neighbors method, got "
                f"{type(origin)}."
            )
        collected = [
            np.asarray(get_neighbors(int(index)), dtype=int)
            for index in core_array
        ]
        neighbors = reduce(
            np.append, collected, np.empty(0, dtype=int)
        ).astype(int)
    else:
        all_indices = np.arange(len(atoms))
        first = np.repeat(core_array, len(all_indices))
        second = np.tile(all_indices, len(core_array))
        distances = atoms.get_distances(first, second, mic=True)
        cutoff = (
            COV_R[atoms.numbers[first]]
            + COV_R[atoms.numbers[second]]
            + float(delta_nbr)
        )
        neighbors = second[distances < cutoff]

    neighbors = np.unique(neighbors)
    return np.setdiff1d(neighbors, core_array)
