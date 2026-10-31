"""Adsorption-site geometry: core set, centre, shell, and direction sphere.

The site is the first factor of the adsorption event. It is defined by a
non-empty core set ``C_s``; its centre is the minimum-image mean of the core
positions and its first-neighbour shell ``J(s)`` is detected from covalent
radii. The direction sphere has radius ``r_s = max_cov(C_s) * scale``, and the
filter set is ``J(s)`` for a single-core site and ``C_s union J(s)`` for a
multi-core site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import covalent_radii as COV_R

from .constants import DELTA_NBR, MAX_CORE_SIZE, default_scale, default_skin
from .shell import first_neighbor_shell, mean_mic, outward_normal

__all__ = ["SiteGeometry", "build_site"]


@dataclass(frozen=True)
class SiteGeometry:
    """The geometric description of one adsorption site.

    The arrays are aligned with the substrate index space (``core_indices``
    and ``neighbor_indices`` index into the substrate atoms). ``core_positions``
    and ``neighbor_positions`` cache ``atoms.positions`` at construction time.

    Attributes:
        core_indices: Indices of the core atoms, shape ``(k,)``.
        neighbor_indices: Indices of the first-neighbour shell, shape ``(m,)``.
        core_positions: Core positions, shape ``(k, 3)``.
        neighbor_positions: Shell positions, shape ``(m, 3)``.
        core_numbers: Atomic numbers of the core atoms, shape ``(k,)``.
        neighbor_numbers: Atomic numbers of the shell, shape ``(m,)``.
        center: Site centre ``O_s``, shape ``(3,)``.
        cell: ``(3, 3)`` cell matrix, or ``None`` for a non-periodic structure.
        pbc: ``(3,)`` periodic-axis mask, or ``None`` when non-periodic.
        scale: Direction-sphere scale factor.
        skin: Steric margin in Angstrom.
        delta_nbr: Neighbour-detection tolerance in Angstrom.
    """

    core_indices: np.ndarray
    neighbor_indices: np.ndarray
    core_positions: np.ndarray
    neighbor_positions: np.ndarray
    core_numbers: np.ndarray
    neighbor_numbers: np.ndarray
    center: np.ndarray
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    scale: float = 1.2
    skin: float = 1.0
    delta_nbr: float = DELTA_NBR

    def __post_init__(self) -> None:
        """Validate the shapes and the scalar parameters."""
        for name in ("core_indices", "neighbor_indices"):
            value = np.asarray(getattr(self, name), dtype=int)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
        for name in ("core_positions", "neighbor_positions"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 2 or value.shape[-1] != 3:
                raise ValueError(f"{name} must be (n, 3), got {value.shape}.")
        for name in ("core_numbers", "neighbor_numbers"):
            value = np.asarray(getattr(self, name), dtype=int)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,):
            raise ValueError(f"center must be (3,), got {center.shape}.")
        if self.cell is not None:
            cell = np.asarray(self.cell, dtype=float)
            if cell.shape != (3, 3):
                raise ValueError(f"cell must be (3, 3), got {cell.shape}.")
        if self.pbc is not None:
            pbc = np.asarray(self.pbc, dtype=bool)
            if pbc.shape != (3,):
                raise ValueError(f"pbc must be (3,), got {pbc.shape}.")
        if len(self.core_positions) == 0:
            raise ValueError("The core set must not be empty.")
        if len(self.neighbor_positions) == 0:
            raise ValueError("The first-neighbour shell must not be empty.")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}.")
        if not np.isfinite(self.skin) or self.skin < 0.0:
            raise ValueError(f"skin must be non-negative, got {self.skin}.")
        if not np.isfinite(self.delta_nbr) or self.delta_nbr < 0.0:
            raise ValueError(
                f"delta_nbr must be non-negative, got {self.delta_nbr}."
            )

    @property
    def core_size(self) -> int:
        """Return the number of core atoms."""
        return int(len(self.core_positions))

    @property
    def radius(self) -> float:
        """Return the direction-sphere radius ``r_s``."""
        cov = np.asarray(COV_R[np.asarray(self.core_numbers, dtype=int)])
        return float(np.max(cov)) * float(self.scale)

    @property
    def filter_positions(self) -> np.ndarray:
        """Return the positions of the steric filter set ``P(s)``."""
        if self.core_size == 1:
            return np.asarray(self.neighbor_positions, dtype=float)
        return np.vstack(
            (
                np.asarray(self.core_positions, dtype=float),
                np.asarray(self.neighbor_positions, dtype=float),
            )
        )

    @property
    def filter_numbers(self) -> np.ndarray:
        """Return the atomic numbers of the steric filter set ``P(s)``."""
        if self.core_size == 1:
            return np.asarray(self.neighbor_numbers, dtype=int)
        return np.concatenate(
            (
                np.asarray(self.core_numbers, dtype=int),
                np.asarray(self.neighbor_numbers, dtype=int),
            )
        )

    @property
    def filter_radii(self) -> np.ndarray:
        """Return the covalent radii of the steric filter set."""
        return np.asarray(COV_R[self.filter_numbers], dtype=float)

    @property
    def outward_normal(self) -> np.ndarray:
        """Return the outward reference direction ``n_s`` of Eq. (4).

        This vector is bookkeeping and visualisation only; it does not
        constrain the ``OA`` grid.
        """
        return outward_normal(
            self.center,
            self.neighbor_positions,
            cell=self.cell,
            pbc=self.pbc,
        )

    def cell_array(self) -> np.ndarray | None:
        """Return the cached cell as a float array, or ``None``."""
        if self.cell is None:
            return None
        return np.asarray(self.cell, dtype=float)

    def pbc_array(self) -> np.ndarray | None:
        """Return the cached periodic-axis mask, or ``None``."""
        if self.pbc is None:
            return None
        return np.asarray(self.pbc, dtype=bool)


def build_site(
    atoms: Atoms,
    core: np.ndarray | list[int] | int,
    *,
    neighbors: np.ndarray | list[int] | None = None,
    origin: object | None = None,
    scale: float | None = None,
    skin: float | None = None,
    delta_nbr: float = DELTA_NBR,
) -> SiteGeometry:
    """Build the geometry of one adsorption site.

    Args:
        atoms: Substrate structure.
        core: Core atom indices (one atom for a top site, two for a bridge,
            three for a hollow).
        neighbors: First-neighbour shell indices. Detected from covalent radii
            when ``None``.
        origin: Optional ``graphatoms`` ``System``/``Cluster`` used for the
            neighbour lookup instead of the covalent-radius detection.
        scale: Direction-sphere scale factor. Defaults to the value tabulated
            in the manuscript for the core size.
        skin: Steric margin. Defaults to the tabulated value for the core size.
        delta_nbr: Neighbour-detection tolerance in Angstrom.

    Returns:
        SiteGeometry: The site geometry.

    Raises:
        ValueError: If the core is empty, out of range, larger than
            ``MAX_CORE_SIZE``, or has no first-neighbour shell.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms)}.")
    core_array = np.unique(np.asarray(core, dtype=int))
    if core_array.size == 0:
        raise ValueError("The core set must not be empty.")
    if core_array.size > MAX_CORE_SIZE:
        raise ValueError(
            "The core size must be less than or equal to "
            f"{MAX_CORE_SIZE}, got {core_array.size}."
        )
    if core_array[0] < 0 or core_array[-1] >= len(atoms):
        raise ValueError(
            f"Core indices must be in [0, {len(atoms)}), got {core_array}."
        )

    if neighbors is None:
        neighbor_array = first_neighbor_shell(
            atoms,
            core_array,
            delta_nbr=delta_nbr,
            origin=origin,
        )
    else:
        neighbor_array = np.unique(np.asarray(neighbors, dtype=int))
        neighbor_array = np.setdiff1d(neighbor_array, core_array)
    if neighbor_array.size == 0:
        raise ValueError(
            f"No first-neighbour shell found for the core {core_array}."
        )
    if neighbor_array[0] < 0 or neighbor_array[-1] >= len(atoms):
        raise ValueError(
            "Neighbour indices must be in "
            f"[0, {len(atoms)}), got {neighbor_array}."
        )

    cell = np.asarray(atoms.cell.array, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if not np.any(pbc) or abs(np.linalg.det(cell)) < 1e-12:
        cell_arg, pbc_arg = None, None
    else:
        cell_arg, pbc_arg = cell, pbc

    core_positions = np.asarray(atoms.positions[core_array], dtype=float)
    neighbor_positions = np.asarray(
        atoms.positions[neighbor_array], dtype=float
    )
    center = mean_mic(core_positions, cell_arg, pbc_arg)

    return SiteGeometry(
        core_indices=core_array,
        neighbor_indices=neighbor_array,
        core_positions=core_positions,
        neighbor_positions=neighbor_positions,
        core_numbers=np.asarray(atoms.numbers[core_array], dtype=int),
        neighbor_numbers=np.asarray(atoms.numbers[neighbor_array], dtype=int),
        center=center,
        cell=cell_arg,
        pbc=pbc_arg,
        scale=default_scale(core_array.size)
        if scale is None
        else float(scale),
        skin=default_skin(core_array.size) if skin is None else float(skin),
        delta_nbr=float(delta_nbr),
    )
