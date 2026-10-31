"""Admissible approach directions: the Fibonacci grid and the steric filter.

The filter removes a grid direction when the probe point at the direction
sphere overlaps any atom of the filter set ``P(s)``. It is a fixed-radius
direction proxy: it is evaluated at ``r_s``, not at the anchor distance ``d``.
The surviving unit vectors are returned together with their original grid
indices so that enumeration is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import DirectionGrid, FibonacciGrid
from .mic import distances_mic
from .site import SiteGeometry

__all__ = ["AdmissibleSet", "admissible_directions", "filter_grid"]


@dataclass(frozen=True)
class AdmissibleSet:
    """The admissible approach directions ``G_OA`` of one site.

    Attributes:
        unit: Surviving unit vectors, shape ``(N_OA, 3)``, ordered by
            increasing original grid index.
        grid_index: Original grid indices of the survivors, shape ``(N_OA,)``.
        n_grid: Number of points of the full grid, ``N``.
    """

    unit: np.ndarray
    grid_index: np.ndarray
    n_grid: int

    def __post_init__(self) -> None:
        """Validate the shapes."""
        unit = np.asarray(self.unit, dtype=float)
        index = np.asarray(self.grid_index, dtype=int)
        if unit.ndim != 2 or unit.shape[-1] != 3:
            raise ValueError(f"unit must be (n, 3), got {unit.shape}.")
        if index.shape != (len(unit),):
            raise ValueError(
                f"grid_index must be ({len(unit)},), got {index.shape}."
            )
        if int(self.n_grid) < len(unit):
            raise ValueError(
                "n_grid must be at least the number of survivors, got "
                f"{self.n_grid} < {len(unit)}."
            )
        if np.any(np.diff(index) <= 0):
            raise ValueError("grid_index must be strictly increasing.")

    @property
    def n_oa(self) -> int:
        """Return the number of admissible directions ``N_OA``."""
        return int(len(self.unit))

    @property
    def fraction(self) -> float:
        """Return the admissible fraction ``f_s = N_OA / N``."""
        return float(self.n_oa) / float(self.n_grid)

    @property
    def accessible_solid_angle(self) -> float:
        """Return the accessible solid angle estimate ``4*pi*N_OA/N``."""
        return 4.0 * np.pi * self.fraction


def filter_grid(
    center: np.ndarray,
    radius: float,
    filter_positions: np.ndarray,
    filter_radii: np.ndarray,
    cell: np.ndarray | None,
    pbc: np.ndarray | None,
    n: int,
    *,
    grid: DirectionGrid | None = None,
    skin: float = 1.0,
) -> AdmissibleSet:
    """Filter a direction grid against a set of atoms.

    Args:
        center: Site centre ``O_s``, shape ``(3,)``.
        radius: Direction-sphere radius ``r_s``.
        filter_positions: Positions of the steric filter set, shape ``(m, 3)``.
        filter_radii: Covalent radii of the filter set, shape ``(m,)``.
        cell: Optional ``(3, 3)`` cell matrix.
        pbc: Optional ``(3,)`` periodic-axis mask.
        n: Requested lattice size.
        grid: Direction grid. Defaults to the spherical Fibonacci lattice.
        skin: Steric margin in Angstrom.

    Returns:
        AdmissibleSet: The surviving directions and their grid indices.

    Raises:
        ValueError: If the radius is not positive, the margin is negative, or
            the filter set is empty.
    """
    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (3,):
        raise ValueError(f"center must be (3,), got {center_array.shape}.")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}.")
    if not np.isfinite(skin) or skin < 0.0:
        raise ValueError(f"skin must be non-negative, got {skin}.")
    positions = np.atleast_2d(np.asarray(filter_positions, dtype=float))
    radii = np.asarray(filter_radii, dtype=float)
    if radii.shape != (len(positions),):
        raise ValueError(
            "filter_radii must have one entry per filter atom, got "
            f"{radii.shape} for {len(positions)} atoms."
        )
    if len(positions) == 0:
        raise ValueError("The steric filter set must not be empty.")

    active_grid: DirectionGrid = FibonacciGrid() if grid is None else grid
    n_grid = int(active_grid.size(n))
    unit = np.asarray(active_grid.points(n), dtype=float)
    if unit.shape != (n_grid, 3):
        raise ValueError(
            "The grid returned a point set of the wrong shape: "
            f"{unit.shape} for size {n_grid}."
        )

    probe = center_array + float(radius) * unit
    distances = distances_mic(probe, positions, cell, pbc)
    keep = np.all(distances > (radii + float(skin))[None, :], axis=1)
    indices = np.flatnonzero(keep)
    return AdmissibleSet(unit=unit[indices], grid_index=indices, n_grid=n_grid)


def admissible_directions(
    site: SiteGeometry,
    n: int,
    *,
    grid: DirectionGrid | None = None,
    scale: float | None = None,
    skin: float | None = None,
) -> AdmissibleSet:
    """Return the admissible directions of a site.

    Args:
        site: Site geometry.
        n: Requested lattice size.
        grid: Direction grid. Defaults to the spherical Fibonacci lattice.
        scale: Override of the site's direction-sphere scale.
        skin: Override of the site's steric margin.

    Returns:
        AdmissibleSet: The admissible directions.
    """
    if scale is None:
        radius = site.radius
    else:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}.")
        cov = np.asarray(site.core_numbers, dtype=int)
        from ase.data import covalent_radii

        radius = float(np.max(covalent_radii[cov])) * float(scale)
    margin = site.skin if skin is None else skin
    return filter_grid(
        center=site.center,
        radius=radius,
        filter_positions=site.filter_positions,
        filter_radii=site.filter_radii,
        cell=site.cell_array(),
        pbc=site.pbc_array(),
        n=n,
        grid=grid,
        skin=float(margin),
    )
