"""Geometric descriptors of a site and its admissible set."""

from __future__ import annotations

import numpy as np

from ..geometry import FibonacciGrid, SiteGeometry, admissible_directions

__all__ = [
    "accessible_solid_angle",
    "admissible_fraction",
    "discretization_residual",
    "high_resolution_fraction",
]


def admissible_fraction(n_oa: int, n_grid: int) -> float:
    """Return the admissible fraction ``f_s = N_OA / N``.

    Args:
        n_oa: Number of admissible directions.
        n_grid: Number of grid points.

    Returns:
        float: The fraction in ``[0, 1]``.

    Raises:
        ValueError: If the counts are invalid.
    """
    if int(n_grid) < 1 or not 0 <= int(n_oa) <= int(n_grid):
        raise ValueError(
            "Expected 0 <= N_OA <= N, got "
            f"N_OA={n_oa}, N={n_grid}."
        )
    return float(n_oa) / float(n_grid)


def accessible_solid_angle(n_oa: int, n_grid: int) -> float:
    """Return the accessible solid angle estimate ``4*pi*N_OA/N``.

    Args:
        n_oa: Number of admissible directions.
        n_grid: Number of grid points.

    Returns:
        float: The solid angle in steradians.
    """
    return 4.0 * float(np.pi) * admissible_fraction(n_oa, n_grid)


def high_resolution_fraction(
    site: SiteGeometry,
    *,
    n: int = 65536,
    grid: object | None = None,
) -> float:
    """Estimate ``f_s`` with a high-resolution deterministic grid.

    This is a finite-``N`` reference, not the analytic cap-union value: it
    replaces the discretisation error by the much smaller error of a dense
    Fibonacci set.

    Args:
        site: The site geometry.
        n: Reference lattice size.
        grid: Optional direction grid; defaults to the Fibonacci grid.

    Returns:
        float: The reference fraction in ``[0, 1]``.
    """
    directions = admissible_directions(
        site,
        int(n),
        grid=FibonacciGrid() if grid is None else grid,
    )
    return directions.fraction


def discretization_residual(
    site: SiteGeometry,
    n: int,
    *,
    n_ref: int = 65536,
    grid: object | None = None,
) -> float:
    """Return ``|f_s(N) - f_s(N_ref)|`` for one site.

    Args:
        site: The site geometry.
        n: Lattice size under test.
        n_ref: Reference lattice size.
        grid: Optional direction grid; defaults to the Fibonacci grid.

    Returns:
        float: The absolute residual.
    """
    active = FibonacciGrid() if grid is None else grid
    measured = admissible_directions(site, int(n), grid=active).fraction
    reference = high_resolution_fraction(site, n=int(n_ref), grid=active)
    return abs(measured - reference)
