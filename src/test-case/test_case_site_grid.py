# ruff: noqa: D103, E501
"""Visual examples of the twelve adsorption-site geometries.

Each case builds one site of the main branch, draws its direction grid with
``N_OA`` points and writes ``results/sites/<site>.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _cases import N_OA, SITE_CASES, centered_view  # type: ignore
from _plot import plot  # type: ignore
from ase import Atoms

from adsorption.geometry import FibonacciGrid, build_site

_SITE_IDS = [case[0] for case in SITE_CASES]


@pytest.mark.parametrize(
    ("name", "structure", "core"), SITE_CASES, ids=_SITE_IDS
)
def test_site_grid(
    name: str,
    structure: str,
    core: tuple[int, ...],
    request: pytest.FixtureRequest,
    result_root: Path,
) -> None:
    """Draw one site with its approach-direction grid."""
    substrate: Atoms = request.getfixturevalue(structure)
    site = build_site(substrate, core)
    unit = FibonacciGrid().points(N_OA)

    show, shift = centered_view(substrate, site.center)
    show.numbers[site.core_indices] = 79
    show.extend(
        Atoms(
            numbers=[8] * len(unit),
            positions=site.center + shift + site.radius * unit,
        )
    )
    radii = np.full(len(show), 1.3)
    radii[substrate.get_global_number_of_atoms() :] = 1.0
    png = plot(show, result_root / "sites" / name, radii=radii)

    assert png.exists()
    assert png.stat().st_size > 0
    assert len(site.core_indices) == len(core)
    assert len(site.neighbor_indices) > 0
