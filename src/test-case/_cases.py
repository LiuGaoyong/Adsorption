"""Shared constants of the migrated example cases.

The cases reproduce the examples that live on ``main`` in
``src/tests/test_ads_site_grid.py`` and ``src/tests/test_adsorption.py``,
with the small grids required by the test protocol: fewer than ten approach
directions (``N_OA``) and fewer than ten molecular orientations (``N_ROT``).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

N_OA = 6
"""Number of approach directions (OA) per site; must stay below ten."""

N_ROT = 4
"""Number of molecular orientations (AB) per case; must stay below ten."""

N_D = 3
"""Number of anchor distances per case."""

D_MIN_BASE = 2.0
"""Contact part of the lower distance bound, in Angstrom."""

D_MIN_RADIUS_SCALE = 0.1
"""Coefficient of the adsorbate radius in the lower distance bound."""

D_MAX_BASE = 3.5
"""Base part of the upper distance bound, in Angstrom."""

D_MAX_RADIUS_SCALE = 0.35
"""Coefficient of the adsorbate radius in the upper distance bound."""

ADSORBATES = (
    "O",
    "CO",
    "H2O",
    "CH4",
    "C6H6",
    "C2H6",
    "CH3CH2OH",
    "CH3OH",
)
"""Adsorbates of the migrated adsorption examples."""


SITE_CASES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("cluster_v_top", "cluster", (303,)),
    ("cluster_v_bri", "cluster", (303, 334)),
    ("cluster_v_fcc", "cluster", (303, 334, 464)),
    ("cluster_e_top", "cluster", (578,)),
    ("cluster_e_bri", "cluster", (578, 638)),
    ("cluster_e_fcc", "cluster", (578, 596, 638)),
    ("cluster_s_top", "cluster", (610,)),
    ("cluster_s_bri", "cluster", (608, 610)),
    ("cluster_s_fcc", "cluster", (607, 608, 610)),
    ("slab_s_top", "slab", (400,)),
    ("slab_s_bri", "slab", (400, 490)),
    ("slab_s_fcc", "slab", (400, 490, 499)),
)
"""The twelve site examples (name, fixture, core indices).

The nine ``cluster_*`` cases use the non-periodic ``Octahedron("Cu", 10)``
cluster, the three ``slab_*`` cases use the periodically repeated
``fcc111("Cu", (10, 10, 5))`` slab.  The core indices are those of the
examples on ``main``.
"""


def centered_view(
    atoms: Atoms, center: np.ndarray
) -> tuple[Atoms, np.ndarray]:
    """Return a plotting copy whose periodic site is centred in the cell.

    A slab site that sits on a cell boundary is split by :meth:`Atoms.wrap`;
    translating the whole structure so that the site centre coincides with
    the cell centre keeps the core atoms and the direction grid in one piece.

    Args:
        atoms: The structure to view.
        center: The site centre to bring to the cell centre.

    Returns:
        tuple[Atoms, np.ndarray]: The view and the applied translation.
    """
    view = atoms.copy()
    shift = np.zeros(3)
    if any(view.pbc):
        shift = np.asarray(view.cell.array, dtype=float).sum(axis=0) / 2.0
        shift = shift - np.asarray(center, dtype=float)
        view.translate(shift)
        view.wrap(pbc=True)
    return view, shift
