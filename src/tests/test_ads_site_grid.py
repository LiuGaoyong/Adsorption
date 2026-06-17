# ruff: noqa D103
from pathlib import Path

import pytest
from ase import Atoms
import numpy as np

from adsorption.common._site import Site, _site_helper


@pytest.mark.parametrize(
    "core,name,use_pbc",
    [
        ([303, 334, 464], "v_fcc", False),  # vertex fcc hollow
        ([303, 334], "v_bri", False),  # vertex bridge
        ([303], "v_top", False),  # vertex top
        ([578], "e_top", False),  # edge top
        ([578, 638], "e_bri", False),  # edge bridge
        ([578, 638, 596], "e_fcc", False),  # edge fcc hollow
        ([607, 608, 610], "s_fcc", False),  # surface fcc hollow
        ([608, 610], "s_bri", False),  # surface bridge
        ([610], "s_top", False),  # surface top
        # use periodic boundary condition
        ([400], "s_top", True),  # surface top
        ([400, 490], "s_bri", True),  # surface bridge
        ([400, 490, 499], "s_fcc", True),  # surface fcc hollow
    ],
)
def test_show_site_grid(
    atoms_nopbc_pbc: tuple[Atoms, Atoms],
    use_pbc: bool,
    core: list[int],
    result_dir: Path,
    name: str,
) -> None:
    """Show the site grid."""
    print()
    k = f"grid_pbc_{name}" if use_pbc else f"grid_nopbc_{name}"
    result_dir.mkdir(exist_ok=True)
    if use_pbc:
        atoms = atoms_nopbc_pbc[1]
        assert all(atoms.pbc), atoms
    else:
        atoms = atoms_nopbc_pbc[0]
        assert not any(atoms.pbc), atoms

    _, core_np, nbrs_np, _, site = _site_helper(atoms, core=core)
    print(core_np, nbrs_np)
    assert isinstance(site, Site)
    grid = site.get_direction_grid(
        core_numbers=atoms.numbers[core_np],
        neighbor_numbers=atoms.numbers[nbrs_np],
        nfibonacci=100,
    )
    result_atoms: Atoms = atoms.copy()
    result_atoms.numbers[core_np] = 79
    result_atoms.extend(Atoms([0] * len(grid), grid))
    result_atoms.write(result_dir.joinpath(f"{k}.png"), format="png")
    print(f"  Write: {result_dir.joinpath(f'{k}.png')}")
