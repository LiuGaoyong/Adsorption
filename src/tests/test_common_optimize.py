# ruff: noqa:  D103
import numpy as np
from ase.build import add_adsorbate, fcc100, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from adsorption.common.optimize import (
    OPTIMIZE_METHODS,
    call_dimer,
    call_neb,
    optimize,
)


def test_optimize_methods() -> None:
    for k, cls in OPTIMIZE_METHODS.items():
        print(k, cls)


def test_optimize() -> None:
    result_lst, converged = optimize(
        molecule("C6H6"),
        EMT(),
        logfile="-",
        fmax=1e-4,
    )
    print(converged)
    print(result_lst[-1])
    for i, r in enumerate(result_lst):
        print(i, r.get_potential_energy())
    assert len(result_lst) == 10
    assert converged


def test_dimer() -> None:
    atoms = fcc100("Pt", size=(2, 2, 1), vacuum=10.0)
    add_adsorbate(atoms, "Pt", 1.611, "hollow")
    mask = [atom.tag > 0 for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))
    displacement_vector = [[0.0] * 3] * 5
    displacement_vector[-1][1] = -0.1
    displacement_vector = np.array(displacement_vector)

    result_lst, converged = call_dimer(
        atoms,
        EMT(),
        displacement=displacement_vector,
        logfile="-",
        # trajectory="dimer.traj",
        # append_trajectory=False,
        fmax=0.01,
    )
    for i, r in enumerate(result_lst):
        print(i, r.get_potential_energy())
    assert len(result_lst) == 19
    assert converged


def test_neb() -> None:
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.center(axis=2, vacuum=4.0)
    mask = [atom.tag > 1 for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))

    last = slab.copy()
    last[-1].x += slab.cell.array[0, 0] / 2

    lst, converged = call_neb(
        slab,
        EMT(),
        last,
        logfile="-",
        trajectory=None,
        method4opt="BFGS",
        nimages=5,
    )
    assert len(lst) == 5 * 17
    assert converged
    arr = np.array([at.get_potential_energy() for at in lst]).reshape(-1, 5)
    np.set_printoptions(precision=5)
    print(arr)
