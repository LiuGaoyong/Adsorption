from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as Rot

from adsorption.interface import add_adsorbate, add_adsorbate_and_optimize


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    # return Octahedron("Cu", 10)
    p = Path(__file__).parent / "OctCu10.xyz"
    return read(p.__fspath__())  # type: ignore


@pytest.mark.parametrize("adsorbate", ["CO", "H2O", "CH4", "C6H6"])
@pytest.mark.parametrize(
    "core",
    [
        [303, 334, 464],  # vertex fcc hollow
        [303, 334],  # vertex bridge
        303,  # vertex top
        578,  # edge top
        [578, 638],  # edge bridge
        [578, 638, 596],  # edge fcc hollow
        [607, 608, 610],  # surface fcc hollow
        [608, 610],  # surface bridge
        [610],  # surface top
    ],
)
def test_add_adsorbate(  # noqa: D103
    atoms,
    adsorbate,
    core: int | list[int],
) -> None:
    m: Atoms = molecule(adsorbate)
    d = 1.7 + np.random.random() * 0.7
    dpdist = np.asarray(pdist(m.positions))

    for _ in range(10):
        r0, r1 = Rot.random(), Rot.random()
        ads0 = add_adsorbate(atoms, m, core, d, r0, r1)
        _core = [core] if isinstance(core, int) else core
        com_core0 = Atoms(ads0[_core]).get_center_of_mass()
        com_ads0 = Atoms(ads0[-len(m) :]).get_center_of_mass()
        dpdist0 = np.asarray(pdist(ads0.positions[-len(m) :]))
        assert pytest.approx(dpdist0, abs=1e-7) == dpdist
        d0 = float(np.linalg.norm(com_ads0 - com_core0))
        assert pytest.approx(d0, abs=1e-7) == d


@pytest.mark.parametrize("adsorbate", ["CO", "H2O", "CH4", "C6H6"])
@pytest.mark.parametrize(
    "core,name",
    [
        ([303, 334, 464], "v_fcc"),  # vertex fcc hollow
        ([303, 334], "v_bri"),  # vertex bridge
        (303, "v_top"),  # vertex top
        (578, "e_top"),  # edge top
        ([578, 638], "e_bri"),  # edge bridge
        ([578, 638, 596], "e_fcc"),  # edge fcc hollow
        ([607, 608, 610], "s_fcc"),  # surface fcc hollow
        ([608, 610], "s_bri"),  # surface bridge
        ([610], "s_top"),  # surface top
    ],
)
@pytest.mark.parametrize("calculator", ["lj", "gfnff", "gfn0"])
def test_add_adsorbate_and_optimize(  # noqa: D103
    atoms,
    adsorbate,
    core: int | list[int],
    calculator: str,
    name: str,
) -> None:  # noqa: D103
    print()
    k = f"{adsorbate}_{name}_{calculator}"
    t0 = perf_counter()
    try:
        result = add_adsorbate_and_optimize(
            atoms=atoms,
            adsorbate=molecule(adsorbate),
            calculator=calculator,
            core=core,
        )
        result.write(f"{k}.xyz", format="extxyz")
        print(f"  Write: {Path(__file__).parent / f'{k}.xyz'}")
    except Exception as e:
        msg = f"  No success: for {k} because of {e}"
        with open(f"{k}.error", "w") as f:
            f.write(msg)
        print(msg)
    print(f"  Time({k}) = {perf_counter() - t0:.4f} s")
