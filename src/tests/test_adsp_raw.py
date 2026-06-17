# ruff: noqa D103
import shutil
from pathlib import Path
from time import perf_counter

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.cluster import Octahedron
from adsorption.runner._tune import plot

from adsorption.interfaces import RawAdsorption


@pytest.fixture(scope="module")
def atoms_nopbc_pbc() -> tuple[Atoms, Atoms]:
    return Atoms(Octahedron("Cu", 10)), fcc111(
        "Cu",
        (10, 10, 5),
        orthogonal=True,
        periodic=True,
        vacuum=10,
    )


@pytest.fixture(scope="module")
def result_dir() -> Path:
    p0 = Path(__file__)
    p = p0.parent / p0.name.split(".")[0]
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True)
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*\n")
    return p


@pytest.mark.parametrize(
    "adsorbate",
    [
        "O",
        "CO",
        "H2O",
        "CH4",
        # "C6H6",
    ],
)
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
def test_add_adsorbate_and_optimize(  # noqa: D103
    atoms_nopbc_pbc: tuple[Atoms, Atoms],
    adsorbate: str,
    use_pbc: bool,
    core: int | list[int],
    result_dir: Path,
    name: str,
) -> None:  # noqa: D103
    print()
    k = f"{adsorbate}_{name}"
    k = f"pbc_{k}" if use_pbc else f"nopbc_{k}"
    t0 = perf_counter()
    result_dir.mkdir(exist_ok=True)
    if use_pbc:
        atoms = atoms_nopbc_pbc[1]
        assert all(atoms.pbc), atoms
    else:
        atoms = atoms_nopbc_pbc[0]
        assert not any(atoms.pbc), atoms

    try:
        obj = RawAdsorption(calculator=None)
        result = obj(atoms=atoms, adsorbate=adsorbate, core=core)[0]
        result.numbers[core] = 79
        fname = result_dir.joinpath(f"{k}.png")
        result.write(fname, format="png")
        print(f"  Write: {fname}")
    except Exception as e:
        msg = f"  No success: for {k} because of {e}"
        fname = result_dir.joinpath(f"{k}.error")
        with fname.open("w") as f:
            f.write(msg)
        print(msg)
        raise e
    finally:
        print(f"  Time({k}) = {perf_counter() - t0:.4f} s")
