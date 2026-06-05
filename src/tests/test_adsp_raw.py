import shutil
from pathlib import Path
from time import perf_counter

import pytest
from ase import Atoms
from ase.cluster import Octahedron
from ase.io import read

from adsorption._interfaces import RawAdsorption


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    return Octahedron("Cu", 10)
    p = Path(__file__).parent / "OctCu10.xyz"
    return read(p.__fspath__())  # type: ignore


@pytest.fixture(scope="module")
def result_dir() -> Path:  # noqa: D103
    p = Path(__file__).parent
    p /= ".test.raw.results"
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
def test_add_adsorbate_and_optimize(  # noqa: D103
    atoms,
    adsorbate,
    core: int | list[int],
    result_dir: Path,
    name: str,
) -> None:  # noqa: D103
    print()
    k = f"{adsorbate}_{name}"
    t0 = perf_counter()
    result_dir.mkdir(exist_ok=True)
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
