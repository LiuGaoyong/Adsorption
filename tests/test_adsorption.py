from pathlib import Path
from time import perf_counter

import pytest
from ase import Atoms
from ase.io import read

from adsorption import Adsorption


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    # return Octahedron("Cu", 10)
    p = Path(__file__).parent / "OctCu10.xyz"
    return read(p.__fspath__())  # type: ignore


@pytest.fixture(scope="module")
def result_dir() -> Path:  # noqa: D103
    p = Path(__file__).parent.parent / "results"
    p.mkdir(exist_ok=True)
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
@pytest.mark.parametrize(
    "mode", ["guess", "guess_opt_total", "guess_opt_sub"]
)  # , "ase", "scipy", "bayesian"])
# @pytest.mark.parametrize("mode", ["guess", "scipy", "bayesian"])
def test_add_adsorbate_and_optimize(  # noqa: D103
    atoms,
    adsorbate,
    core: int | list[int],
    result_dir: Path,
    mode: str,
    name: str,
) -> None:  # noqa: D103
    print()
    k = f"{adsorbate}_{name}"
    t0 = perf_counter()
    result_dir = result_dir.joinpath(f"png_{mode}")
    result_dir.mkdir(exist_ok=True)
    try:
        obj = Adsorption(atoms, adsorbate, "gfnff", core)
        result = obj(mode=mode).as_ase()
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
