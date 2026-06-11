# ruff: noqa: E501 D103
import os.path
import shutil
from pathlib import Path
from time import perf_counter

import pytest
from ase import Atoms
from ase.cluster import Octahedron

from adsorption.interfaces._directAD import DirectAdsorptionAD


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    return Octahedron("Cu", 10)


@pytest.fixture(scope="module")
def result_dir() -> Path:  # noqa: D103
    p = Path(__file__).parent
    p = p / ".test.direct.results.ad"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True)
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*\n")
    return p


# @pytest.mark.skip("Run once enough.")
@pytest.mark.parametrize(
    "adsorbate",
    [
        "C6H6",
        # "CH4",
        # "H2O",
        # "CO",
        # "O",
        # "C2H6",
        # "CH3OH",
        # "CH3CH2OH",
        # "C2H4",
    ],
)
@pytest.mark.parametrize(
    "core,name",
    [
        ([303, 334, 464], "v_fcc"),  # vertex fcc hollow
        # ([303, 334], "v_bri"),  # vertex bridge
        # (303, "v_top"),  # vertex top
        # (578, "e_top"),  # edge top
        # ([578, 638], "e_bri"),  # edge bridge
        # ([578, 638, 596], "e_fcc"),  # edge fcc hollow
        # ([607, 608, 610], "s_fcc"),  # surface fcc hollow
        # ([608, 610], "s_bri"),  # surface bridge
        # ([610], "s_top"),  # surface top
    ],
)
def test_add_adsorbate_and_optimize(  # noqa: D103
    atoms,
    adsorbate,
    core: int | list[int],
    result_dir: Path,
    name: str,
) -> None:  # noqa: D103
    try:
        from nequip.integrations.ase import NequIPCalculator

        print()
        k = f"{name}_{adsorbate}"
        result_dir.mkdir(exist_ok=True)
        t0 = perf_counter()
        try:
            obj = DirectAdsorptionAD(
                calculator=NequIPCalculator.from_compiled_model(
                    compile_path=os.path.expanduser(
                        "~/.local/nequip-oam-0.1/NequIP-OAM-S-0.1.nequip.pth"
                    ),
                    chemical_species_to_atom_type_map=True,
                    device="cpu",
                )
            )
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
    except ImportError:
        return
