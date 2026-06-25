# ruff: noqa D103
from pathlib import Path
from time import perf_counter

import pytest
from ase import Atoms
from adsorption.interfaces import RawAdsorption
from adsorption.common import AdsorptionABC
from adsorption.interfaces import DirectAdsorption
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.io import write


_complex_molecule = [
    "C6H6",
    "C2H6",
    "CH3CH2OH",
    "CH3OH",
]


@pytest.mark.parametrize(
    "adsorbate",
    [
        "O",
        "CO",
        "H2O",
        "CH4",
    ]
    + _complex_molecule,
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
@pytest.mark.parametrize(
    "ADS_CLS,sub_dir,calc",
    [
        (RawAdsorption, "raw-None", None),
        (DirectAdsorption, "direct-None", None),
        (DirectAdsorption, "direct-EMT", EMT()),
        # (DirectAdsorptionAD, "direct_ad-EMT", EMT()),
    ],
)
def test_add_adsorption_class_and_optimize(  # noqa: D103
    atoms_nopbc_pbc: tuple[Atoms, Atoms],
    adsorbate: str,
    use_pbc: bool,
    core: list[int],
    result_dir: Path,
    sub_dir: str,
    ADS_CLS: type[AdsorptionABC],
    calc: Calculator | None,
    name: str,
) -> None:  # noqa: D103
    if "raw" in sub_dir and adsorbate in _complex_molecule:
        return

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

    result_dir = result_dir.joinpath(sub_dir)
    result_dir.mkdir(exist_ok=True)

    try:
        obj = ADS_CLS(
            calculator=calc,
            atoms=atoms,
            core=core,
        )
        result = obj(adsorbate=adsorbate)[0]
        result.numbers[core] = 79
        fname = result_dir.joinpath(f"{k}.png")
        result.write(fname, format="png")
        if hasattr(obj, "_atoms_lst"):
            write(
                result_dir.joinpath(f"{k}.xyz"),
                obj._atoms_lst,
                format="extxyz",
            )
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
