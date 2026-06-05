"""Test for Helper class using for-loop single process approach.

These tests verify the Helper class which provides a unified interface
for both raw and direct adsorption paths. The tests use for-loop single
process execution as requested.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster import Octahedron

from adsorption._interfaces.helper import Helper


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    return Octahedron("Cu", 10)


@pytest.fixture(scope="module")
def result_dir() -> Path:  # noqa: D103
    p = Path(__file__).parent
    p = p / ".test.helper.results"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True)
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*\n")
    return p


def test_helper_initialization(atoms: Atoms, result_dir: Path) -> None:
    """Test Helper class initialization and nrun calculation."""
    obj = Helper(
        calculator=EMT(),
        atoms=atoms,
        adsorbate="O",
        core=[303, 334, 464],
        use_direct=True,
        use_raw=True,
        nfibonacci=10,
        max_steps_for_first_stage=10,
        max_steps_for_second_stage=10,
        distance_lst=np.array([2.0, 2.5]),
    )
    assert obj.nrun > 0
    print(f"  nrun = {obj.nrun}")


def test_helper_raw_path_single_iteration(
    atoms: Atoms, result_dir: Path
) -> None:
    """Test Helper raw adsorption path with single iteration."""
    core = [303, 334, 464]
    obj = Helper(
        calculator=EMT(),
        atoms=atoms,
        adsorbate="O",
        core=core,
        use_direct=True,
        use_raw=True,
        nfibonacci=10,
        max_steps_for_first_stage=0,
        max_steps_for_second_stage=0,
        distance_lst=np.array([2.0, 2.5]),
    )

    # irun=0 triggers raw path (after irun -= 1 becomes -1)
    result = obj(irun=0, outdir=result_dir)

    assert "score" in result
    assert "fmax" in result
    assert "nstage" in result
    assert "atoms" in result
    print(
        f"  Raw path result: score={result['score']:.4f}, "
        f"fmax={result['fmax']:.4f}, nstage={result['nstage']}"
    )
    fname = result_dir.joinpath("0-0.png")
    result["atoms"].numbers[core] = 79
    result["atoms"].write(fname, format="png")


def test_helper_for_loop_single_process_raw(
    atoms: Atoms, result_dir: Path
) -> None:
    """Test Helper using for-loop single process with raw path only.

    This test demonstrates the for-loop single process approach
    using the raw adsorption path, which works correctly.
    """
    core = [303, 334, 464]
    obj = Helper(
        calculator=EMT(),
        atoms=atoms,
        adsorbate="O",
        core=core,
        use_direct=True,
        use_raw=True,
        nfibonacci=10,
        max_steps_for_first_stage=0,
        max_steps_for_second_stage=0,
        distance_lst=np.array([2.0, 2.5]),
    )

    print(f"  Total runs: {obj.nrun}")
    results = []

    # Single iteration using raw path
    # irun=0 becomes -1 after irun -= 1, triggering raw path
    result = obj(irun=0, outdir=result_dir)
    results.append(result)
    print(
        f"  irun=0 (raw): score={result['score']:.4f}, "
        f"fmax={result['fmax']:.4f}, nstage={result['nstage']}"
    )

    assert len(results) == 1
    fname = result_dir.joinpath("0-1.png")
    result["atoms"].numbers[core] = 79
    result["atoms"].write(fname, format="png")


@pytest.mark.parametrize(
    "adsorbate",
    [
        "O",
        "CO",
    ],
)
@pytest.mark.parametrize(
    "core",
    [
        ([303, 334, 464],),  # vertex fcc hollow
        ([303],),  # vertex top
    ],
)
def test_helper_parametrized_raw_path(
    atoms: Atoms,
    adsorbate: str,
    core: tuple[list[int]],
    result_dir: Path,
) -> None:
    """Parametrized test for Helper with raw path."""
    obj = Helper(
        calculator=EMT(),
        atoms=atoms,
        adsorbate=adsorbate,
        core=core[0],
        use_direct=True,
        use_raw=True,
        nfibonacci=10,
        max_steps_for_first_stage=0,
        max_steps_for_second_stage=0,
        distance_lst=np.array([2.0, 2.5]),
    )
    print(f"  Total runs: {obj.nrun}")

    assert obj.nrun > 0
    for irun in range(obj.nrun):
        print(f"  irun={irun}")
        result = obj(irun=irun, outdir=result_dir)
        assert "score" in result
        assert "fmax" in result
        assert "nstage" in result
        assert "atoms" in result

        k = "_".join(map(str, core[0]))
        fname = result_dir.joinpath(f"{adsorbate}-{k}-{irun}.png")
        result["atoms"].numbers[core[0]] = 79
        result["atoms"].write(fname, format="png")
