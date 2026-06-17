# ruff: noqa D103
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.cluster import Octahedron


@pytest.fixture(scope="session")
def atoms_nopbc_pbc() -> tuple[Atoms, Atoms]:
    return Atoms(Octahedron("Cu", 10)), fcc111(
        "Cu",
        (10, 10, 5),
        orthogonal=True,
        periodic=True,
        vacuum=10,
    )


@pytest.fixture(scope="session")
def result_dir() -> Path:
    p0 = Path(__file__).parent
    p = p0.parent / "test-results"
    p.mkdir(exist_ok=True)
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*\n")
    for pp in p.rglob("*.error"):
        pp.unlink()
    return p
