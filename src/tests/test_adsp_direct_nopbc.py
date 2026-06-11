# ruff: noqa: E501 D103
import shutil
from pathlib import Path

import pytest
from ase import Atoms
from ase.cluster import Octahedron
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from adsorption.interfaces import DirectAdsorption
from adsorption.interfaces._direct import get_grid_and_anchor_of_core
from adsorption.interfaces._test import (
    test_add_adsorbate_and_optimize as _test_add_,
)


def _plot(atoms: Atoms, core: list[int], name: str, ax: Axes) -> Atoms:
    grid, anchor = get_grid_and_anchor_of_core(atoms, core, 100)
    new_atoms = atoms.copy()
    new_atoms.numbers[core] = 79
    new_atoms.extend(Atoms([0] * len(grid), grid))
    plot_atoms(new_atoms, ax=ax)
    ax.set_title(f"{name}")
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    return new_atoms


def test_get_grid_of_core_Octahedron(atoms) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(6, 4), dpi=150)
    atoms = Octahedron("Cu", 5)
    for ax, core, name in zip(
        axes.flatten(),
        [
            [61],
            [61, 44],
            [61, 24, 44],
            [60],  # fcc-top
            [60, 67],  # fcc-bri
            [60, 79, 67],  # fcc
        ],
        [
            "Vertex Top",
            "Vertex Bridge",
            "Vertex 3-Fold",
            "FCC Top",
            "FCC Bridge",
            "FCC 3-Fold",
        ],
    ):
        assert isinstance(ax, Axes)
        _plot(atoms, core, name, ax)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"))
    plt.close(fig)


@pytest.fixture(scope="module")
def atoms() -> Atoms:  # noqa: D103
    return Octahedron("Cu", 10)


@pytest.fixture(scope="module")
def result_dir() -> Path:  # noqa: D103
    p = Path(__file__).parent
    p = p / ".test.direct.results.nopbc"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True)
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*\n")
    return p


# @pytest.mark.skip("Run once enough.")
@pytest.mark.parametrize(
    "adsorbate",
    [
        "O",
        "CO",
        "H2O",
        "CH4",
        "C6H6",
        "C2H6",
        "CH3OH",
        "CH3CH2OH",
        "C2H4",
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
    _test_add_(
        atoms=atoms,
        adsorbate=adsorbate,
        core=core,
        cls=DirectAdsorption,
        calculator=None,
        result_dir=result_dir,
        name=name,
    )
