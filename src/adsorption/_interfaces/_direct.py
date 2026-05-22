from typing import Literal, override

import numpy as np
from ase import Atom, Atoms
from ase.data import covalent_radii as COV_R
from ase.geometry import find_mic
from graphatoms.geometry import neighbor_list
from graphatoms.geometry.sample import fibonacci_lattice
from graphatoms.system import Cluster, Gas, System
from numpy.typing import ArrayLike

from .._abc import AdsorptionABC


class DirectAdsorption(AdsorptionABC):
    @override
    def __call__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        adsorbate_index: Literal["com"] | int | None = None,
        core: ArrayLike | list[int] | int = 0,
    ) -> Atoms:
        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        raise NotImplementedError


def get_grid_of_core(
    atoms: Atoms,
    select_core: int | list[int] | np.ndarray,
    nfibonacci: int = 1000,
) -> np.ndarray:
    if isinstance(select_core, int):
        select_core = [select_core]
    core = np.asarray(select_core, dtype=int)
    core = np.unique(core)

    skin = 0.5
    scale = 1.5
    base_direction = None
    neighbors_exclude_core = False
    if len(core) == 1:
        neighbors_exclude_core = True
        skin, scale = 1.0, 1.2
    # elif len(core) == 2:
    #     skin, scale = 1.0, 1.6

    cov_core = COV_R[atoms.numbers[core]].max()
    grid = fibonacci_lattice(nfibonacci) * cov_core * float(scale)
    i, j = neighbor_list("ij", atoms, 5.0, self_interaction=False)
    cond = np.logical_and(np.isin(i, core), np.logical_not(np.isin(j, core)))
    nbrs = np.unique(np.append(np.append(i[cond], j[cond]), core).astype(int))
    if neighbors_exclude_core:
        nbrs = np.setdiff1d(nbrs, core.astype(int))
    cov_r = COV_R[atoms.numbers[nbrs]]
    pos = atoms.positions[nbrs]

    # calculate distance by minimum-image representation
    grid += atoms.positions[core].mean(axis=0)
    if base_direction is not None:
        base_direction = np.asarray(base_direction)
        grid += base_direction.flatten()[:3]
    v = pos[:, np.newaxis, :] - grid[np.newaxis, :, :]  # (n_pos, n_grid, 3)
    _, vlen = find_mic(v.reshape(-1, 3), atoms.cell, True)
    d = vlen.reshape(v.shape[:2])

    matrix_cov_r = np.column_stack([cov_r] * len(grid))
    cond = np.all(matrix_cov_r + float(skin) < d, axis=0)
    return grid[cond]


def test_get_grid_of_core() -> None:
    from pathlib import Path

    from ase import Atoms
    from ase.cluster import Octahedron
    from ase.visualize.plot import plot_atoms
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

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
        grid = get_grid_of_core(atoms, core, 100)
        new_atoms = atoms.copy()
        new_atoms.numbers[core] = 79
        new_atoms.extend(Atoms([0] * len(grid), grid))
        plot_atoms(new_atoms, ax=ax)
        ax.set_title(f"{name}")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"))


if __name__ == "__main__":
    test_get_grid_of_core()
