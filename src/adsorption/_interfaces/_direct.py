from typing import override

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.data import covalent_radii as COV_R
from ase.geometry import find_mic
from graphatoms.geometry import neighbor_list
from graphatoms.geometry.sample import fibonacci_lattice
from graphatoms.system import Cluster, Gas, System
from numpy.typing import ArrayLike

from .._abc import AdsorptionABC


class DirectAdsorption(AdsorptionABC):
    def __init__(
        self,
        calculator: Calculator | None = None,
        *,
        nfibonacci: int = 1000,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = True,
    ) -> None:
        super().__init__(
            calculator=calculator,
            max_steps_for_first_stage=max_steps_for_first_stage,
            max_steps_for_second_stage=max_steps_for_second_stage,
            max_force=max_force,
            debug=debug,
        )
        self.__nfibonacci = int(nfibonacci)

    @override
    def __call__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        *,
        core: ArrayLike | None = 0,
        idx_grid_core: int | None = None,
        grid_core: np.ndarray | None = None,
        grid_ads: np.ndarray | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
    ) -> Atoms:
        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        adsorbate = gas = self._get_adsorbate(adsorbate).copy()

        # A. get the direction of `adsorbate`
        if grid_ads is None:
            grid_ads, anchor_ads = self._get_grids(adsorbate, None)
        else:
            grid_ads = np.asarray(grid_ads, dtype=float)
            anchor_ads = np.mean(adsorbate.positions, axis=0)
        assert grid_ads.ndim == 2 and grid_ads.shape[1] == 3
        assert len(grid_ads) == self.__nfibonacci
        if idx_grid_ads is None:
            idx_grid_ads = np.random.randint(len(grid_ads))
        idx_grid_ads = int(idx_grid_ads)
        # B. rotate adsorbate
        center = anchor_ads
        adsorbate.rotate(
            center + [0, 0, 1],
            grid_ads[idx_grid_ads],
            rotate_cell=False,
            center=center,
        )

        # C get the direction of core
        if grid_core is None:
            grid_core, anchor_core = self._get_grids(atoms, core)
        else:
            if isinstance(core, int):
                core = np.asarray([core])
            core = np.asarray(core, dtype=int)
            core = np.unique(core.flatten())
            anchor_core = np.mean(atoms.positions[core], axis=0)
            grid_core = np.asarray(grid_core, dtype=float)
        assert grid_core.ndim == 2 and grid_core.shape[1] == 3
        if idx_grid_core is None:
            idx_grid_core = np.random.randint(len(grid_core))
        idx_grid_core = int(idx_grid_core)
        direction_core = grid_core[idx_grid_core]
        direction_core /= np.linalg.norm(direction_core)

        # place gas into
        if distance is None:
            d_gas = gas.positions - gas.positions.mean(axis=0)
            d_gas_max: float = np.max(np.linalg.norm(d_gas, axis=0))
            d_gas_min: float = np.max(COV_R[gas.numbers])
            v_core = grid_core - anchor_core
            _, d_core = find_mic(v_core, atoms.cell)
            distance = np.mean(d_core) + d_gas_min  # type: ignore
            distance += 0.5 * (d_gas_max - d_gas_min)  # type: ignore
        assert isinstance(distance, float)

        adsorbate.set_positions(
            adsorbate.positions
            - anchor_ads  #
            + anchor_core
            + direction_core * distance
        )
        result = atoms.copy()
        result.extend(gas)
        result_lst: list[Atoms] = [result]
        if self.calculator is not None:
            result_1, converged_1 = self._first_stage_opt(
                natoms=len(atoms),
                result=result,
                calc=self.calculator,
                fmax=self.max_force,
                max_steps=self.max_steps_for_first_stage,
                debug=self.debug,
            )
            if converged_1:
                result_lst.extend(result_1)
                result_2, converged_2 = self._second_stage_opt(
                    result=result_1[-1],
                    calc=self.calculator,
                    fmax=self.max_force,
                    max_steps=self.max_steps_for_second_stage,
                )
                if converged_2:
                    result_lst.extend(result_2)
        return result_lst[-1]

    def _get_grids(
        self,
        atoms: Atoms,
        core: ArrayLike | None = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(core, int):
            core = np.asarray([core])
        elif core is None:
            core = np.arange(len(atoms))
        core = np.asarray(core, dtype=int)
        core = np.unique(core.flatten())
        anchor = np.mean(atoms.positions[core], axis=0)

        if len(core) != len(atoms):
            grid = get_grid_of_core(
                atoms=atoms,
                select_core=core,
                nfibonacci=self.__nfibonacci,
            )
        else:
            grid = fibonacci_lattice(self.__nfibonacci) + anchor
        return grid, anchor


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
