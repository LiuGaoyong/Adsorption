from typing import Literal, override

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.data import covalent_radii as COV_R
from ase.geometry import find_mic
from graphatoms.geometry import neighbor_list
from graphatoms.geometry.sample import fibonacci_lattice
from graphatoms.system import Cluster, Gas, System
from numpy.typing import ArrayLike

from ..common import AdsorptionABC


class DirectAdsorption(AdsorptionABC):
    def __init__(
        self,
        calculator: Calculator | None = None,
        *,
        nfibonacci: int = 1000,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        super().__init__(
            calculator=calculator,
            max_steps_for_first_stage=max_steps_for_first_stage,
            max_steps_for_second_stage=max_steps_for_second_stage,
            max_force=max_force,
            debug=debug,
        )
        self.__nfibonacci = int(nfibonacci)

    def _combine(
        self,
        atoms: Atoms,
        adsorbate: Atoms,
        *,
        core: ArrayLike | None = 0,
        idx_grid_core: int | None = None,
        grid_core: np.ndarray | None = None,
        anchor_core: np.ndarray | None = None,
        grid_ads: np.ndarray | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
    ) -> Atoms:
        """Combine the substrate and adsorbate."""
        gas = adsorbate
        # A. get the direction of `adsorbate`
        if grid_ads is None:
            grid_ads, anchor_ads = self.__get_grids(adsorbate, None)
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
            grid_core, anchor_core = self.__get_grids(atoms, core)
        else:
            assert anchor_core is not None, "anchor_core must be provided."
            anchor_core = np.asarray(anchor_core, dtype=float)
            grid_core = np.asarray(grid_core, dtype=float)
        assert grid_core.ndim == 2 and grid_core.shape[1] == 3
        if idx_grid_core is None:
            idx_grid_core = np.random.randint(len(grid_core))
        idx_grid_core = int(idx_grid_core)
        direction_core = grid_core[idx_grid_core] - anchor_core
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
            - anchor_ads  # 1. move adsorbate to the zero position
            + anchor_core  # 2. move adsorbate to the core position
            + direction_core * distance  # 3. move adsorbate
        )

        # save some information
        self._adsorbate_pos = adsorbate.positions.copy()
        self._direction_core = direction_core
        self._anchor_core = anchor_core
        self._anchor_ads = anchor_ads
        self._distance = distance

        result = atoms.copy()
        result.extend(gas)
        return result

    @override
    def __call__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        *,
        core: ArrayLike | None = 0,
        idx_grid_core: int | None = None,
        grid_core: np.ndarray | None = None,
        anchor_core: np.ndarray | None = None,
        grid_ads: np.ndarray | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
    ) -> tuple[Atoms, Literal[0, 1, 2]]:
        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        result = self._combine(
            atoms=atoms,
            adsorbate=self._get_adsorbate(adsorbate).copy(),
            core=core,
            idx_grid_core=idx_grid_core,
            grid_core=grid_core,
            anchor_core=anchor_core,
            grid_ads=grid_ads,
            idx_grid_ads=idx_grid_ads,
            distance=distance,
        )
        return self._opt(
            natoms=len(atoms),
            atoms=result,
        )

    def __get_grids(
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
        if len(core) == len(atoms):
            assert not atoms.pbc.any(), "PBC is not supported for 'COM'."
            anchor = np.mean(atoms.positions[core], axis=0)
            grid = fibonacci_lattice(self.__nfibonacci) + anchor
        else:
            grid, anchor = get_grid_and_anchor_of_core(
                atoms=atoms,
                select_core=core,
                nfibonacci=self.__nfibonacci,
            )
        return grid, anchor

    def grid_generation(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        *,
        core: ArrayLike | None = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the grid of core and adsorbate.

        Returns:
            grid_core: The grid of core.
            grid_ads: The grid of adsorbate.
            anchor_core: The anchor of core.
        """
        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        adsorbate = self._get_adsorbate(adsorbate)
        grid_ads, _ = self.__get_grids(adsorbate, None)
        grid_core, anchor_core = self.__get_grids(atoms, core)
        return grid_core, grid_ads, anchor_core


def get_grid_and_anchor_of_core(
    atoms: Atoms,
    select_core: int | list[int] | np.ndarray,
    nfibonacci: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(select_core, int):
        select_core = [select_core]
    core = np.asarray(select_core, dtype=int)
    core = np.unique(core)

    # move core atoms if mic
    _MIC_POS: np.ndarray = np.zeros(3)
    if any(atoms.get_pbc()) and len(core) > 1:
        _MIC_POS = atoms.cell.cartesian_positions([0.5, 0.5, 0.5])
        _MIC_POS -= atoms.positions[core[0]]
        atoms = atoms.copy()
        atoms.translate(_MIC_POS)
        atoms.wrap(pbc=True)
    assert _MIC_POS.shape == (3,)

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
    anchor = atoms.positions[core].mean(axis=0)
    grid: np.ndarray = anchor + grid
    if base_direction is not None:
        base_direction = np.asarray(base_direction)
        grid += base_direction.flatten()[:3]
    v = pos[:, np.newaxis, :] - grid[np.newaxis, :, :]  # (n_pos, n_grid, 3)
    _, vlen = find_mic(v.reshape(-1, 3), atoms.cell, True)
    d = vlen.reshape(v.shape[:2])

    matrix_cov_r = np.column_stack([cov_r] * len(grid))
    cond = np.all(matrix_cov_r + float(skin) < d, axis=0)
    grid = grid[cond] - _MIC_POS
    anchor = anchor - _MIC_POS
    return grid, anchor
