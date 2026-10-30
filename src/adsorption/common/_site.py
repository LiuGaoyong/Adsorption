"""The core data classes by pydantic."""

from functools import cached_property, reduce
from typing import Annotated, Self

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.data import covalent_radii as COV_R
from ase.geometry import complete_cell, find_mic, wrap_positions
from graphatoms.dataclasses import NDArray, numpy_validator
from graphatoms.geometry.sample import fibonacci_lattice
from graphatoms.system import Cluster, System
from pydantic import BaseModel, ConfigDict


class Site(BaseModel):
    """The site for adsorption."""

    model_config = ConfigDict(frozen=True)
    core: Annotated[NDArray, numpy_validator(float, (-1, 3))]
    neighbor: Annotated[NDArray, numpy_validator(float, (-1, 3))]
    cell: Annotated[NDArray, numpy_validator(float, (3, 3))] | None = None

    @classmethod
    def from_structure(
        cls,
        atoms: Atoms | System | Cluster,
        neighbors: npt.ArrayLike | None = None,
        core: npt.ArrayLike | int = 0,
    ) -> Self:
        """Create a site from structure.

        Args:
            atoms (Atoms | System | Cluster): The surface or
                cluster onto which the adsorbate should be added.
            neighbors (npt.ArrayLike | list[int] | None, optional):
                The first hop neighbor of core atoms.
                If None, the code will generated automated.
            core (npt.ArrayLike | list[int] | int, optional):
                The central atoms (core) which will place at.
                Defaults to the first atom, i.e. the 0-th atom.
        """
        return _site_helper(
            core=core,
            atoms=atoms,
            neighbors=neighbors,
        )[-1]  # type: ignore

    @cached_property
    def center(self) -> np.ndarray:
        """The center for adsoption."""
        if self.cell is not None and np.abs(np.linalg.det(self.cell)) > 1e-5:
            pbc = np.any(self.cell, axis=1)
            center0 = np.array([0.5, 0.5, 0.5])
            center0 = center0 @ complete_cell(self.cell)
            center0 = center0 - self.core[0]
            core = self.core + center0  # move core to the center
            core = wrap_positions(core, self.cell, pbc)  # type: ignore
            return np.mean(core, axis=0) - center0
        else:
            return np.mean(self.core, axis=0)

    @cached_property
    def direction(self) -> np.ndarray:
        """The normalized direction vector for adsorption."""
        cell = self.cell if self.cell is not None else np.zeros([3, 3])
        n2c, n2c_norm = find_mic(self.center - self.neighbor, cell)
        n2c_eye = n2c / n2c_norm[:, None]  # the unit vector of n2c
        sorted_norm = n2c_norm[np.argsort(-n2c_norm)]  # sort by norm
        sorted_eye = n2c_eye[np.argsort(n2c_norm)]  # sort by norm
        n2c_new = sorted_eye * sorted_norm[:, None]
        direction = np.mean(n2c_new, axis=0)
        direction_norm = np.linalg.norm(direction)
        return direction / direction_norm

    def get_direction_grid(
        self,
        core_numbers: np.ndarray | list[int],
        neighbor_numbers: np.ndarray | list[int],
        nfibonacci: int = 1000,
    ) -> np.ndarray:
        """The direction grid for adsorption."""
        core_numbers = np.asarray(core_numbers, dtype=int)
        assert core_numbers.shape[0] == self.core.shape[0], (
            f"Invalid core_numbers: {core_numbers}."  #
            f"The number of core_numbers={core_numbers.shape[0]}. "
            f"The number of core atoms={self.core.shape[0]}. "
        )
        neighbor_numbers = np.asarray(neighbor_numbers, dtype=int)
        assert neighbor_numbers.shape[0] == self.neighbor.shape[0], (
            f"Invalid neighbor_numbers: {neighbor_numbers}."  #
            f"The number of neighbor_numbers={neighbor_numbers.shape[0]}. "
            f"The number of neighbor atoms={self.neighbor.shape[0]}. "
        )

        # The following parameters are for the direction grid.
        # Please do not change them unless you know what you are doing.
        skin = 0.5
        scale = 1.5
        neighbors_exclude_core = False
        if len(self.core) == 1:
            neighbors_exclude_core = True
            skin, scale = 1.0, 1.2
        # elif len(core) == 2:
        #     skin, scale = 1.0, 1.6

        cov_core = COV_R[core_numbers].max()
        if neighbors_exclude_core:
            pos_nbrs = self.neighbor.copy()
            num_nbrs = neighbor_numbers.copy()
        else:
            pos_nbrs = np.vstack([self.core, self.neighbor])
            num_nbrs = np.append(core_numbers, neighbor_numbers)
        cov_nbrs = COV_R[num_nbrs]

        grid_unit = fibonacci_lattice(nfibonacci)
        grid = grid_unit * cov_core * float(scale) + self.center
        v = pos_nbrs[:, np.newaxis, :] - grid[np.newaxis, :, :]
        _, vlen = find_mic(v.reshape(-1, 3), self.cell, True)
        d = vlen.reshape(v.shape[:2])
        matrix_cov_r = np.column_stack([cov_nbrs] * len(grid))
        cond = np.all(matrix_cov_r + float(skin) < d, axis=0)
        return grid[cond]

    @staticmethod
    def get_1order_nbr(atoms: Atoms, core: npt.ArrayLike) -> np.ndarray:
        """Get the 1-order neighbors of the core atoms."""
        return _get_1order_nbr(atoms, core)


def _get_1order_nbr(atoms: Atoms, core: npt.ArrayLike) -> np.ndarray:
    """Get the 1-order neighbors of the core atoms."""
    core, all = np.unique(core), np.arange(len(atoms))
    i = np.repeat(core, len(all))
    j = np.tile(all, len(core))
    d = atoms.get_distances(i, j, mic=True)
    d_ij = COV_R[atoms.numbers[i]] + COV_R[atoms.numbers[j]] + 0.3
    return j[d < d_ij]


def _site_helper(
    atoms: Atoms | System | Cluster,
    neighbors: npt.ArrayLike | None = None,
    core: npt.ArrayLike | int = 0,
) -> tuple[Atoms, np.ndarray, np.ndarray, System | Cluster | None, Site]:
    """Helper the Site class.

    Returns:
        tuple[Atoms, np.ndarray, np.ndarray, System | Cluster | None, Site]:
            The atoms, core ids, neighbors ids, origin, and site.
    """
    if not isinstance(atoms, Atoms):
        atoms, _origin = atoms.to_ase(), atoms
    else:
        _origin: System | Cluster | None = None
    assert isinstance(_origin, (System, Cluster)) or _origin is None
    assert isinstance(atoms, Atoms), f"Invalid atoms type({type(atoms)})."
    # B. Convert the core atoms to a list of integers (np.ndarray)
    core = np.asarray([core] if isinstance(core, int) else core, int)
    if len(core) > 6:
        raise ValueError(
            "The core size must be less than or equal"
            f" to 6. The value of core: {core}."
        )
    if neighbors is None:
        if _origin is not None:
            assert isinstance(_origin, (System, Cluster))
            lst = [_origin.get_neighbors(i) for i in core]
            neighbors = reduce(np.append, lst)
        else:
            neighbors = _get_1order_nbr(atoms, core)
    else:
        neighbors = np.asarray(neighbors, int).ravel()
    neighbors = np.setdiff1d(neighbors, core).ravel()
    assert len(neighbors) > 0, (
        f"No 1-hop neighbors found for the core of {core}."
    )
    site = Site(
        neighbor=atoms.positions[neighbors],
        core=atoms.positions[core],
        cell=atoms.cell.array,
    )
    return (atoms, core, neighbors, _origin, site)
