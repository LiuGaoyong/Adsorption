"""Legacy position-based site wrapper.

The geometric construction lives in :mod:`adsorption.geometry`. This module
keeps the historical ``Site`` model and the ``_site_helper`` function working
for existing user code. New code should use ``build_site`` /
``admissible_directions`` from :mod:`adsorption.geometry`.
"""

from functools import cached_property
from typing import Annotated, Self

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.data import covalent_radii as COV_R
from ase.geometry import complete_cell, find_mic, wrap_positions
from graphatoms.dataclasses import NDArray, numpy_validator
from graphatoms.system import Cluster, System
from pydantic import BaseModel, ConfigDict

from ..geometry import (
    MAX_CORE_SIZE,
    default_scale,
    default_skin,
    filter_grid,
    first_neighbor_shell,
)


class Site(BaseModel):
    """The site for adsorption (legacy position-based wrapper)."""

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
        """The center for adsorption."""
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
        """The normalized direction vector for adsorption (legacy)."""
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
        """Return the admissible probe points of the direction sphere.

        This wrapper is kept for backward compatibility. New code should use
        ``adsorption.geometry.admissible_directions``.

        Args:
            core_numbers: Atomic numbers of the core atoms.
            neighbor_numbers: Atomic numbers of the neighbour shell.
            nfibonacci: Number of Fibonacci lattice points. Defaults to 1000.

        Returns:
            np.ndarray: Probe points that pass the steric filter, shape
                ``(N_OA, 3)``.
        """
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
        core_size = len(self.core)
        skin = default_skin(core_size)
        scale = default_scale(core_size)
        if core_size == 1:
            filter_positions = np.asarray(self.neighbor, dtype=float)
            filter_numbers = neighbor_numbers
        else:
            filter_positions = np.vstack(
                (np.asarray(self.core, dtype=float), self.neighbor)
            )
            filter_numbers = np.concatenate((core_numbers, neighbor_numbers))
        radius = float(np.max(COV_R[core_numbers])) * float(scale)
        admissible = filter_grid(
            center=np.asarray(self.center, dtype=float),
            radius=radius,
            filter_positions=filter_positions,
            filter_radii=np.asarray(COV_R[filter_numbers], dtype=float),
            cell=None if self.cell is None else np.asarray(self.cell),
            pbc=None,
            n=nfibonacci,
            skin=skin,
        )
        return admissible.unit * radius + np.asarray(self.center, dtype=float)

    @staticmethod
    def get_1order_nbr(atoms: Atoms, core: npt.ArrayLike) -> np.ndarray:
        """Get the 1-order neighbors of the core atoms."""
        return first_neighbor_shell(atoms, core)


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
    core_array = np.asarray(
        [core] if isinstance(core, int) else core, dtype=int
    )
    if len(core_array) > MAX_CORE_SIZE:
        raise ValueError(
            "The core size must be less than or equal"
            f" to {MAX_CORE_SIZE}. The value of core: {core_array}."
        )
    if neighbors is None:
        neighbor_array = first_neighbor_shell(
            atoms, core_array, origin=_origin
        )
    else:
        neighbor_array = np.setdiff1d(
            np.asarray(neighbors, dtype=int).ravel(), core_array
        )
    if len(neighbor_array) == 0:
        raise ValueError(
            f"No 1-hop neighbors found for the core of {core_array}."
        )
    site = Site(
        neighbor=atoms.positions[neighbor_array],
        core=atoms.positions[core_array],
        cell=atoms.cell.array,
    )
    return (atoms, core_array, neighbor_array, _origin, site)
