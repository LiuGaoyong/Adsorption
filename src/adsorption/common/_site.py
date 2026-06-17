"""The core data classes by pydantic."""

from functools import cached_property
from typing import Annotated

import numpy as np
from ase.geometry import complete_cell, find_mic, wrap_positions
from graphatoms.dataclasses import NDArray, numpy_validator
from pydantic import BaseModel, ConfigDict


class Site(BaseModel):
    """The site for adsorption."""

    model_config = ConfigDict(frozen=True)
    core: Annotated[NDArray, numpy_validator(float, (-1, 3))]
    neighbor: Annotated[NDArray, numpy_validator(float, (-1, 3))]
    cell: Annotated[NDArray, numpy_validator(float, (3, 3))] | None = None



    @cached_property
    def center(self) -> np.ndarray:
        """The center for adsoption."""
        if self.cell is not None and np.abs(np.linalg.det(self.cell)) > 1e-5:
            pbc = np.any(self.cell, axis=1)
            center0 = np.array([0.5, 0.5, 0.5])
            center0 = center0 @ complete_cell(self.cell)
            center0 = center0 - self.core[0]
            core = self.core + center0  # move core to the center
            core = wrap_positions(core, self.cell, pbc)  # wrap core to the cell
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

    @cached_property
    def direction_grid(self) -> np.ndarray:
        """The direction grid for adsorption."""
        raise NotImplementedError
