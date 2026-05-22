from typing import Any

import numpy as np
from ase import Atom, Atoms
from graphatoms.system import Cluster, Gas, System
from numpy.typing import ArrayLike

from .._interfaces._direct import DirectAdsorption


class TuneAdsorption(DirectAdsorption):
    def grid_generation(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        *,
        core: ArrayLike | None = 0,
    ) -> tuple[np.ndarray, np.ndarray]:

        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        adsorbate = self._get_adsorbate(adsorbate)
        grid_ads, _ = self._get_grids(adsorbate, None)
        grid_core, _ = self._get_grids(atoms, core)
        return grid_core, grid_ads

    @staticmethod
    def helper(config: dict[str, Any]) -> dict[str, Any]:
        assert isinstance(config["obj"], TuneAdsorption)
        result: Atoms = config["obj"].__call__(
            **{
                k: v
                for k, v in config.items()
                if k
                in (
                    "core",
                    "atoms",
                    "adsorbate",
                    "idx_grid_ads",
                    "idx_grid_core",
                    "grid_core",
                    "grid_ads",
                    "distance",
                )
            },
        )
        try:
            score = result.get_potential_energy()
        except Exception:
            score = 1.0
        return {"score": score}
