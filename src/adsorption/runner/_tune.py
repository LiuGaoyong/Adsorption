from pathlib import Path
from typing import Any

import numpy as np
from ase import Atom, Atoms
from graphatoms.system import Cluster, Gas, System
from numpy.typing import ArrayLike
from ray import tune
from ray.tune.search import create_searcher

from .._interfaces._direct import DirectAdsorption
from ._plot import plot


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

    def tune(
        self,
        atoms: Atoms,
        adsorbate: Atoms | str,
        output: Path | str = ".",
        core: list[int] = [0],
        nsamples: int = 100,
    ) -> tune.ResultGrid:
        grid_core, grid_ads = self.grid_generation(
            adsorbate=adsorbate,
            atoms=atoms,
            core=core,
        )
        p = Path(output)
        p.mkdir(parents=True, exist_ok=True)

        def helper(config: dict[str, Any]) -> dict[str, Any]:
            result = self.__call__(
                atoms=atoms,
                grid_ads=grid_ads,
                grid_core=grid_core,
                adsorbate=adsorbate,
                core=core,
                **config,
            )
            result_atoms: Atoms = result[0]
            try:
                score = result_atoms.get_potential_energy(False, False)
                force = result_atoms.get_forces(False, False)
                fmax = np.linalg.norm(force, axis=1).max()
            except Exception:
                score = fmax = np.inf

            key: list[str] = []
            for k in sorted(config.keys()):
                if k == "distance":
                    v = config[k] * 100
                    v = f"{int(v):03d}pm"
                else:
                    v = f"{int(config[k]):04d}"
                key.append(f"{k}_{v}")
            key.append(f"E_{int(score * 1000):07d}meV")
            key.append(f"stage_{result[1]:d}")
            s = "--".join(key)

            p.joinpath("png").mkdir(parents=True, exist_ok=True)
            p.joinpath("xyz").mkdir(parents=True, exist_ok=True)
            result_atoms.write(p.joinpath("xyz", f"{s}.xyz"), format="extxyz")
            plot(result_atoms, pngfname=p.joinpath("png", f"{s}.png"))
            return {"score": score, "nstage": result[1], "fmax": fmax}

        tuner = tune.Tuner(
            tune.with_resources(helper, {"cpu": 1}),
            param_space={
                "idx_grid_ads": tune.randint(0, len(grid_ads)),
                "idx_grid_core": tune.randint(0, len(grid_core)),
                "distance": tune.choice(np.arange(1, 5, 0.15).tolist()),
            },
            tune_config=tune.TuneConfig(
                mode="min",
                metric="score",
                search_alg=create_searcher("random"),
                num_samples=int(nsamples),
            ),
        )
        return tuner.fit()
