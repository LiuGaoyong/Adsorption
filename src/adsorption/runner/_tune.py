from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from graphatoms.utils.parser import hydra_parse
from omegaconf import DictConfig
from ray import tune
from ray.tune.search import create_searcher

from ..interfaces._directAD import DirectAdsorptionAD as TuneAdsorption
from ..interfaces.helper import plot


def tune_adsorption(
    cfg: DictConfig,
    atoms: Atoms,
    adsorbate: Atoms | str,
    output: Path | str = ".",
    core: list[int] = [0],
    nsamples: int = 100,
) -> tune.ResultGrid:
    obj = TuneAdsorption(
        calculator=hydra_parse(
            cfg=cfg.calculator,
            cls=Calculator,
        ),
        **cfg.adsorption,
    )
    grid_core, grid_ads, anchor_core = obj.grid_generation(
        adsorbate=adsorbate,
        atoms=atoms,
        core=core,
    )
    p = Path(output)
    p.mkdir(parents=True, exist_ok=True)

    def helper(config: dict[str, Any]) -> dict[str, Any]:
        obj = TuneAdsorption(
            calculator=hydra_parse(
                cfg=cfg.calculator,
                cls=Calculator,
            ),
            **cfg.adsorption,
        )
        result = obj.__call__(
            atoms=atoms,
            grid_ads=grid_ads,
            grid_core=grid_core,
            anchor_core=anchor_core,
            adsorbate=adsorbate,
            core=core,
            **config,
        )
        result_atoms, nstage = result
        assert isinstance(result_atoms, Atoms)
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
        if np.isinf(score):
            score = -0.001
        key.insert(0, f"E_{int(score * 1000):07d}meV")
        key.append(f"stage_{nstage:d}")
        s = "--".join(key)

        p.joinpath("png").mkdir(parents=True, exist_ok=True)
        p.joinpath("xyz").mkdir(parents=True, exist_ok=True)
        result_atoms.write(p.joinpath("xyz", f"{s}.xyz"), format="extxyz")
        plot(result_atoms, pngfname=p.joinpath("png", f"{s}.png"))
        return {"score": score, "nstage": nstage, "fmax": fmax}

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
