import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from graphatoms.utils.parser import hydra_parse
from omegaconf import DictConfig, OmegaConf
from ray import tune

from ._tune import TuneAdsorption

log = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_name="_cli", config_path=".")
def main(cfg: DictConfig) -> None:  # noqa: D103
    log.info("=" * 64)
    log.info("The Configuration:\n" + OmegaConf.to_yaml(cfg))
    assert isinstance(cfg, DictConfig)
    log.info(f"Working directory : {os.getcwd()}")
    hydracfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    outlogfile = hydracfg.job_logging.handlers.file.filename
    log.info(f"Output directory  : {hydracfg.runtime.output_dir}")
    log.info(f"Output logfile    : {outlogfile}")
    log.info("=" * 64)

    p = Path(outlogfile).parent
    with p.joinpath(".gitignore").open("w") as f:
        f.write("*\n")
    outlogfile = p / 'run.log'

    # check something
    if hydracfg.mode == "MULTIRUN":
        raise ValueError(
            "Please delete '--multirun,-m' option "
            "when running this script. The multirun "
            "mode is not supported because this program "
            "will be parallelized by Ray innerly."
        )

    obj = TuneAdsorption(
        calculator=hydra_parse(
            cfg=cfg.calculator,
            cls=Calculator,
        ),
        **cfg.adsorption,
    )
    atoms = hydra_parse(cfg=cfg.system.atoms, cls=Atoms)
    adsorbate = (
        hydra_parse(cfg=cfg.gas, cls=Atoms)
        if "_target_" in cfg.gas
        else str(cfg.gas)
    )
    core = [int(i) for i in cfg.system.core]
    grid_core, grid_ads = obj.grid_generation(
        adsorbate=adsorbate,
        atoms=atoms,
        core=core,
    )

    def helper(config: dict[str, Any]) -> dict[str, Any]:
        result: Atoms = obj.__call__(
            atoms=atoms,
            grid_ads=grid_ads,
            grid_core=grid_core,
            adsorbate=adsorbate,
            core=core,
            **config
        )
        try:
            score = result.get_potential_energy()
        except Exception:
            score = 1.0
        return {"score": score}

    tuner = tune.Tuner(
        tune.with_resources(helper, {"cpu": 1}),
        param_space={
            "idx_grid_ads": tune.grid_search(range(0, len(grid_ads))),
            "idx_grid_core": tune.grid_search(range(0, len(grid_core))),
            "distance": tune.grid_search(np.arange(1, 5, 0.1)),
        }
    )
    tuner.fit()


def test_cli() -> None:
    main()
