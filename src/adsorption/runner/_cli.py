import logging
import os
from pathlib import Path

import hydra
from ase import Atoms
from graphatoms.utils.parser import hydra_parse
from omegaconf import DictConfig, OmegaConf
from ray.tune import ResultGrid

from ._tune import tune_adsorption

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
        f.write("\n".join(["*", "!config.yaml", "!run.sh", ""]))
    outlogfile = p / "run.log"

    # check something
    if hydracfg.mode == "MULTIRUN":
        raise ValueError(
            "Please delete '--multirun,-m' option "
            "when running this script. The multirun "
            "mode is not supported because this program "
            "will be parallelized by Ray innerly."
        )

    result: ResultGrid = tune_adsorption(
        cfg=cfg,
        atoms=hydra_parse(cfg=cfg.system.atoms, cls=Atoms),
        adsorbate=(
            hydra_parse(cfg=cfg.gas, cls=Atoms)
            if "_target_" in cfg.gas
            else str(cfg.gas)
        ),
        core=[int(i) for i in cfg.system.core],
        output=p,
    )
    for i in range(len(result)):
        log.info("=" * 32 + f" {i} " + "=" * 32)
        log.info(result[i])
        print(i, result[i])


def test_cli() -> None:
    main()
