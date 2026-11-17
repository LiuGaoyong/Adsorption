"""Hydra entry point of the campaign runner.

The single configuration file lives next to this module (``run.yaml``).  The
decorated :func:`run` is the CLI entry point; programmatic callers should use
:func:`run_config` from :mod:`adsorption.runner._run`.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from adsorption.runner._run import RunSummary, load_config, run_config

__all__ = ["RunSummary", "load_config", "run", "run_config"]


@hydra.main(
    config_path=str(Path(__file__).resolve().parent),
    config_name="run",
    version_base=None,
)
def run(cfg: DictConfig) -> None:
    """Run one campaign from a Hydra configuration.

    Args:
        cfg: The composed Hydra configuration.
    """
    summary = run_config(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    print(
        f"total={summary.total} accepted={summary.accepted} "
        f"rejected={summary.rejected} seconds={summary.seconds:.2f}"
    )
    print(
        f"sites={summary.n_sites} conformers={summary.n_conformers} "
        f"rotatable_bonds={summary.n_rotatable_bonds}"
    )
    print(f"records: {summary.records_path}")
    print(f"accepted: {summary.accepted_path}")
    print(f"summary: {summary.summary_path}")
