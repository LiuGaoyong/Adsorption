# """Hydra configuration of an adsorption campaign.

# Structural parameters are plain dataclasses registered in the ``ConfigStore``;
# swappable objects (substrate builders, adsorbates, filters, calculators) are
# ``_target_`` nodes instantiated with ``hydra.utils.instantiate``. The parameter
# hash excludes volatile fields so that two runs with the same physics produce
# the same hash.
# """

# from __future__ import annotations

# import json
# import os
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any

# import hydra
# import numpy as np
# from ase import Atoms
# from hydra.core.config_store import ConfigStore
# from omegaconf import MISSING, DictConfig, OmegaConf

# from adsorption.generate import (
#     Conformer,
#     MultiAdsorbateGen,
#     build_orientation,
# )
# from adsorption.geometry import build_site, distance_array, resolve_anchor
# from adsorption.base import (
#     AdsorptionConfigurationGeneratorABC,
#     AdsorptionConfigurationFilterABC,
# )

# os.environ.setdefault("HYDRA_FULL_ERROR", "1")

# CONFIG_DIR = Path(__file__).parent

# __all__ = ["Config", "param_hash", "main"]


# @dataclass
# class GridConfig:
#     """The approach-direction grid."""

#     n: int = 200
#     kind: str = "fibonacci"


# @dataclass
# class DistanceConfig:
#     """The anchor-distance array."""

#     d_min: float = 1.2
#     d_max: float = 3.0
#     step: float = 0.2


# @dataclass
# class OrientationConfig:
#     """The molecular-orientation manifold."""

#     kind: str = "auto"
#     n: int = 200
#     axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
#     linear_tol: float = 1e-3
#     planar_tol: float = 1e-3


# @dataclass
# class ProtocolConfig:
#     """The geometric protocol."""

#     grid: GridConfig = field(default_factory=GridConfig)
#     distances: DistanceConfig = field(default_factory=DistanceConfig)
#     orientation: OrientationConfig = field(default_factory=OrientationConfig)


# @dataclass
# class BudgetConfig:
#     """The candidate-count guard."""

#     max_candidates: int = 10_000_000


# @dataclass
# class Config:
#     """The root configuration of one campaign."""

#     substrate: Any = MISSING
#     adsorbate: Any = MISSING
#     sites: list[list[int]] = field(default_factory=lambda: [[0]])
#     protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
#     filter: Any = MISSING
#     filters: list[Any] = field(default_factory=list)
#     budget: BudgetConfig = field(default_factory=BudgetConfig)
#     outputs: str = "./runs"
#     start: int = 1
#     stop: int | None = None


# cs = ConfigStore.instance()
# cs.store(name="base_config", node=Config)


# def param_hash(cfg: DictConfig | Config) -> str:
#     """Return a stable hash of the physics-bearing configuration.

#     Args:
#         cfg: The composed configuration.

#     Returns:
#         str: The hash, with ``outputs`` and ``hydra`` excluded.
#     """
#     from .....deprecated.src.adsorption.protocol.hashing import stable_hash

#     container = OmegaConf.to_container(cfg, resolve=True)
#     if isinstance(container, dict):
#         for key in ("outputs", "hydra"):
#             container.pop(key, None)
#     return stable_hash(container)


# def _build_generator(cfg: Config) -> MultiAdsorbateGen:
#     """Build the process-A generator from the configuration.

#     Args:
#         cfg: The composed configuration.

#     Returns:
#         MultiAdsorbateGen: The generator.

#     Raises:
#         ValueError: If the site list is empty.
#     """
#     substrate: Atoms = hydra.utils.instantiate(
#         OmegaConf.to_container(cfg.substrate, resolve=True)
#     )
#     adsorbate: Atoms = hydra.utils.instantiate(
#         OmegaConf.to_container(cfg.adsorbate, resolve=True)
#     )
#     raw_sites = [list(core) for core in cfg.sites]
#     if not raw_sites:
#         raise ValueError("sites must not be empty.")
#     sites = [build_site(substrate, core=core) for core in raw_sites]
#     orientation_cfg = cfg.protocol.orientation
#     kind, orientation = build_orientation(
#         orientation_cfg.kind,
#         adsorbate,
#         n=int(orientation_cfg.n),
#         axis=np.asarray(orientation_cfg.axis, dtype=float),
#         linear_tol=float(orientation_cfg.linear_tol),
#         planar_tol=float(orientation_cfg.planar_tol),
#     )
#     del kind
#     distances = distance_array(
#         float(cfg.protocol.distances.d_min),
#         float(cfg.protocol.distances.d_max),
#         float(cfg.protocol.distances.step),
#     )
#     conformer = Conformer(
#         reference=adsorbate,
#         anchor=resolve_anchor(adsorbate, "com"),
#         orientation=orientation,
#         distances=distances,
#         label=adsorbate.get_chemical_formula(),
#     )
#     return MultiAdsorbateGen(
#         substrate,
#         sites,
#         [conformer],
#         n_grid=int(cfg.protocol.grid.n),
#     )


# def _build_filters(cfg: Config) -> list[AdsConfFilterABC]:
#     """Instantiate the filter chain from ``_target_`` nodes.

#     Args:
#         cfg: The composed configuration.

#     Returns:
#         list[AdsConfFilterABC]: The filters, in declared order.
#     """
#     filters: list[AdsConfFilterABC] = []
#     for node in cfg.filters:
#         container = OmegaConf.to_container(node, resolve=True)
#         instance = hydra.utils.instantiate(container)
#         if not isinstance(instance, AdsConfFilterABC):
#             raise TypeError(
#                 "Every filter must implement AdsConfFilterABC, got "
#                 f"{type(instance)}."
#             )
#         filters.append(instance)
#     return filters


# @hydra.main(
#     config_path=CONFIG_DIR.as_posix(),
#     config_name="run",
#     version_base=None,
# )
# def main(cfg: Config) -> None:
#     """Enumerate and filter one campaign, writing a JSONL sidecar.

#     Args:
#         cfg: The composed configuration.

#     Raises:
#         ValueError: If the campaign exceeds ``budget.max_candidates``.
#     """
#     generator = _build_generator(cfg)
#     total = generator.n_candidates
#     if total > int(cfg.budget.max_candidates):
#         raise ValueError(
#             f"{total} candidates exceed budget.max_candidates="
#             f"{cfg.budget.max_candidates}."
#         )
#     filters = _build_filters(cfg)
#     pipeline = AdsConfPipeline(generator, filters)
#     output = Path(str(cfg.outputs))
#     output.mkdir(parents=True, exist_ok=True)
#     path = output / "records.jsonl"
#     with path.open("w", encoding="utf-8") as stream:
#         for record in pipeline.run(int(cfg.start), cfg.stop):
#             payload = {
#                 "index": record.trial.index,
#                 "meta": dict(record.trial.meta),
#                 "first_failed": record.first_failed,
#                 "verdicts": [
#                     {
#                         "accepted": verdict.accepted,
#                         "value": verdict.value,
#                         "reason": verdict.reason,
#                     }
#                     for verdict in record.verdicts
#                 ],
#             }
#             stream.write(json.dumps(payload, default=str) + "\n")


# if __name__ == "__main__":
#     main()
