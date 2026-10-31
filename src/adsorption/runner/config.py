"""Pydantic configuration of a campaign file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..evaluation import (
    Relaxation,
    Stage1Constraints,
    build_calculator,
)
from ..geometry import generate_conformers
from ..protocol import (
    AdsorbateSpec,
    ConformerSpec,
    Protocol,
    SiteSpec,
    prepare_campaign,
)
from ..protocol.campaign import Campaign
from ..screen import LJParameters
from .build import build_from_file, build_from_smiles, build_from_target

__all__ = [
    "AdsorbateConfig",
    "CalculatorConfig",
    "CampaignConfig",
    "EvaluationConfig",
    "ScreenConfig",
    "SiteEntry",
    "StructureSpec",
]


def _validate_vector(value: tuple[float, ...]) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite 3-vector, got {value}.")
    if float(np.linalg.norm(array)) < 1e-12:
        raise ValueError("A 3-vector must not be zero.")
    return tuple(float(x) for x in array)


class StructureSpec(BaseModel):
    """A structure source: a file or a dotted builder path."""

    model_config = ConfigDict(frozen=True)

    file: str | None = None
    index: int = -1
    target: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_source(self) -> StructureSpec:
        if (self.file is None) == (self.target is None):
            raise ValueError(
                "A structure spec must set exactly one of 'file' and 'target'."
            )
        return self

    def build(self) -> Atoms:
        """Return the structure.

        Returns:
            Atoms: The built structure.
        """
        if self.file is not None:
            return build_from_file(self.file, self.index)
        assert self.target is not None
        return build_from_target(self.target, self.kwargs)


class SiteEntry(BaseModel):
    """One site entry of the configuration."""

    model_config = ConfigDict(frozen=True)

    core: tuple[int, ...]
    neighbors: tuple[int, ...] | None = None
    label: str | None = None

    @field_validator("core")
    @classmethod
    def _core(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        core = tuple(int(x) for x in value)
        if not core:
            raise ValueError("The core set must not be empty.")
        if len(set(core)) != len(core):
            raise ValueError(f"The core set must be unique, got {core}.")
        if min(core) < 0:
            raise ValueError(f"Core indices must be non-negative, got {core}.")
        return core


class AdsorbateConfig(BaseModel):
    """The adsorbate source, its anchor, and an optional conformer set."""

    model_config = ConfigDict(frozen=True)

    reference: StructureSpec | None = None
    smiles: str | None = None
    anchor: str | int | tuple[float, float, float] = "com"
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    conformers: tuple[StructureSpec, ...] = ()
    generate: bool = False
    n_conformers: int = 50
    conformer_seed: int = 20261030
    prune_rmsd: float = 0.5

    @field_validator("anchor")
    @classmethod
    def _anchor(
        cls, value: str | int | tuple[float, float, float]
    ) -> str | int | tuple[float, float, float]:
        if isinstance(value, str):
            if value != "com":
                raise ValueError(
                    f"The only accepted string anchor is 'com', got {value!r}."
                )
            return value
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if int(value) < 0:
                raise ValueError(
                    f"An atom anchor must be non-negative, got {value}."
                )
            return int(value)
        if isinstance(value, (tuple, list)):
            return _validate_vector(tuple(float(x) for x in value))
        raise TypeError(f"Unsupported anchor declaration: {value!r}.")

    @field_validator("axis")
    @classmethod
    def _axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return _validate_vector(value)

    @field_validator("n_conformers")
    @classmethod
    def _n_conformers(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(f"n_conformers must be positive, got {value}.")
        return int(value)

    @model_validator(mode="after")
    def _consistent(self) -> AdsorbateConfig:
        if (self.reference is None) == (self.smiles is None):
            raise ValueError(
                "An adsorbate must set exactly one of 'reference' and 'smiles'."
            )
        if self.generate and self.smiles is None:
            raise ValueError("Conformer generation requires a SMILES source.")
        if self.generate and self.conformers:
            raise ValueError(
                "Use either conformer generation or an explicit conformer "
                "list, not both."
            )
        return self

    def to_spec(self) -> AdsorbateSpec:
        """Return the protocol adsorbate specification.

        Returns:
            AdsorbateSpec: The specification.
        """
        if self.smiles is not None:
            reference = build_from_smiles(self.smiles)
        else:
            assert self.reference is not None
            reference = self.reference.build()
        conformers: list[ConformerSpec] = []
        if self.generate:
            assert self.smiles is not None
            generated = generate_conformers(
                self.smiles,
                n_conformers=self.n_conformers,
                seed=self.conformer_seed,
                prune_rmsd=self.prune_rmsd,
            )
            conformers = [
                ConformerSpec(
                    reference=atoms,
                    anchor=self.anchor,
                    axis=self.axis,
                )
                for atoms in generated
            ]
        else:
            conformers = [
                ConformerSpec(
                    reference=spec.build(),
                    anchor=self.anchor,
                    axis=self.axis,
                )
                for spec in self.conformers
            ]
        return AdsorbateSpec(
            reference=reference,
            anchor=self.anchor,
            axis=self.axis,
            conformers=tuple(conformers),
        )


class CalculatorConfig(BaseModel):
    """An ASE calculator factory: a dotted path and keyword arguments."""

    model_config = ConfigDict(frozen=True)

    target: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """The evaluation tier."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["none", "single_point", "two_stage"] = "none"
    calculator: CalculatorConfig | None = None
    stage1_substrate_fixed: bool = True
    stage1_intramolecular_distances: bool = True
    max_steps_stage1: int = 100
    max_steps_stage2: int = 100
    max_force: float = 0.05
    method: str = "LBFGS"
    keep_trajectory: bool = False

    @model_validator(mode="after")
    def _calculator_required(self) -> EvaluationConfig:
        if self.mode != "none" and self.calculator is None:
            raise ValueError(
                f"Evaluation mode {self.mode!r} requires a calculator."
            )
        if self.max_steps_stage1 < 0 or self.max_steps_stage2 < 0:
            raise ValueError("The step limits must be non-negative.")
        if not np.isfinite(self.max_force) or self.max_force <= 0.0:
            raise ValueError(
                f"max_force must be positive, got {self.max_force}."
            )
        return self

    def build(self) -> Relaxation | None:
        """Return the evaluation callable, or ``None`` when disabled."""
        if self.mode == "none":
            return None
        assert self.calculator is not None
        calculator = build_calculator(
            self.calculator.target,
            self.calculator.kwargs,
        )
        return Relaxation(
            calculator=calculator,
            mode=self.mode,
            stage1=Stage1Constraints(
                substrate_fixed=self.stage1_substrate_fixed,
                intramolecular_distances=self.stage1_intramolecular_distances,
            ),
            max_steps_stage1=self.max_steps_stage1,
            max_steps_stage2=self.max_steps_stage2,
            max_force=self.max_force,
            method=self.method,
            keep_trajectory=self.keep_trajectory,
        )


class ScreenConfig(BaseModel):
    """The Lennard-Jones screen configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    kim: bool = False
    epsilon: dict[int, float] | None = None
    sigma: dict[int, float] | None = None
    r_cut: float | None = None
    r_min: float = 1.0
    e_cut: float = 1.0e9
    source: str = "user"

    @model_validator(mode="after")
    def _explicit_table(self) -> ScreenConfig:
        if not self.enabled:
            return self
        if self.kim:
            return self
        if self.epsilon is None or self.sigma is None or self.r_cut is None:
            raise ValueError(
                "An explicit screen requires epsilon, sigma, and r_cut; set "
                "kim: true to use the OpenKIM table."
            )
        return self

    def build(self, numbers: object) -> LJParameters | None:
        """Return the parameter set, or ``None`` when the screen is disabled.

        Args:
            numbers: The atomic numbers of the candidate structures.

        Returns:
            LJParameters | None: The parameters.
        """
        if not self.enabled:
            return None
        if self.kim:
            return LJParameters.from_kim(
                numbers,
                r_min=self.r_min,
                e_cut=self.e_cut,
                r_cut=self.r_cut,
                source=self.source,
            )
        assert self.epsilon is not None
        assert self.sigma is not None
        assert self.r_cut is not None
        return LJParameters(
            epsilon=self.epsilon,
            sigma=self.sigma,
            r_cut=self.r_cut,
            r_min=self.r_min,
            e_cut=self.e_cut,
            source=self.source,
        )


class CampaignConfig(BaseModel):
    """A complete campaign configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = "campaign"
    output: str = "runs/campaign"
    substrate: StructureSpec
    adsorbate: AdsorbateConfig
    sites: tuple[SiteEntry, ...] = ()
    protocol: Protocol = Field(default_factory=Protocol)
    screen: ScreenConfig | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def output_path(self) -> Path:
        """Return the output directory."""
        return Path(self.output)

    def to_campaign(self) -> Campaign:
        """Build the prepared campaign.

        Returns:
            Campaign: The campaign.
        """
        atoms = self.substrate.build()
        adsorbate = self.adsorbate.to_spec()
        sites = tuple(
            SiteSpec(
                core=entry.core,
                neighbors=entry.neighbors,
                label=entry.label,
            )
            for entry in self.sites
        )
        return prepare_campaign(atoms, adsorbate, sites, self.protocol)
