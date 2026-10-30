"""Pydantic specifications of a fixed-budget campaign."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, runtime_checkable
from typing import Protocol as TypingProtocol

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..geometry import (
    DELTA_NBR,
    MAX_CORE_SIZE,
    DirectionGrid,
    DirectionRollSet,
    FibonacciGrid,
    FixedRotation,
    LatLonGrid,
    RotationSet,
    SO3Set,
    SeededRandomGrid,
    ShortestArcSet,
    distance_array,
)

__all__ = [
    "AdsorbateSpec",
    "CoreSetProvider",
    "ConformerSpec",
    "DistanceSpec",
    "FilterSpec",
    "GridSpec",
    "OrientationSpec",
    "Protocol",
    "SiteLabeller",
    "SiteSpec",
]


def _finite_tuple(value: tuple[float, ...]) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {value}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"A 3-vector must be finite, got {value}.")
    if float(np.linalg.norm(array)) < 1e-12:
        raise ValueError("A 3-vector must not be zero.")
    return tuple(float(x) for x in array)


def _check_reference(value: Atoms) -> Atoms:
    if len(value) == 0:
        raise ValueError("The reference adsorbate must not be empty.")
    return value


def _check_anchor(
    value: str | int | tuple[float, float, float],
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
        return _finite_tuple(tuple(float(x) for x in value))
    raise TypeError(f"Unsupported anchor declaration: {value!r}.")


class GridSpec(BaseModel):
    """The direction-grid choice."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["fibonacci", "latlon", "random"] = "fibonacci"
    seed: int | None = None

    def build(self) -> DirectionGrid:
        """Return the corresponding grid object.

        Returns:
            DirectionGrid: The grid.

        Raises:
            ValueError: If a random grid is requested without a seed.
        """
        if self.kind == "fibonacci":
            return FibonacciGrid()
        if self.kind == "latlon":
            return LatLonGrid()
        if self.seed is None:
            raise ValueError("The random grid requires an explicit seed.")
        return SeededRandomGrid(self.seed)


class FilterSpec(BaseModel):
    """The steric-filter parameters; ``None`` means the tabulated default."""

    model_config = ConfigDict(frozen=True)

    scale: float | None = None
    skin: float | None = None
    delta_nbr: float = DELTA_NBR

    @field_validator("scale")
    @classmethod
    def _scale(cls, value: float | None) -> float | None:
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"scale must be positive, got {value}.")
        return value

    @field_validator("skin")
    @classmethod
    def _skin(cls, value: float | None) -> float | None:
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"skin must be non-negative, got {value}.")
        return value

    @field_validator("delta_nbr")
    @classmethod
    def _delta(cls, value: float) -> float:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"delta_nbr must be non-negative, got {value}.")
        return value


class DistanceSpec(BaseModel):
    """The anchor-distance array of Eq. (11)."""

    model_config = ConfigDict(frozen=True)

    d_min: float = 0.8
    d_max: float = 3.0
    step: float = 0.2

    @field_validator("d_min", "d_max", "step")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError(
                f"The distance parameters must be finite, got {value}."
            )
        return value

    def array(self) -> np.ndarray:
        """Return the resolved distance array.

        Returns:
            np.ndarray: The ``M`` distances, shape ``(M,)``.
        """
        return distance_array(self.d_min, self.d_max, self.step)


class OrientationSpec(BaseModel):
    """The orientation-set choice.

    ``shortest_arc`` samples the reference axis on the sphere and applies the
    minimal rotation (the roll is not sampled). ``direction_roll`` adds an
    equally spaced roll grid on top. ``so3`` samples full rotations with a
    deterministic Kronecker/Shoemake construction. ``fixed`` is the identity,
    used when orientation does not matter.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal[
        "shortest_arc",
        "direction_roll",
        "so3",
        "fixed",
    ] = "shortest_arc"
    n_ab: int = 1000
    n_roll: int = 1
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @field_validator("n_ab")
    @classmethod
    def _n_ab(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(f"n_ab must be positive, got {value}.")
        return int(value)

    @field_validator("n_roll")
    @classmethod
    def _n_roll(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(f"n_roll must be positive, got {value}.")
        return int(value)

    @field_validator("axis")
    @classmethod
    def _axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return _finite_tuple(value)

    def build(self, *, axis: np.ndarray | None = None) -> RotationSet:
        """Return the corresponding rotation set.

        Args:
            axis: Optional per-conformer body-frame axis override.

        Returns:
            RotationSet: The rotation set.
        """
        body_axis = np.asarray(
            self.axis if axis is None else axis, dtype=float
        )
        if self.kind == "fixed":
            return FixedRotation()
        if self.kind == "direction_roll":
            return DirectionRollSet(
                body_axis,
                self.n_ab,
                n_roll=self.n_roll,
            )
        if self.kind == "so3":
            return SO3Set(self.n_ab)
        return ShortestArcSet(body_axis, self.n_ab)


class SiteSpec(BaseModel):
    """One site of the pre-registered site list."""

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
        if len(core) > MAX_CORE_SIZE:
            raise ValueError(
                f"The core size must be <= {MAX_CORE_SIZE}, got {len(core)}."
            )
        if min(core) < 0:
            raise ValueError(f"Core indices must be non-negative, got {core}.")
        return core

    @field_validator("neighbors")
    @classmethod
    def _neighbors(
        cls, value: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        if value is None:
            return None
        neighbors = tuple(int(x) for x in value)
        if not neighbors:
            raise ValueError("The neighbour shell must not be empty.")
        if min(neighbors) < 0:
            raise ValueError(
                f"Neighbour indices must be non-negative, got {neighbors}."
            )
        return neighbors


class ConformerSpec(BaseModel):
    """One member of a conformational ensemble.

    The reference axis and the anchor are redeclared per conformer, because a
    principal axis of one conformer need not be one of another (Section 2.6).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reference: Atoms
    anchor: str | int | tuple[float, float, float] = "com"
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @field_validator("reference")
    @classmethod
    def _reference(cls, value: Atoms) -> Atoms:
        return _check_reference(value)

    @field_validator("anchor")
    @classmethod
    def _anchor(
        cls, value: str | int | tuple[float, float, float]
    ) -> str | int | tuple[float, float, float]:
        return _check_anchor(value)

    @field_validator("axis")
    @classmethod
    def _axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return _finite_tuple(value)


class AdsorbateSpec(BaseModel):
    """The reference adsorbate, its anchor, and an optional conformer set.

    ``reference`` is the default conformer. Additional members are declared in
    ``conformers``; the campaign enumerates every conformer separately (Eq. 12,
    Proposition 5). The construction does not generate conformers; an optional
    utility can do so, but the campaign only consumes the declared set.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reference: Atoms
    anchor: str | int | tuple[float, float, float] = "com"
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    conformers: tuple[ConformerSpec, ...] = ()

    @field_validator("reference")
    @classmethod
    def _reference(cls, value: Atoms) -> Atoms:
        return _check_reference(value)

    @field_validator("anchor")
    @classmethod
    def _anchor(
        cls, value: str | int | tuple[float, float, float]
    ) -> str | int | tuple[float, float, float]:
        return _check_anchor(value)

    @field_validator("axis")
    @classmethod
    def _axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return _finite_tuple(value)

    @property
    def conformer_specs(self) -> tuple[ConformerSpec, ...]:
        """Return the full conformer list, default conformer first."""
        default = ConformerSpec(
            reference=self.reference,
            anchor=self.anchor,
            axis=self.axis,
        )
        return (default, *self.conformers)


class Protocol(BaseModel):
    """A complete fixed-budget protocol."""

    model_config = ConfigDict(frozen=True)

    name: str = "protocol"
    n: int = 1000
    grid: GridSpec = Field(default_factory=GridSpec)
    orientation: OrientationSpec = Field(default_factory=OrientationSpec)
    distances: DistanceSpec = Field(default_factory=DistanceSpec)
    filter: FilterSpec = Field(default_factory=FilterSpec)

    @field_validator("n")
    @classmethod
    def _n(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(f"n must be positive, got {value}.")
        return int(value)


@runtime_checkable
class CoreSetProvider(TypingProtocol):
    """A provider of candidate core sets (implemented by ``graphatoms``).

    The package does not implement graph-based site enumeration. A provider
    receives the substrate and returns the pre-registered site list.
    """

    def propose(self, atoms: Atoms) -> Sequence[SiteSpec]:
        """Return candidate sites for a substrate."""
        ...


@runtime_checkable
class SiteLabeller(TypingProtocol):
    """A six-class site labeller (implemented by ``graphatoms``).

    Labels describe the local environment and are used only for coverage
    statistics; they never enter the geometric construction.
    """

    def label(self, atoms: Atoms, site: SiteSpec) -> str | None:
        """Return the class label of a site, or ``None`` when undefined."""
        ...
