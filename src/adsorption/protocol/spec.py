"""Pydantic specifications of a fixed-budget campaign."""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..geometry import (
    DELTA_NBR,
    MAX_CORE_SIZE,
    DirectionGrid,
    FibonacciGrid,
    FixedRotation,
    LatLonGrid,
    RotationSet,
    SeededRandomGrid,
    ShortestArcSet,
    distance_array,
)

__all__ = [
    "AdsorbateSpec",
    "DistanceSpec",
    "FilterSpec",
    "GridSpec",
    "OrientationSpec",
    "Protocol",
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
    """The orientation-set choice."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["shortest_arc", "fixed"] = "shortest_arc"
    n_ab: int = 1000
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @field_validator("n_ab")
    @classmethod
    def _n_ab(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError(f"n_ab must be positive, got {value}.")
        return int(value)

    @field_validator("axis")
    @classmethod
    def _axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return _finite_tuple(value)

    def build(self) -> RotationSet:
        """Return the corresponding rotation set.

        Returns:
            RotationSet: The rotation set.
        """
        if self.kind == "fixed":
            return FixedRotation()
        return ShortestArcSet(np.asarray(self.axis, dtype=float), self.n_ab)


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


class AdsorbateSpec(BaseModel):
    """The reference adsorbate and its anchor declaration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reference: Atoms
    anchor: str | int | tuple[float, float, float] = "com"
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @field_validator("reference")
    @classmethod
    def _reference(cls, value: Atoms) -> Atoms:
        if len(value) == 0:
            raise ValueError("The reference adsorbate must not be empty.")
        return value

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
            return _finite_tuple(tuple(float(x) for x in value))
        raise TypeError(f"Unsupported anchor declaration: {value!r}.")

    @field_validator("axis")
    @classmethod
    def _axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return _finite_tuple(value)


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
