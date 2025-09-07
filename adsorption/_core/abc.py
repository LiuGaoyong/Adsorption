"""The core data classes by pydantic."""

import numpy as np
import pydantic
from numpy.typing import ArrayLike
from typing_extensions import Self


class _XYZ(pydantic.BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Self) -> Self:
        return self.__class__(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: Self) -> Self:
        return self.__class__(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, lst: list[float]) -> Self:
        return cls(x=lst[0], y=lst[1], z=lst[2])


class Point(_XYZ):
    """A point in 3D space."""


class Vector(_XYZ):
    """A vector in 3D space."""

    @property
    def length(self) -> float:
        v = [self.x, self.y, self.z]
        return float(np.linalg.norm(v))

    @property
    def norm(self) -> Self:
        t: float = self.length
        return self.__class__(
            x=self.x / t,
            y=self.y / t,
            z=self.z / t,
        )

    @classmethod
    def from_2points(cls, a: Point, b: Point) -> Self:
        return cls(
            x=b.x - a.x,
            y=b.y - a.y,
            z=b.z - a.z,
        )


class Site(pydantic.BaseModel):
    neighbor: list[Point]
    core: list[Point]

    @property
    def center(self) -> Point:
        core = np.asarray([p.to_list() for p in self.core])
        return Point.from_list(np.mean(core, axis=0))

    @property
    def normal(self) -> Vector:
        center = np.asarray(self.center.to_list())
        nbr = np.asarray([p.to_list() for p in self.neighbor])
        n2c = center - nbr  # the vector from the neighbor to the center
        n2c_norm = np.linalg.norm(n2c, axis=1)  # the norm of n2c
        n2c_eye = n2c / n2c_norm[:, None]  # the unit vector of n2c
        sorted_norm = n2c_norm[np.argsort(-n2c_norm)]  # sort by norm
        sorted_eye = n2c_eye[np.argsort(n2c_norm)]  # sort by norm
        _n2c = sorted_eye * sorted_norm[:, None]
        return Vector.from_list(np.mean(_n2c, axis=0))

    @classmethod
    def from_numpy(cls, nbr: ArrayLike, core: ArrayLike) -> Self:
        nbr, core = np.array(nbr, dtype=float), np.array(core, dtype=float)
        assert core.ndim == 2 and core.shape[1] == 3, "The core must be Nx3."
        assert nbr.ndim == 2 and nbr.shape[1] == 3, "The neighbor must be Nx3."
        return cls(
            core=[Point.from_list(c) for c in core],
            neighbor=[Point.from_list(n) for n in nbr],
        )
