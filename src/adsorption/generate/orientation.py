"""Choose the orientation manifold of an adsorbate.

A linear molecule has a continuous rotation symmetry about its axis, so its
orientation space is ``SO(3) / C_inf_v ~ S^2``. A planar molecule only has a
finite point group, so its orientation space is still three-dimensional and
must be sampled on ``SO(3)``. A single atom has no orientation at all.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms

from adsorption.geometry import (
    FixedRotation,
    RotationSet,
    ShortestArcSet,
    SuperFibonacciSO3,
)

__all__ = ["Shape", "build_orientation", "detect_shape"]

Shape = Literal["fixed", "s2", "so3", "auto"]


def detect_shape(
    adsorbate: Atoms,
    *,
    linear_tol: float = 1e-3,
    planar_tol: float = 1e-3,
) -> Literal["fixed", "s2", "so3"]:
    """Classify the adsorbate by its principal moments.

    Args:
        adsorbate: The adsorbate structure.
        linear_tol: Relative threshold on the smallest principal moment.
        planar_tol: Relative threshold on the planar identity
            ``I3 = I1 + I2``.

    Returns:
        str: ``"fixed"``, ``"s2"`` or ``"so3"``.
    """
    if len(adsorbate) <= 1:
        return "fixed"
    moments = np.asarray(adsorbate.get_moments_of_inertia(), dtype=float)
    total = float(moments.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return "fixed"
    if moments[0] / total < linear_tol:
        return "s2"
    if abs(moments[2] - moments[0] - moments[1]) / total < planar_tol:
        return "so3"
    return "so3"


def build_orientation(
    kind: Shape,
    adsorbate: Atoms,
    *,
    n: int,
    axis: np.ndarray | None = None,
    linear_tol: float = 1e-3,
    planar_tol: float = 1e-3,
) -> tuple[str, RotationSet]:
    """Build the rotation set of an adsorbate.

    Args:
        kind: ``"auto"``, ``"fixed"``, ``"s2"`` or ``"so3"``.
        adsorbate: The adsorbate structure.
        n: Number of orientations.
        axis: Optional body-frame reference axis.
        linear_tol: Linearity threshold forwarded to :func:`detect_shape`.
        planar_tol: Planarity threshold forwarded to :func:`detect_shape`.

    Returns:
        tuple[str, RotationSet]: The resolved kind and the rotation set.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in {"auto", "fixed", "s2", "so3"}:
        raise ValueError(f"Unknown orientation kind: {kind!r}.")
    resolved = (
        detect_shape(adsorbate, linear_tol=linear_tol, planar_tol=planar_tol)
        if kind == "auto"
        else kind
    )
    if resolved == "fixed":
        return resolved, FixedRotation()
    if resolved == "s2":
        return resolved, ShortestArcSet(
            np.array([0.0, 0.0, 1.0]) if axis is None else axis,
            int(n),
        )
    return resolved, SuperFibonacciSO3(int(n), axis=axis)
