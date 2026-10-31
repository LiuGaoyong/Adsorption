"""The one-dimensional anchor-distance array."""

from __future__ import annotations

import numpy as np

__all__ = ["distance_array"]


def distance_array(
    d_min: float,
    d_max: float,
    step: float,
) -> np.ndarray:
    """Return the anchor-centre distances of Eq. (11).

    The array is ``d_k = d_min + (k - 1) * step`` for ``k = 1, ..., M`` with
    ``M = floor((d_max - d_min) / step) + 1``. Every value is within
    ``[d_min, d_max]`` up to floating-point round-off; the endpoint convention
    is inclusive at the lower bound.

    Args:
        d_min: Smallest anchor-centre distance in Angstrom.
        d_max: Largest anchor-centre distance in Angstrom.
        step: Distance step in Angstrom.

    Returns:
        np.ndarray: The ``M`` distances, shape ``(M,)``.

    Raises:
        ValueError: If the bounds or the step are not finite, the step is not
            positive, or ``d_max < d_min``.
    """
    for name, value in (("d_min", d_min), ("d_max", d_max), ("step", step)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}.")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}.")
    if d_max < d_min:
        raise ValueError(
            f"d_max must be at least d_min, got {d_max} < {d_min}."
        )
    count = int(np.floor((d_max - d_min) / step + 1e-9)) + 1
    return float(d_min) + float(step) * np.arange(count, dtype=float)
