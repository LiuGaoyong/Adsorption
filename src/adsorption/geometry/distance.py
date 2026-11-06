"""The one-dimensional anchor-distance array."""

from __future__ import annotations

import numpy as np

__all__ = ["distance_array"]


def distance_array(
    d_min: float,
    d_max: float,
    step: float | None = None,
    *,
    n: int | None = None,
) -> np.ndarray:
    """Return the anchor-centre distances of Eq. (11).

    The array covers ``[d_min, d_max]`` and is specified either by the step
    ``Delta d`` or by the number ``n`` of distances; exactly one of the two
    must be given. With a step the count is
    ``M = floor((d_max - d_min) / Delta d) + 1``; with a count the step is
    ``(d_max - d_min) / (n - 1)`` and both endpoints are included. Every value
    is within ``[d_min, d_max]`` up to floating-point round-off.

    Args:
        d_min: Smallest anchor-centre distance in Angstrom.
        d_max: Largest anchor-centre distance in Angstrom.
        step: Distance step in Angstrom. Mutually exclusive with ``n``.
        n: Number of distances. Mutually exclusive with ``step``.

    Returns:
        np.ndarray: The ``M`` (or ``n``) distances, shape ``(M,)``.

    Raises:
        ValueError: If the bounds are invalid, neither or both of ``step`` and
            ``n`` are given, ``step`` is not positive, or ``n`` is not
            positive.
    """
    if (step is None) == (n is None):
        raise ValueError("Exactly one of 'step' and 'n' must be given.")
    for name, value in (("d_min", d_min), ("d_max", d_max)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}.")
    if d_min < 0.0:
        raise ValueError(f"d_min must be non-negative, got {d_min}.")
    if d_max < d_min:
        raise ValueError(
            f"d_max must be at least d_min, got {d_max} < {d_min}."
        )
    if step is not None:
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"step must be positive, got {step}.")
        count = int(np.floor((d_max - d_min) / step + 1e-9)) + 1
        return float(d_min) + float(step) * np.arange(count, dtype=float)
    count = int(n)  # type: ignore[arg-type]
    if count < 1:
        raise ValueError(f"n must be positive, got {n}.")
    if count == 1:
        return np.array([float(d_min)], dtype=float)
    derived = (float(d_max) - float(d_min)) / (count - 1)
    return float(d_min) + derived * np.arange(count, dtype=float)
