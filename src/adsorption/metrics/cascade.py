"""Cascade metrics: retention ratios and false-rejection fraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "CascadeRatios",
    "cascade_ratios",
    "false_rejection",
    "max_false_rejection",
]


@dataclass(frozen=True)
class CascadeRatios:
    """The retention ratios of the cascade.

    Attributes:
        rho_lj: ``N_LJ / N_total``.
        rho_mlp: ``N_MLP / N_LJ``.
        rho: The product ``rho_lj * rho_mlp``.
    """

    rho_lj: float
    rho_mlp: float
    rho: float


def cascade_ratios(n_total: int, n_lj: int, n_mlp: int) -> CascadeRatios:
    """Return the cascade retention ratios of Eq. (20).

    Args:
        n_total: Number of generated candidates.
        n_lj: Number of candidates that passed the screen.
        n_mlp: Number of candidates evaluated.

    Returns:
        CascadeRatios: The ratios.

    Raises:
        ValueError: If the counts are negative or not nested.
    """
    total, screened, evaluated = int(n_total), int(n_lj), int(n_mlp)
    if min(total, screened, evaluated) < 0:
        raise ValueError("The counts must be non-negative.")
    if not 0 <= evaluated <= screened <= total:
        raise ValueError(
            "The counts must satisfy 0 <= N_MLP <= N_LJ <= N_total, got "
            f"{evaluated}, {screened}, {total}."
        )
    if total == 0:
        return CascadeRatios(0.0, 0.0, 0.0)
    rho_lj = screened / total
    rho_mlp = evaluated / screened if screened else 0.0
    return CascadeRatios(rho_lj=rho_lj, rho_mlp=rho_mlp, rho=rho_lj * rho_mlp)


def false_rejection(
    true_top_k: Sequence[int],
    rejected: Sequence[int],
) -> float:
    """Return the false-rejection fraction at rank ``k`` (Eq. 20).

    Args:
        true_top_k: Identifiers of the ``k`` best candidates in the reference
            pool.
        rejected: Identifiers rejected by the screen.

    Returns:
        float: The fraction of the true top-``k`` that the screen rejected.
    """
    top = set(int(item) for item in true_top_k)
    if not top:
        return 0.0
    dropped = set(int(item) for item in rejected)
    return float(len(top & dropped) / len(top))


def max_false_rejection(n_total: int, n_lj: int, k: int) -> float:
    """Return the worst-case false rejection of Proposition 7.

    At rank ``k`` a screen that keeps ``n_lj / n_total`` of the pool can
    discard at most ``min(1, (1 - rho_lj) * n_total / k)`` of any top-``k``.

    Args:
        n_total: Number of generated candidates.
        n_lj: Number of candidates that passed the screen.
        k: Rank cutoff.

    Returns:
        float: The bound in ``[0, 1]``.

    Raises:
        ValueError: If ``k`` is not positive or the counts are invalid.
    """
    total, screened = int(n_total), int(n_lj)
    if int(k) < 1:
        raise ValueError(f"k must be positive, got {k}.")
    if not 0 <= screened <= total:
        raise ValueError("The counts must satisfy 0 <= N_LJ <= N_total.")
    if total == 0:
        return 0.0
    return float(min(1.0, (1.0 - screened / total) * total / int(k)))
