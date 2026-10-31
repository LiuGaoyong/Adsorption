"""Ranking metrics across calculator tiers."""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "kendall",
    "regret",
    "spearman",
    "top_k_ids",
    "top_k_recall",
]


def _scores(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("scores must be finite.")
    return array


def _paired(left: object, right: object) -> tuple[np.ndarray, np.ndarray]:
    a = _scores(left)
    b = _scores(right)
    if len(a) != len(b):
        raise ValueError(
            f"Scores must have the same length, got {len(a)} and {len(b)}."
        )
    if len(a) < 2:
        raise ValueError("At least two scores are required.")
    return a, b


def spearman(left: object, right: object) -> float:
    """Return Spearman's rank correlation.

    Args:
        left: Scores of the first tier.
        right: Scores of the second tier.

    Returns:
        float: The correlation coefficient.
    """
    a, b = _paired(left, right)
    return float(stats.spearmanr(a, b).statistic)


def kendall(left: object, right: object) -> float:
    """Return Kendall's tau-b rank correlation.

    Args:
        left: Scores of the first tier.
        right: Scores of the second tier.

    Returns:
        float: The correlation coefficient.
    """
    a, b = _paired(left, right)
    return float(stats.kendalltau(a, b).statistic)


def top_k_ids(
    scores: object,
    k: int,
    *,
    higher_is_better: bool = False,
) -> np.ndarray:
    """Return the indices of the ``k`` best scores.

    Energies are lower-is-better by default; set ``higher_is_better`` for
    scores such as affinities.

    Args:
        scores: One score per candidate.
        k: Number of indices to return.
        higher_is_better: Whether larger scores are better.

    Returns:
        np.ndarray: The selected indices.

    Raises:
        ValueError: If ``k`` is outside ``[1, len(scores)]``.
    """
    values = _scores(scores)
    count = len(values)
    if not 1 <= int(k) <= count:
        raise ValueError(f"k must be in [1, {count}], got {k}.")
    order = np.argsort(-values if higher_is_better else values)
    return order[: int(k)]


def top_k_recall(
    reference: object,
    query: object,
    k: int,
    *,
    higher_is_better: bool = False,
) -> float:
    """Return the top-``k`` recall of a cheap ranking against a reference.

    The reference is the high tier (the truth) and the query is the cheap
    tier. The recall is the fraction of the reference top-``k`` that the
    query also places in its top-``k``.

    Args:
        reference: Reference scores.
        query: Query scores.
        k: Rank cutoff.
        higher_is_better: Whether larger scores are better.

    Returns:
        float: The recall in ``[0, 1]``.
    """
    truth = set(
        top_k_ids(reference, k, higher_is_better=higher_is_better).tolist()
    )
    found = set(
        top_k_ids(query, k, higher_is_better=higher_is_better).tolist()
    )
    return float(len(truth & found) / float(k))


def regret(
    reference: object,
    query: object,
    k: int,
    *,
    higher_is_better: bool = False,
) -> float:
    """Return the top-``k`` regret of a cheap ranking.

    The regret is the reference score of the best candidate inside the query
    top-``k`` minus the best reference score overall; zero means the query top
    set contains the reference optimum.

    Args:
        reference: Reference scores.
        query: Query scores.
        k: Rank cutoff.
        higher_is_better: Whether larger scores are better.

    Returns:
        float: The non-negative regret.
    """
    truth = _scores(reference)
    cheap = top_k_ids(query, k, higher_is_better=higher_is_better)
    if higher_is_better:
        return float(truth.max() - truth[cheap].max())
    return float(truth[cheap].min() - truth.min())
