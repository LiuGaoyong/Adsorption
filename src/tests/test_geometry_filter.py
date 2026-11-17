# ruff: noqa: D103
"""Regression cases of the geometric distance floor.

``GeometryFilter`` only judges the substrate-adsorbate cross pairs: the
substrate is rigid and the adsorbate is placed as a rigid body, so their
internal distances cannot change from one candidate to the next.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.base import Trial
from adsorption.filters import GeometryFilter


def _trial(distance: float) -> Trial:
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    return Trial(atoms, index=1, n_total=1, meta={"n_substrate": 1})


def test_far_cross_pair_is_accepted() -> None:
    verdict = GeometryFilter().evaluate(_trial(2.5))
    assert verdict.accepted
    assert verdict.value is not None
    assert verdict.value > 0.0


def test_overlapping_cross_pair_is_rejected() -> None:
    verdict = GeometryFilter().evaluate(_trial(1.0))
    assert not verdict.accepted
    assert verdict.reason == "floor"
    assert verdict.detail is not None
    assert verdict.detail["pair"] == (0, 1)


def test_substrate_substrate_pairs_are_not_checked() -> None:
    atoms = Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    trial = Trial(atoms, index=1, n_total=1, meta={"n_substrate": 2})
    assert GeometryFilter().evaluate(trial).accepted


def test_adsorbate_adsorbate_pairs_are_not_checked() -> None:
    atoms = Atoms(
        "CuH2",
        positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    trial = Trial(atoms, index=1, n_total=1, meta={"n_substrate": 1})
    assert GeometryFilter().evaluate(trial).accepted


def test_vectorised_floors_match_the_scalar_table() -> None:
    filt = GeometryFilter(
        element_overrides={1: 0.9},
        pair_overrides={(29, 8): 1.7},
        floor_check="off",
    )
    left = np.array([29, 29, 8, 1, 6])
    right = np.array([8, 29, 29, 29, 1])
    expected = [filt._floor(a, b) for a, b in zip(left, right)]
    assert np.allclose(filt._pair_floors(left, right), expected)
    assert filt._pair_floors(np.array([29]), np.array([8]))[0] == 1.7


def test_unknown_element_raises_key_error() -> None:
    filt = GeometryFilter(sigma={29: 2.3, 8: 3.0})
    with pytest.raises(KeyError):
        filt._pair_floors(np.array([29]), np.array([1]))
    with pytest.raises(KeyError):
        filt._pair_floors(np.array([29]), np.array([200]))


def test_default_cutoff_is_narrowed_to_the_present_pairs() -> None:
    filt = GeometryFilter()
    full = filt.r_cut
    narrowed = filt._cross_cutoff(np.array([29, 29]), np.array([8, 1]))
    assert narrowed < full
    assert narrowed == pytest.approx(
        1.05 * max(filt._floor(29, 8), filt._floor(29, 1))
    )
    explicit = GeometryFilter(r_cut=6.0)
    assert explicit._cross_cutoff(np.array([29]), np.array([8])) == 6.0


def test_dense_cross_pairs_match_the_neighbor_list_backend() -> None:
    """The dense minimum-image path agrees with ``distance_pairs``.

    Above half the cell length the neighbour list reports several periodic
    images of one pair while the dense path keeps the closest only; the
    verdict depends on the minimum, so the comparison is per pair.
    """
    from arrayff.neighbors import distance_pairs

    from adsorption.filters import geometry as module

    slab = fcc111("Cu", (3, 4, 2), vacuum=6.0, orthogonal=True, periodic=True)
    adsorbate = molecule("CO")
    # Put the adsorbate across the cell boundary so periodic images matter.
    adsorbate.positions += slab.cell.lengths() * [1.0, 0.0, 0.0] + [
        0.1,
        0.2,
        slab.positions[:, 2].max() + 2.0,
    ]
    cell = slab.cell.array
    pbc = slab.pbc
    for cutoff in (2.5, 4.0, 6.0):
        left, right, dist = module._cross_pairs(
            slab.positions, adsorbate.positions, cell, pbc, cutoff
        )
        assert len(set(zip(left.tolist(), right.tolist()))) == len(left)
        first, second, expected = distance_pairs(
            p1=slab.positions,
            p2=adsorbate.positions,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            quantities="ijd",
        )
        closest: dict[tuple[int, int], float] = {}
        for i, j, d in zip(first.tolist(), second.tolist(), expected):
            closest[(i, j)] = min(closest.get((i, j), np.inf), float(d))
        dense = {
            (i, j): float(d)
            for i, j, d in zip(left.tolist(), right.tolist(), dist)
        }
        assert dense.keys() == closest.keys()
        for key, value in dense.items():
            assert np.isclose(value, closest[key])
    # Non-periodic structures use plain Euclidean distances.
    left, right, dist = module._cross_pairs(
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 1.5]]),
        np.zeros((3, 3)),
        np.zeros(3, dtype=bool),
        2.0,
    )
    assert left.tolist() == [0] and right.tolist() == [0]
    assert np.isclose(dist[0], 1.5)


def test_pairs_beyond_the_floor_do_not_change_the_verdict() -> None:
    wide = GeometryFilter(r_cut=10.0)
    narrow = GeometryFilter()
    for distance in (1.0, 2.2, 2.4):
        left = wide.evaluate(_trial(distance))
        right = narrow.evaluate(_trial(distance))
        assert left.accepted == right.accepted
        assert left.reason == right.reason
        assert left.value == pytest.approx(right.value)
