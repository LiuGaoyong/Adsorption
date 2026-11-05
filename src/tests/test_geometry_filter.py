# ruff: noqa: D103
"""Regression cases of the geometric distance floor.

``GeometryFilter`` only judges the substrate-adsorbate cross pairs: the
substrate is rigid and the adsorbate is placed as a rigid body, so their
internal distances cannot change from one candidate to the next.
"""

from __future__ import annotations

from ase import Atoms

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
