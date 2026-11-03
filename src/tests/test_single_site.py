# ruff: noqa D103 E501
"""Tests of the single-site process-A generator."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.generate import SingleSiteGen
from adsorption.geometry import (
    FixedRotation,
    SuperFibonacciSO3,
    build_site,
    resolve_anchor,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def generator(slab: Atoms) -> SingleSiteGen:
    reference = molecule("CO")
    return SingleSiteGen(
        slab,
        build_site(slab, core=[0]),
        reference,
        anchor=resolve_anchor(reference, "com"),
        orientation=SuperFibonacciSO3(4),
        distances=np.array([1.5, 2.0, 2.5]),
        n_grid=6,
    )


def test_budget(generator: SingleSiteGen) -> None:
    assert generator.n_oa == 6
    assert generator.n_rot == 4
    assert generator.n_d == 3
    assert generator.n_candidates == 72
    assert len(generator) == 72


def test_index_order(generator: SingleSiteGen) -> None:
    first = generator.trial(1)
    assert first.meta["i"] == 1
    assert first.meta["j"] == 1
    assert first.meta["k"] == 1
    last = generator.trial(72)
    assert last.meta["i"] == 6
    assert last.meta["j"] == 4
    assert last.meta["k"] == 3
    assert last.meta["distance"] == pytest.approx(2.5)
    with pytest.raises(IndexError):
        generator.trial(0)
    with pytest.raises(IndexError):
        generator.trial(73)


def test_trial_structure(generator: SingleSiteGen, slab: Atoms) -> None:
    trial = generator.trial(5)
    assert len(trial.atoms) == len(slab) + 2
    assert trial.index == 5
    assert trial.n_total == 72
    assert trial.meta["n_substrate"] == len(slab)
    assert trial.atoms.get_chemical_formula() == "CCu8O"


def test_generate_interval(generator: SingleSiteGen) -> None:
    chunk = list(generator.generate(10, 12))
    assert [trial.index for trial in chunk] == [10, 11, 12]
    assert len(list(generator.generate())) == 72
    with pytest.raises(IndexError):
        list(generator.generate(0, 1))
    with pytest.raises(IndexError):
        list(generator.generate(1, 73))
    with pytest.raises(ValueError):
        list(generator.generate(5, 4))


def test_signature_is_stable(generator: SingleSiteGen) -> None:
    left = generator.signature()
    right = generator.signature()
    assert left == right
    assert left["kind"] == "single_site"
    assert left["n_grid"] == 6
    assert left["distances"] == [1.5, 2.0, 2.5]


def test_fixed_rotation_has_one_orientation(slab: Atoms) -> None:
    reference = molecule("CO")
    generator = SingleSiteGen(
        slab,
        build_site(slab, core=[0]),
        reference,
        anchor=resolve_anchor(reference, "com"),
        orientation=FixedRotation(),
        distances=np.array([2.0]),
        n_grid=4,
    )
    assert generator.n_rot == 1
    assert generator.n_candidates == 4
