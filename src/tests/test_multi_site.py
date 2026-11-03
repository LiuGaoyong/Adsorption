# ruff: noqa D103 E501
"""Tests of the multi-site process-A generator."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.generate import MultiSiteGen, SingleSiteGen
from adsorption.geometry import (
    FixedRotation,
    build_site,
    resolve_anchor,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


def make(slab: Atoms, core: list[int], n_grid: int, distances: list[float], index: int) -> SingleSiteGen:
    reference = molecule("CO")
    return SingleSiteGen(
        slab,
        build_site(slab, core=core),
        reference,
        anchor=resolve_anchor(reference, "com"),
        orientation=FixedRotation(),
        distances=np.array(distances),
        n_grid=n_grid,
        site_index=index,
    )


@pytest.fixture(scope="module")
def generator(slab: Atoms) -> MultiSiteGen:
    return MultiSiteGen(
        [
            make(slab, [0], 4, [2.0], 0),
            make(slab, [1], 6, [2.0, 2.5], 1),
        ]
    )


def test_budget(generator: MultiSiteGen) -> None:
    assert generator.n_candidates == 4 + 12
    assert len(generator) == 16
    assert generator.offsets == (0, 4)


def test_global_index_maps_to_site(generator: MultiSiteGen) -> None:
    assert generator.trial(1).meta["site"] == 0
    assert generator.trial(4).meta["site"] == 0
    fifth = generator.trial(5)
    assert fifth.meta["site"] == 1
    assert fifth.meta["i"] == 1
    assert fifth.meta["k"] == 1
    last = generator.trial(16)
    assert last.meta["site"] == 1
    assert last.meta["i"] == 6
    assert last.meta["k"] == 2
    assert last.index == 16
    assert last.n_total == 16


def test_generate_across_the_boundary(generator: MultiSiteGen) -> None:
    chunk = list(generator.generate(3, 6))
    assert [trial.index for trial in chunk] == [3, 4, 5, 6]
    assert [trial.meta["site"] for trial in chunk] == [0, 0, 1, 1]
    assert len(list(generator.generate())) == 16
    with pytest.raises(IndexError):
        list(generator.generate(0, 1))
    with pytest.raises(IndexError):
        list(generator.generate(1, 17))


def test_empty_components() -> None:
    with pytest.raises(ValueError):
        MultiSiteGen([])


def test_signature(generator: MultiSiteGen) -> None:
    signature = generator.signature()
    assert signature["kind"] == "multi_site"
    assert len(signature["components"]) == 2
