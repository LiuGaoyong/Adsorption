# ruff: noqa D103 E501
"""Tests of the multi-adsorbate process-A generator."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.generate import Conformer, MultiAdsorbateGenerator
from adsorption.geometry import (
    SuperFibonacciSO3,
    build_site,
    resolve_anchor,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def sites(slab: Atoms):
    return [build_site(slab, core=[0]), build_site(slab, core=[1])]


def make(adsorbate: Atoms, distances: list[float], label: str) -> Conformer:
    return Conformer(
        adsorbate=adsorbate,
        anchor=resolve_anchor(adsorbate, "com"),
        orientation=SuperFibonacciSO3(2),
        distances=np.array(distances),
        label=label,
    )


@pytest.fixture(scope="module")
def generator(slab: Atoms, sites):
    return MultiAdsorbateGenerator(
        slab,
        sites,
        [
            make(molecule("CO"), [2.0], "co"),
            make(molecule("H2O"), [2.0, 2.5], "water"),
        ],
        n_oa=3,
    )


def test_budget(generator: MultiAdsorbateGenerator) -> None:
    # CO: 3 directions * 2 orientations * 1 distance * 2 sites = 12
    # H2O: 3 * 2 * 2 * 2 = 24
    assert generator.n_candidates == 36
    assert generator.offsets == (0, 12)


def test_conformer_index(generator: MultiAdsorbateGenerator) -> None:
    assert generator.trial(1).meta["conformer"] == 0
    assert generator.trial(12).meta["conformer"] == 0
    assert generator.trial(13).meta["conformer"] == 1
    last = generator.trial(36)
    assert last.meta["conformer"] == 1
    assert last.index == 36
    assert last.n_total == 36


def test_generate_across_conformers(generator: MultiAdsorbateGenerator) -> None:
    chunk = list(generator.generate(11, 14))
    assert [trial.index for trial in chunk] == [11, 12, 13, 14]
    assert [trial.meta["conformer"] for trial in chunk] == [0, 0, 1, 1]
    assert len(list(generator.generate())) == 36
    with pytest.raises(IndexError):
        list(generator.generate(0, 1))
    with pytest.raises(IndexError):
        list(generator.generate(1, 37))


def test_empty_inputs(slab: Atoms, sites) -> None:
    conformer = make(molecule("CO"), [2.0], "co")
    with pytest.raises(ValueError):
        MultiAdsorbateGenerator(slab, [], [conformer])
    with pytest.raises(ValueError):
        MultiAdsorbateGenerator(slab, sites, [])


def test_signature(generator: MultiAdsorbateGenerator) -> None:
    signature = generator.signature()
    assert signature["kind"] == "multi_adsorbate"
    assert len(signature["conformers"]) == 2
    assert signature["conformers"][0]["label"] == "co"


def test_components_derive_their_sites_from_core_indices(
    generator: MultiAdsorbateGenerator,
    slab: Atoms,
) -> None:
    """Each single-site component keeps the core of the site it stands for."""
    for component in generator.components:
        print("1", type(component))
        for single in component.components:
            print("2", type(single))
            assert list(single.core) == list(single.site.core_indices)
            assert np.allclose(
                single.site.center,
                slab.positions[single.core].mean(axis=0),
            )
