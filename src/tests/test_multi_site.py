# ruff: noqa D103 E501
"""Tests of the multi-site candidate space."""

import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.generate import (
    AdsorptionConfigurationGenerator,
    Sampling,
)
from adsorption.geometry import ExplicitSites, FixedRotation


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def generator(slab: Atoms) -> AdsorptionConfigurationGenerator:
    return AdsorptionConfigurationGenerator(
        slab,
        ExplicitSites(
            [[0], [1]],
            samplings=[
                Sampling(
                    n_oa=4,
                    orientation=FixedRotation(),
                    distances=[2.0],
                ),
                Sampling(
                    n_oa=6,
                    orientation=FixedRotation(),
                    distances=[2.0, 2.5],
                ),
            ],
        ),
        molecule("CO"),
    )


def test_budget(generator: AdsorptionConfigurationGenerator) -> None:
    assert generator.n_candidates == 4 + 12
    assert len(generator) == 16
    assert generator.offsets == (0, 4)


def test_global_index_maps_to_site(
    generator: AdsorptionConfigurationGenerator,
) -> None:
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


def test_generate_across_the_boundary(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    chunk = list(generator.generate(3, 6))
    assert [trial.index for trial in chunk] == [3, 4, 5, 6]
    assert [trial.meta["site"] for trial in chunk] == [0, 0, 1, 1]
    assert len(list(generator.generate())) == 16
    with pytest.raises(IndexError):
        list(generator.generate(0, 1))
    with pytest.raises(IndexError):
        list(generator.generate(1, 17))


def test_empty_components(slab: Atoms) -> None:
    with pytest.raises(ValueError):
        AdsorptionConfigurationGenerator(
            slab,
            ExplicitSites([]),
            molecule("CO"),
        )


def test_signature(generator: AdsorptionConfigurationGenerator) -> None:
    signature = generator.signature()
    assert signature["kind"] == "adsorption"
    assert signature["sites"]["n"] == 2
    assert signature["factors"]["n_components"] == 2


def test_trial_carries_the_resolved_site(
    generator: AdsorptionConfigurationGenerator,
    slab: Atoms,
) -> None:
    """Every trial records the core and the centre of its site."""
    first = generator.trial(1)
    assert first.meta["site_core"] == [0]
    assert first.meta["site_center"] == pytest.approx(list(slab.positions[0]))
    last = generator.trial(16)
    assert last.meta["site_core"] == [1]
    assert last.meta["site_center"] == pytest.approx(list(slab.positions[1]))
