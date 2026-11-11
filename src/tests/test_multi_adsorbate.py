# ruff: noqa D103 E501
"""Tests of the multi-adsorbate candidate space."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.generate.adsorbate import ExplicitEnsemble, RigidAdsorbate
from adsorption.generate import AdsorptionConfigurationGenerator, Sampling
from adsorption.geometry import (
    ExplicitSites,
    SuperFibonacciSO3,
    resolve_anchor,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


def make(atoms: Atoms, distances: list[float], label: str) -> RigidAdsorbate:
    return RigidAdsorbate(
        atoms,
        resolve_anchor(atoms, "com"),
        label,
        sampling=Sampling(
            n_oa=3,
            orientation=SuperFibonacciSO3(2),
            distances=np.array(distances),
        ),
    )


@pytest.fixture(scope="module")
def generator(slab: Atoms) -> AdsorptionConfigurationGenerator:
    return AdsorptionConfigurationGenerator(
        slab,
        ExplicitSites([[0], [1]]),
        ExplicitEnsemble(
            [
                make(molecule("CO"), [2.0], "co"),
                make(molecule("H2O"), [2.0, 2.5], "water"),
            ]
        ),
        n_oa=3,
    )


def test_budget(generator: AdsorptionConfigurationGenerator) -> None:
    # CO: 3 directions * 2 orientations * 1 distance * 2 sites = 12
    # H2O: 3 * 2 * 2 * 2 = 24
    assert generator.n_candidates == 36
    # One flat component per (conformer, site) pair.
    assert generator.factors()["n_components"] == 4
    assert generator.offsets == (0, 6, 12, 24)


def test_conformer_index(generator: AdsorptionConfigurationGenerator) -> None:
    assert generator.trial(1).meta["conformer"] == 0
    assert generator.trial(12).meta["conformer"] == 0
    assert generator.trial(13).meta["conformer"] == 1
    last = generator.trial(36)
    assert last.meta["conformer"] == 1
    assert last.index == 36
    assert last.n_total == 36


def test_generate_across_conformers(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    chunk = list(generator.generate(11, 14))
    assert [trial.index for trial in chunk] == [11, 12, 13, 14]
    assert [trial.meta["conformer"] for trial in chunk] == [0, 0, 1, 1]
    assert len(list(generator.generate())) == 36
    with pytest.raises(IndexError):
        list(generator.generate(0, 1))
    with pytest.raises(IndexError):
        list(generator.generate(1, 37))


def test_empty_inputs(slab: Atoms) -> None:
    with pytest.raises(ValueError):
        AdsorptionConfigurationGenerator(
            slab,
            ExplicitSites([]),
            ExplicitEnsemble([]),
        )
    with pytest.raises(ValueError):
        ExplicitEnsemble([]).resolve()


def test_signature(generator: AdsorptionConfigurationGenerator) -> None:
    signature = generator.signature()
    assert signature["kind"] == "adsorption"
    assert signature["adsorbates"]["n"] == 2
    assert signature["factors"]["n_components"] == 4


def test_trial_carries_the_conformer_and_the_site(
    generator: AdsorptionConfigurationGenerator,
    slab: Atoms,
) -> None:
    """Every trial keeps both factor coordinates."""
    trial = generator.trial(13)
    assert trial.meta["conformer"] == 1
    assert trial.meta["conformer_label"] == "water"
    assert trial.meta["site_core"] == [0]
    assert trial.meta["site_center"] == pytest.approx(list(slab.positions[0]))
