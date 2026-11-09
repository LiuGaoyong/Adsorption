# ruff: noqa D103 E501
"""Tests of the single generator's index law and of the new sources."""

import pickle

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.adsorbate import RotamerEnsemble
from adsorption.generate import (
    AdsorptionConfigurationGenerator,
    Sampling,
)
from adsorption.geometry import (
    ExplicitSites,
    GraphAtomsSites,
    SiteSelection,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def generator(slab: Atoms) -> AdsorptionConfigurationGenerator:
    return AdsorptionConfigurationGenerator(
        slab,
        ExplicitSites([[0], [1]]),
        molecule("CO"),
        n_oa=3,
        n_rot=2,
        n_d=2,
    )


def test_generate_matches_trial(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    total = generator.n_candidates
    stream = list(generator.generate())
    direct = [generator.trial(index) for index in range(1, total + 1)]
    assert [trial.index for trial in stream] == [
        trial.index for trial in direct
    ]
    for left, right in zip(stream, direct):
        assert left.meta == right.meta
        assert np.allclose(left.atoms.positions, right.atoms.positions)


def test_any_shard_is_a_contiguous_partition(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    total = generator.n_candidates
    for cut in range(1, total):
        left = list(generator.generate(1, cut))
        right = list(generator.generate(cut + 1, total))
        assert [trial.index for trial in left + right] == list(
            range(1, total + 1)
        )
        assert len({trial.index for trial in left + right}) == total


def test_single_component_attributes(slab: Atoms) -> None:
    generator = AdsorptionConfigurationGenerator(
        slab,
        0,
        molecule("CO"),
        n_oa=4,
        orientation_kind="fixed",
        distances=[2.0, 2.5],
    )
    assert [int(index) for index in generator.core] == [0]
    assert generator.n_oa == 4
    assert generator.n_rot == 1
    assert generator.n_d == 2
    assert generator.distances == pytest.approx([2.0, 2.5])
    assert generator.n_candidates == 8


def test_multi_component_attributes_raise(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    with pytest.raises(AttributeError):
        _ = generator.n_oa
    with pytest.raises(AttributeError):
        _ = generator.site


def test_generator_is_picklable(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    copy = pickle.loads(pickle.dumps(generator))
    assert copy.n_candidates == generator.n_candidates
    assert copy.trial(1).meta == generator.trial(1).meta


def test_graphatoms_sites_are_unique_and_sized(slab: Atoms) -> None:
    resolved = GraphAtomsSites(max_ncore=3).resolve(slab)
    sizes = [len(spec.geometry.core_indices) for spec in resolved.specs]
    assert sizes == [1, 2, 3]


def test_graphatoms_selection_caps_each_core_size(slab: Atoms) -> None:
    source = GraphAtomsSites(
        max_ncore=3,
        selection=SiteSelection(per_core={1: 1, 2: 1, 3: 0}),
    )
    resolved = source.resolve(slab)
    assert [len(spec.geometry.core_indices) for spec in resolved.specs] == [
        1,
        2,
    ]


def test_graphatoms_all_sites_are_a_superset(slab: Atoms) -> None:
    unique = GraphAtomsSites(max_ncore=3, unique=True).resolve(slab)
    every = GraphAtomsSites(max_ncore=3, unique=False).resolve(slab)
    assert len(every.specs) >= len(unique.specs)


def test_rotamer_enumeration_is_deterministic() -> None:
    pytest.importorskip("rdkit")
    first = RotamerEnsemble("CCCCCC", max_rotamers=4).resolve()
    second = RotamerEnsemble("CCCCCC", max_rotamers=4).resolve()
    assert len(first) == len(second)
    assert [tuple(r.provenance["angles"]) for r in first] == [
        tuple(r.provenance["angles"]) for r in second
    ]
    assert all(r.provenance["n_rotors"] == 3 for r in first)
