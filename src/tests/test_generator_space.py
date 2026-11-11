# ruff: noqa D103 E501
"""Tests of the single generator's index law and of the new sources."""

import pickle

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.generate.adsorbate import (
    RotamerEnsemble,
    deduplicate_rigids,
)
from adsorption.generate import (
    AdsorptionConfigurationGenerator,
    Sampling,
)
from adsorption.geometry import (
    ExplicitSites,
    GraphAtomsSites,
    SiteSelection,
    place,
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


def test_build_matches_the_placement_primitive() -> None:
    """The cached build equals ``substrate + place(...)`` for every trial."""
    substrate = fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)
    substrate.set_tags(np.arange(len(substrate)))
    generator = AdsorptionConfigurationGenerator(
        substrate,
        ExplicitSites([[0], [1]]),
        molecule("CO"),
        n_oa=3,
        n_rot=2,
        n_d=2,
    )
    for trial in generator:
        bound = generator._components[trial.meta["site"]].adsorbate
        target = np.asarray(trial.meta["site_center"]) + trial.meta[
            "distance"
        ] * bound.directions[trial.meta["i"] - 1]
        placed = place(
            bound.rigid.atoms,
            anchor=bound.rigid.anchor,
            target=target,
            quaternion=bound.orientation.quaternion(trial.meta["j"] - 1),
        )
        expected = substrate.copy()
        expected.extend(placed)
        assert np.array_equal(trial.atoms.numbers, expected.numbers)
        assert np.allclose(trial.atoms.positions, expected.positions)
        assert np.allclose(trial.atoms.cell.array, expected.cell.array)
        assert np.array_equal(trial.atoms.pbc, expected.pbc)
        assert np.array_equal(trial.atoms.get_tags(), expected.get_tags())
        # Trials never share the mutable position block.
        trial.atoms.positions[-1] += 1.0
        assert not np.allclose(
            generator.trial(trial.index).atoms.positions,
            trial.atoms.positions,
        )


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
    assert [tuple(r.provenance["dihedrals"]) for r in first] == [
        tuple(r.provenance["dihedrals"]) for r in second
    ]
    assert all(r.provenance["n_rotors"] == 3 for r in first)


def test_uniform_dihedral_counts_are_grids() -> None:
    pytest.importorskip("rdkit")
    rigids = RotamerEnsemble(
        "CCCCCC",
        dihedrals=(6, 6, 6),
        max_rotamers=9,
    ).resolve()
    assert len(rigids) == 9
    assert all(rigid.provenance["truncated"] for rigid in rigids)
    assert rigids[0].provenance["dihedrals"] == (0.0, 0.0, 0.0)
    assert rigids[1].provenance["dihedrals"] == (0.0, 0.0, 60.0)


def test_ase_dihedrals_match_the_requested_sequence() -> None:
    pytest.importorskip("rdkit")
    from adsorption.utils.rdutils import prepare_source, rotatable_bonds

    molecule, atoms = prepare_source("CCCCCC", 0, 20261030)
    rotors = rotatable_bonds(molecule)
    rigids = RotamerEnsemble(
        atoms,
        dihedrals=[[60.0, 180.0], [180.0], [60.0, 300.0]],
        max_rotamers=2,
    ).resolve()
    assert len(rigids) == 2
    for rigid in rigids:
        targets = rigid.provenance["dihedrals"]
        for rotor, target in zip(rotors, targets):
            assert rigid.atoms.get_dihedral(*rotor) == pytest.approx(
                target, abs=1e-6
            )


def test_component_starts_cover_every_component(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    starts = list(generator.iter_component_starts())
    assert len(starts) == generator.factors()["n_components"]
    assert [trial.index for trial in starts] == [
        offset + 1 for offset in generator.offsets
    ]


def test_conformer_deduplication_drops_mirror_rotamers() -> None:
    """Furfural's +theta / -theta rotamers collapse to one structure."""
    plain = RotamerEnsemble(
        "O=Cc1ccco1", dihedrals=10, max_rotamers=10, deduplicate=False
    ).resolve()
    deduped = RotamerEnsemble(
        "O=Cc1ccco1", dihedrals=10, max_rotamers=10
    ).resolve()
    assert len(plain) == 10
    assert len(deduped) == 6
    assert deduped[0].atoms == plain[0].atoms


def test_conformer_deduplication_keeps_distinct_rotamers() -> None:
    """A linear chain keeps every rotamer at the cap of the grid."""
    rigids = RotamerEnsemble(
        "CCCCCC", dihedrals=10, max_rotamers=64
    ).resolve()
    assert len(rigids) == 64


def test_deduplicate_rigids_is_a_no_op_for_distinct_structures() -> None:
    """Different structures survive in their original order."""
    rigids = RotamerEnsemble(
        "O=Cc1ccco1", dihedrals=10, max_rotamers=10, deduplicate=False
    ).resolve()
    unique = deduplicate_rigids(rigids)
    assert len(unique) == 6
    order = {rigid.label: i for i, rigid in enumerate(rigids)}
    positions = [order[rigid.label] for rigid in unique]
    assert positions == sorted(positions)
    assert positions[0] == 0
