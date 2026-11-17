"""Tests of the analysis of relaxed structures.

Labels, the distance fingerprint, the grouping into minima and the criteria
of a published configuration. No calculator is involved.
"""

import numpy as np
import pytest
from adsorption._helpers import slab_with_co

from adsorption.analysis import (
    LabelThresholds,
    cluster_minima,
    fingerprint,
    label_structure,
    matches_reference,
)


def test_labels_cover_the_four_classes() -> None:
    """Intact, dissociated, desorbed and reconstructed are told apart."""
    atoms, n_substrate = slab_with_co()
    assert label_structure(atoms, atoms, n_substrate)["label"] == "intact"

    broken = atoms.copy()
    broken.positions[-1] += [2.5, 0.0, 0.0]
    result = label_structure(atoms, broken, n_substrate)
    assert result["label"] == "dissociated"
    assert result["n_bonds_broken"] == 1

    gone = atoms.copy()
    gone.positions[n_substrate:, 2] += 4.0
    assert label_structure(atoms, gone, n_substrate)["label"] == "desorbed"

    moved = atoms.copy()
    moved.positions[0] += [0.0, 0.0, -1.5]
    assert (
        label_structure(atoms, moved, n_substrate)["label"] == "reconstructed"
    )
    loose = LabelThresholds(max_substrate_shift=2.0)
    assert (
        label_structure(atoms, moved, n_substrate, loose)["label"] == "intact"
    )


def test_fingerprint_is_invariant_and_continuous() -> None:
    """Permutation and translation leave the fingerprint unchanged."""
    atoms, n_substrate = slab_with_co()
    base = fingerprint(atoms, n_substrate)

    permuted = atoms.copy()
    order = np.r_[
        np.arange(n_substrate)[::-1], np.arange(n_substrate, len(atoms))
    ]
    permuted = permuted[order]
    assert np.allclose(fingerprint(permuted, n_substrate), base)

    shifted = atoms.copy()
    shifted.positions += atoms.cell[0] / 2.0  # one lattice translation in x
    shifted.wrap()
    assert np.allclose(fingerprint(shifted, n_substrate), base, atol=1e-8)

    jitter = atoms.copy()
    rng = np.random.default_rng(0)
    jitter.positions += rng.normal(scale=0.01, size=jitter.positions.shape)
    assert np.abs(fingerprint(jitter, n_substrate) - base).max() < 0.08


def test_cluster_merges_jitter_and_separates_sites() -> None:
    """Jittered copies merge; a different site is a different minimum."""
    top, n_substrate = slab_with_co()
    bridge, _ = slab_with_co(height=1.5, shift=1.28)
    rng = np.random.default_rng(1)
    structures, energies = [], []
    for base, energy in ((top, -1.0), (bridge, -0.8)):
        for _ in range(4):
            copy = base.copy()
            copy.positions += rng.normal(scale=0.01, size=copy.positions.shape)
            structures.append(copy)
            energies.append(energy + rng.normal(scale=0.002))
    prints = [fingerprint(atoms, n_substrate) for atoms in structures]
    labels = cluster_minima(prints, energies, tau_f=0.1, tau_e=0.02)
    assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # The energy tolerance alone keeps equal geometries apart.
    split = cluster_minima([prints[0], prints[0]], [0.0, 1.0])
    assert split.tolist() == [0, 1]


def test_reference_criteria_count_contacts() -> None:
    """A contact criterion counts adsorbate-substrate pairs within reach."""
    atoms, n_substrate = slab_with_co()
    top = {
        "name": "top",
        "contacts": [
            {
                "adsorbate": "C",
                "substrate": "Cu",
                "r_max": 2.1,
                "n_min": 1,
                "n_max": 1,
            }
        ],
    }
    hollow = {
        "name": "hollow",
        "contacts": [
            {"adsorbate": "C", "substrate": "Cu", "r_max": 2.1, "n_min": 3}
        ],
    }
    assert matches_reference(atoms, n_substrate, top, label="intact")
    assert not matches_reference(atoms, n_substrate, hollow, label="intact")
    wrong_label = {"name": "x", "label": "dissociated"}
    assert not matches_reference(
        atoms, n_substrate, wrong_label, label="intact"
    )
    with pytest.raises(ValueError):
        matches_reference(atoms, n_substrate, {"name": "empty"})
