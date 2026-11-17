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
    adsorbate_azimuth,
    adsorbate_tilt,
    matches_reference,
    nearest_site,
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


def _benzene_on_pt(kind: str, turn: float) -> tuple:
    """Return flat benzene over a site of Pt(111), turned by ``turn`` deg."""
    from ase.build import fcc111, molecule

    slab = fcc111("Pt", (3, 3, 3), a=3.979, vacuum=8.0)
    slab.pbc = True
    top = slab.positions[slab.get_tags() == 1]
    second = slab.positions[slab.get_tags() == 2]
    row = top[1] - top[0]
    other = top[3] - top[0]
    target = {
        "top": top[4],
        "bridge": top[4] + 0.5 * row,
        "hollow_a": top[4] + (row + other) / 3.0,
        "hollow_b": top[4] + (2.0 * row - other) / 3.0,
    }
    point = target[kind] if kind in target else None
    if point is None:
        # fcc or hcp: pick the hollow by what lies under it.
        for name in ("hollow_a", "hollow_b"):
            under = np.linalg.norm(
                second[:, :2] - target[name][:2], axis=1
            ).min()
            if (under < 0.5) == (kind == "hcp"):
                point = target[name]
    ring = molecule("C6H6")
    # In ASE's benzene the C-C bonds are 30 degrees off the x axis, and the
    # first Pt row is along x.
    ring.rotate(turn + 30.0, "z")
    ring.translate(point - ring.get_center_of_mass() + [0.0, 0.0, 2.1])
    atoms = slab.copy()
    atoms.extend(ring)
    return atoms, len(slab)


@pytest.mark.parametrize(
    ("kind", "turn"),
    [("top", 0.0), ("bridge", 30.0), ("fcc", 0.0), ("hcp", 30.0)],
)
def test_site_and_azimuth_identify_benzene(kind: str, turn: float) -> None:
    """Site kind and azimuth tell the published benzene sites apart."""
    atoms, n_substrate = _benzene_on_pt(kind, turn)
    found, distance = nearest_site(atoms, n_substrate)
    assert found == kind and distance < 0.05
    assert adsorbate_azimuth(atoms, n_substrate) == pytest.approx(turn, abs=0.5)
    reference = {
        "name": f"{kind}{turn:.0f}",
        "site": {"kind": kind, "tol": 0.4},
        "azimuth": {"value": turn, "tol": 8.0},
    }
    assert matches_reference(atoms, n_substrate, reference)
    turned = dict(reference, azimuth={"value": 30.0 - turn, "tol": 8.0})
    assert not matches_reference(atoms, n_substrate, turned)
    moved = dict(reference, site={"kind": "top", "tol": 0.4})
    assert matches_reference(atoms, n_substrate, moved) == (kind == "top")
    assert adsorbate_tilt(atoms, n_substrate) == pytest.approx(0.0, abs=1e-6)
    assert matches_reference(
        atoms, n_substrate, dict(reference, tilt={"max": 5})
    )
    upright = dict(reference, tilt={"min": 60})
    assert not matches_reference(atoms, n_substrate, upright)
    hollow = dict(reference, site={"kind": "hollow", "tol": 0.4})
    assert matches_reference(atoms, n_substrate, hollow) == (
        kind in {"fcc", "hcp"}
    )


def test_contact_criterion_respects_substrate_tags() -> None:
    """``substrate_tag`` restricts a contact count to one layer."""
    atoms, n_substrate = slab_with_co()
    tags = np.zeros(len(atoms), dtype=int)
    heights = atoms.positions[:n_substrate, 2]
    tags[:n_substrate] = np.where(heights > heights.max() - 0.5, 1, 2)
    atoms.set_tags(tags)
    entry = {"adsorbate": "C", "substrate": "Cu", "r_max": 2.1}
    on_top = {"name": "t", "contacts": [dict(entry, substrate_tag=1)]}
    below = {"name": "b", "contacts": [dict(entry, substrate_tag=[2])]}
    assert matches_reference(atoms, n_substrate, on_top, label="intact")
    assert not matches_reference(atoms, n_substrate, below, label="intact")


def test_arms_are_summarised_over_rungs_and_seeds() -> None:
    """Replicates of an arm give a recall range and a first-passage budget."""
    from adsorption.analysis import analyse_runs
    from adsorption.analysis.recall import ENERGY_KEY, split_run_name

    assert split_run_name("random/g01000@3") == ("random", "g01000", "3")
    assert split_run_name("deterministic/g00100") == (
        "deterministic",
        "g00100",
        "",
    )

    def relaxed(shift: float, energy: float):
        atoms, n_substrate = slab_with_co(shift=shift)
        atoms.info.update(
            {"label": "intact", "converged": True, ENERGY_KEY: energy}
        )
        return atoms

    top = [relaxed(0.0, -1.0)]
    both = [relaxed(0.0, -1.0), relaxed(1.28, -0.9)]
    report = analyse_runs(
        {
            "deterministic/g1": top,
            "deterministic/g2": both,
            "random/g1@1": top,
            "random/g1@2": both,
            "random/g2@1": both,
            "random/g2@2": both + top,
        }
    )
    assert report["reference"]["n_minima"] == 2
    arms = report["arms"]
    assert arms["deterministic"]["n_seeds"] == 1
    assert arms["deterministic"]["first_passage_budget"] == 2
    assert arms["random"]["n_seeds"] == 2
    rung = arms["random"]["rungs"]["g1"]
    assert rung["n_runs"] == 2
    assert (rung["recall_min"], rung["recall_max"]) == (0.5, 1.0)
    assert rung["recall_median"] == 0.75
    assert arms["random"]["first_passage_by_seed"] == {"1": 2, "2": 2}
    assert arms["random"]["first_passage_budget"] == 2
    rows = {row["run"]: row for row in report["runs"]}
    assert rows["random/g2@2"]["seed"] == "2"
    # Every run names the reference minima it reached.
    assert len(rows["deterministic/g1"]["minimum_ids"]) == 1
    assert set(rows["deterministic/g1"]["minimum_ids"]) < set(
        rows["deterministic/g2"]["minimum_ids"]
    )
    # An arm that never reaches the target has no first-passage budget.
    short = analyse_runs({"deterministic/g1": both, "random/g1@1": top})
    assert short["arms"]["random"]["first_passage_budget"] is None
