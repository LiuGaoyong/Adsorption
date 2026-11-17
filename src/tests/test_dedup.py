# ruff: noqa D103 E501
"""Tests of the trial deduplicator stage."""

import numpy as np
import pytest
from ase.build import molecule
from ase.cluster import Octahedron

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    FilterStage,
    Trial,
)
from adsorption.base.pipeline import evaluate_trial
from adsorption.dedup import TrialDeduplicator
from adsorption.filters import GeometryFilter


def make_cluster_trial(dx: float = 0.0, index: int = 1) -> Trial:
    """Return a non-periodic Cu cluster with a CO adsorbate."""
    cluster = Octahedron("Cu", 3)
    co = molecule("CO")
    apex = cluster.positions[int(np.argmax(cluster.positions[:, 2]))]
    co.positions += apex + np.array([0.0, 0.0, 3.2 + dx])
    atoms = cluster.copy()
    atoms.extend(co)
    return Trial(
        atoms, index=index, n_total=2, meta={"n_substrate": len(cluster)}
    )


def test_the_first_structure_is_unique() -> None:
    stage = TrialDeduplicator()
    verdict = stage.evaluate(make_cluster_trial())
    assert verdict.accepted is True
    assert verdict.reason == "unique"
    assert stage.n_seen == 1
    assert stage.n_duplicates == 0


def test_a_repeated_structure_is_a_duplicate() -> None:
    stage = TrialDeduplicator()
    stage.evaluate(make_cluster_trial(index=1))
    verdict = stage.evaluate(make_cluster_trial(index=2))
    assert verdict.accepted is False
    assert verdict.reason == "duplicate"
    assert stage.n_duplicates == 1
    assert stage.n_seen == 1


def test_a_moved_structure_is_not_a_duplicate() -> None:
    stage = TrialDeduplicator()
    stage.evaluate(make_cluster_trial(dx=0.0))
    assert stage.evaluate(make_cluster_trial(dx=0.6)).reason == "unique"


def test_a_custom_key_replaces_the_bond_list_hash() -> None:
    stage = TrialDeduplicator(key=lambda trial: "same")
    stage.evaluate(make_cluster_trial(dx=0.0))
    assert stage.evaluate(make_cluster_trial(dx=1.0)).reason == "duplicate"


def test_reset_forgets_the_seen_keys() -> None:
    stage = TrialDeduplicator()
    stage.evaluate(make_cluster_trial())
    stage.reset()
    assert stage.n_seen == 0
    assert stage.evaluate(make_cluster_trial()).reason == "unique"


def test_the_cutoff_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TrialDeduplicator(cutoff=0.0)
    with pytest.raises(ValueError):
        TrialDeduplicator(cutoff=float("nan"))


def test_the_deduplicator_is_a_stage_but_not_a_filter() -> None:
    stage = TrialDeduplicator()
    assert isinstance(stage, FilterStage)
    assert not isinstance(stage, AdsorptionConfigurationFilterABC)
    assert isinstance(GeometryFilter(scale=1.0), FilterStage)


def test_the_pipeline_stops_at_the_duplicate() -> None:
    stage = TrialDeduplicator()
    first = evaluate_trial(make_cluster_trial(index=1), [stage])
    second = evaluate_trial(make_cluster_trial(index=2), [stage])
    assert first.first_failed is None
    assert second.first_failed == "TrialDeduplicator"
    assert second.verdicts[-1].reason == "duplicate"


def test_the_deduplicator_runs_after_a_geometric_screen() -> None:
    filters = [GeometryFilter(scale=1.0), TrialDeduplicator()]
    records = [evaluate_trial(make_cluster_trial(index=i), filters) for i in (1, 2)]
    assert records[0].first_failed is None
    assert records[1].first_failed == "TrialDeduplicator"
    assert len(records[1].verdicts) == 2


def test_signature_describes_the_stage() -> None:
    signature = TrialDeduplicator(cutoff=3.5).signature()
    assert signature["kind"] == "deduplicator"
    assert signature["cutoff"] == 3.5


def test_the_deduplicator_reuses_a_recorded_key() -> None:
    """A key stored upstream with the same cutoff is reused."""
    stage = TrialDeduplicator(cutoff=4.0)
    trial = make_cluster_trial()
    trial.meta["canonical_key"] = "stored-key"
    trial.meta["canonical_key_cutoff"] = 4.0
    assert stage.canonical_key(trial) == "stored-key"
    stage.cutoff = 3.0
    assert stage.canonical_key(trial) != "stored-key"
