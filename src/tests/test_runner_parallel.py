# ruff: noqa D103 D101 E501
"""Tests of the parallel runner engines and the filter actors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.db import connect

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    Trial,
    Verdict,
)
from adsorption.base.pipeline import AdsorptionConfigurationPipeline
from adsorption.runner import _run
from adsorption.runner._run import (
    _FilterActor,
    _ray_available,
    _resolve_engine,
    run_config,
)


from _helpers import CountingGenerator

class _Even(AdsorptionConfigurationFilterABC):
    """Accept even candidate indices."""

    def _evaluate(self, trial: Trial) -> Verdict:
        return Verdict(trial.index % 2 == 0, float(trial.index), "even")


class _AboveTwo(AdsorptionConfigurationFilterABC):
    """Accept candidates with an index above two."""

    def _evaluate(self, trial: Trial) -> Verdict:
        return Verdict(trial.index > 2, float(trial.index), "gt2")


def test_filter_actor_matches_the_pipeline() -> None:
    filters = [_Even(), _AboveTwo()]
    actor = _FilterActor(filters)
    pipeline = AdsorptionConfigurationPipeline(
        CountingGenerator(5),
        filters,
    )
    for record, trial in zip(pipeline.run(), CountingGenerator(5)):
        result = actor.evaluate(trial)
        assert result.payload["index"] == record.trial.index
        assert result.payload["first_failed"] == record.first_failed
        assert len(result.payload["verdicts"]) == len(record.verdicts)
        if record.first_failed is None:
            assert isinstance(result.atoms, Atoms)
            assert result.atoms.info["index"] == record.trial.index
        else:
            assert result.atoms is None


def test_resolve_engine_follows_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_run, "_ray_available", lambda: False)
    assert _resolve_engine("auto", 2) == "joblib"
    assert _resolve_engine("auto", 1) == "serial"
    assert _resolve_engine("joblib", 1) == "serial"
    monkeypatch.setattr(_run, "_ray_available", lambda: True)
    assert _resolve_engine("auto", 2) == "ray"
    assert _resolve_engine("ray", 2) == "ray"


def test_resolve_engine_requires_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_run, "_ray_available", lambda: False)
    with pytest.raises(ImportError):
        _resolve_engine("ray", 2)


def test_resolve_engine_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        _resolve_engine("mpi", 2)


def _config(parallel: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "substrate": {
            "_target_": "ase.build.fcc111",
            "symbol": "Cu",
            "size": [2, 2, 2],
            "vacuum": 8.0,
            "orthogonal": True,
            "periodic": True,
        },
        "adsorbate": {
            "_target_": "ase.build.molecule",
            "name": "CO",
            "conformers": "none",
        },
        "sites": [[4], [5]],
        "protocol": {
            "n_oa": 2,
            "n_rot": 1,
            "n_d": 1,
            "d_min": 2.0,
            "d_max": 2.0,
        },
        "filters": [],
        "budget": {"max_candidates": 100},
        "parallel": {"n_jobs": 2, "backend": "loky", **(parallel or {})},
    }


def _db_rows(path: Path) -> list[tuple[Any, ...]]:
    """Return a comparable snapshot of an accepted-structure database."""
    with connect(str(path)) as db:
        return [
            (
                int(row.key_value_pairs["index"]),
                row.toatoms().numbers.tobytes(),
                np.round(row.toatoms().positions, 6).tobytes(),
                sorted(row.key_value_pairs.items()),
            )
            for row in db.select()
        ]


def test_joblib_matches_serial(tmp_path: Path) -> None:
    cfg = _config()
    serial = run_config(cfg, output=tmp_path / "serial", n_jobs=1)
    parallel = run_config(
        cfg,
        output=tmp_path / "parallel",
        n_jobs=2,
        engine="joblib",
    )
    assert serial.engine == "serial"
    assert parallel.engine == "joblib"
    assert parallel.n_actors == 0
    assert serial.total == parallel.total == 4
    assert serial.accepted == parallel.accepted == 4
    assert (tmp_path / "serial" / "records.jsonl").read_bytes() == (
        tmp_path / "parallel" / "records.jsonl"
    ).read_bytes()
    assert _db_rows(tmp_path / "serial" / "accepted.db") == _db_rows(
        tmp_path / "parallel" / "accepted.db"
    )


@pytest.mark.filterwarnings("ignore")
def test_ray_matches_serial(tmp_path: Path) -> None:
    if not _ray_available():
        pytest.skip("ray is not installed")
    cfg = _config()
    serial = run_config(cfg, output=tmp_path / "serial", n_jobs=1)
    parallel = run_config(
        cfg,
        output=tmp_path / "ray",
        n_jobs=2,
        engine="ray",
    )
    assert parallel.engine == "ray"
    assert parallel.n_actors == 2
    assert parallel.total == serial.total == 4
    assert parallel.accepted == serial.accepted == 4
    assert (tmp_path / "serial" / "records.jsonl").read_bytes() == (
        tmp_path / "ray" / "records.jsonl"
    ).read_bytes()
    assert _db_rows(tmp_path / "serial" / "accepted.db") == _db_rows(
        tmp_path / "ray" / "accepted.db"
    )


def _dedup_config() -> dict[str, Any]:
    """A single O atom sampled on SO(3): every orientation is the same
    structure, so each ``(direction, distance)`` pair has ``n_rot - 1``
    duplicates that straddle chunk and actor boundaries."""
    cfg = _config({"chunk_factor": 4})
    cfg["adsorbate"] = {
        "_target_": "ase.Atoms",
        "symbols": "O",
        "conformers": "none",
    }
    cfg["sites"] = [[4]]
    cfg["protocol"] = {
        "n_oa": 8,
        "n_rot": 4,
        "n_d": 2,
        "d_min": 2.0,
        "d_max": 3.0,
        "orientation_kind": "so3",
    }
    cfg["filters"] = [
        {"_target_": "adsorption.dedup.TrialDeduplicator", "cutoff": 4.0}
    ]
    return cfg


def _assert_global_dedup(tmp_path: Path, serial, parallel, name: str) -> None:
    assert serial.total == parallel.total == 64
    assert serial.accepted == parallel.accepted == 16
    assert parallel.first_failed["TrialDeduplicator"] == 48
    assert (tmp_path / "serial" / "records.jsonl").read_bytes() == (
        tmp_path / name / "records.jsonl"
    ).read_bytes()
    assert _db_rows(tmp_path / "serial" / "accepted.db") == _db_rows(
        tmp_path / name / "accepted.db"
    )
    summary = json.loads((tmp_path / name / "summary.json").read_text())
    assert summary["n_duplicates_global"] > 0


def test_joblib_deduplicates_across_chunks(tmp_path: Path) -> None:
    cfg = _dedup_config()
    serial = run_config(cfg, output=tmp_path / "serial", n_jobs=1)
    parallel = run_config(
        cfg, output=tmp_path / "joblib", n_jobs=4, engine="joblib"
    )
    _assert_global_dedup(tmp_path, serial, parallel, "joblib")


@pytest.mark.filterwarnings("ignore")
def test_ray_deduplicates_across_actors(tmp_path: Path) -> None:
    if not _ray_available():
        pytest.skip("ray is not installed")
    cfg = _dedup_config()
    serial = run_config(cfg, output=tmp_path / "serial", n_jobs=1)
    parallel = run_config(
        cfg, output=tmp_path / "ray", n_jobs=2, engine="ray"
    )
    _assert_global_dedup(tmp_path, serial, parallel, "ray")


def test_storage_policy_accepted_only_and_adsorbate(tmp_path: Path) -> None:
    """The record and structure policies shrink both sidecars."""
    cfg = _config()
    cfg["filters"] = [{"_target_": "_helpers.EvenFilter"}]
    cfg["storage"] = {
        "records": "accepted",
        "compression": "gzip",
        "structures": "adsorbate",
    }
    summary = run_config(cfg, output=tmp_path / "out", n_jobs=1)
    assert summary.accepted == 2
    assert (tmp_path / "out" / "records.jsonl.gz").exists()
    assert not (tmp_path / "out" / "records.jsonl").exists()
    import gzip

    with gzip.open(tmp_path / "out" / "records.jsonl.gz", "rt") as stream:
        assert len(stream.readlines()) == 2
    with connect(str(tmp_path / "out" / "accepted.db")) as db:
        rows = list(db.select())
        assert len(rows) == 2
        first = rows[0].toatoms()
        assert len(first) == 2  # the CO adsorbate only
        assert rows[0].key_value_pairs["n_substrate"] == \
            rows[0].key_value_pairs["n_substrate"]
        assert "index" in rows[0].key_value_pairs
