# ruff: noqa D103 E501
"""Tests of the Tier-0 geometric measurements.

The cases are deliberately tiny (a 2x2x2 slab, three directions, two
distances) so the whole module is milliseconds and never touches an energy
engine. The paper-scale numbers come from `campaigns/slurm/light.sbatch`.
"""

import numpy as np
import pytest
import yaml

from adsorption.analysis import light
from adsorption.geometry.grids import FibonacciGrid, LatLonGrid


@pytest.fixture(scope="module")
def campaign() -> dict:
    """Return a minimal process-A campaign configuration."""
    return {
        "substrate": {
            "_target_": "ase.build.fcc111",
            "symbol": "Cu",
            "size": (2, 2, 2),
            "vacuum": 8.0,
            "orthogonal": True,
        },
        "adsorbate": {"_target_": "ase.build.molecule", "name": "CO"},
        "sites": [[0]],
        "protocol": {
            "n_oa": 3,
            "n_rot": 2,
            "n_d": 2,
            "d_min": 1.8,
            "d_max": 2.2,
            "orientation_kind": "fixed",
        },
    }


def test_covering_radius_of_a_tetrahedron() -> None:
    points = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    points /= np.linalg.norm(points, axis=1)[:, None]
    assert light.covering_radius(points) == pytest.approx(
        np.degrees(np.arccos(-1.0 / 3.0)), rel=1e-9
    )


def test_covering_radius_shrinks_with_the_lattice_size() -> None:
    small = light.covering_radius(FibonacciGrid().points(25))
    large = light.covering_radius(FibonacciGrid().points(100))
    assert large < small


def test_height_uniformity_separates_fibonacci_from_latitude() -> None:
    fibonacci = FibonacciGrid().points(100)
    latitude = LatLonGrid().points(100)
    assert light.height_uniformity(fibonacci) < 0.1
    assert light.height_uniformity(latitude) > light.height_uniformity(fibonacci)


def test_so3_covering_radius_of_identical_rotations() -> None:
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
    assert light.so3_covering_radius(quaternions) == pytest.approx(0.0)


def test_so3_quaternions_are_unit_quaternions() -> None:
    quaternions = light.so3_quaternions(16)
    assert quaternions.shape == (16, 4)
    assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0)


def test_index_report_is_bijective(campaign: dict) -> None:
    report = light.index_report(campaign, window=8)
    assert report["n_candidates"] > 1
    assert report["n_index_mismatch"] == 0
    assert report["partition_ok"] is True
    assert report["n_distinct_triples"] <= report["n_candidates"]


def test_budget_report_returns_fractions(campaign: dict) -> None:
    report = light.budget_report(campaign)
    assert report["n_examined"] == report["n_candidates"]
    for key in (
        "outward_fraction",
        "floor_rejection_fraction",
        "low_distance_share",
    ):
        assert 0.0 <= report[key] <= 1.0


def test_dedup_report_has_one_row_per_setting(campaign: dict) -> None:
    report = light.dedup_report(
        campaign, resolutions=(0.01,), cutoffs=(4.0, 5.0)
    )
    assert len(report["table"]) == 2
    for row in report["table"]:
        assert 0.0 <= row["ratio"] <= 1.0
        assert row["n_unique"] <= report["n_floor_survivors"]


def test_throughput_report_examines_the_limit(campaign: dict) -> None:
    report = light.throughput_report(campaign, limit=2)
    assert report["n_examined"] == 2
    assert report["seconds"] >= 0.0


def test_run_suite_writes_reports(campaign: dict, tmp_path) -> None:
    case = tmp_path / "cases" / "demo"
    case.mkdir(parents=True)
    (case / "run.yaml").write_text(yaml.safe_dump(campaign), encoding="utf-8")
    manifest = {
        "grid_sizes": [8],
        "cases": [{"dir": "cases/demo", "samples": 2, "limit": 2}],
    }
    report = light.run_suite(manifest, tmp_path, tmp_path / "out")
    assert report["n_cases"] == 1
    assert (tmp_path / "out" / "grids.json").is_file()
    assert (tmp_path / "out" / "budget" / "cases__demo.json").is_file()
