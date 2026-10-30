# ruff: noqa D103 E501
"""Tests of the reusable evaluation metrics."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from adsorption.geometry import (
    FibonacciGrid,
    LatLonGrid,
    SO3Set,
    build_site,
    matrix_to_quaternion,
    quaternion_to_rotation,
    shortest_arc_rotation,
)
from adsorption.metrics import (
    CascadeRatios,
    accessible_solid_angle,
    admissible_fraction,
    cap_fraction,
    cascade_ratios,
    covering_radius,
    discretization_residual,
    false_rejection,
    high_resolution_fraction,
    kendall,
    matrices_to_quaternions,
    max_false_rejection,
    nearest_neighbor_angles,
    pole_bias,
    quaternion_geodesic,
    regret,
    rotation_covering_radius,
    rotation_nearest_neighbor_angles,
    spearman,
    top_k_ids,
    top_k_recall,
)


@pytest.fixture(scope="module")
def sphere_sets():
    return FibonacciGrid().points(200), LatLonGrid().points(200)


def test_spherical_coverage(sphere_sets) -> None:
    fibonacci, latlon = sphere_sets
    fib_spacing = nearest_neighbor_angles(fibonacci)
    lat_spacing = nearest_neighbor_angles(latlon)
    assert fib_spacing.min() > lat_spacing.min()
    assert covering_radius(fibonacci) < covering_radius(latlon)
    assert pole_bias(fibonacci) < pole_bias(latlon)
    assert pole_bias(latlon) > 1.0


def test_cap_fraction() -> None:
    points = FibonacciGrid().points(512)
    assert cap_fraction(points, [0.0, 0.0, 1.0], np.pi) == pytest.approx(1.0)
    assert cap_fraction(points, [0.0, 0.0, 1.0], 0.0) == 0.0
    half = cap_fraction(points, [0.0, 0.0, 1.0], np.pi / 2)
    assert half == pytest.approx(0.5, abs=0.03)


def test_rotation_metrics() -> None:
    rotations = SO3Set(256)
    quaternions = rotations.quaternions
    assert quaternion_geodesic(quaternions[0], quaternions[0]) == 0.0
    assert quaternion_geodesic(quaternions[0], -quaternions[0]) == 0.0
    half_turn = np.array([0.0, 1.0, 0.0, 0.0])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert quaternion_geodesic(identity, half_turn) == pytest.approx(np.pi)
    angles = rotation_nearest_neighbor_angles(quaternions)
    assert angles.shape == (256,)
    assert np.all(angles > 0.0)
    assert np.all(angles <= np.pi)
    radius = rotation_covering_radius(quaternions)
    assert 0.0 < radius <= np.pi
    with pytest.raises(ValueError):
        rotation_nearest_neighbor_angles(quaternions[:1])


def test_matrix_quaternion_roundtrip() -> None:
    matrices = np.stack(
        [
            shortest_arc_rotation([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            shortest_arc_rotation([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            np.eye(3),
        ]
    )
    quaternions = matrices_to_quaternions(matrices)
    for matrix, quaternion in zip(matrices, quaternions):
        assert np.allclose(quaternion_to_rotation(quaternion), matrix)
        assert np.isclose(np.linalg.norm(quaternion), 1.0)
    single = matrix_to_quaternion(np.eye(3))
    assert np.allclose(single, [1.0, 0.0, 0.0, 0.0])


def test_ranking_metrics() -> None:
    reference = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    perfect = reference.copy()
    assert spearman(reference, perfect) == pytest.approx(1.0)
    assert kendall(reference, perfect) == pytest.approx(1.0)
    reversed_scores = -reference
    assert spearman(reference, reversed_scores) == pytest.approx(-1.0)

    query = np.array([0.0, 1.0, 2.0, 5.0, 4.0, 3.0])
    assert top_k_recall(reference, query, 3) == 1.0
    assert top_k_recall(reference, query, 4) == pytest.approx(0.75)
    assert top_k_recall(reference, reversed_scores, 1) == 0.0
    assert regret(reference, query, 3) == 0.0
    assert regret(reference, reversed_scores, 2) == pytest.approx(4.0)
    assert set(top_k_ids(reference, 2).tolist()) == {0, 1}
    assert set(top_k_ids(-reference, 2, higher_is_better=True).tolist()) == {0, 1}
    with pytest.raises(ValueError):
        top_k_ids(reference, 0)
    with pytest.raises(ValueError):
        spearman(reference, reference[:-1])


def test_cascade_metrics() -> None:
    ratios = cascade_ratios(100, 40, 10)
    assert isinstance(ratios, CascadeRatios)
    assert ratios.rho_lj == pytest.approx(0.4)
    assert ratios.rho_mlp == pytest.approx(0.25)
    assert ratios.rho == pytest.approx(0.1)
    assert cascade_ratios(0, 0, 0) == CascadeRatios(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        cascade_ratios(10, 11, 5)
    with pytest.raises(ValueError):
        cascade_ratios(10, 5, 6)

    assert false_rejection([1, 2, 3], [2, 5]) == pytest.approx(1 / 3)
    assert false_rejection([], [2]) == 0.0
    assert max_false_rejection(100, 40, 5) == pytest.approx(1.0)
    assert max_false_rejection(100, 99, 2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        max_false_rejection(10, 5, 0)


def test_geometry_metrics() -> None:
    assert admissible_fraction(3, 10) == pytest.approx(0.3)
    assert accessible_solid_angle(10, 10) == pytest.approx(4.0 * np.pi)
    with pytest.raises(ValueError):
        admissible_fraction(11, 10)

    slab = fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)
    site = build_site(slab, [0])
    reference = high_resolution_fraction(site, n=2048)
    assert 0.0 <= reference <= 1.0
    assert discretization_residual(site, 2048, n_ref=2048) == 0.0
    assert discretization_residual(site, 64, n_ref=2048) >= 0.0
