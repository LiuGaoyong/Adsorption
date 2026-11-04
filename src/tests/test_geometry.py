# ruff: noqa D103 E501
"""Tests of the calculator-free geometric layer."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.cluster import Octahedron
from ase.data import covalent_radii as COV_R
from ase.geometry import find_mic

from adsorption.geometry import (
    DELTA_NBR,
    DirectionRollSet,
    FibonacciGrid,
    LatLonGrid,
    SO3Set,
    SeededRandomGrid,
    build_site,
    default_scale,
    default_skin,
    displacements_mic,
    distance_array,
    first_neighbor_shell,
    mean_mic,
    apply_quaternion,
    compose_quaternions,
    place,
    resolve_anchor,
    shortest_arc_quaternion,
    unit_vector,
)
from adsorption.geometry._quat import quaternion_invert
from adsorption.utils.conformers import (
    generate_conformers,
    near_congruent,
    prune_by_energy,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (4, 4, 3), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def cluster() -> Atoms:
    return Octahedron("Cu", 4)


def top_surface_index(atoms: Atoms) -> int:
    """Return a top-layer atom near the centre of the cell."""
    z = atoms.positions[:, 2]
    layer = np.flatnonzero(np.isclose(z, z.max()))
    centre = atoms.cell.array.sum(axis=0)[:2] / 2.0
    distance = np.linalg.norm(atoms.positions[layer, :2] - centre, axis=1)
    return int(layer[int(np.argmin(distance))])


def inplane_neighbours(atoms: Atoms, index: int) -> list[int]:
    """Return same-layer neighbours of a surface atom, nearest first."""
    z = atoms.positions[index, 2]
    shell = first_neighbor_shell(atoms, [index])
    same = [int(j) for j in shell if abs(atoms.positions[j, 2] - z) < 0.1]
    same.sort(key=lambda j: atoms.get_distance(index, j, mic=True))
    return same


def cluster_top_index(atoms: Atoms) -> int:
    return int(np.argmax(atoms.positions[:, 2]))


def test_fibonacci_grid_is_deterministic_and_unit() -> None:
    grid = FibonacciGrid()
    n = 64
    first = grid.points(n)
    second = grid.points(n)
    assert grid.size(n) == n
    assert first.shape == (n, 3)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    # Eq. (5): z_i = 1 - (2i + 1) / N
    expected_z = 1.0 - (2.0 * np.arange(n) + 1.0) / n
    assert np.allclose(first[:, 2], expected_z)


def test_latlon_and_random_grids() -> None:
    latlon = LatLonGrid()
    points = latlon.points(200)
    assert points.shape == (latlon.size(200), 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    assert np.array_equal(points, latlon.points(200))

    random_a = SeededRandomGrid(seed=7).points(50)
    random_b = SeededRandomGrid(seed=7).points(50)
    random_c = SeededRandomGrid(seed=8).points(50)
    assert np.array_equal(random_a, random_b)
    assert not np.array_equal(random_a, random_c)
    assert np.allclose(np.linalg.norm(random_a, axis=1), 1.0)


def test_distance_array() -> None:
    distances = distance_array(0.8, 3.0, 0.2)
    assert len(distances) == 12
    assert distances[0] == pytest.approx(0.8)
    assert distances[-1] == pytest.approx(3.0)
    assert len(distance_array(0.0, 1.0, 0.1)) == 11
    assert len(distance_array(0.0, 1.0, 0.3)) == 4
    with pytest.raises(ValueError):
        distance_array(1.0, 0.5, 0.1)
    with pytest.raises(ValueError):
        distance_array(0.0, 1.0, 0.0)


def test_shortest_arc_quaternion() -> None:
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([1.0, 0.0, 0.0])
    quaternion = shortest_arc_quaternion(a, b)
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    assert np.allclose(apply_quaternion(quaternion, a[None, :])[0], b)

    identity = shortest_arc_quaternion(a, a)
    assert np.allclose(apply_quaternion(identity, np.eye(3)), np.eye(3))

    antiparallel = shortest_arc_quaternion(a, -a)
    assert np.allclose(apply_quaternion(antiparallel, a[None, :])[0], -a)

    rng = np.random.default_rng(0)
    for _ in range(5):
        u = unit_vector(rng.normal(size=3))
        v = unit_vector(rng.normal(size=3))
        q = shortest_arc_quaternion(u, v)
        assert np.allclose(apply_quaternion(q, u[None, :])[0], v, atol=1e-10)
        back = apply_quaternion(q, v[None, :], inverse=True)[0]
        assert np.allclose(back, u, atol=1e-10)


def test_anchor_resolution() -> None:
    adsorbate = Atoms("H2O")
    com = resolve_anchor(adsorbate, "com")
    assert com.kind == "com"
    assert np.allclose(com.position, adsorbate.get_center_of_mass())

    atom = resolve_anchor(adsorbate, 1)
    assert atom.kind == "atom" and atom.index == 1
    assert np.allclose(atom.position, adsorbate.positions[1])

    point = resolve_anchor(adsorbate, [1.0, 2.0, 3.0])
    assert point.kind == "point"
    assert np.allclose(point.position, [1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        resolve_anchor(adsorbate, 7)
    with pytest.raises(ValueError):
        resolve_anchor(adsorbate, "binding_atom")
    with pytest.raises(ValueError):
        resolve_anchor(adsorbate, [1.0, 2.0])


def test_place_is_rigid_and_does_not_mutate() -> None:
    reference = Atoms("CO")
    anchor = resolve_anchor(reference, "com")
    before = reference.positions.copy()
    quaternion = shortest_arc_quaternion([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    target = np.array([1.0, 2.0, 3.0])
    placed = place(
        reference,
        anchor=anchor,
        target=target,
        quaternion=quaternion,
    )

    assert np.array_equal(reference.positions, before)
    bond = (before[0] - before[1])[None, :]
    assert np.allclose(
        placed.positions[0] - placed.positions[1],
        apply_quaternion(quaternion, bond)[0],
    )
    # the anchor lands on the target
    relative = before - anchor.position
    assert np.allclose(
        target + apply_quaternion(quaternion, relative),
        placed.positions,
    )


def test_first_neighbor_shell_matches_brute_force(slab: Atoms) -> None:
    core = np.array([top_surface_index(slab)])
    shell = first_neighbor_shell(slab, core)
    indices = np.arange(len(slab))
    distances = slab.get_distances(
        np.repeat(core, len(indices)),
        np.tile(indices, len(core)),
        mic=True,
    )
    cutoff = (
        COV_R[slab.numbers[np.repeat(core, len(indices))]]
        + COV_R[slab.numbers[np.tile(indices, len(core))]]
        + DELTA_NBR
    )
    expected = np.setdiff1d(np.unique(np.tile(indices, len(core))[distances < cutoff]), core)
    assert np.array_equal(shell, expected)
    assert np.allclose(slab.get_distances(core[0], shell, mic=True).min(), 2.55, atol=0.1)


def test_site_geometry_defaults(slab: Atoms, cluster: Atoms) -> None:
    top = top_surface_index(slab)
    single = build_site(slab, [top])
    assert single.core_size == 1
    assert single.scale == pytest.approx(default_scale(1))
    assert single.skin == pytest.approx(default_skin(1))
    assert single.radius == pytest.approx(COV_R[29] * 1.2)
    assert np.allclose(single.center, slab.positions[top])
    # a single-core site excludes the core atom from the filter set
    assert len(single.filter_positions) == len(single.neighbor_positions)

    two = inplane_neighbours(slab, top)[:2]
    multi = build_site(slab, two)
    assert multi.core_size == 2
    assert multi.scale == pytest.approx(default_scale(2))
    assert multi.skin == pytest.approx(default_skin(2))
    # a multi-core site keeps the core atoms in the filter set
    assert len(multi.filter_positions) == len(multi.neighbor_positions) + 2
    assert np.allclose(multi.center, mean_mic(slab.positions[two], slab.cell.array, slab.pbc))

    cluster_index = cluster_top_index(cluster)
    cluster_site = build_site(cluster, [cluster_index])
    assert cluster_site.cell is None
    assert np.allclose(cluster_site.center, cluster.positions[cluster_index])


def test_minimum_image_matches_ase(slab: Atoms) -> None:
    rng = np.random.default_rng(3)
    points_a = rng.uniform(-2, 6, size=(6, 3))
    points_b = rng.uniform(-2, 6, size=(4, 3))
    diff = points_b[None, :, :] - points_a[:, None, :]
    expected = find_mic(diff.reshape(-1, 3), slab.cell, slab.pbc)[0]
    assert np.allclose(
        displacements_mic(diff, slab.cell.array, slab.pbc),
        expected.reshape(diff.shape),
    )


def test_direction_roll_set() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    orientations = DirectionRollSet(axis, 8, n_roll=4)
    assert orientations.count() == 32
    for index in (0, 1, 5, 31):
        quaternion = orientations.quaternion(index)
        assert np.isclose(np.linalg.norm(quaternion), 1.0)
        image = apply_quaternion(quaternion, axis[None, :])[0]
        assert np.allclose(image, orientations.image(index))
    # the first direction carries four rolls, one per roll index
    first = orientations.quaternion(0)
    second = orientations.quaternion(1)
    relative = compose_quaternions(second, quaternion_invert(first))
    angle = 2.0 * np.arccos(np.clip(abs(relative[0]), -1.0, 1.0))
    assert np.isclose(angle, 2.0 * np.pi / 4)
    image = orientations.image(0)
    rolled = apply_quaternion(relative, image[None, :])[0]
    assert np.allclose(rolled, image)
    # the roll leaves the placed axis invariant
    assert np.allclose(orientations.image(7), orientations.image(4))


def test_so3_set_is_uniform_and_deterministic() -> None:
    orientations = SO3Set(1024)
    assert orientations.count() == 1024
    assert np.array_equal(orientations.quaternions, SO3Set(1024).quaternions)
    for index in (0, 1, 500, 1023):
        quaternion = orientations.quaternion(index)
        assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-10)
    axis = np.array([0.0, 0.0, 1.0])
    images = np.array(
        [
            apply_quaternion(orientations.quaternion(i), axis[None, :])[0]
            for i in range(1024)
        ]
    )
    assert np.linalg.norm(images.mean(axis=0)) < 0.02
    covariance = np.cov(images.T)
    assert np.allclose(np.diag(covariance), 1.0 / 3.0, atol=0.02)
    assert np.allclose(covariance - np.diag(np.diag(covariance)), 0.0, atol=0.01)


def test_conformer_generation_is_deterministic() -> None:
    first = generate_conformers("CCCC", n_conformers=3, seed=7)
    second = generate_conformers("CCCC", n_conformers=3, seed=7)
    assert 1 <= len(first) <= 3
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert left.get_chemical_formula() == "C4H10"
        assert np.allclose(left.positions, right.positions, atol=1e-6)


def test_conformer_pruning_and_congruence() -> None:
    water = molecule("H2O")
    rotated = water.copy()
    rotated.rotate(37.0, "z")
    rotated.positions += np.array([1.0, 2.0, 3.0])
    assert near_congruent(water, rotated, tolerance=1e-6)
    methane = molecule("CH4")
    with pytest.raises(ValueError):
        near_congruent(water, methane)

    references = [water, rotated, molecule("H2O")]
    kept = prune_by_energy(references, [0.0, 0.05, 0.2], window=0.1)
    assert len(kept) == 2
    assert kept[0] is water and kept[1] is rotated
    with pytest.raises(ValueError):
        prune_by_energy(references, [0.0], window=0.1)
