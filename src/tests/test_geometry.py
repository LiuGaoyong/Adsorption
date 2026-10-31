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
    AdmissibleSet,
    DirectionRollSet,
    FibonacciGrid,
    LatLonGrid,
    SO3Set,
    SeededRandomGrid,
    admissible_directions,
    build_site,
    default_scale,
    default_skin,
    displacements_mic,
    distance_array,
    distances_mic,
    first_neighbor_shell,
    generate_conformers,
    mean_mic,
    near_congruent,
    outward_normal,
    place,
    prune_by_energy,
    resolve_anchor,
    shortest_arc_rotation,
    unit_vector,
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


def test_shortest_arc_rotation() -> None:
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([1.0, 0.0, 0.0])
    rotation = shortest_arc_rotation(a, b)
    assert np.allclose(rotation @ a, b)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(rotation @ rotation.T, np.eye(3))

    identity = shortest_arc_rotation(a, a)
    assert np.allclose(identity, np.eye(3))

    antiparallel = shortest_arc_rotation(a, -a)
    assert np.allclose(antiparallel @ a, -a)
    assert np.isclose(np.linalg.det(antiparallel), 1.0)

    rng = np.random.default_rng(0)
    for _ in range(5):
        u = unit_vector(rng.normal(size=3))
        v = unit_vector(rng.normal(size=3))
        matrix = shortest_arc_rotation(u, v)
        assert np.allclose(matrix @ u, v, atol=1e-10)
        assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10)


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
    rotation = shortest_arc_rotation([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    target = np.array([1.0, 2.0, 3.0])
    placed = place(reference, anchor=anchor, target=target, rotation=rotation)

    assert np.array_equal(reference.positions, before)
    assert np.allclose(placed.positions[0] - placed.positions[1], rotation @ (before[0] - before[1]))
    # the anchor lands on the target
    relative = before - anchor.position
    assert np.allclose(target + relative @ rotation.T, placed.positions)


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


def test_outward_normal(slab: Atoms) -> None:
    top = top_surface_index(slab)
    site = build_site(slab, [top])
    normal = site.outward_normal
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.allclose(outward_normal(site.center, site.neighbor_positions, site.cell, site.pbc), normal)
    # the reference normal of a (111) top site points away from the surface
    assert normal[2] > 0.0


def test_site_translation_covariance(slab: Atoms) -> None:
    top = top_surface_index(slab)
    shifted = slab.copy()
    shifted.positions = slab.positions + slab.cell.array[0]
    site_a = build_site(slab, [top])
    site_b = build_site(shifted, [top])
    shift = slab.cell.array[0]
    assert np.allclose(site_b.center, site_a.center + shift)
    directions_a = admissible_directions(site_a, 128)
    directions_b = admissible_directions(site_b, 128)
    assert directions_a.n_oa == directions_b.n_oa
    assert np.allclose(directions_a.unit, directions_b.unit)


def test_admissible_directions(slab: Atoms) -> None:
    top = top_surface_index(slab)
    site = build_site(slab, [top])
    admissible = admissible_directions(site, 128)
    assert isinstance(admissible, AdmissibleSet)
    assert admissible.n_grid == 128
    assert 0 <= admissible.n_oa <= 128
    assert np.all(np.diff(admissible.grid_index) > 0)
    assert admissible.fraction == pytest.approx(admissible.n_oa / 128)

    # every survivor clears the filter set at r_s
    radii = site.filter_radii + site.skin
    distances = distances_mic(
        site.center + site.radius * admissible.unit,
        site.filter_positions,
        site.cell,
        site.pbc,
    )
    assert np.all(distances > radii[None, :])

    # deterministic
    again = admissible_directions(site, 128)
    assert np.array_equal(admissible.unit, again.unit)
    assert np.array_equal(admissible.grid_index, again.grid_index)


def test_admissible_filter_parameters(slab: Atoms) -> None:
    top = top_surface_index(slab)
    site = build_site(slab, [top])
    loose = admissible_directions(site, 128, skin=0.2)
    tight = admissible_directions(site, 128, skin=1.5)
    assert tight.n_oa <= loose.n_oa
    with pytest.raises(ValueError):
        admissible_directions(site, 128, skin=-1.0)
    with pytest.raises(ValueError):
        admissible_directions(site, 128, scale=0.0)


def test_minimum_image_matches_ase(slab: Atoms) -> None:
    rng = np.random.default_rng(3)
    points_a = rng.uniform(-2, 6, size=(6, 3))
    points_b = rng.uniform(-2, 6, size=(4, 3))
    got = distances_mic(points_a, points_b, slab.cell.array, slab.pbc)
    diff = points_b[None, :, :] - points_a[:, None, :]
    expected = np.linalg.norm(find_mic(diff.reshape(-1, 3), slab.cell, slab.pbc)[0], axis=1).reshape(6, 4)
    assert np.allclose(got, expected)
    assert np.allclose(displacements_mic(diff, slab.cell.array, slab.pbc), find_mic(diff.reshape(-1, 3), slab.cell, slab.pbc)[0].reshape(diff.shape))


def test_geometry_uses_no_random_numbers(monkeypatch: pytest.MonkeyPatch, slab: Atoms) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("the geometric path must not draw random numbers")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "randint", forbidden)
    monkeypatch.setattr(np.random, "rand", forbidden)
    top = top_surface_index(slab)
    site = build_site(slab, [top])
    admissible = admissible_directions(site, 64)
    assert admissible.n_oa > 0


def test_proposition_2_fixed_radius_proxy(slab: Atoms) -> None:
    """A direction admitted at r_s can still put the anchor inside an atom."""
    top = top_surface_index(slab)
    neighbours = inplane_neighbours(slab, top)
    site = build_site(slab, [top, neighbours[0]])
    admissible = admissible_directions(site, 256)
    anchor_distance = 0.8
    anchors = site.center + anchor_distance * admissible.unit
    radii = site.filter_radii + site.skin
    distances = distances_mic(anchors, site.filter_positions, site.cell, site.pbc)
    overlapping = np.any(distances < radii[None, :], axis=1)
    assert np.any(overlapping)


def test_direct_adsorption_deterministic(slab: Atoms) -> None:
    from adsorption import DirectAdsorption

    top = top_surface_index(slab)
    obj = DirectAdsorption(atoms=slab, core=[top], nfibonacci=64, calculator=None)
    adsorbate = molecule("CO")
    reference = adsorbate.positions.copy()
    first, label = obj(adsorbate=adsorbate)
    second, _ = obj(adsorbate=molecule("CO"))
    assert label == -1
    assert len(first) == len(slab) + 2
    assert np.allclose(first.positions, second.positions)
    # the input adsorbate is not mutated
    assert np.allclose(adsorbate.positions, reference)
    assert obj._distance > 0.0


def test_direct_adsorption_anchor(slab: Atoms) -> None:
    from adsorption import DirectAdsorption

    top = top_surface_index(slab)
    obj = DirectAdsorption(atoms=slab, core=[top], nfibonacci=64, calculator=None, anchor=0)
    result, _ = obj(adsorbate="CO", distance=2.0)
    target = obj._anchor_core + obj._distance * obj._direction_core
    # atom 0 of CO is the carbon, i.e. the anchor
    assert np.allclose(result.positions[len(slab)], target)


def test_direct_seeded_indices_are_reproducible(slab: Atoms) -> None:
    from adsorption import DirectAdsorption

    top = top_surface_index(slab)
    first = DirectAdsorption(atoms=slab, core=[top], nfibonacci=64, seed=11)
    second = DirectAdsorption(atoms=slab, core=[top], nfibonacci=64, seed=11)
    for _ in range(3):
        left, _ = first(adsorbate="CO")
        right, _ = second(adsorbate="CO")
        assert np.allclose(left.positions, right.positions)


def test_raw_adsorption_uses_outward_normal(slab: Atoms) -> None:
    from adsorption import RawAdsorption

    top = top_surface_index(slab)
    obj = RawAdsorption(atoms=slab, core=[top], calculator=None)
    result, _ = obj(adsorbate="O")
    position = result.positions[len(slab)]
    direction = position - obj.site_geometry.center
    direction = direction / np.linalg.norm(direction)
    assert np.allclose(direction, obj.site_geometry.outward_normal)


def test_direction_roll_set() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    orientations = DirectionRollSet(axis, 8, n_roll=4)
    assert orientations.count() == 32
    for index in (0, 1, 5, 31):
        rotation = orientations.rotation(index)
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.isclose(np.linalg.det(rotation), 1.0)
    # the first direction carries four rolls, one per roll index
    first = orientations.rotation(0)
    second = orientations.rotation(1)
    relative = second @ first.T
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    assert np.isclose(angle, 2.0 * np.pi / 4)
    image = orientations.image(0)
    assert np.allclose(relative @ image, image)
    # the roll leaves the placed axis invariant
    assert np.allclose(orientations.image(7), orientations.image(4))


def test_so3_set_is_uniform_and_deterministic() -> None:
    orientations = SO3Set(1024)
    assert orientations.count() == 1024
    assert np.array_equal(orientations.quaternions, SO3Set(1024).quaternions)
    for index in (0, 1, 500, 1023):
        rotation = orientations.rotation(index)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)
    axis = np.array([0.0, 0.0, 1.0])
    images = np.array([orientations.rotation(i) @ axis for i in range(1024)])
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
