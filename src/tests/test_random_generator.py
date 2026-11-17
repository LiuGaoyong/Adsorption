"""Contract tests of the independent random placement generator."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.io import read

from adsorption import (
    AdsorptionConfigurationGenerator,
    RandomConfigurationGenerator,
)
from adsorption.geometry import ExplicitSites
from adsorption.runner import run_config

GRID = {"n_oa": 12, "n_rot": 5, "n_d": 3, "hemisphere": True}


def slab() -> Atoms:
    """Return a small Cu(111) slab."""
    return fcc111("Cu", (3, 4, 3), vacuum=8.0, orthogonal=True, periodic=True)


def pair(adsorbate: Atoms, seed: int = 1, **grid: int):
    """Return the product and the random generator of one protocol."""
    options = {**GRID, **grid}
    sites = ExplicitSites([[30], [30, 31], [30, 31, 33]])
    product = AdsorptionConfigurationGenerator(
        slab(), sites, adsorbate, **options
    )
    random = RandomConfigurationGenerator(
        slab(), sites, adsorbate, seed=seed, **options
    )
    return product, random


def test_budget_matches_the_product_component_by_component() -> None:
    """Both generators spend the same budget on every site."""
    product, random = pair(molecule("H2O"))
    assert random.n_candidates == product.n_candidates
    assert random.offsets == product.offsets
    assert random.factors() == product.factors()
    assert random.signature()["placement"]["kind"] == "random"
    assert random.signature()["placement"]["seed"] == 1


def test_trials_are_a_pure_function_of_seed_and_index() -> None:
    """Random access, streaming and a second instance agree; seeds differ."""
    _, random = pair(molecule("H2O"))
    _, again = pair(molecule("H2O"))
    _, other = pair(molecule("H2O"), seed=2)
    stream = list(random.generate())
    assert len(stream) == random.n_candidates
    for kappa in (1, 7, random.n_candidates):
        direct = random.trial(kappa)
        assert np.array_equal(
            direct.atoms.positions, stream[kappa - 1].atoms.positions
        )
        assert np.array_equal(
            direct.atoms.positions, again.trial(kappa).atoms.positions
        )
        assert not np.allclose(
            direct.atoms.positions, other.trial(kappa).atoms.positions
        )
    middle = list(random.generate(20, 40))
    assert [trial.index for trial in middle] == list(range(20, 41))
    assert np.array_equal(middle[0].atoms.positions, stream[19].atoms.positions)


def test_draws_respect_the_hemisphere_and_the_distance_bounds() -> None:
    """Directions point outward and distances stay inside the array."""
    product, random = pair(molecule("H2O"))
    for position, component in enumerate(random._components):
        directions, quaternions, distances = random._component_draws(position)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
        assert np.all(directions @ component.site.normal >= 0.0)
        assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0)
        assert np.all(quaternions[:, 0] >= 0.0)
        array = product._components[position].adsorbate.distances
        assert distances.min() >= array.min()
        assert distances.max() <= array.max()
        assert np.ptp(distances) > 0.5 * np.ptp(array)


def test_full_sphere_without_the_hemisphere() -> None:
    """Without the restriction the directions cover both hemispheres."""
    _, random = pair(molecule("H2O"), hemisphere=False)
    directions, _, _ = random._component_draws(0)
    normal = random._components[0].site.normal
    inward = int((directions @ normal < 0.0).sum())
    assert 0 < inward < len(directions)


def test_a_smaller_budget_is_a_prefix_of_a_larger_one() -> None:
    """The first draws of a component do not depend on how many follow."""
    _, small = pair(molecule("H2O"), n_oa=8, n_rot=2, n_d=2)
    _, large = pair(molecule("H2O"), n_oa=40, n_rot=9, n_d=3)
    few = small._component_draws(1)
    many = large._component_draws(1)
    count = len(few[0])
    assert count < len(many[0])
    assert np.array_equal(few[0], many[0][:count])
    assert np.array_equal(few[1], many[1][:count])
    # The distance bounds are the same, so the distances are a prefix too.
    assert np.allclose(few[2], many[2][:count])


def test_the_molecule_stays_rigid_and_an_atom_has_no_orientation() -> None:
    """A draw moves and turns the molecule; a single atom is not turned."""
    _, random = pair(molecule("H2O"))
    reference = molecule("H2O")
    trial = random.trial(5)
    placed = trial.atoms[trial.natoms_substrate :]
    assert np.allclose(
        placed.get_all_distances(), reference.get_all_distances(), atol=1e-9
    )
    assert trial.meta["sampling"] == "random"
    assert trial.meta["draw"] == 4
    _, atom = pair(Atoms("O"))
    _, quaternions, _ = atom._component_draws(0)
    assert np.allclose(quaternions, [1.0, 0.0, 0.0, 0.0])


def test_orientations_follow_the_haar_measure() -> None:
    """The rotation angle of a Haar-random rotation has mean 90 + 360/pi^2."""
    _, random = pair(molecule("H2O"), n_oa=200, n_rot=20, n_d=3)
    _, quaternions, _ = random._component_draws(0)
    angles = np.degrees(2.0 * np.arccos(np.clip(quaternions[:, 0], 0.0, 1.0)))
    expected = np.degrees(np.pi / 2.0 + 2.0 / np.pi)
    assert len(angles) > 5000
    assert angles.mean() == pytest.approx(expected, abs=2.0)


def test_runner_selects_the_generator_by_kind(tmp_path) -> None:
    """``protocol.kind`` switches the generator; the budget is unchanged."""
    config = {
        "substrate": {
            "_target_": "ase.build.fcc111",
            "symbol": "Cu",
            "size": [2, 2, 2],
            "vacuum": 8.0,
            "orthogonal": True,
            "periodic": True,
        },
        "adsorbate": {"smiles": "[C-]#[O+]"},
        "sites": [0],
        "protocol": {"n_oa": 8, "n_rot": 3, "n_d": 2, "hemisphere": True},
        "filters": [
            {"_target_": "adsorption.filters.GeometryFilter", "scale": 1.0}
        ],
        "storage": {"xyz": True},
        "parallel": {"n_jobs": 1},
    }
    product = run_config(config, output=tmp_path / "product")
    config["protocol"] = {**config["protocol"], "kind": "random", "seed": 4}
    random = run_config(config, output=tmp_path / "random")
    again = run_config(config, output=tmp_path / "again")
    assert random.total == product.total
    assert random.accepted == again.accepted > 0
    frames = read(str(tmp_path / "random" / "accepted.xyz"), index=":")
    repeat = read(str(tmp_path / "again" / "accepted.xyz"), index=":")
    assert all(atoms.info["sampling"] == "random" for atoms in frames)
    assert all(atoms.info["seed"] == 4 for atoms in frames)
    assert all("draw" in atoms.info for atoms in frames)
    assert np.allclose(frames[0].positions, repeat[0].positions)
    first = read(str(tmp_path / "product" / "accepted.xyz"), index=0)
    assert first.info["sampling"] == "product"
    config["protocol"]["kind"] = "sobol"
    with pytest.raises(ValueError):
        run_config(config, output=tmp_path / "bad")
