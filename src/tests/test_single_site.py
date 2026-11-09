# ruff: noqa D103 E501
"""Tests of the single-site process-A generator."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.generate import AdsorptionConfigurationGenerator
from adsorption.geometry import FixedRotation, SuperFibonacciSO3


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def generator(slab: Atoms) -> AdsorptionConfigurationGenerator:
    return AdsorptionConfigurationGenerator(
        slab,
        0,
        molecule("CO"),
        n_oa=6,
        orientation=SuperFibonacciSO3(4),
        n_d=3,
        d_min=1.5,
        d_max=2.5,
    )


def test_budget(generator: AdsorptionConfigurationGenerator) -> None:
    assert generator.n_oa == 6
    assert generator.n_rot == 4
    assert generator.n_d == 3
    assert generator.n_candidates == 72
    assert len(generator) == 72


def test_site_is_derived_from_the_core_index(
    generator: AdsorptionConfigurationGenerator,
    slab: Atoms,
) -> None:
    """The core index selects the core set and the site centre."""
    assert list(generator.core) == [0]
    assert np.allclose(generator.site.center, slab.positions[0])


def test_multi_core_index_is_accepted(slab: Atoms) -> None:
    """A pair of indices builds a bridge site."""
    generator = AdsorptionConfigurationGenerator(
        slab,
        [0, 1],
        molecule("CO"),
        n_oa=3,
        n_rot=1,
        n_d=1,
    )
    assert list(generator.core) == [0, 1]
    assert generator.site.core_size == 2
    assert generator.n_candidates == 3


def test_distance_array_is_built_from_the_bounds() -> None:
    """``n_d`` and the bounds define a uniform distance array."""
    generator = AdsorptionConfigurationGenerator(
        fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True),
        0,
        molecule("CO"),
        n_oa=2,
        n_rot=1,
        n_d=5,
        d_min=1.0,
        d_max=3.0,
    )
    assert generator.n_d == 5
    assert generator.distances == pytest.approx([1.0, 1.5, 2.0, 2.5, 3.0])


def test_explicit_distances_override_the_bounds(slab: Atoms) -> None:
    """An explicit array wins over ``n_d`` and the bounds."""
    generator = AdsorptionConfigurationGenerator(
        slab,
        0,
        molecule("CO"),
        n_oa=2,
        n_rot=1,
        distances=[2.0, 2.5],
    )
    assert generator.n_d == 2
    assert generator.distances == pytest.approx([2.0, 2.5])


def test_distance_bounds_follow_the_adsorbate_radius(slab: Atoms) -> None:
    """The radius coefficients shift both bounds by the molecular radius."""
    adsorbate = molecule("C6H6")
    center = adsorbate.get_center_of_mass()
    radius = float(np.linalg.norm(adsorbate.positions - center, axis=1).max())
    generator = AdsorptionConfigurationGenerator(
        slab,
        0,
        adsorbate,
        n_oa=1,
        n_rot=1,
        n_d=2,
        d_min=2.0,
        d_max=3.5,
        d_min_radius_scale=0.1,
        d_max_radius_scale=0.35,
    )
    assert generator.distances == pytest.approx(
        [2.0 + 0.1 * radius, 3.5 + 0.35 * radius]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_d": 0},
        {"n_oa": 0},
        {"n_rot": 0},
        {"d_min": 3.0, "d_max": 1.0},
        {"d_min": -1.0},
    ],
)
def test_invalid_parameters_are_rejected(
    slab: Atoms,
    kwargs: dict,
) -> None:
    """Invalid named parameters raise a ``ValueError``."""
    with pytest.raises(ValueError):
        AdsorptionConfigurationGenerator(slab, 0, molecule("CO"), **kwargs)


def test_index_order(generator: AdsorptionConfigurationGenerator) -> None:
    first = generator.trial(1)
    assert first.meta["i"] == 1
    assert first.meta["j"] == 1
    assert first.meta["k"] == 1
    last = generator.trial(72)
    assert last.meta["i"] == 6
    assert last.meta["j"] == 4
    assert last.meta["k"] == 3
    assert last.meta["distance"] == pytest.approx(2.5)
    with pytest.raises(IndexError):
        generator.trial(0)
    with pytest.raises(IndexError):
        generator.trial(73)


def test_trial_structure(
    generator: AdsorptionConfigurationGenerator, slab: Atoms
) -> None:
    trial = generator.trial(5)
    assert len(trial.atoms) == len(slab) + 2
    assert trial.index == 5
    assert trial.n_total == 72
    assert trial.meta["n_substrate"] == len(slab)
    assert trial.atoms.get_chemical_formula() == "CCu8O"


def test_generate_interval(generator: AdsorptionConfigurationGenerator) -> None:
    chunk = list(generator.generate(10, 12))
    assert [trial.index for trial in chunk] == [10, 11, 12]
    assert len(list(generator.generate())) == 72
    with pytest.raises(IndexError):
        list(generator.generate(0, 1))
    with pytest.raises(IndexError):
        list(generator.generate(1, 73))
    with pytest.raises(ValueError):
        list(generator.generate(5, 4))


def test_signature_is_stable(
    generator: AdsorptionConfigurationGenerator,
) -> None:
    left = generator.signature()
    right = generator.signature()
    assert left == right
    assert left["kind"] == "adsorption"
    assert left["factors"]["n_sites"] == 1
    assert left["sampling"]["n_oa"] == 6
    assert left["sampling"]["n_d"] == 3
    assert left["sites"]["digest"].startswith("sha1:")


def test_fixed_rotation_has_one_orientation(slab: Atoms) -> None:
    generator = AdsorptionConfigurationGenerator(
        slab,
        0,
        molecule("CO"),
        n_oa=4,
        orientation=FixedRotation(),
        n_d=1,
    )
    assert generator.n_rot == 1
    assert generator.n_candidates == 4


def test_orientation_kind_selects_the_rotation_set(slab: Atoms) -> None:
    """``orientation_kind`` builds the rotation set when none is given."""
    generator = AdsorptionConfigurationGenerator(
        slab,
        0,
        molecule("CO"),
        n_oa=2,
        n_rot=8,
        orientation_kind="so3",
        n_d=1,
    )
    assert generator.orientation.count() == 8
    assert generator.n_rot == 8
