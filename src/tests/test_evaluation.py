# ruff: noqa D103 E501
"""Tests of the evaluation layer."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLengths

from adsorption.evaluation import (
    EvalResult,
    Relaxation,
    Stage1Constraints,
    build_calculator,
    stage1_constraints,
)


@pytest.fixture(scope="module")
def candidate() -> Atoms:
    slab = fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)
    adsorbate = molecule("CO")
    adsorbate.positions += np.array([0.0, 0.0, 2.2])
    structure = slab.copy()
    structure.extend(adsorbate)
    return structure


def test_build_calculator() -> None:
    calculator = build_calculator("ase.calculators.emt.EMT")
    assert isinstance(calculator, EMT)
    with pytest.raises(ValueError):
        build_calculator("EMT")
    with pytest.raises(ImportError):
        build_calculator("ase.calculators.missing.Missing")
    with pytest.raises(TypeError):
        build_calculator("ase.build.molecule")


def test_stage1_constraints(candidate: Atoms) -> None:
    n_substrate = len(candidate) - 2
    constraints = stage1_constraints(
        candidate, n_substrate, Stage1Constraints()
    )
    assert len(constraints) == 2
    fixed, bond_lengths = constraints
    assert isinstance(fixed, FixAtoms)
    assert isinstance(bond_lengths, FixBondLengths)
    assert list(fixed.get_indices()) == list(range(n_substrate))
    assert bond_lengths.pairs.shape == (1, 2)
    assert list(bond_lengths.pairs[0]) == [n_substrate, n_substrate + 1]

    none = stage1_constraints(
        candidate,
        n_substrate,
        Stage1Constraints(False, False),
    )
    assert none == []


def test_stage1_constraint_validation(candidate: Atoms) -> None:
    with pytest.raises(ValueError):
        stage1_constraints(candidate, 0, Stage1Constraints())
    with pytest.raises(ValueError):
        stage1_constraints(candidate, len(candidate), Stage1Constraints())


def test_single_point(candidate: Atoms) -> None:
    relaxation = Relaxation(EMT(), mode="single_point", with_forces=True)
    result = relaxation(candidate, n_substrate=len(candidate) - 2)
    assert isinstance(result, EvalResult)
    assert np.isfinite(result.energy)
    assert result.single_point
    assert result.n_stages == 0
    assert result.converged is None
    assert result.forces is not None
    assert result.forces.shape == (len(candidate), 3)
    assert len(result.atoms) == len(candidate)


def test_two_stage_relaxation(candidate: Atoms) -> None:
    relaxation = Relaxation(
        EMT(),
        mode="two_stage",
        max_steps_stage1=3,
        max_steps_stage2=3,
        max_force=0.5,
    )
    result = relaxation(candidate, n_substrate=len(candidate) - 2)
    assert np.isfinite(result.energy)
    assert not result.single_point
    assert result.n_stages in {0, 1, 2}
    assert isinstance(result.converged, bool)
    assert len(result.atoms) == len(candidate)
    assert result.trajectory == ()

    with_trajectory = Relaxation(
        EMT(),
        max_steps_stage1=2,
        max_steps_stage2=2,
        max_force=0.5,
        keep_trajectory=True,
    )
    traced = with_trajectory(candidate, n_substrate=len(candidate) - 2)
    assert len(traced.trajectory) > 0


def test_non_convergence_is_retained(candidate: Atoms) -> None:
    relaxation = Relaxation(
        EMT(),
        max_steps_stage1=0,
        max_steps_stage2=0,
        max_force=1.0e-8,
    )
    result = relaxation(candidate, n_substrate=len(candidate) - 2)
    assert result.converged is False
    assert result.n_stages == 0
    assert len(result.atoms) == len(candidate)
    assert np.isfinite(result.energy)


def test_relaxation_validation(candidate: Atoms) -> None:
    with pytest.raises(ValueError):
        Relaxation(EMT(), mode="unknown")
    with pytest.raises(ValueError):
        Relaxation(EMT(), max_steps_stage1=-1)
    with pytest.raises(ValueError):
        Relaxation(EMT(), max_force=0.0)
    relaxation = Relaxation(EMT(), mode="single_point")
    with pytest.raises(ValueError):
        relaxation(candidate, n_substrate=len(candidate))
    with pytest.raises(ValueError):
        relaxation(candidate, n_substrate=0)


def test_adsorbate_free_stage1_is_unconstrained(candidate: Atoms) -> None:
    n_substrate = len(candidate) - 2
    constraints = stage1_constraints(
        candidate,
        n_substrate,
        Stage1Constraints(substrate_fixed=False),
    )
    assert len(constraints) == 1
    assert isinstance(constraints[0], FixBondLengths)
