"""Contract tests of the energetic filters and their constraints."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLengths

from adsorption.base import Trial
from adsorption.filters import (
    LennardJonesEnergyFilter,
    OptimizedEnergyFilter,
)


def make_trial() -> Trial:
    """Return a trial of a Cu(111) slab with a CO adsorbate above it."""
    slab = fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)
    adsorbate = molecule("CO")
    adsorbate.positions += slab.positions.max(axis=0) + 2.0
    atoms = slab.copy()
    atoms.extend(adsorbate)
    return Trial(
        atoms,
        index=1,
        n_total=1,
        meta={"n_substrate": len(slab)},
    )


@pytest.mark.parametrize(
    ("fix_substrate", "fix_bonds", "substrate", "bonds"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (False, False, False, False),
        (False, True, True, True),
    ],
)
def test_fix_flags_are_normalised(
    fix_substrate: bool,
    fix_bonds: bool,
    substrate: bool,
    bonds: bool,
) -> None:
    """Fixing the adsorbate bonds implies fixing the substrate."""
    filt = OptimizedEnergyFilter(
        e_cut=0.0,
        calculator=EMT(),
        fix_substrate=fix_substrate,
        fix_adsorbate_bonds=fix_bonds,
    )
    assert filt.fix_substrate is substrate
    assert filt.fix_adsorbate_bonds is bonds


@pytest.mark.parametrize(
    "e_cut", [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_ceiling_is_rejected(e_cut: float) -> None:
    """A non-finite energy ceiling is a construction error."""
    with pytest.raises(ValueError):
        OptimizedEnergyFilter(e_cut=e_cut, calculator=EMT())


def test_evaluate_accepts_below_the_ceiling() -> None:
    """A relaxed trial under the ceiling is accepted and not mutated."""
    filt = OptimizedEnergyFilter(
        e_cut=1e6,
        calculator=EMT(),
        fmax=0.5,
        max_steps=3,
    )
    trial = make_trial()
    before = trial.atoms.positions.copy()
    verdict = filt.evaluate(trial)
    assert verdict.accepted is True
    assert verdict.reason == "ok"
    assert verdict.value is not None and np.isfinite(verdict.value)
    assert verdict.value <= filt.e_cut
    assert hasattr(verdict.detail, "converged")
    assert np.allclose(trial.atoms.positions, before)


def test_evaluate_rejects_above_the_ceiling() -> None:
    """A relaxed energy above the ceiling is rejected with its value."""
    filt = OptimizedEnergyFilter(
        e_cut=-1e9,
        calculator=EMT(),
        fmax=0.5,
        max_steps=3,
    )
    verdict = filt.evaluate(make_trial())
    assert verdict.accepted is False
    assert verdict.reason == "e_cut"
    assert verdict.value is not None
    assert verdict.value > filt.e_cut


@pytest.mark.parametrize(
    ("reject", "accepted", "reason"),
    [(True, False, "unconverged"), (False, True, "ok")],
)
def test_unconverged_policy(
    reject: bool,
    accepted: bool,
    reason: str,
) -> None:
    """The convergence policy decides the verdict at ``fmax = 0``."""
    filt = OptimizedEnergyFilter(
        e_cut=1e6,
        calculator=EMT(),
        fmax=0.0,
        max_steps=2,
        reject_on_unconverged=reject,
    )
    verdict = filt.evaluate(make_trial())
    assert verdict.accepted is accepted
    assert verdict.reason == reason


def test_adsorbate_bonds_use_the_adsorbate_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constraints cover the substrate and the adsorbate, nothing else."""
    trial = make_trial()
    captured: dict[str, object] = {}

    def fake_call_optimization(
        atoms: object,
        calc: object,
        *,
        method: str = "LBFGS",
        max_steps: int = 100,
        fmax: float = 0.01,
    ) -> SimpleNamespace:
        captured["atoms"] = atoms
        latest = atoms.copy()  # type: ignore[attr-defined]
        latest.calc = calc
        return SimpleNamespace(latest=latest, converged=True)

    monkeypatch.setattr(
        "adsorption.filters.relaxation.call_optimization",
        fake_call_optimization,
    )
    filt = OptimizedEnergyFilter(e_cut=1e6, calculator=EMT())
    verdict = filt.evaluate(trial)
    assert verdict.accepted is True

    atoms = captured["atoms"]
    fixed = [
        item
        for item in atoms.constraints  # type: ignore[attr-defined]
        if isinstance(item, FixAtoms)
    ]
    bonds = [
        item
        for item in atoms.constraints  # type: ignore[attr-defined]
        if isinstance(item, FixBondLengths)
    ]
    assert len(fixed) == 1
    assert len(bonds) == 1
    assert list(fixed[0].index) == list(range(trial.natoms_substrate))
    pairs = np.asarray(bonds[0].pairs)
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert pairs.min() >= trial.natoms_substrate
    assert pairs.max() < len(trial.atoms)
    local = {
        tuple(sorted(row))
        for row in (pairs - trial.natoms_substrate).tolist()
    }
    assert local == {(0, 1)}


def test_single_atom_adsorbate_has_no_bonds() -> None:
    """A single-atom adsorbate carries an empty intramolecular bond set."""
    slab = fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)
    oxygen = Atoms("O", positions=[slab.positions.max(axis=0) + 2.0])
    atoms = slab.copy()
    atoms.extend(oxygen)
    trial = Trial(
        atoms,
        index=1,
        n_total=1,
        meta={"n_substrate": len(slab)},
    )
    assert trial.natoms_adsorbate == 1
    filt = OptimizedEnergyFilter(
        e_cut=1e6,
        calculator=EMT(),
        fmax=0.5,
        max_steps=2,
        reject_on_unconverged=False,
    )
    verdict = filt.evaluate(trial)
    assert verdict.accepted is True
    assert verdict.value is not None and np.isfinite(verdict.value)


def test_signature_records_the_parameters() -> None:
    """The signature is a JSON-serialisable description of the filter."""
    filt = OptimizedEnergyFilter(
        e_cut=1.25,
        calculator=EMT(),
        fmax=0.2,
        max_steps=7,
        method="FIRE",
        fix_substrate=True,
        fix_adsorbate_bonds=False,
        reject_on_unconverged=False,
    )
    signature = filt.signature()
    assert signature["kind"] == "optimized_energy"
    assert signature["e_cut"] == 1.25
    assert signature["fmax"] == 0.2
    assert signature["max_steps"] == 7
    assert signature["method"] == "FIRE"
    assert signature["fix_substrate"] is True
    assert signature["fix_adsorbate_bonds"] is False
    assert signature["reject_on_unconverged"] is False
    assert isinstance(signature["calculator"], dict)


def test_lennard_jones_filter_instantiates_and_signs() -> None:
    """The Lennard-Jones tier constructs (no ``AttributeError``) and signs."""
    filt = LennardJonesEnergyFilter(e_cut=10.0)
    assert isinstance(filt, OptimizedEnergyFilter)
    assert filt.e_cut == 10.0
    assert filt.fix_substrate is True
    assert filt.fix_adsorbate_bonds is True
    signature = filt.signature()
    assert signature["kind"] == "optimized_energy"
    assert signature["e_cut"] == 10.0
    assert signature["reject_on_unconverged"] is True
    assert signature["calculator"] == (
        "Lennard-Jones Potential by OpenKIM Parameters"
    )
