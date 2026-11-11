"""Contract tests of the energetic filters and their constraints."""

import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLengths

from adsorption.base import Trial
from adsorption.filters import (
    EnergyFilter,
    LennardJonesEnergyFilter,
    NequIPOptimizedEnergyFilter,
    OptimizedEnergyFilter,
)
from adsorption.filters.nequip import DEFAULT_NEQUIP_MODEL


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


@pytest.mark.parametrize("e_cut", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_ceiling_is_rejected(e_cut: float) -> None:
    """A non-finite energy ceiling is a construction error."""
    with pytest.raises(ValueError):
        OptimizedEnergyFilter(e_cut=e_cut, calculator=EMT())


def test_none_ceiling_uses_isolated_fragment_energies() -> None:
    """``e_cut=None`` compares against the isolated fragment sum."""
    trial = make_trial()
    n_substrate = trial.natoms_substrate
    substrate = Atoms(
        numbers=trial.atoms.numbers[:n_substrate],
        positions=trial.atoms.positions[:n_substrate],
        cell=trial.atoms.cell,
        pbc=trial.atoms.pbc,
    )
    adsorbate = Atoms(
        numbers=trial.atoms.numbers[n_substrate:],
        positions=trial.atoms.positions[n_substrate:],
    )
    substrate.calc = EMT()
    e_substrate = substrate.get_potential_energy()
    adsorbate.calc = EMT()
    e_adsorbate = adsorbate.get_potential_energy()
    cutoff = e_substrate + e_adsorbate

    filt = OptimizedEnergyFilter(
        e_cut=None,
        calculator=EMT(),
        fmax=0.5,
        max_steps=3,
        reject_on_unconverged=False,
    )
    filt.prepare(trial)
    assert 0 in filt._adsorbate_cache
    assert filt._ceiling(trial) == pytest.approx(cutoff)
    verdict = filt.evaluate(trial)
    assert filt.e_cut is None
    assert filt.signature()["e_cut"] is None
    assert verdict.value is not None
    if verdict.reason == "e_cut":
        assert verdict.value > cutoff
    else:
        assert verdict.value <= cutoff


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
        tuple(sorted(row)) for row in (pairs - trial.natoms_substrate).tolist()
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


def test_nequip_filter_builds_a_calculator(monkeypatch) -> None:
    """The NequIP tier builds its calculator from the deployed model."""

    class FakeNequIPCalculator(EMT):
        @classmethod
        def from_deployed_model(cls, **kwargs):
            return cls()

    package = ModuleType("nequip")
    module = ModuleType("nequip.ase")
    module.NequIPCalculator = FakeNequIPCalculator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nequip", package)
    monkeypatch.setitem(sys.modules, "nequip.ase", module)

    filt = NequIPOptimizedEnergyFilter(
        "model.pth", device="cpu", e_cut=1e6, max_steps=1
    )
    assert filt.calculator is None
    signature = filt.signature()
    assert signature["kind"] == "nequip_optimized_energy"
    assert signature["calculator"] == {
        "model_path": "model.pth",
        "device": "cpu",
    }
    assert signature["fix_substrate"] is False
    assert signature["fix_adsorbate_bonds"] is False

    default = NequIPOptimizedEnergyFilter(e_cut=1e6)
    assert default.signature()["calculator"]["model_path"] == str(
        DEFAULT_NEQUIP_MODEL
    )

    restored = pickle.loads(pickle.dumps(filt))
    assert restored.signature() == signature
    assert restored.calculator is None
    restored.evaluate(make_trial())
    assert restored.calculator is not None


def test_energy_filter_does_not_move_the_trial() -> None:
    """The single-point tier only evaluates the energy, never relaxes."""
    filt = EnergyFilter(e_cut=1e6, calculator=EMT())
    trial = make_trial()
    before = trial.atoms.positions.copy()
    verdict = filt.evaluate(trial)
    assert verdict.accepted is True
    assert verdict.reason == "ok"
    assert verdict.value is not None and np.isfinite(verdict.value)
    assert verdict.value <= 1e6
    assert getattr(verdict.detail, "converged") is True
    assert np.allclose(trial.atoms.positions, before)
    assert filt.signature()["kind"] == "energy"


def test_energy_filter_rejects_above_the_ceiling() -> None:
    """The single-point tier keeps the rejection contract."""
    filt = EnergyFilter(e_cut=-1e9, calculator=EMT())
    verdict = filt.evaluate(make_trial())
    assert verdict.accepted is False
    assert verdict.reason == "e_cut"
    assert verdict.value is not None and verdict.value > filt.e_cut


def test_optimized_energy_filter_inherits_the_energy_filter() -> None:
    """The relaxation tier is the single-point tier plus an optimizer."""
    assert issubclass(OptimizedEnergyFilter, EnergyFilter)
    assert not issubclass(EnergyFilter, OptimizedEnergyFilter)
    relaxed = OptimizedEnergyFilter(e_cut=1e6, calculator=EMT(), max_steps=2)
    assert relaxed.signature()["kind"] == "optimized_energy"
    assert relaxed.signature()["max_steps"] == 2
    plain = EnergyFilter(e_cut=1e6, calculator=EMT())
    assert plain.signature()["kind"] == "energy"


def test_binding_cut_offsets_the_isolated_reference() -> None:
    """``e_bind_cut`` shifts the ``e_cut=None`` ceiling by a fixed energy."""
    trial = make_trial()
    loose = EnergyFilter(e_cut=None, calculator=EMT(), e_bind_cut=1e6)
    tight = EnergyFilter(e_cut=None, calculator=EMT(), e_bind_cut=-1e6)
    assert loose.evaluate(trial).accepted is True
    verdict = tight.evaluate(trial)
    assert verdict.accepted is False
    assert verdict.reason == "e_cut"
    assert tight.signature()["e_bind_cut"] == -1e6


def test_prepare_applies_the_binding_cut_to_the_cached_ceiling() -> None:
    """The prepared ceiling is the isolated reference plus the offset."""
    trial = make_trial()
    filt = EnergyFilter(e_cut=None, calculator=EMT(), e_bind_cut=-0.25)
    filt.prepare(trial)
    reference = filt._reference_cutoff(trial)
    assert filt._ceiling(trial) == pytest.approx(reference - 0.25)


def test_record_key_writes_the_key_of_an_accepted_trial() -> None:
    """The filter stores the canonical key and its cutoff in the metadata."""
    filt = EnergyFilter(
        e_cut=1e6, calculator=EMT(), record_key=True, key_cutoff=4.0
    )
    trial = make_trial()
    assert filt.evaluate(trial).accepted is True
    assert trial.meta["canonical_key"] == trial.get_canonical_key(cutoff=4.0)
    assert trial.meta["canonical_key_cutoff"] == 4.0


def test_record_key_is_off_by_default() -> None:
    """Without the flag the metadata is left untouched."""
    filt = EnergyFilter(e_cut=1e6, calculator=EMT())
    trial = make_trial()
    filt.evaluate(trial)
    assert "canonical_key" not in trial.meta


def test_record_key_skips_a_rejected_trial() -> None:
    """A rejected trial never pays for the key."""
    filt = EnergyFilter(e_cut=-1e9, calculator=EMT(), record_key=True)
    trial = make_trial()
    assert filt.evaluate(trial).accepted is False
    assert "canonical_key" not in trial.meta


def test_record_key_rejects_an_invalid_cutoff() -> None:
    """A non-positive key cutoff is a construction error."""
    with pytest.raises(ValueError):
        EnergyFilter(e_cut=1e6, calculator=EMT(), key_cutoff=0.0)
