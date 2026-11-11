"""Process B, energetic tiers: a single-point ceiling and its relaxation.

:class:`EnergyFilter` evaluates the energy of the placed trial without moving
a single atom: it is the cheap energetic screen. :class:`OptimizedEnergyFilter`
inherits from it and relaxes the trial before applying the same ceiling, so
both tiers share the reference-energy bookkeeping, the acceptance policy and
the shape of the verdict.

With ``e_cut=None`` the ceiling is the sum of the isolated substrate and
adsorbate energies, i.e. the trial must bind: already for the single-point
tier, after the relaxation for the optimized tier. The isolated adsorbate can
be referenced in three ways (``adsorbate_reference``):

* ``"conformer"`` -- the single-point energy of the rigid conformer as placed.
  Exact for a rigid-body screen, where the intramolecular energy is a
  constant that cancels.
* ``"relaxed"`` -- the energy of the conformer relaxed in the gas phase with
  the same calculator and optimizer. Appropriate when the tier lets the
  adsorbate deform, so that intramolecular relaxation is not counted as
  binding.
* ``"shared"`` -- the lowest relaxed energy over every conformer the filter
  has been prepared with, so all conformers of one adsorbate share a single
  gas-phase reference. The campaign runner prepares the filter with one trial
  per component before the candidate loop; a conformer that was not prepared
  lowers the shared reference when it first appears.

An explicit ``e_adsorbate`` pins the isolated-adsorbate energy to a value
computed elsewhere, e.g. a gas-phase global minimum, and overrides the three
modes.

The verdict carries the energy and the evaluation result itself, so a caller
never evaluates or relaxes twice. The relaxation tier also stores the relaxed
structure of an accepted trial in ``trial.meta["relaxed"]`` so a later tier
can start from it (``start_from="previous"``), a basin deduplicator can key
it, and the campaign runner can persist the relaxed rather than the placed
structure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from arrayff.optimize import call_optimization
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths

from adsorption.base import AdsorptionConfigurationFilterABC, Trial, Verdict

__all__ = [
    "AdsorbateReference",
    "EnergyFilter",
    "OptimizedEnergyFilter",
    "RELAXED_META_KEY",
    "SinglePointResult",
]

type AdsorbateReference = Literal["conformer", "relaxed", "shared"]

RELAXED_META_KEY = "relaxed"
"""Metadata key under which a relaxation tier stores its relaxed structure.

The value is a mapping with ``positions`` (``(n_atoms, 3)`` array of the
whole trial), ``energy`` (eV), ``converged`` and ``filter`` (the class name of
the tier). It is an in-memory channel between pipeline stages; the campaign
runner drops it from the JSON records and persists the positions in the
structure database instead.
"""


@dataclass(frozen=True)
class SinglePointResult:
    """The stand-in result of a single-point evaluation.

    It mirrors the two attributes of
    :class:`arrayff.optimize.OptimizationResult` the filters and the record
    sidecar read (``latest`` and ``converged``), so a screening tier and a
    relaxation tier produce the same verdict shape.

    Attributes:
        atoms: The evaluated structure, calculator attached.
        converged: Always ``True``: a single point has no convergence test.
    """

    atoms: Atoms
    converged: bool = True

    @property
    def latest(self) -> Atoms:
        """Return the evaluated structure, as ``OptimizationResult`` does."""
        return self.atoms


class EnergyFilter(AdsorptionConfigurationFilterABC):
    """Reject trials whose single-point energy is above the ceiling."""

    def __init__(
        self,
        e_cut: float | None,
        calculator: Calculator,
        *,
        reject_on_unconverged: bool = True,
        e_bind_cut: float = 0.0,
        adsorbate_reference: AdsorbateReference = "conformer",
        e_adsorbate: float | None = None,
        record_key: bool = False,
        key_cutoff: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Store the evaluator and the acceptance policy.

        Args:
            e_cut: Energy ceiling in eV, or ``None`` to use the sum of the
                isolated substrate and adsorbate energies.
            calculator: The ASE calculator to use.
            reject_on_unconverged: Whether to reject trials whose evaluation
                reports ``converged=False``. A single point always reports
                convergence, so the flag only matters to subclasses.
            e_bind_cut: Binding-energy offset in eV added to the isolated
                fragment reference when ``e_cut`` is ``None``. The trial is
                accepted only when its energy is at most
                ``reference + e_bind_cut``. The default ``0.0`` accepts every
                non-repulsive placement; a negative value requires a minimum
                binding energy. Ignored when ``e_cut`` is set.
            adsorbate_reference: How the isolated adsorbate enters the
                ``e_cut=None`` reference; see the module docstring. The
                single-point tier relaxes nothing, so ``"relaxed"`` and
                ``"shared"`` run an unconstrained gas-phase relaxation with
                the optimizer settings of :class:`OptimizedEnergyFilter`.
            e_adsorbate: Explicit isolated-adsorbate energy in eV; overrides
                ``adsorbate_reference``.
            record_key: Whether to store the canonical key of an accepted
                trial in ``trial.meta``.
            key_cutoff: Neighbor cutoff of that key, in Angstrom.
            **kwargs: unused arguments.

        Raises:
            ValueError: If ``e_cut``, ``e_bind_cut`` or ``e_adsorbate`` is
                not finite, or the reference mode is unknown.
        """
        super().__init__(record_key=record_key, key_cutoff=key_cutoff)
        if e_cut is not None and not np.isfinite(e_cut):
            raise ValueError(f"e_cut must be finite or None, got {e_cut}.")
        if not np.isfinite(e_bind_cut):
            raise ValueError(f"e_bind_cut must be finite, got {e_bind_cut}.")
        if e_adsorbate is not None and not np.isfinite(e_adsorbate):
            raise ValueError(
                f"e_adsorbate must be finite or None, got {e_adsorbate}."
            )
        if adsorbate_reference not in {"conformer", "relaxed", "shared"}:
            raise ValueError(
                "adsorbate_reference must be 'conformer', 'relaxed' or "
                f"'shared', got {adsorbate_reference!r}."
            )
        self.e_cut: float | None = None if e_cut is None else float(e_cut)
        self.e_bind_cut = float(e_bind_cut)
        self.adsorbate_reference: AdsorbateReference = adsorbate_reference
        self.e_adsorbate: float | None = (
            None if e_adsorbate is None else float(e_adsorbate)
        )
        self.reject_on_unconverged = bool(reject_on_unconverged)
        self.calculator = calculator
        self._isolated_cache: dict[str, float] = {}
        self._adsorbate_cache: dict[int, float] = {}
        self._shared_reference: float | None = None

    # -- isolated fragments -------------------------------------------------

    @staticmethod
    def _structure_key(atoms: Atoms) -> str:
        """Return a cache key of one isolated structure."""
        digest = hashlib.sha1()
        digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
        digest.update(np.asarray(atoms.positions, dtype=np.float64).tobytes())
        digest.update(np.asarray(atoms.cell.array, dtype=np.float64).tobytes())
        digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes())
        return digest.hexdigest()

    def _isolated_energy(self, atoms: Atoms) -> float:
        """Return the single-point energy of one isolated fragment."""
        key = self._structure_key(atoms)
        cached = self._isolated_cache.get(key)
        if cached is not None:
            return cached
        self.calculator.reset()
        work = atoms.copy()
        work.calc = self.calculator
        energy = float(work.get_potential_energy())
        self._isolated_cache[key] = energy
        return energy

    def _relax_isolated(self, atoms: Atoms) -> float:
        """Return the energy of one fragment relaxed without constraints.

        The optimizer settings come from :class:`OptimizedEnergyFilter` when
        this filter is one; the single-point tier falls back to the defaults
        of that class.
        """
        key = "relaxed:" + self._structure_key(atoms)
        cached = self._isolated_cache.get(key)
        if cached is not None:
            return cached
        work = atoms.copy()
        work.constraints = []
        result = call_optimization(
            calc=self.calculator,
            atoms=work,
            fmax=float(getattr(self, "fmax", 0.01)),
            max_steps=int(getattr(self, "max_steps", 100)),
            method=str(getattr(self, "method", "LBFGS")),
        )
        energy = float(result.latest.get_potential_energy())
        self._isolated_cache[key] = energy
        return energy

    @staticmethod
    def _fragments(trial: Trial) -> tuple[Atoms, Atoms]:
        """Split a trial into its isolated substrate and adsorbate."""
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
        return substrate, adsorbate

    def _substrate_energy(self, trial: Trial) -> float:
        """Return the isolated substrate energy (cached by structure)."""
        substrate, _ = self._fragments(trial)
        return self._isolated_energy(substrate)

    def _adsorbate_energy(self, trial: Trial) -> float:
        """Return the isolated adsorbate energy under the reference policy."""
        if self.e_adsorbate is not None:
            return self.e_adsorbate
        conformer = int(trial.meta.get("conformer", 0))
        value = self._adsorbate_cache.get(conformer)
        if value is None:
            _, adsorbate = self._fragments(trial)
            if self.adsorbate_reference == "conformer":
                value = self._isolated_energy(adsorbate)
            else:
                value = self._relax_isolated(adsorbate)
            self._adsorbate_cache[conformer] = value
            if self.adsorbate_reference == "shared":
                self._shared_reference = (
                    value
                    if self._shared_reference is None
                    else min(self._shared_reference, value)
                )
        if self.adsorbate_reference == "shared":
            assert self._shared_reference is not None
            return self._shared_reference
        return value

    def _reference_cutoff(self, trial: Trial) -> float:
        """Return the isolated substrate plus adsorbate energy."""
        return self._substrate_energy(trial) + self._adsorbate_energy(trial)

    def _ceiling(self, trial: Trial) -> float | None:
        """Return the ceiling of one trial, or ``None`` when not finite."""
        cutoff = self.e_cut
        if cutoff is None:
            cutoff = self._reference_cutoff(trial) + self.e_bind_cut
        return cutoff if np.isfinite(cutoff) else None

    def prepare(self, trial: Trial) -> None:
        """Resolve the reference energies before a large evaluation loop.

        The isolated fragment energies are cached per substrate and per
        conformer; with ``adsorbate_reference="shared"`` every prepared
        conformer contributes to the shared gas-phase reference, so preparing
        the filter with one trial per component makes the reference identical
        for every candidate of the campaign.

        Args:
            trial: A representative trial of one conformer.
        """
        if self.e_cut is not None:
            return
        self._reference_cutoff(trial)

    # -- evaluation ---------------------------------------------------------

    def _evaluate_atoms(self, trial: Trial) -> Any:
        """Evaluate one trial and return the result carrying ``latest``.

        Args:
            trial: The trial to evaluate.

        Returns:
            Any: A :class:`SinglePointResult` for this tier; subclasses
            return the optimizer result instead.
        """
        atoms = trial.atoms.copy()
        atoms.calc = self.calculator
        return SinglePointResult(atoms=atoms)

    def _verdict(
        self,
        energy: float,
        cutoff: float,
        result: Any,
    ) -> Verdict:
        """Apply the ceiling and the convergence policy to one energy."""
        if not np.isfinite(energy):
            return Verdict(False, energy, "nan", result)
        if energy > cutoff:
            return Verdict(False, energy, "e_cut", result)
        if not result.converged and self.reject_on_unconverged:
            return Verdict(False, energy, "unconverged", result)
        return Verdict(True, energy, "ok", result)

    def _evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial with a single-point energy.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted when the energy is finite and at most the
            ceiling. The ceiling is ``e_cut`` when set, otherwise the sum of
            the isolated fragment energies.
        """
        cutoff = self._ceiling(trial)
        if cutoff is None:
            return Verdict(False, float("nan"), "nan", None)
        self.calculator.reset()
        result = self._evaluate_atoms(trial)
        energy = float(result.latest.get_potential_energy())
        return self._verdict(energy, cutoff, result)

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "energy",
            "calculator": self.calculator.todict(),
            "e_cut": self.e_cut,
            "e_bind_cut": self.e_bind_cut,
            "adsorbate_reference": self.adsorbate_reference,
            "e_adsorbate": self.e_adsorbate,
            "reject_on_unconverged": self.reject_on_unconverged,
        }


class OptimizedEnergyFilter(EnergyFilter):
    """Relax the trial, then reject it above the energy ceiling."""

    def __init__(
        self,
        e_cut: float | None,
        calculator: Calculator,
        *,
        fmax: float = 0.01,
        max_steps: int = 100,
        method: str = "LBFGS",
        fix_substrate: bool = True,
        fix_adsorbate_bonds: bool = True,
        reject_on_unconverged: bool = True,
        e_bind_cut: float = 0.0,
        adsorbate_reference: AdsorbateReference | None = None,
        e_adsorbate: float | None = None,
        start_from: Literal["trial", "previous"] = "trial",
        record_relaxed: bool = True,
        record_key: bool = False,
        key_cutoff: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Store the optimizer and the acceptance policy.

        Args:
            e_cut: Energy ceiling in eV, or ``None`` to use the sum of the
                isolated substrate and adsorbate energies.
            calculator: The ASE calculator to use.
            fmax: Force tolerance in eV/A.
            max_steps: Maximum number of steps.
            method: Optimization method.
            fix_substrate: Whether to fix the substrate atoms; forced on when
                ``fix_adsorbate_bonds`` is set.
            fix_adsorbate_bonds: Whether to constrain every intramolecular
                distance of the adsorbate to its reference value.
            reject_on_unconverged: Whether to reject trials on unconverged
                optimizations.
            e_bind_cut: Binding-energy offset of the ``e_cut=None`` ceiling.
            adsorbate_reference: Isolated-adsorbate reference policy. When
                ``None`` it follows the constraints: ``"conformer"`` while
                the intramolecular distances are fixed (the molecule is a
                rigid body and its internal energy cancels), ``"relaxed"``
                otherwise (the molecule may deform, so gas-phase relaxation
                must not count as binding).
            e_adsorbate: Explicit isolated-adsorbate energy in eV.
            start_from: ``"trial"`` relaxes the placed structure;
                ``"previous"`` starts from the relaxed structure that an
                earlier relaxation tier stored in ``trial.meta``, falling
                back to the placed structure when none is present.
            record_relaxed: Whether to store the relaxed structure of an
                accepted trial under ``trial.meta["relaxed"]``.
            record_key: Whether to store the canonical key of an accepted
                trial in ``trial.meta``.
            key_cutoff: Neighbor cutoff of that key, in Angstrom.
            **kwargs: unused arguments.
        """
        if fix_adsorbate_bonds:
            fix_substrate = True
        elif not fix_substrate:
            fix_adsorbate_bonds = False
        if start_from not in {"trial", "previous"}:
            raise ValueError(
                f"start_from must be 'trial' or 'previous', got {start_from!r}."
            )
        self.fix_substrate = bool(fix_substrate)
        self.fix_adsorbate_bonds = bool(fix_adsorbate_bonds)
        self.max_steps = max_steps
        self.method = method
        self.fmax = fmax
        self.start_from = start_from
        self.record_relaxed = bool(record_relaxed)
        if adsorbate_reference is None:
            adsorbate_reference = (
                "conformer" if self.fix_adsorbate_bonds else "relaxed"
            )
        super().__init__(
            e_cut,
            calculator,
            reject_on_unconverged=reject_on_unconverged,
            e_bind_cut=e_bind_cut,
            adsorbate_reference=adsorbate_reference,
            e_adsorbate=e_adsorbate,
            record_key=record_key,
            key_cutoff=key_cutoff,
            **kwargs,
        )

    def _starting_structure(self, trial: Trial) -> Atoms:
        """Return the structure the relaxation starts from."""
        atoms = trial.atoms.copy()
        if self.start_from != "previous":
            return atoms
        previous = trial.meta.get(RELAXED_META_KEY)
        if not isinstance(previous, Mapping):
            return atoms
        positions = np.asarray(previous.get("positions"), dtype=float)
        if positions.shape != atoms.positions.shape:
            return atoms
        atoms.set_positions(positions)
        return atoms

    def _evaluate_atoms(self, trial: Trial) -> Any:
        """Relax one trial and return the optimizer result."""
        atoms = self._starting_structure(trial)
        if self.fix_substrate:
            ids = list(range(trial.natoms_substrate))
            atoms.constraints = [FixAtoms(indices=ids)]
        if self.fix_adsorbate_bonds:
            from graphatoms.system.sysGas import Gas  # noqa: PLC0415

            z = atoms.numbers[-trial.natoms_adsorbate :]
            r = atoms.positions[-trial.natoms_adsorbate :]
            gas_atoms = Atoms(numbers=z, positions=r)
            edges = Gas.from_ase(gas_atoms).pair
            assert isinstance(edges, np.ndarray)
            edges += trial.natoms_substrate
            atoms.constraints = [
                *atoms.constraints,
                FixBondLengths(pairs=edges),
            ]
        return call_optimization(
            calc=self.calculator,
            atoms=atoms,
            fmax=self.fmax,
            max_steps=self.max_steps,
            method=self.method,
        )

    def _evaluate(self, trial: Trial) -> Verdict:
        """Relax, judge, and record the relaxed structure of a survivor."""
        verdict = super()._evaluate(trial)
        if (
            verdict.accepted
            and self.record_relaxed
            and verdict.detail is not None
            and isinstance(trial.meta, MutableMapping)
        ):
            latest = verdict.detail.latest
            trial.meta[RELAXED_META_KEY] = {
                "positions": np.array(latest.positions, dtype=float),
                "energy": float(verdict.value),  # type: ignore[arg-type]
                "converged": bool(verdict.detail.converged),
                "filter": type(self).__name__,
            }
            trial.meta["relaxed_energy"] = float(verdict.value)  # type: ignore[arg-type]
        return verdict

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        signature = dict(super().signature())
        signature["kind"] = "optimized_energy"
        signature.update(
            {
                "fmax": self.fmax,
                "max_steps": self.max_steps,
                "method": self.method,
                "fix_substrate": self.fix_substrate,
                "fix_adsorbate_bonds": self.fix_adsorbate_bonds,
                "start_from": self.start_from,
                "record_relaxed": self.record_relaxed,
            }
        )
        return signature
