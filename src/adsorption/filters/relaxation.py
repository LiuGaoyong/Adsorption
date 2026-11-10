"""Process B, energetic tiers: a single-point ceiling and its relaxation.

:class:`EnergyFilter` evaluates the energy of the placed trial without moving
a single atom: it is the cheap energetic screen. :class:`OptimizedEnergyFilter`
inherits from it and relaxes the trial before applying the same ceiling, so
both tiers share the reference-energy bookkeeping, the acceptance policy and
the shape of the verdict.

With ``e_cut=None`` the ceiling is the sum of the isolated substrate and
adsorbate single-point energies, i.e. the trial must bind: already for the
single-point tier, after the relaxation for the optimized tier. The verdict
carries the energy and the evaluation result itself, so a caller never
evaluates or relaxes twice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from arrayff.optimize import call_optimization
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths

from adsorption.base import AdsorptionConfigurationFilterABC, Trial, Verdict

__all__ = ["EnergyFilter", "OptimizedEnergyFilter", "SinglePointResult"]


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
        record_key: bool = False,
        key_cutoff: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Store the evaluator and the acceptance policy.

        Args:
            e_cut: Energy ceiling in eV, or ``None`` to use the sum of the
                isolated substrate and adsorbate single-point energies.
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
            record_key: Whether to store the canonical key of an accepted
                trial in ``trial.meta``.
            key_cutoff: Neighbor cutoff of that key, in Angstrom.
            **kwargs: unused arguments.

        Raises:
            ValueError: If ``e_cut`` or ``e_bind_cut`` is not finite.
        """
        super().__init__(record_key=record_key, key_cutoff=key_cutoff)
        if e_cut is not None and not np.isfinite(e_cut):
            raise ValueError(f"e_cut must be finite or None, got {e_cut}.")
        if not np.isfinite(e_bind_cut):
            raise ValueError(f"e_bind_cut must be finite, got {e_bind_cut}.")
        self.e_cut: float | None = None if e_cut is None else float(e_cut)
        self.e_bind_cut = float(e_bind_cut)
        self.reject_on_unconverged = bool(reject_on_unconverged)
        self.calculator = calculator
        self._isolated_cache: dict[str, float] = {}
        self._cutoffs: dict[int, float] = {}

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

    def _reference_cutoff(self, trial: Trial) -> float:
        """Return the isolated substrate plus adsorbate energy."""
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
        return self._isolated_energy(substrate) + self._isolated_energy(
            adsorbate
        )

    def _ceiling(self, trial: Trial) -> float | None:
        """Return the ceiling of one trial, or ``None`` when not finite."""
        cutoff = self.e_cut
        if cutoff is not None:
            return cutoff
        key = int(trial.meta.get("conformer", 0))
        cutoff = self._cutoffs.get(key)
        if cutoff is None:
            cutoff = self._reference_cutoff(trial) + self.e_bind_cut
            self._cutoffs[key] = cutoff
        return cutoff if np.isfinite(cutoff) else None

    def prepare(self, trial: Trial) -> None:
        """Resolve the reference ceiling before a large evaluation loop.

        The cutoff is cached per conformer index; the filter then reuses it
        for every subsequent trial instead of recomputing the isolated
        fragment energies.

        Args:
            trial: A representative trial of one conformer.
        """
        if self.e_cut is not None:
            return
        key = int(trial.meta.get("conformer", 0))
        if key not in self._cutoffs:
            self._cutoffs[key] = (
                self._reference_cutoff(trial) + self.e_bind_cut
            )

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
        record_key: bool = False,
        key_cutoff: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Store the optimizer and the acceptance policy.

        Args:
            e_cut: Energy ceiling in eV, or ``None`` to use the sum of the
                isolated substrate and adsorbate single-point energies.
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
            record_key: Whether to store the canonical key of an accepted
                trial in ``trial.meta``.
            key_cutoff: Neighbor cutoff of that key, in Angstrom.
            **kwargs: unused arguments.
        """
        if fix_adsorbate_bonds:
            fix_substrate = True
        elif not fix_substrate:
            fix_adsorbate_bonds = False
        self.fix_substrate = bool(fix_substrate)
        self.fix_adsorbate_bonds = bool(fix_adsorbate_bonds)
        self.max_steps = max_steps
        self.method = method
        self.fmax = fmax
        super().__init__(
            e_cut,
            calculator,
            reject_on_unconverged=reject_on_unconverged,
            e_bind_cut=e_bind_cut,
            record_key=record_key,
            key_cutoff=key_cutoff,
            **kwargs,
        )

    def _evaluate_atoms(self, trial: Trial) -> Any:
        """Relax one trial and return the optimizer result."""
        atoms = trial.atoms.copy()
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
            }
        )
        return signature
