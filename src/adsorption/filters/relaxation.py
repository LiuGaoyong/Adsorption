"""Process B, energetic tier: relaxation followed by an energy threshold.

The filter wraps an ASE calculator: it relaxes a copy of the trial structure
with :func:`arrayff.optimize.call_optimization` and accepts the trial when the
relaxed energy is finite, at most the ceiling and the convergence policy
allows it. With ``e_cut=None`` the ceiling is the sum of the isolated
substrate and adsorbate single-point energies. The verdict carries the energy
and the optimization result itself, so a caller never relaxes twice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from arrayff.optimize import call_optimization
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths
from graphatoms.system.sysGas import Gas

from adsorption.base import AdsorptionConfigurationFilterABC, Trial, Verdict

__all__ = ["OptimizedEnergyFilter"]


class OptimizedEnergyFilter(AdsorptionConfigurationFilterABC):
    """Reject trials above the energy ceiling or without convergence."""

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
        **kwargs,
    ) -> None:
        """Store the evaluator and the acceptance policy.

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
            **kwargs: unused arguments.
        """
        if fix_adsorbate_bonds:
            fix_substrate = True
        elif not fix_substrate:
            fix_adsorbate_bonds = False
        if e_cut is not None and not np.isfinite(e_cut):
            raise ValueError(f"e_cut must be finite or None, got {e_cut}.")
        self.e_cut: float | None = None if e_cut is None else float(e_cut)
        self._isolated_cache: dict[str, float] = {}
        self._cutoffs: dict[int, float] = {}
        self.fix_substrate = fix_substrate
        self.fix_adsorbate_bonds = fix_adsorbate_bonds
        self.reject_on_unconverged = reject_on_unconverged
        self.calculator = calculator
        self.max_steps = max_steps
        self.method = method
        self.fmax = fmax

    @staticmethod
    def _structure_key(atoms: Atoms) -> str:
        """Return a cache key of one isolated structure."""
        digest = hashlib.sha1()
        digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
        digest.update(np.asarray(atoms.positions, dtype=np.float64).tobytes())
        digest.update(
            np.asarray(atoms.cell.array, dtype=np.float64).tobytes()
        )
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
            self._cutoffs[key] = self._reference_cutoff(trial)

    def evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial with the wrapped relaxation.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted when the energy is finite, at most the ceiling
            and the convergence policy allows it. The ceiling is ``e_cut``
            when set, otherwise the sum of the isolated fragment energies.
        """
        cutoff = self.e_cut
        if cutoff is None:
            key = int(trial.meta.get("conformer", 0))
            cutoff = self._cutoffs.get(key)
            if cutoff is None:
                cutoff = self._reference_cutoff(trial)
                self._cutoffs[key] = cutoff
            if not np.isfinite(cutoff):
                return Verdict(False, cutoff, "nan", None)
        self.calculator.reset()
        atoms = trial.atoms.copy()
        if self.fix_substrate:
            ids = list(range(trial.natoms_substrate))
            atoms.constraints = [FixAtoms(indices=ids)]
        if self.fix_adsorbate_bonds:
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

        result = call_optimization(
            calc=self.calculator,
            atoms=atoms,
            fmax=self.fmax,
            max_steps=self.max_steps,
            method=self.method,
        )
        energy = result.latest.get_potential_energy()
        if not np.isfinite(energy):
            return Verdict(False, energy, "nan", result)
        if energy > cutoff:
            return Verdict(False, energy, "e_cut", result)
        if not result.converged and self.reject_on_unconverged:
            return Verdict(False, energy, "unconverged", result)
        return Verdict(True, energy, "ok", result)

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "optimized_energy",
            "calculator": self.calculator.todict(),
            "e_cut": self.e_cut,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            "method": self.method,
            "fix_substrate": self.fix_substrate,
            "fix_adsorbate_bonds": self.fix_adsorbate_bonds,
            "reject_on_unconverged": self.reject_on_unconverged,
        }
