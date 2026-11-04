"""Process B, energetic tier: relaxation followed by an energy threshold.

The filter wraps an ASE calculator: it relaxes a copy of the trial structure
with :func:`arrayff.optimize.call_optimization` and accepts the trial when the
relaxed energy is finite, at most ``e_cut`` and the convergence policy allows
it. The verdict carries the energy and the optimization result itself, so a
caller never relaxes twice.
"""

from __future__ import annotations

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
        e_cut: float,
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
            e_cut: Energy ceiling in eV.
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
        if not np.isfinite(e_cut):
            raise ValueError(f"e_cut must be finite, got {e_cut}.")
        self.e_cut = float(e_cut)
        self.fix_substrate = fix_substrate
        self.fix_adsorbate_bonds = fix_adsorbate_bonds
        self.reject_on_unconverged = reject_on_unconverged
        self.calculator = calculator
        self.max_steps = max_steps
        self.method = method
        self.fmax = fmax

    def evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial with the wrapped relaxation.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted when the energy is finite, at most ``e_cut``
            and the convergence policy allows it.
        """
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
        if energy > self.e_cut:
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
