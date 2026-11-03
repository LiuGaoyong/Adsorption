"""Process B, steric tier: the repulsive Lennard-Jones energy.

Only the repulsive branch is summed, and the filter only removes: a trial is
accepted when the total repulsive energy stays below the registered ceiling.
The filter never ranks candidates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from arrayff.potentials.pairwise.calculator import LJCalculator

from adsorption.filters.relaxation import OptimizedEnergyFilter

__all__ = ["LennardJonesEnergyFilter"]


class LennardJonesEnergyFilter(OptimizedEnergyFilter):
    """Reject adsorption configurations with a total Lennard-Jones energy."""

    def __init__(
        self,
        e_cut: float,
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
            fmax: Force tolerance in eV/A.
            max_steps: Maximum number of steps.
            method: Optimization method.
            fix_substrate: Whether to fix the substrate atoms.
                Only effective if ``fix_adsorbate_bonds`` is ``False``.
            fix_adsorbate_bonds: Whether to fix the adsorbate bonds.
            reject_on_unconverged: Whether to reject trials on unconverged
                optimizations.
            **kwargs: unused arguments.
        """
        super().__init__(
            e_cut=e_cut,
            calculator=LJCalculator(),
            fmax=fmax,
            max_steps=max_steps,
            method=method,
            fix_substrate=fix_substrate,
            fix_adsorbate_bonds=fix_adsorbate_bonds,
            reject_on_unconverged=reject_on_unconverged,
            **kwargs,
        )

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        res = dict({k: v for k, v in super().signature().items()})
        res["calculator"] = "Lennard-Jones Potential by OpenKIM Parameters"
        return res
