"""Process B, steric tier: the repulsive Lennard-Jones energy.

Only the repulsive branch is summed, and the filter only removes: a trial is
accepted when the total repulsive energy stays below the registered ceiling.
The filter never ranks candidates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from arrayff.potentials import LennardJones
from ase.calculators.calculator import Calculator

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    Trial,
    Verdict,
)

__all__ = ["LJFilter"]


class LJFilter(AdsorptionConfigurationFilterABC):
    """Reject adsorption configurations with a total repulsive Lennard-Jones energy.

    Attributes:
        epsilon: Per-element epsilon in eV.
        sigma: Per-element sigma in Angstrom.
        r_cut: Pair cutoff in Angstrom.
        e_cut: Repulsive-energy ceiling in eV.
    """

    def __init__(self, e_cut: float) -> None:
        """Store the parameter table and the cutoff."""
        if not np.isfinite(e_cut):
            raise ValueError(f"e_cut must be finite, got {e_cut}.")
        self.e_cut = float(e_cut)
        self.__lj: Calculator = LennardJones()

    def evaluate(self, trial: Trial) -> Verdict:
        """Return the repulsive energy of one trial.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted when ``e_rep <= e_cut``; ``value`` is
            ``e_rep`` and ``detail`` reports the pair count.
        """
        atoms = trial.atoms.copy()
        atoms.calc = self.__lj
        atoms.calc.reset()
        energy = float(atoms.get_potential_energy())
        accepted = energy <= self.e_cut
        return Verdict(
            accepted=accepted,
            value=energy,
            reason="ok" if accepted else "e_cut",
            detail={"energy": energy},
        )

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "lennard_jones",
            "e_cut": self.e_cut,
        }
