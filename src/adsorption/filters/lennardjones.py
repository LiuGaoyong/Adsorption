"""Process B, steric tiers: a Lennard-Jones screen and its relaxation.

:class:`LennardJonesScreenFilter` evaluates the single-point Lennard-Jones
energy of the placed trial and keeps it only when the energy stays below the
registered ceiling: no optimizer runs, so it is the cheap steric screen.
:class:`LennardJonesEnergyFilter` is the same tier with the relaxation of
:class:`~adsorption.filters.relaxation.OptimizedEnergyFilter`. With
``e_cut=None`` the ceiling is the sum of the isolated substrate and adsorbate
Lennard-Jones energies. Both only remove: neither ranks candidates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from arrayff.potentials.pairwise.calculator import LJCalculator

from adsorption.filters.relaxation import EnergyFilter, OptimizedEnergyFilter

__all__ = ["LennardJonesEnergyFilter", "LennardJonesScreenFilter"]


class LennardJonesScreenFilter(EnergyFilter):
    """Reject configurations with a single-point Lennard-Jones energy.

    The filter is the energetic screen of
    :mod:`adsorption.filters.relaxation` with a Lennard-Jones calculator and
    no optimizer: it evaluates the placed trial as it is, which makes it a
    cheap stage before a relaxation tier. With ``e_cut=None`` the ceiling is
    the sum of the isolated substrate and adsorbate single-point energies.
    """

    def __init__(
        self,
        e_cut: float | None = None,
        *,
        reject_on_unconverged: bool = True,
        e_bind_cut: float = 0.0,
        **kwargs,
    ) -> None:
        """Store the evaluator and the acceptance policy.

        Args:
            e_cut: Energy ceiling in eV, or ``None`` to use the sum of the
                isolated substrate and adsorbate single-point energies.
            reject_on_unconverged: Whether to reject trials whose evaluation
                reports ``converged=False``.
            e_bind_cut: Binding-energy offset of the ``e_cut=None`` ceiling.
            **kwargs: unused arguments.
        """
        super().__init__(
            e_cut,
            LJCalculator(),
            reject_on_unconverged=reject_on_unconverged,
            e_bind_cut=e_bind_cut,
            **kwargs,
        )

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        res = dict(super().signature())
        res["kind"] = "lennard_jones_screen"
        res["calculator"] = "Lennard-Jones Potential by OpenKIM Parameters"
        return res


class LennardJonesEnergyFilter(OptimizedEnergyFilter):
    """Reject adsorption configurations with a relaxed Lennard-Jones energy.

    The filter is the energetic tier of :mod:`adsorption.filters.relaxation`
    with a Lennard-Jones calculator and the same fixed-budget relaxation
    protocol; the ceiling applies to the relaxed energy.
    """

    def __init__(
        self,
        e_cut: float | None,
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
