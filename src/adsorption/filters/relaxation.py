"""Process B, energetic tier: relaxation followed by an energy threshold.

The filter wraps any callable with the signature of
:class:`adsorption.evaluation.Relaxation`: it receives a structure and the
number of substrate atoms and returns an ``EvalResult``. The verdict carries
the energy and the result itself, so a caller never relaxes twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np

from ....deprecated.src.adsorption.evaluation import EvalResult
from ..process import AdsConfFilterABC, Trial, Verdict

__all__ = ["EnergyFilter"]


class EnergyFilter(AdsConfFilterABC):
    """Reject trials above the energy ceiling or without convergence.

    Attributes:
        relaxation: The evaluation callable.
        e_cut: Energy ceiling in eV.
        on_unconverged: ``"reject"`` (default) or ``"keep"``.
    """

    def __init__(
        self,
        relaxation: Callable[..., EvalResult],
        *,
        e_cut: float,
        on_unconverged: Literal["reject", "keep"] = "reject",
    ) -> None:
        """Store the evaluator and the acceptance policy.

        Args:
            relaxation: Callable ``(atoms, *, n_substrate) -> EvalResult``.
            e_cut: Energy ceiling in eV.
            on_unconverged: Whether a non-converged relaxation is rejected.

        Raises:
            ValueError: If ``e_cut`` is not finite or the policy is unknown.
        """
        if not np.isfinite(e_cut):
            raise ValueError(f"e_cut must be finite, got {e_cut}.")
        if on_unconverged not in {"reject", "keep"}:
            raise ValueError(
                f"Unknown on_unconverged policy: {on_unconverged!r}."
            )
        self.relaxation = relaxation
        self.e_cut = float(e_cut)
        self.on_unconverged = on_unconverged

    def _n_substrate(self, trial: Trial) -> int:
        """Return the substrate size recorded in the trial metadata.

        Args:
            trial: The trial.

        Returns:
            int: The number of substrate atoms.

        Raises:
            ValueError: If the metadata does not carry a valid size.
        """
        value = trial.meta.get("n_substrate")
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                "The trial metadata must carry a positive integer "
                f"'n_substrate', got {value!r}."
            )
        return value

    def evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial with the wrapped relaxation.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted when the energy is finite, at most ``e_cut``
            and the convergence policy allows it.
        """
        result = self.relaxation(
            trial.atoms, n_substrate=self._n_substrate(trial)
        )
        energy = float(result.energy)
        if not np.isfinite(energy):
            return Verdict(False, energy, "nan", result)
        if energy > self.e_cut:
            return Verdict(False, energy, "e_cut", result)
        if result.converged is False and self.on_unconverged == "reject":
            return Verdict(False, energy, "unconverged", result)
        return Verdict(True, energy, "ok", result)

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        name = getattr(self.relaxation, "__class__", type(None)).__name__
        return {
            "kind": "energy",
            "e_cut": self.e_cut,
            "on_unconverged": self.on_unconverged,
            "relaxation": name,
        }
