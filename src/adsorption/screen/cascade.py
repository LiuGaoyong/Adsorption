"""The deterministic screening cascade: generate, screen, evaluate.

The cascade only narrows: a candidate rejected by the steric screen is never
relaxed, and the evaluation tier never re-admits it. The screen therefore never
ranks; the evaluation tier is the only tier that produces an energy used for
ordering.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from ase import Atoms

from ..protocol import Campaign, Candidate
from .lj import LJDecision, RepulsiveScreen

__all__ = ["Cascade", "CascadeRecord", "Evaluator"]


class Evaluator(Protocol):
    """A callable that evaluates or relaxes one candidate."""

    def __call__(self, atoms: Atoms) -> Any:
        """Return the evaluation result of a candidate structure."""
        ...


@dataclass(frozen=True)
class CascadeRecord:
    """One candidate with its screen verdict and optional evaluation.

    Attributes:
        candidate: The generated candidate.
        decision: The Lennard-Jones screen verdict.
        evaluation: The evaluator result for a surviving candidate, or
            ``None`` when the candidate was rejected or no evaluator is set.
    """

    candidate: Candidate
    decision: LJDecision
    evaluation: Any | None = None


class Cascade:
    """Run the screen and then, for survivors, the evaluation tier.

    Args:
        screen: The steric screen.
        evaluator: Optional callable applied to each survivor. Without it the
            cascade stops at the screen verdict.
    """

    def __init__(
        self,
        screen: RepulsiveScreen,
        evaluator: Evaluator | None = None,
    ) -> None:
        """Store the tiers."""
        self.screen = screen
        self.evaluator = evaluator

    def run(
        self,
        campaign: Campaign,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[CascadeRecord]:
        """Screen a candidate interval.

        Args:
            campaign: The prepared campaign.
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the campaign end when
                None.

        Yields:
            CascadeRecord: One record per candidate, in index order.
        """
        n_substrate = len(campaign.atoms)
        for candidate in campaign.candidates(start, stop):
            decision = self.screen.evaluate(
                candidate.atoms,
                n_substrate=n_substrate,
            )
            evaluation = None
            if decision.passed and self.evaluator is not None:
                evaluation = self.evaluator(candidate.atoms)
            yield CascadeRecord(
                candidate=candidate,
                decision=decision,
                evaluation=evaluation,
            )
