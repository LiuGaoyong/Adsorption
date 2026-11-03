"""Compose process A and process B: generate, filter, record.

The pipeline evaluates the filters in the declared order and stops at the
first rejection. It records every verdict it saw, so the sidecar can attribute
a rejection to the tier that produced it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._base import (
    AdsorptionConfigurationFilterABC,
    AdsorptionConfigurationGeneratorABC,
    Trial,
    Verdict,
)

__all__ = ["AdsorptionConfigurationPipeline", "Record"]


@dataclass(frozen=True)
class Record:
    """One trial with the verdicts it received.

    Attributes:
        trial: The generated trial.
        verdicts: The verdicts of the filters that were evaluated, in order.
        first_failed: Name of the first rejecting filter, or ``None``.
        result: The payload of the last verdict, e.g. an ``EvalResult``.
    """

    trial: Trial
    verdicts: tuple[Verdict, ...] = field(default_factory=tuple)
    first_failed: str | None = None
    result: Any | None = None


class AdsorptionConfigurationPipeline:
    """Run a generator through an ordered filter chain.

    Attributes:
        generator: Process A.
        filters: Process B tiers, applied in order.
    """

    def __init__(
        self,
        generator: AdsorptionConfigurationGeneratorABC,
        filters: Sequence[AdsorptionConfigurationFilterABC],
    ) -> None:
        """Store the generator and the filter chain.

        Args:
            generator: Process A.
            filters: Process B tiers; may be empty.
        """
        self.generator = generator
        self.filters = tuple(filters)

    def run(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Record]:
        """Yield the records of an index interval.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end when ``None``.

        Yields:
            Record: One record per trial, in index order.
        """
        for trial in self.generator.generate(start, stop):
            verdicts: list[Verdict] = []
            first_failed: str | None = None
            for filt in self.filters:
                verdict = filt.evaluate(trial)
                verdicts.append(verdict)
                if not verdict.accepted:
                    first_failed = type(filt).__name__
                    break
            yield Record(
                trial=trial,
                verdicts=tuple(verdicts),
                first_failed=first_failed,
                result=verdicts[-1].detail if verdicts else None,
            )

    def accepted(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Record]:
        """Yield only the records that survive every filter.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end when ``None``.

        Yields:
            Record: The surviving records, in index order.
        """
        for record in self.run(start, stop):
            if record.first_failed is None:
                yield record
