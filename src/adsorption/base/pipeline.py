"""Compose process A and process B: generate, filter, record.

The pipeline evaluates the filters in the declared order and stops at the
first rejection. It records every verdict it saw, so the sidecar can attribute
a rejection to the tier that produced it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from adsorption.base._base import (
    AdsorptionConfigurationGeneratorABC,
    FilterStage,
    Trial,
    Verdict,
)

__all__ = ["AdsorptionConfigurationPipeline", "Record", "evaluate_trial"]

_LOG = logging.getLogger(__name__)


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


_REPORTED_EXCEPTIONS: set[tuple[str, str]] = set()


def _report_exception(
    filt: FilterStage,
    trial: Trial,
    error: Exception,
) -> None:
    """Log one filter exception, once per filter and exception type.

    A campaign can evaluate hundreds of millions of trials, and a relaxation
    that blows up on a few of them is expected (a very short anchor distance
    can push a Lennard-Jones step to non-finite coordinates). The first
    occurrence of every ``(filter, exception type)`` pair keeps its traceback
    so the failure stays diagnosable; later ones are logged at debug level so
    they cannot swamp the logs of a long run.

    Args:
        filt: The filter that raised.
        trial: The trial being evaluated.
        error: The exception that escaped the filter.
    """
    key = (type(filt).__name__, type(error).__name__)
    if key in _REPORTED_EXCEPTIONS:
        _LOG.debug(
            "Filter %s raised %s on candidate %s.",
            key[0],
            key[1],
            trial.index,
        )
        return
    _REPORTED_EXCEPTIONS.add(key)
    _LOG.warning(
        "Filter %s raised %s on candidate %s; the trial is rejected and the "
        "campaign continues (further %s/%s errors are logged at debug).",
        key[0],
        key[1],
        trial.index,
        key[0],
        key[1],
        exc_info=True,
    )


def evaluate_trial(
    trial: Trial,
    filters: Sequence[FilterStage],
) -> Record:
    """Evaluate one trial through an ordered filter chain.

    The chain stops at the first rejection, so a record never carries a
    verdict from a filter that did not run. This is the single-trial kernel
    shared by the in-process pipeline and by the filter actors of the runner.

    Args:
        trial: The generated trial.
        filters: Process B tiers, applied in order.

    Returns:
        Record: The trial with the verdicts that were evaluated.
    """
    verdicts: list[Verdict] = []
    first_failed: str | None = None
    for filt in filters:
        try:
            verdict = filt.evaluate(trial)
        except Exception as error:  # noqa: BLE001 - one bad trial must not
            # abort a campaign: a crashed relaxation (e.g. non-finite
            # coordinates from a very short anchor distance) is a rejection,
            # not a reason to stop the remaining candidates.
            _report_exception(filt, trial, error)
            verdict = Verdict(
                False,
                None,
                f"exception:{type(error).__name__}",
                None,
            )
        verdicts.append(verdict)
        if not verdict.accepted:
            first_failed = type(filt).__name__
            break
    return Record(
        trial=trial,
        verdicts=tuple(verdicts),
        first_failed=first_failed,
        result=verdicts[-1].detail if verdicts else None,
    )


class AdsorptionConfigurationPipeline:
    """Run a generator through an ordered filter chain.

    Attributes:
        generator: Process A.
        filters: Process B tiers, applied in order.
    """

    def __init__(
        self,
        generator: AdsorptionConfigurationGeneratorABC,
        filters: Sequence[FilterStage],
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
            yield evaluate_trial(trial, self.filters)

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
