# ruff: noqa D103 E501
"""Tests of the A/B pipeline."""

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    Trial,
    Verdict,
)
from adsorption.base.pipeline import AdsorptionConfigurationPipeline

from _helpers import CountingGenerator


class _Even(AdsorptionConfigurationFilterABC):
    def evaluate(self, trial: Trial) -> Verdict:
        return Verdict(trial.index % 2 == 0, float(trial.index), "even")


class _AboveTwo(AdsorptionConfigurationFilterABC):
    def evaluate(self, trial: Trial) -> Verdict:
        return Verdict(trial.index > 2, float(trial.index), "gt2")


def test_pipeline_stops_at_first_failure() -> None:
    pipeline = AdsorptionConfigurationPipeline(CountingGenerator(5), [_Even(), _AboveTwo()])
    records = list(pipeline.run())
    assert [record.trial.index for record in records] == [1, 2, 3, 4, 5]
    assert records[0].first_failed == "_Even"
    assert len(records[0].verdicts) == 1
    assert records[1].first_failed == "_AboveTwo"
    assert len(records[1].verdicts) == 2
    assert records[3].first_failed is None


def test_pipeline_accepts() -> None:
    pipeline = AdsorptionConfigurationPipeline(CountingGenerator(6), [_Even(), _AboveTwo()])
    survivors = list(pipeline.accepted())
    assert [record.trial.index for record in survivors] == [4, 6]


def test_pipeline_index_interval() -> None:
    pipeline = AdsorptionConfigurationPipeline(CountingGenerator(6), [])
    records = list(pipeline.run(2, 4))
    assert [record.trial.index for record in records] == [2, 3, 4]
    assert all(record.first_failed is None for record in records)
