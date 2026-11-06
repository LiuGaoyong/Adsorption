# ruff: noqa D103 E501
"""Contract tests of the A/B process abstractions."""

from collections.abc import Generator, Sequence

import pytest
from ase import Atoms

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    AdsorptionConfigurationGeneratorABC,
    Trial,
    Verdict,
)

from _helpers import CountingGenerator


def test_abstract_processes_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        AdsorptionConfigurationGeneratorABC()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        AdsorptionConfigurationFilterABC()  # type: ignore[abstract]


def test_trial_validation() -> None:
    atoms = Atoms("H2")
    Trial(atoms, index=1, n_total=1, meta={"n_substrate": 1})
    with pytest.raises(ValueError):
        Trial(atoms, index=0, n_total=1)
    with pytest.raises(ValueError):
        Trial(atoms, index=2, n_total=1)
    with pytest.raises(TypeError):
        Trial("H", index=1, n_total=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Trial(Atoms(), index=1, n_total=1)


def test_trial_requires_a_substrate_size() -> None:
    atoms = Atoms("H2O")
    with pytest.raises(ValueError):
        Trial(atoms, index=1, n_total=1)
    with pytest.raises(ValueError):
        Trial(atoms, index=1, n_total=1, meta={"n_substrate": 0})
    with pytest.raises(ValueError):
        Trial(atoms, index=1, n_total=1, meta={"n_substrate": 3})
    with pytest.raises(ValueError):
        Trial(atoms, index=1, n_total=1, meta={"n_substrate": "2"})


def test_trial_derives_the_adsorbate_count() -> None:
    trial = Trial(Atoms("H2O"), index=1, n_total=1, meta={"n_substrate": 2})
    assert trial.natoms_substrate == 2
    assert trial.natoms_adsorbate == 1


def test_verdict_defaults_and_validation() -> None:
    ok = Verdict(True)
    assert ok.accepted is True
    assert ok.value is None
    assert ok.reason == "ok"
    assert ok.detail is None
    with pytest.raises(ValueError):
        Verdict(True, reason="")
    with pytest.raises(TypeError):
        Verdict(1)  # type: ignore[arg-type]


class _EvenFilter(AdsorptionConfigurationFilterABC):
    def evaluate(self, trial: Trial) -> Verdict:
        accepted = trial.index % 2 == 0
        return Verdict(accepted, value=float(trial.index))


def test_generator_contract() -> None:
    generator = CountingGenerator(5)
    assert generator.n_candidates == 5
    assert len(generator) == 5
    assert generator.signature() == {"n": 5}
    trials = list(generator.generate())
    assert [trial.index for trial in trials] == [1, 2, 3, 4, 5]
    assert trials[2].meta["i"] == 3
    assert [trial.index for trial in generator.generate(2, 4)] == [2, 3, 4]


def test_filter_contract_keeps_order() -> None:
    generator = CountingGenerator(5)
    filt = _EvenFilter()
    assert filt.accepts(CountingGenerator(1).generate().__next__()) is False
    survivors = list(filt.filter(generator.generate()))
    assert [trial.index for trial in survivors] == [2, 4]


def test_generator_is_a_sequence() -> None:
    generator = CountingGenerator(5)
    assert isinstance(generator, Sequence)
    assert len(generator) == 5
    assert generator[0].index == 1
    assert generator[4].index == 5
    assert generator[-1].index == 5
    assert [trial.index for trial in generator[1:3]] == [2, 3]
    assert [trial.index for trial in generator[::2]] == [1, 3, 5]
    assert [trial.index for trial in generator] == [1, 2, 3, 4, 5]
    with pytest.raises(IndexError):
        generator[5]
    with pytest.raises(IndexError):
        generator[-6]


def test_each_iteration_starts_a_new_stream() -> None:
    """``iter()`` always starts a fresh stream (Sequence semantics)."""
    generator = CountingGenerator(3)
    assert [trial.index for trial in generator] == [1, 2, 3]
    assert [trial.index for trial in generator] == [1, 2, 3]
    first = iter(generator)
    second = iter(generator)
    assert first is not second
    assert next(first).index == 1
    assert next(second).index == 1
    # consuming a stream leaves the other one untouched
    assert [trial.index for trial in first] == [2, 3]
    assert [trial.index for trial in second] == [2, 3]


def test_the_generator_cursor_does_not_consume_the_streams() -> None:
    """Advancing ``next()`` leaves fresh iteration unaffected."""
    generator = CountingGenerator(4)
    assert next(generator).index == 1
    assert next(generator).index == 2
    assert [trial.index for trial in generator] == [1, 2, 3, 4]
    # a stream started before the cursor moves is also unaffected
    stream = iter(generator)
    assert next(generator).index == 3
    assert [trial.index for trial in stream] == [1, 2, 3, 4]


def test_generator_protocol_cursor() -> None:
    generator = CountingGenerator(3)
    assert isinstance(generator, Generator)
    assert next(generator).index == 1
    assert next(generator).index == 2
    assert generator.send(None).index == 3
    with pytest.raises(StopIteration):
        next(generator)
    # the cursor is exhausted, the sequence is not
    assert [trial.index for trial in generator] == [1, 2, 3]


def test_generator_cursor_can_be_closed() -> None:
    generator = CountingGenerator(3)
    assert next(generator).index == 1
    generator.close()
    with pytest.raises(StopIteration):
        next(generator)
