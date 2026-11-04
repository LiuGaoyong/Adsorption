# ruff: noqa D103 E501
"""Contract tests of the A/B process abstractions."""

from collections.abc import Iterator

import pytest
from ase import Atoms

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    AdsorptionConfigurationGeneratorABC,
    Trial,
    Verdict,
)


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


class _CountingGen(AdsorptionConfigurationGeneratorABC):
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def n_candidates(self) -> int:
        return self._n

    def generate(
        self, start: int = 1, stop: int | None = None
    ) -> Iterator[Trial]:
        last = self._n if stop is None else stop
        for index in range(start, last + 1):
            yield Trial(
                Atoms("H2"),
                index,
                self._n,
                {"i": index, "n_substrate": 1},
            )

    def signature(self) -> dict[str, object]:
        return {"n": self._n}


class _EvenFilter(AdsorptionConfigurationFilterABC):
    def evaluate(self, trial: Trial) -> Verdict:
        accepted = trial.index % 2 == 0
        return Verdict(accepted, value=float(trial.index))


def test_generator_contract() -> None:
    generator = _CountingGen(5)
    assert generator.n_candidates == 5
    assert len(generator) == 5
    assert generator.signature() == {"n": 5}
    trials = list(generator.generate())
    assert [trial.index for trial in trials] == [1, 2, 3, 4, 5]
    assert trials[2].meta["i"] == 3
    assert [trial.index for trial in generator.generate(2, 4)] == [2, 3, 4]


def test_filter_contract_keeps_order() -> None:
    generator = _CountingGen(5)
    filt = _EvenFilter()
    assert filt.accepts(_CountingGen(1).generate().__next__()) is False
    survivors = list(filt.filter(generator.generate()))
    assert [trial.index for trial in survivors] == [2, 4]
