# ruff: noqa D103 E501
"""Tests of the Lennard-Jones screen and the screening cascade."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.protocol import (
    AdsorbateSpec,
    DistanceSpec,
    OrientationSpec,
    Protocol,
    SiteSpec,
    prepare_campaign,
)
from adsorption.screen import (
    Cascade,
    CascadeRecord,
    LJParameters,
    RepulsiveScreen,
)


def parameters(**overrides) -> LJParameters:
    values = {
        "epsilon": {29: 1.0, 6: 1.0, 8: 1.0},
        "sigma": {29: 2.0, 6: 2.0, 8: 2.0},
        "r_cut": 5.0,
        "r_min": 1.0,
        "e_cut": 1.0e9,
    }
    values.update(overrides)
    return LJParameters(**values)


def dimer(distance: float) -> Atoms:
    return Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]],
        cell=np.zeros((3, 3)),
        pbc=False,
    )


def test_analytic_dimer() -> None:
    screen = RepulsiveScreen(parameters())
    decision = screen.evaluate(dimer(2.5))
    expected = 4.0 * (2.0 / 2.5) ** 12
    assert decision.n_pairs == 1
    assert decision.e_rep == pytest.approx(expected)
    assert decision.r_close == pytest.approx(2.5)
    assert decision.passed


def test_thresholds_reject() -> None:
    close = RepulsiveScreen(parameters(r_min=3.0)).evaluate(dimer(2.5))
    assert close.r_close < 3.0
    assert not close.passed

    energy = RepulsiveScreen(parameters(e_cut=1.0e-6)).evaluate(dimer(2.5))
    assert energy.e_rep > 1.0e-6
    assert not energy.passed


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        parameters(epsilon={29: 0.0})
    with pytest.raises(ValueError):
        parameters(sigma={29: -1.0})
    with pytest.raises(ValueError):
        parameters(r_cut=0.0)
    with pytest.raises(ValueError):
        parameters(r_min=-1.0)
    with pytest.raises(ValueError):
        parameters(epsilon={29: 1.0}, sigma={6: 1.0})
    with pytest.raises(KeyError):
        table = LJParameters(
            epsilon={6: 1.0},
            sigma={6: 2.0},
            r_cut=5.0,
            r_min=1.0,
            e_cut=1.0e9,
        )
        RepulsiveScreen(table).evaluate(dimer(2.5))


def test_pair_mixing() -> None:
    table = LJParameters(
        epsilon={6: 4.0, 8: 9.0},
        sigma={6: 2.0, 8: 4.0},
        r_cut=5.0,
        r_min=1.0,
        e_cut=1.0e9,
    )
    epsilon, sigma = table.pair(6, 8)
    assert epsilon == pytest.approx(6.0)
    assert sigma == pytest.approx(3.0)
    assert table.pair(8, 6) == pytest.approx((epsilon, sigma))


def test_from_kim() -> None:
    table = LJParameters.from_kim([29], r_min=1.0, e_cut=10.0)
    assert set(table.epsilon) == {29}
    assert table.source.startswith("OpenKIM")
    assert table.r_cut > 0.0
    with pytest.raises(KeyError):
        LJParameters.from_kim([119], r_min=1.0, e_cut=10.0)


def test_periodic_self_images_are_counted_once() -> None:
    atom = Atoms(
        "Cu1",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3) * 2.0,
        pbc=True,
    )
    decision = RepulsiveScreen(
        parameters(r_cut=2.5, r_min=0.0, e_cut=1.0e9)
    ).evaluate(atom)
    # six face images at 2.0 A; the 12 edge images at 2.83 A are outside
    assert decision.n_pairs == 6
    expected = 6.0 * 4.0 * (2.0 / 2.0) ** 12
    assert decision.e_rep == pytest.approx(expected)


def test_non_periodic_pairs_are_unique() -> None:
    atoms = Atoms(
        "Cu4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        cell=np.zeros((3, 3)),
        pbc=False,
    )
    decision = RepulsiveScreen(
        parameters(r_cut=3.0, r_min=0.0, e_cut=1.0e9)
    ).evaluate(atoms)
    assert decision.n_pairs == 6
    assert decision.r_close == pytest.approx(2.0)


def test_screen_never_ranks() -> None:
    screen = RepulsiveScreen(parameters())
    good = screen.evaluate(dimer(2.5))
    bad = screen.evaluate(dimer(1.5))
    assert bad.e_rep > good.e_rep
    assert bad.r_close < good.r_close
    assert good.passed
    # the screen has no ordering API: the verdict is a boolean only
    assert set(vars(good)) == {
        "e_rep",
        "e_rep_adsorbate",
        "r_close",
        "passed",
        "n_pairs",
    }


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def campaign(slab: Atoms):
    protocol = Protocol(
        n=16,
        orientation=OrientationSpec(n_ab=2),
        distances=DistanceSpec(d_min=1.0, d_max=2.0, step=0.5),
    )
    return prepare_campaign(
        slab,
        AdsorbateSpec(reference=molecule("CO")),
        [SiteSpec(core=(0,))],
        protocol,
    )


class Recorder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, atoms: Atoms) -> str:
        self.calls.append(len(atoms))
        return "evaluated"


def test_cascade_rejects_before_evaluation(campaign) -> None:
    recorder = Recorder()
    screen = RepulsiveScreen(parameters(r_min=100.0, e_cut=1.0e9))
    assert len(campaign) > 0
    records = list(Cascade(screen, recorder).run(campaign))
    assert len(records) == len(campaign)
    assert all(isinstance(record, CascadeRecord) for record in records)
    assert all(not record.decision.passed for record in records)
    assert all(record.evaluation is None for record in records)
    assert recorder.calls == []
    assert [record.candidate.kappa for record in records] == list(
        range(1, len(campaign) + 1)
    )


def test_cascade_evaluates_survivors(campaign) -> None:
    recorder = Recorder()
    screen = RepulsiveScreen(parameters(r_min=0.0, e_cut=1.0e12))
    records = list(Cascade(screen, recorder).run(campaign))
    assert all(record.decision.passed for record in records)
    assert all(record.evaluation == "evaluated" for record in records)
    assert len(recorder.calls) == len(campaign)
    assert all(count == len(campaign.atoms) + 2 for count in recorder.calls)


def test_cascade_without_evaluator(campaign) -> None:
    screen = RepulsiveScreen(parameters(r_min=0.0, e_cut=1.0e12))
    records = list(Cascade(screen).run(campaign))
    assert all(record.evaluation is None for record in records)


def test_adsorbate_diagnostic(campaign) -> None:
    screen = RepulsiveScreen(parameters(r_min=0.0, e_cut=1.0e12))
    decision = screen.evaluate(campaign.support(1).atoms, n_substrate=len(campaign.atoms))
    assert decision.e_rep_adsorbate is not None
    assert 0.0 <= decision.e_rep_adsorbate <= decision.e_rep
