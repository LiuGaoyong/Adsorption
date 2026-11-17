# ruff: noqa D103 E501
"""Tests of the orientation-manifold selection."""

import pytest
from ase import Atoms
from ase.build import molecule

from adsorption.generate import build_orientation, detect_shape
from adsorption.geometry import SuperFibonacciSO3, ShortestArcSet


def test_detect_shape() -> None:
    assert detect_shape(Atoms("He")) == "fixed"
    assert detect_shape(molecule("CO")) == "s2"
    assert detect_shape(molecule("H2O")) == "so3"
    assert detect_shape(molecule("CH4")) == "so3"
    assert detect_shape(molecule("C6H6")) == "so3"


def test_explicit_kind() -> None:
    adsorbate = molecule("CO")
    kind, orientation = build_orientation("fixed", adsorbate, n=8)
    assert kind == "fixed"
    assert orientation.count() == 1
    kind, orientation = build_orientation("s2", adsorbate, n=8)
    assert kind == "s2"
    assert isinstance(orientation, ShortestArcSet)
    kind, orientation = build_orientation("so3", adsorbate, n=8)
    assert kind == "so3"
    assert isinstance(orientation, SuperFibonacciSO3)


def test_auto() -> None:
    kind, orientation = build_orientation("auto", molecule("H2O"), n=8)
    assert kind == "so3"
    assert orientation.count() == 8


def test_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_orientation("maybe", molecule("CO"), n=8)  # type: ignore[arg-type]
