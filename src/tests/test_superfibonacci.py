# ruff: noqa D103 E501
"""Tests of the Super-Fibonacci SO(3) sequence."""

import numpy as np
import pytest
from scipy import stats

from adsorption.geometry import SuperFibonacciSO3


def test_validation() -> None:
    with pytest.raises(ValueError):
        SuperFibonacciSO3(0)
    with pytest.raises(ValueError):
        SuperFibonacciSO3(4, axis=np.zeros(3))


def test_quaternions_are_unit_and_deterministic() -> None:
    sequence = SuperFibonacciSO3(16)
    assert sequence.count() == 16
    for index in range(16):
        q = sequence.quaternion(index)
        assert q.shape == (4,)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert np.array_equal(q, sequence.quaternion(index))
    with pytest.raises(IndexError):
        sequence.quaternion(16)
    with pytest.raises(IndexError):
        sequence.quaternion(-1)


def test_rotations_are_proper() -> None:
    sequence = SuperFibonacciSO3(8)
    for index in range(8):
        quaternion = sequence.quaternion(index)
        assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
        image = sequence.image(index)
        assert np.linalg.norm(image) == pytest.approx(1.0)


def test_axis_marginal_is_uniform() -> None:
    sequence = SuperFibonacciSO3(2000)
    z = np.array([sequence.image(i)[2] for i in range(sequence.count())])
    p_value = stats.kstest(z, "uniform", args=(-1.0, 2.0)).pvalue
    assert p_value > 0.01


def test_rotation_angle_follows_haar() -> None:
    sequence = SuperFibonacciSO3(2000)
    angles = np.array(
        [
            2.0
            * np.arccos(
                np.clip(abs(sequence.quaternion(i)[0]), 0.0, 1.0)
            )
            for i in range(sequence.count())
        ]
    )
    cdf = (angles - np.sin(angles)) / np.pi
    p_value = stats.kstest(cdf, "uniform").pvalue
    assert p_value > 0.01


def test_no_nested_refinement() -> None:
    small = SuperFibonacciSO3(10)
    large = SuperFibonacciSO3(20)
    assert not np.allclose(small.quaternion(0), large.quaternion(0))
