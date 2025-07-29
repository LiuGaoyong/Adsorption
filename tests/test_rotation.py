import numpy as np
import pytest
from ase.build import molecule

from adsorption.rotation import Rot, kabsch, new_rotation_from, rotate


def test_rotation() -> None:  # noqa: D103
    ben = molecule("C6H6")
    points = ben.positions + np.random.random(3)
    center = points.mean(axis=0)
    print(f"Points center: {center}")

    rotation = Rot.random()
    print(f"Random rotation:\n{rotation.as_quat()}")

    rotated = rotate(points, rotation, center=[0, 0, 0])
    print(f"Rotated points around zero point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) != center

    rotated = rotate(points, rotation, center)
    print(f"Rotated points around center point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) == center

    # Default center point as the geometry center:
    rotated = rotate(points, rotation)
    print(f"Rotated points around center point:\n{rotated}")
    assert pytest.approx(rotated.mean(axis=0)) == center


def test_kabsch() -> None:  # noqa: D103
    B = molecule("C6H6").positions + np.random.random(3)
    R, T = Rot.random(), np.random.random(3)

    A = rotate(B, R) + T
    assert pytest.approx((A - T).mean(axis=0)) == B.mean(axis=0)

    R0, T0, rmsd = kabsch(A, B)
    assert isinstance(T0, np.ndarray)
    assert pytest.approx(T0) == T
    assert isinstance(R0, Rot)
    assert pytest.approx(R0.as_matrix()) == R.as_matrix()
    assert rmsd <= 1e-5
    assert pytest.approx(rotate(B, R0) + T0) == A


def test_new_rotation_from() -> None:  # noqa: D103
    A, B = np.random.rand(2, 3)
    A /= np.linalg.norm(A)
    B /= np.linalg.norm(B)
    print(A, B)

    R = new_rotation_from(A, B, C=[0, 0, 0])
    assert pytest.approx(R.apply(B), abs=1e-7) == A
