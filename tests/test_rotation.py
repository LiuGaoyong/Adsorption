from random import random

import pytest
from ase.build import molecule

from adsorption.rotation import Rot, rotate


def test_rotation() -> None:  # noqa: D103
    ben = molecule("C6H6")
    points = ben.positions + [random()] * 3
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
