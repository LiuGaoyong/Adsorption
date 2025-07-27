import numpy as np
from numpy import typing as npt
from scipy.spatial.transform import Rotation as Rot


def rotate(
    points: npt.ArrayLike,
    rotation: Rot = Rot.random(),
    center: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Rotate 3D points by the provided rotation around a given center.

    Args:
        points (npt.ArrayLike): The 3D points to rotate.
        rotation (Rot, optional): The rotation. Defaults to Rot.random().
        center (npt.ArrayLike, optional): The center of the rotation.
            Defaults to None which means that rotation will take
                place around the geometry center of input points.

    Returns:
        np.ndarray: The points after rotation.
    """
    assert isinstance(rotation, Rot), (
        "Rotation must be an instance of scipy.spatial.transform.Rotation."
    )

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("Points must be an array of 3D points.")

    if center is None:
        center = points.mean(axis=0)
    else:
        center = np.asarray(center, dtype=float)
    assert (
        isinstance(center, np.ndarray)
        and center.ndim == 1
        and center.shape[0] == 3
    ), "Center must be a 3D vector."

    return rotation.apply(points - center) + center
