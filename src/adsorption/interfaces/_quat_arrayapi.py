"""Array API compatible version of quaternion operations."""

import numpy as np
from graphatoms.arrayapi import Array, ArrayNamespace
from graphatoms.arrayapi import get_namespace as array_namespace


def quaternion_apply(quaternion: Array, point: Array) -> Array:
    """Apply the rotation given by a quaternion to a 3D point.

    Usual array API rules for broadcasting apply.

    Args:
        quaternion: Array of quaternions, real part first, of shape (..., 4).
        point: Array of 3D points of shape (..., 3).

    Returns:
        Array of rotated points of shape (..., 3).
    """
    xp: ArrayNamespace = array_namespace(quaternion, point)

    if point.shape[-1] != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")

    real_parts = xp.zeros(
        point.shape[:-1] + (1,),
        dtype=point.dtype,
        device=point.device,
    )
    point_as_quaternion = xp.concat((real_parts, point), axis=-1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def standardize_quaternion(quaternions: Array) -> Array:
    """Convert a unit quaternion to a standard form.

    i.e. get one in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as array of shape (..., 4).

    Returns:
        Standardized quaternions as array of shape (..., 4).
    """
    xp: ArrayNamespace = array_namespace(quaternions)
    return xp.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_invert(quaternion: Array) -> Array:
    """Given a quaternion representing rotation.

    i.e. get the quaternion representing its inverse.

    Args:
        quaternion: Quaternions as array of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, an array of quaternions of shape (..., 4).
    """
    xp: ArrayNamespace = array_namespace(quaternion)
    scaling = xp.asarray([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_raw_multiply(a: Array, b: Array) -> Array:
    """Multiply two quaternions.

    Usual array API rules for broadcasting apply.

    Args:
        a: Quaternions as array of shape (..., 4), real part first.
        b: Quaternions as array of shape (..., 4), real part first.
        xp: Array namespace (optional, inferred from inputs if not provided).

    Returns:
        The product of a and b, an array of quaternions shape (..., 4).
    """
    xp: ArrayNamespace = array_namespace(a, b)

    aw, ax, ay, az = xp.unstack(a, axis=-1)
    bw, bx, by, bz = xp.unstack(b, axis=-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return xp.stack((ow, ox, oy, oz), axis=-1)


def random_quaternions(n: int, *, xp: ArrayNamespace | None = None) -> Array:
    """Generate random quaternions representing rotations.

    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        xp: Array namespace (optional, uses numpy if not provided).

    Returns:
        Quaternions as array of shape (N, 4).
    """
    if xp is None:
        xp = np
    assert xp is not None

    # Generate random quaternions
    o = np.random.rand(n, 4)
    s = np.sum(o * o, axis=1)
    o = o / _copysign(np.sqrt(s), o[:, 0])[:, None]
    return xp.asarray(o, dtype=float)


def _copysign(a: Array, b: Array) -> Array:
    """Return an array where each element has the absolute value.

    taken from the corresponding element of a, with sign taken from
    the corresponding element of b. This is like the standard copysign
    floating-point operation, but is not careful about negative 0 and NaN.

    Args:
        a: source array.
        b: array whose signs will be used, of the same shape as a.
        xp: Array namespace (optional, inferred from inputs if not provided).

    Returns:
        Array of the same shape as a with the signs of b.
    """
    xp: ArrayNamespace = array_namespace(a, b)

    signs_differ = (a < 0) != (b < 0)
    return xp.where(signs_differ, -a, a)
