from typing import TYPE_CHECKING

import array_api_compat

if TYPE_CHECKING:
    from array_api_compat.common._helpers import _ArrayApiObj as Array


def quaternion_apply(quaternion, point) -> Array:
    """Apply the rotation given by a quaternion to a 3D point.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    np = array_api_compat.array_namespace(quaternion, point)
    if point.shape(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = np.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]  # type: ignore


def standardize_quaternion(quaternions: Array) -> Array:
    """Convert a unit quaternion to a standard form.

    one in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return array_api_compat.array_namespace(quaternions).where(
        quaternions[..., 0:1] < 0,  # type: ignore
        -quaternions,  # type: ignore
        quaternions,  # type: ignore
    )


def quaternion_invert(quaternion: Array) -> Array:
    """Got the quaternion representing its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    np = array_api_compat.array_namespace(quaternion)
    scaling = np.asarray([1, -1, -1, -1])
    return quaternion * scaling


def quaternion_raw_multiply(a: Array, b: Array) -> Array:
    """Multiply two quaternions.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    np = array_api_compat.array_namespace(a, b)
    aw, ax, ay, az = np.unbind(a, -1)
    bw, bx, by, bz = np.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return np.stack((ow, ox, oy, oz), -1)


def random_quaternions(
    n: int,
    x: Array,
    dtype: str | None = None,
    device: str | None = None,
) -> Array:
    """Generate random quaternions representing rotations.

    Args:
        n: Number of quaternions in a batch to return.
        x: ...
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    np = array_api_compat.get_namespace(x)
    if isinstance(device, str):
        device = np.device(device)
    o = np.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(np.sqrt(s), o[:, 0])[:, None]  # type: ignore
    return o


def _copysign(a: Array, b: Array) -> Array:
    """Helper function for random quaternions generation.

    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """  #
    np = array_api_compat.get_namespace(a, b)
    signs_differ = (a < 0) != (b < 0)  # type: ignore
    return np.where(signs_differ, -a, a)  # type: ignore
