from graphatoms.arrayapi import Array, ArrayNamespace, get_namespace


def cutoff_function(r: Array, rc: Array, ro: Array) -> Array:
    """Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    Taken from https://github.com/google/jax-md.

    """
    xp: ArrayNamespace = get_namespace(r, rc, ro)
    return xp.where(
        r < ro,
        1.0,
        xp.where(
            r < rc,
            (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3,
            0.0,
        ),
    )


def d_cutoff_function(r: Array, rc: Array, ro: Array) -> Array:
    """Derivative of smooth cutoff function wrt r.

    Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
    we need to multiply `2*r_ij`. This gives rise to the factor 2
    above, the `r_ij` is cancelled out by the remaining derivative
    `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
    """
    xp: ArrayNamespace = get_namespace(r, rc, ro)
    return xp.where(
        r < ro,
        0.0,
        xp.where(
            r < rc,
            6 * (rc - r) * (ro - r) / (rc - ro) ** 3,
            0.0,
        ),
    )
