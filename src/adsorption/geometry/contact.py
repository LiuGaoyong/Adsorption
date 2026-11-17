"""Hard-sphere contact distances derived from the KIM Lennard-Jones radii.

The pairwise floor of the geometric screen and the lower bound of the
anchor-distance array share one table: the ``sigma`` of the OpenKIM universal
Lennard-Jones model, which is built from covalent radii
(``sigma_a = 2 R_a^cov / 2^(1/6)``). The floor of a pair is
``scale * sigma_ab`` with the Lorentz-Berthelot mean
``sigma_ab = (sigma_a + sigma_b) / 2``; at ``scale = 1`` it is the
Lennard-Jones zero crossing, ``0.89 (R_a^cov + R_b^cov)``.

:func:`contact_distance` turns that floor into an anchor-centre distance for a
site: the distance along the site normal at which a point probe carrying the
smallest adsorbate floor touches the core atoms. For a top site the core atom
sits at the centre and the contact distance is the floor itself; for a bridge
or hollow site the core atoms lie at an in-plane offset ``rho_k`` from the
centre, so the probe touches them lower, at ``sqrt(floor^2 - rho_k^2)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from arrayff.potentials.pairwise.lj_kim import KIM_LJ_PARAMETERS

from adsorption.geometry.mic import displacements_mic
from adsorption.geometry.site import SiteGeometry

__all__ = ["DEFAULT_SIGMA", "contact_distance", "pair_floor"]

DEFAULT_SIGMA: Mapping[int, float] = {
    number: parameters[2] for number, parameters in KIM_LJ_PARAMETERS.items()
}
"""Per-element Lennard-Jones ``sigma`` in Angstrom (KIM universal model)."""


def pair_floor(
    left: int,
    right: int,
    *,
    scale: float = 1.0,
    sigma: Mapping[int, float] | None = None,
) -> float:
    """Return the hard-sphere floor of one element pair.

    Args:
        left: Atomic number of the first atom.
        right: Atomic number of the second atom.
        scale: Coefficient of the Lorentz-Berthelot ``sigma_ab``.
        sigma: Per-element ``sigma`` table; the KIM table when ``None``.

    Returns:
        float: ``scale * (sigma_left + sigma_right) / 2`` in Angstrom.

    Raises:
        KeyError: If an element has no ``sigma``.
        ValueError: If ``scale`` is not positive.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}.")
    table = DEFAULT_SIGMA if sigma is None else sigma
    try:
        left_sigma = float(table[int(left)])
        right_sigma = float(table[int(right)])
        return 0.5 * float(scale) * (left_sigma + right_sigma)
    except KeyError as error:
        raise KeyError(f"No sigma for Z={error.args[0]}.") from error


def contact_distance(
    site: SiteGeometry,
    adsorbate_numbers: Iterable[int],
    *,
    scale: float = 1.0,
    sigma: Mapping[int, float] | None = None,
) -> float:
    """Return the anchor-centre distance at which the probe touches the core.

    For every core atom ``k`` at in-plane offset ``rho_k = |R_k - O_s|`` the
    probe clears the atom along the site normal once
    ``d >= sqrt(max(f_k^2 - rho_k^2, 0))``, where ``f_k`` is the smallest
    floor between ``k`` and any heavy adsorbate element. Hydrogen is ignored
    unless the adsorbate consists of hydrogen only: the anchor sits on the
    heavy-atom skeleton, and a hydrogen floor would place the skeleton inside
    the core. The contact distance is the largest of these values, so the
    probe clears every core atom. Core atoms are assumed to lie in the plane
    through ``O_s`` perpendicular to the normal, which is exact for coplanar
    cores and conservative otherwise.

    Args:
        site: The site geometry.
        adsorbate_numbers: Atomic numbers of the adsorbate.
        scale: Coefficient of the pairwise floor.
        sigma: Per-element ``sigma`` table; the KIM table when ``None``.

    Returns:
        float: The contact distance in Angstrom.

    Raises:
        ValueError: If the adsorbate has no atoms.
    """
    numbers = sorted({int(value) for value in adsorbate_numbers})
    if len(numbers) == 0:
        raise ValueError("adsorbate_numbers must not be empty.")
    heavy = [number for number in numbers if number > 1]
    if heavy:
        numbers = heavy
    offsets = displacements_mic(
        np.asarray(site.core_positions, dtype=float)
        - np.asarray(site.center, dtype=float),
        site.cell,
        site.pbc,
    )
    rho = np.linalg.norm(offsets, axis=1)
    distance = 0.0
    for core_number, offset in zip(site.core_numbers, rho):
        floor = min(
            pair_floor(int(core_number), number, scale=scale, sigma=sigma)
            for number in numbers
        )
        distance = max(distance, float(np.sqrt(max(floor**2 - offset**2, 0.0))))
    return distance
