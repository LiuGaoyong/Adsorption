"""Default parameters of the geometric construction.

The values follow Table 1 of the manuscript in ``paper-1/``. They are
implementation defaults, not universal constants: the protocol layer records
every value that enters a campaign.
"""

from __future__ import annotations

DELTA_NBR: float = 0.3
"""Neighbour-detection tolerance in Angstrom."""

SCALE_SINGLE: float = 1.2
"""Direction-sphere scale factor for a single-core site."""

SCALE_MULTI: float = 1.5
"""Direction-sphere scale factor for a multi-core site."""

SKIN_SINGLE: float = 1.0
"""Steric margin in Angstrom for a single-core site."""

SKIN_MULTI: float = 0.5
"""Steric margin in Angstrom for a multi-core site."""

MAX_CORE_SIZE: int = 6
"""Largest supported core set."""

GOLDEN_RATIO: float = (1.0 + 5.0**0.5) / 2.0
"""The golden ratio used by the spherical Fibonacci lattice."""


def default_scale(core_size: int) -> float:
    """Return the default direction-sphere scale for a core set.

    Args:
        core_size: Number of atoms in the core set.

    Returns:
        float: ``SCALE_SINGLE`` for one core atom, ``SCALE_MULTI`` otherwise.
    """
    return SCALE_SINGLE if int(core_size) == 1 else SCALE_MULTI


def default_skin(core_size: int) -> float:
    """Return the default steric margin for a core set.

    Args:
        core_size: Number of atoms in the core set.

    Returns:
        float: ``SKIN_SINGLE`` for one core atom, ``SKIN_MULTI`` otherwise.
    """
    return SKIN_SINGLE if int(core_size) == 1 else SKIN_MULTI
