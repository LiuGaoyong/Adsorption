"""Process A: generators of adsorption configurations.

Two generators share the sites, the adsorbates and the budget:
:class:`AdsorptionConfigurationGenerator` enumerates the Cartesian product of
a direction grid, a rotation set and a distance array, and
:class:`RandomConfigurationGenerator` draws every candidate independently.
"""

from adsorption.generate.generator import AdsorptionConfigurationGenerator
from adsorption.generate.orientation import (
    OrientationShape,
    build_orientation,
    detect_shape,
)
from adsorption.generate.random import RandomConfigurationGenerator
from adsorption.generate.sampling import (
    DEFAULT_D_CONTACT_SCALE,
    DEFAULT_D_MAX,
    DEFAULT_D_MAX_RADIUS_SCALE,
    DEFAULT_D_MIN,
    DEFAULT_D_MIN_RADIUS_SCALE,
    DEFAULT_HEMISPHERE,
    DEFAULT_N_D,
    DEFAULT_N_OA,
    DEFAULT_N_ROT,
    BoundAdsorbate,
    Sampling,
)

__all__ = [
    "AdsorptionConfigurationGenerator",
    "BoundAdsorbate",
    "DEFAULT_D_CONTACT_SCALE",
    "DEFAULT_D_MAX",
    "DEFAULT_D_MAX_RADIUS_SCALE",
    "DEFAULT_D_MIN",
    "DEFAULT_D_MIN_RADIUS_SCALE",
    "DEFAULT_HEMISPHERE",
    "DEFAULT_N_D",
    "DEFAULT_N_OA",
    "DEFAULT_N_ROT",
    "RandomConfigurationGenerator",
    "Sampling",
    "OrientationShape",
    "build_orientation",
    "detect_shape",
]
