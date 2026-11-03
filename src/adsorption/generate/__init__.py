"""Process A: deterministic generators of adsorption configurations."""

from .multi_adsorbate import Conformer, MultiAdsorbateGen
from .multi_site import MultiSiteGen
from .single_site import SingleSiteGen

__all__ = [
    "Conformer",
    "MultiAdsorbateGen",
    "MultiSiteGen",
    "SingleSiteGen",
]

from .orientation import Shape, build_orientation, detect_shape

__all__ += ["Shape", "build_orientation", "detect_shape"]
