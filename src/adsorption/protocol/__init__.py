"""The fixed-budget protocol layer.

The layer turns the geometric primitives into a deterministic candidate
stream: a protocol specification, a prepared campaign, the mixed-radix index
of Eqs. (15)-(16), the streaming enumerator, and the sidecar record. It does
not evaluate energies and it does not import an ASE calculator.
"""

from __future__ import annotations

from .campaign import Campaign, Candidate, prepare_campaign
from .hashing import parameter_hash, stable_hash
from .index import Event
from .plan import BudgetPlan, SiteVariant
from .sidecar import SidecarRecord, SidecarWriter, read_sidecar
from .spec import (
    AdsorbateSpec,
    ConformerSpec,
    DistanceSpec,
    FilterSpec,
    GridSpec,
    OrientationSpec,
    Protocol,
    SiteSpec,
)

__all__ = [
    "AdsorbateSpec",
    "BudgetPlan",
    "Campaign",
    "Candidate",
    "ConformerSpec",
    "DistanceSpec",
    "Event",
    "FilterSpec",
    "GridSpec",
    "OrientationSpec",
    "Protocol",
    "SidecarRecord",
    "SidecarWriter",
    "SiteSpec",
    "SiteVariant",
    "parameter_hash",
    "prepare_campaign",
    "read_sidecar",
    "stable_hash",
]
