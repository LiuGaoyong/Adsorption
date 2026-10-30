"""The engine-agnostic evaluation layer.

The layer owns the calculator object, the single-point and two-stage
relaxation modes, the stage-1 constraint switches, and the structured result.
It knows nothing about the geometric construction or the protocol index.
"""

from __future__ import annotations

from .calculator import build_calculator
from .relaxation import Relaxation, Stage1Constraints, stage1_constraints
from .result import EvalResult

__all__ = [
    "EvalResult",
    "Relaxation",
    "Stage1Constraints",
    "build_calculator",
    "stage1_constraints",
]
