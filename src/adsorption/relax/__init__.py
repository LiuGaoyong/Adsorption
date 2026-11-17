"""Relaxation of placed structures with an interatomic potential.

This module is separate from the placement core on purpose. The core
(:mod:`adsorption.generate`, :mod:`adsorption.filters`,
:mod:`adsorption.runner`) places adsorbates and screens the placements with
geometric and Lennard-Jones criteria; it writes the survivors as an
extended-XYZ file. This module takes such a file and relaxes every structure
with a machine-learned potential. The two steps share nothing but the file,
so the relaxation can run on another machine.

What is done with the relaxed structures (labels, minima, basin recall) is
analysis and lives in :mod:`adsorption.analysis`; the ``adsorption-relax``
command exposes both steps (``run`` and ``recall``).

No potential is imported until a calculator is built.
"""

from adsorption.relax.calculators import build_calculator
from adsorption.relax.relax import (
    RelaxSettings,
    read_frames,
    relax_structure,
    run_relax,
)

__all__ = [
    "RelaxSettings",
    "build_calculator",
    "read_frames",
    "relax_structure",
    "run_relax",
]
