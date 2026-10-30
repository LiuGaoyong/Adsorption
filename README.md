# Adsorption

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/adsorption.svg)](https://anaconda.org/conda-forge/adsorption)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/adsorption.svg)](https://anaconda.org/conda-forge/adsorption)
[![Pypi version](https://img.shields.io/pypi/v/adsorption)](https://pypi.org/project/adsorption/)
[![PyPI Downloads](https://static.pepy.tech/badge/adsorption)](https://pepy.tech/projects/adsorption)

Geometric pre-screening of molecular adsorption configurations on clusters and
periodic surfaces, built on [ASE](https://ase-lib.org).

The package places an adsorbate on a site defined by a core set of substrate
atoms. The placement is described by four quantities — the site centre `O`, the
approach direction `OA`, the molecular orientation `AB`, and the anchor
distance `d` — and the geometric stage never evaluates an energy. A downstream
ASE calculator (EMT, a machine-learned potential, or DFT) performs the optional
relaxation.

The method and its pre-registered validation protocol are described in
`paper-1/` (the live manuscript) and `paper/` (an earlier archived version of
the manuscript).

## Installation

```bash
pip install adsorption
```

Or with pixi:

```bash
pixi install
```

Requires Python 3.12–3.13, ASE, NumPy, SciPy, and
[graphatoms](https://github.com/LiuGaoyong/graphatoms).

## Quick start

```python
from ase.build import fcc111
from ase.calculators.emt import EMT
from adsorption import DirectAdsorption, RawAdsorption

slab = fcc111("Cu", size=(4, 4, 3), vacuum=12, orthogonal=True)
site = [0, 1, 2]  # core atoms; top = 1 atom, bridge = 2, hollow = 3

# Fibonacci-grid placement: scan the approach directions, then relax.
direct = DirectAdsorption(
    atoms=slab,
    core=site,
    calculator=EMT(),   # None keeps the unrelaxed structure
    nfibonacci=200,
    distance=2.0,       # optional; a hard-sphere heuristic is used by default
)
atoms, stage = direct(adsorbate="CO")
# stage: -1 not converged, 0/1/2 the number of converged stages

# Conventional placement: anchor on an atom (or the centre of mass).
raw = RawAdsorption(atoms=slab, core=site, calculator=None)
atoms, stage = raw(adsorbate="CO", adsorbate_index=0)
```

`DirectAdsorption` accepts `nfibonacci`, `distance`, `anchor` (`"com"`, an atom
index, or an explicit point), `scale`, `skin`, `delta_nbr`, and an optional
`seed`. Without a seed the placement is deterministic: the grid index defaults
to `0`; pass a seed only when stochastic index selection is wanted.

## Geometry primitives

The geometric layer is available directly, without building an adsorption
object:

```python
import numpy as np
from adsorption.geometry import (
    FibonacciGrid,
    admissible_directions,
    build_site,
    distance_array,
    resolve_anchor,
    shortest_arc_rotation,
)

site = build_site(slab, core=[0, 1, 2], scale=1.5, skin=0.5)
grid = admissible_directions(site, n=200, grid=FibonacciGrid())
print(site.center, site.radius, grid.n_oa, grid.fraction)

# d_min, d_max, step -> M anchor distances (paper Eq. 11)
d = distance_array(0.8, 3.0, 0.2)  # length 12

# Shortest-arc rotation maps a body axis onto a grid direction.
rotation = shortest_arc_rotation([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
anchor = resolve_anchor(atoms, anchor="com")
```

An `AdmissibleSet` keeps both the surviving unit vectors and their original
grid indices, so a candidate can be regenerated from its index alone.

## Relaxation

`AdsorptionOptimizeABC.optimize_adsorption` performs the two-stage protocol: the
substrate is frozen and the intra-adsorbate distances are constrained in stage
one, and the whole system is relaxed in stage two. Supply an ASE calculator to
enable it; without a calculator the unrelaxed structure is returned with stage
label `-1`.

`adsorption.common.optimize` also provides `optimize`, `call_dimer`, and
`call_neb` for plain optimization, dimer saddle-point search, and NEB/DyNEB
minimum-energy paths.

## Package layout

```
src/adsorption/
├── geometry/      # site, grids, admissibility, rotation, distance, placement
├── common/        # AdsorptionABC, Site (legacy wrapper), optimize()
├── interfaces/    # RawAdsorption, DirectAdsorption
├── pairwise/      # Lennard-Jones calculator (arrayff backend)
└── runner/        # campaign runner (protocol layer, under construction)
```

The `geometry` subpackage is calculator-free: nothing in it imports ASE
calculators, so candidate generation is engine-independent.

## Deprecations

The Torch/autodiff path (`DirectAdsorptionAD`, `Helper`, the Ray Tune runner,
and the `adsorption-tune` command) has been removed. `adsorption.common.Site`
and `Site.get_direction_grid` remain as wrappers around the `geometry` layer and
will be removed after the protocol layer lands.

## Roadmap

The fixed-budget protocol layer is being implemented on
`refactor/paper1-protocol-layer`: the κ index bijection, the distance array in
the public API, the Lennard-Jones steric screen, sidecar records, the campaign
CLI, and orientation sets on `S²`, `S² × S¹`, and `SO(3)`. The manuscript in
`paper-1/` is updated together with the code.

## Development

```bash
pixi run test        # pytest -s -vv
pytest src/tests     # or with an existing virtual environment
```

The test suite includes the geometric primitives (`src/tests/test_geometry.py`),
the adsorption interfaces, and the quaternion module.

## Citation and license

Code: <https://github.com/LiuGaoyong/Adsorption>, GPL-3.0-or-later. See
`paper-1/` for the method and the pre-registered evaluation protocol.
