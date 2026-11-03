# Adsorption

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/adsorption.svg)](https://anaconda.org/conda-forge/adsorption)
[![PyPI version](https://img.shields.io/pypi/v/adsorption)](https://pypi.org/project/adsorption/)
[![PyPI Downloads](https://static.pepy.tech/badge/adsorption)](https://pepy.tech/projects/adsorption)

Deterministic generation and geometric/energetic pre-screening of molecular
adsorption configurations on clusters and periodic surfaces, built on
[ASE](https://ase-lib.org).

The package is organised as two processes:

- **A — generation** (`AdsConfGenABC`): a fixed-budget, deterministic
  enumeration of the site `O`, the approach direction `OA` on a Fibonacci
  sphere, the molecular orientation on `SO(3)` (Super-Fibonacci quaternions),
  and the anchor distance `d`.
- **B — filtering** (`AdsConfFilterABC`): a cheap pairwise distance floor, a
  repulsive Lennard-Jones tier, and an optional relaxation tier.

Generation never imports an ASE calculator; the energy engine is injected
only in process B.

## Installation

```bash
pip install adsorption
```

Or with pixi:

```bash
pixi install
```

Requires Python 3.12–3.13, ASE, NumPy, SciPy, Hydra, and
[graphatoms](https://github.com/LiuGaoyong/graphatoms).

## Quick start

```python
from ase import Atoms
from ase.build import fcc111
from adsorption import AdsConfPipeline, GeometryFilter, SingleSiteGen
from adsorption.geometry import (
    FixedRotation,
    build_site,
    distance_array,
    resolve_anchor,
)

slab = fcc111("Cu", size=(3, 4, 3), vacuum=10, orthogonal=True)
oxygen = Atoms("O")
generator = SingleSiteGen(
    slab,
    build_site(slab, core=[0]),
    oxygen,
    anchor=resolve_anchor(oxygen, "com"),
    orientation=FixedRotation(),          # one atom: no orientation
    distances=distance_array(2.0, 3.0, 0.2),
    n_grid=50,
)
filt = GeometryFilter({29: 2.338, 8: 3.0})   # Cu and O sigma
pipeline = AdsConfPipeline(generator, [filt])

for record in pipeline.accepted():
    print(record.trial.index, record.trial.meta["distance"])
```

For a molecule, build the rotation set with
`build_orientation("auto", adsorbate, n=200)`: linear molecules use `S^2`,
planar and three-dimensional molecules use `SO(3)`.

## Hydra campaign

```bash
python -m adsorption.config                        # run.yaml defaults
python -m adsorption.config --multirun protocol.orientation.n=200,500
```

`run.yaml` composes `substrate`, `adsorbate` and `filter` groups; swappable
objects are `_target_` nodes instantiated through
`graphatoms.utils.parser.hydra_parse`. A run writes `records.jsonl` with one
verdict per candidate, and refuses to start when the candidate count exceeds
`budget.max_candidates`.

## The two processes

```text
AdsConfGenABC                       AdsConfFilterABC
├── SingleSiteGen   one site        ├── GeometryFilter   pairwise floor
├── MultiSiteGen    many sites      ├── LJFilter         repulsive energy
└── MultiAdsorbateGen  many         └── EnergyFilter     relaxation + E cut
    conformers (consumed)
```

`AdsConfPipeline` applies the filters in order and stops at the first
rejection. Every `Trial` carries its one-based index and budget; every
`Verdict` carries the measured value and the reason.

## Package layout

```text
src/adsorption/
├── process/       # Trial, Verdict, the two ABCs, pipeline
├── generate/      # SingleSiteGen, MultiSiteGen, MultiAdsorbateGen
├── filters/       # GeometryFilter, LJFilter, EnergyFilter
├── geometry/      # site, grids, SuperFibonacciSO3, rotation, placement
├── protocol/      # index, plan, sidecar, hashing
├── evaluation/    # single-point and two-stage relaxation
├── metrics/       # ranking, cascade, spherical, rotation metrics
└── config/        # Hydra dataclasses, run.yaml and config groups
```

## Deprecated

The previous three-layer `AdsorptionABC` hierarchy, the `admissible_directions`
filter, the `screen` and `runner` packages, and the pydantic protocol spec are
frozen under [`deprecated/`](deprecated/README.md). They are not imported,
tested, linted, or packaged.

## Development

```bash
pixi run test              # pytest -s -vv
pytest src/tests -m "not slow"
```

## Citation and license

Code: <https://github.com/LiuGaoyong/Adsorption>, GPL-3.0-or-later. See
`paper-1/` for the method and the pre-registered evaluation protocol.
