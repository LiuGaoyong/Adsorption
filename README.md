# Adsorption

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/adsorption.svg)](https://anaconda.org/conda-forge/adsorption)
[![PyPI version](https://img.shields.io/pypi/v/adsorption)](https://pypi.org/project/adsorption/)
[![PyPI Downloads](https://static.pepy.tech/badge/adsorption)](https://pepy.tech/projects/adsorption)

Deterministic generation and geometric/energetic pre-screening of molecular
adsorption configurations on clusters and periodic surfaces, built on
[ASE](https://ase-lib.org).

The package is organised as two processes:

- **A — generation** (`AdsorptionConfigurationGeneratorABC`): a fixed-budget, deterministic
  enumeration of the site `O`, the approach direction `OA` on a deterministic
  near-uniform spherical point set (the spherical Fibonacci lattice by default),
  the molecular orientation on `SO(3)` (Super-Fibonacci quaternions),
  and the anchor distance `d`.
- **B — filtering** (`AdsorptionConfigurationFilterABC`): a cheap pairwise distance floor, a
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
from adsorption import AdsorptionConfigurationPipeline, GeometryFilter
from adsorption import AdsorptionConfigurationGenerator

slab = fcc111("Cu", size=(3, 4, 3), vacuum=10, orthogonal=True)
oxygen = Atoms("O")
generator = AdsorptionConfigurationGenerator(
    slab,
    0,                     # adsorption centre: atom index 0
    oxygen,                # adsorbate
    n_oa=50,               # approach directions
    n_rot=1,               # orientations (a single atom has none)
    n_d=6,                 # anchor distances
    d_min=2.0,
    d_max=3.0,
)
filt = GeometryFilter({29: 2.338, 8: 3.0})  # Cu and O sigma
pipeline = AdsorptionConfigurationPipeline(generator, [filt])

for record in pipeline.accepted():
    print(record.trial.index, record.trial.meta["distance"])
```

For a molecule, build the rotation set with
`build_orientation("auto", adsorbate, n=200)`: linear molecules use `S^2`,
planar and three-dimensional molecules use `SO(3)`.

## Campaign runner

```bash
adsorption-run                         # Hydra entry point, packaged run.yaml
python -m adsorption.runner            # same entry point
adsorption-run parallel.n_jobs=4       # Hydra config override
adsorption-run start=1 stop=100 outputs=./runs/demo
```

`run.yaml` is a **single** file next to the runner package
(`src/adsorption/runner/run.yaml`): it declares the
substrate, the adsorbate, the site declaration, the generation protocol, the
filter chain and the budget. Swappable objects are `_target_` nodes
instantiated with `hydra.utils.instantiate`.

The runner builds one `AdsorptionConfigurationGenerator` from three inputs:
a site source (`sites: auto` selects every top-layer atom; a flat list is one
core set; a list of lists declares several sites; a mapping can select the
GraphAtoms enumeration), an adsorbate source (the rotatable-bond detection
selects an ETKDG ensemble, otherwise the single adsorbate), and the sampling
protocol. Candidate intervals are evaluated in parallel with `joblib` (default
backend `loky`);
`records.jsonl` holds one JSON line per candidate, `accepted.xyz` holds the
accepted structures as one extended-XYZ trajectory, and `summary.json` holds
the campaign totals. A run refuses to start when the candidate count exceeds
`budget.max_candidates`.

## The two processes

```text
AdsConfGenABC                          AdsConfFilterABC
└── AdsorptionConfigurationGenerator    ├── GeometryFilter   pairwise floor
    ├── sites: ExplicitSites            ├── LJFilter         repulsive energy
    │           TopLayerSites           └── EnergyFilter     relaxation + E cut
    │           GraphAtomsSites
    └── adsorbates: SingleAdsorbate
                    ExplicitEnsemble
                    ETKDGEnsemble
                    RotamerEnsemble
```

`AdsConfPipeline` applies the filters in order and stops at the first
rejection. Every `Trial` carries its one-based index and budget; every
`Verdict` carries the measured value and the reason.

## Package layout

```text
src/adsorption/
├── process/       # Trial, Verdict, the two ABCs, pipeline
├── generate/      # AdsorptionConfigurationGenerator, Sampling, orientation
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

The 96 single-site examples live in `examples/0-SingleSite/<site>/<adsorbate>/`;
the six-system multi-site gallery is in `examples/1-MultiSite/`, and the
flexible-adsorbate gallery (n-hexane and 1,5-hexadiene) is in
`examples/2-FlexibleAdsorbate/`. Each folder has its own `run_all.sh`.

## Citation and license

Code: <https://github.com/LiuGaoyong/Adsorption>, GPL-3.0-or-later. See
`paper-1/` for the method and the pre-registered evaluation protocol.
