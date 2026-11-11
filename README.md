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

Requires Python 3.12–3.13, ASE, NumPy, SciPy, RDKit, Hydra, and
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
filt = GeometryFilter(sigma={29: 2.338, 8: 3.0})  # Cu and O sigma
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
a site source (`sites: auto` selects the symmetry-unique exposed sites — one
top, bridge and hollow core per local environment on a slab, the unique
GraphAtoms sites on a cluster; `sites: top_layer` keeps every top-layer atom;
a flat list is one core set; a list of lists declares several sites; a mapping can select the
GraphAtoms enumeration), an adsorbate source (the rotatable-bond detection
selects an ETKDG ensemble, otherwise the single adsorbate), and the sampling
protocol. Candidate intervals are evaluated in parallel with `joblib` (default
backend `loky`) or, when the optional `ray` package is installed, by Ray
filter actors behind a `ray.util.ActorPool`. The engine defaults to `auto`:
Ray when it can be imported, joblib otherwise; a single-job run stays in
the calling process. Install the optional backend
with `pip install 'adsorption[ray]'` (or `uv sync --extra ray`) and force one
engine with `parallel.engine=joblib` / `parallel.engine=ray`. With Ray,
`parallel.n_jobs` is the actor count and `parallel.threads_per_worker` is the
CPU reservation and thread budget of each actor. Each actor owns a private
copy of the filter chain and consumes the `Trial` objects yielded by the
generator; `records.jsonl` stays in candidate-index order regardless of the
engine.
`records.jsonl` holds one JSON line per candidate (by default only the
accepted ones, see `storage.records`), `accepted.db` is an ASE database with
one row per accepted structure (by default only the adsorbate atoms plus the
substrate size, see `storage.structures`), and `summary.json` holds the
campaign totals (including the resolved `engine` and `n_actors`). A run
refuses to start when the candidate count exceeds `budget.max_candidates`.

## The two processes

```text
AdsorptionConfigurationGeneratorABC     AdsorptionConfigurationFilterABC
└── AdsorptionConfigurationGenerator    ├── GeometryFilter              pairwise floor
    ├── sites: AutoSites                ├── LennardJonesScreenFilter    single-point LJ ceiling
    │           ExplicitSites           ├── LennardJonesEnergyFilter    relaxed LJ ceiling
    │           TopLayerSites           ├── EnergyFilter / OptimizedEnergyFilter
    │           GraphAtomsSites         │                               any ASE calculator
    └── adsorbates: SingleAdsorbate     └── NequIPOptimizedEnergyFilter MLP relaxation
                    ExplicitEnsemble
                    ETKDGEnsemble       TrialDeduplicator  (stages, not filters:
                    RotamerEnsemble     BasinDeduplicator   placed / relaxed structure)
```

`AdsorptionConfigurationPipeline` applies the stages in order and stops at the
first rejection. Every `Trial` carries its one-based index and budget; every
`Verdict` carries the measured value and the reason. `TrialDeduplicator`
rejects a placed structure whose bond-list hash was already accepted (the
hash is value-sorted, so it is invariant to the atom order and collapses
symmetry-equivalent placements); `BasinDeduplicator`, placed after a
relaxation tier, rejects a trial whose relaxed structure and energy repeat an
earlier basin. In a parallel campaign the runner repeats both checks globally
in the driver process, so the result does not depend on the worker count.

The sampling protocol has two site-dependent defaults. The lower bound of the
anchor-distance array is the hard-sphere contact distance of the site and
adsorbate (`d_min: null`), derived from the same KIM Lennard-Jones radii as
`GeometryFilter`, so no candidate starts inside the core; a number restores an
absolute `d_min`. With `hemisphere: true` only the approach directions on the
outward side of each site are enumerated. A relaxation tier stores the relaxed
structure of a survivor in `trial.meta["relaxed"]`; the runner persists that
structure (and `relaxed_energy`) in `accepted.db`, and a later tier can start
from it with `start_from: previous`. The isolated-adsorbate reference of
`e_cut: null` is the rigid conformer when the intramolecular bonds are fixed
and the gas-phase relaxed conformer otherwise (`adsorbate_reference`).

## Package layout

```text
src/adsorption/
├── base/          # Trial, Verdict, the two ABCs, pipeline, canonical_key
├── generate/      # AdsorptionConfigurationGenerator, Sampling, adsorbate
│                  # sources (Single/Explicit/ETKDG/Rotamer), orientation
├── filters/       # GeometryFilter, Lennard-Jones tiers, EnergyFilter,
│                  # OptimizedEnergyFilter, NequIPOptimizedEnergyFilter
├── geometry/      # sites, grids, SuperFibonacciSO3, rotations, placement
├── dedup.py       # TrialDeduplicator
├── runner/        # Hydra entry point, run.yaml, joblib / Ray engines
└── utils/         # RDKit helpers, four-view plotting
```

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
