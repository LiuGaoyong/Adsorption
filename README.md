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
  single-point Lennard-Jones screen, and a rigid-body Lennard-Jones relaxation.

The core stops there: it places adsorbates and screens the placements with
geometric and Lennard-Jones criteria. Relaxing the survivors with a
machine-learned potential is a separate module with its own command,
`adsorption-relax`; the two steps share one extended-XYZ file and can run on
different machines (see [Relaxation](#relaxation-with-a-machine-learned-potential)).

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
campaign totals (including the resolved `engine` and `n_actors`). With
`storage.xyz: true` the accepted structures are also written to
`accepted.xyz` as complete extended-XYZ frames. A run
refuses to start when the candidate count exceeds `budget.max_candidates`.

## Relaxation with a machine-learned potential

```bash
adsorption-run storage.xyz=true outputs=./runs/demo      # placement + LJ
adsorption-relax run runs/demo/accepted.xyz --out runs/demo/relaxed \
    --calculator nequip --model-path model.nequip.pth --free-layers 2 \
    --engine ray --n-jobs 16 --cpus-per-actor 4
adsorption-relax recall --run fibonacci/n200=runs/demo/relaxed --out runs/demo/analysis
```

`accepted.xyz` is self-contained: every frame holds the substrate atoms first
and the adsorbate after them, and its header carries `n_substrate`, the
candidate index, the site and the conformer. `adsorption-relax run` relaxes
every frame (`nequip`, or any `module:factory` that
returns an ASE calculator), writes `relaxed.xyz` and `relax.jsonl`, and
labels each result `intact`, `dissociated`, `desorbed` or `reconstructed`;
convergence is recorded separately. Every frame is kept, usable or not. A run
is split with `--start/--stop` and resumes from `relax.jsonl`.
Frames are relaxed in the calling process, by joblib workers, or by Ray
actors behind a `ray.util.ActorPool` (`--engine ray`): `--n-jobs` is the
actor count and `--cpus-per-actor` the CPUs each actor reserves, which is
also its Torch thread budget; `--ray-address` joins a running cluster.
`adsorption-relax recall` (the code is in `adsorption.analysis`) pools one or more runs, groups the relaxed
structures into minima with a tolerance-based distance fingerprint, and
reports the basin recall of every run and which published configurations
(`--references reference.yaml`, geometric criteria) were recovered. NequIP
is an optional dependency: `pip install 'adsorption[nequip]'`.

## The two processes

```text
AdsorptionConfigurationGeneratorABC     AdsorptionConfigurationFilterABC
└── AdsorptionConfigurationGenerator    ├── GeometryFilter              pairwise floor
    ├── sites: AutoSites                ├── LennardJonesScreenFilter    single-point LJ ceiling
    │           ExplicitSites           ├── LennardJonesEnergyFilter    relaxed LJ ceiling
    │           TopLayerSites           └── EnergyFilter / OptimizedEnergyFilter
    │           GraphAtomsSites                                         bases of the LJ tiers
    └── adsorbates: SingleAdsorbate
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
├── filters/       # GeometryFilter, Lennard-Jones tiers and their bases,
│                  # FixRigidBody
├── geometry/      # sites, grids, SuperFibonacciSO3, rotations, placement
├── dedup.py       # TrialDeduplicator, BasinDeduplicator
├── runner/        # adsorption-run: Hydra entry point, run.yaml, engines
├── relax/         # adsorption-relax: calculators and the relaxation of
│                  # extxyz frames (separate from the core)
├── analysis/      # no energy is evaluated here:
│                  #   light   geometric diagnostics (adsorption-light)
│                  #   labels  intact / dissociated / desorbed / reconstructed
│                  #   basins  which relaxed structures are one minimum
│                  #   recall  basin recall, published configurations
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

Code: <https://github.com/LiuGaoyong/Adsorption>, GPL-3.0-or-later.
