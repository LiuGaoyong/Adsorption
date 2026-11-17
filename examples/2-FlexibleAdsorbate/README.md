# Flexible-adsorbate runner examples

Flexible-adsorbate cases built on the same six systems as
`examples/1-MultiSite`. The layout mirrors `0-SingleSite` and `1-MultiSite`:
one folder per system, one folder per flexible adsorbate, and a generated
`run.yaml` in every adsorbate folder.

```text
examples/2-FlexibleAdsorbate/
├── README.md
├── gallery_index.csv
├── run_all.sh
├── plot_sites.py
├── sites/
├── structures/S6-disordered-Cu.xyz
├── S1-Cu-octahedron/
│   ├── n-hexane/run.yaml       # generated, ignored
│   └── 1,5-hexadiene/run.yaml  # generated, ignored
├── S2-Cu111/...
...
└── S6-disordered-Cu/...
```

## Adsorbates

| Adsorbate | SMILES | Rotatable bonds | Rotamers at `dihedrals: 6` |
|:--|:--|--:|--:|
| n-hexane | `CCCCCC` | 3 | 64 (capped from `6^3 = 216`) |
| 1,5-hexadiene | `C=CCCC=C` | 3 | 64 (capped from `6^3 = 216`) |

Both use the RDKit-detected rotatable bonds and ASE's dihedral rotation:
`conformers: rotamers`, `dihedrals: 6` means six uniform points in
`[0, 360)` per rotor, and `max_rotamers: 64` caps the Cartesian product.
The cap is recorded in each structure's provenance (`truncated: true`).

## Systems and sites

The six systems and substrates are the same as `examples/1-MultiSite`.
Each generated `run.yaml` uses `sites: {kind: graphatoms, ...}` with the same
per-system core caps as the multi-site gallery; the curated site metadata
remains in [`sites/`](sites/) for reference. See
[`gallery_index.csv`](gallery_index.csv) for the 12 system–adsorbate rows.

## Batch parameters

`run_all.sh` renders one `run.yaml` per case from
[`manifest.yaml`](manifest.yaml) and [`run.template.yaml`](run.template.yaml)
before running. The manifest stores each system's extxyz substrate and
GraphAtoms selection; the template stores the shared protocol, rotamer grid,
filter chain and budget. Generated `run.yaml` files are ignored by git.

```bash
./run_all.sh                    # render the cases, then run them
GENERATE_CASES=0 ./run_all.sh   # reuse the existing generated files
```

### Relaxation with a machine-learned potential

The examples stop at the Lennard-Jones tier. Every case writes its accepted
placements to `results/accepted.xyz` (`storage.xyz: true`): complete
extended-XYZ frames, substrate atoms first, with `n_substrate` and the
candidate index in the frame header. That file is the input of the separate
relaxation command, which can run on another machine:

```bash
adsorption-relax run results/accepted.xyz --config ../../../relax.yaml --out relaxed
```

[`../relax.yaml`](../relax.yaml) selects the potential and the optimizer
settings. The potential is an optional dependency
(`pip install 'adsorption[nequip]'`).

For an in-process sweep (one scientific-stack import instead of one per
case) use the batch driver:

```bash
BATCH_DRIVER=1 STOP=8 ./run_all.sh
```

`parallel.chunk_factor` controls how many chunks are created per worker, and
`parallel.threads_per_worker` limits BLAS/Torch threads in each worker
(`0` disables the limit).

`START`, `STOP`, `N_JOBS` and `CONTINUE_ON_ERROR` override the generated
configuration and the runner for a quick sweep:

```bash
START=1 STOP=8 N_JOBS=2 CONTINUE_ON_ERROR=1 ./run_all.sh
```


Substrates are in [`structures/`](structures/) as extxyz files; the two
adsorbates are SMILES strings in the manifest.

## Protocol

The paper-scale matrix is summarized in `examples/1-MultiSite`. These flexible
examples use a reduced grid to keep the conformer campaigns manageable:

```yaml
protocol:
  n_oa: 10
  n_rot: 10
  n_d: 6
  d_min: 0.8
  d_max: 3.0
```

The filter chain is `GeometryFilter(scale=1.0)` followed by
`LennardJonesEnergyFilter(e_cut: null)` with `max_steps: 10`, `fmax: 0.3`,
`reject_on_unconverged: false`.

## Running

```bash
# one system-adsorbate case
# from this directory: render the generated run.yaml files once
python ../build_cases.py manifest.yaml run.template.yaml --root .
cd S2-Cu111/n-hexane
adsorption-run --config-path "$PWD" --config-name run

# all twelve cases, sequentially
cd examples/2-FlexibleAdsorbate
RUNNER=adsorption-run ./run_all.sh

# only cases without results yet
SKIP_EXISTING=1 RUNNER=adsorption-run ./run_all.sh
```

## Site overview images

```bash
python examples/2-FlexibleAdsorbate/plot_sites.py
python examples/2-FlexibleAdsorbate/plot_sites.py --only S2-Cu111
```

## Notes

- The configs generate conformers from SMILES; no conformer structures are
  committed.
- S5 uses the `fcc211` Cu(211) slab written to extxyz with all three axes
  periodic and 15 A of vacuum along z, so the full filter chain runs on it.
- `1,5-hexadiene` is written as `1,5-hexadiene` in folder and config names.
