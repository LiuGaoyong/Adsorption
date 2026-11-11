# Site runner examples

Twelve site folders, one per site. Each site contains one folder per
adsorbate:

```text
cluster_vertex_top/
├── O/
├── CO/
├── H2O/
├── CH4/
├── C6H6/
├── C2H6/
├── CH3CH2OH/
└── CH3OH/
```

Every adsorbate folder is one case. Its `run.yaml` is rendered at run time
from [`manifest.yaml`](manifest.yaml) and
[`run.template.yaml`](run.template.yaml); the generated file is ignored by
git. Running a case writes `records.jsonl`, `accepted.xyz` (the accepted
structures as an extended-XYZ trajectory) and `summary.json` under its
`results/` directory.

```bash
# one example
# from this directory: render the generated run.yaml files once
python ../build_cases.py manifest.yaml run.template.yaml --root .
cd cluster_vertex_top/CO
adsorption-run --config-path "$PWD" --config-name run

# all 96 examples, sequentially
RUNNER=adsorption-run ./run_all.sh

# only the examples that have no results yet
SKIP_EXISTING=1 RUNNER=adsorption-run ./run_all.sh
```

## Batch parameters

`run_all.sh` renders one `run.yaml` per case from
[`manifest.yaml`](manifest.yaml) and [`run.template.yaml`](run.template.yaml)
before running. The manifest stores the extxyz substrate and the core indices
of every site folder; the template stores the shared protocol, rotamer grid,
filter chain and budget.

```bash
./run_all.sh                    # render the cases, then run them
GENERATE_CASES=0 ./run_all.sh   # reuse the existing generated files
```

### NequIP tier

[`run.nequip.template.yaml`](run.nequip.template.yaml) replaces the
Lennard-Jones relaxation with a single-point Lennard-Jones screen and a
`NequIPOptimizedEnergyFilter`:

```text
GeometryFilter -> LennardJonesScreenFilter -> NequIP
```

It needs NequIP and a deployed model; the default is
`~/.local/nequip-oam-0.1/NequIP-OAM-S-0.1.nequip.pth`. The tiers relax the
full structure (`fix_substrate: false`, `fix_adsorbate_bonds: false`).
Duplicate rotamers of a flexible adsorbate are dropped when the ensemble is
resolved (`deduplicate: true`, the default), before any candidate is built.

```bash
# the default model is used automatically when it exists
# ~/.local/nequip-oam-0.1/NequIP-OAM-S-0.1.nequip.pth
./run_all.sh

# force the plain LJ chain
TEMPLATE=run.template.yaml ./run_all.sh

# or point at another model; NEQUIP_MODEL selects the NequIP template
NEQUIP_MODEL=/path/to/model.pth ./run_all.sh
```

For an in-process sweep (one scientific-stack import instead of one per
case) use the batch driver:

```bash
BATCH_DRIVER=1 STOP=8 ./run_all.sh
```

`parallel.chunk_factor` controls how many chunks are created per worker, and
`parallel.threads_per_worker` limits BLAS/Torch threads in each worker
(`0` disables the limit). NequIP examples use `chunk_factor: 2`.

`START`, `STOP`, `N_JOBS` and `CONTINUE_ON_ERROR` override the generated
configuration and the runner for a quick sweep:

```bash
START=1 STOP=8 N_JOBS=2 CONTINUE_ON_ERROR=1 ./run_all.sh
```


To add a case, create the site folder, add it to `manifest.yaml`, and add the
adsorbate SMILES to the manifest. `structures/cluster.xyz` and
`structures/slab.xyz` are the two substrates.

## Site images

`plot_sites.py` redraws the site geometry of every site folder with the
four-view helper and writes `<site>/site.png`:

```bash
python examples/0-SingleSite/plot_sites.py
python examples/0-SingleSite/plot_sites.py --only cluster_vertex_top
```

The highlighted atoms are the site core.

Every example applies `GeometryFilter(scale=1.0)` followed by
`LennardJonesEnergyFilter(e_cut: null)`, whose ceiling is the sum of the
isolated substrate and adsorbate Lennard-Jones energies. The runner resolves
that ceiling once before the candidate loop. The examples use a short LJ
relaxation budget (`max_steps: 10`, `fmax: 0.3`,
`reject_on_unconverged: false`) so the gallery stays runnable; the filter
defaults remain 100 steps and `fmax=0.01`. The distance bounds are size-aware.

The adsorbates use rigid rotamers: RDKit locates the rotatable bonds and ASE
sets the declared dihedrals. The examples set `conformers: rotamers`,
`dihedrals: 6` (six uniform points in `[0, 360)` per rotor) and
`max_rotamers: 64`; a molecule with no rotatable bond keeps a single
structure.
