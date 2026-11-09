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
[`run.yaml.template`](run.yaml.template); the generated file is ignored by
git. Running a case writes `records.jsonl`, `accepted.xyz` (the accepted
structures as an extended-XYZ trajectory) and `summary.json` under its
`results/` directory.

```bash
# one example
# from this directory: render the generated run.yaml files once
python ../build_cases.py manifest.yaml run.yaml.template --root .
cd cluster_vertex_top/CO
adsorption-run --config-path "$PWD" --config-name run

# all 96 examples, sequentially
RUNNER=adsorption-run ./run_all.sh

# only the examples that have no results yet
SKIP_EXISTING=1 RUNNER=adsorption-run ./run_all.sh
```

## Batch parameters

`run_all.sh` renders one `run.yaml` per case from
[`manifest.yaml`](manifest.yaml) and [`run.yaml.template`](run.yaml.template)
before running. The manifest stores the extxyz substrate and the core indices
of every site folder; the template stores the shared protocol, rotamer grid,
filter chain and budget.

```bash
./run_all.sh                    # render the cases, then run them
GENERATE_CASES=0 ./run_all.sh   # reuse the existing generated files
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
