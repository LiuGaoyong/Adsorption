# CO on Cu(211)

Reference: Ozbek, Turk. J. Chem. 46 (2022) 1199. Details, DFT settings and the published configurations
are in [`reference.yaml`](reference.yaml).

| File | Content |
|:--|:--|
| `reference.yaml` | source, surface model, published configurations and their geometric criteria |
| `substrate.xyz` | the slab, atoms tagged by layer (1 is the outermost) |
| `sites.yaml` | the site cores used for placement |
| `relax.yaml`, `relax.sbatch` | options and slurm template of `adsorption-relax run` |
| `recall.sh` | basin recall of the relaxed runs (`adsorption-relax recall`) |
| `manual.xyz` | hand-placed starting geometries in the manner of the paper |
| `runs/deterministic/<rung>/` | the Cartesian-product campaign: `run.yaml`, `results/accepted.xyz` |
| `runs/random/s<seed>/<rung>/` | the independent random campaign of one seed |

## Notes

- One terrace period times four atoms along the step: 0.25 ML with one CO, the paper's lowest coverage. a = 3.643 A.
- ASE's `fcc211` slab with five layers is used; the paper describes five layers with six at the step edge, so the bottom of the slab may differ in detail. `free_layers: 9` frees all but the six lowest height levels, the equivalent of the paper's two frozen bulk layers.
- Substrate tags: 1 step-edge row, 2 middle terrace row, 3 row at the foot of the step. Eight site cores are used: three top sites, the step-edge bridge and four three-fold hollows of the terrace.
- Step-edge top (T1), step-edge bridge (B) and terrace top (T2) are matched individually. The four published hollows are grouped into two references (with or without a step-edge atom), because their assignment was read from the text only; fcc and hcp are not told apart, and a hollow at the foot of the step also touches the edge atom of the terrace above.
- `manual.xyz` holds one upright CO per site core, carbon down along the site normal, 1.9 A clear of every Cu atom; the paper prints no starting geometry.
- The paper uses PBE without dispersion.
- `relax.sbatch` also holds the ablation with the Lennard-Jones pre-relaxation (`--lj-prerelax`).
