# Benzene on Cu(111)

Reference: Liu et al., New J. Phys. 15 (2013) 053046. Details, DFT settings and the published configurations
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

- 3x3 cell, six layers, a = 3.572 A, two upper layers free.
- All eight starts of the paper are minima within 0.10 eV of each other: benzene is physisorbed on a nearly flat energy surface. This case tests how a flat landscape is sampled; recovering one particular site is not a sharp criterion.
- `manual.xyz` holds the paper's eight starts, the ring 3.0 A above the top layer.
