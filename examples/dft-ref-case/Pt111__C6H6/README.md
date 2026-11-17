# Benzene on Pt(111)

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

- 3x3 cell, six layers, a = 3.979 A (the paper's PBE+vdW-surf lattice constant), two upper layers free.
- Five minima are published (bri30, hcp0, fcc0, bri0, atop0). Each is matched by the site under the ring centre, the azimuth of the C-C bonds against the metal rows, and a flat ring (tilt below 15 degrees).
- `manual.xyz` holds the paper's eight starts (four sites, two azimuths), the ring 2.5 A above the top layer; the starting height is not printed in the paper.
- Energies are PBE+vdW-surf values.
