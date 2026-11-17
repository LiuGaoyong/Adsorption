# Ethyl formate on CaO(001)

Reference: Bechthold et al., J. Phys. Chem. Solids 185 (2024) 111780. Details, DFT settings and the published configurations
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

- Same slab as the formic-acid case.
- `manual.xyz` rebuilds the 12 hand-placed starts (three approaches at four sites). The paper numbers its sites 1-4 without naming them; the four sites of the formic-acid paper are assumed, and the starting height is chosen (lowest atom 2.0 A above the surface).
- The molecule is placed as one rigid conformer (RDKit geometry of `CCOC=O`); the paper also used a single gas-phase structure.
- The reference minimum (P3, -1.21 eV) forms a C-O bond to a surface oxygen while keeping its own bonds; it is matched by that contact.
