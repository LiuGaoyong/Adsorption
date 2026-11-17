# Formic acid on CaO(001)

Reference: Bechthold et al., Appl. Surf. Sci. 583 (2022) 152296. Details, DFT settings and the published configurations
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

- The slab follows the paper: a = 4.84 A, 4x4 primitive surface cells, five layers, 30 A of vacuum.
- `manual.xyz` rebuilds the 40 hand-placed starts of Table S1. The paper prints neither the starting height nor the in-plane azimuth; here the lowest atom starts 2.0 A above the surface.
- The paper's minima are dissociative. `adsorption-relax` labels them `dissociated`, which counts as a valid minimum.
- Only the two lowest minima are described geometrically, and the text does not separate them, so one structural reference covers both. The other energy levels of Table S1 are listed in `reference.yaml` for comparison by energy only.
- The paper uses PBE-D2; compare energies with a potential by order and trend, not by value.
