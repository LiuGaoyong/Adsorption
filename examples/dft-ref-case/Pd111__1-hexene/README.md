# 1-Hexene on Pd(111)

Reference: van der Hoeven et al., J. Phys. Chem. C 126 (2022) 15710. Details, DFT settings and the published configurations
are in [`reference.yaml`](reference.yaml).

| File | Content |
|:--|:--|
| `reference.yaml` | source, surface model, published configurations and their geometric criteria |
| `substrate.xyz` | the slab, atoms tagged by layer (1 is the outermost) |
| `sites.yaml` | the site cores used for placement |
| `relax.yaml`, `relax.sbatch` | options and slurm template of `adsorption-relax run` |
| `recall.sh` | basin recall of the relaxed runs (`adsorption-relax recall`) |
| `runs/deterministic/<rung>/` | the Cartesian-product campaign: `run.yaml`, `results/accepted.xyz` |
| `runs/random/s<seed>/<rung>/` | the independent random campaign of one seed |

## Notes

- The paper gives one adsorption energy (-1.67 eV, optPBE-vdW) and does not describe the geometry in its text, so this is an energy reference, not a structural one. The criterion only asks that the C=C carbons are bound to Pd.
- A correction to the article exists and was not read; check it before quoting the energy.
- The Pd lattice constant is not printed in the paper; 3.89 A (experiment) is used. The 4x4, six-layer cell is stated for the alloy model and assumed for pure Pd.
- Flexible adsorbate: six rigid rotamers (three rotatable bonds, three dihedral values each, a stratified subset), so the placement has 24 components (four sites times six rotamers). The generated budget of each rung is doubled.
- There is no `manual.xyz`: the paper does not say how the molecule was placed.
