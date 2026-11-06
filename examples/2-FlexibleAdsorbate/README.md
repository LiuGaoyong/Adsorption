# Flexible-adsorbate runner examples

Flexible-adsorbate cases built on the same six systems as
`examples/1-MultiSite`. The layout mirrors `0-SingleSite` and `1-MultiSite`:
one folder per system, one folder per flexible adsorbate, and a `run.yaml` in
every adsorbate folder.

```text
examples/2-FlexibleAdsorbate/
├── README.md
├── gallery_index.csv
├── run_all.sh
├── plot_sites.py
├── sites/
├── structures/S6-disordered-Cu.xyz
├── S1-Cu-octahedron/
│   ├── n-hexane/run.yaml
│   └── 1,5-hexadiene/run.yaml
├── S2-Cu111/...
...
└── S6-disordered-Cu/...
```

## Adsorbates

| Adsorbate | SMILES | Rotatable bonds | Conformers at `n_conformers=8` |
|:--|:--|--:|--:|
| n-hexane | `CCCCCC` | 3 | 3 |
| 1,5-hexadiene | `C=CCCC=C` | 3 | 5 |

Both are generated with RDKit ETKDG (seed 20261030, RMSD pruning 0.5 Å).
The runner detects the rotatable bonds automatically, builds the conformer
ensemble and uses `MultiAdsorbateGenerator`; the conformer count is recorded in
`summary.json`.

## Systems and sites

The six systems, substrates and all site cores are the same as
`examples/1-MultiSite`; see [`gallery_index.csv`](gallery_index.csv) for the
12 system–adsorbate rows and [`sites/`](sites/) for the frozen site metadata.

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
- S5 uses `ase.build.fcc211`, which returns mixed PBC; the current
  `GeometryFilter` requires all-periodic or non-periodic inputs, so the S5
  configs are structural documentation until mixed-PBC support is added.
- `1,5-hexadiene` is written as `1,5-hexadiene` in folder and config names.
