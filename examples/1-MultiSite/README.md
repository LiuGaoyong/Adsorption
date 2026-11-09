# Multi-site runner examples

Multi-site cases summarized from `add-pipeline`, extended to the full
`0-SingleSite` adsorbate set (methane, methanol, ethanol, ethane, ...). The file
layout mirrors `examples/0-SingleSite`: one folder per system, one folder per
adsorbate, and a `run.yaml` in every adsorbate folder.

```text
examples/1-MultiSite/
├── README.md
├── case_index.csv              # 23 paper-matrix rows
├── gallery_index.csv           # all 56 configs, including extensions
├── run_all.sh
├── plot_sites.py
├── sites/                      # frozen per-site N_OA and cores
├── structures/S6-disordered-Cu.xyz
├── S1-Cu-octahedron/
│   ├── O/run.yaml
│   ├── CO/run.yaml
│   ├── H2O/run.yaml
│   ├── CH4/run.yaml
│   ├── C6H6/run.yaml
│   ├── C2H6/run.yaml
│   ├── CH3OH/run.yaml
│   ├── CH3CH2OH/run.yaml
│   ├── n-hexane/run.yaml
│   └── furfural/run.yaml
├── S2-Cu111/...
...
└── S6-disordered-Cu/...
```

Each `run.yaml` lists **all** site cores of its system, so the runner builds
the multi-site candidate space (with an ETKDG ensemble when an adsorbate has
multiple conformers). The frozen site cores are in [`sites/`](sites/), and
the S6 structure is in [`structures/`](structures/).

There are now **56 configs**: the 23 paper-matrix cases plus 33 extensions.
The extension adds CH4 to every system, C2H6, CH3OH, CH3CH2OH to every system,
and furfural to S1, S2, S4, S5 and S6. Every system therefore carries the same
eight base adsorbates as `0-SingleSite`, plus n-hexane (S1/S2) and furfural
(S1/S2/S3/S4/S5/S6).

## Cases

| System | Substrate | Sites | Adsorbates | Tiers in the paper matrix |
|:--|:--|--:|:--|:--|
| S1 | Cu octahedron, 670 atoms | 6 | O, CO, H2O, CH4, C6H6, C2H6, CH3OH, CH3CH2OH, n-hexane, furfural | tier0, tier1, tier2 |
| S2 | Cu(111) 10×10×5, 500 atoms | 3 | O, CO, H2O, CH4, C6H6, C2H6, CH3OH, CH3CH2OH, n-hexane, furfural | tier0, tier1, tier2, tier3 |
| S3 | Pd(111) 10×10×5, 500 atoms | 3 | O, CO, H2O, CH4, C6H6, C2H6, CH3OH, CH3CH2OH, furfural | tier0, tier1, tier2, tier3 |
| S4 | Fe(110) 8×10×5, 400 atoms | 3 | O, CO, H2O, CH4, C6H6, C2H6, CH3OH, CH3CH2OH, furfural | tier0, tier1, tier3 |
| S5 | Cu(211) stepped, 96 atoms | 4 | O, CO, H2O, CH4, C6H6, C2H6, CH3OH, CH3CH2OH, furfural | tier0, tier1, tier3 |
| S6 | disordered Cu cluster, 200 atoms | 2 | O, CO, H2O, CH4, C6H6, C2H6, CH3OH, CH3CH2OH, furfural | tier0, tier1 |

The original matrix has 66 tier configurations: 23 `tier0-glj` (runnable),
23 `tier1-mlp` (needs a MACE engine), 14 `tier2-emt` (runnable), and 6
`tier3-dft-template` (VASP/POTCAR). The 33 extension configs use the same
`GeometryFilter` + `LennardJonesEnergyFilter` chain and are not part of that
paper matrix. The paper case index is [`case_index.csv`](case_index.csv), and the full
56-config gallery is [`gallery_index.csv`](gallery_index.csv). The original
`pipeline/` source tree is not required to inspect these cases.

## Protocol of these examples

The paper protocol uses `N = 200`, `N_AB = 200`, `M = 12`, and site-specific
`N_OA` (12–82). A faithful run of the 23 paper cases is 10^5–10^6 candidates each.
The example configs therefore use a reduced grid:

```yaml
protocol:
  n_oa: 10
  n_rot: 10
  n_d: 6
  d_min: 0.8
  d_max: 3.0
```

They keep the same multi-site cores and the same adsorbate definitions, but
the candidate counts are demonstration-scale, not paper-scale.

## Running

```bash
# one system-adsorbate case
cd S2-Cu111/CO
adsorption-run --config-path "$PWD" --config-name run

# all 23 cases, sequentially
cd examples/1-MultiSite
RUNNER=adsorption-run ./run_all.sh

# only cases without results yet
SKIP_EXISTING=1 RUNNER=adsorption-run ./run_all.sh
```

The configs use `GeometryFilter(scale=1.0)` followed by
`LennardJonesEnergyFilter(e_cut: null)` with a short relaxation budget
(`max_steps: 10`, `fmax: 0.3`, `reject_on_unconverged: false`). No campaign
results are committed yet.

## Site overview images

`plot_sites.py` highlights every site core of each system and writes
`<system>/sites.png`:

```bash
python examples/1-MultiSite/plot_sites.py
python examples/1-MultiSite/plot_sites.py --only S2-Cu111
```

## Notes

- `n-hexane` and `furfural` come from SMILES (`CCCCCC`, `O=Cc1ccco1`) through
  RDKit ETKDG; the other adsorbates come from `ase.build.molecule`.
- S5 uses `ase.build.fcc211`, which returns a mixed-PBC structure. The
  `GeometryFilter` currently requires all-periodic or non-periodic inputs, so
  the S5 configs are documentation of the sites rather than a guaranteed run
  until mixed-PBC support is added.
- The exact paper-scale configs came from add-pipeline's `pipeline/configs/`;
  they are not duplicated here.
