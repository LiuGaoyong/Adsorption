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

Every adsorbate folder is a single Hydra configuration. Running it writes
`records.jsonl`, `accepted.xyz` (the accepted structures as an extended-XYZ
trajectory) and `summary.json` under its `results/` directory.

```bash
# one example
cd cluster_vertex_top/CO
adsorption-run --config-path "$PWD" --config-name run

# all 96 examples, sequentially
RUNNER=adsorption-run ./run_all.sh

# only the examples that have no results yet
SKIP_EXISTING=1 RUNNER=adsorption-run ./run_all.sh
```

## Site images

`plot_sites.py` redraws the site geometry of every site folder with the
four-view helper and writes `<site>/site.png`:

```bash
python examples/0-SingleSite/plot_sites.py
python examples/0-SingleSite/plot_sites.py --only cluster_vertex_top
```

The highlighted atoms are the site core.

The examples use rigid adsorbates (`conformers: none`),
`GeometryFilter(scale=1.0)` and the size-aware distance bounds.
