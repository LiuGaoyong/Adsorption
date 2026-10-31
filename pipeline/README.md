# Pipeline: test cases and the pre-registered evaluation matrix

This directory contains the generated test cases of the evaluation matrix of
`paper-1/` (Table 4: systems; Table 5: adsorbates and tiers; Section 3.3: tier
definitions). Every case is a campaign configuration under `pipeline/configs/`.

**No campaign was run in the background, and no full campaign was launched
here.** Generation and validation only build structures, site geometries,
budgets, screen parameters, and calculator objects. Two single-candidate smoke
tests were run in the foreground to verify the runnable tiers (see
[Smoke tests](#smoke-tests)).

## Layout

```
pipeline/
├── matrix.py            # matrix, adsorbates, protocol, tiers, site selection
├── generate.py          # writes configs/, sites/, structures/ (no campaign)
├── validate.py          # code-level runnability report (no campaign)
├── subset.py            # pre-registered DFT subset selection (no energies)
├── configs/             # 66 YAML campaign configurations
├── sites/               # pre-registered site lists and their N_OA
├── structures/          # S6 disordered Cu cluster (deterministic)
├── reports/
│   ├── matrix.csv       # one row per configuration
│   └── runnability.md   # generated runnability table
└── runs/                # campaign outputs (git-ignored)
```

Regenerate and validate (both foreground, neither runs a campaign):

```bash
python -m pipeline.generate --clean
python -m pipeline.validate
```

## Matrix

| System | Substrate | Adsorbates | Tiers | DFT subset |
|:--|:--|:--|:--|:--|
| S1 | Cu octahedron, `Octahedron("Cu", 10)`, 670 atoms | O, CO, H2O, CH4, C6H6, n-hexane | G+LJ, MLP, EMT | — |
| S2 | Cu(111) 10×10×5, 15 Å vacuum, 500 atoms | O, CO, H2O, C6H6, n-hexane | G+LJ, MLP, EMT | CO, C6H6 |
| S3 | Pd(111) 10×10×5, 15 Å vacuum, 500 atoms | CO, C6H6, furfural | G+LJ, MLP, EMT | CO |
| S4 | Fe(110) 8×10×5, 15 Å vacuum, 400 atoms | O, CO, C6H6 | G+LJ, MLP | O, CO |
| S5 | Cu(211) stepped, 15 Å vacuum, 96 atoms | O, CO, C6H6 | G+LJ, MLP | CO |
| S6 | disordered Cu cluster, 200 atoms | O, CO, C6H6 | G+LJ, MLP | — |

Tier definitions and configuration names:

| Tier | Config name | Meaning | Code-level status (this environment) |
|:--|:--|:--|:--|
| 0 | `tier0-glj` | geometry + Fibonacci grid + distance array + KIM Lennard-Jones screen, no energy engine | **runnable now** (23 configs) |
| 1 | `tier1-mlp` | two-stage MLP relaxation with MACE | **needs engine**: `mace.calculators` is not installed and `models/mace.model` is absent (23 configs) |
| 2 | `tier2-emt` | two-stage EMT relaxation (Cu/Pd only) | **runnable now** (14 configs) |
| 3 | `tier3-dft-template` | single-point VASP evaluation of the pre-registered subset | **template**: ASE `Vasp` constructs, but execution requires the VASP executable and POTCARs; apply `pipeline/subset.py` first (6 configs) |

Runnability counts from `pipeline/reports/matrix.csv`: **66 configs**, 37
runnable, 23 needing an engine, 6 templates, **0 invalid**, **0 sites with
`N_OA = 0`**. Candidate budgets range from 124,800 to 926,400 per campaign
(N = 200, N_AB = 200, M = 12; n-hexane has two generated conformers, so its
campaigns multiply the per-site budget).

## Protocol and placeholders

`matrix.py` uses the manuscript's illustrative values: N = 200, N_AB = 200,
d = 0.8–3.0 Å in 0.2 Å steps (M = 12), default `scale`/`skin`, δ_nbr = 0.3 Å.
These are protocol choices, not recommended optima. The following values are
placeholders because the manuscript marks them `[TO BE FIXED]`:

- screen thresholds `r_min = 1.0 Å`, `e_cut = 50 eV`;
- the MACE model path `models/mace.model`;
- the VASP input line (`xcc: PBE`, `encut: 450`, `kpts: [3, 3, 1]`).

Freeze them in a real campaign and record them in the sidecar hash (the
parameter hash covers the whole protocol, screen, sites, and adsorbate).

## Pre-registered sites

The site indices are computed once and stored in `pipeline/sites/`; the
manuscript requires the site list to be frozen before a campaign. Labels are
descriptive construction labels, **not** the `graphatoms` six-class taxonomy.

| System | Formula, atoms | Sites (`N_OA` at N=200) |
|:--|:--|:--|
| S1 | Cu670 | vertex_top 79, vertex_bridge 37, vertex_three_fold 21, facet_top 12, facet_bridge 23, facet_three_fold 21 |
| S2 | Cu500 | facet_top 12, facet_bridge 19, facet_three_fold 21 |
| S3 | Pd500 | facet_top 16, facet_bridge 24, facet_three_fold 24 |
| S4 | Fe400 | facet_top 11, facet_bridge 22, facet_three_fold 22 |
| S5 | Cu96 | step_top 25, step_bridge 38, terrace_top 11, terrace_bridge 22 |
| S6 | Cu200 | coord_top 82, coord_bridge 42 |

S6 is a deterministic greedy random packing (seed 20261030) followed by 20
FIRE/EMT steps; the committed XYZ makes the case reproducible. It is a test
structure for site-coordination behaviour, not a physical sample.

Adsorbates: O, CO, H2O, CH4, C6H6 come from the ASE molecule database;
n-hexane (`CCCCCC`) and furfural (`O=Cc1ccco1`) use deterministic RDKit ETKDG
embedding (seed 20261030, four requested, RMSD pruning to 0.5 Å), which yields
2 and 1 conformers in this environment. The generated ensemble defines the
reference as well, so reference and conformers share one composition
(C6H14, C5H4O2).

## Code-level runnability analysis

`python -m pipeline.validate` loads every configuration, builds the substrate,
the adsorbate (including generated conformers), the campaign plan, the site
geometries, the budget, the KIM parameter table (over substrate **and**
adsorbate elements), and resolves the evaluation calculator. It does not
generate candidate structures, evaluate energies, or launch a campaign. The
table is in `pipeline/reports/runnability.md`.

Two defects were found by this exercise and fixed in the code:

1. **Screen elements were taken from the substrate only**, so a KIM table
   without C/O/H failed at the first candidate with `KeyError: Z=6`. The runner
   and the validator now collect the substrate plus every conformer's elements.
2. **The Lennard-Jones screen summed every pair within the cutoff**, including
   intramolecular bonds and the substrate–substrate baseline. A C–O bond
   (1.13 Å) and a 500-atom Cu slab produced thousands of eV, so every candidate
   was rejected. The screen now uses substrate–adsorbate contact pairs only
   (`C(χ)` in the manuscript); paper-1 Eq. (18) and Algorithm 3 were updated to
   match.
3. **The Tier-3 subset is non-contiguous**, so the CLI gained `--indices` to
   evaluate an explicit index list (see below) instead of forcing a range.

## Smoke tests (foreground)

These two checks were run in the foreground; neither is a full campaign:

```bash
# Tier 0: one candidate (the d = 0.8 Å stress point) is rejected by the screen.
adsorption-campaign run pipeline/configs/S2-Cu111/CO/tier0-glj.yaml --limit 50
# -> 50 processed, 16 survive; kappa 1: e_rep = 3.25e14 eV, r_close = 0.14 Å

# Tier 2: one screened survivor relaxed with EMT.
adsorption-campaign run pipeline/configs/S2-Cu111/CO/tier2-emt.yaml --start 9 --stop 9
# -> kappa 9: e_rep = 30.4 eV, r_close = 1.74 Å, EMT two-stage converged,
#    E = 70.07 eV
```

The first case shows the fixed-radius filter accepting an anchor at 0.8 Å and
the LJ tier rejecting it, which is exactly the division of labour between
Sections 2.4 and 2.8. The second shows the evaluation tier on a survivor.

## Running a case

Run from the repository root; configuration paths are relative to it. Keep
runs in the foreground:

```bash
# Budget only (no candidates generated)
adsorption-campaign enumerate pipeline/configs/S2-Cu111/CO/tier0-glj.yaml

# Tier 0 or Tier 2, one candidate at a time
adsorption-campaign run pipeline/configs/S2-Cu111/CO/tier0-glj.yaml --limit 1
adsorption-campaign run pipeline/configs/S2-Cu111/CO/tier2-emt.yaml --start 9 --stop 9

# Resume an interrupted run
adsorption-campaign run pipeline/configs/S2-Cu111/CO/tier0-glj.yaml --resume

# Distributed across local workers (still foreground for this shell)
adsorption-campaign run pipeline/configs/S2-Cu111/CO/tier0-glj.yaml --workers 4
```

Do not launch a full matrix casually: the budgets are paper-scale (10^5–10^6
candidates per cell). A typical smoke test is `--limit 1` (Tier 0) or a single
surviving index (Tier 2).

### Tier 1 (MLP)

Install an MLP package and provide its model, then edit
`tier1-mlp`'s `evaluation.calculator.target`/`kwargs` (the config currently
uses `mace.calculators.MACECalculator` with `models/mace.model`). The validator
reports the tier as `needs engine` until the import and the model path resolve.

### Tier 3 (DFT subset)

The pre-registered subset is selected from a cheap tier's sidecar:

```bash
python -m pipeline.subset \
  --sidecar pipeline/runs/S2-Cu111/CO/tier2-emt/sidecar.jsonl \
  --output pipeline/runs/S2-Cu111/CO/dft-subset.json \
  --top 10 --per-decile 2 --deciles 10 --control 5 --seed 0
```

The rule takes the best 10 by energy, one sample per rank decile of the rest,
and five uniform controls; it uses only cheap-tier ranks. Then evaluate exactly
those indices with the DFT template (external VASP required):

```bash
adsorption-campaign run pipeline/configs/S2-Cu111/CO/tier3-dft-template.yaml \
  --indices pipeline/runs/S2-Cu111/CO/dft-subset.json
```

## Baselines

B1–B7 apply to every matrix row and are implemented in `bench/baselines.py`
(see `bench/README.md`). They are driver functions, not campaign
configurations, because equal-budget accounting is a campaign-level concern.

## Limitations

- Site selection is a pre-registered descriptive construction, not the
  graph-based enumeration or the six-class labels; those belong to
  `graphatoms` (adapters in `adsorption.providers`).
- S6 is a synthetic disordered test structure, not a physical sample.
- The thresholds, MLP model path, and DFT settings are placeholders.
- Tier 1 and Tier 3 were not executed here; their code paths were resolved,
  not run, because the engines are external.
