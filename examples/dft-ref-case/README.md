# DFT reference cases

Seven adsorption systems whose configurations were published in DFT studies,
rebuilt to compare two ways of placing an adsorbate at the same budget:

- **deterministic**: the Cartesian product of a spherical Fibonacci grid of
  approach directions, a rotation set (Super-Fibonacci; a Fibonacci set of
  axis directions for a linear molecule) and an equidistant array of anchor
  distances;
- **random**: every placement drawn on its own, with a uniform random
  direction, a Haar-random orientation and a uniform random distance between
  the two ends of that array. Five seeds.

The question is whether the placements of either arm, once relaxed with a
machine-learned potential, find the configurations the papers report, and
how many distinct minima each arm reaches.

| Case | Published configurations | Source |
|:--|:--|:--|
| `CaO001__HCOOH` | two dissociative minima (-2.38, -2.07 eV) among 40 hand-placed starts | Bechthold 2022, 10.1016/j.apsusc.2021.152296 |
| `CaO001__ethyl-formate` | one dominant minimum (-1.21 eV) among 12 starts | Bechthold 2024, 10.1016/j.jpcs.2023.111780 |
| `Pt111__C6H6` | five minima: bri30, hcp0, fcc0, bri0, atop0 | Liu 2013, 10.1088/1367-2630/15/5/053046 |
| `Pd111__C6H6` | five minima: bri30, hcp0, fcc0, bri0, atop0 | same |
| `Cu111__C6H6` | eight near-degenerate physisorbed minima | same |
| `Pd111__1-hexene` | one adsorption energy (-1.67 eV); flexible adsorbate | van der Hoeven 2022, 10.1021/acs.jpcc.2c04982 |
| `Cu211__CO` | step-edge top and bridge, terrace top, four hollows | Ozbek 2022, 10.55730/1300-0527.3427 |

Each case folder has its own `README.md` with what was taken from the paper
and what had to be assumed.

## What is here

Only placement. `build_refs.py` rebuilds every slab from the lattice constant
and cell of its paper, places the adsorbate with `adsorption-run`, and keeps
the placements that pass the geometric floor. No energy is evaluated. Each
campaign writes `results/accepted.xyz`: complete extended-XYZ frames that
`adsorption-relax run` relaxes, on this machine or another.

- **Rungs** `g00100`, `g00300`, `g01000`, `g03000`, `g10000` are budgets of
  generated placements: about 100, 300, 1000, 3000 and 10000 per case
  (doubled for 1-hexene, which has 24 site-conformer components instead of
  four to eight). The grid of the deterministic arm is the one whose
  candidate count is closest to the target; the random arm generates exactly
  as many candidates on every site and conformer.
- Both arms take directions from the outward hemisphere of each site and
  distances from the same interval: from the hard-sphere contact distance of
  the site to 3.0 A plus the molecular radius.
- The screening chain is the geometric floor alone. How many placements pass
  it is an outcome: it is the number of relaxations a run costs.
- `manual.xyz` (six cases) holds hand-placed starts in the manner of each
  paper. Relaxing them first checks the potential before the sweep is read.

[`placements.txt`](placements.txt) lists, for every rung, the grid, the
generated placements and the placements each run accepted. Roughly one
placement in four or five passes the floor. For the larger adsorbates the
deterministic arm passes about ten percent more than the random arm at the
two largest rungs.

## Commands

```bash
python build_refs.py build       # slabs, sites, run.yaml, relax.yaml, manual.xyz
python build_refs.py run         # every campaign; finished ones are skipped
python build_refs.py status      # generated and accepted placements
python build_refs.py calibrate   # how the grids of the rungs were chosen
```

`runs/` is ignored by git (about 1.1 GB of structures); `build` and `run`
regenerate it in about twelve minutes on one core. The random arm is
reproducible: a placement is a function of the seed and its index.

## Relaxation and analysis (not run here)

```bash
cd Pt111__C6H6
STAGE=manual sbatch relax.sbatch                       # gate: the paper's starts
STAGE=sweep RUNGS="g00100 g00300" sbatch relax.sbatch  # selected rungs
sbatch relax.sbatch     # everything: manual starts, then every campaign
./recall.sh             # basin recall; seeds of the random arm are pooled
```

The accepted placements of all seven cases add up to about 134 000
relaxations (22 000 more for the ablation of `Cu211__CO`), so relax the
manual starts and the small rungs first; every run resumes.

`relax.yaml` frees the layers the paper frees and asks for the gas-phase and
clean-slab references. `relax.sbatch` is a template: adapt the partition, the
environment and the CPU count. `recall.sh` names the runs `arm/rung@seed`,
so the report gives, for each arm and rung, the recall over the seeds and
the first-passage budget. The published configurations are matched by the
geometric criteria of `reference.yaml` (contacts, site kind, azimuth, tilt),
because none of the papers prints coordinates.

## Candidates not built yet

Ethanol on CaO(001) (10.1016/j.apsusc.2019.144254), ethylene on Pd(111) and
propene on Pt(111) (di-sigma against pi), furfural on Pd, Cu and Pt(111),
formic acid on anatase TiO2(101), and the alloy systems of the local library
(CO on PdCu(111), water and methanol on PdZn, acetylene on Pd-Ag). Their
sources were found but not checked in detail.
