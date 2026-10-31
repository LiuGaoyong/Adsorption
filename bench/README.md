# Benchmarks and the evaluation protocol

This directory contains the driver-side code of the pre-registered evaluation
in `paper-1/`. It is deliberately **not** part of the installed package: the
manuscript archives the evaluation scripts with the follow-up study, while the
reusable metric functions live in `adsorption.metrics`.

## Baselines

`baselines.py` implements the schemes of Table 6/7:

| ID | Function | Scheme |
|:--|:--|:--|
| B1 | `manual_placement` | one configuration per site, outward normal, hard-sphere distance |
| B2 | `conventional_placement` | the package's `RawAdsorption` geometry |
| B3 | `random_placement` | uniform random rigid body, seeded |
| B4 | `ase_add_adsorbate` | ASE `add_adsorbate` |
| B5 | `basin_hopping` | basin hopping with a seeded global state |
| B6 | `dense_grid_placements` | deterministic latitude-longitude x distance grid |
| B7 | `oracle_scores` | evaluate the full candidate set |

A comparison is only meaningful at equal budget. The campaign API supplies the
fixed budget (`len(campaign)`, `sidecar.jsonl` evaluation counts); the baselines
above are generator functions so a driver can draw the same number of
structures from each scheme, evaluate them with the same calculator and
relaxation protocol, and then compute the metrics with `adsorption.metrics`.

## Metrics

```python
from adsorption.metrics import (
    covering_radius, nearest_neighbor_angles, pole_bias,
    rotation_covering_radius, rotation_nearest_neighbor_angles,
    spearman, kendall, top_k_recall, regret,
    cascade_ratios, false_rejection, max_false_rejection,
    accessible_solid_angle, discretization_residual,
)
```

Spherical metrics act on `(n, 3)` unit vectors; rotation metrics act on
`(n, 4)` quaternions (`w, x, y, z`, as returned by `SO3Set.quaternions`).
Ranking metrics treat scores as energies (lower is better) unless
`higher_is_better=True`.

## Equal-budget accounting

- Count failed and non-converged evaluations: charge them to the scheme that
  produced them.
- Report both equal evaluation counts and equal wall-clock.
- Use pre-registered seeds for B3 and B5; the campaign itself is
  deterministic and is reported once.
- Keep the calculator, relaxation protocol, and convergence criterion
  identical within a comparison.
