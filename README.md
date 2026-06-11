# Adsorption

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/adsorption.svg)](https://anaconda.org/conda-forge/adsorption)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/adsorption.svg)](https://anaconda.org/conda-forge/adsorption)
[![Pypi version](https://img.shields.io/pypi/v/adsorption)](https://pypi.org/project/adsorption/)
[![PyPI Downloads](https://static.pepy.tech/badge/adsorption)](https://pepy.tech/projects/adsorption)

A Python package for simulating molecular adsorption on clusters or surfaces, built on top of ASE (Atomic Simulation Environment).

## Overview

The `adsorption` package provides tools for placing and optimizing adsorbates (molecules or atoms) on surface or cluster substrates. It automates the process of:

- Generating adsorption sites on surfaces/clusters
- Placing adsorbates at optimal positions and orientations
- Performing geometry optimization using two-stage relaxation
- Running transition state searches (dimer and NEB methods)
- Running high-throughput adsorption studies with Ray Tune

## Features

- **Multiple Adsorption Strategies**:
  - `RawAdsorption`: Places adsorbates based on geometric site analysis (top, bridge, hollow sites)
  - `DirectAdsorption`: Uses Fibonacci lattice sampling for comprehensive orientation exploration
  - `DirectAdsorptionAD`: Extends `DirectAdsorption` with NequIP-based torch autodiff optimization for GPU-accelerated relaxation

- **Flexible Adsorbate Input**:
  - ASE `Atoms` objects
  - Chemical symbols (e.g., `"O"`, `"C"`)
  - Molecule names recognized by ASE (e.g., `"H2O"`, `"CO"`)
  - SMILES strings (e.g., `"C=C-C-C-C-C"`)

- **Two-Stage Optimization**:
  1. First stage: Fixed substrate + bond length constraints on adsorbate
  2. Second stage: Full relaxation of the entire system

- **Transition State Search**:
  - Dimer method (`call_dimer`) for saddle point searches
  - NEB / DyNEB method (`call_neb`) for minimum energy path searches
  - Built-in trajectory recording with per-step energy snapshots

- **High-Throughput Screening**:
  - Integration with Ray Tune for parallel optimization
  - Automatic grid generation for systematic adsorption site exploration
  - Unified `Helper` interface for iterating over both raw and direct adsorption paths

## Installation

```bash
pip install adsorption
```

Or using pixi:

```bash
pixi install
```

## Quick Start

### Basic Usage

```python
from ase.cluster import Octahedron
from adsorption import RawAdsorption, DirectAdsorption

# Create a copper cluster
cluster = Octahedron("Cu", 10)

# Method 1: RawAdsorption (geometric site-based)
ads = RawAdsorption(calculator=None)  # Replace with actual calculator
result, stage = ads(
    atoms=cluster,
    adsorbate="CO",
    core=[303, 334, 464],  # FCC hollow site
)

# Method 2: DirectAdsorption (Fibonacci lattice sampling)
ads = DirectAdsorption(calculator=None, nfibonacci=100)
result, stage = ads(
    atoms=cluster,
    adsorbate="C6H6",  # Benzene
    core=[454],  # Top site
)
```

### NequIP-Based Autodiff Optimization

```python
from ase.cluster import Octahedron
from adsorption.interfaces import DirectAdsorptionAD
from nequip.ase import NequIPCalculator

cluster = Octahedron("Cu", 10)
calc = NequIPCalculator.from_compiled_model("path/to/model.pth")

ads = DirectAdsorptionAD(calculator=calc, nfibonacci=100)
result, stage = ads(atoms=cluster, adsorbate="CO", core=[454])
```

### Unified Helper Interface

```python
from ase.cluster import Octahedron
from adsorption.interfaces import Helper
from ase.calculators.emt import EMT

cluster = Octahedron("Cu", 10)

helper = Helper(
    calculator=EMT(),
    atoms=cluster,
    adsorbate="CO",
    core=[454],
    use_direct=True,
    use_raw=True,
)

# Run raw adsorption path
result = helper(irun=0, outdir="./results")

# Run direct adsorption path (iterates over orientation grids)
for i in range(1, helper.nrun):
    result = helper(irun=i, outdir="./results")
```

### Transition State Search

```python
from ase.build import fcc100
from adsorption.common.optimize import call_dimer, call_neb
from ase.calculators.emt import EMT

# Create a slab
slab = fcc100("Cu", size=(3, 3, 3), vacuum=10)

# Dimer method for saddle point search
trajectory, converged = call_dimer(
    slab, EMT(), displacement=..., max_steps=1000, fmax=0.02
)

# NEB method for minimum energy path
initial = slab.copy()
final = slab.copy()
# ... set up initial and final states ...
trajectory, converged = call_neb(
    initial, EMT(), final, nimages=5, climb=True
)
```

### Command Line Interface

The package provides a CLI tool for running high-throughput adsorption studies:

```bash
adsorption-tune -cn config.yaml
```

Example configuration (`config.yaml`):

```yaml
output: ./outputs
calculator:
  _target_: ase.calculators.emt.EMT
adsorption:
  nfibonacci: 100
  max_steps_for_first_stage: 100
  max_steps_for_second_stage: 100
  max_force: 0.05
system:
  atoms:
    _target_: ase.build.fcc111
    symbol: Cu
    size: [10, 10, 5]
    vacuum: 15
  core: [454]
gas: C6H6
```

## Architecture

### Core Components

```
src/adsorption/
├── common/
│   ├── __init__.py       # Exports AdsorptionABC, Point, Vector, Site
│   ├── _interface.py     # Abstract base class (AdsorptionABC)
│   ├── _dataclass.py     # Data structures (Point, Vector, Site)
│   └── optimize.py       # optimize(), call_dimer(), call_neb()
├── interfaces/
│   ├── __init__.py       # Exports RawAdsorption, DirectAdsorption, DirectAdsorptionAD
│   ├── _raw.py           # RawAdsorption implementation
│   ├── _direct.py        # DirectAdsorption implementation
│   ├── _directAD.py      # DirectAdsorptionAD (NequIP + torch autodiff)
│   └── helper.py         # Helper class + plot() visualization
└── runner/
    ├── _cli.py           # Hydra-based CLI (adsorption-tune entry point)
    ├── _cli.yaml         # Default Hydra configuration
    └── _tune.py          # Ray Tune integration
```

### Key Classes

#### `AdsorptionABC` (Abstract Base Class)

The abstract base class that defines the common interface and optimization workflow:

- **Two-stage optimization**:
  - Stage 1: Fixes substrate atoms, applies bond length constraints to adsorbate
  - Stage 2: Full relaxation without constraints

- **Adsorbate parsing**: Converts various input types (string, Atom, Atoms, Gas) to ASE Atoms

#### `RawAdsorption`

Places adsorbates based on geometric analysis of adsorption sites:

- Automatically identifies neighbor atoms around the core site
- Calculates optimal adsorption direction based on site geometry
- Supports top, bridge, and hollow sites

**Parameters**:
- `adsorbate_index`: Index of anchoring atom in adsorbate (or `"com"` for center of mass)
- `nbr1hop`: First-neighbor shell indices (auto-detected if not provided)
- `core`: Core atom indices defining the adsorption site

#### `DirectAdsorption`

Uses Fibonacci lattice sampling for comprehensive orientation exploration:

- Generates uniform sampling points on a sphere
- Supports custom grid definitions for both adsorbate and substrate
- Ideal for high-throughput screening

**Parameters**:
- `nfibonacci`: Number of Fibonacci lattice points (default: 1000)
- `grid_core`: Custom grid for substrate orientations
- `grid_ads`: Custom grid for adsorbate orientations
- `distance`: Adsorbate-substrate distance

#### `DirectAdsorptionAD`

Extends `DirectAdsorption` with NequIP-based torch autodiff optimization:

- Requires a `NequIPCalculator` instance
- Uses `torch.optim.LBFGS` for GPU-accelerated relaxation
- Optimizes distance, quaternion rotations via autodiff gradients
- Early stopping with configurable patience and tolerance

#### `Helper`

Unified interface that wraps both `RawAdsorption` and `DirectAdsorption`:

- Initializes grids and configurations for both adsorption paths
- `irun=0`: Runs the raw adsorption path
- `irun>=1`: Runs the direct adsorption path (iterates over orientation x distance grids)
- Returns structured results with energy, convergence stage, and structure

### Data Structures

All defined in `common/_dataclass.py` as Pydantic models:

- **`Point`**: 3D point with arithmetic operations
- **`Vector`**: 3D vector with `length` and `normalize` properties
- **`Site`**: Adsorption site with `neighbor`, `core`, computed `center` and `direction`

### Optimization Utilities

`common/optimize.py` provides:

- **`optimize(atoms, calc, method, max_steps, fmax)`**: General ASE structure optimization with trajectory recording. Returns `(trajectory, converged)`.
- **`call_dimer(atoms, calc, displacement, mask, max_steps, fmax)`**: Dimer method for transition state search. Returns `(trajectory, converged)`.
- **`call_neb(atoms, calc, final_atoms, nimages, climb, ...)`**: DyNEB method for minimum energy path search. Returns `(trajectory, converged)`.

## Examples

### Example 1: Benzene on Cu(111) Surface

```python
from ase.build import fcc111
from adsorption import DirectAdsorption
from ase.calculators.emt import EMT

# Create Cu(111) surface
surface = fcc111("Cu", size=(10, 10, 5), vacuum=15, orthogonal=True)

# Initialize adsorption calculator
ads = DirectAdsorption(
    calculator=EMT(),
    nfibonacci=100,
    max_steps_for_first_stage=100,
    max_steps_for_second_stage=100,
)

# Place benzene on surface
result, stage = ads(
    atoms=surface,
    adsorbate="C6H6",
    core=[454],  # Surface site index
)
```

### Example 2: CO on Copper Cluster

```python
from ase.cluster import Octahedron
from adsorption import RawAdsorption

# Create octahedral Cu cluster
cluster = Octahedron("Cu", 10)

# Place CO on FCC hollow site
ads = RawAdsorption(calculator=None)
result, stage = ads(
    atoms=cluster,
    adsorbate="CO",
    core=[303, 334, 464],  # FCC hollow site
    adsorbate_index=0,  # Anchor on carbon atom
)
```

## High-Throughput Screening

The `tune_adsorption` function enables parallel exploration of multiple adsorption configurations:

```python
from pathlib import Path
from adsorption.runner._tune import tune_adsorption
from ase.build import fcc111

surface = fcc111("Cu", size=(10, 10, 5), vacuum=15)

results = tune_adsorption(
    cfg=config,  # DictConfig with calculator and adsorption settings
    atoms=surface,
    adsorbate="C6H6",
    core=[454],
    nsamples=100,  # Number of random samples
    output=Path("./results"),
)
```

Results are saved as:
- `.xyz` files with energy and optimization stage in filename
- `.png` visualization files with multiple viewing angles

## Dependencies

- **ASE**: Atomic Simulation Environment for atom manipulation
- **graphatoms**: Graph-based atom system utilities
- **Ray[default, tune]**: Distributed hyperparameter optimization
- **Hydra-core**: Configuration management
- **typing-extensions**: Type hints support
- **NequIP** (optional): Neural network interatomic potential for `DirectAdsorptionAD`

## Development

### Running Tests

```bash
pixi run test
# or
pytest -s -vv
```

### Code Style

The project uses:
- **Ruff** for linting and formatting
- **Google-style docstrings**
- Line length: 80 characters

## License

GPL-3.0-or-later

## Author

LiuGaoyong (liugaoyong_88@163.com)

## Links

- [Homepage](https://github.com/LiuGaoyong/Adsorption)
- [Repository](https://github.com/LiuGaoyong/Adsorption)
- [Issues](https://github.com/LiuGaoyong/Adsorption/issues/)
