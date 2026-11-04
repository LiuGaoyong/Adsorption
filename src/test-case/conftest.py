# ruff: noqa: D103
"""Fixtures of the migrated example cases."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.cluster import Octahedron


@pytest.fixture(scope="session")
def cluster() -> Atoms:
    """Return the non-periodic copper octahedron of the main examples."""
    return Octahedron("Cu", 10)


@pytest.fixture(scope="session")
def slab() -> Atoms:
    """Return the periodic Cu(111) slab of the main examples."""
    return fcc111(
        "Cu",
        (10, 10, 5),
        vacuum=10.0,
        orthogonal=True,
        periodic=True,
    )


@pytest.fixture(scope="session")
def result_root() -> Path:
    """Return the PNG root, ignoring its contents in git."""
    root = Path(__file__).resolve().parent / "results"
    root.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("*\n", encoding="utf-8")
    return root
