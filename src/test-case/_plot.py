"""Plotting helpers of the migrated example cases.

The two-by-two function is the ``plot(atoms, pngfname)`` helper of the main
branch (``src/adsorption/interfaces/helper.py``): four views of one structure
are rendered into a single PNG.  It is used for every site grid and for every
surviving adsorption configuration.
"""

from __future__ import annotations

# ruff: noqa: E402

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "adsorption-mpl-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

from ase import Atoms
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

__all__ = ["plot"]

_ROTATIONS = (
    "0x, 0y, 0z",
    "-90x, 0y, 0z",
    "0x, 90y, 0z",
    "45x, 45y, 45z",
)


def plot(
    atoms: Atoms,
    pngfname: Path,
    *,
    dpi: int = 100,
    figsize: tuple[float, float] = (12.0, 9.0),
    scale: float = 20.0,
    maxwidth: float = 500.0,
    radii: Sequence[float] | None = None,
) -> Path:
    """Render one structure from four views into a PNG.

    Args:
        atoms: The structure to render.
        pngfname: Destination path; the suffix is replaced by ``.png``.
        dpi: Figure resolution.
        figsize: Figure size in inches.
        scale: Atom-radius scale forwarded to ``plot_atoms``.
        maxwidth: Maximum image width forwarded to ``plot_atoms``.
        radii: Optional per-atom radii forwarded to ``plot_atoms``.

    Returns:
        Path: The written PNG path.
    """
    png = Path(pngfname).with_suffix(".png")
    png.parent.mkdir(parents=True, exist_ok=True)

    show = atoms.copy()
    # Do not wrap periodic structures here: wrapping splits a slab at the
    # cell boundary and hides the site that the figure is meant to show.

    fig, axes = plt.subplots(2, 2, dpi=dpi, figsize=figsize)
    for ax, rotation in zip(axes.flatten(), _ROTATIONS):
        assert isinstance(ax, Axes)
        plot_atoms(
            show,
            ax=ax,
            rotation=rotation,
            radii=radii,
            bbox=None,
            colors=None,
            scale=scale,
            maxwidth=maxwidth,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(png)
    plt.close(fig)
    return png
