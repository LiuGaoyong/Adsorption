from pathlib import Path

import matplotlib
from ase import Atoms
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

matplotlib.use("Agg")


def plot(atoms: Atoms, pngfname: Path) -> None:
    atoms.wrap(pbc=any(atoms.pbc))
    fig, axes = plt.subplots(2, 2, dpi=150, figsize=(16, 12))
    for ax, rot in zip(
        axes.flatten(),
        [
            "0x, 0y, 0z",  # top
            "-90x, 0y, 0z",  #
            "0x, 90y, 0z",  #
            "45x, 45y, 45z",  #
        ],
    ):
        assert isinstance(ax, Axes)
        # atoms.write()
        plot_atoms(
            atoms,
            ax=ax,
            rotation=rot,
            radii=None,
            bbox=None,
            colors=None,
            scale=20,
            maxwidth=500,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(pngfname.with_suffix(".png"))
