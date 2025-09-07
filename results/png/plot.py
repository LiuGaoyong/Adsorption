from pathlib import Path

from matplotlib import pyplot as plt


def plot(k="O"):  # noqa: D103
    pngs = sorted(Path(__file__).parent.glob(f"{k}*.png"))
    assert len(pngs) == 9, len(pngs)
    idx = [
        (0, 0, 0),
        (0, 1, 9),
        (0, 2, 1),
        (0, 3, 4),
        (0, 4, 7),
        (1, 0, 3),
        (1, 1, 6),
        (1, 2, 2),
        (1, 3, 5),
        (1, 4, 8),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(5 * 5, 2 * 5))
    for i, j, id in idx:
        axes[i, j].axis("off")
        if id != 0:
            f = pngs[id - 1]
            axes[i, j].imshow(plt.imread(f))
            axes[i, j].set_title(
                ",".join(f.name.split(".")[0].split("_")[1:3]),
                fontsize=20,
            )
    fig.tight_layout()
    plt.savefig(f"ZZZ_{k}.png")


for k in ("O", "CO", "H2O", "CH4"):
    plot(k)
