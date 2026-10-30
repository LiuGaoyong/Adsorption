"""Direct adsorption: Fibonacci-grid placement of a rigid adsorbate."""

from typing import Literal, override

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.data import covalent_radii as COV_R
from graphatoms.system import Cluster, Gas, System
from numpy.typing import ArrayLike

from ..common import AdsorptionABC
from ..geometry import (
    DELTA_NBR,
    AdmissibleSet,
    Anchor,
    FibonacciGrid,
    admissible_directions,
    displacements_mic,
    place,
    resolve_anchor,
    shortest_arc_rotation,
)


class DirectAdsorption(AdsorptionABC):
    """Place an adsorbate with a spherical Fibonacci direction grid.

    The admissible approach directions are computed once, from the site
    geometry. For every call the construction picks one admissible direction
    (``OA``), one molecular reference direction (``AB``) and one anchor
    distance, rotates the rigid adsorbate with the shortest-arc rotation, and
    translates it so that the anchor lands on ``O_s + d * u_OA``.

    When no grid index is supplied the construction is deterministic and uses
    index ``0``. Pass ``seed`` to draw the indices from a seeded generator
    instead; the deterministic path never touches a random number generator.

    Attributes:
        nfibonacci: Number of points of the direction grid.
        anchor: Anchor declaration (``"com"``, an atom index, or a point).
        seed: Seed of the stochastic index mode, or ``None``.
        admissible: The admissible approach directions of the site.
    """

    def __init__(
        self,
        atoms: Atoms | System | Cluster,
        *,
        core: ArrayLike = 0,
        nfibonacci: int = 1000,
        neighbors: ArrayLike | None = None,
        calculator: Calculator | None = None,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        scale: float | None = None,
        skin: float | None = None,
        delta_nbr: float = DELTA_NBR,
        anchor: Anchor = "com",
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the direct adsorption calculation.

        Args:
            atoms (Atoms | System | Cluster): The surface or cluster onto
                which the adsorbate should be added.
            core (ArrayLike, optional): The core atom indices.
                Defaults to the first atom.
            nfibonacci (int, optional): The number of points of the direction
                grid. Defaults to 1000.
            neighbors (ArrayLike | None, optional): The first-neighbour shell.
                Detected automatically when None. Defaults to None.
            calculator (Calculator | None, optional): The ASE calculator used
                by the optional two-stage relaxation. Defaults to None.
            max_steps_for_first_stage (int, optional): Step limit of the first
                relaxation stage. Defaults to 100.
            max_steps_for_second_stage (int, optional): Step limit of the
                second relaxation stage. Defaults to 100.
            max_force (float, optional): Force criterion in eV/Angstrom.
                Defaults to 0.05.
            scale (float | None, optional): The direction-sphere scale.
                Defaults to the value tabulated for the core size.
            skin (float | None, optional): The steric margin in Angstrom.
                Defaults to the value tabulated for the core size.
            delta_nbr (float, optional): The neighbour-detection tolerance.
                Defaults to 0.3 Angstrom.
            anchor (Anchor, optional): The anchor declaration: ``"com"`` (the
                centre of mass), an atom index, or an explicit point.
                Defaults to ``"com"``.
            seed (int | None, optional): Seed of the stochastic index mode.
                When None the placement is deterministic. Defaults to None.
            debug (bool, optional): Whether to print debug information.
                Defaults to False.
        """
        super().__init__(
            atoms=atoms,
            neighbors=neighbors,
            calculator=calculator,
            max_steps_for_first_stage=max_steps_for_first_stage,
            max_steps_for_second_stage=max_steps_for_second_stage,
            max_force=max_force,
            scale=scale,
            skin=skin,
            delta_nbr=delta_nbr,
            debug=debug,
            core=core,
        )
        self.nfibonacci = int(nfibonacci)
        if self.nfibonacci < 1:
            raise ValueError(
                f"nfibonacci must be positive, got {self.nfibonacci}."
            )
        if isinstance(anchor, str) and anchor != "com":
            raise ValueError(
                "The only accepted string anchor is 'com', got "
                f"{anchor!r}."
            )
        self.anchor = anchor
        self.seed = None if seed is None else int(seed)
        self._rng: np.random.Generator | None = (
            None if self.seed is None else np.random.default_rng(self.seed)
        )
        self._grid_ads = FibonacciGrid().points(self.nfibonacci)
        self.admissible: AdmissibleSet = admissible_directions(
            self.site_geometry,
            self.nfibonacci,
        )
        self._anchor_core: np.ndarray = np.asarray(
            self.site_geometry.center, dtype=float
        )

    def _draw_index(self, size: int) -> int:
        """Return a deterministic or seeded grid index.

        Args:
            size: Number of available indices.

        Returns:
            int: Index ``0`` in the deterministic mode, otherwise a seeded
            draw from ``[0, size)``.
        """
        if size < 1:
            raise ValueError(f"The index space must be positive, got {size}.")
        if self._rng is None:
            return 0
        return int(self._rng.integers(size))

    @staticmethod
    def _check_index(index: int, size: int, name: str) -> int:
        value = int(index)
        if value < 0 or value >= size:
            raise ValueError(f"{name} must be in [0, {size}), got {index}.")
        return value

    def _default_distance(
        self,
        adsorbate: Atoms,
        anchor: np.ndarray,
    ) -> float:
        """Return the hard-sphere heuristic distance used when none is given.

        Args:
            adsorbate: Reference adsorbate.
            anchor: Anchor position in the reference frame.

        Returns:
            float: The heuristic anchor-centre distance in Angstrom.
        """
        relative = np.asarray(adsorbate.positions, dtype=float) - anchor
        radius = float(np.linalg.norm(relative, axis=1).max())
        contract = float(np.max(COV_R[np.asarray(adsorbate.numbers, int)]))
        direction_sphere = self.site_geometry.radius
        vectors = (
            np.asarray(self.admissible.unit, dtype=float) * direction_sphere
        )
        if len(vectors) == 0:
            return direction_sphere + 0.5 * contract
        micro = displacements_mic(
            vectors,
            self.site_geometry.cell_array(),
            self.site_geometry.pbc_array(),
        )
        core_distance = float(np.linalg.norm(micro, axis=1).mean())
        return core_distance + contract + 0.5 * (radius - contract)

    @override
    def try_adsorption(  # noqa: D417
        self,
        *,
        adsorbate: Atoms,
        idx_grid_core: int | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
        anchor: Anchor | None = None,
        **kwargs,
    ) -> Atoms:
        """Place the adsorbate once and return the unrelaxed structure.

        Args:
            adsorbate (Atoms): The reference adsorbate.
            idx_grid_core (int | None, optional): Index of the admissible
                approach direction. Drawn from the deterministic/seeded mode
                when None.
            idx_grid_ads (int | None, optional): Index of the molecular
                reference direction. Drawn from the deterministic/seeded mode
                when None.
            distance (float | None, optional): The anchor-centre distance.
                When None a hard-sphere heuristic is used.
            anchor (Anchor | None, optional): Per-call anchor override.
                Defaults to the constructor value.

        Returns:
            Atoms: The substrate with the placed adsorbate.
        """
        gas = adsorbate
        if len(gas) == 0:
            raise ValueError("The adsorbate must have at least one atom.")
        resolved = resolve_anchor(
            gas, self.anchor if anchor is None else anchor
        )

        n_admissible = self.admissible.n_oa
        if n_admissible == 0:
            raise ValueError(
                "The site has no admissible approach direction; relax the "
                "filter parameters (scale/skin) or choose another site."
            )
        index_core = (
            self._draw_index(n_admissible)
            if idx_grid_core is None
            else self._check_index(
                idx_grid_core, n_admissible, "idx_grid_core"
            )
        )
        index_ads = (
            self._draw_index(len(self._grid_ads))
            if idx_grid_ads is None
            else self._check_index(
                idx_grid_ads, len(self._grid_ads), "idx_grid_ads"
            )
        )

        direction_core = np.asarray(
            self.admissible.unit[index_core], dtype=float
        )
        direction_ads = np.asarray(self._grid_ads[index_ads], dtype=float)
        rotation = shortest_arc_rotation(
            np.array([0.0, 0.0, 1.0]), direction_ads
        )

        if distance is None:
            distance_value = self._default_distance(gas, resolved.position)
        else:
            distance_value = float(distance)
            if not np.isfinite(distance_value) or distance_value <= 0.0:
                raise ValueError(
                    f"distance must be positive and finite, got {distance!r}."
                )

        target = self._anchor_core + distance_value * direction_core
        placed = place(
            gas,
            anchor=resolved,
            target=target,
            rotation=rotation,
        )

        self._distance = distance_value
        self._direction_core = direction_core
        self._direction_ads = direction_ads
        self._anchor_ref = resolved
        self._adsorbate_pos = np.asarray(placed.positions, dtype=float).copy()

        result = self.atoms.copy()
        result.extend(placed)
        return result

    @override
    def __call__(
        self,
        *,
        adsorbate: Atoms | Gas | Atom | str,
        idx_grid_core: int | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
        anchor: Anchor | None = None,
        **kwargs,
    ) -> tuple[Atoms, Literal[-1, 0, 1, 2]]:
        """Run the direct adsorption calculation.

        Args:
            adsorbate (Atoms | Gas | Atom | str): The adsorbate: an Atoms
                object, an Atom object, a chemical symbol, an ASE molecule
                name, or a SMILES string.
            idx_grid_core (int | None, optional): Index of the admissible
                approach direction. Deterministic/seeded when None.
            idx_grid_ads (int | None, optional): Index of the molecular
                reference direction. Deterministic/seeded when None.
            distance (float | None, optional): The anchor-centre distance.
                A hard-sphere heuristic is used when None.
            anchor (Anchor | None, optional): Per-call anchor override.
            **kwargs: Additional keyword arguments for `try_adsorption`.

        Returns:
            tuple[Atoms, Literal[-1, 0, 1, 2]]:
                The optimized adsorption and the convergence label.
                -   -1: neither stage converged (or no calculator).
                -    0: both stages converged.
                -    1: only the second stage converged.
                -    2: only the first stage converged.
        """
        return self.optimize_adsorption(
            atoms=self.try_adsorption(
                adsorbate=self.get_adsorbate(adsorbate=adsorbate),
                idx_grid_core=idx_grid_core,
                idx_grid_ads=idx_grid_ads,
                distance=distance,
                anchor=anchor,
                **kwargs,
            ),
            natoms=len(self.atoms),
        )
