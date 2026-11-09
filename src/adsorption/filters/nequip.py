"""Process B: NequIP-relaxed energy tier.

The filter is the generic :class:`OptimizedEnergyFilter` with a NequIP
calculator built from a deployed model. It is a separate module so the
optional NequIP dependency is only imported when the filter is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from adsorption.filters.relaxation import OptimizedEnergyFilter

__all__ = ["DEFAULT_NEQUIP_MODEL", "NequIPOptimizedEnergyFilter"]

DEFAULT_NEQUIP_MODEL = (
    Path.home() / ".local/nequip-oam-0.1/NequIP-OAM-S-0.1.nequip.pth"
)
"""Default deployed NequIP OAM model path."""

_CALCULATOR_CACHE: dict[tuple[Any, ...], Calculator] = {}


def _calculator_key(
    model_path: str,
    device: str,
    species_to_type_name: Mapping[int, str] | None,
    chemical_symbols: Sequence[str] | None,
    chemical_species_to_atom_type_map: Mapping[str, str] | bool | None,
    neighborlist_backend: str,
) -> tuple[Any, ...]:
    """Return a hashable cache key of one calculator configuration."""
    species_key = (
        None
        if species_to_type_name is None
        else tuple(
            sorted((int(k), str(v)) for k, v in species_to_type_name.items())
        )
    )
    symbols_key = None if chemical_symbols is None else tuple(chemical_symbols)
    if isinstance(chemical_species_to_atom_type_map, Mapping):
        map_key: Any = tuple(
            sorted(
                (str(key), str(value))
                for key, value in chemical_species_to_atom_type_map.items()
            )
        )
    else:
        map_key = chemical_species_to_atom_type_map
    return (
        str(model_path),
        str(device),
        species_key,
        symbols_key,
        map_key,
        str(neighborlist_backend),
    )


def _cached_calculator(
    model_path: str,
    device: str,
    species_to_type_name: Mapping[int, str] | None,
    chemical_symbols: Sequence[str] | None,
    chemical_species_to_atom_type_map: Mapping[str, str] | bool | None,
    neighborlist_backend: str,
) -> Calculator:
    """Return a process-local NequIP calculator, loading it once."""
    key = _calculator_key(
        model_path,
        device,
        species_to_type_name,
        chemical_symbols,
        chemical_species_to_atom_type_map,
        neighborlist_backend,
    )
    cached = _CALCULATOR_CACHE.get(key)
    if cached is None:
        cached = _build_calculator(
            model_path,
            device,
            species_to_type_name,
            chemical_symbols,
            chemical_species_to_atom_type_map,
            neighborlist_backend,
        )
        _CALCULATOR_CACHE[key] = cached
    return cached


def _build_calculator(
    model_path: str,
    device: str,
    species_to_type_name: Mapping[int, str] | None,
    chemical_symbols: Sequence[str] | None,
    chemical_species_to_atom_type_map: Mapping[str, str] | bool | None,
    neighborlist_backend: str,
) -> Calculator:
    """Build a NequIP ASE calculator from a compiled or deployed model.

    Args:
        model_path: Path of the NequIP model.
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
        species_to_type_name: Optional atomic-number mapping for the older
            ``from_deployed_model`` API.
        chemical_symbols: Optional ordered chemical symbols for the older API.
        chemical_species_to_atom_type_map: Optional mapping for the newer
            ``from_compiled_model`` API.
        neighborlist_backend: Neighbor-list backend of the newer API.

    Returns:
        Calculator: The NequIP ASE calculator.

    Raises:
        ImportError: If NequIP is not installed.
        ValueError: If the model path is empty.
    """
    if not model_path:
        raise ValueError("model_path must not be empty.")
    try:
        from nequip.integrations.ase import NequIPCalculator
    except ImportError:
        try:
            from nequip.ase import NequIPCalculator
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "NequIP is required for NequIPOptimizedEnergyFilter; "
                "install nequip and provide a compiled model path."
            ) from error

    if hasattr(NequIPCalculator, "from_deployed_model"):
        options: dict[str, Any] = {
            "model_path": str(model_path),
            "device": str(device),
        }
        if species_to_type_name is not None:
            options["species_to_type_name"] = {
                int(key): str(value)
                for key, value in species_to_type_name.items()
            }
        if chemical_symbols is not None:
            options["chemical_symbols"] = [
                str(value) for value in chemical_symbols
            ]
        return NequIPCalculator.from_deployed_model(**options)

    if not hasattr(NequIPCalculator, "from_compiled_model"):
        raise ImportError(
            "This NequIP version exposes neither from_deployed_model nor "
            "from_compiled_model."
        )
    options = {
        "compile_path": str(model_path),
        "device": str(device),
        "neighborlist_backend": str(neighborlist_backend),
    }
    if chemical_species_to_atom_type_map is not None:
        options["chemical_species_to_atom_type_map"] = (
            chemical_species_to_atom_type_map
        )
    return NequIPCalculator.from_compiled_model(**options)


class NequIPOptimizedEnergyFilter(OptimizedEnergyFilter):
    """Relax with a NequIP potential and apply an energy threshold.

    The calculator is built lazily on the first trial that reaches this tier.
    A chunk whose trials are all rejected by earlier filters therefore never
    loads the NequIP model.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "cpu",
        species_to_type_name: Mapping[int, str] | None = None,
        chemical_symbols: Sequence[str] | None = None,
        chemical_species_to_atom_type_map: Mapping[str, str]
        | bool
        | None = None,
        neighborlist_backend: str = "matscipy",
        e_cut: float | None = None,
        fmax: float = 0.05,
        max_steps: int = 50,
        method: str = "LBFGS",
        fix_substrate: bool = False,
        fix_adsorbate_bonds: bool = False,
        reject_on_unconverged: bool = False,
        precompute_reference: bool = False,
        **kwargs: Any,
    ) -> None:
        """Build the NequIP calculator and the acceptance policy.

        Args:
            model_path: Path of the deployed NequIP model. Defaults to
                ``DEFAULT_NEQUIP_MODEL`` when empty.
            device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
            species_to_type_name: Optional mapping from atomic number to the
                model's type name.
            chemical_symbols: Optional ordered chemical symbols of the model.
            chemical_species_to_atom_type_map: Optional mapping for the newer
                ``from_compiled_model`` API.
            neighborlist_backend: Neighbor-list backend of the compiled-model
                API.
            e_cut: Energy ceiling in eV, or ``None`` for the isolated-fragment
                reference.
            fmax: Force tolerance in eV/A.
            max_steps: Maximum number of optimization steps.
            method: ASE optimizer name.
            fix_substrate: Whether to fix the substrate atoms. Defaults to
                ``False`` so the full structure relaxes.
            fix_adsorbate_bonds: Whether to constrain the adsorbate bonds.
                Defaults to ``False``.
            reject_on_unconverged: Whether to reject unconverged trials.
            precompute_reference: Whether to compute the isolated-fragment
                reference in ``prepare``; off by default so workers with no
                NequIP survivor never load the model.
            **kwargs: Forwarded to the base filter.

        Raises:
            ValueError: If the model path is empty.
        """
        resolved = (
            DEFAULT_NEQUIP_MODEL
            if model_path is None or str(model_path).strip() == ""
            else Path(str(model_path)).expanduser()
        )
        self.model_path = str(resolved)
        self.device = str(device)
        self.species_to_type_name = (
            None
            if species_to_type_name is None
            else {
                int(key): str(value)
                for key, value in species_to_type_name.items()
            }
        )
        self.chemical_symbols = (
            None
            if chemical_symbols is None
            else tuple(str(value) for value in chemical_symbols)
        )
        self.chemical_species_to_atom_type_map = (
            chemical_species_to_atom_type_map
        )
        self.neighborlist_backend = str(neighborlist_backend)
        self.calculator = None
        self.precompute_reference = bool(precompute_reference)
        super().__init__(
            e_cut=e_cut,
            calculator=None,  # type: ignore[arg-type]
            fmax=fmax,
            max_steps=max_steps,
            method=method,
            fix_substrate=fix_substrate,
            fix_adsorbate_bonds=fix_adsorbate_bonds,
            reject_on_unconverged=reject_on_unconverged,
            **kwargs,
        )

    def _ensure_calculator(self) -> None:
        """Load the NequIP calculator on first use in this process."""
        if self.calculator is None:
            self.calculator = _cached_calculator(
                self.model_path,
                self.device,
                self.species_to_type_name,
                self.chemical_symbols,
                self.chemical_species_to_atom_type_map,
                self.neighborlist_backend,
            )

    def prepare(self, trial) -> None:
        """Prepare the reference ceiling only when explicitly requested."""
        if self.e_cut is not None:
            return
        if not getattr(self, "precompute_reference", False):
            return
        self._ensure_calculator()
        super().prepare(trial)

    def evaluate(self, trial):
        """Evaluate after loading the calculator lazily."""
        self._ensure_calculator()
        return super().evaluate(trial)

    def __getstate__(self) -> dict[str, Any]:
        """Drop the Torch model; it is rebuilt in the worker process."""
        state = self.__dict__.copy()
        state["calculator"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild or reuse a process-local calculator after unpickling."""
        self.__dict__.update(state)
        self.calculator = None

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the filter."""
        return {
            "kind": "nequip_optimized_energy",
            "precompute_reference": self.precompute_reference,
            "calculator": {
                "model_path": self.model_path,
                "device": self.device,
            },
            "e_cut": self.e_cut,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            "method": self.method,
            "fix_substrate": self.fix_substrate,
            "fix_adsorbate_bonds": self.fix_adsorbate_bonds,
            "reject_on_unconverged": self.reject_on_unconverged,
        }
