"""ASE calculators for the relaxation module, built by name.

Every backend is imported lazily, so the module itself needs none of them:

``nequip``
    A compiled or deployed NequIP model (``model_path``).
``emt``
    ASE's effective-medium theory. It only has serious parameters for a few
    fcc metals and is meant for tests, not for adsorption energies.
``module:factory``
    Any importable callable that returns an ASE calculator; the remaining
    options are passed to it as keyword arguments.

A calculator is cached per process and configuration, so a worker loads a
model once however many structures it relaxes.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

__all__ = ["DEFAULT_NEQUIP_MODEL", "build_calculator", "calculator_signature"]

DEFAULT_NEQUIP_MODEL = (
    Path.home() / ".local/nequip-oam-0.1/NequIP-OAM-S-0.1.nequip.pth"
)
"""Default NequIP model path, used when ``model_path`` is empty."""

_CACHE: dict[str, Calculator] = {}


def _nequip(options: Mapping[str, Any]) -> Calculator:
    """Build a NequIP calculator from a compiled or deployed model."""
    values = dict(options)
    raw_path = values.pop("model_path", None)
    model_path = (
        DEFAULT_NEQUIP_MODEL
        if raw_path is None or str(raw_path).strip() == ""
        else Path(str(raw_path)).expanduser()
    )
    device = str(values.pop("device", "cpu"))
    try:
        from nequip.integrations.ase import NequIPCalculator
    except ImportError:
        try:
            from nequip.ase import NequIPCalculator
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'nequip' calculator needs NequIP; install it with "
                "pip install 'adsorption[nequip]'."
            ) from error

    if hasattr(NequIPCalculator, "from_deployed_model"):
        call: dict[str, Any] = {"model_path": str(model_path), "device": device}
        for name in ("species_to_type_name", "chemical_symbols"):
            if values.get(name) is not None:
                call[name] = values[name]
        return NequIPCalculator.from_deployed_model(**call)
    if not hasattr(NequIPCalculator, "from_compiled_model"):
        raise ImportError(
            "This NequIP version exposes neither from_deployed_model nor "
            "from_compiled_model."
        )
    call = {
        "compile_path": str(model_path),
        "device": device,
        "neighborlist_backend": str(
            values.get("neighborlist_backend", "matscipy")
        ),
        "chemical_species_to_atom_type_map": values.get(
            "chemical_species_to_atom_type_map", True
        ),
    }
    return NequIPCalculator.from_compiled_model(**call)


def _emt(options: Mapping[str, Any]) -> Calculator:
    """Build ASE's effective-medium calculator (tests only)."""
    from ase.calculators.emt import EMT

    return EMT(**dict(options))


_BUILDERS = {
    "nequip": _nequip,
    "emt": _emt,
}


def calculator_signature(name: str, options: Mapping[str, Any]) -> str:
    """Return a stable text key of one calculator configuration.

    Args:
        name: The calculator name.
        options: Its options.

    Returns:
        str: A JSON string, usable as a cache key and in a run record.
    """
    return json.dumps(
        {"name": str(name), "options": dict(options)},
        sort_keys=True,
        default=str,
    )


def build_calculator(
    name: str,
    options: Mapping[str, Any] | None = None,
    *,
    cache: bool = True,
) -> Calculator:
    """Return the calculator of one configuration.

    Args:
        name: ``"nequip"``, ``"emt"`` or an
            importable ``"module:factory"``.
        options: Keyword options of the backend.
        cache: Whether to reuse the calculator within this process.

    Returns:
        Calculator: The ASE calculator.

    Raises:
        ValueError: If the name is unknown.
        ImportError: If the backend is not installed.
    """
    values = dict(options or {})
    key = calculator_signature(name, values)
    if cache and key in _CACHE:
        return _CACHE[key]
    label = str(name).strip()
    if label.lower() in _BUILDERS:
        calculator = _BUILDERS[label.lower()](values)
    elif ":" in label:
        module_name, attribute = label.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
        calculator = factory(**values)
    else:
        raise ValueError(
            f"Unknown calculator {name!r}; expected one of "
            f"{sorted(_BUILDERS)} or 'module:factory'."
        )
    if cache:
        _CACHE[key] = calculator
    return calculator
