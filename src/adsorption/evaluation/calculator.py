"""Build an ASE calculator from a dotted path and keyword arguments."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from ase.calculators.calculator import Calculator

__all__ = ["build_calculator"]


def build_calculator(
    target: str,
    kwargs: Mapping[str, Any] | None = None,
) -> Calculator:
    """Instantiate a calculator from a dotted path.

    The path is split at the last dot: ``ase.calculators.emt.EMT`` imports
    ``ase.calculators.emt`` and calls its ``EMT`` attribute. No configuration
    framework is involved, so the same path works from a YAML file and from
    Python.

    Args:
        target: Dotted path of the calculator class or factory.
        kwargs: Keyword arguments forwarded to the constructor.

    Returns:
        Calculator: The calculator instance.

    Raises:
        TypeError: If the resolved object is not an ASE calculator.
        ImportError: If the module or the attribute cannot be imported.
    """
    if not isinstance(target, str) or "." not in target:
        raise ValueError(
            f"The calculator target must be a dotted path, got {target!r}."
        )
    module_name, attribute = target.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Cannot import the calculator module {module_name!r}."
        ) from error
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise ImportError(
            f"The module {module_name!r} has no attribute {attribute!r}."
        ) from error
    calculator = factory(**dict(kwargs or {}))
    if not isinstance(calculator, Calculator):
        raise TypeError(
            f"{target!r} did not produce an ASE Calculator, got "
            f"{type(calculator)}."
        )
    return calculator
