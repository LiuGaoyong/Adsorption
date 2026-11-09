# ruff: noqa D103
"""Tests of the absolute-import contract of the package.

The package must stay importable through absolute intra-package imports only:
every module is reachable from the ``adsorption`` root, every name declared in
``__all__`` resolves, no source file falls back on a relative import, and the
root package and every subpackage import in a fresh interpreter.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import adsorption

PACKAGE = adsorption.__name__
PACKAGE_ROOT = Path(adsorption.__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
TIMEOUT = 300

REQUIRED_EXPORTS = frozenset(
    {
        "AdsorbateSource",
        "AdsorptionConfigurationFilterABC",
        "AdsorptionConfigurationGenerator",
        "AdsorptionConfigurationGeneratorABC",
        "AdsorptionConfigurationPipeline",
        "BoundAdsorbate",
        "Conformer",
        "ETKDGEnsemble",
        "ExplicitEnsemble",
        "ExplicitSites",
        "GeometryFilter",
        "GraphAtomsSites",
        "LennardJonesEnergyFilter",
        "OptimizedEnergyFilter",
        "Record",
        "ResolvedSites",
        "RigidAdsorbate",
        "RotamerEnsemble",
        "Sampling",
        "SingleAdsorbate",
        "SiteSelection",
        "SiteSource",
        "SiteSpec",
        "TopLayerSites",
        "Trial",
        "Verdict",
    }
)


def _module_names() -> list[str]:
    """Return every module of the package, derived from the source tree."""
    names = {PACKAGE}
    for path in PACKAGE_ROOT.rglob("*.py"):
        parts = list(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.add(".".join([PACKAGE, *parts]))
    return sorted(names)


def _subpackage_names() -> list[str]:
    """Return the first-level subpackages of the package, from the tree."""
    return sorted(
        f"{PACKAGE}.{path.name}"
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.joinpath("__init__.py").is_file()
    )


def _relative_imports(path: Path) -> list[str]:
    """Return the relative imports of a source file, as ``file:line: module``."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            target = "." * node.level + (node.module or "")
            found.append(
                f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {target}"
            )
    return found


def _isolated_env() -> dict[str, str]:
    """Return an environment that imports the package from the source tree."""
    env = dict(os.environ)
    roots = [str(SOURCE_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env


MODULES = _module_names()
PACKAGES = [PACKAGE, *_subpackage_names()]


def test_discovery_finds_the_whole_package() -> None:
    assert len(MODULES) > 10
    assert PACKAGE in MODULES
    assert f"{PACKAGE}.geometry.site" in MODULES
    assert f"{PACKAGE}.filters.relaxation" in MODULES
    assert len(PACKAGES) > 3


@pytest.mark.parametrize("name", MODULES)
def test_module_is_importable(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__name__ == name
    assert module.__file__ is not None


@pytest.mark.parametrize("name", MODULES)
def test_declared_exports_exist(name: str) -> None:
    module = importlib.import_module(name)
    missing = [
        exported
        for exported in getattr(module, "__all__", ())
        if not hasattr(module, exported)
    ]
    assert missing == [], f"{name} declares names it does not define"


def test_source_uses_absolute_imports_only() -> None:
    offenders = [
        entry
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for entry in _relative_imports(path)
    ]
    assert offenders == [], "relative imports found: " + ", ".join(offenders)


def test_root_exports_the_public_api() -> None:
    assert set(adsorption.__all__) >= REQUIRED_EXPORTS
    for name in adsorption.__all__:
        assert getattr(adsorption, name) is not None


def test_root_exports_are_the_same_objects_as_their_modules() -> None:
    assert adsorption.Trial is sys.modules[f"{PACKAGE}.base"].Trial
    assert (
        adsorption.AdsorptionConfigurationPipeline
        is sys.modules[
            f"{PACKAGE}.base.pipeline"
        ].AdsorptionConfigurationPipeline
    )
    assert (
        adsorption.GeometryFilter
        is sys.modules[f"{PACKAGE}.filters.geometry"].GeometryFilter
    )
    assert (
        adsorption.AdsorptionConfigurationGenerator
        is sys.modules[
            f"{PACKAGE}.generate.generator"
        ].AdsorptionConfigurationGenerator
    )


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports_in_a_fresh_interpreter(name: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(SOURCE_ROOT),
        env=_isolated_env(),
    )
    assert completed.returncode == 0, completed.stderr


def test_submodule_import_in_a_fresh_interpreter() -> None:
    """Importing a leaf module directly must not need the root to go first."""
    completed = subprocess.run(
        [sys.executable, "-c", f"import {PACKAGE}.geometry.site"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=str(SOURCE_ROOT),
        env=_isolated_env(),
    )
    assert completed.returncode == 0, completed.stderr
