"""Canonical hashing of the parameter set.

Two campaigns are comparable only when they share the same parameter hash:
the protocol values, the site list, the reference adsorbate, and the software
versions. The hash is recorded with every candidate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

__all__ = ["parameter_hash", "stable_hash"]


_PACKAGES = ("adsorption", "ase", "numpy", "graphatoms")


def _software_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "unknown"
    return found


def _round(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Non-finite parameter in the hash: {value!r}.")
    if value == 0.0:
        return 0.0
    return float(f"{value:.12g}")


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if hasattr(value, "model_dump"):
        return _canonical(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return _round(value)
    raise TypeError(f"Unsupported value in the parameter hash: {type(value)}.")


def stable_hash(payload: Any) -> str:
    """Return the SHA-256 of the canonical JSON form of ``payload``.

    Args:
        payload: A JSON-composable object with optional NumPy arrays, tuples,
            and pydantic models.

    Returns:
        str: The hexadecimal digest.
    """
    canonical = _canonical(payload)
    text = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parameter_hash(
    protocol: Any,
    sites: Sequence[Any],
    adsorbate: Any,
) -> str:
    """Hash the parameter set of a campaign.

    Args:
        protocol: The protocol model (or any object with ``model_dump``).
        sites: The pre-registered site list; ``core`` and ``neighbors`` are
            the hashed fields.
        adsorbate: The adsorbate specification; the reference composition and
            geometry, the anchor declaration, and the reference axis are
            hashed.

    Returns:
        str: The hexadecimal SHA-256 digest.
    """
    specs = getattr(adsorbate, "conformer_specs", None)
    if specs is None:
        specs = (adsorbate,)
    conformer_records = []
    for spec in specs:
        reference = getattr(spec, "reference", None)
        if reference is None:
            raise TypeError(
                "each conformer must expose a 'reference' Atoms object."
            )
        conformer_records.append(
            {
                "numbers": np.asarray(
                    reference.numbers, dtype=int
                ).tolist(),
                "positions": np.asarray(
                    reference.positions, dtype=float
                ).tolist(),
                "anchor": getattr(spec, "anchor", "com"),
                "axis": getattr(spec, "axis", (0.0, 0.0, 1.0)),
            }
        )
    site_records = [
        {
            "core": list(getattr(site, "core", ())),
            "neighbors": (
                None
                if getattr(site, "neighbors", None) is None
                else list(getattr(site, "neighbors"))
            ),
        }
        for site in sites
    ]
    payload = {
        "protocol": protocol,
        "sites": site_records,
        "adsorbate": {"conformers": conformer_records},
        "software": _software_versions(),
    }
    return stable_hash(payload)
