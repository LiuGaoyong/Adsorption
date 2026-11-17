"""Basin recall of relaxation runs, and recovery of published configurations.

The analysis reads relaxed structures; it imports no calculator and does not
care which engine produced them. A structure needs ``n_substrate``, a
``label``, a ``converged`` flag and an energy (``mlp_energy``) in its ``info``,
which is what ``adsorption-relax run`` writes.

A *run* is the set of relaxed structures of one placement campaign (one
direction set at one budget); its name has the form ``arm/label``, e.g.
``fibonacci/n200``. The analysis pools the valid structures of all runs of
one system, groups them into minima (:mod:`adsorption.analysis.basins`) and
takes the minima within an energy window of the lowest one as the reference
set. For every run it then reports

* the budget (number of relaxations, usable or not),
* the recall: the share of the reference minima the run reached,
* whether the run reached the pooled lowest energy within 0.1 eV, and
* which published configurations it recovered.

The reference set is a pool, so it can miss minima that no run reached. Two
estimators of the unseen-species problem quantify this: the Good-Turing
sample coverage and the Chao1 lower bound on the number of minima.

A published configuration is described by geometric criteria, not by
coordinates, because most papers print none::

References:
      - name: site-I
        e_ads: -2.38                 # eV, as published (informative)
        label: dissociated           # optional
        contacts:                    # every entry must hold
          - {adsorbate: O, substrate: Ca, r_max: 2.6, n_min: 2}
          - {adsorbate: H, substrate: O, r_max: 1.1, n_min: 1, n_max: 1}

A ``contacts`` entry counts the adsorbate-substrate atom pairs of the two
elements that are closer than ``r_max`` (periodic images included).
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list

from adsorption.analysis.basins import cluster_minima, fingerprint

__all__ = [
    "ENERGY_KEY",
    "analyse_runs",
    "load_run",
    "matches_reference",
    "write_report",
]

ENERGY_KEY = "mlp_energy"
"""``info`` key of the relaxed energy of a structure."""

VALID_LABELS = ("intact", "dissociated", "reconstructed")
"""Labels of structures that count as adsorption minima by default."""

SUCCESS_WINDOW = 0.1
"""A run succeeds when its lowest energy is within this many eV of the pool."""


def load_run(paths: Sequence[str | Path] | str | Path) -> list[Atoms]:
    """Read the relaxed structures of one run.

    Args:
        paths: One ``relaxed.xyz`` file or run directory, or several (the
            shards of one run), in order.

    Returns:
        list[Atoms]: The relaxed structures with their ``info``.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames: list[Atoms] = []
    for path in paths:
        source = Path(path).expanduser()
        if source.is_dir():
            source = source / "relaxed.xyz"
        if not source.exists():
            raise FileNotFoundError(f"No relaxed structures at {source}")
        frames.extend(read(str(source), index=":"))
    return frames


def _number(symbol: Any) -> int:
    """Return the atomic number of a symbol or number."""
    if isinstance(symbol, (int, np.integer)):
        return int(symbol)
    return int(atomic_numbers[str(symbol)])


def matches_reference(
    atoms: Atoms,
    n_substrate: int,
    reference: Mapping[str, Any],
    label: str | None = None,
) -> bool:
    """Return whether a structure satisfies the criteria of a reference.

    Args:
        atoms: The relaxed structure, substrate atoms first.
        n_substrate: Number of substrate atoms.
        reference: The criteria: optional ``label`` and ``contacts``.
        label: The label of the structure; read from ``atoms.info`` when
            ``None``.

    Returns:
        bool: True when every criterion holds.

    Raises:
        ValueError: If the reference states no criterion.
    """
    contacts = list(reference.get("contacts") or [])
    wanted = reference.get("label")
    if not contacts and wanted is None:
        raise ValueError(
            f"Reference {reference.get('name')!r} has no criterion."
        )
    if wanted is not None:
        found = atoms.info.get("label") if label is None else label
        if str(found) != str(wanted):
            return False
    if not contacts:
        return True
    n_substrate = int(n_substrate)
    numbers = np.asarray(atoms.numbers, dtype=int)
    reach = max(float(entry["r_max"]) for entry in contacts)
    first, second, distance = neighbor_list("ijd", atoms, reach)
    cross = (first >= n_substrate) & (second < n_substrate)
    first, second, distance = first[cross], second[cross], distance[cross]
    for entry in contacts:
        mask = (
            (numbers[first] == _number(entry["adsorbate"]))
            & (numbers[second] == _number(entry["substrate"]))
            & (distance <= float(entry["r_max"]))
        )
        count = int(mask.sum())
        if count < int(entry.get("n_min", 1)):
            return False
        upper = entry.get("n_max")
        if upper is not None and count > int(upper):
            return False
    return True


def _coverage(counts: Sequence[int]) -> dict[str, Any]:
    """Return the Good-Turing coverage and the Chao1 estimate."""
    total = int(sum(counts))
    observed = len(counts)
    f1 = sum(1 for value in counts if value == 1)
    f2 = sum(1 for value in counts if value == 2)
    if f2 > 0:
        chao1 = observed + f1 * f1 / (2.0 * f2)
    else:
        chao1 = observed + f1 * (f1 - 1) / 2.0
    return {
        "n_observations": total,
        "n_minima": observed,
        "f1": f1,
        "f2": f2,
        "coverage": (1.0 - f1 / total) if total else float("nan"),
        "chao1": float(chao1),
    }


def analyse_runs(
    runs: Mapping[str, Sequence[Atoms]],
    *,
    window: float = 0.3,
    tau_f: float = 0.1,
    tau_e: float = 0.02,
    r_cut: float = 6.0,
    length: int = 32,
    valid_labels: Sequence[str] = VALID_LABELS,
    require_converged: bool = True,
    references: Sequence[Mapping[str, Any]] = (),
    recall_target: float = 0.95,
) -> dict[str, Any]:
    """Analyse the relaxation runs of one system.

    Args:
        runs: Relaxed structures by run name (``arm/label``).
        window: Energy window of the reference minima above the lowest, eV.
        tau_f: Fingerprint tolerance in Angstrom.
        tau_e: Energy tolerance in eV.
        r_cut: Clamping distance of the fingerprint, Angstrom.
        length: Distances kept per element pair in the fingerprint.
        valid_labels: Labels that count as adsorption minima.
        require_converged: Whether an unconverged relaxation is unusable.
        references: Published configurations (see the module docstring).
        recall_target: Recall that defines the first-passage budget of an
            arm.

    Returns:
        dict[str, Any]: ``reference`` (the pooled minima and the coverage
        estimates), ``runs`` (one row per run), ``arms`` (the smallest
        budget of every arm that reaches ``recall_target``) and
        ``literature`` (per published configuration, the runs that
        recovered it).

    Raises:
        ValueError: If no run holds a usable structure.
    """
    allowed = {str(value) for value in valid_labels}
    owner: list[str] = []
    pool: list[Atoms] = []
    budgets: dict[str, int] = {}
    wasted: dict[str, Counter[str]] = {}
    for name, frames in runs.items():
        budgets[name] = len(frames)
        wasted[name] = Counter()
        for atoms in frames:
            label = str(atoms.info.get("label", "unknown"))
            converged = bool(atoms.info.get("converged", False))
            usable = (
                label in allowed
                and ENERGY_KEY in atoms.info
                and (converged or not require_converged)
            )
            if usable:
                owner.append(name)
                pool.append(atoms)
            else:
                reason = label if label not in allowed else "unconverged"
                wasted[name][reason] += 1
    if not pool:
        raise ValueError("No run holds a usable relaxed structure.")

    sizes = [int(atoms.info["n_substrate"]) for atoms in pool]
    energies = np.array([float(atoms.info[ENERGY_KEY]) for atoms in pool])
    prints = np.array(
        [
            fingerprint(atoms, size, r_cut=r_cut, length=length)
            for atoms, size in zip(pool, sizes)
        ]
    )
    minimum = cluster_minima(prints, energies, tau_f=tau_f, tau_e=tau_e)
    n_minima = int(minimum.max()) + 1
    lowest = np.array(
        [energies[minimum == index].min() for index in range(n_minima)]
    )
    e_zero = float(lowest.min())
    in_window = np.flatnonzero(lowest - e_zero <= float(window))
    reference_ids = set(in_window.tolist())
    hits = Counter(
        int(value) for value in minimum if int(value) in reference_ids
    )

    matched: dict[str, np.ndarray] = {}
    for entry in references:
        matched[str(entry["name"])] = np.array(
            [
                matches_reference(atoms, size, entry)
                for atoms, size in zip(pool, sizes)
            ],
            dtype=bool,
        )

    owners = np.array(owner)
    rows: list[dict[str, Any]] = []
    for name in runs:
        mine = owners == name
        found = {int(value) for value in minimum[mine]} & reference_ids
        best = float(energies[mine].min()) if mine.any() else None
        arm, _, tag = name.partition("/")
        row: dict[str, Any] = {
            "run": name,
            "arm": arm,
            "tag": tag,
            "n_relaxed": budgets[name],
            "n_usable": int(mine.sum()),
            "n_minima_found": len(found),
            "recall": len(found) / len(reference_ids),
            "e_lowest": best,
            "success": bool(
                best is not None and best - e_zero <= SUCCESS_WINDOW
            ),
            "wasted": dict(wasted[name]),
        }
        for key, mask in matched.items():
            row[f"lit:{key}"] = bool((mask & mine).any())
        rows.append(row)

    arms: dict[str, Any] = {}
    for row in rows:
        entry = arms.setdefault(
            row["arm"], {"n_runs": 0, "first_passage_budget": None}
        )
        entry["n_runs"] += 1
        if row["recall"] >= float(recall_target):
            current = entry["first_passage_budget"]
            if current is None or row["n_relaxed"] < current:
                entry["first_passage_budget"] = row["n_relaxed"]

    literature: dict[str, Any] = {}
    for entry in references:
        key = str(entry["name"])
        mask = matched[key]
        ids = sorted({int(value) for value in minimum[mask]})
        literature[key] = {
            "e_ads_published": entry.get("e_ads"),
            "found": bool(mask.any()),
            "minimum_ids": ids,
            "e_lowest": float(energies[mask].min()) if mask.any() else None,
            "runs": [row["run"] for row in rows if row[f"lit:{key}"]],
        }

    return {
        "parameters": {
            "window": float(window),
            "tau_f": float(tau_f),
            "tau_e": float(tau_e),
            "r_cut": float(r_cut),
            "length": int(length),
            "valid_labels": sorted(allowed),
            "require_converged": bool(require_converged),
            "recall_target": float(recall_target),
        },
        "reference": {
            "e_lowest": e_zero,
            "n_minima_all": n_minima,
            "minima": [
                {
                    "id": int(index),
                    "e_rel": float(lowest[index] - e_zero),
                    "n_hits": int(hits[int(index)]),
                }
                for index in in_window
            ],
            **_coverage([hits[int(index)] for index in in_window]),
        },
        "runs": rows,
        "arms": arms,
        "literature": literature,
    }


def _clean(value: Any) -> Any:
    """Replace non-finite floats so the report is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def write_report(report: Mapping[str, Any], out: str | Path) -> Path:
    """Write ``summary.json`` and ``recall.csv`` of one analysis.

    Args:
        report: The result of :func:`analyse_runs`.
        out: Output directory.

    Returns:
        Path: The path of ``summary.json``.
    """
    out_dir = Path(out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.json"
    summary.write_text(
        json.dumps(_clean(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            key: (json.dumps(value) if isinstance(value, dict) else value)
            for key, value in row.items()
        }
        for row in report["runs"]
    ]
    with (out_dir / "recall.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary
