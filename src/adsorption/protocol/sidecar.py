"""Sidecar records: one JSON object per candidate.

The record is fixed by the construction and carries the index, the radices,
the geometry parameters, the screening decision, the evaluation result, and
the parameter hash. It lives next to the structure files (``.xyz``) and is
appended as JSON Lines so a run can be resumed and audited.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

__all__ = ["SidecarRecord", "SidecarWriter", "read_sidecar"]


@dataclass
class SidecarRecord:
    """The audit record of one candidate.

    Attributes:
        kappa: One-based candidate index.
        site: Zero-based site index.
        conformer: Zero-based conformer index.
        i: One-based admissible-direction index.
        j: One-based orientation index.
        k: One-based distance index.
        n_oa: Number of admissible directions of the site.
        offset: Offset of the site variant in the index space.
        r_s: Direction-sphere radius in Angstrom.
        n_filter: Size of the steric filter set ``P(s)``.
        distance: Anchor-centre distance in Angstrom.
        param_hash: SHA-256 of the parameter set.
        structure: Relative path of the structure file, when written.
        e_rep: Repulsive Lennard-Jones energy, when screened.
        r_close: Closest interatomic distance, when screened.
        lj_pass: Whether the candidate passed the screen.
        energy: Relaxed or single-point energy, when evaluated.
        n_stages: Number of converged relaxation stages.
        converged: Whether the evaluation converged.
    """

    kappa: int
    site: int
    conformer: int
    i: int
    j: int
    k: int
    n_oa: int
    offset: int
    r_s: float
    n_filter: int
    distance: float
    param_hash: str
    structure: str | None = None
    e_rep: float | None = None
    r_close: float | None = None
    lj_pass: bool | None = None
    energy: float | None = None
    n_stages: int | None = None
    converged: bool | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the record as a plain dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Return the compact JSON form of the record."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class SidecarWriter:
    """Append sidecar records to a JSON Lines file.

    Args:
        path: Destination path. Parent directories are created on open.
    """

    def __init__(self, path: str | Path) -> None:
        """Store the destination path."""
        self.path = Path(path)
        self._file = None

    def __enter__(self) -> SidecarWriter:
        """Open the file and return the writer."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the file."""
        self.close()

    def write(self, record: SidecarRecord) -> None:
        """Append one record.

        Args:
            record: The record to write.

        Raises:
            RuntimeError: If the writer is not open.
        """
        if self._file is None:
            raise RuntimeError("The sidecar writer is not open.")
        self._file.write(record.to_json() + "\n")

    def close(self) -> None:
        """Close the file if it is open."""
        if self._file is not None:
            self._file.close()
            self._file = None


def read_sidecar(path: str | Path) -> list[SidecarRecord]:
    """Read a sidecar file into records.

    Args:
        path: Path of the JSON Lines file.

    Returns:
        list[SidecarRecord]: The records in file order.

    Raises:
        ValueError: If a line does not match the record schema.
    """
    known = {field.name for field in fields(SidecarRecord)}
    records: list[SidecarRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            unknown = set(payload) - known
            if unknown:
                raise ValueError(
                    f"Unknown fields on line {line_number} of {path}: "
                    f"{sorted(unknown)}."
                )
            records.append(SidecarRecord(**payload))
    return records
