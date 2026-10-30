"""Adapters for the ``graphatoms`` site enumerator and six-class labeller.

Graph-based core-set enumeration and the automatic site taxonomy are
implemented upstream in ``graphatoms``, not in this package. The campaign
consumes them through two small protocols:

* ``CoreSetProvider.propose(atoms) -> Sequence[SiteSpec]``
* ``SiteLabeller.label(atoms, site) -> str | None``

The factories below look for the upstream entry points. When the installed
``graphatoms`` does not expose them, they raise a clear ``NotImplementedError``
instead of silently falling back to a private implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module

from .protocol.spec import CoreSetProvider, SiteLabeller, SiteSpec

__all__ = [
    "CallableCoreSetProvider",
    "CallableSiteLabeller",
    "graphatoms_core_provider",
    "graphatoms_site_labeller",
]


class CallableCoreSetProvider:
    """Wrap a callable as a :class:`CoreSetProvider`.

    Args:
        function: A callable ``atoms -> iterable`` returning ``SiteSpec``
            objects or raw core-index iterables.
    """

    def __init__(self, function: Callable[..., Iterable[object]]) -> None:
        """Store the callable."""
        self.function = function

    def propose(self, atoms: object) -> tuple[SiteSpec, ...]:
        """Return the sites proposed by the callable.

        Args:
            atoms: The substrate.

        Returns:
            tuple[SiteSpec, ...]: The candidate sites.

        Raises:
            ValueError: If the callable returns no site.
        """
        produced = tuple(self.function(atoms))
        if not produced:
            raise ValueError(
                "The core-set provider returned no site; a site list must "
                "not be empty."
            )
        sites: list[SiteSpec] = []
        for item in produced:
            if isinstance(item, SiteSpec):
                sites.append(item)
            else:
                sites.append(SiteSpec(core=tuple(int(index) for index in item)))  # type: ignore[arg-type]
        return tuple(sites)


class CallableSiteLabeller:
    """Wrap a callable as a :class:`SiteLabeller`.

    Args:
        function: A callable ``(atoms, site) -> str | None``.
    """

    def __init__(self, function: Callable[..., str | None]) -> None:
        """Store the callable."""
        self.function = function

    def label(self, atoms: object, site: SiteSpec) -> str | None:
        """Return the label of a site."""
        return self.function(atoms, site)


_CORE_PROVIDER_CANDIDATES = (
    ("graphatoms.site", "propose_cores"),
    ("graphatoms.system", "propose_cores"),
    ("graphatoms.sites", "propose_cores"),
)

_LABELLER_CANDIDATES = (
    ("graphatoms.site", "label_site"),
    ("graphatoms.site", "classify_site"),
    ("graphatoms.sites", "label_site"),
)


def _resolve(
    candidates: Iterable[tuple[str, str]],
) -> Callable[..., object] | None:
    for module_name, attribute in candidates:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        function = getattr(module, attribute, None)
        if callable(function):
            return function
    return None


def graphatoms_core_provider() -> CoreSetProvider:
    """Return the ``graphatoms`` core-set provider.

    Returns:
        CoreSetProvider: The upstream provider.

    Raises:
        NotImplementedError: If the installed ``graphatoms`` does not expose
            ``propose_cores``; pass an explicit site list or a custom
            provider instead.
    """
    function = _resolve(_CORE_PROVIDER_CANDIDATES)
    if function is None:
        raise NotImplementedError(
            "The installed graphatoms does not expose a core-set enumerator "
            "(expected graphatoms.site.propose_cores). Upgrade graphatoms or "
            "supply the site list in the protocol; this package does not "
            "implement graph-based enumeration."
        )
    return CallableCoreSetProvider(function)


def graphatoms_site_labeller() -> SiteLabeller:
    """Return the ``graphatoms`` six-class site labeller.

    Returns:
        SiteLabeller: The upstream labeller.

    Raises:
        NotImplementedError: If the installed ``graphatoms`` does not expose
            ``label_site``; labels are optional and the geometry is unchanged
            without them.
    """
    function = _resolve(_LABELLER_CANDIDATES)
    if function is None:
        raise NotImplementedError(
            "The installed graphatoms does not expose a site labeller "
            "(expected graphatoms.site.label_site). Labels are optional; this "
            "package does not implement the six-class taxonomy."
        )
    return CallableSiteLabeller(function)
