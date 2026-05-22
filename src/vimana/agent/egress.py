"""Egress allowlist used by the Vimana agent.

The autonomous loop does not call real HTTP endpoints in the demo, so
the allowlist is mostly belt-and-braces: it guarantees that if a
future planner module decides to hit a live API, the host has to be
declared up front.

The contract mirrors birddog.Birddog. When birddog is installed (it is,
in this venv), we use it for real. When it is not, the fallback is a
tiny in-process check that raises the same DomainDeniedError shape. We
keep the fallback so vimana itself does not hard-fail import if a
downstream user has not installed birddog yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

try:  # pragma: no cover - import-time branch
    from birddog import DomainDeniedError
    _BIRDDOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BIRDDOG_AVAILABLE = False

    class DomainDeniedError(Exception):  # type: ignore[no-redef]
        """Raised when an egress target is not in the allowlist."""


_DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "metrics.simulated.local",
        "api.simulated.local",
    }
)


@dataclass
class EgressGuard:
    """Allowlist guard for the agent's outbound calls.

    Use `assert_allowed(url)` before making a request. If the host is
    not in the allowlist, DomainDeniedError is raised. The agent
    surfaces this to the operator instead of swallowing it: a rogue
    decision target is a real safety event.
    """

    allowed: frozenset[str] = field(default_factory=lambda: _DEFAULT_ALLOWLIST)

    def assert_allowed(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        if host in self.allowed:
            return
        raise DomainDeniedError(
            f"host {host!r} not in allowlist {sorted(self.allowed)!r}"
        )

    def host_allowed(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host in self.allowed

    @property
    def backend(self) -> str:
        return "birddog" if _BIRDDOG_AVAILABLE else "inline"
