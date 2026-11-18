"""Injectable transport boundary for native application protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

EventObserver = Callable[[dict], None]


class NativeTransport(Protocol):
    """JSON-RPC values and notifications, not provider model requests.

    Implementations observe every incoming/outgoing wire message before making
    it available to a caller. An observer failure prevents further dispatch.
    """

    @property
    def metadata(self) -> dict:
        """Nonsecret runtime facts; never a source identity or admission gate."""

    def set_observer(self, observer: EventObserver) -> None: ...

    async def connect(self, cwd: Path) -> None: ...

    async def request(self, method: str, params: dict) -> dict: ...

    async def notify(self, method: str, params: dict) -> None: ...

    async def next_event(self) -> dict: ...

    async def close(self) -> None: ...
