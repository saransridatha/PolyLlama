"""Inference backend contract.

llama.cpp is the only shipping implementation, but the lifecycle manager depends
only on this protocol. A future backend (for example vLLM or a remote provider)
must provide start/stop, a loopback base URL, health, and process stats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    process: Any
    port: int | None
    log_path: Path | None
    version: str

    @property
    def alive(self) -> bool: ...

    @property
    def base_url(self) -> str: ...

    async def start(self, config: Any) -> None: ...

    async def stop(self, *, force: bool = False) -> None: ...

    async def health(self) -> dict[str, Any]: ...

    def process_stats(self) -> dict[str, Any]: ...

    def read_log_tail(self, lines: int = 80) -> str: ...

    async def close(self) -> None: ...
