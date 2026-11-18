"""Two-listener server runner.

The public OpenAI API and the admin dashboard intentionally bind different ports.
Running both ASGI apps in one process keeps model residency and metrics coherent
without duplicating the inference lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

import uvicorn

from .api.admin import create_admin_app
from .api.public import create_public_app
from .config import Settings
from .logging_setup import configure_logging
from .runtime import AppRuntime

log = logging.getLogger("modelswitch.server")


class _NoSignalServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):  # type: ignore[override]
        yield


async def serve(settings: Settings) -> int:
    configure_logging(settings.log_level, settings.log_json)
    runtime = AppRuntime(settings)
    if runtime.db.count_api_keys() == 0:
        print(
            "\nModelSwitch is not initialized: no API keys exist.\n"
            "Run:  .venv/bin/python -m app init   (or ./scripts/start.sh)\n",
            file=sys.stderr,
        )
        runtime.db.close()
        return 2
    if not runtime.llama_binary:
        log.warning(
            "llama-server binary was not found; the gateway will start but inference "
            "will return backend_unavailable until it is installed."
        )
    public_app = create_public_app(runtime)
    admin_app = create_admin_app(runtime)
    public_config = uvicorn.Config(
        public_app,
        host=settings.public_host,
        port=settings.public_port,
        log_config=None,
        access_log=False,
        lifespan="off",
        proxy_headers=settings.trust_proxy,
        server_header=False,
        timeout_graceful_shutdown=15,
    )
    admin_config = uvicorn.Config(
        admin_app,
        host=settings.admin_host,
        port=settings.admin_port,
        log_config=None,
        access_log=False,
        lifespan="off",
        proxy_headers=settings.trust_proxy,
        server_header=False,
        timeout_graceful_shutdown=15,
    )
    public_server = _NoSignalServer(public_config)
    admin_server = _NoSignalServer(admin_config)
    shutdown = asyncio.Event()

    def request_shutdown(*_: Any) -> None:
        shutdown.set()
        public_server.should_exit = True
        admin_server.should_exit = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, request_shutdown)

    await runtime.start()
    tasks = [
        asyncio.create_task(public_server.serve(), name="public-api"),
        asyncio.create_task(admin_server.serve(), name="admin-api"),
    ]
    print(f"\n  ModelSwitch {settings.version}")
    print(f"  Public API : {settings.public_url('/v1')}")
    print(f"  Dashboard  : {settings.admin_url('/admin')}")
    print(f"  Health     : {settings.public_url('/healthz')}")
    print(f"  llama.cpp  : {runtime.llama_binary or 'NOT FOUND'}")
    print(f"  Hardware   : {runtime.hardware.llama_backend}\n", flush=True)
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc:
                log.error("listener stopped with error: %s", exc)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # If one listener exits unexpectedly, stop the other rather than running half a gateway.
        if not shutdown.is_set():
            request_shutdown()
        return 0
    finally:
        public_server.should_exit = True
        admin_server.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
        await runtime.close()
        print("ModelSwitch stopped.", flush=True)


def main(settings: Settings | None = None) -> int:
    return asyncio.run(serve(settings or Settings()))


if __name__ == "__main__":
    raise SystemExit(main())
