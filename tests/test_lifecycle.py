from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.hardware import SystemMonitor
from app.lifecycle import ModelManager, ModelUnavailable
from app.registry import ModelRegistry
from app.runtime import AppRuntime
from tests.conftest import TEST_MODEL, model_available


@pytest.fixture
def model_paths(tmp_path: Path):
    if not model_available():
        pytest.skip("models unavailable")
    other = Path("/home/n13/models/SmolLM3-3B-Q4_K_M.gguf")
    if not other.is_file():
        pytest.skip("second model unavailable")
    return TEST_MODEL, other


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_load_is_coalesced(runtime, model_paths):
    model = runtime.registry.add_model(path=str(model_paths[0]), context_length=2048)
    results = await asyncio.gather(
        runtime.manager.load(model["id"]),
        runtime.manager.load(model["id"]),
        runtime.manager.load(model["id"]),
    )
    assert results[0] is results[1] is results[2]
    assert results[0].load_count == 1
    assert results[0].state == "loaded"
    await runtime.manager.unload(model["id"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resident_limit_evicts_idle_model(runtime, model_paths):
    first = runtime.registry.add_model(path=str(model_paths[0]), context_length=2048)
    second = runtime.registry.add_model(
        path=str(model_paths[1]), context_length=2048, name="Second"
    )
    await runtime.manager.load(first["id"])
    assert runtime.manager.is_loaded(first["id"])
    await runtime.manager.load(second["id"])
    assert runtime.manager.is_loaded(second["id"])
    assert not runtime.manager.is_loaded(first["id"])
    assert len(runtime.manager.loaded_models()) <= runtime.settings.max_resident_models
    await runtime.manager.unload(second["id"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_insufficient_memory_rejected(runtime, model_paths):
    model = runtime.registry.add_model(path=str(model_paths[0]), context_length=2048)
    old = runtime.settings.min_free_ram_mb
    runtime.settings.min_free_ram_mb = 10_000_000
    try:
        with pytest.raises(ModelUnavailable) as exc:
            await runtime.manager.load(model["id"])
        assert exc.value.status_code == 503
        assert "memory" in exc.value.message.lower() or "resident" in exc.value.message.lower()
    finally:
        runtime.settings.min_free_ram_mb = old
    assert not runtime.manager.is_loaded(model["id"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_missing_file_rejected(tmp_path: Path, settings: Settings):
    runtime = AppRuntime(settings)
    await runtime.start()
    try:
        model_id = "missing-model"
        runtime.db.create_model(
            {
                "id": model_id,
                "name": "Missing Model",
                "path": str(tmp_path / "does-not-exist.gguf"),
                "aliases": [],
                "capabilities": ["chat"],
                "context_length": 2048,
                "max_context_length": 2048,
                "quantization": "Q4_K_M",
                "parameter_size": "2B",
                "size_bytes": 1234,
                "enabled": True,
            }
        )
        with pytest.raises(ModelUnavailable) as exc:
            await runtime.manager.load(model_id)
        assert exc.value.code == "model_file_missing"
    finally:
        await runtime.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_idle_timeout_unloads(tmp_path: Path):
    if not model_available():
        pytest.skip("model unavailable")
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path),
        allow_external_model_paths=True,
        default_context_length=2048,
        max_resident_models=1,
        backend_idle_timeout=1,
        idle_check_interval=0.1,
        llama_threads=6,
        log_level="WARNING",
    )
    settings.ensure_dirs()
    runtime = AppRuntime(settings)
    await runtime.start()
    try:
        model = runtime.registry.add_model(path=str(TEST_MODEL), context_length=2048)
        await runtime.manager.load(model["id"])
        assert runtime.manager.is_loaded(model["id"])
        await asyncio.sleep(3.2)
        assert not runtime.manager.is_loaded(model["id"])
    finally:
        await runtime.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unload_all_on_shutdown(tmp_path: Path):
    if not model_available():
        pytest.skip("model unavailable")
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path),
        allow_external_model_paths=True,
        default_context_length=2048,
        max_resident_models=1,
        llama_threads=6,
        log_level="WARNING",
    )
    settings.ensure_dirs()
    runtime = AppRuntime(settings)
    await runtime.start()
    model = runtime.registry.add_model(path=str(TEST_MODEL), context_length=2048)
    await runtime.manager.load(model["id"])
    backend = runtime.manager.runtimes[model["id"]].backend
    assert backend and backend.alive
    await runtime.close()
    assert not backend.alive
