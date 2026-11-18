from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from app.api.admin import create_admin_app
from app.api.public import create_public_app
from app.config import Settings
from app.runtime import AppRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_MODEL = Path(
    os.environ.get("MS_TEST_MODEL", "/home/n13/models/Qwen3.5-2B_Q4_K_M.gguf")
)
TEST_CODER_MODEL = Path(
    os.environ.get("MS_TEST_CODER_MODEL", "/home/n13/Downloads/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
)
TEST_EMBED_MODEL = Path(
    os.environ.get("MS_TEST_EMBED_MODEL", str(REPO_ROOT / "models" / "nomic-embed-text-v1.5.Q4_K_M.gguf"))
)


def model_available(path: Path = TEST_MODEL) -> bool:
    return path.is_file()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path / "models"),
        allow_external_model_paths=True,
        max_resident_models=1,
        default_context_length=2048,
        max_context_length=8192,
        backend_idle_timeout=300,
        queue_timeout=30,
        request_timeout=120,
        llama_threads=6,
        log_json=False,
        log_level="WARNING",
        environment="test",
    )
    settings.ensure_dirs()
    (tmp_path / "models").mkdir(exist_ok=True)
    return settings


@pytest_asyncio.fixture
async def runtime(settings: Settings):
    runtime = AppRuntime(settings)
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.close()


@pytest_asyncio.fixture
async def keys(runtime: AppRuntime):
    admin, admin_raw = runtime.keys.create_key(
        name="test-admin",
        role="admin",
        permissions={"inference": True},
        rate_limit_rpm=600,
        rate_limit_tpm=1_000_000,
    )
    inference, inference_raw = runtime.keys.create_key(
        name="test-inference",
        role="inference",
        permissions={"models": ["*"], "allow_streaming": True, "allow_embeddings": True},
        rate_limit_rpm=300,
        rate_limit_tpm=500_000,
    )
    return {
        "admin": (admin, admin_raw),
        "inference": (inference, inference_raw),
        "admin_headers": {"Authorization": f"Bearer {admin_raw}"},
        "inference_headers": {"Authorization": f"Bearer {inference_raw}"},
    }


@pytest_asyncio.fixture
async def public_client(runtime: AppRuntime):
    app = create_public_app(runtime)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://public.test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def admin_client(runtime: AppRuntime):
    app = create_admin_app(runtime)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://admin.test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def loaded_model(runtime: AppRuntime, keys):
    if not model_available():
        pytest.skip(f"test GGUF not available at {TEST_MODEL}")
    model = runtime.registry.add_model(
        path=str(TEST_MODEL),
        name="Test Qwen 2B",
        context_length=2048,
        max_concurrency=2,
        priority=60,
    )
    return model

