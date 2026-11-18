from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.db import Database
from app.registry import ModelRegistry, RegistryError
from tests.conftest import TEST_MODEL, model_available


@pytest.fixture
def registry(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path / "models"),
        allow_external_model_paths=True,
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    yield ModelRegistry(db, settings)
    db.close()


def test_registers_valid_model(registry: ModelRegistry):
    if not model_available():
        pytest.skip("model unavailable")
    model = registry.add_model(path=str(TEST_MODEL), context_length=2048)
    assert model["enabled"] is True
    assert model["quantization"] != "unknown"
    assert model["context_length"] == 2048
    assert model["max_context_length"] >= 2048
    assert "chat" in model["capabilities"]
    assert model["aliases"]


def test_duplicate_path_rejected(registry: ModelRegistry):
    if not model_available():
        pytest.skip("model unavailable")
    registry.add_model(path=str(TEST_MODEL), context_length=2048)
    with pytest.raises(RegistryError, match="already registered"):
        registry.add_model(path=str(TEST_MODEL), context_length=2048)


def test_invalid_extension_rejected(registry: ModelRegistry, tmp_path: Path):
    bad = tmp_path / "model.bin"
    bad.write_bytes(b"GGUF")
    with pytest.raises(RegistryError, match="gguf"):
        registry.add_model(path=str(bad))


def test_missing_file_rejected(registry: ModelRegistry, tmp_path: Path):
    with pytest.raises(RegistryError):
        registry.add_model(path=str(tmp_path / "missing.gguf"))


def test_corrupt_gguf_rejected(registry: ModelRegistry, tmp_path: Path):
    bad = tmp_path / "models" / "corrupt.gguf"
    bad.parent.mkdir(exist_ok=True)
    import struct

    bad.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    )
    with pytest.raises(RegistryError, match="Invalid GGUF"):
        registry.add_model(path=str(bad))


def test_path_outside_roots_blocked(tmp_path: Path):
    if not model_available():
        pytest.skip("model unavailable")
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path / "only-here"),
        allow_external_model_paths=False,
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    registry = ModelRegistry(db, settings)
    with pytest.raises(RegistryError, match="outside"):
        registry.add_model(path=str(TEST_MODEL), context_length=2048)
    db.close()


def test_traversal_path_blocked(registry: ModelRegistry):
    with pytest.raises(RegistryError):
        registry.add_model(path="/etc/passwd")
    with pytest.raises(RegistryError):
        registry.add_model(path="../../etc/shadow.gguf")


def test_reserved_aliases_rejected(registry: ModelRegistry):
    if not model_available():
        pytest.skip("model unavailable")
    with pytest.raises(RegistryError, match="reserved"):
        registry.add_model(path=str(TEST_MODEL), context_length=2048, aliases=["auto"])


def test_alias_normalization_and_update(registry: ModelRegistry):
    if not model_available():
        pytest.skip("model unavailable")
    model = registry.add_model(
        path=str(TEST_MODEL), context_length=2048, aliases=["Workhorse", "workhorse"]
    )
    assert "workhorse" in model["aliases"]
    assert model["aliases"].count("workhorse") == 1
    with pytest.raises(RegistryError, match="Invalid alias"):
        registry.update_model(model["id"], {"aliases": ["bad alias"]})
    with pytest.raises(RegistryError):
        registry.update_model(model["id"], {"aliases": ["coding"]})
    updated = registry.update_model(model["id"], {"priority": 99, "enabled": False})
    assert updated["priority"] == 99
    assert updated["enabled"] is False


def test_context_bounds(registry: ModelRegistry):
    if not model_available():
        pytest.skip("model unavailable")
    model = registry.add_model(path=str(TEST_MODEL), context_length=2048)
    with pytest.raises(RegistryError):
        registry.update_model(model["id"], {"context_length": 12})
    with pytest.raises(RegistryError):
        registry.update_model(model["id"], {"context_length": 10_000_000})


def test_alias_collision_across_models(registry: ModelRegistry):
    if not model_available():
        pytest.skip("model unavailable")
    first = registry.add_model(path=str(TEST_MODEL), context_length=2048, aliases=["shared"])
    second_path = Path("/home/n13/models/SmolLM3-3B-Q4_K_M.gguf")
    if not second_path.is_file():
        pytest.skip("second model unavailable")
    with pytest.raises(RegistryError, match="already in use"):
        registry.add_model(path=str(second_path), context_length=2048, aliases=["shared"])


def test_delete_preserves_file(registry: ModelRegistry):
    if not model_available():
        pytest.skip("model unavailable")
    model = registry.add_model(path=str(TEST_MODEL), context_length=2048)
    assert registry.delete_model(model["id"])
    assert TEST_MODEL.exists()
    assert registry.get(model["id"]) is None


def test_scan_discovers_models(tmp_path: Path):
    if not model_available():
        pytest.skip("model unavailable")
    import os
    from app.db import Database

    root = Path(__file__).resolve().parent.parent / ".cache" / "scan-root"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "scan-me.gguf"
    if target.exists():
        target.unlink()
    try:
        os.link(TEST_MODEL, target)
    except OSError:
        # The sandbox may block hard links even on the same apparent device.
        import shutil

        shutil.copyfile(TEST_MODEL, target)
    settings = Settings(
        data_dir=tmp_path / "scan-data",
        model_roots=str(root),
        allow_external_model_paths=False,
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    try:
        scan_registry = ModelRegistry(db, settings)
        found = scan_registry.scan()
        discovered = next(item for item in found if item["path"] == str(target))
        assert discovered["valid"] is True
        assert "chat" in discovered["capabilities"]
    finally:
        db.close()
        target.unlink(missing_ok=True)
