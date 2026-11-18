from __future__ import annotations

import struct
from pathlib import Path

import pytest

from app.gguf import GGUFError, estimate_memory_bytes, infer_capabilities, read_gguf_metadata
from tests.conftest import TEST_MODEL, model_available


def test_reads_real_model_metadata():
    if not model_available():
        pytest.skip("real GGUF fixture unavailable")
    meta = read_gguf_metadata(TEST_MODEL)
    assert meta.architecture
    assert meta.context_length >= 2048
    assert meta.block_count > 0
    assert meta.quantization != "unknown"
    assert meta.file_size == TEST_MODEL.stat().st_size
    caps = infer_capabilities(meta, meta.name)
    assert "chat" in caps
    assert "instruct" in caps


def test_memory_estimate_is_conservative():
    if not model_available():
        pytest.skip("real GGUF fixture unavailable")
    meta = read_gguf_metadata(TEST_MODEL)
    estimate = estimate_memory_bytes(meta, context_length=2048, slots=2)
    assert estimate["ram_bytes"] > meta.file_size
    assert estimate["kv_cache_bytes"] > 0
    assert estimate["runtime_overhead_bytes"] >= 256 * 1024 * 1024
    bigger = estimate_memory_bytes(meta, context_length=8192, slots=2)
    assert bigger["kv_cache_bytes"] > estimate["kv_cache_bytes"]


def test_gpu_offload_estimate_splits_weights():
    if not model_available():
        pytest.skip("real GGUF fixture unavailable")
    meta = read_gguf_metadata(TEST_MODEL)
    cpu = estimate_memory_bytes(meta, context_length=2048, slots=1, gpu_layers=0)
    split = estimate_memory_bytes(
        meta, context_length=2048, slots=1, gpu_layers=max(1, meta.block_count // 2)
    )
    assert split["vram_bytes"] > 0
    assert split["ram_bytes"] < cpu["ram_bytes"]


def test_rejects_bad_magic(tmp_path: Path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOTGGUF" + b"\0" * 64)
    with pytest.raises(GGUFError, match="magic"):
        read_gguf_metadata(path)


def test_rejects_truncated_header(tmp_path: Path):
    path = tmp_path / "short.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3))
    with pytest.raises(GGUFError):
        read_gguf_metadata(path)


def test_rejects_missing_file(tmp_path: Path):
    with pytest.raises(GGUFError, match="not found"):
        read_gguf_metadata(tmp_path / "missing.gguf")


def test_rejects_unsupported_version(tmp_path: Path):
    path = tmp_path / "v99.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 99) + b"\0" * 16)
    with pytest.raises(GGUFError, match="version"):
        read_gguf_metadata(path)
