"""Minimal, dependency-free GGUF metadata reader.

We intentionally do not parse tensors. Reading the key/value header is enough to
make resource-aware scheduling decisions and populate model cards without
loading gigabytes of weights.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# GGML file type ids (general.file_type)
QUANTIZATION_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_2",
    5: "Q4_3",
    6: "Q5_0",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
}

_SCALAR_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}
_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<B",
    10: "<Q",
    11: "<q",
    12: "<d",
}


class GGUFError(ValueError):
    pass


@dataclass(slots=True)
class GGUFMetadata:
    path: str
    file_size: int
    version: int
    architecture: str = "unknown"
    name: str = ""
    context_length: int = 4096
    block_count: int = 0
    embedding_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    parameter_count: int = 0
    file_type: int | None = None
    quantization: str = "unknown"
    parameter_size: str = "unknown"
    chat_template: bool = False
    tools: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        if self.key_length:
            return self.key_length
        if self.head_count:
            return max(1, self.embedding_length // self.head_count)
        return 0

    def kv_cache_bytes(self, context_length: int, slots: int = 1) -> int:
        """Conservative f16 KV-cache estimate, including both K and V."""
        if not self.block_count or not self.head_dim:
            return 0
        kv_heads = self.head_count_kv or self.head_count or 1
        per_token = 2 * self.block_count * kv_heads * self.head_dim * 2
        # Padding/graph overhead and per-slot allocator slack.
        return int(per_token * context_length * max(1, slots) * 1.20)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "name": self.name,
            "context_length": self.context_length,
            "block_count": self.block_count,
            "embedding_length": self.embedding_length,
            "head_count": self.head_count,
            "head_count_kv": self.head_count_kv,
            "parameter_count": self.parameter_count,
            "quantization": self.quantization,
            "parameter_size": self.parameter_size,
            "file_size": self.file_size,
            "chat_template": self.chat_template,
            "tools": self.tools,
        }


class _Reader:
    def __init__(self, fh: BinaryIO, file_size: int):
        self.fh = fh
        self.file_size = file_size

    def read(self, n: int) -> bytes:
        if n < 0 or n > self.file_size:
            raise GGUFError("invalid metadata length")
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFError("truncated GGUF file")
        return data

    def scalar(self, type_id: int) -> Any:
        fmt = _SCALAR_FORMATS.get(type_id)
        if not fmt:
            raise GGUFError(f"unsupported scalar type {type_id}")
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = struct.unpack("<Q", self.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise GGUFError("unreasonable string length in GGUF metadata")
        return self.read(length).decode("utf-8", errors="replace")

    def skip(self, n: int) -> None:
        if n < 0:
            raise GGUFError("invalid skip length")
        pos = self.fh.tell()
        if pos + n > self.file_size:
            raise GGUFError("truncated GGUF file")
        self.fh.seek(pos + n)

    def skip_value(self, type_id: int, depth: int = 0) -> None:
        if depth > 4:
            raise GGUFError("metadata nesting too deep")
        if type_id == 8:
            length = struct.unpack("<Q", self.read(8))[0]
            if length > 16 * 1024 * 1024:
                raise GGUFError("unreasonable string length in GGUF metadata")
            self.skip(length)
            return
        if type_id == 9:
            elem_type = struct.unpack("<I", self.read(4))[0]
            count = struct.unpack("<Q", self.read(8))[0]
            if count > 100_000_000:
                raise GGUFError("unreasonable array length in GGUF metadata")
            size = _SCALAR_SIZES.get(elem_type)
            if size:
                self.skip(size * count)
                return
            for _ in range(count):
                self.skip_value(elem_type, depth + 1)
            return
        size = _SCALAR_SIZES.get(type_id)
        if not size:
            raise GGUFError(f"unsupported metadata type {type_id}")
        self.skip(size)

    def value(self, type_id: int, depth: int = 0) -> Any:
        if depth > 4:
            raise GGUFError("metadata nesting too deep")
        if type_id == 8:
            return self.string()
        if type_id == 9:
            elem_type = struct.unpack("<I", self.read(4))[0]
            count = struct.unpack("<Q", self.read(8))[0]
            if count > 100_000_000:
                raise GGUFError("unreasonable array length in GGUF metadata")
            # Large arrays are never needed for routing; skip fixed-width arrays
            # with a seek, and stream variable-width arrays in bounded chunks.
            if elem_type in _SCALAR_SIZES:
                self.read(_SCALAR_SIZES[elem_type] * count)
                return None
            values = []
            for _ in range(min(count, 4096)):
                values.append(self.value(elem_type, depth + 1))
            if count > 4096:
                for _ in range(count - 4096):
                    self.value(elem_type, depth + 1)
                return values[:32]
            return values
        return self.scalar(type_id)


_INTERESTING_SUFFIXES = (
    "context_length",
    "block_count",
    "embedding_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "expert_count",
    "expert_used_count",
    "parameter_count",
)


def _iter_metadata(fh: BinaryIO, reader: _Reader, kv_count: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    # We only keep bounded scalar values. Arrays (tokens, scores) are skipped.
    for _ in range(kv_count):
        key = reader.string()
        type_id = struct.unpack("<I", reader.read(4))[0]
        keep = (
            key in _INTERESTING_KEYS
            or key.startswith("general.")
            or key.endswith(_INTERESTING_SUFFIXES)
        )
        if type_id == 9 and not key.endswith("attention.head_count_kv"):
            # Tokenizer/tag arrays can contain hundreds of thousands of entries and
            # are irrelevant to routing. Seek past them without decoding.
            reader.skip_value(type_id)
            continue
        value = reader.value(type_id)
        if value is not None and keep:
            metadata[key] = value
    return metadata


_INTERESTING_KEYS = {
    "general.name",
    "general.architecture",
    "general.file_type",
    "general.parameter_count",
    "general.quantization_version",
    "general.tags",
    "general.description",
    "tokenizer.chat_template",
    "tokenizer.ggml.add_generation_prompt",
}


def _arch_value(metadata: dict[str, Any], arch: str, suffix: str) -> Any:
    return metadata.get(f"{arch}.{suffix}")


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, list) and value and isinstance(value[0], int):
        return int(value[0])
    return default


def _infer_parameter_size(name: str, parameter_count: int) -> str:
    if parameter_count > 0:
        for unit, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if parameter_count >= divisor:
                return f"{parameter_count / divisor:.1f}{unit}".replace(".0", "")
    match = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", name)
    if match:
        return f"{match.group(1)}B"
    match = re.search(r"(\d+(?:\.\d+)?)[bB][_-]", name)
    if match:
        return f"{match.group(1)}B"
    return "unknown"


def read_gguf_metadata(path: Path | str) -> GGUFMetadata:
    path = Path(path)
    if not path.is_file():
        raise GGUFError(f"Model file not found: {path}")
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if fh.read(4) != GGUF_MAGIC:
                raise GGUFError("Not a GGUF file (bad magic)")
            version = struct.unpack("<I", fh.read(4))[0]
            if version not in (2, 3):
                raise GGUFError(f"Unsupported GGUF version {version}")
            if size < 24:
                raise GGUFError("Corrupted GGUF header")
            _tensor_count = struct.unpack("<Q", fh.read(8))[0]
            kv_count = struct.unpack("<Q", fh.read(8))[0]
            if kv_count > 1_000_000:
                raise GGUFError("Unreasonable GGUF metadata count")
            reader = _Reader(fh, size)
            metadata = _iter_metadata(fh, reader, kv_count)
    except OSError as exc:
        raise GGUFError(f"Cannot read model file: {exc}") from exc

    arch = str(metadata.get("general.architecture") or "unknown")
    file_type = metadata.get("general.file_type")
    file_type_int = _as_int(file_type, -1)
    quantization = QUANTIZATION_TYPES.get(file_type_int, "unknown")
    if quantization == "unknown":
        name = str(metadata.get("general.name") or path.stem)
        m = re.search(r"(IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*|BF16|F16|F32)", name, re.I)
        if m:
            quantization = m.group(1).upper()
    parameter_count = _as_int(metadata.get("general.parameter_count"), 0)
    if not parameter_count:
        # Some converters store it under the architecture.
        parameter_count = _as_int(_arch_value(metadata, arch, "parameter_count"), 0)
    context_length = _as_int(_arch_value(metadata, arch, "context_length"), 4096) or 4096
    head_count = _as_int(_arch_value(metadata, arch, "attention.head_count"), 0)
    head_count_kv = _as_int(
        metadata.get(f"{arch}.attention.head_count_kv")
        or _arch_value(metadata, arch, "attention.head_count_kv"),
        0,
    )
    if not head_count_kv:
        # GQA models sometimes expose this only via the first layer array; the
        # metadata reader preserves a short list.
        raw_kv = _arch_value(metadata, arch, "attention.head_count_kv")
        if isinstance(raw_kv, list) and raw_kv:
            head_count_kv = _as_int(raw_kv[0], 0)
    name = str(metadata.get("general.name") or path.stem)
    template = str(metadata.get("tokenizer.chat_template") or "")
    meta = GGUFMetadata(
        path=str(path),
        file_size=size,
        version=version,
        architecture=arch,
        name=name,
        context_length=max(256, context_length),
        block_count=_as_int(_arch_value(metadata, arch, "block_count"), 0),
        embedding_length=_as_int(_arch_value(metadata, arch, "embedding_length"), 0),
        head_count=head_count,
        head_count_kv=head_count_kv,
        key_length=_as_int(_arch_value(metadata, arch, "attention.key_length"), 0),
        value_length=_as_int(_arch_value(metadata, arch, "attention.value_length"), 0),
        parameter_count=parameter_count,
        file_type=file_type_int if file_type_int >= 0 else None,
        quantization=quantization,
        parameter_size=_infer_parameter_size(f"{name} {path.stem}", parameter_count),
        chat_template=bool(template),
        tools="tool" in template.lower(),
        raw=metadata,
    )
    return meta


def infer_capabilities(meta: GGUFMetadata, model_name: str = "") -> list[str]:
    text = f"{meta.name} {model_name}".lower()
    caps = {
        "chat",
        "general",
        "summarization",
        "extraction",
        "translation",
        "creative",
        "json",
        "classification",
    }
    if re.search(r"cod(er|e)|starcoder|deepseek-coder|qwen.*coder", text):
        caps.update({"coding", "code_generation", "debugging"})
    if re.search(r"reason|thinking|think|r1|qwq|o1|o3|deepseek-v4", text):
        caps.add("reasoning")
    if re.search(r"math|deepseek-math|qwen.*math", text):
        caps.add("math")
    is_embedding = bool(re.search(r"embed|bge|e5-|gte-|nomic-", text))
    if is_embedding:
        # Embedding-only GGUFs are launched with llama.cpp --embeddings, which
        # restricts the server to the embeddings endpoint. Never advertise chat.
        return ["embeddings"]
    if re.search(r"vision|vl|llava|multimodal|omni", text):
        caps.add("vision")
    if meta.context_length >= 16384:
        caps.add("long_context")
    if meta.parameter_count and meta.parameter_count <= 7_000_000_000:
        caps.add("fast")
    if meta.chat_template:
        caps.add("instruct")
    return sorted(caps)


def estimate_memory_bytes(
    meta: GGUFMetadata,
    *,
    context_length: int,
    slots: int = 1,
    gpu_layers: int = 0,
) -> dict[str, int]:
    """Heuristic resident-memory estimate used only for admission control."""
    weights = meta.file_size
    kv = meta.kv_cache_bytes(context_length, slots)
    if meta.block_count and gpu_layers:
        offloaded_fraction = min(1.0, gpu_layers / max(1, meta.block_count))
    else:
        offloaded_fraction = 0.0
    # Activations, graph buffers, tokenizer, and allocator overhead.
    overhead = max(256 * 1024 * 1024, int(weights * 0.075))
    cpu_weights = int(weights * (1.0 - offloaded_fraction))
    vram_weights = int(weights * offloaded_fraction)
    return {
        "weights_bytes": weights,
        "kv_cache_bytes": kv,
        "runtime_overhead_bytes": overhead,
        "ram_bytes": cpu_weights + kv + overhead,
        "vram_bytes": vram_weights + (kv if offloaded_fraction > 0 else 0),
        "offloaded_fraction": int(offloaded_fraction * 1000),
    }
