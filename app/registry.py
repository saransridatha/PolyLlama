"""Model registry, GGUF validation, import safety, and discovery."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database, iso
from .gguf import (
    GGUFError,
    GGUFMetadata,
    estimate_memory_bytes,
    infer_capabilities,
    read_gguf_metadata,
)

RESERVED_ALIASES = {"auto", "fast", "coding", "reasoning", "default", "router"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ALIAS_RE = ID_RE


class RegistryError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(slots=True)
class ModelMetaCache:
    mtime_ns: int
    size: int
    meta: GGUFMetadata
    cached_at: float


class ModelRegistry:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self._cache: dict[str, ModelMetaCache] = {}

    # -- path and validation --------------------------------------------
    def _allowed_roots(self) -> list[Path]:
        return [p.resolve() for p in self.settings.model_root_paths]

    def validate_path(self, raw_path: str, *, require_exists: bool = True) -> Path:
        if not raw_path or "\x00" in raw_path:
            raise RegistryError("A model path is required")
        if len(raw_path) > 4096:
            raise RegistryError("Model path is too long")
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=require_exists)
        except (OSError, RuntimeError) as exc:
            raise RegistryError(f"Invalid model path: {exc}") from exc
        if resolved.suffix.lower() != ".gguf":
            raise RegistryError("Only .gguf files can be registered")
        if require_exists:
            if not resolved.is_file():
                raise RegistryError("Model path is not a regular file")
            if not self.settings.allow_external_model_paths:
                if not any(self._is_within(resolved, root) for root in self._allowed_roots()):
                    roots = ", ".join(str(r) for r in self._allowed_roots())
                    raise RegistryError(
                        "Model path is outside the configured model roots "
                        f"({roots}). Move the file into a model root or set "
                        "MS_ALLOW_EXTERNAL_MODEL_PATHS=true."
                    )
        return resolved

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def metadata(self, path: Path) -> GGUFMetadata:
        try:
            st = path.stat()
        except OSError as exc:
            raise RegistryError(f"Model file is unavailable: {exc}", 404) from exc
        cached = self._cache.get(str(path))
        if cached and cached.mtime_ns == st.st_mtime_ns and cached.size == st.st_size:
            return cached.meta
        try:
            meta = read_gguf_metadata(path)
        except GGUFError as exc:
            raise RegistryError(f"Invalid GGUF model: {exc}", 422) from exc
        self._cache[str(path)] = ModelMetaCache(st.st_mtime_ns, st.st_size, meta, time.time())
        return meta

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-._")
        slug = re.sub(r"-{2,}", "-", slug)[:64]
        return slug or "model"

    def _unique_id(self, base: str) -> str:
        model_id = self._slug(base)
        existing = {m["id"] for m in self.db.list_models()}
        if model_id not in existing:
            return model_id
        for _ in range(100):
            candidate = f"{model_id[:56]}-{uuid.uuid4().hex[:6]}"
            if candidate not in existing:
                return candidate
        return f"model-{uuid.uuid4().hex}"

    def _normalize_aliases(
        self, aliases: list[str] | None, *, model_id: str, path: Path
    ) -> list[str]:
        result: list[str] = []
        other_models = [m for m in self.db.list_models() if m["id"] != model_id]
        other_ids = {m["id"] for m in other_models}
        other_aliases = {alias for m in other_models for alias in m.get("aliases", [])}
        for alias in aliases or []:
            alias = str(alias).strip().lower()
            if not alias:
                continue
            if alias in RESERVED_ALIASES:
                raise RegistryError(f"'{alias}' is a reserved routing alias")
            if not ALIAS_RE.match(alias):
                raise RegistryError(f"Invalid alias '{alias}'")
            if alias == model_id:
                continue
            if alias in other_ids or alias in other_aliases:
                raise RegistryError(f"Alias '{alias}' is already in use", 409)
            if alias not in result:
                result.append(alias)
        # A filename-derived alias is convenient, but never collide with another
        # model or a reserved routing role.
        stem_alias = self._slug(path.stem)
        if (
            stem_alias != model_id
            and stem_alias not in RESERVED_ALIASES
            and stem_alias not in result
        ):
            existing = self.db.find_model(stem_alias)
            if not existing:
                result.append(stem_alias)
        return result[:8]

    @staticmethod
    def _extract_parameter_count(size_label: str) -> int:
        m = re.match(r"^(\d+(?:\.\d+)?)([MBT])$", size_label or "")
        if not m:
            return 0
        return int(float(m.group(1)) * {"M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)])

    def add_model(
        self,
        *,
        path: str,
        name: str | None = None,
        aliases: list[str] | None = None,
        capabilities: list[str] | None = None,
        context_length: int | None = None,
        priority: int = 50,
        residency: str = "on_demand",
        max_concurrency: int | None = None,
        n_gpu_layers: int | None = None,
        threads: int | None = None,
        description: str = "",
        enabled: bool = True,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved = self.validate_path(path)
        existing = self.db.query_one("SELECT id FROM models WHERE path=?", (str(resolved),))
        if existing:
            raise RegistryError(f"Model path is already registered as '{existing['id']}'", 409)
        meta = self.metadata(resolved)
        model_name = (name or meta.name or resolved.stem).strip()[:120]
        model_id = self._unique_id(model_name)
        if not ID_RE.match(model_id):
            raise RegistryError("Unable to derive a safe model identifier")
        max_ctx = min(meta.context_length, self.settings.max_context_length)
        ctx = int(context_length or min(self.settings.default_context_length, max_ctx))
        if ctx < 256:
            raise RegistryError("Context length must be at least 256")
        if ctx > max_ctx:
            raise RegistryError(f"Context length exceeds model capability ({max_ctx})")
        if residency not in ("on_demand", "keep_loaded"):
            raise RegistryError("residency must be on_demand or keep_loaded")
        caps = [
            c.strip().lower()
            for c in (capabilities if capabilities is not None else infer_capabilities(meta, model_name))
            if c.strip()
        ]
        model = {
            "id": model_id,
            "name": model_name,
            "path": str(resolved),
            "aliases": self._normalize_aliases(aliases, model_id=model_id, path=resolved),
            "capabilities": sorted(set(caps)),
            "context_length": ctx,
            "max_context_length": max_ctx,
            "quantization": meta.quantization,
            "parameter_size": meta.parameter_size,
            "parameter_count": meta.parameter_count or self._extract_parameter_count(meta.parameter_size),
            "size_bytes": meta.file_size,
            "priority": max(0, min(100, int(priority))),
            "enabled": enabled,
            "residency": residency,
            "max_concurrency": max(1, min(16, int(max_concurrency or self.settings.model_concurrency))),
            "n_gpu_layers": n_gpu_layers,
            "threads": threads,
            "description": description.strip()[:500],
            "extra_args": [str(a) for a in (extra_args or [])][:32],
            "created_at": iso(),
            "updated_at": iso(),
        }
        self.db.create_model(model)
        self.db.add_event(
            "info",
            "model.registered",
            f"Registered model {model_id}",
            model_id,
            {"size_bytes": meta.file_size, "quantization": meta.quantization},
        )
        return self.db.get_model(model_id)  # type: ignore[return-value]

    def update_model(self, model_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        model = self.db.get_model(model_id)
        if not model:
            raise RegistryError("Model not found", 404)
        update: dict[str, Any] = {}
        for key in ("name", "description", "priority", "enabled", "residency", "max_concurrency"):
            if key in fields and fields[key] is not None:
                update[key] = fields[key]
        if "priority" in update:
            update["priority"] = max(0, min(100, int(update["priority"])))
        if "max_concurrency" in update:
            update["max_concurrency"] = max(1, min(16, int(update["max_concurrency"])))
        if "residency" in update and update["residency"] not in ("on_demand", "keep_loaded"):
            raise RegistryError("residency must be on_demand or keep_loaded")
        if "name" in update:
            update["name"] = str(update["name"]).strip()[:120] or model["name"]
        if "description" in update:
            update["description"] = str(update["description"])[:500]
        if "aliases" in fields and fields["aliases"] is not None:
            update["aliases"] = self._normalize_aliases(
                list(fields["aliases"]), model_id=model_id, path=Path(model["path"])
            )
        if "capabilities" in fields and fields["capabilities"] is not None:
            update["capabilities"] = sorted(
                {str(c).strip().lower() for c in fields["capabilities"] if str(c).strip()}
            )
        if "context_length" in fields and fields["context_length"] is not None:
            ctx = int(fields["context_length"])
            max_ctx = int(model.get("max_context_length") or model["context_length"])
            if ctx < 256 or ctx > max_ctx:
                raise RegistryError(f"Context length must be between 256 and {max_ctx}")
            update["context_length"] = ctx
        if "n_gpu_layers" in fields:
            value = fields["n_gpu_layers"]
            update["n_gpu_layers"] = None if value is None else max(-1, int(value))
        if "threads" in fields:
            value = fields["threads"]
            update["threads"] = None if value is None else max(1, min(256, int(value)))
        if "extra_args" in fields and fields["extra_args"] is not None:
            update["extra_args"] = [str(a) for a in fields["extra_args"]][:32]
        if not update:
            return model
        self.db.update_model(model_id, update)
        return self.db.get_model(model_id)  # type: ignore[return-value]

    def delete_model(self, model_id: str) -> bool:
        exists = self.db.get_model(model_id)
        if not exists:
            raise RegistryError("Model not found", 404)
        deleted = self.db.delete_model(model_id)
        if deleted:
            self._cache.pop(exists["path"], None)
            self.db.add_event(
                "info", "model.unregistered", f"Removed registration for {model_id}", model_id
            )
        return deleted

    def scan(self) -> list[dict[str, Any]]:
        registered = {m["path"] for m in self.db.list_models()}
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for root in self._allowed_roots():
            if not root.is_dir():
                continue
            try:
                candidates = list(root.rglob("*.gguf"))[:500]
            except OSError:
                continue
            for path in candidates:
                try:
                    resolved = path.resolve(strict=True)
                except OSError:
                    continue
                key = str(resolved)
                if key in seen or key in registered or not resolved.is_file():
                    continue
                seen.add(key)
                try:
                    meta = self.metadata(resolved)
                except RegistryError as exc:
                    found.append(
                        {
                            "path": key,
                            "name": path.stem,
                            "valid": False,
                            "error": exc.message,
                            "registered": False,
                        }
                    )
                    continue
                found.append(
                    {
                        "path": key,
                        "name": meta.name or path.stem,
                        "valid": True,
                        "size_bytes": meta.file_size,
                        "quantization": meta.quantization,
                        "parameter_size": meta.parameter_size,
                        "context_length": meta.context_length,
                        "capabilities": infer_capabilities(meta, path.stem),
                        "registered": False,
                    }
                )
        found.sort(key=lambda item: item.get("size_bytes", 0))
        return found

    def warm_metadata_cache(self) -> int:
        """Parse GGUF headers once at startup so first-request routing stays fast."""
        warmed = 0
        for model in self.list():
            try:
                if self.availability(model)["available"]:
                    self.metadata(Path(model["path"]))
                    warmed += 1
            except Exception:  # nosec B112 - one unreadable model must not block warmup
                continue
        return warmed

    def get(self, model_id: str) -> dict[str, Any] | None:
        return self.db.get_model(model_id)

    def find(self, name: str) -> dict[str, Any] | None:
        return self.db.find_model(name)

    def list(self, include_disabled: bool = True) -> list[dict[str, Any]]:
        models = self.db.list_models()
        return models if include_disabled else [m for m in models if m["enabled"]]

    def availability(self, model: dict[str, Any]) -> dict[str, Any]:
        path = Path(model["path"])
        try:
            st = path.stat()
            exists = path.is_file()
        except OSError:
            return {"available": False, "exists": False, "reason": "file missing"}
        if not exists:
            return {"available": False, "exists": False, "reason": "not a regular file"}
        if int(model.get("size_bytes") or 0) and st.st_size != int(model["size_bytes"]):
            return {
                "available": True,
                "exists": True,
                "reason": "file size changed since registration; reload to refresh metadata",
                "changed": True,
            }
        return {"available": True, "exists": True, "reason": ""}

    def estimate_memory(
        self,
        model: dict[str, Any],
        *,
        context_length: int | None = None,
        slots: int | None = None,
        gpu_layers: int = 0,
    ) -> dict[str, Any]:
        meta = self.metadata(Path(model["path"]))
        estimate = estimate_memory_bytes(
            meta,
            context_length=context_length or int(model["context_length"]),
            slots=slots or int(model.get("max_concurrency") or 1),
            gpu_layers=gpu_layers,
        )
        estimate["model_id"] = model["id"]
        estimate["file_size"] = meta.file_size
        return estimate

    def public_card(self, model: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": model["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "modelswitch-local",
            "name": model["name"],
            "aliases": model.get("aliases", []),
            "capabilities": model.get("capabilities", []),
            "context_length": model.get("context_length"),
            "quantization": model.get("quantization"),
            "parameter_size": model.get("parameter_size"),
        }

    def admin_card(self, model: dict[str, Any]) -> dict[str, Any]:
        card = self.public_card(model)
        card.update(
            {
                "path": model["path"],
                "enabled": model["enabled"],
                "priority": model["priority"],
                "residency": model["residency"],
                "max_concurrency": model["max_concurrency"],
                "n_gpu_layers": model.get("n_gpu_layers"),
                "threads": model.get("threads"),
                "description": model.get("description", ""),
                "size_bytes": model.get("size_bytes"),
                "max_context_length": model.get("max_context_length"),
                "parameter_count": model.get("parameter_count"),
                "extra_args": model.get("extra_args", []),
                "created_at": model.get("created_at"),
                "updated_at": model.get("updated_at"),
                "availability": self.availability(model),
            }
        )
        return card
