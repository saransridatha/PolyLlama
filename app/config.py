"""Runtime configuration for ModelSwitch.

All knobs are environment-first (``MS_*``) with a local ``.env`` fallback. The
defaults are deliberately conservative: loopback-only listeners, a small resident
model budget, and no remote tunnel auto-start.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _split_paths(raw: str) -> list[Path]:
    paths: list[Path] = []
    for part in raw.replace(";", os.pathsep).split(","):
        part = part.strip()
        if not part:
            continue
        paths.append(Path(part).expanduser().resolve())
    return paths


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MS_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ModelSwitch"
    version: str = "1.0.0"
    environment: str = "production"
    log_level: str = "INFO"
    log_json: bool = True

    public_host: str = "127.0.0.1"
    public_port: int = Field(default=8000, ge=1, le=65535)
    admin_host: str = "127.0.0.1"
    admin_port: int = Field(default=8001, ge=1, le=65535)
    public_base_url: str = "http://127.0.0.1:8000"
    admin_base_url: str = "http://127.0.0.1:8001"
    cors_origins: str = ""
    trust_proxy: bool = False

    data_dir: Path = PROJECT_ROOT / "data"
    model_roots: str = "~/models,./models,~/Downloads"
    allow_external_model_paths: bool = False

    llama_server_bin: str = ""
    llama_threads: int = 0
    llama_n_gpu_layers: int = -1
    llama_batch_size: int = 2048
    llama_ubatch_size: int = 512
    backend_start_timeout: float = Field(default=120.0, ge=5, le=900)
    backend_idle_timeout: int = Field(default=900, ge=0)
    max_resident_models: int = Field(default=2, ge=1, le=32)
    model_concurrency: int = Field(default=2, ge=1, le=16)
    backend_ready_grace: float = 5.0
    backend_restart_attempts: int = 1
    flash_attn: str = "auto"

    request_timeout: float = Field(default=300.0, ge=1, le=3600)
    queue_timeout: float = Field(default=120.0, ge=0.1, le=3600)
    max_queue_size: int = Field(default=64, ge=1, le=10000)
    max_request_bytes: int = Field(default=2 * 1024 * 1024, ge=1024)
    default_max_tokens: int = Field(default=1024, ge=1, le=131072)
    default_context_length: int = Field(default=8192, ge=256, le=1048576)
    max_context_length: int = Field(default=32768, ge=256, le=1048576)
    memory_safety_factor: float = Field(default=0.82, ge=0.1, le=0.99)
    min_free_ram_mb: int = 1024
    idle_check_interval: float = 15.0

    default_rate_limit_rpm: int = 120
    default_rate_limit_tpm: int = 200_000
    admin_rate_limit_rpm: int = 600
    key_pepper: str = Field(
        default="",
        validation_alias=AliasChoices("MS_KEY_PEPPER", "MODELSWITCH_KEY_PEPPER"),
    )

    router_strategy: str = "balanced"
    router_llm_enabled: bool = False
    router_llm_model: str = ""
    router_llm_timeout_ms: int = 2500
    router_ambiguity_threshold: float = 0.55
    router_history_retention_days: int = 30
    router_prioritize_loaded: bool = True

    cloudflare_bin: str = ""
    cloudflare_tunnel_token: str = Field(
        default="",
        validation_alias=AliasChoices("CLOUDFLARE_TUNNEL_TOKEN", "MS_CLOUDFLARE_TUNNEL_TOKEN"),
    )
    cloudflare_public_url: str = ""

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, value: object) -> Path:
        return Path(str(value)).expanduser()

    @field_validator("model_roots", mode="before")
    @classmethod
    def _coerce_model_roots(cls, value: object) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    @property
    def data_path(self) -> Path:
        return self.data_dir.resolve()

    @property
    def db_path(self) -> Path:
        return self.data_path / "modelswitch.db"

    @property
    def log_dir(self) -> Path:
        return self.data_path / "logs"

    @property
    def run_dir(self) -> Path:
        return self.data_path / "run"

    @property
    def dashboard_dir(self) -> Path:
        return PROJECT_ROOT / "app" / "static" / "dashboard"

    @property
    def model_root_paths(self) -> list[Path]:
        roots = _split_paths(self.model_roots)
        # Always permit an in-repo models directory to make local setup obvious.
        repo_models = (PROJECT_ROOT / "models").resolve()
        if repo_models not in roots:
            roots.append(repo_models)
        return roots

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for path in (self.data_path, self.log_dir, self.run_dir):
            path.mkdir(parents=True, exist_ok=True)
            try:
                path.chmod(0o700)
            except OSError:
                pass

    def public_url(self, path: str = "") -> str:
        base = self.public_base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}" if path else base

    def admin_url(self, path: str = "") -> str:
        base = self.admin_base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}" if path else base


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
