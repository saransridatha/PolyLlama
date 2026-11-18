from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.remote import CloudflareTunnel, RemoteError


def test_public_url_validation(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", model_roots=str(tmp_path))
    settings.ensure_dirs()
    tunnel = CloudflareTunnel(settings)
    assert tunnel._validate_public_url("https://llm.example.com/") == "https://llm.example.com"
    with pytest.raises(RemoteError):
        tunnel._validate_public_url("http://llm.example.com")
    with pytest.raises(RemoteError):
        tunnel._validate_public_url("https://localhost:8000")
    with pytest.raises(RemoteError):
        tunnel._validate_public_url("not a url")


@pytest.mark.asyncio
async def test_token_mode_requires_token(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path),
        cloudflare_tunnel_token="",
        cloudflare_bin="/bin/true",
    )
    settings.ensure_dirs()
    tunnel = CloudflareTunnel(settings)
    with pytest.raises(RemoteError, match="token"):
        await tunnel.start(mode="token")


@pytest.mark.asyncio
async def test_missing_binary(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path),
        cloudflare_bin=str(tmp_path / "missing-cloudflared"),
    )
    settings.ensure_dirs()
    tunnel = CloudflareTunnel(settings)
    tunnel.binary = None
    with pytest.raises(RemoteError, match="not installed"):
        await tunnel.start(mode="quick")


@pytest.mark.asyncio
async def test_quick_tunnel_process_lifecycle_with_fake_connector(tmp_path: Path):
    script = tmp_path / "fake-cloudflared"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "print('INF Requesting new quick Tunnel on trycloudflare.com...')\n"
        "print('INF https://unit-test.trycloudflare.com')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path),
        cloudflare_bin=str(script),
        public_port=18300,
        admin_port=18301,
    )
    settings.ensure_dirs()
    tunnel = CloudflareTunnel(settings)
    status = await tunnel.start(mode="quick")
    assert status["running"] is True
    assert status["public_url"] == "https://unit-test.trycloudflare.com"
    assert status["api_base_url"].endswith("/v1")
    assert status["admin_exposed"] is False
    stopped = await tunnel.stop()
    assert stopped["running"] is False
    assert tunnel.process is None
