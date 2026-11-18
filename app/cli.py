"""ModelSwitch command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import Settings
from .db import Database
from .hardware import detect_hardware, SystemMonitor
from .registry import ModelRegistry, RegistryError
from .security import KeyService


def _settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def cmd_init(args: argparse.Namespace) -> int:
    settings = _settings()
    db = Database(settings.db_path)
    try:
        keys = KeyService(db, settings.key_pepper)
        if db.count_api_keys() > 0:
            print("ModelSwitch is already initialized. Refusing to generate new bootstrap keys.")
            print("Use `python -m app keys create --name <name> --role inference` to add a key.")
            return 1
        generated = keys.bootstrap_if_empty()
        print("\n" + "=" * 74)
        print(" ModelSwitch initialized")
        print("=" * 74)
        print(f" Database : {settings.db_path}")
        print(f" Data dir : {settings.data_path}")
        print("\n Credentials are shown ONCE and stored only as salted hashes.")
        print(" Copy them into your secret manager now.\n")
        for label, raw in generated:
            print(f" {label.upper():<10} : {raw}")
        print("\n Public API :", settings.public_url("/v1"))
        print(" Dashboard  :", settings.admin_url("/admin"))
        print("=" * 74 + "\n")
        return 0
    finally:
        db.close()


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    settings = _settings()
    if args.public_port:
        settings.public_port = args.public_port
        settings.public_base_url = f"http://{settings.public_host}:{args.public_port}"
    if args.admin_port:
        settings.admin_port = args.admin_port
        settings.admin_base_url = f"http://{settings.admin_host}:{args.admin_port}"
    return asyncio.run(serve(settings))


def cmd_keys(args: argparse.Namespace) -> int:
    settings = _settings()
    db = Database(settings.db_path)
    try:
        keys = KeyService(db, settings.key_pepper)
        if args.keys_command == "create":
            record, raw = keys.create_key(
                name=args.name,
                role=args.role,
                permissions=_permissions(args),
                rate_limit_rpm=args.rpm
                or (settings.admin_rate_limit_rpm if args.role == "admin" else settings.default_rate_limit_rpm),
                rate_limit_tpm=args.tpm
                or (1_000_000 if args.role == "admin" else settings.default_rate_limit_tpm),
            )
            print(f"Created {args.role} key '{record['name']}' ({record['id']})")
            print(raw)
            print("This key is shown once; only a hash is stored.")
        elif args.keys_command == "list":
            for key in db.list_api_keys():
                state = "revoked" if key.get("revoked_at") else "active"
                print(
                    f"{key['id']}  {key['prefix']:<12}  {key['role']:<9}  {state:<8}  "
                    f"{key['rate_limit_rpm']:>6} rpm  {key['name']}"
                )
        elif args.keys_command == "revoke":
            if not db.revoke_api_key(args.key_id):
                print("Key not found or already revoked", file=sys.stderr)
                return 1
            print(f"Revoked {args.key_id}")
        elif args.keys_command == "rotate":
            try:
                record, raw = keys.rotate(args.key_id)
            except (KeyError, ValueError) as exc:
                print(f"Cannot rotate: {exc}", file=sys.stderr)
                return 1
            print(f"Rotated to {record['id']}. Old key revoked.")
            print(raw)
        return 0
    finally:
        db.close()


def _permissions(args: argparse.Namespace) -> dict[str, object]:
    perms: dict[str, object] = {}
    if args.models:
        perms["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.no_stream:
        perms["allow_streaming"] = False
    if args.max_tokens:
        perms["max_tokens"] = args.max_tokens
    return perms


def cmd_models(args: argparse.Namespace) -> int:
    settings = _settings()
    db = Database(settings.db_path)
    try:
        registry = ModelRegistry(db, settings)
        if args.models_command == "list":
            models = registry.list()
            if not models:
                print("No models registered.")
            for model in models:
                state = "enabled" if model["enabled"] else "disabled"
                print(
                    f"{model['id']:<32} {model['parameter_size']:<8} "
                    f"{model['quantization']:<10} ctx={model['context_length']:<7} "
                    f"{state:<8} {model['path']}"
                )
        elif args.models_command == "scan":
            for item in registry.scan():
                marker = "valid" if item.get("valid") else f"INVALID: {item.get('error')}"
                print(f"{item['name']}  {item['path']}  [{marker}]")
        elif args.models_command == "add":
            model = registry.add_model(
                path=args.path,
                name=args.name,
                context_length=args.context,
                priority=args.priority,
                residency=args.residency,
                max_concurrency=args.concurrency,
            )
            print(json.dumps(registry.admin_card(model), indent=2))
        elif args.models_command == "remove":
            registry.delete_model(args.model_id)
            print(f"Removed registration {args.model_id}; model file was not deleted.")
        return 0
    except RegistryError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 2
    finally:
        db.close()


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings()
    hardware = detect_hardware()
    monitor = SystemMonitor(hardware)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    checks.append(("data_dir", settings.data_path.exists(), str(settings.data_path)))
    checks.append(("llama_binary", bool(_llama_binary(settings)), str(_llama_binary(settings) or "missing")))
    checks.append(("llama.cpp", bool(_llama_binary(settings)), "run scripts/build_llama.sh if missing"))
    db = Database(settings.db_path)
    try:
        checks.append(("database", True, str(settings.db_path)))
        checks.append(("api_keys", db.count_api_keys() > 0, f"{db.count_api_keys()} active"))
        registry = ModelRegistry(db, settings)
        models = registry.list()
        available = [m for m in models if registry.availability(m)["available"]]
        checks.append(("models", bool(models), f"{len(models)} registered, {len(available)} available"))
    finally:
        db.close()
    checks.append(("cloudflared", bool(_cloudflared(settings)), str(_cloudflared(settings) or "optional, not installed")))
    for name, ok, detail in checks:
        print(f"[{'ok' if ok else '!!'}] {name:<14} {detail}")
    print("\nHardware:")
    print(json.dumps(hardware.to_dict(monitor.snapshot().to_dict()), indent=2))
    print("\nRecommendation:")
    print(json.dumps(monitor.recommend(), indent=2))
    critical = [c for c in checks if c[0] in ("python", "data_dir", "database", "api_keys")]
    return 0 if all(ok for _, ok, _ in critical) else 1


def _llama_binary(settings: Settings):
    from .backend.llama import find_llama_server

    return find_llama_server(settings)


def _cloudflared(settings: Settings):
    from .remote import find_cloudflared

    return find_cloudflared(settings)


def cmd_version(args: argparse.Namespace) -> int:
    settings = _settings()
    print(f"ModelSwitch {settings.version}")
    print(f"Data: {settings.data_path}")
    binary = _llama_binary(settings)
    print(f"llama.cpp: {binary or 'not found'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelswitch",
        description="Hardware-adaptive local multi-model LLM inference gateway",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize the database and generate bootstrap API keys")
    p_init.set_defaults(func=cmd_init)

    p_serve = sub.add_parser("serve", help="Start the public API and admin dashboard")
    p_serve.add_argument("--public-port", type=int)
    p_serve.add_argument("--admin-port", type=int)
    p_serve.set_defaults(func=cmd_serve)

    p_keys = sub.add_parser("keys", help="Manage API keys")
    keys_sub = p_keys.add_subparsers(dest="keys_command", required=True)
    kc = keys_sub.add_parser("create")
    kc.add_argument("--name", required=True)
    kc.add_argument("--role", choices=["admin", "inference"], default="inference")
    kc.add_argument("--rpm", type=int)
    kc.add_argument("--tpm", type=int)
    kc.add_argument("--models", help="Comma-separated allowed model ids")
    kc.add_argument("--no-stream", action="store_true")
    kc.add_argument("--max-tokens", type=int)
    kc.set_defaults(func=cmd_keys)
    kl = keys_sub.add_parser("list")
    kl.set_defaults(func=cmd_keys)
    kr = keys_sub.add_parser("revoke")
    kr.add_argument("key_id")
    kr.set_defaults(func=cmd_keys)
    kt = keys_sub.add_parser("rotate")
    kt.add_argument("key_id")
    kt.set_defaults(func=cmd_keys)
    p_keys.set_defaults(func=cmd_keys)

    p_models = sub.add_parser("models", help="Manage the GGUF model registry")
    models_sub = p_models.add_subparsers(dest="models_command", required=True)
    ml = models_sub.add_parser("list")
    ml.set_defaults(func=cmd_models)
    ms = models_sub.add_parser("scan")
    ms.set_defaults(func=cmd_models)
    ma = models_sub.add_parser("add")
    ma.add_argument("--path", required=True)
    ma.add_argument("--name")
    ma.add_argument("--context", type=int)
    ma.add_argument("--priority", type=int, default=50)
    ma.add_argument("--residency", choices=["on_demand", "keep_loaded"], default="on_demand")
    ma.add_argument("--concurrency", type=int)
    ma.set_defaults(func=cmd_models)
    mr = models_sub.add_parser("remove")
    mr.add_argument("model_id")
    mr.set_defaults(func=cmd_models)

    p_doctor = sub.add_parser("doctor", help="Check installation, hardware, and configuration")
    p_doctor.set_defaults(func=cmd_doctor)
    p_version = sub.add_parser("version")
    p_version.set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
