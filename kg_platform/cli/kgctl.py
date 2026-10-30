from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import uvicorn

from kg_platform.config.loader import load_profile, materialize_profile
from kg_platform.config.runtime import build_backend
from kg_platform.service import create_app


def _load(args):
    path = args.config or args.profile
    if not path:
        raise ValueError("--profile or --config is required")
    return load_profile(path)



async def probe(profile, service_backend: bool = False) -> dict:
    backend = build_backend(profile, service_backend=service_backend)
    try:
        metadata = await backend.metadata()
        expected = profile.graphs[profile.active_graph].expected_capabilities
        if metadata.capabilities.scope != expected.scope:
            raise ValueError("backend scope does not match expected capabilities")
        if expected.reverse_edges_materialized is not None and metadata.capabilities.reverse_edges_materialized != expected.reverse_edges_materialized:
            raise ValueError("reverse edge capability mismatch")
        if expected.supports_literals is not None and metadata.capabilities.supports_literals != expected.supports_literals:
            raise ValueError("literal capability mismatch")
        return {
            "status": "ready",
            "graph_id": metadata.graph.graph_id,
            "snapshot_id": metadata.graph.snapshot_id,
            "capabilities": {"scope": metadata.capabilities.scope},
            "counts": dict(metadata.counts),
        }
    finally:
        await backend.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="kgctl")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "probe", "inspect"):
        command = sub.add_parser(name)
        command.add_argument("--profile")
        command.add_argument("--config")
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--profile", required=True)
    materialize.add_argument("--run-dir", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--profile")
    serve.add_argument("--config")
    args = parser.parse_args()

    if args.command == "materialize":
        output, profile = materialize_profile(args.profile, args.run_dir)
        print(json.dumps({"resolved_config": str(output), "digest": profile.resolved_digest}, indent=2))
        return
    profile = _load(args)
    if args.command == "validate":
        print(json.dumps({"status": "valid", "profile": profile.profile_name, "active_graph": profile.active_graph}, indent=2))
    elif args.command in ("prepare", "probe"):
        print(json.dumps(asyncio.run(probe(profile, service_backend=args.command == "prepare")), indent=2))
    elif args.command == "inspect":
        print(json.dumps(profile.model_dump(mode="json"), indent=2))
    elif args.command == "serve":
        graph = profile.graphs[profile.active_graph]
        if graph.transport.kind in {"embedded", "direct"}:
            print(json.dumps({"status": "no-op", "reason": f"{graph.transport.kind} profile"}))
            return
        backend = build_backend(profile, service_backend=True)
        uvicorn.run(create_app(backend), host=graph.service.host, port=graph.service.port, workers=1)


if __name__ == "__main__":
    main()
