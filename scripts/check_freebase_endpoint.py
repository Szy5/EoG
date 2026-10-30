#!/usr/bin/env python3
"""Health-check a Freebase SPARQL endpoint for KG-Self-Play."""

import argparse
import json
import os
import subprocess
import sys

import requests


def run_probe(endpoint: str, timeout: float, query: str, require_bindings: bool) -> None:
    response = requests.get(
        endpoint,
        params={"query": query, "format": "application/sparql-results+json"},
        timeout=timeout,
    )
    response.raise_for_status()
    print("SPARQL endpoint probe status:", response.status_code)
    print("SPARQL endpoint probe body:", response.text[:400])
    try:
        payload = response.json()
    except ValueError:
        return

    bindings = payload.get("results", {}).get("bindings")
    if require_bindings:
        if not bindings:
            raise SystemExit("SPARQL query returned no bindings.")
        print("SPARQL endpoint first binding:", json.dumps(bindings[0], ensure_ascii=False))


def run_pytest(repo_root: str, endpoint: str) -> None:
    env = dict(os.environ)
    env["KGQA_SPARQL_ENDPOINT"] = endpoint
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/kg_ssp/integration/test_kg_tool_roundtrip.py",
        "-q",
        "-m",
        "requires_kg_server",
    ]
    print("RUNNING:", " ".join(command))
    completed = subprocess.run(command, cwd=repo_root, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a Freebase SPARQL endpoint.")
    parser.add_argument("--endpoint", required=True, help="SPARQL endpoint, e.g. http://127.0.0.1:3001/sparql")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--repo-root", default="/path/to/project/KG-Self-Play")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--query", default="ASK {}", help="SPARQL query to execute for the probe.")
    parser.add_argument("--require-bindings", action="store_true", help="Fail when a SELECT query returns no bindings.")
    args = parser.parse_args()

    run_probe(args.endpoint, args.timeout, args.query, args.require_bindings)
    if args.run_pytest:
        run_pytest(args.repo_root, args.endpoint)


if __name__ == "__main__":
    main()
