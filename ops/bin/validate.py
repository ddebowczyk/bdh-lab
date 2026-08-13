#!/usr/bin/env python3
"""Validate the small operations catalogue without changing repository state."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
OPS_ROOT = ROOT / "ops"
SCHEMA_ROOT = OPS_ROOT / "schema"


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping with a useful error for malformed data."""

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must be a YAML mapping")
    return loaded


def validate_schema(document: Path, schema: Path) -> list[str]:
    """Run the repository-standard YAML Schema checker."""

    executable = shutil.which("ys")
    if executable is None:
        return ["ys is required to validate ops manifests"]
    completed = subprocess.run(
        [executable, "--json", "--schema", str(schema), str(document)],
        capture_output=True,
        check=False,
        text=True,
        cwd=ROOT,
    )
    if completed.returncode == 0:
        return []
    detail = completed.stdout.strip() or completed.stderr.strip() or "schema validation failed"
    return [f"{document.relative_to(ROOT)}: {detail}"]


def validate_catalogue() -> list[str]:
    """Check schemas, active selections, and duplicate claimed paths."""

    errors = validate_schema(OPS_ROOT / "ops.yaml", SCHEMA_ROOT / "ops.schema.yaml")
    manifests = sorted(OPS_ROOT.glob("*/capability.yaml"))
    capability_data: dict[str, dict[str, Any]] = {}
    owner_index: dict[str, str] = {}
    route_index: dict[str, str] = {}
    for manifest in manifests:
        errors.extend(validate_schema(manifest, SCHEMA_ROOT / "capability.schema.yaml"))
        try:
            data = load_yaml(manifest)
        except (OSError, ValueError, yaml.YAMLError) as error:
            errors.append(str(error))
            continue
        identifier = data.get("id")
        if not isinstance(identifier, str):
            continue
        if identifier in capability_data:
            errors.append(f"duplicate capability ID: {identifier}")
        capability_data[identifier] = data
        for owned_path in data.get("owns", []):
            previous_owner = owner_index.setdefault(owned_path, identifier)
            if previous_owner != identifier:
                errors.append(
                    f"{owned_path} is claimed by both {previous_owner} and {identifier}"
                )
        for command in data.get("commands", []):
            route = command.get("route")
            if not isinstance(route, str):
                continue
            previous_owner = route_index.setdefault(route, identifier)
            if previous_owner != identifier:
                errors.append(f"{route} is declared by both {previous_owner} and {identifier}")

    try:
        active = load_yaml(OPS_ROOT / "ops.yaml").get("active", {})
    except (OSError, ValueError, yaml.YAMLError) as error:
        errors.append(str(error))
        active = {}
    if isinstance(active, dict):
        for operation, identifier in active.items():
            if identifier not in capability_data:
                errors.append(f"active operation {operation} selects unknown capability {identifier}")
    return errors


def main() -> int:
    """Print all failures and use a conventional process status."""

    errors = validate_catalogue()
    if errors:
        print("Operations validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("Operations manifests, ownership, and active selections are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
