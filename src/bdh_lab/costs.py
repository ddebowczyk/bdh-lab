"""Collect immutable, Pod-scoped Runpod billing evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from bdh_lab.metadata import ROOT, MetadataError, validate_cost_document


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _cost_id(pod_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%S%fZ").lower()
    return f"cost-{timestamp}-pod-{pod_id.lower()}"


def _normalise_entry(entry: dict[str, Any], pod_id: str) -> dict[str, Any]:
    """Convert the documented Runpod response fields to the lab's stable format."""

    amount = entry.get("amount")
    billed = entry.get("timeBilledMs")
    time = entry.get("time")
    if not isinstance(amount, (int, float)):
        raise MetadataError("Runpod billing response has no numeric amount")
    if not isinstance(billed, int):
        raise MetadataError("Runpod billing response has no integer timeBilledMs")
    if not isinstance(time, str):
        raise MetadataError("Runpod billing response has no RFC3339 time")
    normalized: dict[str, Any] = {
        "amount_usd": float(amount),
        "time": time,
        "time_billed_ms": billed,
    }
    optional_fields = {
        "diskSpaceBilledGb": "disk_space_billed_gb",
        "gpuTypeId": "gpu_type_id",
        "podId": "pod_id",
    }
    for source, target in optional_fields.items():
        value = entry.get(source)
        if value is not None:
            normalized[target] = value
    normalized.setdefault("pod_id", pod_id)
    return normalized


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def collect_pod_cost(
    *,
    experiment_id: str,
    pod_id: str,
    started_at: str,
    ended_at: str,
    run_ids: list[str] | None = None,
    output_root: Path | None = None,
) -> Path:
    """Query one Pod's billing history and append a schema-valid cost record.

    An empty provider reply means the billing feed is not ready, not a zero-cost run.
    A failed query is recorded as unavailable so the experiment remains auditable.
    """

    output_root = output_root or ROOT / "var" / "runs"
    costs_root = output_root / experiment_id / "costs"
    costs_root.mkdir(parents=True, exist_ok=True)
    identifier = _cost_id(pod_id)
    raw_name = f"{identifier}.json"
    raw_path = costs_root / raw_name
    record_path = costs_root / f"{identifier}.yaml"
    collection = {
        "started_at": started_at,
        "ended_at": ended_at,
        "collected_at": _now(),
        "bucket_size": "hour",
        "raw_artifact": raw_name,
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "id": identifier,
        "experiment_id": experiment_id,
        "provider": "runpod",
        "pod_id": pod_id,
        "run_ids": run_ids or [],
        "collection": collection,
    }
    executable = shutil.which("runpodctl")
    if executable is None:
        raw_path.write_text("[]\n", encoding="utf-8")
        record.update(
            {
                "status": "unavailable",
                "failure": {
                    "type": "FileNotFoundError",
                    "message": "runpodctl is not installed",
                },
            }
        )
    else:
        command = [
            executable,
            "--output",
            "json",
            "billing",
            "pods",
            "--pod-id",
            pod_id,
            "--grouping",
            "podId",
            "--bucket-size",
            "hour",
            "--start-time",
            started_at,
            "--end-time",
            ended_at,
        ]
        completed = subprocess.run(command, capture_output=True, check=False, text=True)
        raw_path.write_text(completed.stdout or "[]\n", encoding="utf-8")
        if completed.returncode != 0:
            record.update(
                {
                    "status": "unavailable",
                    "failure": {
                        "type": "RunpodBillingError",
                        "message": completed.stderr.strip() or "Runpod billing command failed",
                    },
                }
            )
        else:
            try:
                payload = json.loads(completed.stdout)
                if not isinstance(payload, list):
                    raise MetadataError("Runpod billing response must be a JSON array")
                entries = [_normalise_entry(entry, pod_id) for entry in payload]
            except (json.JSONDecodeError, MetadataError, TypeError) as error:
                record.update(
                    {
                        "status": "unavailable",
                        "failure": {"type": type(error).__name__, "message": str(error)},
                    }
                )
            else:
                if entries:
                    record.update(
                        {
                            "status": "reported",
                            "amount_usd": sum(entry["amount_usd"] for entry in entries),
                            "time_billed_ms": sum(
                                entry["time_billed_ms"] for entry in entries
                            ),
                            "entries": entries,
                        }
                    )
                else:
                    record["status"] = "pending"

    _write_yaml(record_path, record)
    errors = validate_cost_document(record_path)
    if errors:
        raise MetadataError("cost record schema validation failed: " + "; ".join(errors))
    return record_path
