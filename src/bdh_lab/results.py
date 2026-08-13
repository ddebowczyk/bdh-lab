"""Read immutable run records and summarize comparable experiment evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from bdh_lab.metadata import (
    ROOT,
    MetadataError,
    load_yaml,
    validate_cost_document,
    validate_result_document,
)


@dataclass(frozen=True)
class ResultRecord:
    """One schema-valid generated result record."""

    path: Path
    data: dict[str, Any]


@dataclass(frozen=True)
class CostRecord:
    """One schema-valid provider billing record."""

    path: Path
    data: dict[str, Any]


def load_results(root: Path | None = None) -> list[ResultRecord]:
    """Load every valid result below the ignored local results root."""

    root = root or ROOT / "var" / "runs"
    if not root.exists():
        return []
    records: list[ResultRecord] = []
    for path in sorted(root.rglob("result.yaml")):
        errors = validate_result_document(path)
        if errors:
            raise MetadataError("result validation failed:\n" + "\n".join(errors))
        records.append(ResultRecord(path=path, data=load_yaml(path)))
    return records


def load_costs(root: Path | None = None) -> list[CostRecord]:
    """Load immutable provider billing records below the local results root."""

    root = root or ROOT / "var" / "runs"
    if not root.exists():
        return []
    records: list[CostRecord] = []
    for path in sorted(root.rglob("cost-*.yaml")):
        errors = validate_cost_document(path)
        if errors:
            raise MetadataError("cost validation failed:\n" + "\n".join(errors))
        records.append(CostRecord(path=path, data=load_yaml(path)))
    return records


def summarize_results(
    records: list[ResultRecord],
    *,
    costs: list[CostRecord] | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate completed runs by experiment without hiding failed runs."""

    if experiment_id is not None:
        records = [
            record for record in records if record.data["experiment_id"] == experiment_id
        ]
    groups: dict[str, list[ResultRecord]] = {}
    for record in records:
        groups.setdefault(str(record.data["experiment_id"]), []).append(record)
    cost_groups: dict[str, list[CostRecord]] = {}
    for record in costs or []:
        if experiment_id is None or record.data["experiment_id"] == experiment_id:
            identifier = str(record.data["experiment_id"])
            cost_groups.setdefault(identifier, []).append(record)
            groups.setdefault(identifier, [])

    experiments: list[dict[str, Any]] = []
    for identifier, group in sorted(groups.items()):
        completed = [record for record in group if record.data["status"] == "completed"]
        failed = [record for record in group if record.data["status"] == "failed"]
        metric_names = sorted(
            {name for record in completed for name in record.data["metrics"]}
        )
        metrics: dict[str, dict[str, float | int]] = {}
        for name in metric_names:
            # Metrics can be added in a later experiment-record version. Do not
            # make historical evidence unreadable when a newer run reports a
            # more specific metric; show the sample count for each metric.
            values = [
                float(record.data["metrics"][name])
                for record in completed
                if name in record.data["metrics"]
            ]
            metrics[name] = {
                "count": len(values),
                "mean": fmean(values),
                "sample_standard_deviation": stdev(values) if len(values) > 1 else 0.0,
            }
        experiments.append(
            {
                "experiment_id": identifier,
                "run_count": len(group),
                "completed_count": len(completed),
                "failed_count": len(failed),
                "metrics": metrics,
                "cost": _summarize_costs(cost_groups.get(identifier, [])),
            }
        )

    return {
        "schema_version": 1,
        "result_count": len(records),
        "experiments": experiments,
    }


def _summarize_costs(records: list[CostRecord]) -> dict[str, float | int]:
    """Show only provider-reported spend; pending data is never treated as zero."""

    reported = [record for record in records if record.data["status"] == "reported"]
    return {
        "record_count": len(records),
        "reported_count": len(reported),
        "pending_count": sum(record.data["status"] == "pending" for record in records),
        "unavailable_count": sum(record.data["status"] == "unavailable" for record in records),
        "reported_amount_usd": sum(float(record.data["amount_usd"]) for record in reported),
        "reported_time_billed_ms": sum(
            int(record.data["time_billed_ms"]) for record in reported
        ),
    }
