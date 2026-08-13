"""Run reproducible mechanism experiments and write immutable evidence."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import yaml
from torch import nn

from bdh_lab.datasets import HeldOutSplits, accuracy, resolve_held_out_splits
from bdh_lab.metadata import ROOT, validate_result_document
from bdh_lab.models import build_model, make_associative_recall_batch


@dataclass(frozen=True)
class RunOutcome:
    """One persisted seed result and its optional execution error."""

    output_directory: Path
    error: Exception | None = None


class ExperimentExecutionError(RuntimeError):
    """Raised after each requested seed has written either evidence or failure data."""

    def __init__(self, outcomes: list[RunOutcome]) -> None:
        self.outcomes = outcomes
        failed = [outcome for outcome in outcomes if outcome.error is not None]
        details = "; ".join(
            f"{outcome.output_directory.name}: {outcome.error}" for outcome in failed
        )
        super().__init__(f"{len(failed)} seed run(s) failed: {details}")


def select_device(requested: str) -> torch.device:
    """Resolve an explicit device request without silently using an unavailable GPU."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def source_revision() -> str:
    """Return the current revision without requiring that the first commit exists."""

    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _time_now() -> str:
    return datetime.now(UTC).isoformat()


def _run_id(seed: int) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%S%fZ").lower()
    return f"run-{timestamp}-seed-{seed}"


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _provenance(
    *, device: str, seed: int, held_out_splits: HeldOutSplits | None = None
) -> dict[str, Any]:
    """Build safe provenance from local state and optional remote runner values."""

    provenance: dict[str, Any] = {
        "source_revision": os.environ.get("BDH_LAB_SOURCE_REVISION", source_revision()),
        "torch_version": str(torch.__version__),
        "device": device,
        "seed": seed,
    }
    optional_environment = {
        "source_snapshot_sha256": "BDH_LAB_SOURCE_SNAPSHOT_SHA256",
        "provider": "BDH_LAB_PROVIDER",
        "pod_id": "BDH_LAB_RUNPOD_POD_ID",
        "template_id": "BDH_LAB_RUNPOD_TEMPLATE_ID",
    }
    for field, environment in optional_environment.items():
        value = os.environ.get(environment)
        if value:
            provenance[field] = value
    if held_out_splits is not None:
        provenance.update(
            {
                "validation_dataset_id": held_out_splits.validation_id,
                "validation_dataset_checksum_sha256": held_out_splits.validation_checksum_sha256,
                "golden_evaluation_dataset_id": held_out_splits.golden_evaluation_id,
                "golden_evaluation_dataset_checksum_sha256": (
                    held_out_splits.golden_evaluation_checksum_sha256
                ),
            }
        )
    return provenance


def _write_result(path: Path, result: dict[str, Any]) -> None:
    """Persist and immediately validate a generated result record."""

    _write_yaml(path, result)
    errors = validate_result_document(path)
    if errors:
        raise RuntimeError("result schema validation failed: " + "; ".join(errors))


def _run_seed(
    *,
    plan: dict[str, Any],
    output_root: Path,
    seed: int,
) -> RunOutcome:
    """Run one seed, recording a valid completed or failed result in all normal cases."""

    experiment = plan["experiment"]
    assembly = plan["assembly"]
    task = experiment["task"]["config"]
    training = experiment["training"]
    run_id = _run_id(seed)
    output_directory = output_root / run_id
    output_directory.mkdir(parents=True, exist_ok=False)
    started_at = _time_now()
    requested_device = str(training["device"])
    held_out_splits: HeldOutSplits | None = None

    result_base = {
        "schema_version": 1,
        "id": run_id,
        "experiment_id": experiment["id"],
        "assembly_id": assembly["id"],
        "profile_id": plan["profile"]["id"],
        "started_at": started_at,
        "resolved": plan,
    }

    try:
        symbols = int(task["symbols"])
        device = select_device(requested_device)
        held_out_splits = resolve_held_out_splits(plan)
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        model: nn.Module = build_model(assembly, symbols=symbols).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
        final_loss = 0.0

        model.train()
        for _step in range(int(training["steps"])):
            batch = make_associative_recall_batch(
                symbols=symbols,
                associations_per_sequence=int(task["associations_per_sequence"]),
                batch_size=int(task["batch_size"]),
                generator=generator,
                device=device,
                excluded_example_ids=held_out_splits.excluded_training_example_ids,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.tokens)
            loss = functional.cross_entropy(logits, batch.targets)
            final_loss = float(loss.detach().cpu())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_accuracy = accuracy(
                model, held_out_splits.validation_examples, device
            )
            golden_evaluation_accuracy = accuracy(
                model, held_out_splits.golden_evaluation_examples, device
            )

        result = {
            **result_base,
            "status": "completed",
            "finished_at": _time_now(),
            "provenance": _provenance(
                device=str(device), seed=seed, held_out_splits=held_out_splits
            ),
            "metrics": {
                "validation_query_accuracy": validation_accuracy,
                "golden_evaluation_query_accuracy": golden_evaluation_accuracy,
                "final_train_loss": final_loss,
            },
        }
        _write_result(output_directory / "result.yaml", result)
        rendered_plan = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        (output_directory / "plan.json").write_text(rendered_plan, encoding="utf-8")
        return RunOutcome(output_directory)
    except Exception as error:
        result = {
            **result_base,
            "status": "failed",
            "finished_at": _time_now(),
            "provenance": _provenance(
                device=requested_device, seed=seed, held_out_splits=held_out_splits
            ),
            "metrics": {},
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        _write_result(output_directory / "result.yaml", result)
        rendered_plan = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        (output_directory / "plan.json").write_text(rendered_plan, encoding="utf-8")
        return RunOutcome(output_directory, error)


def run_experiment(
    plan: dict[str, Any],
    output_root: Path | None = None,
    *,
    runner_provider: str = "local",
) -> list[Path]:
    """Run every declared seed for one active experiment and persist each result."""

    experiment = plan["experiment"]
    profile = plan["profile"]
    if experiment["status"] != "active":
        raise ValueError(f"experiment {experiment['id']} is not active")
    if profile["provider"] != runner_provider:
        raise ValueError(
            f"{runner_provider} runner cannot execute a {profile['provider']} profile"
        )

    if output_root is None:
        output_root = ROOT / profile["artifacts"]["local_root"] / experiment["id"]
    outcomes = [
        _run_seed(plan=plan, output_root=output_root, seed=int(seed))
        for seed in experiment["seeds"]
    ]
    if any(outcome.error is not None for outcome in outcomes):
        raise ExperimentExecutionError(outcomes)
    return [outcome.output_directory for outcome in outcomes]
