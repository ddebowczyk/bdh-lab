"""Bounded, credential-safe Runpod execution through ``runpodctl`` and SSH."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bdh_lab.costs import collect_pod_cost
from bdh_lab.metadata import ROOT, Document, MetadataError, require_experiment


@dataclass(frozen=True)
class PodPlan:
    """A rendered Pod command that has not yet created a provider resource."""

    experiment_id: str
    experiment_path: str
    experiment_status: str
    profile_id: str
    command: tuple[str, ...]
    gpu_type_ids: tuple[str, ...]
    template_id: str
    template_environment: str | None
    checkout_path: str
    artifacts_root: str
    maximum_runtime_minutes: int
    maximum_cost_usd: float
    terminates_at: str

    def display(self) -> str:
        """Render a copyable command without interpolating environment values."""

        return " ".join(self.command)


@dataclass(frozen=True)
class ProvisionedPod:
    """Provider response for one newly created Pod."""

    pod_id: str
    data: dict[str, Any]


@dataclass(frozen=True)
class RemoteRun:
    """The local evidence retrieved from one bounded remote execution."""

    pod_id: str
    output_directories: tuple[Path, ...]
    cost_record: Path


class PodNotReadyError(MetadataError):
    """Runpod accepted the Pod but has not yet published SSH connection data."""


def _time_now() -> str:
    return datetime.now(UTC).isoformat()


def _rfc3339(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_seconds(value: str) -> int:
    """Convert a schema-validated single-unit duration to seconds."""

    units = {"s": 1, "m": 60, "h": 60 * 60}
    return int(value[:-1]) * units[value[-1]]


def _require_runpod_profile(experiment: Document, profiles: dict[str, Document]) -> Document:
    profile = profiles[experiment.data["profile_id"]]
    if profile.data.get("provider") != "runpod":
        raise MetadataError(
            f"{experiment.data['id']} uses {profile.data['id']}; "
            "select a Runpod profile before rendering a Pod plan"
        )
    return profile


def _require_budget(experiment: Document) -> dict[str, Any]:
    budget = experiment.data.get("budget")
    if not isinstance(budget, dict):
        raise MetadataError(
            f"{experiment.data['id']} has no Runpod budget; set maximum runtime and cost first"
        )
    return budget


def render_pod_plan(experiment_path: Path) -> PodPlan:
    """Resolve a remote experiment to one capped ``runpodctl pod create`` command."""

    catalogue, experiment = require_experiment(experiment_path)
    profile = _require_runpod_profile(experiment, catalogue.profiles)
    budget = _require_budget(experiment)
    execution = profile.data["execution"]
    pod = execution["pod"]
    if pod["cloud_type"] == "COMMUNITY" and not pod["public_ip"]:
        raise MetadataError("a community Pod needs public_ip for the SSH wait and remote runner")
    runtime_minutes = int(budget["maximum_runtime_minutes"])
    terminates_at = _rfc3339(datetime.now(UTC) + timedelta(minutes=runtime_minutes))
    default_template_id = str(execution["template_id"])
    configured_environment = execution.get("template_id_environment")
    template_environment = (
        str(configured_environment) if isinstance(configured_environment, str) else None
    )
    template_id = (
        os.environ.get(template_environment, default_template_id)
        if template_environment
        else default_template_id
    )
    name = f"{pod['name_prefix']}-{experiment.data['id']}"
    command_parts = [
        str(execution["command"]),
        "pod",
        "create",
        "--name",
        name,
        "--template-id",
        template_id,
        "--gpu-id",
        str(pod["gpu_type_ids"][0]),
        "--cloud-type",
        str(pod["cloud_type"]),
        "--container-disk-in-gb",
        str(pod["container_disk_gb"]),
        "--volume-mount-path",
        str(pod["volume_mount_path"]),
        "--ports",
        "22/tcp",
        "--wait",
        "--wait-timeout",
        str(pod["wait_timeout"]),
        "--terminate-after",
        terminates_at,
    ]
    if pod["public_ip"]:
        command_parts.append("--public-ip")
    minimum_cuda = pod.get("min_cuda_version")
    if minimum_cuda is not None:
        command_parts.extend(["--min-cuda-version", str(minimum_cuda)])
    return PodPlan(
        experiment_id=str(experiment.data["id"]),
        experiment_path=experiment.path.relative_to(ROOT).as_posix(),
        experiment_status=str(experiment.data["status"]),
        profile_id=str(profile.data["id"]),
        command=tuple(command_parts),
        gpu_type_ids=tuple(str(gpu_id) for gpu_id in pod["gpu_type_ids"]),
        template_id=template_id,
        template_environment=template_environment,
        checkout_path=str(execution["remote"]["checkout_path"]),
        artifacts_root=str(profile.data["artifacts"]["local_root"]),
        maximum_runtime_minutes=runtime_minutes,
        maximum_cost_usd=float(budget["maximum_cost_usd"]),
        terminates_at=terminates_at,
    )


def doctor() -> list[str]:
    """Report local prerequisites without reading or printing any credential."""

    executable = shutil.which("runpodctl")
    lines = ["Runpod local readiness:"]
    if executable is None:
        lines.append("- runpodctl: missing; install it before remote execution")
    else:
        completed = subprocess.run(
            [executable, "version"], capture_output=True, check=False, text=True
        )
        version = completed.stdout.strip() or completed.stderr.strip()
        lines.append(f"- runpodctl: {version or 'installed'}")
    template = os.environ.get("RUNPOD_TEMPLATE_ID")
    lines.append(
        "- RUNPOD_TEMPLATE_ID: set (overrides the profile default)"
        if template
        else "- RUNPOD_TEMPLATE_ID: not set (the profile default will be used)"
    )
    lines.append("- API key: configure with `runpodctl doctor`; do not commit it to this project")
    return lines


def _apply_inputs(plan: PodPlan) -> tuple[str, list[str]]:
    if plan.experiment_status != "active":
        raise MetadataError(
            f"experiment {plan.experiment_id} is {plan.experiment_status}; "
            "promote it to active before creating a Pod"
        )
    executable = shutil.which(plan.command[0])
    if executable is None:
        raise MetadataError("runpodctl is not installed; run `just provider runpod doctor`")
    command = [executable, *plan.command[1:]]
    return executable, command


def _pod_from_response(response: str) -> ProvisionedPod:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise MetadataError("Runpod did not return JSON for the created Pod") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise MetadataError("Runpod did not return an ID for the created Pod")
    return ProvisionedPod(pod_id=payload["id"], data=payload)


def _hourly_cost_usd(pod: ProvisionedPod) -> float:
    value = pod.data.get("adjustedCostPerHr", pod.data.get("costPerHr"))
    try:
        hourly_cost = float(value)
    except (TypeError, ValueError) as error:
        raise MetadataError("Runpod did not return a usable hourly Pod cost") from error
    if not math.isfinite(hourly_cost) or hourly_cost < 0:
        raise MetadataError("Runpod returned an invalid hourly Pod cost")
    return hourly_cost


def _delete_pod(executable: str, pod_id: str) -> None:
    completed = subprocess.run(
        [executable, "pod", "delete", pod_id], capture_output=True, check=False, text=True
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "Runpod Pod deletion failed"
        raise MetadataError(detail)


def _command_for_gpu(command: list[str], gpu_type_id: str) -> list[str]:
    """Return a Pod-create command with one profile-approved GPU type."""

    updated = list(command)
    updated[updated.index("--gpu-id") + 1] = gpu_type_id
    return updated


def _is_capacity_error(detail: str) -> bool:
    """Recognize retryable allocation errors without retrying other failures."""

    normalized = detail.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "does not have the resources",
            "no longer any instances available",
            "no instances available",
        )
    )


def create_pod(plan: PodPlan) -> ProvisionedPod:
    """Create a Pod on an approved GPU and enforce the projected price cap."""

    executable, command = _apply_inputs(plan)
    capacity_errors: list[str] = []
    for gpu_type_id in plan.gpu_type_ids:
        completed = subprocess.run(
            _command_for_gpu(command, gpu_type_id),
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "Runpod Pod creation failed"
            if _is_capacity_error(detail):
                capacity_errors.append(f"{gpu_type_id}: {detail}")
                continue
            raise MetadataError(detail)
        pod = _pod_from_response(completed.stdout)
        try:
            projected_cost = _hourly_cost_usd(pod) * plan.maximum_runtime_minutes / 60
        except MetadataError:
            _delete_pod(executable, pod.pod_id)
            raise
        if projected_cost > plan.maximum_cost_usd:
            _delete_pod(executable, pod.pod_id)
            raise MetadataError(
                f"Pod {pod.pod_id} would cost up to ${projected_cost:.2f} for "
                f"{plan.maximum_runtime_minutes} minutes; cap is ${plan.maximum_cost_usd:.2f}"
            )
        return pod
    details = "; ".join(capacity_errors)
    raise MetadataError(f"no approved Runpod GPU has capacity: {details}")


def apply_pod_plan(plan: PodPlan) -> str:
    """Create a capped Pod after the caller chose the explicit apply operation."""

    return create_pod(plan).pod_id


def _ssh_command(executable: str, pod_id: str) -> tuple[str, ...]:
    completed = subprocess.run(
        [executable, "--output", "json", "ssh", "info", pod_id],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"could not get SSH information for Pod {pod_id}"
        raise PodNotReadyError(detail)
    try:
        payload = json.loads(completed.stdout)
        command = payload.get("ssh_command", payload.get("sshCommand"))
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise PodNotReadyError("Runpod has not yet published SSH information") from error
    if not isinstance(command, str):
        raise PodNotReadyError("Runpod has not yet published an SSH command")
    parts = shlex.split(command)
    if not parts or Path(parts[0]).name != "ssh":
        raise MetadataError("Runpod SSH information is not an ssh command")
    return tuple(str(Path(part).expanduser()) if part.startswith("~/") else part for part in parts)


def _wait_for_ssh_command(
    executable: str, pod_id: str, *, timeout_seconds: int
) -> tuple[str, ...]:
    """Wait briefly for Runpod to expose SSH after the Pod is allocated."""

    deadline = time.monotonic() + timeout_seconds
    last_error: PodNotReadyError | None = None
    while time.monotonic() < deadline:
        try:
            return _ssh_command(executable, pod_id)
        except PodNotReadyError as error:
            last_error = error
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    detail = str(last_error) if last_error is not None else "no response"
    raise MetadataError(
        f"Runpod did not publish SSH information within {timeout_seconds}s: {detail}"
    )


def _with_ephemeral_known_hosts(
    ssh_command: tuple[str, ...], known_hosts_path: Path
) -> tuple[str, ...]:
    """Use trust-on-first-use for one short-lived Pod, not stale host keys."""

    return (
        ssh_command[0],
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        *ssh_command[1:],
    )


def _run_ssh(
    ssh_command: tuple[str, ...], remote_command: str, *, input_data: bytes | None = None
) -> bytes:
    completed = subprocess.run(
        [*ssh_command, remote_command],
        capture_output=True,
        check=False,
        input=input_data,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip() or "remote command failed"
        raise MetadataError(detail)
    return completed.stdout


_ARCHIVE_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "var",
}
_ARCHIVE_EXCLUDED_NAMES = {".DS_Store", ".env"}


def _include_in_source_archive(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        not any(part in _ARCHIVE_EXCLUDED_PARTS for part in relative.parts)
        and path.name not in _ARCHIVE_EXCLUDED_NAMES
    )


def _source_archive(path: Path) -> str:
    """Archive the current checkout without credentials or generated experiment output."""

    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for source in sorted(ROOT.rglob("*")):
            if _include_in_source_archive(source):
                archive.add(source, arcname=source.relative_to(ROOT).as_posix(), recursive=False)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def render_remote_command(plan: PodPlan, pod_id: str, snapshot_sha256: str) -> str:
    """Render the shell command executed inside the prepared Pod."""

    catalogue, experiment = require_experiment(ROOT / plan.experiment_path)
    profile = catalogue.profiles[experiment.data["profile_id"]].data
    remote = profile["execution"]["remote"]
    checkout = str(remote["checkout_path"])
    python_command = shlex.quote(str(remote["python_command"]))
    uv_version = str(remote["uv_version"])
    environment = {
        "BDH_LAB_SOURCE_REVISION": _source_revision(),
        "BDH_LAB_SOURCE_SNAPSHOT_SHA256": snapshot_sha256,
        "BDH_LAB_PROVIDER": "runpod",
        "BDH_LAB_RUNPOD_POD_ID": pod_id,
        "BDH_LAB_RUNPOD_TEMPLATE_ID": plan.template_id,
    }
    assignments = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in environment.items() if value
    )
    cuda_probe = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is not available in this environment'; "
        "print(torch.cuda.get_device_name(0))"
    )
    initial_cuda_probe = f"{python_command} -c {shlex.quote(cuda_probe)}"
    synchronized_cuda_probe = f"uv run {python_command} -c {shlex.quote(cuda_probe)}"
    return (
        "set -eu; "
        "if ! command -v uv >/dev/null 2>&1; then "
        f"{python_command} -m pip install --disable-pip-version-check "
        f"uv=={shlex.quote(uv_version)}; "
        "fi; "
        "if ! command -v ys >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq; "
        "apt-get install -y --no-install-recommends pkg-config libssl-dev; "
        "if ! command -v cargo >/dev/null 2>&1; then "
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal; "
        "fi; "
        "export PATH=\"$HOME/.cargo/bin:$PATH\"; "
        "cargo install --locked yaml-schema --version 0.9.1; "
        "fi; "
        "export PATH=\"$HOME/.cargo/bin:$PATH\"; "
        f"cd {shlex.quote(checkout)}; "
        "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >&2; "
        f"{initial_cuda_probe} >&2; "
        "uv sync --frozen --extra dev; "
        f"{synchronized_cuda_probe} >&2; "
        f"{assignments} uv run bdh-lab run {shlex.quote(plan.experiment_path)} "
        "--runner-provider runpod"
    )


def _safe_extract(archive_bytes: bytes, destination: Path) -> None:
    """Extract a remote result archive without allowing paths or links to escape it."""

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = destination / member.name
            try:
                target.resolve().relative_to(destination.resolve())
            except ValueError as error:
                raise MetadataError("remote result archive contains an unsafe path") from error
            if not (member.isdir() or member.isfile()):
                raise MetadataError("remote result archive contains a link or device")
        archive.extractall(destination, filter="data")


def _retrieve_results(plan: PodPlan, ssh_command: tuple[str, ...]) -> tuple[Path, ...]:
    local_root = ROOT / plan.artifacts_root / plan.experiment_id
    remote_root = f"{plan.checkout_path}/{plan.artifacts_root}/{plan.experiment_id}"
    archive_bytes = _run_ssh(
        ssh_command,
        f"test -d {shlex.quote(remote_root)} && tar -C {shlex.quote(remote_root)} -czf - .",
    )
    local_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_root.parent) as temporary:
        staging = Path(temporary)
        _safe_extract(archive_bytes, staging)
        for source in staging.iterdir():
            destination = local_root / source.name
            if destination.exists():
                raise MetadataError(
                    f"will not overwrite existing local run evidence: {destination}"
                )
            shutil.move(str(source), destination)
    outputs = tuple(sorted(path.parent for path in local_root.rglob("result.yaml")))
    if not outputs:
        raise MetadataError("the remote runner returned no result record")
    return outputs


def launch_remote_experiment(plan: PodPlan, *, keep_pod: bool = False) -> RemoteRun:
    """Create, execute, retrieve, cost, and normally terminate one isolated Pod."""

    existing_root = ROOT / plan.artifacts_root / plan.experiment_id
    if existing_root.exists() and any(existing_root.rglob("result.yaml")):
        raise MetadataError(
            f"{plan.experiment_id} already has immutable run evidence; create a new version"
        )
    pod = create_pod(plan)
    executable, _command = _apply_inputs(plan)
    started_at = _time_now()
    outputs: tuple[Path, ...] = ()
    execution_error: MetadataError | None = None
    try:
        wait_timeout = plan.command[plan.command.index("--wait-timeout") + 1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            ssh_command = _wait_for_ssh_command(
                executable,
                pod.pod_id,
                timeout_seconds=_duration_seconds(wait_timeout),
            )
            ssh_command = _with_ephemeral_known_hosts(
                ssh_command, temporary_path / "known_hosts"
            )
            archive_path = temporary_path / "source.tar.gz"
            snapshot_sha256 = _source_archive(archive_path)
            _run_ssh(
                ssh_command,
                f"mkdir -p {shlex.quote(plan.checkout_path)} && "
                f"tar -xzf - -C {shlex.quote(plan.checkout_path)}",
                input_data=archive_path.read_bytes(),
            )
            try:
                _run_ssh(ssh_command, render_remote_command(plan, pod.pod_id, snapshot_sha256))
            except MetadataError as error:
                execution_error = error
            try:
                outputs = _retrieve_results(plan, ssh_command)
            except MetadataError as error:
                if execution_error is None:
                    execution_error = error
                else:
                    execution_error = MetadataError(f"{execution_error}; result retrieval: {error}")
    finally:
        ended_at = _time_now()
        if not keep_pod:
            try:
                _delete_pod(executable, pod.pod_id)
            except MetadataError as error:
                if execution_error is None:
                    execution_error = error
                else:
                    execution_error = MetadataError(f"{execution_error}; Pod cleanup: {error}")
        cost_record = collect_pod_cost(
            experiment_id=plan.experiment_id,
            pod_id=pod.pod_id,
            started_at=started_at,
            ended_at=ended_at,
            run_ids=[output.name for output in outputs],
        )
    if execution_error is not None:
        raise execution_error
    return RemoteRun(pod_id=pod.pod_id, output_directories=outputs, cost_record=cost_record)


def plan_as_dict(plan: PodPlan) -> dict[str, Any]:
    """Return machine-readable plan data that contains no credential values."""

    return {
        "experiment_id": plan.experiment_id,
        "experiment_path": plan.experiment_path,
        "experiment_status": plan.experiment_status,
        "profile_id": plan.profile_id,
        "command": list(plan.command),
        "gpu_type_ids": list(plan.gpu_type_ids),
        "template_id": plan.template_id,
        "template_environment": plan.template_environment,
        "artifacts_root": plan.artifacts_root,
        "maximum_runtime_minutes": plan.maximum_runtime_minutes,
        "maximum_cost_usd": plan.maximum_cost_usd,
        "terminates_at": plan.terminates_at,
        "creates_remote_resource": True,
    }
