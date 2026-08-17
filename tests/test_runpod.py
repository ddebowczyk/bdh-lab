import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import bdh_lab.runpod as runpod
from bdh_lab.metadata import MetadataError, repository_root
from bdh_lab.runpod import (
    apply_pod_plan,
    create_pod,
    launch_remote_experiment,
    plan_as_dict,
    render_pod_plan,
    render_remote_command,
)


def test_runpod_plan_uses_profile_default_without_credential_values() -> None:
    plan = render_pod_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-fastweight-runpod-v1.yaml"
    )

    rendered = plan_as_dict(plan)

    assert rendered["profile_id"] == "runpod-pod-v1"
    assert "runpod-torch-v280" in rendered["command"]
    assert rendered["template_id"] == "runpod-torch-v280"
    assert "RUNPOD_API_KEY" not in str(rendered)
    assert "--terminate-after" in rendered["command"]
    assert rendered["maximum_runtime_minutes"] == 45
    assert rendered["maximum_cost_usd"] == 2.0
    assert rendered["creates_remote_resource"] is True


def test_runpod_smoke_plan_is_small_and_capped() -> None:
    plan = render_pod_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-fastweight-runpod-smoke-v1.yaml"
    )

    assert plan.experiment_status == "active"
    assert plan.maximum_runtime_minutes == 30
    assert plan.maximum_cost_usd == 0.20
    assert plan.command[plan.command.index("--gpu-id") + 1] == "NVIDIA RTX A5000"
    assert plan.gpu_type_ids == (
        "NVIDIA RTX A5000",
        "NVIDIA RTX A4000",
        "NVIDIA RTX A4500",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 4090",
    )


def test_runpod_v2_plan_uses_current_cuda_compatible_gpu_types() -> None:
    plan = render_pod_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-bdh-gpu-streaming-runpod-v2.yaml"
    )

    assert plan.command[plan.command.index("--gpu-id") + 1] == "NVIDIA GeForce RTX 4090"
    assert plan.command[plan.command.index("--min-cuda-version") + 1] == "13.0"
    assert plan.gpu_type_ids == (
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A6000",
        "NVIDIA GeForce RTX 3090",
    )


def test_remote_command_contains_reproducibility_evidence_not_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID", "template-safe-to-record")
    plan = render_pod_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-fastweight-runpod-v1.yaml"
    )

    command = render_remote_command(plan, "pod-123", "a" * 64)

    assert "--runner-provider runpod" in command
    assert "BDH_LAB_SOURCE_SNAPSHOT_SHA256" in command
    assert "RUNPOD_API_KEY" not in command
    assert "template-safe-to-record" in command
    assert "apt-get install -y --no-install-recommends pkg-config libssl-dev" in command
    assert "cargo install --locked yaml-schema --version 0.9.1" in command
    assert "torch.cuda.is_available" in command


def test_proposed_runpod_experiment_cannot_create_a_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = render_pod_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-fastweight-runpod-v1.yaml"
    )
    with pytest.raises(MetadataError, match="promote it to active"):
        apply_pod_plan(plan)

    with pytest.raises(MetadataError, match="promote it to active"):
        launch_remote_experiment(plan)


def test_pod_is_deleted_if_returned_price_exceeds_declared_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = render_pod_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-fastweight-runpod-v1.yaml"
    )
    plan = replace(plan, experiment_status="active")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[1:3] == ["pod", "create"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"id": "pod-123", "adjustedCostPerHr": 10.0}),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("bdh_lab.runpod.shutil.which", lambda _name: "runpodctl")
    monkeypatch.setattr("bdh_lab.runpod.subprocess.run", fake_run)

    with pytest.raises(MetadataError, match="would cost up to"):
        create_pod(plan)

    assert any(command[1:3] == ["pod", "delete"] for command in calls)


def test_pod_tries_next_profile_gpu_after_capacity_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = render_pod_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-fastweight-runpod-v1.yaml"
    )
    plan = replace(
        plan,
        experiment_status="active",
        gpu_type_ids=("NVIDIA RTX A5000", "NVIDIA RTX A4000"),
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        gpu_type_id = command[command.index("--gpu-id") + 1]
        if gpu_type_id == "NVIDIA RTX A5000":
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="There are no longer any instances available",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"id": "pod-456", "adjustedCostPerHr": 0.1}),
            stderr="",
        )

    monkeypatch.setattr("bdh_lab.runpod.shutil.which", lambda _name: "runpodctl")
    monkeypatch.setattr("bdh_lab.runpod.subprocess.run", fake_run)

    pod = create_pod(plan)

    assert pod.pod_id == "pod-456"
    assert [call[call.index("--gpu-id") + 1] for call in calls] == [
        "NVIDIA RTX A5000",
        "NVIDIA RTX A4000",
    ]


def test_wait_for_ssh_command_retries_until_runpod_publishes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_ssh_command(_executable: str, _pod_id: str) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise runpod.PodNotReadyError("still starting")
        return ("ssh", "root@example.test")

    monkeypatch.setattr("bdh_lab.runpod._ssh_command", fake_ssh_command)
    monkeypatch.setattr("bdh_lab.runpod.time.sleep", lambda _seconds: None)

    command = runpod._wait_for_ssh_command("runpodctl", "pod-123", timeout_seconds=5)

    assert command == ("ssh", "root@example.test")
    assert calls == 2


def test_ssh_command_accepts_current_runpod_cli_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bdh_lab.runpod.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ssh_command": "ssh root@example.test -p 22"}),
            stderr="",
        ),
    )

    command = runpod._ssh_command("runpodctl", "pod-123")

    assert command == ("ssh", "root@example.test", "-p", "22")


def test_ssh_uses_one_ephemeral_trust_on_first_use_file() -> None:
    command = runpod._with_ephemeral_known_hosts(
        ("ssh", "root@example.test", "-p", "22"), Path("/tmp/known_hosts")
    )

    assert command == (
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=/tmp/known_hosts",
        "root@example.test",
        "-p",
        "22",
    )
