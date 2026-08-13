import json
import subprocess
from pathlib import Path

import yaml

from bdh_lab.costs import collect_pod_cost
from bdh_lab.results import load_costs, summarize_results


def test_collect_pod_cost_records_provider_response(
    monkeypatch, tmp_path: Path
) -> None:
    payload = [
        {
            "amount": 0.42,
            "diskSpaceBilledGb": 30,
            "gpuTypeId": "NVIDIA RTX 4090",
            "podId": "pod-123",
            "time": "2026-08-13T10:00:00+00:00",
            "timeBilledMs": 120000,
        }
    ]
    monkeypatch.setattr("bdh_lab.costs.shutil.which", lambda _name: "runpodctl")
    monkeypatch.setattr(
        "bdh_lab.costs.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    path = collect_pod_cost(
        experiment_id="synthetic-associative-recall-fastweight-runpod-v1",
        pod_id="pod-123",
        started_at="2026-08-13T10:00:00+00:00",
        ended_at="2026-08-13T10:02:00+00:00",
        run_ids=["run-example"],
        output_root=tmp_path,
    )

    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert record["status"] == "reported"
    assert record["amount_usd"] == 0.42
    assert record["time_billed_ms"] == 120000
    raw = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert raw == payload
    report = summarize_results([], costs=load_costs(tmp_path))
    assert report["experiments"][0]["cost"]["reported_amount_usd"] == 0.42


def test_empty_billing_reply_is_pending_not_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("bdh_lab.costs.shutil.which", lambda _name: "runpodctl")
    monkeypatch.setattr(
        "bdh_lab.costs.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[]", stderr=""
        ),
    )

    path = collect_pod_cost(
        experiment_id="synthetic-associative-recall-fastweight-runpod-v1",
        pod_id="pod-123",
        started_at="2026-08-13T10:00:00+00:00",
        ended_at="2026-08-13T10:02:00+00:00",
        output_root=tmp_path,
    )

    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert record["status"] == "pending"
    assert "amount_usd" not in record
