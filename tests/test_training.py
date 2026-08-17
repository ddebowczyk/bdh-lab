from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bdh_lab.metadata import repository_root, resolved_plan
from bdh_lab.training import ExperimentExecutionError, run_experiment


def test_smoke_experiment_writes_resolved_inputs_and_metrics(tmp_path: Path) -> None:
    plan = resolved_plan(
        repository_root() / "experiments" / "synthetic-associative-recall-smoke-v1.yaml"
    )

    outputs = run_experiment(plan, tmp_path)
    assert len(outputs) == 1
    output = outputs[0]
    result = yaml.safe_load((output / "result.yaml").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert result["resolved"]["assembly"]["id"] == "bdh-fastweight-recall-v1"
    assert len(result["provenance"]["source_snapshot_sha256"]) == 64
    assert 0.0 <= result["metrics"]["validation_query_accuracy"] <= 1.0
    assert 0.0 <= result["metrics"]["golden_evaluation_query_accuracy"] <= 1.0
    assert result["provenance"]["validation_dataset_id"] == (
        "synthetic-associative-recall-s8-a3-validation-v1"
    )
    assert result["provenance"]["golden_evaluation_dataset_id"] == (
        "synthetic-associative-recall-s8-a3-golden-evaluation-v1"
    )


def test_experiment_writes_one_result_per_seed(tmp_path: Path) -> None:
    plan = resolved_plan(
        repository_root() / "experiments" / "synthetic-associative-recall-smoke-v1.yaml"
    )
    plan = deepcopy(plan)
    plan["experiment"]["seeds"] = [3, 5]
    plan["experiment"]["training"]["steps"] = 1

    outputs = run_experiment(plan, tmp_path)

    assert len(outputs) == 2
    seeds = {
        yaml.safe_load((output / "result.yaml").read_text(encoding="utf-8"))["provenance"][
            "seed"
        ]
        for output in outputs
    }
    assert seeds == {3, 5}


def test_failed_seed_writes_a_failure_record(tmp_path: Path) -> None:
    plan = resolved_plan(
        repository_root() / "experiments" / "synthetic-associative-recall-smoke-v1.yaml"
    )
    plan = deepcopy(plan)
    plan["experiment"]["training"]["device"] = "cuda"

    with pytest.raises(ExperimentExecutionError) as error:
        run_experiment(plan, tmp_path)

    outcome = error.value.outcomes[0]
    result = yaml.safe_load((outcome.output_directory / "result.yaml").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "RuntimeError"
