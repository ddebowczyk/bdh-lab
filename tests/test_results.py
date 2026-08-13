from pathlib import Path

from bdh_lab.results import ResultRecord, summarize_results


def test_report_keeps_historical_and_new_metric_sets_comparable() -> None:
    records = [
        ResultRecord(
            path=Path("old-result.yaml"),
            data={
                "experiment_id": "synthetic-associative-recall-smoke-v1",
                "status": "completed",
                "metrics": {"query_accuracy": 0.5, "final_train_loss": 1.0},
            },
        ),
        ResultRecord(
            path=Path("new-result.yaml"),
            data={
                "experiment_id": "synthetic-associative-recall-smoke-v1",
                "status": "completed",
                "metrics": {
                    "validation_query_accuracy": 0.75,
                    "golden_evaluation_query_accuracy": 0.25,
                    "final_train_loss": 0.8,
                },
            },
        ),
    ]

    report = summarize_results(records)
    metrics = report["experiments"][0]["metrics"]

    assert metrics["query_accuracy"]["count"] == 1
    assert metrics["validation_query_accuracy"]["count"] == 1
    assert metrics["golden_evaluation_query_accuracy"]["count"] == 1
    assert metrics["final_train_loss"]["count"] == 2
