from copy import deepcopy

import pytest
import torch

from bdh_lab.datasets import (
    DatasetIntegrityError,
    materialize_synthetic_recall_dataset,
    resolve_held_out_splits,
)
from bdh_lab.metadata import repository_root, resolved_plan
from bdh_lab.models import make_associative_recall_batch, recall_example_id


def _smoke_plan() -> dict[str, object]:
    return resolved_plan(
        repository_root() / "experiments" / "synthetic-associative-recall-smoke-v1.yaml"
    )


def test_held_out_splits_are_checksum_pinned_and_disjoint() -> None:
    splits = resolve_held_out_splits(_smoke_plan())

    assert len(splits.validation_examples) == 16
    assert len(splits.golden_evaluation_examples) == 16
    assert not ({example.id for example in splits.validation_examples} & {
        example.id for example in splits.golden_evaluation_examples}
    )
    assert len(splits.excluded_training_example_ids) == 32


def test_generated_training_examples_exclude_both_held_out_splits() -> None:
    plan = _smoke_plan()
    splits = resolve_held_out_splits(plan)
    task = plan["experiment"]["task"]["config"]
    generator = torch.Generator(device="cpu").manual_seed(7)

    for _ in range(16):
        batch = make_associative_recall_batch(
            symbols=int(task["symbols"]),
            associations_per_sequence=int(task["associations_per_sequence"]),
            batch_size=int(task["batch_size"]),
            generator=generator,
            device=torch.device("cpu"),
            excluded_example_ids=splits.excluded_training_example_ids,
        )
        assert {
            recall_example_id(tokens, target)
            for tokens, target in zip(batch.tokens, batch.targets, strict=True)
        }.isdisjoint(splits.excluded_training_example_ids)


def test_dataset_checksum_mismatch_is_rejected() -> None:
    plan = _smoke_plan()
    invalid = deepcopy(plan["datasets"]["validation"])
    invalid["generation"]["checksum_sha256"] = "0" * 64

    with pytest.raises(DatasetIntegrityError, match="checksum mismatch"):
        materialize_synthetic_recall_dataset(invalid)


def test_long_stream_curve_splits_are_checksum_pinned_and_disjoint() -> None:
    plan = resolved_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-public-bdh-long-stream-v2.yaml"
    )
    splits = resolve_held_out_splits(plan)

    assert len(splits.validation_examples) == 64
    assert len(splits.golden_evaluation_examples) == 128
    assert [split.id for split in splits.evaluation_curve] == ["long-delay-a5", "long-delay-a7"]
    assert [
        len(split.golden_evaluation_examples) for split in splits.evaluation_curve
    ] == [128, 128]
    assert len(splits.excluded_training_example_ids) == 576


def test_capacity_diagnostic_splits_are_fresh_relative_to_the_failed_curve() -> None:
    completed_plan = resolved_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-public-bdh-long-stream-v2.yaml"
    )
    diagnostic_plan = resolved_plan(
        repository_root()
        / "experiments"
        / "synthetic-associative-recall-public-bdh-capacity-diagnostic-v3.yaml"
    )

    assert resolve_held_out_splits(completed_plan).excluded_training_example_ids.isdisjoint(
        resolve_held_out_splits(diagnostic_plan).excluded_training_example_ids
    )
