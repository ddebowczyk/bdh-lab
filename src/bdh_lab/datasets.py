"""Checksum-pinned synthetic data splits for associative-recall experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from bdh_lab.models import RecallBatch, make_associative_recall_batch, recall_example_id


class DatasetIntegrityError(ValueError):
    """Raised when a declared frozen split cannot be recreated exactly."""


@dataclass(frozen=True)
class RecallExample:
    """One canonical input/answer pair from a frozen synthetic split."""

    tokens: tuple[int, ...]
    target: int

    @property
    def id(self) -> str:
        return recall_example_id(self.tokens, self.target)

    def canonical(self) -> str:
        return json.dumps(
            {"target": self.target, "tokens": list(self.tokens)},
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class HeldOutSplits:
    """Separate selection and final-evaluation data for one task configuration."""

    validation_id: str
    validation_checksum_sha256: str
    validation_examples: tuple[RecallExample, ...]
    golden_evaluation_id: str
    golden_evaluation_checksum_sha256: str
    golden_evaluation_examples: tuple[RecallExample, ...]

    @property
    def excluded_training_example_ids(self) -> frozenset[str]:
        return frozenset(
            example.id
            for example in (*self.validation_examples, *self.golden_evaluation_examples)
        )


def _seed_value(seed: str) -> int:
    """Derive a portable positive torch seed from a versioned string seed."""

    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big") % (2**63)


def _examples_checksum(examples: Iterable[RecallExample]) -> str:
    """Hash the canonical serialized split, including its stable ordering."""

    payload = "".join(f"{example.canonical()}\n" for example in examples).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def materialize_synthetic_recall_dataset(dataset: dict[str, Any]) -> tuple[RecallExample, ...]:
    """Regenerate and verify one checksum-pinned synthetic recall split."""

    if dataset.get("kind") != "synthetic_associative_recall_v1":
        raise DatasetIntegrityError(f"{dataset.get('id')}: unsupported dataset kind")
    generation = dataset.get("generation")
    if not isinstance(generation, dict):
        raise DatasetIntegrityError(f"{dataset.get('id')}: missing generation definition")
    if generation.get("algorithm") != "torch-synthetic-associative-recall-v1":
        raise DatasetIntegrityError(f"{dataset.get('id')}: unsupported generation algorithm")
    config = dataset.get("config")
    if not isinstance(config, dict):
        raise DatasetIntegrityError(f"{dataset.get('id')}: missing task configuration")
    try:
        count = int(generation["count"])
        symbols = int(config["symbols"])
        associations_per_sequence = int(config["associations_per_sequence"])
        seed = str(generation["seed"])
        expected_checksum = str(generation["checksum_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetIntegrityError(
            f"{dataset.get('id')}: invalid generation definition"
        ) from error
    generator = torch.Generator(device="cpu").manual_seed(_seed_value(seed))
    examples: list[RecallExample] = []
    seen: set[str] = set()
    while len(examples) < count:
        batch = make_associative_recall_batch(
            symbols=symbols,
            associations_per_sequence=associations_per_sequence,
            batch_size=1,
            generator=generator,
            device=torch.device("cpu"),
        )
        example = RecallExample(
            tokens=tuple(int(value) for value in batch.tokens[0].tolist()),
            target=int(batch.targets[0]),
        )
        if example.id not in seen:
            examples.append(example)
            seen.add(example.id)
    actual_checksum = _examples_checksum(examples)
    if actual_checksum != expected_checksum:
        raise DatasetIntegrityError(
            f"{dataset.get('id')}: checksum mismatch; expected {expected_checksum}, "
            f"got {actual_checksum}"
        )
    return tuple(examples)


def recall_batch_from_examples(
    examples: Iterable[RecallExample], device: torch.device
) -> RecallBatch:
    """Move a frozen set of examples to the requested training or evaluation device."""

    materialized = tuple(examples)
    if not materialized:
        raise DatasetIntegrityError("a held-out split cannot be empty")
    return RecallBatch(
        tokens=torch.tensor(
            [example.tokens for example in materialized], dtype=torch.long, device=device
        ),
        targets=torch.tensor(
            [example.target for example in materialized], dtype=torch.long, device=device
        ),
    )


def resolve_held_out_splits(plan: dict[str, Any]) -> HeldOutSplits:
    """Load the validation and golden splits and prove they do not overlap."""

    datasets = plan.get("datasets")
    if not isinstance(datasets, dict):
        raise DatasetIntegrityError("resolved plan has no held-out datasets")
    validation = datasets.get("validation")
    golden = datasets.get("golden_evaluation")
    if not isinstance(validation, dict) or not isinstance(golden, dict):
        raise DatasetIntegrityError("resolved plan has incomplete held-out datasets")
    validation_examples = materialize_synthetic_recall_dataset(validation)
    golden_examples = materialize_synthetic_recall_dataset(golden)
    validation_ids = {example.id for example in validation_examples}
    golden_ids = {example.id for example in golden_examples}
    if validation_ids & golden_ids:
        raise DatasetIntegrityError("validation and golden evaluation splits overlap")
    return HeldOutSplits(
        validation_id=str(validation["id"]),
        validation_checksum_sha256=str(validation["generation"]["checksum_sha256"]),
        validation_examples=validation_examples,
        golden_evaluation_id=str(golden["id"]),
        golden_evaluation_checksum_sha256=str(golden["generation"]["checksum_sha256"]),
        golden_evaluation_examples=golden_examples,
    )


def accuracy(
    model: torch.nn.Module, examples: Iterable[RecallExample], device: torch.device
) -> float:
    """Measure exact associative recall on one complete frozen split."""

    batch = recall_batch_from_examples(examples, device)
    return float((model(batch.tokens).argmax(dim=-1) == batch.targets).float().mean().item())
