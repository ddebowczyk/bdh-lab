from pathlib import Path

from bdh_lab.metadata import (
    Catalogue,
    Document,
    reference_errors,
    repository_root,
    resolved_plan,
    validate_catalogue,
)


def test_checked_in_metadata_is_valid() -> None:
    assert validate_catalogue() == []


def test_plan_resolves_exact_component_versions() -> None:
    plan = resolved_plan(
        repository_root() / "experiments" / "synthetic-associative-recall-fastweight-v1.yaml"
    )

    assert plan["assembly"]["id"] == "bdh-fastweight-recall-v1"
    assert {component["id"] for component in plan["components"]} == {
        "activation-positive-relu-v1",
        "memory-hebbian-fastweight-v1",
        "representation-token-embedding-v1",
    }


def test_unregistered_experiment_path_is_rejected(tmp_path: Path) -> None:
    unregistered = tmp_path / "experiment.yaml"
    unregistered.write_text("schema_version: 1\n", encoding="utf-8")

    try:
        resolved_plan(unregistered)
    except ValueError as error:
        assert "not a registered experiment" in str(error)
    else:
        raise AssertionError("unregistered experiment path was accepted")


def test_unknown_experiment_lineage_is_rejected(tmp_path: Path) -> None:
    experiment = Document(
        path=tmp_path / "candidate.yaml",
        data={
            "id": "candidate-v1",
            "assembly_id": "assembly-v1",
            "profile_id": "profile-v1",
            "lineage": {"parent_experiment_ids": ["missing-v1"]},
        },
    )
    catalogue = Catalogue(
        components={},
        assemblies={"assembly-v1": Document(tmp_path / "assembly.yaml", {"status": "active"})},
        datasets={},
        experiments={"candidate-v1": experiment},
        profiles={"profile-v1": Document(tmp_path / "profile.yaml", {})},
    )

    assert f"{experiment.path}: unknown parent experiment missing-v1" in reference_errors(catalogue)
