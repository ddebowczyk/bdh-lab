"""Load and validate the versioned experiment control plane."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "schemas" / "v1"

DOCUMENT_KINDS = {
    "components": ("components/*.yaml", "component.schema.yaml"),
    "assemblies": ("assemblies/*.yaml", "assembly.schema.yaml"),
    "datasets": ("datasets/*.yaml", "dataset.schema.yaml"),
    "experiments": ("experiments/*.yaml", "experiment.schema.yaml"),
    "profiles": ("profiles/*.yaml", "profile.schema.yaml"),
}


class MetadataError(ValueError):
    """Raised when metadata cannot be resolved into a runnable experiment."""


@dataclass(frozen=True)
class Document:
    """One parsed metadata document and its on-disk source."""

    path: Path
    data: dict[str, Any]


@dataclass(frozen=True)
class Catalogue:
    """All versioned candidate inputs, indexed by document type and ID."""

    components: dict[str, Document]
    assemblies: dict[str, Document]
    datasets: dict[str, Document]
    experiments: dict[str, Document]
    profiles: dict[str, Document]


def repository_root() -> Path:
    """Return the checkout root without depending on the caller's directory."""

    return ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping and report the source when its shape is invalid."""

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise MetadataError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise MetadataError(f"{path}: document must be a YAML mapping")
    return loaded


def _documents(root: Path, pattern: str) -> list[Document]:
    return [Document(path, load_yaml(path)) for path in sorted(root.glob(pattern))]


def _index(documents: list[Document], kind: str) -> dict[str, Document]:
    indexed: dict[str, Document] = {}
    for document in documents:
        identifier = document.data.get("id")
        if not isinstance(identifier, str):
            continue
        if identifier in indexed:
            raise MetadataError(
                f"{document.path}: duplicate {kind} id {identifier}; "
                f"first used by {indexed[identifier].path}"
            )
        indexed[identifier] = document
    return indexed


def load_catalogue(root: Path | None = None) -> Catalogue:
    """Read metadata into stable ID indexes without applying any decision."""

    root = root or ROOT
    parsed = {
        kind: _documents(root, pattern)
        for kind, (pattern, _schema_name) in DOCUMENT_KINDS.items()
    }
    return Catalogue(
        components=_index(parsed["components"], "component"),
        assemblies=_index(parsed["assemblies"], "assembly"),
        datasets=_index(parsed["datasets"], "dataset"),
        experiments=_index(parsed["experiments"], "experiment"),
        profiles=_index(parsed["profiles"], "profile"),
    )


def _validate_with_ys(root: Path, document: Path, schema: Path) -> list[str]:
    ys = shutil.which("ys")
    if ys is None:
        return ["ys is required to validate versioned YAML schemas; install yaml-schema"]
    completed = subprocess.run(
        [ys, "--json", "--schema", str(schema), str(document)],
        capture_output=True,
        check=False,
        cwd=root,
        text=True,
    )
    if completed.returncode == 0:
        return []
    detail = completed.stdout.strip() or completed.stderr.strip() or "unknown schema error"
    try:
        label = document.relative_to(root)
    except ValueError:
        label = document
    return [f"{label}: {detail}"]


def schema_errors(root: Path | None = None) -> list[str]:
    """Validate every checked-in control record against its versioned schema."""

    root = root or ROOT
    errors: list[str] = []
    for pattern, schema_name in DOCUMENT_KINDS.values():
        for document in sorted(root.glob(pattern)):
            try:
                data = load_yaml(document)
            except MetadataError as error:
                errors.append(str(error))
                continue
            version = data.get("schema_version")
            if not isinstance(version, int) or version < 1:
                errors.append(f"{document}: schema_version must be a positive integer")
                continue
            schema = root / "schemas" / f"v{version}" / schema_name
            if not schema.is_file():
                errors.append(f"{document}: no {schema_name} exists for schema_version {version}")
                continue
            errors.extend(_validate_with_ys(root, document, schema))
    return errors


def reference_errors(catalogue: Catalogue) -> list[str]:
    """Check relationships that a single-file YAML schema cannot express."""

    errors: list[str] = []
    for assembly in catalogue.assemblies.values():
        data = assembly.data
        for reference in data.get("components", []):
            component_id = reference.get("component_id")
            component = catalogue.components.get(component_id)
            if component is None:
                errors.append(f"{assembly.path}: unknown component {component_id}")
            elif data.get("status") == "active" and component.data.get("status") != "active":
                errors.append(
                    f"{assembly.path}: active assembly uses non-active component {component_id}"
                )

    for experiment in catalogue.experiments.values():
        data = experiment.data
        assembly_id = data.get("assembly_id")
        profile_id = data.get("profile_id")
        assembly = catalogue.assemblies.get(assembly_id)
        if assembly is None:
            errors.append(f"{experiment.path}: unknown assembly {assembly_id}")
        elif data.get("status") == "active" and assembly.data.get("status") != "active":
            errors.append(
                f"{experiment.path}: active experiment uses non-active assembly {assembly_id}"
            )
        if profile_id not in catalogue.profiles:
            errors.append(f"{experiment.path}: unknown profile {profile_id}")
        data_splits = data.get("data")
        if isinstance(data_splits, dict):
            validation_id = data_splits.get("validation_dataset_id")
            golden_id = data_splits.get("golden_evaluation_dataset_id")
            split_pairs: list[tuple[str, object, object]] = [
                ("base", validation_id, golden_id)
            ]
            evaluation_curve = data_splits.get("evaluation_curve", [])
            if isinstance(evaluation_curve, list):
                for condition in evaluation_curve:
                    if isinstance(condition, dict):
                        split_pairs.append(
                            (
                                str(condition.get("id", "unnamed")),
                                condition.get("validation_dataset_id"),
                                condition.get("golden_evaluation_dataset_id"),
                            )
                        )
            expected_ids = {
                dataset_id
                for _name, validation, golden in split_pairs
                for dataset_id in (validation, golden)
            }
            no_overlap_with = data_splits.get("training", {}).get("no_overlap_with", [])
            if set(no_overlap_with) != expected_ids:
                errors.append(
                    f"{experiment.path}: training no_overlap_with must name every held-out split"
                )
            task_config = data.get("task", {}).get("config", {})
            seen_conditions: set[str] = set()
            for condition_name, condition_validation_id, condition_golden_id in split_pairs:
                if condition_name in seen_conditions:
                    errors.append(
                        f"{experiment.path}: duplicate evaluation condition {condition_name}"
                    )
                seen_conditions.add(condition_name)
                condition_datasets: list[Document] = []
                for dataset_id, expected_role in (
                    (condition_validation_id, "validation"),
                    (condition_golden_id, "golden_evaluation"),
                ):
                    dataset = catalogue.datasets.get(dataset_id)
                    if dataset is None:
                        errors.append(f"{experiment.path}: unknown dataset {dataset_id}")
                    elif dataset.data.get("status") != "active":
                        errors.append(
                            f"{experiment.path}: held-out dataset {dataset_id} is not active"
                        )
                    elif dataset.data.get("role") != expected_role:
                        errors.append(
                            f"{experiment.path}: dataset {dataset_id} must have role "
                            f"{expected_role}"
                        )
                    else:
                        condition_datasets.append(dataset)
                if len(condition_datasets) != 2:
                    continue
                validation_config = condition_datasets[0].data.get("config")
                golden_config = condition_datasets[1].data.get("config")
                if validation_config != golden_config:
                    errors.append(
                        f"{experiment.path}: {condition_name} validation and golden "
                        "configurations differ"
                    )
                    continue
                if (
                    not isinstance(validation_config, dict)
                    or validation_config.get("symbols") != task_config.get("symbols")
                ):
                    errors.append(
                        f"{experiment.path}: {condition_name} datasets do not match task symbols"
                    )
                    continue
                associations = validation_config.get("associations_per_sequence")
                if not isinstance(associations, int) or associations < int(
                    task_config.get("associations_per_sequence", 0)
                ):
                    errors.append(
                        f"{experiment.path}: {condition_name} datasets use an invalid "
                        "association count"
                    )
                if condition_name == "base" and associations != task_config.get(
                    "associations_per_sequence"
                ):
                    errors.append(
                        f"{experiment.path}: base datasets do not match task association count"
                    )
        lineage = data.get("lineage", {})
        if isinstance(lineage, dict):
            for parent_id in lineage.get("parent_experiment_ids", []):
                if parent_id not in catalogue.experiments:
                    errors.append(f"{experiment.path}: unknown parent experiment {parent_id}")
                elif parent_id == data.get("id"):
                    errors.append(f"{experiment.path}: experiment cannot parent itself")
    return errors


def validate_catalogue(root: Path | None = None) -> list[str]:
    """Return all schema and relationship errors without modifying repository state."""

    root = root or ROOT
    errors = schema_errors(root)
    try:
        errors.extend(reference_errors(load_catalogue(root)))
    except MetadataError as error:
        errors.append(str(error))
    return errors


def validate_result_document(path: Path) -> list[str]:
    """Validate one generated run record against the versioned result contract."""

    return _validate_with_ys(ROOT, path, SCHEMA_ROOT / "result.schema.yaml")


def validate_cost_document(path: Path) -> list[str]:
    """Validate one immutable provider billing record."""

    return _validate_with_ys(ROOT, path, SCHEMA_ROOT / "cost.schema.yaml")


def require_experiment(
    experiment_path: Path, root: Path | None = None
) -> tuple[Catalogue, Document]:
    """Resolve one file to a known experiment after validating the complete catalogue."""

    root = root or ROOT
    errors = validate_catalogue(root)
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise MetadataError(f"metadata validation failed:\n{details}")
    catalogue = load_catalogue(root)
    resolved_path = experiment_path.resolve()
    for experiment in catalogue.experiments.values():
        if experiment.path.resolve() == resolved_path:
            return catalogue, experiment
    raise MetadataError(f"{experiment_path}: not a registered experiment document")


def resolved_plan(experiment_path: Path, root: Path | None = None) -> dict[str, Any]:
    """Return exact inputs for one candidate without starting a training run."""

    catalogue, experiment = require_experiment(experiment_path, root)
    data = experiment.data
    assembly = catalogue.assemblies[data["assembly_id"]]
    profile = catalogue.profiles[data["profile_id"]]
    components = [
        catalogue.components[reference["component_id"]].data
        for reference in assembly.data["components"]
    ]
    data_splits = data["data"]
    evaluation_curve = []
    for condition in data_splits.get("evaluation_curve", []):
        evaluation_curve.append(
            {
                "id": condition["id"],
                "validation": catalogue.datasets[condition["validation_dataset_id"]].data,
                "golden_evaluation": catalogue.datasets[
                    condition["golden_evaluation_dataset_id"]
                ].data,
            }
        )
    return {
        "experiment": data,
        "assembly": assembly.data,
        "profile": profile.data,
        "components": components,
        "datasets": {
            "validation": catalogue.datasets[data_splits["validation_dataset_id"]].data,
            "golden_evaluation": catalogue.datasets[
                data_splits["golden_evaluation_dataset_id"]
            ].data,
            "evaluation_curve": evaluation_curve,
        },
    }
