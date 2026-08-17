"""Validate and render compact research-objective records."""

from __future__ import annotations

import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bdh_lab.metadata import ROOT, Catalogue, MetadataError, load_catalogue, load_yaml
from bdh_lab.results import load_costs, load_results, summarize_results

OBJECTIVE_PATTERN = "research/objectives/*.yaml"
PROGRAM_PATTERN = "research/programs/*.yaml"
OBJECTIVE_SCHEMA_NAME = "research-objective.schema.yaml"
PROGRAM_SCHEMA_NAME = "research-program.schema.yaml"


class ResearchError(ValueError):
    """Raised when research records cannot support a trustworthy status view."""


@dataclass(frozen=True)
class ObjectiveDocument:
    """One validated research objective and its on-disk source."""

    path: Path
    data: dict[str, Any]


@dataclass(frozen=True)
class ProgramDocument:
    """One versioned directed research program and its on-disk source."""

    path: Path
    data: dict[str, Any]


def _schema_errors(root: Path, document: Path, schema_name: str) -> list[str]:
    executable = shutil.which("ys")
    if executable is None:
        return ["ys is required to validate research-objective records"]
    completed = subprocess.run(
        [
            executable,
            "--json",
            "--schema",
            str(root / "schemas" / "v1" / schema_name),
            str(document),
        ],
        capture_output=True,
        check=False,
        cwd=root,
        text=True,
    )
    if completed.returncode == 0:
        return []
    detail = completed.stdout.strip() or completed.stderr.strip() or "schema validation failed"
    return [f"{document.relative_to(root)}: {detail}"]


def load_objectives(root: Path | None = None) -> dict[str, ObjectiveDocument]:
    """Load unique research-objective documents without changing their status."""

    root = root or ROOT
    objectives: dict[str, ObjectiveDocument] = {}
    for path in sorted(root.glob(OBJECTIVE_PATTERN)):
        try:
            data = load_yaml(path)
        except MetadataError as error:
            raise ResearchError(str(error)) from error
        identifier = data.get("id")
        if not isinstance(identifier, str):
            continue
        if identifier in objectives:
            raise ResearchError(
                f"{path.relative_to(root)}: duplicate objective ID {identifier}; "
                f"first used by {objectives[identifier].path.relative_to(root)}"
            )
        objectives[identifier] = ObjectiveDocument(path=path, data=data)
    return objectives


def load_programs(root: Path | None = None) -> dict[str, ProgramDocument]:
    """Load unique research-program documents without changing their status."""

    root = root or ROOT
    programs: dict[str, ProgramDocument] = {}
    for path in sorted(root.glob(PROGRAM_PATTERN)):
        try:
            data = load_yaml(path)
        except MetadataError as error:
            raise ResearchError(str(error)) from error
        identifier = data.get("id")
        if not isinstance(identifier, str):
            continue
        if identifier in programs:
            raise ResearchError(
                f"{path.relative_to(root)}: duplicate program ID {identifier}; "
                f"first used by {programs[identifier].path.relative_to(root)}"
            )
        programs[identifier] = ProgramDocument(path=path, data=data)
    return programs


def _safe_relative_path(root: Path, value: str) -> Path | None:
    path = (root / value).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def reference_errors(
    objectives: dict[str, ObjectiveDocument], catalogue: Catalogue, root: Path
) -> list[str]:
    """Check gate invariants and links that a single YAML schema cannot express."""

    errors: list[str] = []
    for objective in objectives.values():
        data = objective.data
        label = objective.path.relative_to(root)
        gates = data.get("gates", [])
        if not isinstance(gates, list):
            continue
        gate_ids = [gate.get("id") for gate in gates if isinstance(gate, dict)]
        if len(gate_ids) != len(set(gate_ids)):
            errors.append(f"{label}: gate IDs must be unique")
        current = [
            gate
            for gate in gates
            if isinstance(gate, dict) and gate.get("status") == "current"
        ]
        if data.get("status") == "active" and len(current) != 1:
            errors.append(f"{label}: active objective must have exactly one current gate")
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            evidence = gate.get("evidence", [])
            if gate.get("status") in {"passed", "failed"} and not evidence:
                errors.append(f"{label}: {gate.get('id')} needs evidence when {gate.get('status')}")
            for item in evidence if isinstance(evidence, list) else []:
                if not isinstance(item, dict):
                    continue
                if item.get("kind") in {"source", "test"}:
                    value = item.get("path")
                    path = _safe_relative_path(root, value) if isinstance(value, str) else None
                    if path is None or not path.is_file():
                        errors.append(f"{label}: missing {item.get('kind')} evidence {value}")
                if item.get("kind") == "experiment":
                    identifier = item.get("experiment_id")
                    if identifier not in catalogue.experiments:
                        errors.append(f"{label}: unknown evidence experiment {identifier}")

    for documents in (
        catalogue.components.values(),
        catalogue.assemblies.values(),
        catalogue.experiments.values(),
    ):
        for document in documents:
            identifier = document.data.get("research_objective_id")
            if identifier is not None and identifier not in objectives:
                errors.append(
                    f"{document.path.relative_to(root)}: unknown research objective {identifier}"
                )
    return errors


def _program_nodes(program: ProgramDocument) -> dict[str, dict[str, Any]]:
    """Index one program's node mappings by ID after schema validation."""

    return {
        node["id"]: node
        for node in program.data.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _requirements_are_satisfied(node: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> bool:
    """Return true only when every declared predecessor has its required outcome."""

    requirements = node.get("requires", [])
    if not isinstance(requirements, list):
        return False
    for requirement in requirements:
        if not isinstance(requirement, dict):
            return False
        predecessor = requirement.get("node_id")
        if (
            predecessor not in nodes
            or nodes[predecessor].get("status") != requirement.get("outcome")
        ):
            return False
    return True


def _node_is_ready(node: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> bool:
    """Return true when a pending or current node has all required outcomes."""

    return node.get("status") in {"pending", "current"} and _requirements_are_satisfied(
        node, nodes
    )


def _node_readiness(node: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    """Return the user-facing derived readiness label for one program node."""

    status = node.get("status")
    if status == "parked":
        return "parked"
    if status in {"passed", "failed"}:
        return "complete"
    return "ready" if _node_is_ready(node, nodes) else "waiting"


def _has_program_cycle(nodes: dict[str, dict[str, Any]]) -> bool:
    """Detect dependency cycles, including cycles through outcome-specific edges."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for requirement in nodes[node_id].get("requires", []):
            if not isinstance(requirement, dict):
                continue
            predecessor = requirement.get("node_id")
            if isinstance(predecessor, str) and predecessor in nodes and visit(predecessor):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in nodes)


def program_errors(
    programs: dict[str, ProgramDocument],
    objectives: dict[str, ObjectiveDocument],
    catalogue: Catalogue,
    root: Path,
) -> list[str]:
    """Check DAG routing, status, and cross-record references for programs."""

    errors: list[str] = []
    for program in programs.values():
        label = program.path.relative_to(root)
        source_plan = program.data.get("source_plan")
        source_path = (
            _safe_relative_path(root, source_plan)
            if isinstance(source_plan, str)
            else None
        )
        if source_path is None or not source_path.is_file():
            errors.append(f"{label}: missing source plan {source_plan}")

        nodes = _program_nodes(program)
        declared_nodes = program.data.get("nodes", [])
        if len(nodes) != len(declared_nodes):
            errors.append(f"{label}: node IDs must be unique")
        if _has_program_cycle(nodes):
            errors.append(f"{label}: program dependencies must be acyclic")

        for node_id, node in nodes.items():
            objective_id = node.get("objective_id")
            if objective_id is not None and objective_id not in objectives:
                errors.append(f"{label}: {node_id} has unknown objective {objective_id}")
            requirements = node.get("requires", [])
            seen_requirements: set[tuple[str, str]] = set()
            for requirement in requirements if isinstance(requirements, list) else []:
                if not isinstance(requirement, dict):
                    continue
                predecessor = requirement.get("node_id")
                outcome = requirement.get("outcome")
                if predecessor not in nodes:
                    errors.append(f"{label}: {node_id} requires unknown node {predecessor}")
                elif predecessor == node_id:
                    errors.append(f"{label}: {node_id} cannot require itself")
                if isinstance(predecessor, str) and isinstance(outcome, str):
                    key = (predecessor, outcome)
                    if key in seen_requirements:
                        errors.append(f"{label}: {node_id} has duplicate requirement {predecessor}")
                    seen_requirements.add(key)
            evidence = node.get("evidence", [])
            if node.get("status") in {"passed", "failed"} and not evidence:
                errors.append(f"{label}: {node_id} needs evidence when {node.get('status')}")
            for item in evidence if isinstance(evidence, list) else []:
                if not isinstance(item, dict):
                    continue
                if item.get("kind") in {"source", "test"}:
                    value = item.get("path")
                    path = _safe_relative_path(root, value) if isinstance(value, str) else None
                    if path is None or not path.is_file():
                        errors.append(f"{label}: {node_id} has missing evidence {value}")
                if item.get("kind") == "experiment":
                    identifier = item.get("experiment_id")
                    if identifier not in catalogue.experiments:
                        errors.append(f"{label}: {node_id} has unknown experiment {identifier}")
            if (
                node.get("status") in {"current", "passed", "failed"}
                and not _requirements_are_satisfied(node, nodes)
            ):
                errors.append(f"{label}: {node_id} does not satisfy its required outcomes")

        current_nodes = [node for node in nodes.values() if node.get("status") == "current"]
        if program.data.get("status") == "active" and not current_nodes:
            errors.append(f"{label}: active program must have at least one current node")
    return errors


def validate_research(root: Path | None = None) -> list[str]:
    """Return research schema and link errors without changing record status."""

    root = root or ROOT
    errors: list[str] = []
    for path in sorted(root.glob(OBJECTIVE_PATTERN)):
        errors.extend(_schema_errors(root, path, OBJECTIVE_SCHEMA_NAME))
    for path in sorted(root.glob(PROGRAM_PATTERN)):
        errors.extend(_schema_errors(root, path, PROGRAM_SCHEMA_NAME))
    try:
        objectives = load_objectives(root)
        programs = load_programs(root)
        catalogue = load_catalogue(root)
    except (MetadataError, ResearchError) as error:
        return [*errors, str(error)]
    errors.extend(reference_errors(objectives, catalogue, root))
    errors.extend(program_errors(programs, objectives, catalogue, root))
    return errors


def _current_gate(objective: ObjectiveDocument) -> dict[str, Any] | None:
    for gate in objective.data.get("gates", []):
        if isinstance(gate, dict) and gate.get("status") == "current":
            return gate
    return None


def render_status(root: Path | None = None) -> str:
    """Render a short read-only research status table."""

    root = root or ROOT
    errors = validate_research(root)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ResearchError(f"research validation failed:\n{detail}")
    rows = ["objective\tstatus\tcurrent_gate\tevidence\tnext_action\tdecision"]
    for identifier, objective in sorted(load_objectives(root).items()):
        current = _current_gate(objective)
        evidence_count = len(current["evidence"]) if current is not None else 0
        action = objective.data["next_action"]
        rows.append(
            "\t".join(
                (
                    identifier,
                    str(objective.data["status"]),
                    str(current["id"]) if current is not None else "-",
                    str(evidence_count),
                    str(action["target"]),
                    str(objective.data["decision"]),
                )
            )
        )
    return "\n".join(rows)


def render_program_status(root: Path | None = None) -> str:
    """Render every program node with computed readiness, without changing state."""

    root = root or ROOT
    errors = validate_research(root)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ResearchError(f"research validation failed:\n{detail}")
    rows = ["program\tprogram_status\tnode\tnode_status\treadiness\tnext_action"]
    for program_id, program in sorted(load_programs(root).items()):
        nodes = _program_nodes(program)
        for node_id, node in nodes.items():
            readiness = _node_readiness(node, nodes)
            rows.append(
                "\t".join(
                    (
                        program_id,
                        str(program.data["status"]),
                        node_id,
                        str(node["status"]),
                        readiness,
                        str(node["next_action"]),
                    )
                )
            )
    return "\n".join(rows)


def resolved_program(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Resolve one program into its nodes and computed readiness without side effects."""

    root = root or ROOT
    errors = validate_research(root)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ResearchError(f"research validation failed:\n{detail}")
    target = path.resolve()
    programs = load_programs(root)
    program = next((item for item in programs.values() if item.path.resolve() == target), None)
    if program is None:
        raise ResearchError(f"{path}: not a registered research program")
    nodes = _program_nodes(program)
    resolved_nodes = []
    for _node_id, node in nodes.items():
        resolved_nodes.append(
            {
                **node,
                "readiness": _node_readiness(node, nodes),
            }
        )
    return {"program": deepcopy(program.data), "nodes": resolved_nodes}


def resolved_objective_plan(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Resolve one objective to its linked candidates and existing evidence summaries."""

    root = root or ROOT
    errors = validate_research(root)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise ResearchError(f"research validation failed:\n{detail}")
    target = path.resolve()
    objectives = load_objectives(root)
    objective = next((item for item in objectives.values() if item.path.resolve() == target), None)
    if objective is None:
        raise ResearchError(f"{path}: not a registered research objective")
    catalogue = load_catalogue(root)
    candidates: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    for kind, documents in (
        ("component", catalogue.components.values()),
        ("assembly", catalogue.assemblies.values()),
        ("experiment", catalogue.experiments.values()),
    ):
        for document in documents:
            if document.data.get("research_objective_id") != objective.data["id"]:
                continue
            candidates.append(
                {
                    "kind": kind,
                    "id": str(document.data["id"]),
                    "status": str(document.data["status"]),
                    "path": str(document.path.relative_to(root)),
                }
            )
            if kind == "experiment":
                reports.extend(
                    summarize_results(
                        load_results(), costs=load_costs(), experiment_id=str(document.data["id"])
                    )["experiments"]
                )
    return {
        "objective": objective.data,
        "candidates": sorted(candidates, key=lambda item: (item["kind"], item["id"])),
        "evidence": reports,
    }
