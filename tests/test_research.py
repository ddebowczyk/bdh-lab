from bdh_lab.metadata import load_catalogue, repository_root
from bdh_lab.research import (
    ObjectiveDocument,
    ProgramDocument,
    program_errors,
    reference_errors,
    render_program_status,
    render_status,
    validate_research,
)


def test_checked_in_research_records_are_valid() -> None:
    assert validate_research() == []


def test_research_status_shows_concluded_objective_and_next_action() -> None:
    status = render_status()

    assert "bdh-streaming-state-mechanism-v1" in status
    assert "concluded" in status
    assert "bdh-gpu-streaming-runpod-v2.md" in status


def test_research_program_status_shows_a_completed_failure_diagnostic() -> None:
    status = render_program_status()

    assert "bdh-promise-v1" in status
    assert "public-source-snapshot\tpassed\tcomplete" in status
    assert "public-baseline-conformance\tpassed\tcomplete" in status
    assert "long-stream-retention-curve\tfailed\tcomplete" in status
    assert "retention-failure-diagnosis\tpassed\tcomplete" in status


def test_active_objective_without_a_current_gate_is_rejected() -> None:
    root = repository_root()
    objective = ObjectiveDocument(
        path=root / "research" / "objectives" / "test-objective-v1.yaml",
        data={
            "id": "test-objective-v1",
            "status": "active",
            "gates": [
                {
                    "id": "only-gate",
                    "status": "pending",
                    "evidence": [],
                }
            ],
        },
    )

    errors = reference_errors({"test-objective-v1": objective}, load_catalogue(), root)

    assert any("exactly one current gate" in error for error in errors)


def test_program_cycle_and_unsatisfied_current_node_are_rejected(tmp_path) -> None:
    source_plan = tmp_path / "plan.md"
    source_plan.write_text("# plan\n", encoding="utf-8")
    program = ProgramDocument(
        path=tmp_path / "program.yaml",
        data={
            "id": "test-program-v1",
            "status": "active",
            "source_plan": "plan.md",
            "nodes": [
                {
                    "id": "first",
                    "status": "current",
                    "requires": [{"node_id": "second", "outcome": "passed"}],
                    "evidence": [],
                },
                {
                    "id": "second",
                    "status": "pending",
                    "requires": [{"node_id": "first", "outcome": "passed"}],
                    "evidence": [],
                },
            ],
        },
    )

    errors = program_errors({"test-program-v1": program}, {}, load_catalogue(), tmp_path)

    assert any("acyclic" in error for error in errors)
    assert any("does not satisfy" in error for error in errors)
