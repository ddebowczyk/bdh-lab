"""The small command surface for validated experiment work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from bdh_lab.costs import collect_pod_cost
from bdh_lab.metadata import MetadataError, repository_root, resolved_plan, validate_catalogue
from bdh_lab.results import load_costs, load_results, summarize_results
from bdh_lab.runpod import (
    apply_pod_plan,
    doctor,
    launch_remote_experiment,
    plan_as_dict,
    render_pod_plan,
)
from bdh_lab.training import ExperimentExecutionError, run_experiment


def _experiment_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repository_root() / path
    return path


def _parse_args(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bdh-lab")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate", help="validate all checked-in experiment metadata")

    plan = subcommands.add_parser("plan", help="resolve an experiment without executing it")
    plan.add_argument("experiment")

    run = subcommands.add_parser("run", help="run one active local experiment")
    run.add_argument("experiment")
    run.add_argument("--output-root", type=Path)
    run.add_argument("--runner-provider", choices=("local", "runpod"), default="local")

    report = subcommands.add_parser("report", help="summarize schema-valid local run records")
    report.add_argument("--experiment-id")
    report.add_argument("--format", choices=("yaml", "json"), default="yaml")

    runpod = subcommands.add_parser("runpod", help="plan or apply a Runpod Pod")
    runpod_subcommands = runpod.add_subparsers(dest="runpod_command", required=True)
    runpod_subcommands.add_parser("doctor", help="check local Runpod prerequisites")
    runpod_plan = runpod_subcommands.add_parser("plan", help="render a Pod command")
    runpod_plan.add_argument("experiment")
    runpod_submit = runpod_subcommands.add_parser("submit", help="create the planned Pod")
    runpod_submit.add_argument("experiment")
    runpod_submit.add_argument("--apply", action="store_true", help="permit remote Pod creation")
    runpod_launch = runpod_subcommands.add_parser(
        "launch", help="create, run, retrieve, cost, and terminate one Pod"
    )
    runpod_launch.add_argument("experiment")
    runpod_launch.add_argument("--apply", action="store_true", help="permit remote Pod creation")
    runpod_launch.add_argument(
        "--keep-pod", action="store_true", help="do not delete the Pod after evidence retrieval"
    )
    runpod_cost = runpod_subcommands.add_parser(
        "cost", help="append a billing record for one existing Runpod Pod"
    )
    runpod_cost.add_argument("experiment")
    runpod_cost.add_argument("--pod-id", required=True)
    runpod_cost.add_argument("--started-at", required=True)
    runpod_cost.add_argument("--ended-at", required=True)
    runpod_cost.add_argument("--run-id", action="append", default=[])
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run one command and return a process status suitable for Just recipes."""

    args = _parse_args(arguments)
    try:
        if args.command == "validate":
            errors = validate_catalogue()
            if errors:
                print("Metadata validation failed:", file=sys.stderr)
                print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
                return 1
            print("Metadata schemas and references are valid.")
            return 0
        if args.command == "plan":
            print(yaml.safe_dump(resolved_plan(_experiment_path(args.experiment)), sort_keys=False))
            return 0
        if args.command == "run":
            plan = resolved_plan(_experiment_path(args.experiment))
            outputs = run_experiment(
                plan, args.output_root, runner_provider=args.runner_provider
            )
            for output in outputs:
                print(output.relative_to(repository_root()))
            return 0
        if args.command == "report":
            report = summarize_results(
                load_results(), costs=load_costs(), experiment_id=args.experiment_id
            )
            if args.format == "json":
                print(json.dumps(report, indent=2) + "\n")
            else:
                print(yaml.safe_dump(report, sort_keys=False))
            return 0
        if args.command == "runpod" and args.runpod_command == "doctor":
            print("\n".join(doctor()))
            return 0
        if args.command == "runpod":
            pod_plan = render_pod_plan(_experiment_path(args.experiment))
            if args.runpod_command == "plan":
                print(json.dumps(plan_as_dict(pod_plan), indent=2) + "\n")
                return 0
            if args.runpod_command == "cost":
                record = collect_pod_cost(
                    experiment_id=pod_plan.experiment_id,
                    pod_id=args.pod_id,
                    started_at=args.started_at,
                    ended_at=args.ended_at,
                    run_ids=args.run_id,
                )
                print(record.relative_to(repository_root()))
                return 0
            if not args.apply:
                raise MetadataError("Pod creation requires the explicit --apply flag")
            if args.runpod_command == "submit":
                print(apply_pod_plan(pod_plan))
                return 0
            if args.runpod_command == "launch":
                remote_run = launch_remote_experiment(pod_plan, keep_pod=args.keep_pod)
                print(f"pod_id: {remote_run.pod_id}")
                for output in remote_run.output_directories:
                    print(output.relative_to(repository_root()))
                print(remote_run.cost_record.relative_to(repository_root()))
                return 0
    except ExperimentExecutionError as error:
        for outcome in error.outcomes:
            print(outcome.output_directory.relative_to(repository_root()))
        print(f"bdh-lab: {error}", file=sys.stderr)
        return 1
    except (MetadataError, OSError, RuntimeError, ValueError) as error:
        print(f"bdh-lab: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
