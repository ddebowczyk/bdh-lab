# BDH Lab agent guide

Use Simplified Technical English. State what changed and name the evidence.

## Purpose and orientation

This repository is a metadata-driven experimental lab for BDH-inspired
fast-weight models and transformer baselines. It records a reproducible chain:

```text
component -> assembly -> experiment -> immutable run and cost evidence
```

Start here:

- `README.md` gives the repository overview.
- `docs/experiment-system.md` defines the evidence, data-split, and candidate
  lifecycle rules.
- `docs/plans/` holds dated Markdown source plans for research programs.
- `research/programs/` holds the validated DAG that routes research nodes.
- `research/objectives/` holds the evidence gates for each research question.
- `components/` defines replaceable architectural choices.
- `assemblies/` pins a model implementation to exact component versions.
- `experiments/` binds an assembly, datasets, execution profile, hypothesis,
  and budget.
- `profiles/` defines local or Runpod execution settings.
- `src/bdh_lab/` implements validation, training, evidence, and Runpod launch.
- `var/runs/` contains generated evidence. It is ignored by Git. Never edit a
  prior run record.

## Rules

- Use `uv` for every Python command. Do not use a global virtual environment.
- Use `just` for project commands. Run `just --list --unsorted` before using
  an unfamiliar recipe.
- Treat `components/`, `assemblies/`, `experiments/`, and `profiles/` as
  versioned metadata. Use `yq` to inspect or preview YAML changes and `ys` to
  validate them.
- Keep provider credentials only in environment variables or provider-managed
  user configuration. Never write credentials to profiles, run records, or
  command output.
- A result can support promotion or rejection, but it must not silently edit a
  component or assembly. Record design decisions as reviewed metadata changes.
- An agent may create and validate a `proposed` candidate. It must not promote,
  reject, overwrite, or delete candidate metadata without explicit review.

## Navigate `ops/`

`ops/` is the lab control plane. It owns repository checks and provider
planning. Model code must not import it.

- `ops/README.md` describes the operations layer.
- `ops/ops.yaml` selects the active capability for each operations interface.
- `ops/control/` validates the operations manifests and ownership boundaries.
- `ops/check/` exposes repository quality checks.
- `ops/runpod/` exposes bounded Runpod Pod planning, launch, and cost capture.
- `ops/research/` validates objective and DAG-program records, then renders
  decision-gate status and ready program nodes.
- Each capability has a `capability.yaml` manifest and a thin `justfile`.

Use these safe discovery and validation commands:

```sh
just ops
just ops validate
just check all
just ops research status
just ops research programs
```

## Run a local experiment

First inspect the experiment and confirm that it is `active` and uses a local
profile. Planning does not train:

```sh
just experiment list
just experiment plan experiments/synthetic-associative-recall-smoke-v1.yaml
just experiment smoke
just experiment report-experiment synthetic-associative-recall-smoke-v1
```

For another reviewed local experiment, use `just experiment run <experiment>`.
The runner writes a new immutable result below `var/runs/`.

## Run a bounded Runpod experiment

Read `docs/runpod.md` before any paid launch. Runpod credentials are in the
local `runpodctl` user configuration, outside this checkout. Do not request,
print, or commit the API key.

`plan` is read-only. `launch` creates a paid Pod, retrieves evidence, records
cost evidence, and terminates the Pod by default. It accepts only an `active`
experiment. The launcher enforces the experiment runtime and projected-cost
caps, but a cost record can remain `pending` until Runpod posts billing data.

Repeat the approved GPU smoke test with these exact commands:

```sh
just check all
just ops validate
just ops runpod doctor
just ops runpod plan experiments/synthetic-associative-recall-fastweight-runpod-smoke-v1.yaml
just ops runpod launch experiments/synthetic-associative-recall-fastweight-runpod-smoke-v1.yaml
just experiment report-experiment synthetic-associative-recall-fastweight-runpod-smoke-v1
```

Run only this active smoke experiment unless the operator explicitly approves a
metadata promotion. Do not launch the proposed comparison experiments. A valid
smoke result has `provider: runpod` and `device: cuda`; report cost as pending
unless Runpod returns a final amount.
