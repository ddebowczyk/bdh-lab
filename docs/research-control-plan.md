# Research-control plan

## Goal

Add a small, validated research layer to the lab. It must answer five questions
at any time:

1. What decision is this work intended to support?
2. What claim is under test now?
3. What evidence can change the decision?
4. What is the next decision gate?
5. What is the exact next action?

It must not become a task tracker, a duplicate result store, or a place for
unverified performance claims.

## Design

```text
research program DAG
    | decision route and predecessor outcomes
    v
research objective
    | question, scope, current evidence gate, next action
    v
source assessment ----> component / assembly ----> experiment
       |                                         |
       +------------------- evidence links ------+
                                                  v
                                        immutable run and cost records
```

The existing experiment records stay the source of truth for a specific
hypothesis, data split, metric, budget, and result. A research objective only
gives the higher-level reason for one or more experiments.

A research program is above objectives. It records only the decision graph:
which node follows which required result, the next action, and evidence for a
completed branch. It does not copy experiment settings or automatically change
any status. This keeps branches and joins visible without creating a second
task tracker.

`ops/` owns the commands and validation. The records live in `research/` so
that `ops/` stays a control plane and never becomes model or research content.

## New records

Add `research/objectives/<id>.yaml`. Each record is a small, versioned research
question. It has these fields:

```yaml
schema_version: 1
id: bdh-streaming-state-mechanism-v1
status: active
question: >-
  Does paper-derived recurrent state preserve useful information across calls?
decision: Decide whether to invest in a matched long-context comparison.
scope:
  includes:
    - A paper-derived BDH-GPU streaming reference.
    - State retained across a defined inference boundary.
  excludes:
    - Claims to reproduce private BDH-CQ details.
    - Claims about reasoning, continual learning, or brain equivalence.
gates:
  - id: source-boundary
    status: passed
    exit_criteria: Public equations, public code, and missing details are separated.
    evidence:
      - research/sources/bdh-public-boundary-v1.md
  - id: streaming-reference
    status: current
    exit_criteria: Step and batched execution agree for every output and state update.
    evidence: []
  - id: retained-state-mechanism
    status: pending
    exit_criteria: Retained state and reset state are compared on fixed episodes.
    evidence: []
  - id: matched-comparison
    status: pending
    exit_criteria: Three seeds compare BDH-GPU state, fast-weight control, and Transformer.
    evidence: []
  - id: GPU-cost-quality
    status: pending
    exit_criteria: >-
      A GPU run reports performance, latency, state size, and provider cost
      state.
    evidence: []
next_action:
  kind: implementation
  target: assemblies/bdh-gpu-streaming-reference-v1.yaml
  completion: Add a proposed assembly and local parity tests.
```

Use only these objective statuses: `active`, `waiting`, `concluded`, and
`parked`. A gate has only `pending`, `current`, `passed`, or `failed`. Exactly
one gate is `current` for an active objective. A failed gate is evidence, not a
reason to delete the objective.

Add `research/programs/<id>.yaml` for a top-level decision graph. Every node
has an ID, kind, status, summary, dependency requirements, evidence, and next
action. A requirement names a predecessor node and its required `passed` or
`failed` outcome. The validator rejects cycles, unknown predecessors, missing
evidence for terminal nodes, and current nodes whose requirements are not met.
A program may have more than one ready node after a branch; it does not select
or launch one automatically.

`research/sources/` contains short Markdown source assessments. A source
assessment states what is public, what is inferred, and what remains unknown.
It does not copy a paper or make a model-quality claim.

## Links and validation

Add an optional `research_objective_id` to component, assembly, and experiment
schemas. It must point to an existing objective. This makes the path from a
research decision to immutable run evidence explicit.

The research validator must check:

- every objective follows its schema;
- IDs are unique;
- an active objective has exactly one current gate;
- a passed or failed gate has at least one evidence link;
- every evidence path exists, or every evidence experiment ID is known; and
- each `research_objective_id` points to a known objective.

The validator must not decide that a gate has passed. That remains a reviewed
metadata change.

## Command surface

Add one `ops/research` capability. It owns its manifest, thin Justfile,
research schemas, and research-specific validation code.

```sh
just ops research validate
just ops research status
just ops research plan research/objectives/<id>.yaml
just ops research programs
just ops research program research/programs/<id>.yaml
```

`status` renders a read-only table with objective, current gate, evidence,
next action, and the decision that is due. It derives this view from records;
there is no manually maintained dashboard.

`plan` resolves one objective and its linked candidates, experiments, and known
run/cost summaries. It does not launch work or change status.

`programs` renders every node with derived readiness. `program` resolves one
program with the same read-only readiness calculation. Neither command changes
the program, objective, or candidate lifecycle.

There is intentionally no `start`, `complete`, `promote`, or `close` command.
Those operations change the research record and must occur in a reviewed
commit, as component and experiment promotion already does.

## BDH streaming-state work program

The first objective is `bdh-streaming-state-mechanism-v1`.

### Source boundary

Record the public BDH-GPU equations, the public-code gap, and excluded BDH-CQ
details. The decision evidence is a source assessment that separates fact,
inference, and unknown detail.

### Streaming reference

Add `forward_step(token, state)` and batched `forward(tokens)`. Keep the
implementation proposed. Unit tests must show numerical output and state parity
at each token.

### Retained-state mechanism

Create fixed episodes with demonstrations in one call and a query in another.
Compare retained `rho` to reset `rho`. The decision evidence is held-out
accuracy and a reset-state ablation.

### Matched comparison

Compare BDH-GPU state, the existing `FastWeightRecall`, and causal Transformer.
Sweep demonstration count and delay. The decision evidence is three independent
seeds, fixed held-out splits, and equal training budget.

### GPU cost and quality

Run only after the local gates pass. Record accuracy, state bytes, tokens per
second, latency, and a Runpod cost record.

The proposed experiment hypothesis is:

> A paper-derived BDH-GPU recurrent state preserves demonstrations across calls,
> improves held-out query accuracy over reset state, and has a state footprint
> independent of prior-token count.

This plan does not call the result a reproduction of BDH or BDH-CQ. A passing
mechanism test supports only the stated mechanism claim.

## Delivery sequence

1. Add the objective schema, directory, first objective, and source assessment.
2. Add the `research` operation capability, validator, and read-only commands.
3. Link existing and new candidate metadata to objectives where useful.
4. Add the proposed BDH-GPU streaming-reference component and assembly without
   changing `bdh-fastweight-recall-v1`.
5. Implement the local parity and retained-state tests.
6. Review the first two gates before promotion or any Runpod quality run.

## Constraints and risks

- The public paper is sufficient for a paper-derived mechanism test, not exact
  numerical reproduction of the authors' internal system.
- The current task format supports one sequence. The retained-state gate needs
  an explicit multi-call episode protocol and new held-out datasets.
- State size must be reported with the exact tensor shape and dtype. A claim of
  constant state means constant in token count, not small in model size.
- GPU smoke evidence remains lifecycle evidence only. It does not pass any
  model-quality gate.

## Deliberate omissions

- No tickets, time estimates, owners, daily updates, or automatic gate changes.
- No duplicated metrics or copied run outputs in the research record.
- No automatic promotion, rejection, or external resource launch.
