# Experiment-system design

## Purpose

The lab needs to compare many architecture choices without losing lineage or
turning automated search into an unreviewed source of claims. The control plane
therefore uses data records, not filenames or code comments, as the source of
truth.

## Records

```text
component version -> assembly version -> experiment version -> run record
        ^                  ^                    ^                  |
        |                  |                    |                  v
  design decision    exact composition     task, budget, splits  evidence

dataset version --------------------------^    ^
checksum-pinned held-out samples                |
                                                  v
                                   training rejects held-out examples
```

Research objectives sit above this chain. They state one decision, scope, and
current verification gate; source assessments and immutable run evidence link
back to that objective. This is deliberately not a task tracker: an active
objective has one current gate, while a concluded objective keeps its decision,
evidence, and the condition for a separately scoped follow-up.

A research program is a separate, versioned dependency graph above objectives.
It records only node routing, required pass/fail outcomes, progress, and links
to evidence. `ops/research` derives which pending nodes are ready; it never
starts a node or changes a status. This allows branches and merges without
duplicating experiment settings or becoming a second task tracker.

A component describes one replaceable choice, such as a memory update rule,
gate, positional method, or optimizer. An assembly binds specific component
versions to a model implementation. An experiment binds the assembly to a
task and execution profile. A run writes a resolved copy of those inputs and
its measured outputs.

## Data protocol

Every associative-recall experiment names two active dataset records: one
validation split and one golden-evaluation split. Each record contains a
deterministic generator definition, sample count, and SHA-256 checksum of its
canonical samples. The runner materializes the samples, verifies the checksum,
and rejects the run if the splits overlap.

The online training generator receives both held-out example ID sets and
rejects a generated training sample that matches either one. Run provenance
stores both dataset IDs and checksums. Completed results store separate
`validation_query_accuracy` and `golden_evaluation_query_accuracy` metrics.

Validation supports development choices. The golden metric is a final-report
metric: do not use it to choose architectures, hyperparameters, or stop
criteria. The software prevents sample overlap and generator drift; it cannot
prevent a person from tuning to a visible golden score.

Every experiment declares a structured hypothesis: its claim, expected effect,
and falsifier. A descendant proposal can name its parent experiment IDs in
`lineage`; this makes an automated search traceable without allowing the
automation to promote itself.

Each declared seed creates one independent immutable result record. A failed
seed also writes a record with its failure, so a report cannot silently hide
partial execution. A provider billing record is separate from the training
result because Runpod can publish final billed usage after training ends. It
has three explicit states: `reported`, `pending`, and `unavailable`.

## Candidate lifecycle

`proposed` means the component has a stated hypothesis but no promotion
decision. `active` means it can appear in an active assembly. `rejected` means
new active assemblies must not use it. `superseded` records a replacement, and
`retired` is preserved only for historical replay.

Validation forbids an active assembly from referencing a non-active component
and forbids an active experiment from using a non-active assembly. Historical
records stay valid; no past run is edited.

## Proposal sequence

An automated search must create new versioned records; it must never rewrite a
candidate that already has evidence:

1. add a `proposed` component version with its parent IDs and falsifier;
2. add a `proposed` assembly that names exact component versions;
3. add a `proposed` experiment with a hypothesis, budget, and parent experiment
   IDs when it repeats or extends prior work;
4. review and promote the required component and assembly versions to `active`;
5. run the exact active experiment and inspect its result and cost records; and
6. record a reviewed `rejected`, `superseded`, or `retired` decision instead of
   deleting the losing candidate.

This sequence permits recursive proposal and execution while keeping semantic
promotion and rejection under review.

## Automation boundary

An automated experimenter may:

- materialize a proposal from a template;
- validate metadata and cross-file references;
- render a provider plan;
- run an explicitly named experiment; and
- write immutable result and provider-cost records.

It may not promote, reject, overwrite, or delete candidate metadata. That
operation changes the hypothesis catalogue and needs an explicit reviewed
change. This keeps recursive improvement measurable: each next proposal names
its parent evidence and its claimed improvement.
