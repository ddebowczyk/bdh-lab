# BDH deep-evidence research program v2

## Purpose

Determine whether the public BDH-GPU direction has distinct practical or
scientific value beyond a Transformer and ordinary linear attention.

This program succeeds `bdh-promise-v1`. It does not reopen or alter that
program. Version 1 established public-baseline conformance and found that one
small a3-trained associative-recall model did not robustly extrapolate to a5
and a7. It did not complete the matched-control or language branches described
in its source plan. That result closes the registered synthetic protocol, not
the broader BDH question.

The new program evaluates five possible sources of value:

1. language quality and sample efficiency;
2. fixed-state continuity across coherent chunks;
3. state and cache memory at increasing stream length;
4. BDH-specific effects beyond generic linear attention; and
5. sparse-state and graph interpretability after a quality gate passes.

It does not test biological equivalence, general intelligence, or the private
BDH-CQ implementation.

## Research question

Does a public-BDH-conformant recurrent model occupy a useful quality, memory,
latency, or interpretability frontier when it is trained on coherent language
streams and compared with matched controls?

## Evidence inherited from version 1

- The pinned public model is a byte-level causal language model with shared
  recurrent depth, a positive expanded latent space, RoPE-keyed causal
  attention, and dense quadratic reference execution.
- The lab's recurrent RoPE implementation matches the pinned public batched
  computation and preserves equivalent state across chunk boundaries.
- Retained state helped mean associative-recall accuracy at some delays, but
  the effect was unstable across seeds and capacity alone did not fix it.
- The earlier task trained only on a3 episodes, evaluated a5 and a7, used a
  final-query objective, and did not include the planned matched Transformer
  and ordinary-linear-attention comparisons.

These facts justify a corrected program. They do not count as language-quality
or practical-efficiency evidence.

## Program graph

```text
H0 inherited v1 evidence (passed)
 |
 +--> L1 public Shakespeare calibration (current)
 |     | pass
 |     +--> L2 matched Shakespeare comparison
 |     +--> K1 recurrent-kernel efficiency
 |
 +--> T1 state capacity and stability analysis
       | pass
       +--> M1 repaired memory benchmark

L2 pass + K1 pass
        |
        v
E1 coherent Europarl pilot, 120M tokens
        | pass
        v
A1 BDH architecture attribution
        | pass
        v
E2 full Europarl study, up to 1.2B tokens
        | pass                    | fail
        +--> I1 interpretability  +--> D2 negative decision
        +--> D1 practical-value decision

Each principal experiment has a named failure-review branch. A failure does
not silently promote a modified model or erase the completed evidence.
```

## Work packages

### H0: Inherited evidence boundary

Treat the v1 source snapshot, conformance proof, retention curve, and capacity
diagnostic as immutable inputs. Do not rerun or rewrite those experiments.

### L1: Public Shakespeare calibration

Freeze the public Tiny Shakespeare byte corpus, checksum, train/validation
split, source revision, model configuration, optimizer, and evaluation
procedure. Preserve the upstream source unchanged and add instrumentation in
new lab records.

Record validation bits per byte, training loss, non-finite values, throughput,
step latency, peak accelerator memory, checkpoint identity, and samples. A text
sample is diagnostic only.

The goal is pipeline calibration. This node does not claim long memory,
reasoning, or superiority.

### T1: State capacity and stability analysis

Analyze the recurrent state before another recall-quality claim. Measure:

- state norm and numerical growth with stream length;
- effective rank and occupied capacity;
- key collision and cross-talk;
- overwrite and correction behavior;
- temporal decay with RoPE, identity, and damping transitions; and
- reset and detach semantics across training boundaries.

Use untrained, trained, randomized, and generic-linear-attention controls where
the comparison is meaningful.

### M1: Repaired memory benchmark

Replace the exhausted a3-to-a7 protocol with a versioned task that varies
association count, delay, distractor count, and overwrite events independently.
Train across a declared range and report both in-distribution and extrapolation
results. Use a next-token objective over the sequence in addition to exact
query accuracy.

Compare public-conformant BDH, an identity-transition BDH ablation, a
parameter-matched Transformer, ordinary linear attention, and the generic
fast-weight positive control. Match training tokens and report a separate
compute-matched view. Use document-level held-out generation seeds and at least
three model seeds for a promotion decision.

This branch diagnoses the memory mechanism. Its failure does not, by itself,
forbid a language experiment.

### L2: Matched Shakespeare comparison

Use the same raw bytes, split, context lengths, training-token checkpoints, and
evaluation code for all models. Compare:

- the pinned public dense BDH path;
- the conformant recurrent BDH path;
- a causal nanoGPT-style Transformer; and
- an ordinary linear-attention control.

Produce two views:

1. parameter-matched, to test parameter efficiency; and
2. compute- or wall-time-matched, to test practical training efficiency.

Tiny Shakespeare is a calibration corpus. A passing result opens coherent
streaming work but does not verify the paper's scaling claim.

### K1: Recurrent-kernel efficiency

Benchmark the dense public path and algebraically conformant recurrent path on
identical weights and tokens. Sweep chunk and stream length. Record numerical
error, state bytes, peak memory, training throughput, prefill latency, warmed
single-token latency, and kernel utilization.

Fixed state bytes are not sufficient. The state must also remain numerically
stable and useful as the stream grows.

### E1: Coherent Europarl pilot

Reconstruct a small version of the paper-relevant task using raw UTF-8 bytes,
document-coherent source/translation streams, explicit source and target
language markers, 2,048-token chunks, and truncated backpropagation through
time.

Carry BDH state between related chunks. Carry an explicitly bounded
TransformerXL cache for the Transformer control. Reset both only at declared
document boundaries. Split by complete documents, not random chunks.

Use approximately 120M presented tokens and one seed first. Measure throughput
and final projected cost before approving a full run. All paid records require
hard runtime and projected-cost caps.

### A1: BDH architecture attribution

Hold data and training-token budget fixed while testing one architectural
choice at a time:

- shared versus unshared recurrent depth;
- positive ReLU versus a matched dense or signed activation;
- latent expansion and recurrent-state capacity;
- RoPE versus identity and damped transitions;
- state normalization, selective forgetting, and conditional gating; and
- BDH state versus ordinary linear-attention state.

An effect counts as BDH-specific only when a generic linear-attention control
does not explain it.

### E2: Full Europarl study

Run up to 1.2B presented tokens only for candidates that pass calibration,
kernel, coherent-stream, and attribution gates. Use at least three seeds for
the final candidate and matched Transformer control. Freeze model sizes,
training data, token budgets, evaluation documents, hardware class, and cost
caps before the first golden run.

This stage tests the small-scale form of the paper's language and translation
claim. It does not reproduce a 10M-to-1B-parameter scaling curve unless a
separate, explicitly funded program is approved.

### I1: Interpretability analysis

Only after a candidate meets the quality gate, compare trained, randomized,
and matched-control models on activation sparsity, state sparsity, effective
graph degree, modularity, heavy-tailed structure, and concept selectivity.

Use quantitative tests and correction for multiple comparisons. Selected
examples can illustrate a measured effect but cannot establish
monosemanticity or biological plausibility by themselves.

### D1 and D2: Decisions

The positive branch decides whether BDH merits larger public-corpus scaling,
kernel engineering, or a separate private document-stream program. The
negative branch records which value axes failed and preserves any mechanism or
tooling that remains useful.

## Program-level measurements

Every comparative language experiment reports:

- validation bits per byte and its confidence interval;
- loss versus presented tokens and wall time;
- parameter count and estimated training FLOPs;
- state or KV-cache bytes per stream;
- peak accelerator memory;
- training tokens per second;
- prefill and warmed decode latency;
- retained-versus-reset quality;
- performance at the training length and longer lengths; and
- provider runtime and final or pending cost evidence.

## Initial promotion rule

A candidate remains promising when it establishes at least one material
advantage without an unacceptable loss on the other axes. Before golden runs,
each objective must freeze exact thresholds. The initial program targets are:

- language quality within 5% relative bits per byte of the best fair control;
- at least 10% fewer presented tokens to reach a matched quality target; or
- no more than 25% of the matched Transformer's cache bytes at four times the
  training context, with no more than 5% relative quality degradation.

Runtime is always reported. A final candidate that is more than twice as slow
in warmed decoding must show a compensating quality or memory advantage and
must receive an explicit decision review.

These are program targets, not retroactive interpretations. Node objectives
may make them stricter but must not relax them after golden evidence is seen.

## Data and evaluation rules

- Validation data supports development. Golden data supports one declared
  decision.
- Natural-language splits use complete documents or plays, not random chunks
  from the same document on both sides.
- Training streams preserve temporal coherence and record all reset and detach
  boundaries.
- Model and data seeds are independent and recorded.
- Parameter matching and compute matching are separate reported comparisons.
- One bounded diagnostic may follow a failure. It changes one declared factor
  and uses fresh held-out data.

## Cost policy

Run local conformance and tiny-data checks first. Then use this sequence:

1. a short GPU throughput and peak-memory pilot;
2. one 120M-token Europarl seed;
3. a reviewed cost projection based on measured throughput; and
4. the 1.2B-token, three-seed study only after explicit approval.

Each paid experiment owns a versioned runtime cap, projected USD cap, Pod ID,
and billing record. Provider prices are refreshed before launch.

## Out of scope

- claims of brain equivalence or biological plausibility;
- reproduction claims for private BDH-CQ dimensions or training methods;
- autonomous promotion of model candidates;
- a production knowledge system over a private document corpus; and
- a full parameter scaling curve above the approved small-model study.

That private corpus may later support a separate document-coherence program. It
must retain provenance and permissions and must compare against a
retrieval-grounded system.

## Immediate next action

Complete the current `bdh-public-shakespeare-calibration-v1` protocol-freeze
gate. Create new dataset, assembly, and experiment records. Do not modify or
reuse the completed v1 experiment records.
