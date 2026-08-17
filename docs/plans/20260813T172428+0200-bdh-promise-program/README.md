# BDH promise research program

## Purpose

Determine whether the public BDH-GPU direction is promising as a practical
long-stream model. The target claim is narrow: a trained model can retain
useful information beyond its training context with fixed recurrent-state
memory, while preserving useful quality and runtime trade-offs.

This plan does not test brain equivalence, general reasoning, or the private
BDH-CQ system.

## Starting evidence

The completed `bdh-streaming-state-mechanism-v1` objective established that the
lab's paper-derived, no-RoPE reference has a causal per-layer `rho` state.
Step-wise and batched paths agree; retaining `rho` over an inference boundary
helps fixed associative recall; and its state bytes do not grow with token
count.

The same evidence also limits the claim. The reference used identity `U`, not
RoPE, and it has not been compared directly with the current public Pathway
baseline. On three-association recall, generic fast weights substantially
outperformed the BDH reference. The test did not vary context length, delay,
capacity, model scale, or language data.

## Research question and decision

Question: does a public BDH-GPU implementation offer a useful quality versus
fixed-memory trade-off on streams that are longer than those seen in training?

Decision: continue toward language-scale experiments only if public-baseline
conformance and long-stream evaluation show that the fixed-state architecture
has a real retention or efficiency benefit. Otherwise retain the reference as
a mechanism baseline and stop before large paid runs.

## Constraints

- Preserve all completed experiments and run records. New work uses new,
  versioned records.
- Pin public source revisions before using them as a conformance oracle.
- Use validation data for development and reserve the golden split for each
  declared decision.
- Run local conformance and small training gates before any Runpod launch.
- Each paid experiment declares a time and cost cap and records its own cost
  state.
- BDH-CQ is an independent, speculative program. Its undisclosed dimensions,
  update rules, and training method prohibit a reproduction claim.

## Program graph

```text
P0 public-source snapshot
 |
 v
P1 public baseline conformance (RoPE-keyed recurrent state)
 | pass
 v
P2 long-stream retention and memory curve
 |-------------------------------|
 | pass                          | fail
 v                               v
P3 BDH attribution/scaling       P2R diagnose interference or close direction
 | pass
 +------------------------------+
 |                              |
 v                              v
P4 language quality/efficiency   P5 sparse and graph analysis
 | pass                          |
 v                               |
P6 larger quality/cost study <---+

Q1 BDH-CQ latent-workspace exploration (independent and parked)
```

## Nodes

### P0: Public-source snapshot

Pin the exact `pathwaycom/bdh` revision, its license, and the relevant
`bdh.py` / training configuration. Record what the public code implements and
what the paper leaves as a choice.

Evidence: a short source assessment and a reproducible source reference.

### P1: Public baseline conformance

Implement a RoPE-keyed recurrent state in the streaming reference.
With identical seeded parameters and tokens, assert that the local batched path
matches the pinned public baseline at every token. Then assert that chunked
`forward_step` produces the same logits and final state as the batched path,
including a cross-call boundary.

This answers whether the lab is testing the public BDH-GPU baseline rather
than merely a related fast-weight model.

### P2: Long-stream retention and memory curve

Train on bounded synthetic streams, then evaluate held-out streams across
increasing association count and delay. Measure retained versus reset state,
interference, state bytes, Transformer KV-cache bytes, throughput, and warmed
decode latency.

Compare the RoPE BDH reference to identity-`U` BDH, a parameter-matched causal
Transformer, and a matched ordinary linear-attention control. Retain the
existing generic fast-weight model as a positive-control upper bound, not as a
claim of architectural equivalence.

The direction is promising only if BDH preserves useful quality beyond the
training horizon and its fixed state produces a material practical trade-off.

### P2R: Diagnose or close

If P2 fails, use a small ablation only to determine whether the cause is
capacity, positional state, or cross-talk. Close the practical-long-stream
claim if no controlled variant changes the result. Do not spend on language
training merely because a mechanism test passes.

### P3: BDH attribution and scaling

Only after P2 passes, vary the paper-specific choices while holding the task
and training-token budget fixed: particle count `n`, low-rank dimension `d`,
shared versus unshared blocks, positive ReLU versus dense activation, and
identity versus RoPE/damped `U`.

The result must attribute any retention benefit to a BDH-specific choice rather
than to parameter count or a generic linear-attention state.

### P4: Language quality and efficiency

Train small matched models on a public language corpus with document-aware
state reset. Evaluate perplexity at the training length and at longer lengths.
Measure fixed recurrent state, Transformer cache memory, decode latency, and
tokens per second on the same hardware.

This is the first test of a practical language-model claim. It is not a test of
general reasoning.

### P5: Sparse and graph analysis

Only analyze sparsity, modularity, heavy-tailed structure, and concept-level
activation examples once a model has passed the quality gate. Use matched
random and trained baselines. These measurements may support an
interpretability claim; they cannot establish biological plausibility.

### P6: Larger quality and cost study

Run only after P4 and P5 identify a stable candidate. Pre-register parameter
budget, data amount, training length, evaluation lengths, seeds, hardware,
and cost cap. This is the earliest stage that can examine the paper's scaling
and performance claims.

### Q1: BDH-CQ exploration

Keep this separate and parked. A recurrent memory plus iterative latent
workspace can be evaluated as an original research design, with no claim to
reproduce BDH-CQ.

## Execution policy

Each program node should point to one or more versioned research objectives and
experiments. The program controls dependency and routing only; objectives own
the question and evidence gates, while experiments own hypotheses, splits,
budgets, and immutable results. A node becomes ready only when all required
predecessors pass. A failure follows its named diagnostic or closure branch.

This preserves a short decision structure instead of duplicating experiment
details in a second tracker.
