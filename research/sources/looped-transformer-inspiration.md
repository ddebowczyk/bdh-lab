# Looped transformer inspiration: RLT and MELT

## Source identity

- Project page: <https://yifanzhang-pro.github.io/recurrent-looped-tranformer/>
- RLT report: <https://yifanzhang-pro.github.io/recurrent-looped-tranformer/Recurrent_Looped_Transformer.pdf>
- Associated project repository: <https://github.com/yifanzhang-pro/recurrent-looped-tranformer>
- MELT paper: <https://arxiv.org/pdf/2605.07721>
- MELT record: <https://arxiv.org/abs/2605.07721>
- Capture date: 2026-09-13

The project page and report describe Yifan Zhang's *Recurrent Looped
Transformer* (RLT), a September 2026 technical report. The arXiv source is
Victor Conchello Vendrell et al.'s *Memory-Efficient Looped Transformer:
Decoupling Compute from Memory in Looped Language Models* (arXiv version 2,
19 May 2026).

The GitHub URL is retained as an associated source link only. Per the research
request, this note does not inspect or compare the repository implementation.
No claim below treats either source as evidence that the BDH Lab implements
these designs.

## Why these sources matter

Both sources move some reasoning work into recurrent internal computation, but
they address different bottlenecks:

- RLT defines one complete state transition across prompt and response tokens,
  then specifies the execution, training, and policy-replay consequences.
- MELT changes how a looped Transformer's key-value state is retained so that
  reasoning depth does not multiply KV-cache memory, then reports a
  pretrained-model adaptation recipe and benchmark results.

The shared inspiration is not a particular layer layout. It is the insistence
that recurrent state, cache lifetime, update order, training gradients, serving
boundaries, and replay metadata form one explicit contract.

## RLT: source claims

### Complete prompt-to-response state

RLT uses a causal encoder and a recurrent decoder. The encoder builds
prefix-restricted key-value memory. The decoder carries a recurrent output and
layerwise sliding-window attention (SWA) KV cache through every token. Its
complete decoder state is described as:

```text
H_t = (s_t, C_t^D)
```

The state is initialized once. The prompt/response boundary is a serving
boundary, not a model reset. The same transition consumes observed prompt
tokens and sampled response tokens. The decoder cache keeps only a bounded
window, while encoder-derived memory grows with the processed prefix.

The reference tied configuration uses 48 encoder and 48 decoder layers, with
compatible attention and FFN weights shared across the two stages. This is
parameter reuse with different attention wiring, not copying encoder
activations into the decoder. The report uses the example to explain an
extensible temporal path: after `t` tokens, the recurrent path has traversed
`t * L_D` decoder blocks while each token still executes a fixed number of
logical blocks.

### Execution and memory boundary

RLT separates three quantities that should not be conflated:

1. the fixed number of blocks evaluated for each token;
2. the sequential recurrent path, whose structural depth grows with history;
3. the memory stores, including growing encoder memory and bounded decoder SWA
   history.

The report explicitly says that encoder parallelism does not make prompt
prefill fully parallel: every prompt token still receives a decoder update.
Independent sequences can contribute ready recurrent updates to a batch, but
the nonlinear decoder recurrence is not assumed to admit an exact parallel
scan. The report also presents batching, weight reuse, fusion, and
checkpointing as implementation targets, not measured speedups.

Therefore RLT is not an end-to-end constant-memory design. Its decoder SWA
cache is bounded by the window, but encoder-derived memory remains a function
of sequence length. Parameter tying reduces stored weights without removing
the second logical pass or either cache role.

### Training and policy replay

The proposed training contract includes full BPTT through recurrent outputs,
decoder KV, encoder memory, and the encoder-side continuation state.
Activation checkpointing may trade recomputation for storage, but detaching
state changes the computation. For SFT, prompt and context tokens still update
the differentiable state even when the loss is masked to assistant targets.

For RL, RLT proposes:

- sample with a behavior policy and retain the actual behavior log-probability
  and sampling metadata;
- rebuild the full prompt and response history under current parameters;
- replay the complete state transition before scoring each sampled action;
- retain the original behavior probabilities as the ratio denominator; and
- rebuild current-policy state after every parameter update.

The report warns that top-k or top-p sampling changes the support of the
distribution and that correct behavior log-probabilities do not by themselves
make an arbitrary off-policy objective unbiased. An exact prefix snapshot must
include encoder continuation state, encoder-derived memory, recurrent output,
every decoder SWA cache, token and position metadata, window semantics, model
version, and any required sampler state.

### Evidence boundary

RLT is a design report. It develops equations, execution schedules, cache
semantics, and propositions about causality and cached-state exactness. It says
that it does not report measured efficiency or scaling results. The report's
hardware and RL benefits are hypotheses and engineering targets, not reproduced
results in this lab.

## MELT: source claims and reported evidence

### Depth-independent loop cache

MELT starts from a looped language model in which the same Transformer stack is
applied for multiple reasoning iterations. Standard looped execution appends
KV entries at each loop, so the source describes cache growth as proportional
to sequence length times reasoning depth.

MELT instead maintains a latent state per layer and updates the current token's
state with an element-wise gate:

```text
z_t = sigmoid(x_t W_z + h_(t-1) U_z + b_z)
h_t = z_t * h_(t-1) + (1 - z_t) * x_t
```

Keys and values are projected from the updated latent state. A token adds one
row to the layer cache, and later reasoning loops update that row rather than
adding another row. The stated asymptotic target is therefore `O(N * L)` KV
memory rather than `O(N * L * T)` for `N` layers, sequence length `L`, and
reasoning depth `T`.

This shifts the burden from explicit loop-indexed cache storage to learned
gating. Information may be preserved, blended, or overwritten by the gate; a
constant cache is not a guarantee of constant information retention.

### Two-phase training recipe

MELT adapts pretrained Ouro parameters instead of training the changed cache
architecture from scratch. The source reports three linked mechanisms:

1. **Chunk-wise training.** Chunks are processed sequentially, while work
   inside a chunk is parallel. Smaller chunks better match autoregressive KV
   dependencies; larger chunks improve throughput but increase the training
   and inference mismatch.
2. **Interpolated transition.** The standard LoopLM KV and MELT KV are both
   computed during the transition. A coefficient `alpha` increases from zero
   to one, moving from the original cache behavior to MELT dynamics.
3. **Attention-aligned distillation.** After the transition, a frozen LoopLM
   teacher supplies distillation targets, including post-attention alignment at
   every layer and reasoning loop.

The reported MELT-1.6B setup used 24 layers, hidden size 2048, four recurrent
steps, 500-token chunks, 320K tokens per batch, and 10K-token training
sequences. Phase 1 used 160M tokens and phase 2 used 96M tokens. The main run
used 1,040 GPU-hours on eight H100 80 GB GPUs; the paper reports approximately
20,000 GPU-hours for the overall project including preliminary work.

### Reported benchmark and memory results

The paper reports MELT-1.6B results against Ouro-1.4B-Thinking and several
non-looped models. Selected source-reported MELT scores include:

- AIME24 pass@1: 46.7; AIME25 pass@1: 33.3; AIME26 pass@1: 41.0.
- AMC23 pass@1: 80.2; MATH-500 accuracy: 93.4; OlympiadBench accuracy:
  64.7.
- GPQA accuracy: 42.6; HLE accuracy: 2.0; MMLU-Red accuracy: 74.2;
  HumanEval accuracy: 81.7.

For a 32K-token generation, the paper reports the following derived memory
comparison from vLLM KV metrics and a 2-byte-per-parameter model-memory
estimate:

- MELT-1.6B: 6.29 GB KV, 3.272 GB model, 9.49 GB total.
- Ouro-1.4B-Thinking: 25.17 GB KV, 2.869 GB model, 27.97 GB total.
- Qwen3-1.7B: 3.67 GB KV, 3.442 GB model, 7.07 GB total.

The source characterizes MELT as approximately four times lower in KV memory
and 2.95 times lower in total derived memory than Ouro for this setup. These
are source-reported measurements for specific models, tools, prompts, and
limits; they are not independent measurements by BDH Lab.

The ablations support the training recipe rather than isolating only the cache
formula. The paper reports that removing attention-aligned distillation,
interpolation, all-loop distillation, and finally chunk-wise training causes
progressive degradation. The no-chunk-wise-training variant reports zero on
the four listed first-stage benchmarks. The authors also report that simpler
mean, EMA, final-loop, and scalar-gated alternatives underperform the proposed
element-wise gate after phase 1.

### Limitations and reproducibility caveats

The paper states that the number of recurrent loops is fixed at inference time,
that GQA was not explored, and that sequential KV updates constrain training
parallelism. It presents adaptive loop depth and GQA as future work.

The paper also reports that the authors could not fully reproduce the reported
Ouro results from the released artifacts and therefore rely on their own
measurements for the comparison. That is an important source caveat, not a
finding reproduced by this lab.

## What is useful as inspiration for BDH Lab

This source set suggests several hypotheses and contract fields, but it does
not create a new objective or candidate in the lab.

### State and boundary contracts

- Define the complete recurrent state, not only the visible output tensor.
- State whether prompt tokens and response tokens share one transition.
- Make reset, append, eviction, and external-input semantics explicit.
- Record which state parts are differentiable during training and which are
  reconstructed during replay.

### Memory and compute accounting

- Separate parameter memory, context-derived memory, recurrent state, local
  cache, activation storage, and temporary workspace.
- Report memory as a function of both token length and reasoning depth.
- Do not call a design constant-memory if another store still grows with
  context length.
- Measure sequential dependency and throughput separately from arithmetic
  FLOPs and structural depth.

### Training and evidence design

- Test sequential reference execution against any chunked or batched schedule.
- Treat chunk size as an explicit fidelity-versus-throughput factor.
- Preserve state-update order and cache provenance in immutable run records.
- If policy replay is ever studied, record behavior sampling transforms,
  current-policy rebuild rules, cache validity, and support assumptions.
- Keep source-reported results, local reproductions, and proposed hypotheses in
  separate evidence categories.

## Decision boundary for this lab

The sources justify a future controlled study of recurrent-state contracts,
loop-depth versus memory scaling, and training-schedule fidelity. They do not
justify calling RLT or MELT a BDH reproduction, changing an existing BDH
objective, or promoting an architecture.

No GitHub implementation comparison was performed. No source-reported
benchmark, memory number, or GPU-hour figure above has been independently
reproduced in this checkout.
