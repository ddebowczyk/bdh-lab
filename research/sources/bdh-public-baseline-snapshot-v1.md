# BDH public baseline snapshot v1

## Source identity

- Repository: <https://github.com/pathwaycom/bdh>
- Pinned commit: `2b0d7a45b058d4309c84a10e0768d541fe18bdc2`
- Commit date: 2026-05-15
- Commit message: `Update README.md`
- License in repository: MIT
- `bdh.py` SHA-256:
  `cfe24008f920965cc3c8236feff52c89ca794a31e52324acf9ddb4cd6fd50ac9`
- `train.py` SHA-256:
  `25059c567c99feb989a36174ad984ccaab6e3cb6bb72385667b3775207431534`

The source was shallow-cloned from the public repository at this exact commit
on 2026-08-13. Future conformance work must use this commit or record a new
snapshot version.

## Public baseline behavior

`bdh.py` defines a six-layer character-level model with embedding dimension
256, four heads, dropout 0.1, and 128 times internal latent expansion. The
same encoder and decoder tensors are reused at every layer.

Each layer projects activations to a ReLU-positive latent vector. Attention
applies RoPE to the shared query/key tensor, uses strict causal scores, then
multiplies those scores by the value tensor. A second positive projection is
multiplied with the first, decoded, normalized, and added as a residual.

`train.py` trains the default configuration on byte-level Tiny Shakespeare with
512-token blocks, batch size 32, AdamW, and 3,000 iterations. It is a small
demonstration training script, not the paper's reported scaling setup.

## Conformance boundary

The public baseline implements full-sequence attention. It does not expose a
persistent `rho` inference API or a recurrent state transition across calls.
The lab's streaming implementation is therefore an algebraic state-space
extension of the public causal computation, not a direct execution path from
the repository.

The next node must first match this public RoPE batched computation with
identical weights and tokens. It must then prove that its chunked state-space
path has identical logits and recurrent state. Only then may a long-stream
experiment call the model a public-baseline-conformant streaming variant.

## Excluded claims

This snapshot does not reproduce the authors' reported 10M-to-1B parameter
training, translation experiments, scaling laws, graph analysis, or private
BDH-CQ system. It is a reproducible source boundary for the next local gate.
