# BDH public boundary

## Purpose

This assessment defines what this lab can test from public BDH material. It is
not a claim to reproduce the authors' internal training system or results.

## Public primary sources

- [The Dragon Hatchling paper](https://arxiv.org/html/2509.26507v1), Definition
  4, Equation 8, and Appendix E.
- [The BDH-CQ paper](https://arxiv.org/html/2608.09888v1), Sections 3.2 and
  3.3.

## Public and usable

The original paper describes a per-layer state `rho` with shape `n x d`. At a
new token, each layer reads the prior state, computes ReLU-low-rank values, and
writes an outer product into the state. Appendix E gives a PyTorch tensor
listing with shared low-rank blocks and causal linear attention. The listing
uses a fixed context window, but the paper says the same linear attention has a
state-space form.

The lab reference uses the Appendix E tensor layout in evaluation mode. Its
streaming kernel replaces the causal token-pair matrix with a per-head
`key x value` state. This is algebraically the causal linear-attention state
for the listed `Q = K = x` operation when positional rotation is disabled.

## Public but not reproduced

The paper permits several choices for LayerNorm, residuals, heads, and the
state transition matrix `U`. The appendix leaves RoPE to the reader. This lab
uses parameter-free LayerNorm, residuals in the Appendix E order, one or more
shared recurrent blocks, and identity `U`. These are explicit reference choices
for a mechanism test, not verified author settings.

The paper reports broader training and scaling results. This lab cannot verify
them from the public equations or the small mechanism task.

## Not public enough to reproduce

BDH-CQ describes a recurrent contextual memory `S` and a separate repeated
latent workspace `H`. It explicitly leaves dimensions, exact update rules, and
the complete training recipe proprietary. Therefore no BDH-CQ result in this
lab can be called a faithful reproduction.

## Decision use

The next evidence gate is not model quality. It is numerical parity between
step-wise state execution and batched causal execution, plus a retained-state
versus reset-state test on fixed episodes.
