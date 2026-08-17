# BDH-GPU streaming local evidence v1

## Scope

This evidence evaluates the public, no-RoPE BDH-GPU state-kernel reference on
checksum-pinned synthetic associative-recall episodes. It tests a fixed
cross-call state boundary. It does not reproduce the authors' full training
system or test language reasoning.

## Fixed setup

- Three seeds: 11, 17, and 23.
- Two hidden splits of 64 episodes each, with no training overlap.
- Each run has 512 training steps, batch size 16, and the same CPU profile.
- The BDH-GPU reference has two shared recurrent blocks, two heads, 64
  particles, identity state transition, and 8,192 bytes of rho per example.

## Results

| Model | Golden accuracy by seed | Golden mean |
| --- | --- | --- |
| BDH-GPU retained rho | 42.2%, 26.6%, 34.4% | 34.4% |
| BDH-GPU reset rho | 9.4%, 15.6%, 14.1% | 13.0% |
| Generic fast-weight control | 95.3%, 100.0%, 100.0% | 98.4% |
| Causal Transformer control | 28.1%, 34.4%, 98.4% | 53.6% |

Retained rho exceeds reset rho in all three seeds. The aggregate golden margin
is 41 / 192 correct episodes, or 21.4 percentage points. The state footprint
does not depend on sequence length. The generic fast-weight control is much
stronger on this simple associative-recall task. The Transformer result has
high seed variance.

## Evidence records

- `var/runs/synthetic-associative-recall-bdh-gpu-streaming-v1/`
- `var/runs/synthetic-associative-recall-fastweight-streaming-v1/`
- `var/runs/synthetic-associative-recall-transformer-streaming-v1/`

All nine records contain source snapshot
`534cedefd3737aa7243f9410ca0cef346325e724fe7f8175675c9b0bfe566f18`.
The earlier one-seed smoke record is preserved but excluded here because it
predates local source-snapshot provenance.

## Decision

The fixed recurrent state mechanism passes its local interface and usefulness
gate: it carries useful task information over a call boundary. The matched
comparison does not show a quality advantage over the generic fast-weight
control. A bounded GPU run is useful only to measure execution, throughput,
latency, state size, and provider billing; it cannot change that quality
comparison claim.
