# Public BDH long-stream retention v1

## Scope

This is the pre-registered local test of the public-BDH-conformant RoPE
state-space adapter. It tests fixed synthetic associative recall. It does not
test language modelling, reasoning, the authors' reported scale, or BDH-CQ.

The query always asks for the first written association. The a5 and a7 cases
therefore increase its fixed delay before the cross-call query. The model trains
only on random-query a3 episodes. All six held-out splits are checksum-pinned
and excluded from generated training examples.

## Protocol

- Assembly: `bdh-public-streaming-recall-v2`.
- Seeds: 11, 17, and 23.
- Training: 1,024 CPU steps; batch size 16; learning rate 0.003.
- State: two layers, two heads, hidden size 16, internal latent size 32 per
  head; 8,192 bytes per example.
- Advance criterion: at least a 0.10 retained-minus-reset golden margin on a5
  and a7 in every seed.

## Results

<!-- markdownlint-disable MD013 -->

| Golden condition | Seed 11 | Seed 17 | Seed 23 | Mean retained | Mean reset | Mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a5 | +22.7 pp | +28.1 pp | +4.7 pp | 32.6% | 14.1% | +18.5 pp |
| a7 | +0.8 pp | +9.4 pp | +7.8 pp | 17.7% | 11.7% | +6.0 pp |

<!-- markdownlint-enable MD013 -->

All completed records report 8,192 state bytes per example. The average
training rate was 27,639 tokens/second and the mean step latency was 6.37 ms.
Those small CPU timings are implementation measurements only; they do not show
a scaled efficiency advantage.

## Decision

The long-stream retention node **fails** its advance criterion. The a7 margin
is below 0.10 in every seed. The a5 margin also misses in seed 23. The result
still shows a useful retained-state signal at shorter and medium delays, but it
does not support the practical long-delay claim required to start BDH-specific
attribution.

The registered next action is one clean, bounded capacity diagnostic with fresh
held-out splits. It changes state capacity only. It will not alter this protocol
or reuse these golden splits for selection.

## Evidence records

- Experiment: `synthetic-associative-recall-public-bdh-long-stream-v2`.
- Immutable results:
  `var/runs/synthetic-associative-recall-public-bdh-long-stream-v2/`.
