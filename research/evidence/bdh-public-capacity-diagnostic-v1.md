# Public BDH capacity diagnostic v1

## Scope

This is the one registered failure-branch diagnostic after the failed public
BDH long-stream curve. It changes state capacity only. It uses fresh,
checksum-pinned held-out splits that are disjoint from the completed curve.

The adapter keeps the public-BDH-conformant RoPE state-space computation, the
fixed first-association query, the optimizer, the 1,024-step budget, and the
a3 training distribution. Hidden size increases from 16 to 32. The recurrent
state therefore increases from 8,192 to 32,768 bytes per example.

## Results

<!-- markdownlint-disable MD013 -->

| Golden condition | Seed 31 | Seed 37 | Seed 41 | Mean retained | Mean reset | Mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a5 | +35.2 pp | +2.3 pp | +23.4 pp | 31.5% | 11.2% | +20.3 pp |
| a7 | +23.4 pp | -4.7 pp | +17.2 pp | 25.5% | 13.5% | +12.0 pp |

<!-- markdownlint-enable MD013 -->

The larger state improves the mean a7 margin relative to the completed curve
(+12.0 rather than +6.0 points). It is not robust: seed 37 has negative a7
margin and a5 margin far below 0.10. The registered all-three-seed criterion
therefore fails.

Average training throughput was 20,800 tokens/second and mean step latency
was 8.46 ms, compared with 27,639 tokens/second and 6.37 ms in the small-state
curve. This small CPU workload shows a local capacity cost, not a scale result.

## Decision

Capacity can help this mechanism in some runs, but capacity alone does not
make public-BDH streaming retention reliable on this controlled long-delay
task. The practical long-stream branch is closed at this point. No
BDH-specific attribution, language, sparse-graph, or paid quality study is
opened from this evidence.

This does not disprove fixed recurrent memory in general. It establishes the
narrower result that the public conformant adapter has not met a robust
long-delay usefulness gate under the registered protocol.

## Evidence records

- Experiment: `synthetic-associative-recall-public-bdh-capacity-diagnostic-v3`.
- Immutable results:
  `var/runs/synthetic-associative-recall-public-bdh-capacity-diagnostic-v3/`.
