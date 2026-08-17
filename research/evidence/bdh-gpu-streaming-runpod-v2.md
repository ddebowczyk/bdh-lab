# BDH-GPU streaming Runpod evidence v2

## Scope

This is a bounded CUDA execution measurement of the public, no-RoPE BDH-GPU
reference. It repeats the local BDH-GPU configuration on a Runpod Pod. It is
not a new architecture-quality comparison and it does not reproduce the full
BDH training system.

## Execution record

- Profile: `runpod-pod-v2`, which requested a current-catalogue GPU type and
  required CUDA visibility before and after environment synchronization.
- Pod: `afbdc5atd2q2jx`; it was deleted after retrieval.
- Provider and device: `runpod` and `cuda` in all three completed result
  records.
- Seeds: 11, 17, and 23; 512 steps per seed; fixed validation and golden
  splits.
- Source snapshot: `0a42de06bdd1ba34b18d1b7929a7778e1a1be18d3f2730f2cf9f7b02b2c54321`.

## Results

| Measure | Seed 11 | Seed 17 | Seed 23 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Golden retained-rho accuracy | 35.9% | 28.1% | 35.9% | 33.3% |
| Golden reset-rho accuracy | 17.2% | 14.1% | 15.6% | 15.6% |
| Training tokens/second | 22,032 | 23,825 | 23,243 | 23,033 |
| Mean step latency | 7.99 ms | 7.39 ms | 7.57 ms | 7.65 ms |

The rho state is 8,192 bytes per example for every run. The retained-state
margin is 17.7 percentage points on the golden split. This repeats the
mechanism result on CUDA; it does not overturn the local matched comparison,
where the generic fast-weight control was much stronger on this task.

The small model is slower here than the local CPU measurement. This is expected
for a workload that is too small to amortize GPU launch and transfer overhead;
it is not an efficiency conclusion for a scaled model.

## Cost and limits

The experiment declared a 30-minute runtime limit and a USD 0.20 projected
cost cap. The Pod collection window was 2026-08-13 15:00:54 UTC to 15:03:41
UTC. Runpod had not returned final billing at collection time, so the
Pod-scoped cost record is **pending**. No cost estimate is reported as final.

## Evidence records

- `var/runs/synthetic-associative-recall-bdh-gpu-streaming-runpod-v2/`
- `var/runs/synthetic-associative-recall-bdh-gpu-streaming-runpod-v2/costs/cost-20260813t150342221314z-pod-afbdc5atd2q2jx.yaml`

## Decision

The CUDA execution and billing-state gate passes. The conclusion remains
narrow: this reference is a verified fixed-state streaming mechanism baseline.
Any test of long-delay quality, context scaling, or positional encoding needs a
new objective and a new versioned experiment.
