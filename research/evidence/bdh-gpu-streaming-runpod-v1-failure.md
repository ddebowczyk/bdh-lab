# BDH-GPU streaming Runpod v1 provider failure

## Result

The bounded v1 Runpod measurement created Pod `aqgtnre8vdikeq`, ran for about
two and a half minutes, retrieved three immutable failure records, deleted the
Pod, and collected one pending billing record. The provider did not publish a
final amount at collection time.

Each seed failed before training with the same error:

```text
CUDA was requested but is not available
```

The synchronized environment contained PyTorch `2.13.0+cu130`. This is a
provider GPU-visibility failure. It is not evidence about BDH-GPU accuracy,
state persistence, or efficiency.

## Cause and replacement

The v1 profile preferred A5000, A4000, and A4500 GPU types. A live catalogue
check on 2026-08-13 showed those types were not in the available community
set. The replacement v2 profile uses the current RTX 4090, RTX A6000, and RTX
3090 options, requests CUDA 13.0 compatibility, and verifies both template
and synchronized-environment CUDA visibility before training.

## Immutable evidence

- `var/runs/synthetic-associative-recall-bdh-gpu-streaming-runpod-v1/`
- Pod ID: `aqgtnre8vdikeq`
- Cost record: `pending`, not zero.
