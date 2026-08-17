# BDH public baseline conformance v1

## Scope

This record tests the local `BDHPublicStreaming` implementation against the
pinned public Pathway `bdh.py` baseline. It is an execution-conformance result,
not a training-quality or efficiency result.

## Fixed source and setup

- Public file: `resources/github/pathwaycom-bdh-2b0d7a45/bdh.py`.
- SHA-256: `cfe24008f920965cc3c8236feff52c89ca794a31e52324acf9ddb4cd6fd50ac9`.
- Test configuration: two layers, 16 embedding dimensions, two heads, 32
  latent values per head, dropout disabled, vocabulary size 31.
- The public model parameters are loaded unchanged into `BDHPublicStreaming`.
- The test has two fixed seven-token input rows.

The streaming state stores, for each layer and head, the sum of rotated-key
outer products with the layer input values. It also stores the absolute token
position. The public baseline has RoPE but no explicit persistent transition
matrix; retaining keys at their original RoPE positions gives the same causal
attention computation across calls.

## Verification

On 2026-08-13, this command completed successfully:

```sh
uv run pytest tests/test_models.py
```

It reported `7 passed`. The relevant tests are:

- `test_public_bdh_resource_matches_the_recorded_source_hash`
- `test_public_bdh_streaming_batched_logits_match_pinned_public_baseline`
- `test_public_bdh_streaming_step_and_chunks_match_the_batched_state_space_form`

They use `torch.testing.assert_close` with `atol=1e-6` and `rtol=1e-5`. They
check every token logit, every layer state after every step, and a three-token
plus four-token inference split. The state position reaches seven and the
state tensor footprint is unchanged by the split.

## Decision

The public-baseline conformance node passes. The next ready work is a separate
long-stream retention objective with registered length and delay splits. This
result does not show that the model retains useful information after training,
is efficient at scale, or matches any broader author claim.
