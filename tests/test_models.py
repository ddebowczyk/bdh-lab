import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

from bdh_lab.models import (
    BDHGPUStreamingRecall,
    BDHPublicStreaming,
    BDHPublicStreamingConfig,
    BDHPublicStreamingRecall,
    FastWeightRecall,
    hebbian_read,
    hebbian_write,
    make_associative_recall_batch,
    split_recall_episode,
)

PINNED_PUBLIC_BDH = (
    Path(__file__).parents[1] / "resources/github/pathwaycom-bdh-2b0d7a45/bdh.py"
)
PINNED_PUBLIC_BDH_SHA256 = "cfe24008f920965cc3c8236feff52c89ca794a31e52324acf9ddb4cd6fd50ac9"


def _pinned_public_module() -> object:
    """Load the immutable public baseline without adding it to the package API."""

    module_name = "pinned_pathway_bdh"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, PINNED_PUBLIC_BDH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bdh_gpu_reference() -> BDHGPUStreamingRecall:
    torch.manual_seed(13)
    return BDHGPUStreamingRecall(
        symbols=8,
        hidden_size=16,
        particles=64,
        heads=2,
        layers=2,
        dropout=0.0,
    ).eval()


def _public_bdh_streaming_reference() -> tuple[object, BDHPublicStreaming]:
    public = _pinned_public_module()
    config = public.BDHConfig(
        n_layer=2,
        n_embd=16,
        dropout=0.0,
        n_head=2,
        mlp_internal_dim_multiplier=4,
        vocab_size=31,
    )
    torch.manual_seed(31)
    baseline = public.BDH(config).eval()
    candidate = BDHPublicStreaming(
        BDHPublicStreamingConfig(
            n_layer=config.n_layer,
            n_embd=config.n_embd,
            dropout=config.dropout,
            n_head=config.n_head,
            mlp_internal_dim_multiplier=config.mlp_internal_dim_multiplier,
            vocab_size=config.vocab_size,
        )
    ).eval()
    candidate.load_state_dict(baseline.state_dict())
    return baseline, candidate


def test_hebbian_state_recovers_an_orthogonal_written_value() -> None:
    state = torch.zeros((1, 2, 2))
    key = torch.tensor([[1.0, 0.0]])
    value = torch.tensor([[0.0, 1.0]])

    stored = hebbian_write(state, key, value)

    assert torch.allclose(hebbian_read(stored, key), value)


def test_fast_weight_candidate_returns_one_value_logit_per_symbol() -> None:
    batch = make_associative_recall_batch(
        symbols=4,
        associations_per_sequence=2,
        batch_size=3,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )

    logits = FastWeightRecall(symbols=4, hidden_size=8)(batch.tokens)

    assert logits.shape == (3, 4)


def test_fixed_query_index_always_selects_the_first_written_association() -> None:
    batch = make_associative_recall_batch(
        symbols=8,
        associations_per_sequence=3,
        batch_size=4,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
        query_association_index=0,
    )

    assert torch.equal(batch.tokens[:, -1], batch.tokens[:, 1])
    assert torch.equal(batch.targets, batch.tokens[:, 2] - 11)


def test_bdh_gpu_step_and_batched_forms_match_every_logit_and_state() -> None:
    batch = make_associative_recall_batch(
        symbols=8,
        associations_per_sequence=3,
        batch_size=2,
        generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
    )
    model = _bdh_gpu_reference()

    with torch.no_grad():
        batched = model.forward_batched(batch.tokens, return_state_trace=True)
        state = None
        for position in range(batch.tokens.size(1)):
            logits, state = model.forward_step(batch.tokens[:, position], state)
            assert torch.allclose(logits, batched.logits[:, position], atol=1e-6, rtol=1e-5)
            assert state is not None
            assert batched.state_trace is not None
            for layer, trace in zip(state.layers, batched.state_trace, strict=True):
                assert torch.allclose(layer, trace[:, :, position], atol=1e-6, rtol=1e-5)


def test_bdh_gpu_is_causal_and_retains_state_across_an_inference_boundary() -> None:
    batch = make_associative_recall_batch(
        symbols=8,
        associations_per_sequence=3,
        batch_size=2,
        generator=torch.Generator().manual_seed(23),
        device=torch.device("cpu"),
    )
    changed = batch.tokens.clone()
    changed[:, -1] = 2
    model = _bdh_gpu_reference()

    with torch.no_grad():
        original = model.forward_batched(batch.tokens, return_state_trace=True)
        later_changed = model.forward_batched(changed, return_state_trace=True)
        demonstrations = batch.tokens[:, :-2]
        query = batch.tokens[:, -2:]
        _, retained_state = model.forward_with_state(demonstrations)
        retained_logits, retained_end = model.forward_with_state(query, retained_state)
        reset_logits, _ = model.forward_with_state(query)

    assert torch.allclose(original.logits[:, :-1], later_changed.logits[:, :-1])
    assert original.state_trace is not None
    assert later_changed.state_trace is not None
    for before, after in zip(original.state_trace, later_changed.state_trace, strict=True):
        assert torch.allclose(before[:, :, :-1], after[:, :, :-1])
    assert torch.allclose(retained_logits, original.logits[:, -1], atol=1e-6, rtol=1e-5)
    for retained, full in zip(retained_end.layers, original.state.layers, strict=True):
        assert torch.allclose(retained, full, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(retained_logits, reset_logits)
    assert model.forward_batched(batch.tokens[:, :3]).state.nbytes == original.state.nbytes


def test_public_bdh_resource_matches_the_recorded_source_hash() -> None:
    """Protect the public conformance baseline from accidental local changes."""

    assert hashlib.sha256(PINNED_PUBLIC_BDH.read_bytes()).hexdigest() == PINNED_PUBLIC_BDH_SHA256


def test_public_bdh_streaming_batched_logits_match_pinned_public_baseline() -> None:
    """Prove equal full-sequence logits with identical public parameters and tokens."""

    baseline, candidate = _public_bdh_streaming_reference()
    tokens = torch.tensor(
        [[1, 5, 9, 3, 2, 7, 6], [4, 8, 0, 12, 18, 30, 11]], dtype=torch.long
    )

    with torch.no_grad():
        expected, _ = baseline(tokens)
        actual = candidate.forward_batched(tokens)

    torch.testing.assert_close(actual.logits, expected, atol=1e-6, rtol=1e-5)


def test_public_bdh_streaming_step_and_chunks_match_the_batched_state_space_form() -> None:
    """Prove state and logits agree across every one-token and chunk boundary."""

    _, candidate = _public_bdh_streaming_reference()
    tokens = torch.tensor(
        [[1, 5, 9, 3, 2, 7, 6], [4, 8, 0, 12, 18, 30, 11]], dtype=torch.long
    )

    with torch.no_grad():
        full = candidate.forward_batched(tokens, return_state_trace=True)
        state = None
        for position in range(tokens.size(1)):
            logits, state = candidate.forward_step(tokens[:, position], state)
            torch.testing.assert_close(logits, full.logits[:, position], atol=1e-6, rtol=1e-5)
            assert full.state_trace is not None
            for layer, trace in zip(state.layers, full.state_trace, strict=True):
                torch.testing.assert_close(layer, trace[:, :, position], atol=1e-6, rtol=1e-5)

        first = candidate.forward_batched(tokens[:, :3])
        second = candidate.forward_batched(tokens[:, 3:], first.state)

    torch.testing.assert_close(
        torch.cat((first.logits, second.logits), dim=1), full.logits, atol=1e-6, rtol=1e-5
    )
    for chunked, whole in zip(second.state.layers, full.state.layers, strict=True):
        torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=1e-5)
    assert second.state.position == tokens.size(1)
    assert first.state.nbytes == second.state.nbytes == full.state.nbytes


def test_public_bdh_recall_adapter_preserves_state_across_the_episode_boundary() -> None:
    batch = make_associative_recall_batch(
        symbols=8,
        associations_per_sequence=3,
        batch_size=2,
        generator=torch.Generator().manual_seed(29),
        device=torch.device("cpu"),
        query_association_index=0,
    )
    model = BDHPublicStreamingRecall(
        symbols=8,
        hidden_size=16,
        heads=2,
        layers=2,
        mlp_internal_dim_multiplier=4,
        dropout=0.0,
    ).eval()

    with torch.no_grad():
        full_logits, full_state = model.forward_with_state(batch.tokens)
        demonstrations, query = split_recall_episode(batch.tokens)
        _, demonstration_state = model.forward_with_state(demonstrations)
        retained_logits, retained_state = model.forward_with_state(query, demonstration_state)
        reset_logits, _ = model.forward_with_state(query)

    torch.testing.assert_close(retained_logits, full_logits, atol=1e-6, rtol=1e-5)
    for retained, full in zip(retained_state.layers, full_state.layers, strict=True):
        torch.testing.assert_close(retained, full, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(retained_logits, reset_logits)
    assert model.state_nbytes(batch_size=2) == full_state.nbytes
