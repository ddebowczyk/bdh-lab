import torch

from bdh_lab.models import (
    FastWeightRecall,
    hebbian_read,
    hebbian_write,
    make_associative_recall_batch,
)


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
