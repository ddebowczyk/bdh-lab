"""Small, matched mechanism-test models for associative recall."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

WRITE_TOKEN = 0
READ_TOKEN = 1
PAD_TOKEN = 2
KEY_OFFSET = 3


def vocabulary_size(symbols: int) -> int:
    """Return the vocabulary for distinct write, read, key, and value tokens."""

    return KEY_OFFSET + (2 * symbols)


def value_offset(symbols: int) -> int:
    """Return the first value token ID for a task vocabulary."""

    return KEY_OFFSET + symbols


def is_key(tokens: Tensor, symbols: int) -> Tensor:
    """Return a boolean mask for key tokens."""

    return (tokens >= KEY_OFFSET) & (tokens < value_offset(symbols))


def is_value(tokens: Tensor, symbols: int) -> Tensor:
    """Return a boolean mask for value tokens."""

    return (tokens >= value_offset(symbols)) & (tokens < vocabulary_size(symbols))


@dataclass(frozen=True)
class RecallBatch:
    """A batch of key-value writes followed by one key query."""

    tokens: Tensor
    targets: Tensor


def recall_example_id(tokens: Sequence[int] | Tensor, target: int) -> str:
    """Return a stable ID for one complete recall example and its answer."""

    if isinstance(tokens, Tensor):
        values = [int(value) for value in tokens.detach().cpu().tolist()]
    else:
        values = [int(value) for value in tokens]
    encoded = json.dumps(
        {"tokens": values, "target": int(target)}, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_associative_recall_batch(
    *,
    symbols: int,
    associations_per_sequence: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    excluded_example_ids: Collection[str] = (),
) -> RecallBatch:
    """Create a fresh batch, rejecting examples reserved by a held-out split."""

    if associations_per_sequence > symbols:
        raise ValueError("associations_per_sequence cannot exceed symbols")

    sequence_length = (associations_per_sequence * 3) + 2
    tokens = torch.full((batch_size, sequence_length), PAD_TOKEN, dtype=torch.long)
    targets = torch.empty(batch_size, dtype=torch.long)
    excluded = set(excluded_example_ids)
    for row in range(batch_size):
        for _attempt in range(10_000):
            keys = torch.randperm(symbols, generator=generator)[:associations_per_sequence]
            values = torch.randperm(symbols, generator=generator)[:associations_per_sequence]
            candidate = torch.full((sequence_length,), PAD_TOKEN, dtype=torch.long)
            position = 0
            for key, value in zip(keys.tolist(), values.tolist(), strict=True):
                candidate[position] = WRITE_TOKEN
                candidate[position + 1] = KEY_OFFSET + key
                candidate[position + 2] = value_offset(symbols) + value
                position += 3
            query_index = int(torch.randint(associations_per_sequence, (1,), generator=generator))
            candidate[position] = READ_TOKEN
            candidate[position + 1] = KEY_OFFSET + int(keys[query_index])
            target = int(values[query_index])
            if recall_example_id(candidate, target) not in excluded:
                tokens[row] = candidate
                targets[row] = target
                break
        else:
            raise RuntimeError("could not sample outside the held-out recall splits")
    return RecallBatch(tokens=tokens.to(device), targets=targets.to(device))


def hebbian_write(state: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Add normalized key-value outer products to an associative state matrix."""

    normalized_key = functional.normalize(key, dim=-1)
    normalized_value = functional.normalize(value, dim=-1)
    return state + torch.einsum("bi,bj->bij", normalized_key, normalized_value)


def hebbian_read(state: Tensor, query: Tensor) -> Tensor:
    """Read a value-like vector from state with a normalized query."""

    normalized_query = functional.normalize(query, dim=-1)
    return torch.bmm(normalized_query.unsqueeze(1), state).squeeze(1)


class TransformerRecall(nn.Module):
    """A compact causal-Transformer control with the same task vocabulary."""

    def __init__(
        self,
        *,
        symbols: int,
        hidden_size: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size(symbols), hidden_size)
        self.position = nn.Embedding(512, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(hidden_size, symbols)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.size(1)
        positions = torch.arange(sequence_length, device=tokens.device)
        encoded = self.embedding(tokens) + self.position(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones((sequence_length, sequence_length), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.output(self.encoder(encoded, mask=causal_mask)[:, -1])


class FastWeightRecall(nn.Module):
    """A writable, fixed-size associative-memory candidate for the mechanism test."""

    def __init__(self, *, symbols: int, hidden_size: int) -> None:
        super().__init__()
        self.symbols = symbols
        self.embedding = nn.Embedding(vocabulary_size(symbols), hidden_size)
        self.key_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.value_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.query_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.output = nn.Linear(hidden_size, symbols)

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, sequence_length = tokens.shape
        hidden_size = self.embedding.embedding_dim
        state = torch.zeros((batch_size, hidden_size, hidden_size), device=tokens.device)
        pending_key = torch.zeros((batch_size, hidden_size), device=tokens.device)

        for position in range(sequence_length):
            token_ids = tokens[:, position]
            vector = self.embedding(token_ids)
            key_mask = is_key(token_ids, self.symbols).unsqueeze(-1)
            value_mask = is_value(token_ids, self.symbols).unsqueeze(-1)
            key = self.key_projection(vector)
            value = self.value_projection(vector)
            pending_key = torch.where(key_mask, key, pending_key)
            normalized_key = functional.normalize(pending_key, dim=-1)
            normalized_value = functional.normalize(value, dim=-1)
            write = torch.einsum("bi,bj->bij", normalized_key, normalized_value)
            state = state + (value_mask.unsqueeze(-1) * write)

        query = self.query_projection(self.embedding(tokens[:, -1]))
        return self.output(hebbian_read(state, query))


def build_model(assembly: dict[str, object], *, symbols: int) -> nn.Module:
    """Build only an explicitly registered model implementation."""

    implementation = assembly["implementation"]
    if not isinstance(implementation, dict):
        raise ValueError("assembly implementation must be a mapping")
    kind = implementation["kind"]
    config = implementation["config"]
    if not isinstance(config, dict):
        raise ValueError("assembly implementation config must be a mapping")
    if kind == "transformer_recall_v1":
        return TransformerRecall(
            symbols=symbols,
            hidden_size=int(config["hidden_size"]),
            heads=int(config["heads"]),
            layers=int(config["layers"]),
            dropout=float(config["dropout"]),
        )
    if kind == "fast_weight_recall_v1":
        return FastWeightRecall(symbols=symbols, hidden_size=int(config["hidden_size"]))
    raise ValueError(f"unregistered assembly implementation {kind}")
