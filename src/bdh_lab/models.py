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


def split_recall_episode(tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Split a fixed-format recall episode into demonstration and query calls."""

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape batch x sequence")
    read_positions = (tokens == READ_TOKEN).nonzero(as_tuple=False)
    if read_positions.size(0) != tokens.size(0):
        raise ValueError("each recall episode must contain exactly one read token")
    positions = read_positions[:, 1]
    if not torch.equal(positions, positions[0].expand_as(positions)):
        raise ValueError("each batch must use one shared inference boundary")
    boundary = int(positions[0])
    if boundary < 1 or boundary >= tokens.size(1) - 1:
        raise ValueError("recall inference boundary must leave demonstrations and a query")
    return tokens[:, :boundary], tokens[:, boundary:]


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
    query_association_index: int | None = None,
) -> RecallBatch:
    """Create a fresh batch, rejecting examples reserved by a held-out split."""

    if associations_per_sequence > symbols:
        raise ValueError("associations_per_sequence cannot exceed symbols")
    if query_association_index is not None and not 0 <= query_association_index < (
        associations_per_sequence
    ):
        raise ValueError("query_association_index must name one written association")

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
            query_index = (
                query_association_index
                if query_association_index is not None
                else int(torch.randint(associations_per_sequence, (1,), generator=generator))
            )
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


@dataclass(frozen=True)
class BDHGPUState:
    """The fixed causal linear-attention state for every shared BDH-GPU block."""

    layers: tuple[Tensor, ...]

    @property
    def nbytes(self) -> int:
        """Return the actual tensor footprint, including batch dimension and dtype."""

        return sum(layer.numel() * layer.element_size() for layer in self.layers)


@dataclass(frozen=True)
class BDHGPUSequenceOutput:
    """Token logits, final recurrent state, and optional state after each token."""

    logits: Tensor
    state: BDHGPUState
    state_trace: tuple[Tensor, ...] | None = None


class BDHGPUStreamingRecall(nn.Module):
    """Paper-derived Appendix E BDH-GPU reference with an explicit rho state.

    This matches the public Appendix E tensor layout in evaluation mode: shared
    low-rank ReLU blocks, strict-causal linear attention, residual updates, and
    parameter-free LayerNorm. `forward_batched` evaluates the causal matrix form;
    `forward_step` evaluates the algebraically equivalent state-space form.
    """

    def __init__(
        self,
        *,
        symbols: int,
        hidden_size: int,
        particles: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if particles % heads != 0:
            raise ValueError("particles must divide evenly across heads")
        if layers < 1:
            raise ValueError("layers must be positive")
        self.symbols = symbols
        self.hidden_size = hidden_size
        self.particles = particles
        self.heads = heads
        self.layers = layers
        self.particles_per_head = particles // heads
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, bias=False)
        self.embedding = nn.Embedding(vocabulary_size(symbols), hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Parameter(torch.empty((particles, hidden_size)))
        self.decoder_x = nn.Parameter(torch.empty((heads, hidden_size, self.particles_per_head)))
        self.decoder_y = nn.Parameter(torch.empty((heads, hidden_size, self.particles_per_head)))
        self.readout = nn.Parameter(torch.empty((hidden_size, symbols)))
        for parameter in (self.encoder, self.decoder_x, self.decoder_y, self.readout):
            nn.init.normal_(parameter, std=0.02)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> BDHGPUState:
        """Create zero rho tensors with shape layer x batch x head x key x value."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        state_dtype = dtype or self.embedding.weight.dtype
        layer = torch.zeros(
            (
                batch_size,
                self.heads,
                self.particles_per_head,
                self.hidden_size,
            ),
            device=device,
            dtype=state_dtype,
        )
        return BDHGPUState(layers=tuple(layer.clone() for _ in range(self.layers)))

    def state_nbytes(self, *, batch_size: int = 1) -> int:
        """Return persistent rho bytes for a given batch, independent of token count."""

        return self.initial_state(batch_size=batch_size, device=self.embedding.weight.device).nbytes

    def _validated_state(self, tokens: Tensor, state: BDHGPUState | None) -> BDHGPUState:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape batch x sequence")
        if state is None:
            return self.initial_state(batch_size=tokens.size(0), device=tokens.device)
        if len(state.layers) != self.layers:
            raise ValueError("state layer count does not match the model")
        expected = (
            tokens.size(0),
            self.heads,
            self.particles_per_head,
            self.hidden_size,
        )
        for layer in state.layers:
            if tuple(layer.shape) != expected:
                raise ValueError(f"state shape {tuple(layer.shape)} does not match {expected}")
            if layer.device != tokens.device:
                raise ValueError("state and tokens must use the same device")
        return state

    def forward_batched(
        self,
        tokens: Tensor,
        state: BDHGPUState | None = None,
        *,
        return_state_trace: bool = False,
    ) -> BDHGPUSequenceOutput:
        """Evaluate the public strict-causal tensor form and return its final rho state."""

        state = self._validated_state(tokens, state)
        batch_size, sequence_length = tokens.shape
        if sequence_length < 1:
            raise ValueError("tokens must contain at least one position")
        value = self.norm(self.embedding(tokens))
        final_layers: list[Tensor] = []
        traces: list[Tensor] = []

        for layer_index in range(self.layers):
            prior = state.layers[layer_index]
            key = torch.einsum("btd,hdn->bhtn", value, self.decoder_x).relu()
            scores = torch.einsum("bhtn,bhsn->bhts", key, key)
            scores = scores.tril(diagonal=-1)
            state_read = torch.einsum("bhtn,bhnd->bhtd", key, prior)
            causal_read = torch.einsum("bhts,bsd->bhtd", scores, value)
            attention = state_read + causal_read
            update = torch.einsum("bhtn,btd->bhtnd", key, value)
            trace = prior.unsqueeze(2) + update.cumsum(dim=2)
            final_layers.append(trace[:, :, -1])
            if return_state_trace:
                traces.append(trace)

            decoded = torch.einsum("bhtd,hdn->bhtn", self.norm(attention), self.decoder_y)
            activity = decoded.relu() * key
            particles = activity.permute(0, 2, 1, 3).reshape(
                batch_size, sequence_length, self.particles
            )
            value = self.norm(value + self.norm(self.dropout(particles) @ self.encoder))

        return BDHGPUSequenceOutput(
            logits=value @ self.readout,
            state=BDHGPUState(layers=tuple(final_layers)),
            state_trace=tuple(traces) if return_state_trace else None,
        )

    def forward_step(
        self, token: Tensor, state: BDHGPUState | None = None
    ) -> tuple[Tensor, BDHGPUState]:
        """Evaluate one token with the fixed rho state-space kernel."""

        if token.ndim != 1:
            raise ValueError("token must have shape batch")
        batch_tokens = token.unsqueeze(1)
        state = self._validated_state(batch_tokens, state)
        value = self.norm(self.embedding(token))
        final_layers: list[Tensor] = []

        for layer_index in range(self.layers):
            prior = state.layers[layer_index]
            key = torch.einsum("bd,hdn->bhn", value, self.decoder_x).relu()
            attention = torch.einsum("bhn,bhnd->bhd", key, prior)
            updated = prior + torch.einsum("bhn,bd->bhnd", key, value)
            final_layers.append(updated)

            decoded = torch.einsum("bhd,hdn->bhn", self.norm(attention), self.decoder_y)
            activity = decoded.relu() * key
            particles = activity.reshape(token.size(0), self.particles)
            value = self.norm(value + self.norm(self.dropout(particles) @ self.encoder))

        return value @ self.readout, BDHGPUState(layers=tuple(final_layers))

    def forward_with_state(
        self, tokens: Tensor, state: BDHGPUState | None = None
    ) -> tuple[Tensor, BDHGPUState]:
        """Read a sequence through rho and return only its final recall logits."""

        output = self.forward_batched(tokens, state)
        return output.logits[:, -1], output.state

    def forward(self, tokens: Tensor) -> Tensor:
        """Provide the standard classifier interface used by the existing trainer."""

        return self.forward_batched(tokens).logits[:, -1]


@dataclass(frozen=True)
class BDHPublicStreamingConfig:
    """Configuration with the field names and tensor shapes of public ``bdh.py``."""

    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


@dataclass(frozen=True)
class BDHPublicStreamingState:
    """Persistent rotated-key/value sums for each shared public-BDH layer."""

    layers: tuple[Tensor, ...]
    position: int = 0

    @property
    def nbytes(self) -> int:
        """Return the tensor footprint, independent of the processed token count."""

        return sum(layer.numel() * layer.element_size() for layer in self.layers)


@dataclass(frozen=True)
class BDHPublicStreamingOutput:
    """Full token logits and the persistent state after a public-BDH sequence."""

    logits: Tensor
    state: BDHPublicStreamingState
    state_trace: tuple[Tensor, ...] | None = None


class _BDHPublicAttention(nn.Module):
    """The RoPE functions and frequency buffer from the pinned public baseline."""

    def __init__(self, *, latent_size: int) -> None:
        super().__init__()
        positions = torch.arange(latent_size, dtype=torch.float32)
        quantized = (positions / 2).floor() * 2
        frequencies = 1.0 / ((2**16) ** (quantized / latent_size)) / (2 * torch.pi)
        self.register_buffer("freqs", frequencies.view(1, 1, 1, latent_size))

    @staticmethod
    def rope(phases: Tensor, values: Tensor) -> Tensor:
        """Apply the pinned public RoPE function to an even-sized latent axis."""

        rotated = torch.stack((-values[..., 1::2], values[..., ::2]), dim=-1).view(
            *values.size()
        )
        wrapped = (phases % 1) * (2 * torch.pi)
        return (values * torch.cos(wrapped)).to(values.dtype) + (
            rotated * torch.sin(wrapped)
        ).to(values.dtype)


class BDHPublicStreaming(nn.Module):
    """State-space execution of the pinned public Pathway BDH causal computation.

    The parameter names and tensor layouts deliberately match
    ``resources/github/pathwaycom-bdh-2b0d7a45/bdh.py``. Each state layer is
    ``sum(rotated_key outer value)`` for the processed prefix. The state stores
    absolute RoPE positions in the keys, so it is algebraically equivalent to
    the public strict-causal attention, including across call boundaries.
    """

    def __init__(self, config: BDHPublicStreamingConfig) -> None:
        super().__init__()
        if config.n_layer < 1:
            raise ValueError("n_layer must be positive")
        if config.n_embd < 1 or config.n_head < 1:
            raise ValueError("n_embd and n_head must be positive")
        if config.n_embd * config.mlp_internal_dim_multiplier % config.n_head != 0:
            raise ValueError("the public latent size must divide evenly across heads")
        self.config = config
        self.latent_size = config.n_embd * config.mlp_internal_dim_multiplier // config.n_head
        if self.latent_size % 2:
            raise ValueError("the public RoPE latent size must be even")

        self.decoder = nn.Parameter(
            torch.zeros((config.n_head * self.latent_size, config.n_embd)).normal_(std=0.02)
        )
        self.encoder = nn.Parameter(
            torch.zeros((config.n_head, config.n_embd, self.latent_size)).normal_(std=0.02)
        )
        self.attn = _BDHPublicAttention(latent_size=self.latent_size)
        self.ln = nn.LayerNorm(config.n_embd, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(
            torch.zeros((config.n_head, config.n_embd, self.latent_size)).normal_(std=0.02)
        )
        self.lm_head = nn.Parameter(
            torch.zeros((config.n_embd, config.vocab_size)).normal_(std=0.02)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> BDHPublicStreamingState:
        """Return zero recurrent state for a batch at absolute position zero."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        layer = torch.zeros(
            (batch_size, self.config.n_head, self.latent_size, self.config.n_embd),
            device=device,
            dtype=dtype or self.embed.weight.dtype,
        )
        return BDHPublicStreamingState(
            layers=tuple(layer.clone() for _ in range(self.config.n_layer))
        )

    def state_nbytes(self, *, batch_size: int = 1) -> int:
        """Return persistent state bytes; this does not depend on stream length."""

        return self.initial_state(batch_size=batch_size, device=self.embed.weight.device).nbytes

    def _validated_state(
        self, tokens: Tensor, state: BDHPublicStreamingState | None
    ) -> BDHPublicStreamingState:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape batch x sequence")
        if state is None:
            return self.initial_state(batch_size=tokens.size(0), device=tokens.device)
        if state.position < 0:
            raise ValueError("state position must not be negative")
        if len(state.layers) != self.config.n_layer:
            raise ValueError("state layer count does not match the model")
        expected = (tokens.size(0), self.config.n_head, self.latent_size, self.config.n_embd)
        for layer in state.layers:
            if tuple(layer.shape) != expected:
                raise ValueError(f"state shape {tuple(layer.shape)} does not match {expected}")
            if layer.device != tokens.device:
                raise ValueError("state and tokens must use the same device")
            if layer.dtype != self.embed.weight.dtype:
                raise ValueError("state and model parameters must use the same dtype")
        return state

    def _phases(self, *, start: int, count: int) -> Tensor:
        positions = torch.arange(
            start,
            start + count,
            device=self.attn.freqs.device,
            dtype=self.attn.freqs.dtype,
        ).view(1, 1, -1, 1)
        return positions * self.attn.freqs

    def forward_batched(
        self,
        tokens: Tensor,
        state: BDHPublicStreamingState | None = None,
        *,
        return_state_trace: bool = False,
    ) -> BDHPublicStreamingOutput:
        """Evaluate a full causal sequence and extend its persistent state."""

        state = self._validated_state(tokens, state)
        batch_size, sequence_length = tokens.shape
        if sequence_length < 1:
            raise ValueError("tokens must contain at least one position")
        phases = self._phases(start=state.position, count=sequence_length)
        value = self.ln(self.embed(tokens).unsqueeze(1))
        final_layers: list[Tensor] = []
        traces: list[Tensor] = []

        for layer_index in range(self.config.n_layer):
            prior = state.layers[layer_index]
            x_sparse = (value @ self.encoder).relu()
            rotated = self.attn.rope(phases, x_sparse)
            scores = (rotated @ rotated.mT).tril(diagonal=-1)
            prior_read = rotated @ prior
            current_read = scores @ value
            updates = torch.einsum("bhtn,btd->bhtnd", rotated, value.squeeze(1))
            trace = prior.unsqueeze(2) + updates.cumsum(dim=2)
            final_layers.append(trace[:, :, -1])
            if return_state_trace:
                traces.append(trace)

            y_kv = self.ln(prior_read + current_read)
            y_sparse = (y_kv @ self.encoder_v).relu()
            xy_sparse = self.drop(x_sparse * y_sparse)
            y_mlp = xy_sparse.transpose(1, 2).reshape(
                batch_size, 1, sequence_length, self.config.n_head * self.latent_size
            ) @ self.decoder
            value = self.ln(value + self.ln(y_mlp))

        logits = value.view(batch_size, sequence_length, self.config.n_embd) @ self.lm_head
        return BDHPublicStreamingOutput(
            logits=logits,
            state=BDHPublicStreamingState(
                layers=tuple(final_layers), position=state.position + sequence_length
            ),
            state_trace=tuple(traces) if return_state_trace else None,
        )

    def forward_step(
        self,
        token: Tensor,
        state: BDHPublicStreamingState | None = None,
    ) -> tuple[Tensor, BDHPublicStreamingState]:
        """Evaluate one token with the recurrent form of public causal attention."""

        if token.ndim != 1:
            raise ValueError("token must have shape batch")
        state = self._validated_state(token.unsqueeze(1), state)
        phases = self._phases(start=state.position, count=1)
        value = self.ln(self.embed(token))
        final_layers: list[Tensor] = []

        for layer_index in range(self.config.n_layer):
            prior = state.layers[layer_index]
            x_sparse = torch.einsum("bd,hdn->bhn", value, self.encoder).relu()
            rotated = self.attn.rope(phases, x_sparse.unsqueeze(2)).squeeze(2)
            y_kv = self.ln(torch.einsum("bhn,bhnd->bhd", rotated, prior))
            updated = prior + torch.einsum("bhn,bd->bhnd", rotated, value)
            final_layers.append(updated)

            y_sparse = torch.einsum("bhd,hdn->bhn", y_kv, self.encoder_v).relu()
            xy_sparse = self.drop(x_sparse * y_sparse)
            y_mlp = xy_sparse.reshape(token.size(0), 1, 1, -1) @ self.decoder
            value = self.ln(value + self.ln(y_mlp).squeeze(1).squeeze(1))

        return value @ self.lm_head, BDHPublicStreamingState(
            layers=tuple(final_layers), position=state.position + 1
        )

    def forward_with_state(
        self,
        tokens: Tensor,
        state: BDHPublicStreamingState | None = None,
    ) -> tuple[Tensor, BDHPublicStreamingState]:
        """Return the final-token logits and state for one sequence call."""

        output = self.forward_batched(tokens, state)
        return output.logits[:, -1], output.state

    def forward(self, tokens: Tensor) -> Tensor:
        """Match the public baseline's full-token logits interface."""

        return self.forward_batched(tokens).logits


class BDHPublicStreamingRecall(nn.Module):
    """Recall-task adapter around the public-conformant streaming core.

    The core retains the public vocabulary-level output head. This adapter
    selects only the value-token rows, so it can train with the lab's existing
    symbol-class recall targets without altering the conformance model.
    """

    def __init__(
        self,
        *,
        symbols: int,
        hidden_size: int,
        heads: int,
        layers: int,
        mlp_internal_dim_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.symbols = symbols
        self.core = BDHPublicStreaming(
            BDHPublicStreamingConfig(
                n_layer=layers,
                n_embd=hidden_size,
                dropout=dropout,
                n_head=heads,
                mlp_internal_dim_multiplier=mlp_internal_dim_multiplier,
                vocab_size=vocabulary_size(symbols),
            )
        )

    def _value_logits(self, logits: Tensor) -> Tensor:
        return logits[..., value_offset(self.symbols) : vocabulary_size(self.symbols)]

    def state_nbytes(self, *, batch_size: int = 1) -> int:
        """Return the core's persistent state bytes for this batch size."""

        return self.core.state_nbytes(batch_size=batch_size)

    def forward_with_state(
        self,
        tokens: Tensor,
        state: BDHPublicStreamingState | None = None,
    ) -> tuple[Tensor, BDHPublicStreamingState]:
        """Return value-class logits and public-conformant state after one call."""

        logits, updated = self.core.forward_with_state(tokens, state)
        return self._value_logits(logits), updated

    def forward(self, tokens: Tensor) -> Tensor:
        """Return the final value-class logits required by the recall trainer."""

        return self._value_logits(self.core.forward_batched(tokens).logits[:, -1])


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
    if kind == "bdh_gpu_streaming_recall_v1":
        return BDHGPUStreamingRecall(
            symbols=symbols,
            hidden_size=int(config["hidden_size"]),
            particles=int(config["particles"]),
            heads=int(config["heads"]),
            layers=int(config["layers"]),
            dropout=float(config["dropout"]),
        )
    if kind == "bdh_public_streaming_recall_v1":
        return BDHPublicStreamingRecall(
            symbols=symbols,
            hidden_size=int(config["hidden_size"]),
            heads=int(config["heads"]),
            layers=int(config["layers"]),
            mlp_internal_dim_multiplier=int(config["mlp_internal_dim_multiplier"]),
            dropout=float(config["dropout"]),
        )
    raise ValueError(f"unregistered assembly implementation {kind}")
