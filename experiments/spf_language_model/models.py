"""Learned causal byte LM with a canonical phase-mixer Cartesian reference.

Execution is REFERENCE_CARTESIAN, not native SPF arithmetic. The differentiable
carrier is a real/imaginary pair and never claims the packed or clocked storage
contract. Each layer uses the canonical recurrence (C is NOT conjugated)::

    advanced = A[t] * state
    y[t] = sum_n C[t,n] * advanced[n] + D * V[t]
    state = advanced + B[t,:,None] * V[t,None,:]

The learned real controller generates token-local phase addresses. A fixed
sparse integer character bank selects transition/write/read phases, omega is
fixed to one, and learned decay and B/C log gains are static across tokens.
Thus the feature program is representable by PhaseFeatureNetwork; embedding,
address generation, projections, normalization and the byte head remain float.
No bidirectional scan, attention, oracle continuation, or Mamba claim is made.

Training cost is O(B*T*N*P), streaming carrier O(B*N*P), P=d_model. Autograd
retains the trajectory. Canonical packed native accumulation has per-update
rounding and small-write loss; the current clocked SPFBlockV3 executor retains
records O(T) and reads O(T**2). Neither is reproduced by this float accumulator.
Native export rounds phases to 16-bit ticks, gains to 1/256 log2 codes, and
decay to Q16 log2 units. Replay differences are observations, not learning gates.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

EXECUTION_KIND = "REFERENCE_CARTESIAN"


def multiply_cartesian_reference(a: Tensor, b: Tensor) -> Tensor:
    """Pair arithmetic only for this explicitly labelled reference path."""
    return torch.stack(
        (
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ),
        -1,
    )


def polar_cartesian_reference(magnitude: Tensor, angle: Tensor) -> Tensor:
    return torch.stack((magnitude * angle.cos(), magnitude * angle.sin()), -1)


def numpy_complex_reference(pair: Tensor) -> np.ndarray:
    array = pair.detach().cpu().numpy()
    return array[..., 0] + 1j * array[..., 1]


class CartesianPhaseSequenceMixerReference(nn.Module):
    """Learned ingress around a fixed canonical sparse-character program."""

    execution_kind = EXECUTION_KIND

    def __init__(self, d_model: int, state_size: int):
        super().__init__()
        self.d_model, self.state_size = d_model, state_size
        self.n_address_channels = max(2, state_size)
        self.norm = nn.LayerNorm(d_model)
        self.address_projection = nn.Linear(d_model, self.n_address_channels)
        self.value_projection = nn.Linear(d_model, 2 * d_model)
        self.output_projection = nn.Linear(2 * d_model, d_model)
        # Two sparse characters per row, integer coefficients fixed throughout
        # learning. No dense learned phase feature operator replaces the program.
        characters = torch.zeros(3 * state_size, self.n_address_channels)
        for row in range(3 * state_size):
            characters[row, row % self.n_address_channels] += 1
            characters[row, (row * 3 + 1) % self.n_address_channels] += (
                1 if row // state_size == 1 else -1
            )
        self.register_buffer("characters", characters)
        self.register_buffer("transition_frequencies", torch.ones(state_size))
        half_lives = torch.logspace(math.log10(2), math.log10(128), state_size)
        rates = half_lives.reciprocal()  # log2 attenuation per token
        self.decay_raw = nn.Parameter(torch.log(torch.expm1(rates)))
        scale = -0.5 * math.log2(state_size)
        self.write_log2_gain = nn.Parameter(torch.full((state_size,), scale))
        self.read_log2_gain = nn.Parameter(torch.full((state_size,), scale))
        self.diagonal = nn.Parameter(torch.tensor(1.0))

    def parameters_cartesian_reference(self, x: Tensor) -> dict[str, Tensor]:
        addresses = self.control_addresses(x)
        features = F.linear(addresses, self.characters)
        transition, write, read = features.split(self.state_size, dim=-1)
        decay_log2 = -F.softplus(self.decay_raw)
        a = polar_cartesian_reference(
            torch.exp2(decay_log2), transition * self.transition_frequencies
        )
        b = polar_cartesian_reference(torch.exp2(self.write_log2_gain), write)
        c = polar_cartesian_reference(torch.exp2(self.read_log2_gain), read)
        v = self.v_values(x)
        return {
            "a": a,
            "b": b,
            "c": c,
            "v": v,
            "addresses": addresses,
            "features": features,
            "decay_log2": decay_log2,
        }

    def control_addresses(self, input_hidden: Tensor) -> Tensor:
        """Float token-local phase ingress in radians, [B,T,address_channels]."""
        return math.pi * torch.tanh(self.address_projection(self.norm(input_hidden)))

    def v_values(self, input_hidden: Tensor) -> Tensor:
        """Float Cartesian payload ingress [B,T,P,2]; outer adapter only."""
        return self.value_projection(self.norm(input_hidden)).reshape(
            *input_hidden.shape[:-1], self.d_model, 2
        )

    def apply_readout(self, y: Tensor | np.ndarray, input_hidden: Tensor) -> Tensor:
        """Float outer seam after the native core, accepting pairs or complex."""
        if isinstance(y, np.ndarray):
            array = np.stack((y.real, y.imag), -1) if np.iscomplexobj(y) else y
            y = torch.as_tensor(array, dtype=input_hidden.dtype, device=input_hidden.device)
        elif y.is_complex():
            y = torch.stack((y.real, y.imag), -1).to(input_hidden)
        if y.shape != (*input_hidden.shape[:-1], self.d_model, 2):
            raise ValueError("readout expects complex [B,T,P] or Cartesian pairs [B,T,P,2]")
        return input_hidden + self.output_projection(y.flatten(-2))

    def forward_reference(
        self,
        x: Tensor,
        state: Tensor | None = None,
        reset_before: Tensor | None = None,
        capture: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch, length, _ = x.shape
        planes = self.parameters_cartesian_reference(x)
        expected = (batch, self.state_size, self.d_model, 2)
        if state is None:
            state = x.new_zeros(expected)
        if state.shape != expected or state.device != x.device or state.dtype != x.dtype:
            raise ValueError("state shape, device and dtype must match the reference carrier")
        outputs, states, reads = [], [], []
        for t in range(length):
            if reset_before is not None:
                state = torch.where(
                    reset_before[:, t, None, None, None], torch.zeros_like(state), state
                )
            advanced = multiply_cartesian_reference(planes["a"][:, t, :, None], state)
            read = multiply_cartesian_reference(planes["c"][:, t, :, None], advanced).sum(1)
            output = read + self.diagonal * planes["v"][:, t]
            state = advanced + multiply_cartesian_reference(
                planes["b"][:, t, :, None], planes["v"][:, t, None]
            )
            outputs.append(output)
            if capture:
                states.append(state.detach().clone())
                reads.append(read.detach().clone())
        mixed = torch.stack(outputs, 1) if outputs else x.new_empty(batch, 0, self.d_model, 2)
        result = self.apply_readout(mixed, x)
        diagnostics = {}
        if capture:
            diagnostics = {key: value.detach().clone() for key, value in planes.items()}
            diagnostics["mixed"] = mixed.detach().clone()
            diagnostics["state_trajectory"] = (
                torch.stack(states, 1) if states else state[:, None][:, :0]
            )
            diagnostics["strict_read"] = torch.stack(reads, 1) if reads else mixed.detach()
            diagnostics["output_hidden"] = result.detach().clone()
        return result, state, diagnostics

    def native_program(self) -> Any:
        """Build canonical fixed program; no private codec or primitive exists here."""
        from spf.tensor.core import encode_words
        from spf.tensor.sequence_mixer import (
            PhaseFeatureNetwork,
            PhaseMixerDirection,
            PhaseSequenceMixer,
        )

        character_array = self.characters.detach().cpu().numpy().astype(np.int64)
        rows = tuple(
            tuple((int(i), int(row[i])) for i in np.flatnonzero(row)) for row in character_array
        )
        n = self.state_size
        direction = PhaseMixerDirection(
            tuple(range(n)),
            tuple(range(n, 2 * n)),
            tuple(range(2 * n, 3 * n)),
            tuple(np.rint(self.write_log2_gain.detach().cpu().numpy() * 256).astype(int)),
            tuple(np.rint(self.read_log2_gain.detach().cpu().numpy() * 256).astype(int)),
        )
        return PhaseSequenceMixer(
            PhaseFeatureNetwork(self.n_address_channels, rows),
            direction,
            tuple(self.transition_frequencies.detach().cpu().numpy().astype(int)),
            decay_q16=tuple(
                np.rint(-F.softplus(self.decay_raw).detach().cpu().numpy() * 65536).astype(int)
            ),
            diagonal=int(encode_words(np.asarray(self.diagonal.detach().cpu().item()))),
        )

    def install_native_mixer(self) -> Any:
        """Snapshot the fixed learned gains/decay into a canonical native header."""
        return self.native_program()

    def export_native_inputs(self, diagnostics: dict[str, Tensor]) -> dict[str, Any]:
        """Quantized hybrid ingress; native recurrence replay belongs to caller.

        Fallback conversions count Cartesian address/value elements encoded.
        Real adapters and native mixing must be reported separately.
        """
        from spf.tensor.core import encode_words

        angles = diagnostics["addresses"].cpu().numpy()
        addresses = encode_words(np.cos(angles) + 1j * np.sin(angles))
        values = encode_words(numpy_complex_reference(diagnostics["v"]))
        program = self.native_program()
        return {
            "addresses": addresses,
            "values": values,
            "program": program,
            "parameters": program.generate(addresses),
            "clocked_parameters": program.generate_clocked(addresses),
            "fallback_conversions": int(addresses.size + values.size + 1),
            "execution_kind": "HYBRID_FLOAT_INGRESS_NATIVE_SPF_EXPORT",
        }


class SPFByteLanguageModelReference(nn.Module):
    execution_kind = EXECUTION_KIND

    def __init__(
        self, vocab_size: int = 256, d_model: int = 24, n_layers: int = 2, state_size: int = 8
    ):
        super().__init__()
        if min(vocab_size, d_model, n_layers, state_size) < 1:
            raise ValueError("model dimensions must be positive")
        self.config = dict(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, state_size=state_size
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            CartesianPhaseSequenceMixerReference(d_model, state_size) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.architecture = {
            "kind": "phase_sequence_mixer_byte_lm",
            **self.config,
            "execution_kind": EXECUTION_KIND,
            "controller": "learned_token_local_phase_ingress_fixed_sparse_integer_characters",
            "transition_frequencies": "fixed_integer_ones",
            "decay_and_log_gains": "learned_static_per_state",
            "recurrence": "advance_read_before_write_plus_scalar_diagonal",
            "payload_channels": d_model,
            "carrier_axes": ["batch", "state", "payload", "real_imag"],
            "tied_embeddings": False,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }
        # Enable only when a microscope consumer needs the detached trajectory.
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, Any] = {}

    def forward_reference(
        self,
        ids: Tensor,
        state: list[Tensor] | None = None,
        reset_before: Tensor | None = None,
        capture: bool = False,
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("ids must be an integer [batch,time] tensor")
        if state is not None and len(state) != len(self.layers):
            raise ValueError("one recurrent state is required for every layer")
        if reset_before is not None and (
            reset_before.shape != ids.shape or reset_before.dtype != torch.bool
        ):
            raise ValueError("reset_before must be a boolean [batch,time] tensor")
        hidden, states, layer_diagnostics = self.embedding(ids), [], []
        for i, layer in enumerate(self.layers):
            hidden, final, detail = layer.forward_reference(
                hidden, None if state is None else state[i], reset_before, capture
            )
            states.append(final)
            if capture:
                layer_diagnostics.append(detail)
        logits = self.head(self.final_norm(hidden))
        diagnostics = {"execution_kind": EXECUTION_KIND, "layers": layer_diagnostics}
        if capture:
            diagnostics["final_hidden"] = hidden.detach().clone()
        self.last_diagnostics = diagnostics
        return logits, states, diagnostics

    def forward(self, ids: Tensor) -> Tensor:
        return self.forward_reference(ids, capture=self.capture_diagnostics)[0]

    def model_sha256(self) -> str:
        digest = hashlib.sha256(repr(sorted(self.config.items())).encode())
        for name, value in self.state_dict().items():
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def capture_carrier(self, ids: Tensor, cut_at: int) -> dict[str, Any]:
        """cut_at is the number of consumed tokens, with a post-injection cut."""
        if not 0 <= cut_at <= ids.shape[1]:
            raise ValueError("cut_at must be within the supplied prefix")
        logits, states, _ = self.forward_reference(ids[:, :cut_at])
        return {
            "state_planes": {f"layer.{i}.state": s.detach().clone() for i, s in enumerate(states)},
            "model_sha256": self.model_sha256(),
            "processed_tokens": cut_at,
            "batch_size": ids.shape[0],
            "execution_kind": EXECUTION_KIND,
            "next_logits": logits[:, -1].clone() if cut_at else None,
            "prefix_ids": ids[:, :cut_at].clone(),
        }

    def _carrier_states(self, carrier: dict[str, Any]) -> list[Tensor]:
        if carrier["model_sha256"] != self.model_sha256():
            raise ValueError("carrier belongs to a different model checkpoint")
        return [carrier["state_planes"][f"layer.{i}.state"] for i in range(len(self.layers))]

    @torch.no_grad()
    def continue_from_carrier(self, carrier: dict[str, Any], remaining_ids: Tensor) -> Tensor:
        if remaining_ids.shape[0] != carrier["batch_size"]:
            raise ValueError("carrier and suffix batch sizes differ")
        return self.forward_reference(remaining_ids, self._carrier_states(carrier))[0]

    @torch.no_grad()
    def generate_from_carrier(
        self,
        carrier: dict[str, Any],
        next_logits: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> Tensor:
        """Return only generated bytes; every step consumes its generated history.

        next_logits is the already-computed prefix readout. Carrier ablation at
        this post-injection seam cannot retroactively change that first readout.
        """
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("generation budget and temperature must be nonnegative")
        states = self._carrier_states(carrier)
        generated = []
        for _ in range(max_new_tokens):
            token = (
                next_logits.argmax(-1)
                if temperature == 0
                else torch.multinomial((next_logits / temperature).softmax(-1), 1).squeeze(-1)
            )
            generated.append(token)
            logits, states, _ = self.forward_reference(token[:, None], states)
            next_logits = logits[:, 0]
        return (
            torch.stack(generated, 1)
            if generated
            else carrier["prefix_ids"].new_empty(carrier["batch_size"], 0)
        )

    @torch.no_grad()
    def generate(self, ids: Tensor, max_new_tokens: int, temperature: float = 0.0) -> Tensor:
        if ids.shape[1] == 0:
            raise ValueError("generation requires a nonempty byte prefix")
        carrier = self.capture_carrier(ids, ids.shape[1])
        generated = self.generate_from_carrier(
            carrier, carrier["next_logits"], max_new_tokens, temperature
        )
        return torch.cat((ids, generated), 1)

    @torch.no_grad()
    def export_reference_planes(self, ids: Tensor) -> list[dict[str, Any]]:
        """Export per-layer reference ingress for same-input native comparisons.

        Later layer inputs come from the reference preceding layer. Whole-model
        native replay must propagate each native output through the float adapter
        before obtaining the next layer ingress; this export alone is per-layer.
        """
        _, _, diagnostics = self.forward_reference(ids, capture=True)
        return [
            {
                **{key: numpy_complex_reference(detail[key]) for key in ("a", "b", "c", "v")},
                "diagonal": complex(layer.diagonal.item()),
                "addresses": detail["addresses"].cpu().numpy(),
                "execution_kind": EXECUTION_KIND,
            }
            for layer, detail in zip(self.layers, diagnostics["layers"])
        ]


def build_spf_model(
    vocab_size: int = 256, d_model: int = 24, n_layers: int = 2, state_size: int = 8, seed: int = 0
) -> SPFByteLanguageModelReference:
    """Seed model initialization without perturbing the caller's RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SPFByteLanguageModelReference(vocab_size, d_model, n_layers, state_size)
        model.architecture["seed"] = int(seed)
        return model
