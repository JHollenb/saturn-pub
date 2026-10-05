"""Typed donor-free SourceWrite Acts for GPT-NeoX attention boundaries.

The module is intentionally small and model-runtime agnostic.  It defines the
compiled artifact, the TITAN ``ActSpec`` describing its authority, exact
direct-source decomposition helpers, and a scoped hook runtime.  Donor trace
capture and consumer-closed optimization live in the experiment worker; a
hydrated artifact can only read the three slots declared here.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mdb.titan import ActRegistry, ActSpec

SOURCE_WRITE_ACT_SCHEMA = "saturn-ar-consumer-closed-source-write-act-v1"
SOURCE_WRITE_ACT_V2_SCHEMA = "saturn-ar-source-necessary-write-act-v2"
SOURCE_WRITE_ACT_NAME = "ar.attention.source-write"
SOURCE_WRITE_RUNTIME_ABI = "gpt-neox-query-source-write-hook-v1"


class SourceWriteActError(ValueError):
    """Raised when a SourceWrite Act or runtime frame fails closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourceWriteActError("SourceWrite value is not canonical JSON") from exc


def digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    raw = array.astype("<f4", copy=False).tobytes()
    return {
        "encoding": "little-endian-float32-gzip-base64",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_base64": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
    }


def _tensor_from_payload(payload: Mapping[str, Any]) -> torch.Tensor:
    if payload.get("encoding") != "little-endian-float32-gzip-base64":
        raise SourceWriteActError("unrecognized SourceWrite tensor encoding")
    try:
        shape = tuple(int(value) for value in payload["shape"])
        raw = gzip.decompress(base64.b64decode(str(payload["gzip_base64"])))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise SourceWriteActError("malformed SourceWrite tensor payload") from exc
    if not shape or any(value < 0 for value in shape):
        raise SourceWriteActError("invalid SourceWrite tensor shape")
    if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
        raise SourceWriteActError("SourceWrite tensor content hash mismatch")
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != math.prod(shape):
        raise SourceWriteActError("SourceWrite tensor payload geometry mismatch")
    return torch.from_numpy(values.copy()).reshape(shape)


def source_write_act_spec(
    *, hidden_size: int, layer: int, selected_heads: Sequence[int]
) -> ActSpec:
    """Declare the exact read/write authority of one compiled SourceWrite Act."""

    if hidden_size < 1 or layer < 0:
        raise SourceWriteActError("SourceWrite ABI dimensions are invalid")
    heads = tuple(int(value) for value in selected_heads)
    if not heads or len(heads) != len(set(heads)) or min(heads) < 0:
        raise SourceWriteActError("SourceWrite ABI requires unique non-negative heads")
    prefix = f"gpt_neox.layer{layer}"
    return ActSpec(
        name=SOURCE_WRITE_ACT_NAME,
        abi_version=f"v1.layer{layer}.h{'-'.join(str(value) for value in heads)}.d{hidden_size}",
        reads=(
            f"{prefix}.query.pre_qkv",
            f"{prefix}.source.pre_qkv",
            f"{prefix}.query.attention_write.native",
        ),
        writes=(f"{prefix}.query.direct_source_write.delta",),
        numerical_contract="float32-finite-additive-query-write",
        tolerance={"maximum_absolute_nonfinite": 0.0},
        write_kinds={f"{prefix}.query.direct_source_write.delta": "tensor"},
        write_abis={
            f"{prefix}.query.direct_source_write.delta": {
                "dtype": "float32-or-runtime-model-dtype",
                "shape": ["batch", hidden_size],
                "selected_heads": list(heads),
                "write_mode": "additive-delta-at-attention-dense-output",
                "scope": "declared-query-row-only",
            }
        },
    )


class SourceWriteAct(nn.Module):
    """A compact source/query-conditioned delta compiler.

    The bottleneck MLP sees only the layer-local query state, declared earlier
    source state, and native query attention write.  Its output is an additive
    delta at the same query write.  No token ids, gold labels, logits, or donor
    runtime values are accepted by the forward signature.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck: int,
        layer: int,
        selected_heads: Sequence[int],
        donor_identity: str,
        recipient_identity: str,
        seed: int,
    ) -> None:
        super().__init__()
        heads = tuple(int(value) for value in selected_heads)
        if hidden_size < 1 or bottleneck < 1 or layer < 0:
            raise SourceWriteActError("SourceWrite Act dimensions are invalid")
        if not heads or len(heads) != len(set(heads)) or min(heads) < 0:
            raise SourceWriteActError("SourceWrite Act head selection is invalid")
        if not donor_identity or not recipient_identity:
            raise SourceWriteActError("SourceWrite Act requires bound checkpoint identities")
        if seed < 0:
            raise SourceWriteActError("SourceWrite Act seed must be non-negative")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.layer = int(layer)
        self.selected_heads = heads
        self.donor_identity = str(donor_identity)
        self.recipient_identity = str(recipient_identity)
        self.seed = int(seed)
        self.down = nn.Linear(3 * hidden_size, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, hidden_size, bias=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        nn.init.xavier_uniform_(self.down.weight, generator=generator)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @property
    def act_spec(self) -> ActSpec:
        return source_write_act_spec(
            hidden_size=self.hidden_size,
            layer=self.layer,
            selected_heads=self.selected_heads,
        )

    def forward(
        self,
        query_state: torch.Tensor,
        source_state: torch.Tensor,
        native_write: torch.Tensor,
    ) -> torch.Tensor:
        values = (query_state, source_state, native_write)
        if any(value.ndim != 2 for value in values):
            raise SourceWriteActError("SourceWrite Act inputs must be rank-two tensors")
        if any(value.shape != query_state.shape for value in values[1:]):
            raise SourceWriteActError("SourceWrite Act input geometries do not align")
        if query_state.shape[-1] != self.hidden_size:
            raise SourceWriteActError("SourceWrite Act hidden width changed")
        features = torch.cat(tuple(value.float() for value in values), dim=-1)
        features = F.layer_norm(features, (features.shape[-1],))
        delta = self.up(F.gelu(self.down(features), approximate="tanh"))
        if not bool(torch.isfinite(delta).all().item()):
            raise SourceWriteActError("SourceWrite Act emitted a non-finite delta")
        return delta.to(dtype=native_write.dtype)

    def manifest(self) -> dict[str, Any]:
        state = self.state_dict()
        state_manifest = {
            name: {
                "shape": list(value.shape),
                "sha256": _tensor_payload(value)["sha256"],
            }
            for name, value in sorted(state.items())
        }
        body = {
            "schema": SOURCE_WRITE_ACT_SCHEMA,
            "runtime_abi": SOURCE_WRITE_RUNTIME_ABI,
            "architecture": {
                "kind": "three-slot-layernorm-gelu-bottleneck-delta",
                "hidden_size": self.hidden_size,
                "bottleneck": self.bottleneck,
                "layer": self.layer,
                "selected_heads": list(self.selected_heads),
                "parameter_count": self.parameter_count,
            },
            "checkpoint_binding": {
                "donor": self.donor_identity,
                "recipient": self.recipient_identity,
            },
            "seed": self.seed,
            "act_spec": self.act_spec.to_dict(),
            "state": state_manifest,
            "serving_forbidden_inputs": [
                "gold_answer",
                "teacher_or_donor_logits",
                "donor_runtime_state",
                "prompt_identity_lookup",
            ],
            "serving_reads": list(self.act_spec.reads),
            "serving_write": self.act_spec.writes[0],
        }
        return {**body, "fingerprint": digest_value(body)}

    def to_payload(self, *, training_receipt: Mapping[str, Any]) -> dict[str, Any]:
        manifest = self.manifest()
        body = {
            "schema": SOURCE_WRITE_ACT_SCHEMA,
            "manifest": manifest,
            "state": {
                name: _tensor_payload(value) for name, value in sorted(self.state_dict().items())
            },
            "training_receipt": dict(training_receipt),
        }
        return {**body, "content_sha256": digest_value(body)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SourceWriteAct:
        if payload.get("schema") != SOURCE_WRITE_ACT_SCHEMA:
            raise SourceWriteActError("unrecognized SourceWrite Act artifact")
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload.get("content_sha256") != digest_value(body):
            raise SourceWriteActError("SourceWrite Act artifact fingerprint changed")
        manifest = payload.get("manifest")
        state_payload = payload.get("state")
        if not isinstance(manifest, Mapping) or not isinstance(state_payload, Mapping):
            raise SourceWriteActError("SourceWrite Act artifact is incomplete")
        architecture = manifest.get("architecture")
        binding = manifest.get("checkpoint_binding")
        if not isinstance(architecture, Mapping) or not isinstance(binding, Mapping):
            raise SourceWriteActError("SourceWrite Act manifest is incomplete")
        act = cls(
            hidden_size=int(architecture["hidden_size"]),
            bottleneck=int(architecture["bottleneck"]),
            layer=int(architecture["layer"]),
            selected_heads=tuple(int(value) for value in architecture["selected_heads"]),
            donor_identity=str(binding["donor"]),
            recipient_identity=str(binding["recipient"]),
            seed=int(manifest["seed"]),
        )
        expected = set(act.state_dict())
        if set(state_payload) != expected:
            raise SourceWriteActError("SourceWrite Act state inventory changed")
        state = {name: _tensor_from_payload(state_payload[name]) for name in expected}
        act.load_state_dict(state, strict=True)
        if act.manifest() != manifest:
            raise SourceWriteActError("SourceWrite Act manifest does not match hydrated state")
        return act


class SourceWriteActV2(nn.Module):
    """A source-necessary, host-conditioned SourceWrite compiler.

    V1 concatenated all three slots before an ordinary MLP. That made the
    declared source available but did not make it causally necessary: query
    and native-write slots could learn a redundant repair. V2 removes that
    bypass. A bias-free source carrier is multiplied by a gate derived from
    the two host slots, then decoded by a bias-free output projection.

    Therefore ``source_state == 0`` implies an exactly zero delta for every
    possible parameter value. Correct-versus-wrong source specificity remains
    a learned property and must still pass contrastive controls.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck: int,
        layer: int,
        selected_heads: Sequence[int],
        donor_identity: str,
        recipient_identity: str,
        seed: int,
    ) -> None:
        super().__init__()
        heads = tuple(int(value) for value in selected_heads)
        if hidden_size < 1 or bottleneck < 1 or layer < 0:
            raise SourceWriteActError("SourceWrite Act dimensions are invalid")
        if not heads or len(heads) != len(set(heads)) or min(heads) < 0:
            raise SourceWriteActError("SourceWrite Act head selection is invalid")
        if not donor_identity or not recipient_identity:
            raise SourceWriteActError("SourceWrite Act requires bound checkpoint identities")
        if seed < 0:
            raise SourceWriteActError("SourceWrite Act seed must be non-negative")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.layer = int(layer)
        self.selected_heads = heads
        self.donor_identity = str(donor_identity)
        self.recipient_identity = str(recipient_identity)
        self.seed = int(seed)
        self.source_down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.host_down = nn.Linear(2 * hidden_size, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        nn.init.xavier_uniform_(self.source_down.weight, generator=generator)
        nn.init.xavier_uniform_(self.host_down.weight, generator=generator)
        nn.init.zeros_(self.host_down.bias)
        nn.init.zeros_(self.up.weight)

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @property
    def act_spec(self) -> ActSpec:
        return source_write_act_spec(
            hidden_size=self.hidden_size,
            layer=self.layer,
            selected_heads=self.selected_heads,
        )

    def forward(
        self,
        query_state: torch.Tensor,
        source_state: torch.Tensor,
        native_write: torch.Tensor,
    ) -> torch.Tensor:
        values = (query_state, source_state, native_write)
        if any(value.ndim != 2 for value in values):
            raise SourceWriteActError("SourceWrite Act inputs must be rank-two tensors")
        if any(value.shape != query_state.shape for value in values[1:]):
            raise SourceWriteActError("SourceWrite Act input geometries do not align")
        if query_state.shape[-1] != self.hidden_size:
            raise SourceWriteActError("SourceWrite Act hidden width changed")
        source = source_state.float()
        # No affine offset: zero source remains zero through the carrier.
        source = source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1.0e-6)
        source_carrier = F.gelu(self.source_down(source), approximate="tanh")
        host = torch.cat((query_state.float(), native_write.float()), dim=-1)
        host = F.layer_norm(host, (host.shape[-1],))
        host_gate = torch.sigmoid(self.host_down(host))
        delta = self.up(source_carrier * host_gate)
        if not bool(torch.isfinite(delta).all().item()):
            raise SourceWriteActError("SourceWrite Act emitted a non-finite delta")
        return delta.to(dtype=native_write.dtype)

    def manifest(self) -> dict[str, Any]:
        state_manifest = {
            name: {"shape": list(value.shape), "sha256": _tensor_payload(value)["sha256"]}
            for name, value in sorted(self.state_dict().items())
        }
        body = {
            "schema": SOURCE_WRITE_ACT_V2_SCHEMA,
            "runtime_abi": SOURCE_WRITE_RUNTIME_ABI,
            "architecture": {
                "kind": "source-keyed-multiplicative-host-gate-delta",
                "hidden_size": self.hidden_size,
                "bottleneck": self.bottleneck,
                "layer": self.layer,
                "selected_heads": list(self.selected_heads),
                "parameter_count": self.parameter_count,
                "source_zero_implies_delta_zero": True,
                "source_path_bias_free": True,
                "output_path_bias_free": True,
            },
            "checkpoint_binding": {
                "donor": self.donor_identity,
                "recipient": self.recipient_identity,
            },
            "seed": self.seed,
            "act_spec": self.act_spec.to_dict(),
            "state": state_manifest,
            "serving_forbidden_inputs": [
                "gold_answer",
                "teacher_or_donor_logits",
                "donor_runtime_state",
                "prompt_identity_lookup",
            ],
            "serving_reads": list(self.act_spec.reads),
            "serving_write": self.act_spec.writes[0],
        }
        return {**body, "fingerprint": digest_value(body)}

    def to_payload(self, *, training_receipt: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema": SOURCE_WRITE_ACT_V2_SCHEMA,
            "manifest": self.manifest(),
            "state": {
                name: _tensor_payload(value) for name, value in sorted(self.state_dict().items())
            },
            "training_receipt": dict(training_receipt),
        }
        return {**body, "content_sha256": digest_value(body)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SourceWriteActV2:
        if payload.get("schema") != SOURCE_WRITE_ACT_V2_SCHEMA:
            raise SourceWriteActError("unrecognized SourceWrite v2 Act artifact")
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload.get("content_sha256") != digest_value(body):
            raise SourceWriteActError("SourceWrite Act artifact fingerprint changed")
        manifest = payload.get("manifest")
        state_payload = payload.get("state")
        if not isinstance(manifest, Mapping) or not isinstance(state_payload, Mapping):
            raise SourceWriteActError("SourceWrite Act artifact is incomplete")
        architecture = manifest.get("architecture")
        binding = manifest.get("checkpoint_binding")
        if not isinstance(architecture, Mapping) or not isinstance(binding, Mapping):
            raise SourceWriteActError("SourceWrite Act manifest is incomplete")
        act = cls(
            hidden_size=int(architecture["hidden_size"]),
            bottleneck=int(architecture["bottleneck"]),
            layer=int(architecture["layer"]),
            selected_heads=tuple(int(value) for value in architecture["selected_heads"]),
            donor_identity=str(binding["donor"]),
            recipient_identity=str(binding["recipient"]),
            seed=int(manifest["seed"]),
        )
        expected = set(act.state_dict())
        if set(state_payload) != expected:
            raise SourceWriteActError("SourceWrite Act state inventory changed")
        state = {name: _tensor_from_payload(state_payload[name]) for name in expected}
        act.load_state_dict(state, strict=True)
        if act.manifest() != manifest:
            raise SourceWriteActError("SourceWrite Act manifest does not match hydrated state")
        return act


def load_source_write_act(payload: Mapping[str, Any]) -> SourceWriteAct | SourceWriteActV2:
    """Hydrate a versioned SourceWrite artifact without weakening either schema."""

    schema = payload.get("schema")
    if schema == SOURCE_WRITE_ACT_SCHEMA:
        return SourceWriteAct.from_payload(payload)
    if schema == SOURCE_WRITE_ACT_V2_SCHEMA:
        return SourceWriteActV2.from_payload(payload)
    raise SourceWriteActError("unrecognized SourceWrite Act artifact")


def direct_source_writes(
    *,
    attentions: torch.Tensor,
    qkv: torch.Tensor,
    dense_weight: torch.Tensor,
    query_positions: torch.Tensor,
    source_positions: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Return each head's exact bias-free contribution from one source address.

    The result has shape ``[batch, head, hidden]`` and sums to the selected
    source token's contribution after the GPT-NeoX output projection.
    """

    if attentions.ndim != 4 or qkv.ndim != 3 or dense_weight.ndim != 2:
        raise SourceWriteActError("direct-source inputs have invalid ranks")
    batch, tokens, width3 = qkv.shape
    hidden = dense_weight.shape[0]
    if (
        width3 != 3 * hidden
        or dense_weight.shape[1] != hidden
        or hidden % num_heads
        or attentions.shape[:2] != (batch, num_heads)
        or attentions.shape[2:] != (tokens, tokens)
        or query_positions.shape != (batch,)
        or source_positions.shape != (batch,)
    ):
        raise SourceWriteActError("direct-source geometry changed")
    if (
        bool((query_positions < 0).any().item())
        or bool((query_positions >= tokens).any().item())
        or bool((source_positions < 0).any().item())
        or bool((source_positions >= tokens).any().item())
    ):
        raise SourceWriteActError("direct-source address escaped the sequence")
    head_dim = hidden // num_heads
    values = qkv.reshape(batch, tokens, num_heads, 3 * head_dim)[..., 2 * head_dim :]
    row = torch.arange(batch, device=qkv.device)
    source_values = values[row, source_positions]
    source_weights = attentions[row, :, query_positions, source_positions]
    contexts = source_values * source_weights.unsqueeze(-1)
    projected = torch.einsum(
        "ohd,bhd->bho",
        dense_weight.reshape(hidden, num_heads, head_dim).float(),
        contexts.float(),
    )
    if not bool(torch.isfinite(projected).all().item()):
        raise SourceWriteActError("direct-source decomposition is non-finite")
    return projected


def _primary_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise SourceWriteActError("SourceWrite boundary has no primary tensor")


def _replace_primary(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return value
    if isinstance(output, tuple):
        return (value, *output[1:])
    raise SourceWriteActError("SourceWrite boundary output cannot be replaced")


class SourceWritePatchRuntime:
    """Scoped hook runtime for one compiled SourceWrite Act.

    A frame must explicitly bind query/source addresses, validity mask, and
    dose.  Outside a frame the hooks are inert.  Nested frames, stale captures,
    invalid addresses, and multiple boundary executions fail closed.
    """

    def __init__(self, model: Any, act: SourceWriteAct) -> None:
        self.model = model
        self.act = act
        try:
            layer = model.gpt_neox.layers[act.layer]
            self._qkv = layer.attention.query_key_value
            self._dense = layer.attention.dense
        except (AttributeError, IndexError, TypeError) as exc:
            raise SourceWriteActError("model does not expose the bound GPT-NeoX boundary") from exc
        self._frame: dict[str, Any] | None = None
        self._hidden: torch.Tensor | None = None
        self.last_delta: torch.Tensor | None = None
        self.last_native: torch.Tensor | None = None
        self.last_patched: torch.Tensor | None = None
        self.executions = 0
        self._handles = (
            self._qkv.register_forward_pre_hook(self._capture_hidden),
            self._dense.register_forward_hook(self._apply_delta),
        )
        self.installed = True

    def _capture_hidden(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self._frame is None:
            return
        if self._hidden is not None:
            raise SourceWriteActError("SourceWrite query-state capture executed more than once")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise SourceWriteActError("SourceWrite query-state boundary changed")
        self._hidden = inputs[0]

    def _apply_delta(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self._frame is None:
            return output
        if self._hidden is None:
            raise SourceWriteActError("SourceWrite attention write lacks its declared state")
        if self.executions:
            raise SourceWriteActError("SourceWrite attention write executed more than once")
        tensor = _primary_output(output)
        hidden = self._hidden
        query = self._frame["query_positions"].to(device=tensor.device, dtype=torch.long)
        source = self._frame["source_positions"].to(device=tensor.device, dtype=torch.long)
        mask = self._frame["attention_mask"].to(device=tensor.device)
        batch, tokens, width = tensor.shape
        if (
            hidden.shape != tensor.shape
            or width != self.act.hidden_size
            or query.shape != (batch,)
            or source.shape != (batch,)
            or mask.shape != (batch, tokens)
        ):
            raise SourceWriteActError("SourceWrite runtime frame geometry changed")
        row = torch.arange(batch, device=tensor.device)
        if (
            bool((query < 0).any().item())
            or bool((query >= tokens).any().item())
            or bool((source < 0).any().item())
            or bool((source >= tokens).any().item())
            or not bool(mask[row, query].all().item())
            or not bool(mask[row, source].all().item())
        ):
            raise SourceWriteActError("SourceWrite runtime address is invalid or masked")
        native = tensor[row, query]
        source_state = hidden[row, source]
        source_zero_mask = self._frame.get("source_zero_mask")
        if source_zero_mask is not None:
            zero_mask = source_zero_mask.to(device=tensor.device, dtype=torch.bool)
            if zero_mask.shape != (batch,):
                raise SourceWriteActError("SourceWrite source-zero audit mask is invalid")
            source_state = torch.where(
                zero_mask[:, None], torch.zeros_like(source_state), source_state
            )
        delta = self.act(hidden[row, query], source_state, native)
        raw_dose = self._frame["dose"]
        if isinstance(raw_dose, torch.Tensor):
            dose = raw_dose.to(device=tensor.device, dtype=torch.float32)
            if dose.shape != (batch,) or not bool(torch.isfinite(dose).all().item()):
                raise SourceWriteActError("SourceWrite runtime dose vector is invalid")
            dose_value: torch.Tensor | float = dose[:, None]
            exact_zero = bool((dose == 0).all().item())
        else:
            dose_value = float(raw_dose)
            if not math.isfinite(dose_value):
                raise SourceWriteActError("SourceWrite runtime dose is non-finite")
            exact_zero = dose_value == 0.0
        self.last_delta = delta
        self.last_native = native
        self.executions += 1
        if exact_zero:
            self.last_patched = native
            return output
        changed = tensor.clone()
        patched = native + dose_value * delta
        changed[row, query] = patched
        self.last_patched = patched
        return _replace_primary(output, changed)

    @contextmanager
    def frame(
        self,
        *,
        query_positions: torch.Tensor,
        source_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        dose: float | torch.Tensor = 1.0,
        source_zero_mask: torch.Tensor | None = None,
    ) -> Iterator[SourceWritePatchRuntime]:
        if not self.installed:
            raise SourceWriteActError("SourceWrite runtime is uninstalled")
        if self._frame is not None:
            raise SourceWriteActError("SourceWrite runtime frames cannot be nested")
        self._frame = {
            "query_positions": query_positions,
            "source_positions": source_positions,
            "attention_mask": attention_mask,
            "dose": dose,
            "source_zero_mask": source_zero_mask,
        }
        self._hidden = None
        self.last_delta = None
        self.last_native = None
        self.last_patched = None
        self.executions = 0
        try:
            yield self
            if self.executions != 1:
                raise SourceWriteActError("SourceWrite frame did not execute exactly once")
        finally:
            self._frame = None
            self._hidden = None

    def uninstall(self) -> None:
        if not self.installed:
            return
        if self._frame is not None:
            raise SourceWriteActError("cannot uninstall SourceWrite runtime inside a frame")
        for handle in reversed(self._handles):
            handle.remove()
        self.installed = False


def install_source_write_act(model: Any, act: SourceWriteAct) -> SourceWritePatchRuntime:
    return SourceWritePatchRuntime(model, act)


def register_source_write_act(
    registry: ActRegistry,
    act: SourceWriteAct,
    *,
    implementation: str = "compiled-bottleneck-mlp",
) -> ActSpec:
    """Register the pure tensor operation behind Saturn's fail-closed registry."""

    if not isinstance(registry, ActRegistry):
        raise SourceWriteActError("SourceWrite registration requires an ActRegistry")
    spec = act.act_spec

    def execute(values: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        query_name, source_name, native_name = spec.reads
        write_name = spec.writes[0]
        delta = act(values[query_name], values[source_name], values[native_name])
        return {write_name: delta}, {
            "numerical_contract_ok": bool(torch.isfinite(delta).all().item()),
            "parameter_count": act.parameter_count,
            "model_invocations": 0,
            "donor_runtime_reads": 0,
        }

    registry.register(spec, implementation, execute)
    return spec


__all__ = [
    "SOURCE_WRITE_ACT_NAME",
    "SOURCE_WRITE_ACT_SCHEMA",
    "SOURCE_WRITE_ACT_V2_SCHEMA",
    "SOURCE_WRITE_RUNTIME_ABI",
    "SourceWriteAct",
    "SourceWriteActV2",
    "SourceWriteActError",
    "SourceWritePatchRuntime",
    "digest_value",
    "direct_source_writes",
    "install_source_write_act",
    "load_source_write_act",
    "register_source_write_act",
    "source_write_act_spec",
]
