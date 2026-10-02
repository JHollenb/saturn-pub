"""Git-like context captures for Saturn's autoregressive microscope.

This extension composes two existing Saturn primitives instead of introducing
another cache format:

* :class:`~saturn.future_forest.PersistentFutureForest` stores immutable
  capture nodes, probe branches, observations, and decisions;
* :class:`~saturn.ar.persistent_kv_history.PersistentKVHistoryStore` stores
  exact KV pages as a Merkle history whose children contain only appended
  pages.

The register is deliberately an evidence format.  It records token/KV state,
typed layer-cut carriers, and causal probe results.  It does not claim that a
hidden state is a transcript of a model's private reasoning or that a probe
has identified a unique semantic circuit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mdb.ar.persistent_kv_history import (
    PersistentKVHistoryError,
    PersistentKVHistoryStore,
    PersistentKVMetadata,
    PersistentKVRootManifest,
)
from mdb.future_forest import PersistentFutureForest
from mdb.qwen2_machine import AutoregressiveState, ModelIdentity, NormalizedKVCache

THOTT_TOOL_NAME = "Thott"
THOTT_TOOL_ID = "saturn-thott"
# Kept as import aliases so clients written against the first prototype do not
# break while all new manifests, receipts, and CLI help use Thott.
JANUS_TOOL_NAME = THOTT_TOOL_NAME
JANUS_TOOL_ID = THOTT_TOOL_ID
CONTEXT_REGISTER_SCHEMA = "saturn-context-register-v1"
CONTEXT_SNAPSHOT_SCHEMA = "saturn-context-snapshot-v1"
CONTEXT_PROBE_SCHEMA = "saturn-context-probe-v1"
CONTEXT_DIFF_SCHEMA = "saturn-context-diff-v1"
CONTEXT_RESIDUAL_SCHEMA = "saturn-context-representation-residual-v1"


class ContextRegisterError(RuntimeError):
    """Raised when a context register cannot prove its custody or ABI."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContextRegisterError("context register value is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical(value))


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContextRegisterError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ContextRegisterError(f"{label} must be finite")
    return result


def _json_mapping(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        payload = json.loads(_canonical(value).decode("utf-8"))
    except ContextRegisterError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContextRegisterError(f"{label} is not JSON-like") from exc
    if not isinstance(payload, dict):
        raise ContextRegisterError(f"{label} must be an object")
    return payload


def _tensor_bundle(tensors: Mapping[str, torch.Tensor]) -> bytes:
    if not tensors:
        raise ContextRegisterError("tensor bundle cannot be empty")
    prepared: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if not isinstance(name, str) or not name:
            raise ContextRegisterError("tensor bundle names must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise ContextRegisterError(f"tensor bundle entry {name!r} is not a tensor")
        prepared[name] = value.detach().to(device="cpu").contiguous()
    return bytes(save_safetensors(prepared))


def _normalised_cache(value: Any) -> NormalizedKVCache:
    if isinstance(value, NormalizedKVCache):
        return value
    to_normalized = getattr(value, "to_normalized", None)
    if callable(to_normalized):
        result = to_normalized()
        if isinstance(result, NormalizedKVCache):
            return result
    raise ContextRegisterError("context register requires a normalized or paged KV cache")


def _cache_prefix_equal(left: NormalizedKVCache, right: NormalizedKVCache, length: int) -> bool:
    if len(left.layers) != len(right.layers):
        return False
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        if left_layer is None or right_layer is None:
            if left_layer is not right_layer:
                return False
            continue
        if left_layer.key.shape[-2] < length or right_layer.key.shape[-2] < length:
            return False
        if not torch.equal(left_layer.key[..., :length, :], right_layer.key[..., :length, :]):
            return False
        if not torch.equal(left_layer.value[..., :length, :], right_layer.value[..., :length, :]):
            return False
    return True


def _cache_exact_equal(left: NormalizedKVCache, right: NormalizedKVCache) -> bool:
    if len(left.layers) != len(right.layers):
        return False
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        if left_layer is None or right_layer is None:
            if left_layer is not right_layer:
                return False
            continue
        if not torch.equal(left_layer.key, right_layer.key):
            return False
        if not torch.equal(left_layer.value, right_layer.value):
            return False
    return True


def _cache_append(
    current: NormalizedKVCache,
    parent: NormalizedKVCache,
    parent_length: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    appends: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, (current_layer, parent_layer) in enumerate(
        zip(current.layers, parent.layers, strict=True)
    ):
        if current_layer is None or parent_layer is None:
            raise ContextRegisterError("append custody requires complete KV layers")
        if current_layer.key.shape[-2] <= parent_length:
            raise ContextRegisterError("append custody requires a strictly longer child cache")
        appends[index] = (
            current_layer.key[..., parent_length:, :],
            current_layer.value[..., parent_length:, :],
        )
    return appends


def _layer_residual_row(
    reference_key: torch.Tensor,
    reference_value: torch.Tensor,
    approximation_key: torch.Tensor,
    approximation_value: torch.Tensor,
    *,
    token_ids: Sequence[int] | None,
    layer: int,
) -> dict[str, Any]:
    reference_key = reference_key.detach().float()
    reference_value = reference_value.detach().float()
    approximation_key = approximation_key.detach().float()
    approximation_value = approximation_value.detach().float()
    key_delta = reference_key - approximation_key
    value_delta = reference_value - approximation_value
    reference_energy = float(
        reference_key.square().sum().item() + reference_value.square().sum().item()
    )
    residual_energy = float(
        key_delta.square().sum().item() + value_delta.square().sum().item()
    )
    reference_norm = math.sqrt(max(reference_energy, 0.0))
    residual_norm = math.sqrt(max(residual_energy, 0.0))
    position_energy = key_delta.square().sum(dim=(0, 1, 3)) + value_delta.square().sum(
        dim=(0, 1, 3)
    )
    positions = int(position_energy.numel())
    top_count = min(8, positions)
    top = torch.topk(position_energy, k=top_count) if top_count else None
    top_positions = []
    if top is not None:
        for position, energy in zip(top.indices.tolist(), top.values.tolist(), strict=True):
            row: dict[str, Any] = {"position": int(position), "energy": float(energy)}
            if token_ids is not None and int(position) < len(token_ids):
                row["token_id"] = int(token_ids[int(position)])
            top_positions.append(row)
    quarter = max(1, positions // 4)
    prefix_energy = float(position_energy[:quarter].mean().item()) if positions else 0.0
    tail_energy = float(position_energy[-quarter:].mean().item()) if positions else 0.0
    return {
        "layer": int(layer),
        "positions": positions,
        "reference_l2": reference_norm,
        "residual_l2": residual_norm,
        "relative_l2": residual_norm / reference_norm if reference_norm else 0.0,
        "residual_energy_share": residual_energy / reference_energy if reference_energy else 0.0,
        "max_abs": float(max(key_delta.abs().max().item(), value_delta.abs().max().item())),
        "position_energy_prefix_quarter_mean": prefix_energy,
        "position_energy_tail_quarter_mean": tail_energy,
        "tail_to_prefix_position_energy": tail_energy / prefix_energy if prefix_energy else None,
        "top_positions": top_positions,
    }


def representation_residual_profile(
    reference_cache: Any,
    approximation_cache: Any,
    *,
    token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Measure the cache representation residual consumed by a reconstruction.

    The residual is defined as ``reference KV - approximation KV``.  It is a
    representation error, not a model-thinking channel.  The profile keeps
    layer growth, position concentration, and token labels so a later causal
    intervention can test whether the measured residue matters to the native
    consumer.
    """

    reference = _normalised_cache(reference_cache)
    approximation = _normalised_cache(approximation_cache)
    if len(reference.layers) != len(approximation.layers):
        raise ContextRegisterError("residual caches have different layer counts")
    rows: list[dict[str, Any]] = []
    for index, (reference_layer, approximation_layer) in enumerate(
        zip(reference.layers, approximation.layers, strict=True)
    ):
        if reference_layer is None or approximation_layer is None:
            if reference_layer is not approximation_layer:
                raise ContextRegisterError("residual caches disagree on an empty layer")
            continue
        if reference_layer.key.shape != approximation_layer.key.shape:
            raise ContextRegisterError("residual caches disagree on KV geometry")
        rows.append(
            _layer_residual_row(
                reference_layer.key,
                reference_layer.value,
                approximation_layer.key,
                approximation_layer.value,
                token_ids=token_ids,
                layer=index,
            )
        )
    relative = [float(row["relative_l2"]) for row in rows]
    shares = [float(row["residual_energy_share"]) for row in rows]
    relative_pairs = sum(
        right >= left for left, right in zip(relative[:-1], relative[1:], strict=True)
    )
    share_pairs = sum(
        right >= left for left, right in zip(shares[:-1], shares[1:], strict=True)
    )
    pair_count = max(1, len(rows) - 1)
    return {
        "schema": CONTEXT_RESIDUAL_SCHEMA,
        "definition": "native_reference_minus_reconstructed_representation",
        "layers": rows,
        "relative_l2_by_layer": relative,
        "residual_energy_share_by_layer": shares,
        "relative_l2_growing_pair_fraction": relative_pairs / pair_count,
        "energy_share_growing_pair_fraction": share_pairs / pair_count,
        "trend_label": (
            "growing_with_depth"
            if relative and relative_pairs / pair_count >= 0.75
            else "mixed_or_flat"
        ),
    }


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Tensor-free handle for one context-register node."""

    node_id: str
    label: str
    stage: str
    token_cut: int
    kv_sequence_length: int
    frame_id: str
    kv_root_sha256: str
    storage_mode: str
    residual_profile: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_SNAPSHOT_SCHEMA,
            "tool": {"name": THOTT_TOOL_NAME, "id": THOTT_TOOL_ID},
            "node_id": self.node_id,
            "label": self.label,
            "stage": self.stage,
            "token_cut": self.token_cut,
            "kv_sequence_length": self.kv_sequence_length,
            "frame_id": self.frame_id,
            "kv_root_sha256": self.kv_root_sha256,
            "storage_mode": self.storage_mode,
            "residual_profile": None
            if self.residual_profile is None
            else dict(self.residual_profile),
        }


class ContextRegister:
    """Persistent context register backed by Saturn's existing CAS systems."""

    def __init__(
        self,
        root: Path,
        *,
        forest: PersistentFutureForest,
        kv_store: PersistentKVHistoryStore,
        model_identity: Mapping[str, Any],
        register_manifest: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.forest = forest
        self.kv_store = kv_store
        self.model_identity = dict(model_identity)
        self.register_manifest = dict(register_manifest)

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        model_identity: Mapping[str, Any],
        register_id: str = "context-register",
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextRegister:
        path = Path(root).expanduser().resolve()
        if (path / "register.json").exists():
            raise ContextRegisterError(f"context register already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)
        forest = PersistentFutureForest(path / "forest", forest_id=register_id)
        kv_store = PersistentKVHistoryStore(path / "kv")
        identity = _json_mapping(model_identity, label="model_identity")
        identity_ref = forest.put_json_object(
            {"schema": CONTEXT_REGISTER_SCHEMA, "model_identity": identity},
            media_type="application/vnd.saturn.context-identity+json",
        )
        root_node = forest.create_root(
            state_refs={"context/model_identity": identity_ref},
            metadata=_json_mapping(metadata, label="register metadata"),
        )
        manifest_body = {
            "schema": CONTEXT_REGISTER_SCHEMA,
            "tool": {"name": THOTT_TOOL_NAME, "id": THOTT_TOOL_ID},
            "register_id": register_id,
            "model_identity": identity,
            "root_node_id": root_node,
            "storage": {
                "forest": "forest/",
                "persistent_kv": "kv/",
                "tensor_payloads": "forest/objects/",
            },
        }
        manifest = {**manifest_body, "content_sha256": _digest(manifest_body)}
        (path / "register.json").write_bytes(_canonical(manifest) + b"\n")
        return cls(
            path,
            forest=forest,
            kv_store=kv_store,
            model_identity=identity,
            register_manifest=manifest,
        )

    @classmethod
    def reopen(cls, root: str | Path) -> ContextRegister:
        path = Path(root).expanduser().resolve()
        try:
            manifest = json.loads((path / "register.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContextRegisterError(f"cannot read context register: {path}") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema") != CONTEXT_REGISTER_SCHEMA:
            raise ContextRegisterError("context register schema changed")
        unsigned = {key: value for key, value in manifest.items() if key != "content_sha256"}
        if manifest.get("content_sha256") != _digest(unsigned):
            raise ContextRegisterError("context register manifest hash changed")
        forest = PersistentFutureForest.reopen(path / "forest")
        root_node = str(manifest.get("root_node_id"))
        if root_node not in forest.list_nodes():
            raise ContextRegisterError("context register root node is missing")
        forest.verify_node(root_node)
        return cls(
            path,
            forest=forest,
            kv_store=PersistentKVHistoryStore(path / "kv"),
            model_identity=_json_mapping(manifest.get("model_identity"), label="model_identity"),
            register_manifest=dict(manifest),
        )

    @property
    def root_node_id(self) -> str:
        return str(self.register_manifest["root_node_id"])

    def head(self) -> str:
        return self.forest.head("accepted")

    def _validate_state(self, state: AutoregressiveState) -> NormalizedKVCache:
        if not isinstance(state, AutoregressiveState):
            raise ContextRegisterError("context snapshot requires an AutoregressiveState")
        if state.model_identity.to_dict() != self.model_identity:
            raise ContextRegisterError("context snapshot model identity does not match register")
        return _normalised_cache(state.kv_cache)

    def _kv_root_for_state(
        self,
        cache: NormalizedKVCache,
        *,
        parent: ContextSnapshot | None,
        parent_state: AutoregressiveState | None,
    ) -> tuple[PersistentKVRootManifest, str]:
        if parent is not None and parent_state is not None:
            parent_cache = self._validate_state(parent_state)
            parent_length = parent_cache.common_sequence_length()
            current_length = cache.common_sequence_length()
            if (
                current_length > parent_length
                and _cache_prefix_equal(cache, parent_cache, parent_length)
                and parent.kv_root_sha256
            ):
                appends = _cache_append(cache, parent_cache, parent_length)
                try:
                    manifest = self.kv_store.append(parent.kv_root_sha256, appends)
                except PersistentKVHistoryError as exc:
                    raise ContextRegisterError("persistent KV append custody failed") from exc
                return manifest, "append"
            if current_length == parent_length and _cache_exact_equal(cache, parent_cache):
                manifest = self.kv_store.load_root(parent.kv_root_sha256)
                return manifest, "shared"
        try:
            metadata = PersistentKVMetadata.from_cache(
                cache,
                model_fingerprint=_digest(self.model_identity),
                block_size=128,
            )
            manifest = self.kv_store.publish_genesis(cache, metadata=metadata)
        except PersistentKVHistoryError as exc:
            raise ContextRegisterError("persistent KV genesis custody failed") from exc
        return manifest, "genesis"

    def capture_state(
        self,
        state: AutoregressiveState,
        *,
        label: str,
        stage: str = "token-cut",
        parent: ContextSnapshot | None = None,
        parent_state: AutoregressiveState | None = None,
        cycle: Any | None = None,
        residual_profile: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextSnapshot:
        """Capture an exact state and optional layer-cut carrier evidence."""

        cache = self._validate_state(state)
        if not isinstance(label, str) or not label:
            raise ContextRegisterError("context snapshot label must be non-empty")
        if not isinstance(stage, str) or not stage:
            raise ContextRegisterError("context snapshot stage must be non-empty")
        parent_node = parent.node_id if parent is not None else self.head()
        manifest, storage_mode = self._kv_root_for_state(
            cache, parent=parent, parent_state=parent_state
        )
        token_bundle = _tensor_bundle(
            {"token_ids": state.token_ids, "attention_mask": state.attention_mask}
        )
        token_ref = self.forest.put_blob(
            token_bundle,
            media_type="application/vnd.saturn.context-tensors+safetensors",
            metadata={"token_cut": int(state.token_ids.shape[1])},
        )
        kv_root_ref = self.forest.put_blob(
            _canonical(manifest.to_dict()),
            media_type="application/vnd.saturn.persistent-kv-root+json",
            metadata={"root_sha256": manifest.root_sha256},
        )
        descriptor: dict[str, Any] = {
            "schema": CONTEXT_SNAPSHOT_SCHEMA,
            "label": label,
            "stage": stage,
            "model_identity": self.model_identity,
            "frame_id": state.frame_id,
            "token_cut": int(state.token_ids.shape[1]),
            "kv_sequence_length": int(cache.common_sequence_length()),
            "prompt_length": int(state.prompt_length),
            "cursor": int(state.cursor),
            "max_new_tokens": int(state.max_new_tokens),
            "eos_token_ids": [int(value) for value in state.eos_token_ids],
            "finished": bool(state.finished),
            "finish_reason": state.finish_reason,
            "sampler": str(state.sampler),
            "pending_token_id": int(state.pending_token_id),
            "kv_runtime_fingerprint": str(cache.fingerprint),
            "storage_mode": storage_mode,
            "metadata": _json_mapping(metadata, label="snapshot metadata"),
        }
        descriptor_ref = self.forest.put_json_object(
            descriptor,
            media_type="application/vnd.saturn.context-snapshot+json",
        )
        diff_refs: dict[str, Mapping[str, Any]] = {
            "context/tensors": token_ref,
            "context/kv_root": kv_root_ref,
            "context/descriptor": descriptor_ref,
        }
        cut_descriptors: list[dict[str, Any]] = []
        if cycle is not None:
            cuts = tuple(getattr(cycle, "cuts", ()))
            for cut in cuts:
                hidden = getattr(cut, "hidden_state", None)
                if not isinstance(hidden, torch.Tensor):
                    continue
                layer = int(cut.layer_index)
                ref = self.forest.put_blob(
                    _tensor_bundle({"hidden_state": hidden}),
                    media_type="application/vnd.saturn.layer-cut+safetensors",
                    metadata={"layer": layer, "source": "cycle.cut"},
                )
                address = f"carrier/layer/{layer:04d}"
                diff_refs[address] = ref
                cut_descriptors.append(
                    {
                        "layer": layer,
                        "shape": [int(item) for item in hidden.shape],
                        "hidden_sha256": str(getattr(cut, "hidden_sha256", "")),
                        "position_indices": [
                            int(item) for item in getattr(cut, "position_indices", ())
                        ],
                        "resumable": bool(getattr(cut, "resumable", False)),
                    }
                )
        if residual_profile is not None:
            residual_payload = _json_mapping(residual_profile, label="residual profile")
            diff_refs["context/residual_profile"] = self.forest.put_json_object(
                residual_payload,
                media_type="application/vnd.saturn.context-residual+json",
            )
        node_id = self.forest.fork(
            parent_node,
            transition={
                "kind": "context-capture",
                "label": label,
                "stage": stage,
                "token_cut": int(state.token_ids.shape[1]),
                "storage_mode": storage_mode,
            },
            diff_refs=diff_refs,
            metadata={
                **_json_mapping(metadata, label="node metadata"),
                "snapshot_schema": CONTEXT_SNAPSHOT_SCHEMA,
                "cycle_cut_layers": [row["layer"] for row in cut_descriptors],
            },
        )
        return ContextSnapshot(
            node_id=node_id,
            label=label,
            stage=stage,
            token_cut=int(state.token_ids.shape[1]),
            kv_sequence_length=int(cache.common_sequence_length()),
            frame_id=state.frame_id,
            kv_root_sha256=manifest.root_sha256,
            storage_mode=storage_mode,
            residual_profile=None if residual_profile is None else dict(residual_profile),
        )

    def record_probe(
        self,
        parent: ContextSnapshot,
        *,
        label: str,
        scores: Mapping[str, float],
        decision: str = "retain",
        reason: str = "retain causal evidence for later review",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a branch-local probe observation and an explicit decision."""

        score_payload = {
            str(key): _finite(value, label=f"probe score {key}")
            for key, value in scores.items()
        }
        evidence = self.forest.put_json_object(
            {
                "schema": CONTEXT_PROBE_SCHEMA,
                "label": label,
                "scores": score_payload,
                "metadata": _json_mapping(metadata, label="probe metadata"),
            },
            media_type="application/vnd.saturn.context-probe+json",
        )
        node_id = self.forest.fork(
            parent.node_id,
            transition={"kind": "causal-probe", "label": label},
            diff_refs={f"probe/{label}": evidence},
            metadata=_json_mapping(metadata, label="probe node metadata"),
        )
        observation_id = self.forest.observe(
            node_id,
            stage="native-consumer",
            scores=score_payload,
            evidence_refs=(evidence,),
            metadata={
                "label": label,
                **_json_mapping(metadata, label="probe observation metadata"),
            },
        )
        decision_id = self.forest.decide(
            node_id,
            decision=decision,  # type: ignore[arg-type]
            observation_ids=(observation_id,),
            reason=reason,
        )
        return {
            "schema": CONTEXT_PROBE_SCHEMA,
            "tool": {"name": THOTT_TOOL_NAME, "id": THOTT_TOOL_ID},
            "label": label,
            "node_id": node_id,
            "observation_id": observation_id,
            "decision_id": decision_id,
            "scores": score_payload,
        }

    def snapshot_descriptor(self, node_id: str) -> dict[str, Any]:
        state = self.forest.effective_state(node_id)
        reference = state.get("context/descriptor")
        if reference is None:
            raise ContextRegisterError(f"node {node_id} has no context descriptor")
        try:
            payload = json.loads(self.forest.get_blob(reference).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContextRegisterError("context descriptor is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema") != CONTEXT_SNAPSHOT_SCHEMA:
            raise ContextRegisterError("context descriptor schema changed")
        return payload

    def restore_state(
        self,
        node_id: str,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[AutoregressiveState, dict[str, Any]]:
        """Verify and hydrate a captured Thott state for live model execution.

        Storage ancestry reconstructs bytes through the persistent-KV Merkle
        chain. Causal ancestry remains the Future Forest node history. Neither
        is silently substituted for the other in the restore receipt.
        """

        state_refs = self.forest.effective_state(node_id)
        descriptor = self.snapshot_descriptor(node_id)
        token_ref = state_refs.get("context/tensors")
        kv_ref = state_refs.get("context/kv_root")
        if token_ref is None or kv_ref is None:
            raise ContextRegisterError("context snapshot has no executable token/KV closure")
        required = (
            "prompt_length",
            "cursor",
            "max_new_tokens",
            "eos_token_ids",
            "finished",
            "finish_reason",
            "sampler",
        )
        missing = [key for key in required if key not in descriptor]
        if missing:
            raise ContextRegisterError(
                f"context snapshot predates executable restore fields: {missing}"
            )
        try:
            token_bundle = self.forest.get_blob(token_ref)
            tensors = load_safetensors(token_bundle)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ContextRegisterError("context token tensor closure failed to hydrate") from exc
        if set(tensors) != {"attention_mask", "token_ids"}:
            raise ContextRegisterError("context token tensor inventory changed")
        token_ids = tensors["token_ids"]
        attention_mask = tensors["attention_mask"]
        if device is not None:
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
        try:
            kv_manifest = PersistentKVRootManifest.from_dict(
                json.loads(self.forest.get_blob(kv_ref).decode("utf-8"))
            )
            hydrated = self.kv_store.hydrate(kv_manifest.root_sha256, device=device)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            PersistentKVHistoryError,
        ) as exc:
            raise ContextRegisterError("context KV closure failed to hydrate") from exc
        identity = ModelIdentity.from_mapping(self.model_identity)
        if descriptor.get("model_identity") != identity.to_dict():
            raise ContextRegisterError("context descriptor model identity changed")
        restored = AutoregressiveState(
            token_ids=token_ids,
            attention_mask=attention_mask,
            kv_cache=hydrated.cache.to_normalized(),
            prompt_length=int(descriptor["prompt_length"]),
            cursor=int(descriptor["cursor"]),
            max_new_tokens=int(descriptor["max_new_tokens"]),
            eos_token_ids=tuple(int(value) for value in descriptor["eos_token_ids"]),
            model_identity=identity,
            frame_id=str(descriptor["frame_id"]),
            finished=bool(descriptor["finished"]),
            finish_reason=descriptor["finish_reason"],
            sampler=str(descriptor["sampler"]),
        )
        if restored.pending_token_id != int(descriptor["pending_token_id"]):
            raise ContextRegisterError("restored pending token changed")
        if int(restored.token_ids.shape[1]) != int(descriptor["token_cut"]):
            raise ContextRegisterError("restored token cut changed")
        if restored.kv_cache.common_sequence_length() != int(
            descriptor["kv_sequence_length"]
        ):
            raise ContextRegisterError("restored KV sequence length changed")
        causal_node = self.forest.verify_node(node_id)
        receipt = {
            "schema": "saturn-context-live-restore-v1",
            "tool": {"name": THOTT_TOOL_NAME, "id": THOTT_TOOL_ID},
            "node_id": node_id,
            "frame_id": restored.frame_id,
            "model_identity": identity.to_dict(),
            "token_cut": int(restored.token_ids.shape[1]),
            "kv_sequence_length": restored.kv_cache.common_sequence_length(),
            "storage_ancestry": {
                "kv_root_sha256": hydrated.receipt.root_sha256,
                "kv_root_chain": list(hydrated.root_chain),
                "closure": hydrated.receipt.to_dict(),
            },
            "causal_ancestry": {
                "primary_parent": causal_node.get("primary_parent"),
                "parents": list(causal_node.get("parents", ())),
            },
            "execution_authority": {
                "state_hydrated": True,
                "model_invocations": 0,
                "act_executions": 0,
                "caller_must_bind_matching_resident_model": True,
            },
        }
        receipt["content_sha256"] = _digest(receipt)
        return restored, receipt

    def diff(self, left_node_id: str, right_node_id: str) -> dict[str, Any]:
        left = self.forest.effective_state(left_node_id)
        right = self.forest.effective_state(right_node_id)
        addresses = sorted(set(left) | set(right))
        changed = [
            address
            for address in addresses
            if left.get(address) != right.get(address)
        ]
        return {
            "schema": CONTEXT_DIFF_SCHEMA,
            "tool": {"name": THOTT_TOOL_NAME, "id": THOTT_TOOL_ID},
            "left_node_id": left_node_id,
            "right_node_id": right_node_id,
            "left_is_ancestor": left_node_id in self.forest.ancestors(
                right_node_id, include_self=True
            ),
            "right_is_ancestor": right_node_id in self.forest.ancestors(
                left_node_id, include_self=True
            ),
            "changed_addresses": changed,
            "unchanged_addresses": [address for address in addresses if address not in changed],
            "left_descriptor": self.snapshot_descriptor(left_node_id)
            if "context/descriptor" in left
            else None,
            "right_descriptor": self.snapshot_descriptor(right_node_id)
            if "context/descriptor" in right
            else None,
        }

    def verify(self) -> dict[str, Any]:
        closure = self.forest.verify_closure()
        roots: set[str] = set()
        descriptors = 0
        for node_id in self.forest.list_nodes():
            state = self.forest.effective_state(node_id)
            descriptor_ref = state.get("context/descriptor")
            kv_ref = state.get("context/kv_root")
            if descriptor_ref is not None:
                self.snapshot_descriptor(node_id)
                descriptors += 1
            if kv_ref is None:
                continue
            raw = self.forest.get_blob(kv_ref)
            try:
                manifest = PersistentKVRootManifest.from_dict(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError, PersistentKVHistoryError) as exc:
                raise ContextRegisterError("context KV-root descriptor is invalid") from exc
            if manifest.root_sha256 != str(kv_ref.get("metadata", {}).get("root_sha256")):
                raise ContextRegisterError("context KV-root reference metadata changed")
            roots.add(manifest.root_sha256)
        kv_receipts = [self.kv_store.verify(root).to_dict() for root in sorted(roots)]
        return {
            "schema": CONTEXT_REGISTER_SCHEMA,
            "tool": {"name": THOTT_TOOL_NAME, "id": THOTT_TOOL_ID},
            "register_id": self.register_manifest.get("register_id"),
            "root_node_id": self.root_node_id,
            "accepted_head": self.head(),
            "context_descriptors": descriptors,
            "persistent_kv_roots": kv_receipts,
            "forest_closure": closure,
            "all_context_dependencies_verified": True,
        }

    def inspect(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_REGISTER_SCHEMA,
            "tool": {"name": THOTT_TOOL_NAME, "id": THOTT_TOOL_ID},
            "path": str(self.root),
            "manifest": dict(self.register_manifest),
            "verification": self.verify(),
        }


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=THOTT_TOOL_ID,
        description="Inspect and verify Saturn's persistent context register.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("register", type=Path)
    diff = subparsers.add_parser("diff")
    diff.add_argument("register", type=Path)
    diff.add_argument("left_node")
    diff.add_argument("right_node")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    register = ContextRegister.reopen(args.register)
    if args.command == "inspect":
        payload = register.inspect()
    elif args.command == "verify":
        payload = register.verify()
    else:
        payload = register.diff(args.left_node, args.right_node)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = [
    "THOTT_TOOL_ID",
    "THOTT_TOOL_NAME",
    "JANUS_TOOL_ID",
    "JANUS_TOOL_NAME",
    "CONTEXT_DIFF_SCHEMA",
    "CONTEXT_PROBE_SCHEMA",
    "CONTEXT_REGISTER_SCHEMA",
    "CONTEXT_RESIDUAL_SCHEMA",
    "CONTEXT_SNAPSHOT_SCHEMA",
    "ContextRegister",
    "ContextRegisterError",
    "ContextSnapshot",
    "main",
    "representation_residual_profile",
]


if __name__ == "__main__":
    raise SystemExit(main())
