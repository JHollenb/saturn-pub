"""Typed learned source-address policies and SourceWrite v2 composition.

The policy is deliberately separate from the payload.  It selects one live
candidate row (or abstains); ``SourceWriteActV2`` decides how that selected
state is converted into a recipient-local write.  This makes address selection,
source necessity, and consumer closure independently replayable claims.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mdb.ar.capability_source_write import (
    SOURCE_WRITE_ACT_V2_SCHEMA,
    SourceWriteActV2,
    load_source_write_act,
)
from mdb.titan import ActSpec

SOURCE_ADDRESS_POLICY_SCHEMA = "saturn-source-address-policy-v1"
ADDRESSED_SOURCE_WRITE_SCHEMA = "saturn-addressed-source-write-program-v1"
SOURCE_ADDRESS_POLICY_NAME = "ar.source-address-policy"
POSITION_FEATURE_MODES = frozenset({"none", "relative_log_distance_v1"})


class SourceAddressPolicyError(ValueError):
    """Raised when a policy, candidate surface, or package fails closed."""


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
        raise SourceAddressPolicyError("source-address value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().cpu().float().contiguous().numpy()
    raw = array.astype("<f4", copy=False).tobytes()
    return {
        "encoding": "little-endian-float32-gzip-base64",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_base64": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
    }


def _tensor_from_payload(payload: Mapping[str, Any]) -> torch.Tensor:
    if payload.get("encoding") != "little-endian-float32-gzip-base64":
        raise SourceAddressPolicyError("unrecognized source-address tensor encoding")
    try:
        shape = tuple(int(value) for value in payload["shape"])
        raw = gzip.decompress(base64.b64decode(str(payload["gzip_base64"])))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise SourceAddressPolicyError("malformed source-address tensor payload") from exc
    if not shape or any(value < 0 for value in shape):
        raise SourceAddressPolicyError("invalid source-address tensor shape")
    if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
        raise SourceAddressPolicyError("source-address tensor content hash mismatch")
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != math.prod(shape):
        raise SourceAddressPolicyError("source-address tensor geometry mismatch")
    return torch.from_numpy(values.copy()).reshape(shape)


def source_address_policy_spec(*, hidden_size: int, layer: int) -> ActSpec:
    if hidden_size < 1 or layer < -1:
        raise SourceAddressPolicyError("source-address ABI dimensions are invalid")
    prefix = "model.token_embedding" if layer == -1 else f"model.layer{layer}"
    return ActSpec(
        name=SOURCE_ADDRESS_POLICY_NAME,
        abi_version=f"v1.layer{layer}.d{hidden_size}",
        reads=(
            f"{prefix}.query.state",
            f"{prefix}.candidate-source.address-keys",
            f"{prefix}.candidate-source.mask",
            f"{prefix}.candidate-source.positions",
        ),
        writes=(
            f"{prefix}.selected-source.position",
            f"{prefix}.selected-source.confidence",
            f"{prefix}.selected-source.abstained",
        ),
        numerical_contract="finite-shared-candidate-scorer-with-explicit-abstention",
        tolerance={"maximum_absolute_nonfinite": 0.0},
        write_kinds={
            f"{prefix}.selected-source.position": "tensor",
            f"{prefix}.selected-source.confidence": "tensor",
            f"{prefix}.selected-source.abstained": "tensor",
        },
        write_abis={
            f"{prefix}.selected-source.position": {
                "dtype": "int64",
                "shape": ["batch"],
                "semantics": "absolute token row; -1 means abstain",
            },
            f"{prefix}.selected-source.confidence": {
                "dtype": "float32",
                "shape": ["batch"],
                "range": [0.0, 1.0],
            },
            f"{prefix}.selected-source.abstained": {
                "dtype": "bool",
                "shape": ["batch"],
            },
        },
    )


class SourceAddressPolicy(nn.Module):
    """A shared candidate scorer with calibrated explicit abstention.

    The scorer is permutation equivariant over candidate address keys.  The
    key need not be the payload state: a key/value task can expose a local
    predecessor or another recipient-native routing key while SourceWrite
    fetches the selected payload row.  It cannot learn an
    absolute source index in its default mode.  The explicitly versioned
    relative-geometry mode receives only differences among candidate and query
    coordinates; candidate order and absolute positions are otherwise used
    only to return the selected live row.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        bottleneck: int,
        layer: int,
        model_identity: str,
        tokenizer_identity: str,
        seed: int,
        abstain_threshold: float = 0.5,
        position_feature_mode: str = "none",
    ) -> None:
        super().__init__()
        if hidden_size < 1 or bottleneck < 1 or layer < -1 or seed < 0:
            raise SourceAddressPolicyError("source-address dimensions are invalid")
        if not model_identity or not tokenizer_identity:
            raise SourceAddressPolicyError("source-address policy requires bound identities")
        threshold = float(abstain_threshold)
        if not 0.0 <= threshold <= 1.0 or not math.isfinite(threshold):
            raise SourceAddressPolicyError("abstain threshold must be in [0, 1]")
        mode = str(position_feature_mode)
        if mode not in POSITION_FEATURE_MODES:
            raise SourceAddressPolicyError("source-address position feature mode is unsupported")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.layer = int(layer)
        self.model_identity = str(model_identity)
        self.tokenizer_identity = str(tokenizer_identity)
        self.seed = int(seed)
        self.abstain_threshold = threshold
        self.position_feature_mode = mode
        self.query_projection = nn.Linear(hidden_size, bottleneck, bias=False)
        self.candidate_projection = nn.Linear(hidden_size, bottleneck, bias=False)
        # This path is deliberately relative-only. It exposes neither absolute
        # token position nor token identity: each live candidate receives its
        # distance to the declared query normalized by the live candidate span.
        # Keeping the module absent for ``none`` preserves legacy package
        # state dictionaries and manifests byte-for-byte.
        self.position_projection = (
            nn.Linear(2, bottleneck, bias=False)
            if mode == "relative_log_distance_v1"
            else None
        )
        self.interaction = nn.Linear(3 * bottleneck, bottleneck, bias=True)
        self.score = nn.Linear(bottleneck, 1, bias=False)
        self.abstain = nn.Linear(hidden_size, 1, bias=True)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        nn.init.xavier_uniform_(self.query_projection.weight, generator=generator)
        nn.init.xavier_uniform_(self.candidate_projection.weight, generator=generator)
        if self.position_projection is not None:
            nn.init.zeros_(self.position_projection.weight)
        nn.init.xavier_uniform_(self.interaction.weight, generator=generator)
        nn.init.zeros_(self.interaction.bias)
        nn.init.xavier_uniform_(self.score.weight, generator=generator)
        nn.init.zeros_(self.abstain.weight)
        nn.init.zeros_(self.abstain.bias)

    @property
    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @property
    def act_spec(self) -> ActSpec:
        return source_address_policy_spec(hidden_size=self.hidden_size, layer=self.layer)

    def assert_compatible(
        self,
        *,
        model_identity: str,
        tokenizer_identity: str,
        hidden_size: int,
        layer: int,
    ) -> None:
        """Fail closed before applying a package to a resident recipient."""

        observed = (str(model_identity), str(tokenizer_identity), int(hidden_size), int(layer))
        expected = (
            self.model_identity,
            self.tokenizer_identity,
            self.hidden_size,
            self.layer,
        )
        if observed != expected:
            raise SourceAddressPolicyError(
                "source-address policy is bound to another model/tokenizer ABI"
            )

    def _validate(
        self,
        query_state: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_positions: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
    ) -> None:
        if query_state.ndim != 2 or candidate_states.ndim != 3 or candidate_mask.ndim != 2:
            raise SourceAddressPolicyError("source-address inputs have invalid ranks")
        batch, candidates, width = candidate_states.shape
        if (
            query_state.shape != (batch, width)
            or candidate_mask.shape != (batch, candidates)
            or width != self.hidden_size
        ):
            raise SourceAddressPolicyError("source-address input geometry changed")
        if candidate_positions is not None and candidate_positions.shape != (batch, candidates):
            raise SourceAddressPolicyError("source-address positions do not align")
        if candidate_positions is not None and candidate_positions.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise SourceAddressPolicyError("source-address positions must be integral")
        if query_positions is not None and (
            query_positions.shape != (batch,)
            or query_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise SourceAddressPolicyError("source-address query positions do not align")
        if self.position_feature_mode != "none" and (
            candidate_positions is None or query_positions is None
        ):
            raise SourceAddressPolicyError(
                "relative source-address policy requires declared query and candidate positions"
            )

    def _relative_position_features(
        self,
        candidate_positions: torch.Tensor,
        query_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded, recipient-derived relative source/query geometry."""

        if self.position_feature_mode != "relative_log_distance_v1":
            raise SourceAddressPolicyError("relative source-address features are unavailable")
        query = query_positions.to(device=candidate_positions.device, dtype=torch.float32)[:, None]
        candidates = candidate_positions.to(device=candidate_positions.device, dtype=torch.float32)
        distance = query - candidates
        valid = candidate_mask.to(device=candidate_positions.device, dtype=torch.bool)
        if valid.any() and not bool((distance[valid] > 0.0).all().item()):
            raise SourceAddressPolicyError(
                "relative source-address candidate is not strictly before its query"
            )
        safe_distance = distance.clamp_min(1.0)
        # Normalize only against other live *relative* distances.  Dividing
        # by the absolute query coordinate would quietly break translation
        # invariance and leak an absolute-position proxy into the ABI.
        span = (safe_distance * valid.to(dtype=safe_distance.dtype)).amax(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        features = torch.stack(
            (
                safe_distance / span,
                torch.log1p(safe_distance) / torch.log1p(span),
            ),
            dim=-1,
        )
        # Padding coordinates are not inputs to the scorer.  Clearing them
        # avoids accidental dependence on how a caller happened to fill a
        # masked position tensor.
        return torch.where(valid[:, :, None], features, torch.zeros_like(features))

    def forward(
        self,
        query_state: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        candidate_positions: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(
            query_state,
            candidate_states,
            candidate_mask,
            candidate_positions,
            query_positions,
        )
        query = F.normalize(query_state.float(), dim=-1, eps=1.0e-6)
        candidates = F.normalize(candidate_states.float(), dim=-1, eps=1.0e-6)
        candidate_features = self.candidate_projection(candidates)
        if self.position_projection is not None:
            if candidate_positions is None or query_positions is None:
                raise SourceAddressPolicyError("relative source-address positions disappeared")
            candidate_features = candidate_features + self.position_projection(
                self._relative_position_features(
                    candidate_positions,
                    query_positions,
                    candidate_mask,
                )
            )
        query_features = self.query_projection(query)[:, None, :].expand_as(candidate_features)
        features = torch.cat(
            (query_features, candidate_features, query_features * candidate_features), dim=-1
        )
        scores = self.score(F.gelu(self.interaction(features), approximate="tanh")).squeeze(-1)
        mask = candidate_mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))
        abstain_logit = self.abstain(query).squeeze(-1)
        if not bool(torch.isfinite(abstain_logit).all().item()):
            raise SourceAddressPolicyError("source-address policy emitted non-finite abstention")
        valid_rows = mask.any(dim=1)
        finite_candidate = torch.isfinite(scores[valid_rows]).any(dim=1)
        if valid_rows.any() and not bool(finite_candidate.all().item()):
            raise SourceAddressPolicyError("source-address policy emitted no finite candidate")
        return scores, abstain_logit

    def select(
        self,
        query_state: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_positions: torch.Tensor,
        *,
        threshold: float | None = None,
        query_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate(
            query_state,
            candidate_states,
            candidate_mask,
            candidate_positions,
            query_positions,
        )
        scores, abstain_logit = self(
            query_state,
            candidate_states,
            candidate_mask,
            candidate_positions=candidate_positions,
            query_positions=query_positions,
        )
        mask = candidate_mask.to(device=scores.device, dtype=torch.bool)
        valid = mask.any(dim=1)
        safe_scores = torch.where(valid[:, None], scores, torch.zeros_like(scores))
        probabilities = torch.softmax(
            torch.cat((safe_scores, abstain_logit[:, None]), dim=1), dim=1
        )
        candidate_probabilities = probabilities[:, :-1] * mask.float()
        normalizer = candidate_probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        conditional = candidate_probabilities / normalizer
        confidence, candidate_index = conditional.max(dim=1)
        abstain_probability = probabilities[:, -1]
        use_threshold = self.abstain_threshold if threshold is None else float(threshold)
        if not 0.0 <= use_threshold <= 1.0 or not math.isfinite(use_threshold):
            raise SourceAddressPolicyError("selection threshold must be in [0, 1]")
        abstained = (
            (~valid)
            | (confidence < use_threshold)
            | (abstain_probability >= candidate_probabilities.max(dim=1).values)
        )
        selected = candidate_positions.to(device=scores.device, dtype=torch.long).gather(
            1, candidate_index[:, None]
        )[:, 0]
        selected = torch.where(abstained, torch.full_like(selected, -1), selected)
        return {
            "selected_positions": selected,
            "selected_candidate_indices": candidate_index,
            "confidence": confidence,
            "abstain_probability": abstain_probability,
            "abstained": abstained,
            "candidate_probabilities": conditional,
            "candidate_scores": scores,
        }

    def manifest(self) -> dict[str, Any]:
        state = {
            name: {"shape": list(value.shape), "sha256": _tensor_payload(value)["sha256"]}
            for name, value in sorted(self.state_dict().items())
        }
        body = {
            "schema": SOURCE_ADDRESS_POLICY_SCHEMA,
            "architecture": {
                "kind": "permutation-equivariant-query-candidate-interaction-scorer",
                "hidden_size": self.hidden_size,
                "bottleneck": self.bottleneck,
                "layer": self.layer,
                "parameter_count": self.parameter_count,
                "absolute_position_features": False,
                "explicit_abstention": True,
                **(
                    {
                        "position_feature_mode": self.position_feature_mode,
                        "relative_position_features": [
                            "normalized_source_query_distance",
                            "normalized_log_source_query_distance",
                        ],
                    }
                    if self.position_feature_mode != "none"
                    else {}
                ),
            },
            "binding": {
                "model": self.model_identity,
                "tokenizer": self.tokenizer_identity,
            },
            "seed": self.seed,
            "abstain_threshold": self.abstain_threshold,
            "act_spec": self.act_spec.to_dict(),
            "state": state,
            "serving_forbidden_inputs": [
                "gold_answer",
                "answer_id",
                "oracle_source_position",
                "teacher_or_donor_logits",
                "donor_runtime_state",
                "prompt_identity_lookup",
            ],
        }
        return {**body, "fingerprint": _digest(body)}

    def to_payload(self, *, training_receipt: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema": SOURCE_ADDRESS_POLICY_SCHEMA,
            "manifest": self.manifest(),
            "state": {
                name: _tensor_payload(value) for name, value in sorted(self.state_dict().items())
            },
            "training_receipt": dict(training_receipt),
        }
        return {**body, "content_sha256": _digest(body)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SourceAddressPolicy:
        if payload.get("schema") != SOURCE_ADDRESS_POLICY_SCHEMA:
            raise SourceAddressPolicyError("unrecognized source-address policy artifact")
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload.get("content_sha256") != _digest(body):
            raise SourceAddressPolicyError("source-address policy fingerprint changed")
        manifest = payload.get("manifest")
        state_payload = payload.get("state")
        if not isinstance(manifest, Mapping) or not isinstance(state_payload, Mapping):
            raise SourceAddressPolicyError("source-address policy artifact is incomplete")
        architecture = manifest.get("architecture")
        binding = manifest.get("binding")
        if not isinstance(architecture, Mapping) or not isinstance(binding, Mapping):
            raise SourceAddressPolicyError("source-address policy manifest is incomplete")
        policy = cls(
            hidden_size=int(architecture["hidden_size"]),
            bottleneck=int(architecture["bottleneck"]),
            layer=int(architecture["layer"]),
            model_identity=str(binding["model"]),
            tokenizer_identity=str(binding["tokenizer"]),
            seed=int(manifest["seed"]),
            abstain_threshold=float(manifest["abstain_threshold"]),
            position_feature_mode=str(architecture.get("position_feature_mode", "none")),
        )
        expected = set(policy.state_dict())
        if set(state_payload) != expected:
            raise SourceAddressPolicyError("source-address policy state inventory changed")
        policy.load_state_dict(
            {name: _tensor_from_payload(state_payload[name]) for name in expected}, strict=True
        )
        if policy.manifest() != manifest:
            raise SourceAddressPolicyError("source-address policy manifest changed after hydration")
        return policy


class AddressedSourceWriteProgram(nn.Module):
    """Compose discrete address selection with a source-necessary v2 payload."""

    def __init__(self, policy: SourceAddressPolicy, act: SourceWriteActV2) -> None:
        super().__init__()
        if policy.hidden_size != act.hidden_size or policy.layer != act.layer:
            raise SourceAddressPolicyError("policy and SourceWrite v2 ABI do not align")
        self.policy = policy
        self.act = act

    def forward(
        self,
        query_state: torch.Tensor,
        candidate_address_keys: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_positions: torch.Tensor,
        native_write: torch.Tensor,
        *,
        candidate_payload_states: torch.Tensor | None = None,
        dose: float = 1.0,
        threshold: float | None = None,
        query_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if native_write.shape != query_state.shape:
            raise SourceAddressPolicyError("native write does not align with query state")
        payload_states = (
            candidate_address_keys if candidate_payload_states is None else candidate_payload_states
        )
        if payload_states.shape != candidate_address_keys.shape:
            raise SourceAddressPolicyError(
                "candidate payload states do not align with address keys"
            )
        selection = self.policy.select(
            query_state,
            candidate_address_keys,
            candidate_mask,
            candidate_positions,
            threshold=threshold,
            query_positions=query_positions,
        )
        row = torch.arange(query_state.shape[0], device=query_state.device)
        source = payload_states[row, selection["selected_candidate_indices"]]
        source = torch.where(selection["abstained"][:, None], torch.zeros_like(source), source)
        delta = self.act(query_state, source, native_write)
        dose_value = float(dose)
        if not math.isfinite(dose_value):
            raise SourceAddressPolicyError("addressed SourceWrite dose is non-finite")
        patched = native_write + dose_value * delta
        return {**selection, "selected_source_state": source, "delta": delta, "patched": patched}

    def to_payload(self, *, training_receipt: Mapping[str, Any]) -> dict[str, Any]:
        policy_payload = self.policy.to_payload(training_receipt=training_receipt)
        act_payload = self.act.to_payload(training_receipt=training_receipt)
        body = {
            "schema": ADDRESSED_SOURCE_WRITE_SCHEMA,
            "policy": policy_payload,
            "source_write_v2": act_payload,
            "binding": {
                "policy_fingerprint": policy_payload["manifest"]["fingerprint"],
                "act_fingerprint": act_payload["manifest"]["fingerprint"],
                "hidden_size": self.policy.hidden_size,
                "layer": self.policy.layer,
            },
            "serving_custody": {
                "oracle_source_position": False,
                "gold_answer": False,
                "donor_model": False,
                "donor_runtime_state": False,
                "optimizer": False,
            },
        }
        return {**body, "content_sha256": _digest(body)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AddressedSourceWriteProgram:
        if payload.get("schema") != ADDRESSED_SOURCE_WRITE_SCHEMA:
            raise SourceAddressPolicyError("unrecognized addressed SourceWrite package")
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if payload.get("content_sha256") != _digest(body):
            raise SourceAddressPolicyError("addressed SourceWrite package fingerprint changed")
        policy_payload = payload.get("policy")
        act_payload = payload.get("source_write_v2")
        if not isinstance(policy_payload, Mapping) or not isinstance(act_payload, Mapping):
            raise SourceAddressPolicyError("addressed SourceWrite package is incomplete")
        policy = SourceAddressPolicy.from_payload(policy_payload)
        act = load_source_write_act(act_payload)
        if (
            not isinstance(act, SourceWriteActV2)
            or act_payload.get("schema") != SOURCE_WRITE_ACT_V2_SCHEMA
        ):
            raise SourceAddressPolicyError(
                "addressed SourceWrite requires a v2 source-necessary Act"
            )
        program = cls(policy, act)
        binding = payload.get("binding")
        if not isinstance(binding, Mapping) or (
            binding.get("policy_fingerprint") != policy.manifest()["fingerprint"]
            or binding.get("act_fingerprint") != act.manifest()["fingerprint"]
        ):
            raise SourceAddressPolicyError("addressed SourceWrite binding changed")
        return program


__all__ = [
    "ADDRESSED_SOURCE_WRITE_SCHEMA",
    "SOURCE_ADDRESS_POLICY_NAME",
    "SOURCE_ADDRESS_POLICY_SCHEMA",
    "POSITION_FEATURE_MODES",
    "AddressedSourceWriteProgram",
    "SourceAddressPolicy",
    "SourceAddressPolicyError",
    "source_address_policy_spec",
]
