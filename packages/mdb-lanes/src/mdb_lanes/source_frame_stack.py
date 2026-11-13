"""Persistent source scopes beside executable StateCuts.

One active stack walks an immutable Thott/Future Forest graph. Stack checkout
changes source visibility only; the caller's native adapter restores StateCuts.
This is a single-writer experimental control plane, not a second model runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .live_statecut import LiveStateCutStore
from .source_program_surface import (
    SourceProgramArtifact, SourceProgramSelection, ThottSourceProgramRegister,
    compile_source_program_plan,
)
from .spectral_address import (
    CanonicalSymbolKey, SpectralBinding, SpectralFrame, SpectralInstanceKey,
    SpectralLogicalAddress, SpectralQuery, SpectralReadKernelSpec,
    SpectralRoleTransformSpec, SupportResult, read_bindings,
)

SOURCE_FRAME_SCHEMA = "saturn-source-frame-v1"


class SourceFrameError(ValueError):
    """Source scope, state custody, or execution identity cannot be verified."""


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _address(value: Mapping[str, Any]) -> SpectralLogicalAddress:
    return SpectralLogicalAddress(
        CanonicalSymbolKey(**value["canonical_key"]),
        SpectralInstanceKey(**value["instance_key"]),
        SpectralRoleTransformSpec(**value["role_transform"]),
        SpectralFrame(**value["frame"]),
    )


@dataclass(frozen=True)
class FrameSourceBinding:
    """A spectral name bound to a native source coalition in a Thott store."""
    address: SpectralLogicalAddress
    store_id: str
    source_addresses: tuple[str, ...] = ()


class SourceFrameStack:
    """Push, inspect, resolve, and retain sibling scopes with verified cuts.

Each frame seals its own StateCut, local exports, resolution/evidence metadata,
and parent. Sources inherit through Thott; lexical exports inherit through the
frame chain. Identical full spectral names shadow in the nearest frame.
"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cuts = LiveStateCutStore(self.root / "statecuts")
        thott = self.root / "thott"
        self.register = (ThottSourceProgramRegister.reopen(thott) if thott.exists()
                         else ThottSourceProgramRegister(thott, forest_id="source-frame-stack"))
        self.cursor_path = self.root / "active.json"
        self.current = None
        if self.cursor_path.exists():
            cursor = json.loads(self.cursor_path.read_text())
            if cursor.get("schema") != SOURCE_FRAME_SCHEMA:
                raise SourceFrameError("frame cursor schema changed")
            self.current = cursor["current"]
            if self.current is not None:
                self.inspect(self.current)

    def _checkout(self, node: str | None) -> None:
        if node is not None:
            self.inspect(node)
        raw = _json({"schema": SOURCE_FRAME_SCHEMA, "current": node})
        from .bench_publication import stage_write
        if stage_write(self.cursor_path, raw, immutable=False):
            self.current = node
            return
        descriptor, temporary = tempfile.mkstemp(dir=self.root, prefix=".active-")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cursor_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.current = node

    def inspect(self, node: str | None = None) -> dict[str, Any]:
        node = node or self.current
        if node is None:
            raise SourceFrameError("source frame stack is empty")
        native = self.register.forest.verify_node(node)
        if native.get("metadata", {}).get("kind") != SOURCE_FRAME_SCHEMA:
            raise SourceFrameError("node is not a source frame")
        state = self.register.forest.effective_state(node)
        raw = self.register.forest.get_blob(state["frame-stack/current"])
        body = json.loads(raw)
        digest = body.pop("fingerprint")
        if body.get("schema") != SOURCE_FRAME_SCHEMA or hashlib.sha256(_json(body)).hexdigest() != digest:
            raise SourceFrameError("source frame seal changed")
        cut = self.cuts.query(body["statecut_ref"])
        if cut["model_identity"] != body["model_identity"]:
            raise SourceFrameError("frame and StateCut model identities disagree")
        # A detached metadata reference cannot silently point at an unrelated cut.
        return {**body, "fingerprint": digest, "node_id": node}

    def chain(self, node: str | None = None) -> list[dict[str, Any]]:
        cursor = node or self.current
        rows, seen = [], set()
        while cursor is not None:
            if cursor in seen:
                raise SourceFrameError("frame parent cycle")
            seen.add(cursor)
            frame = self.inspect(cursor)
            rows.append(frame)
            cursor = frame["parent_frame"]
        return rows

    def push(
        self, *, label: str, statecut_ref: str,
        artifacts: Mapping[str, SourceProgramArtifact] | None = None,
        bindings: Sequence[FrameSourceBinding] = (),
        resolution: Mapping[str, Any] | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> str:
        if not label:
            raise SourceFrameError("frame label is empty")
        cut = self.cuts.verify_or_raise(statecut_ref).cut
        parent = self.current
        inventory: dict[str, SourceProgramArtifact] = {}
        if parent is not None:
            previous = self.inspect(parent)
            if previous["model_identity"] != dict(cut.model_identity):
                raise SourceFrameError("child StateCut has a different model identity")
            if cut.ref != previous["statecut_ref"] and cut.parent_cut != previous["statecut_ref"]:
                raise SourceFrameError("child StateCut is not bound to its stack parent")
            inventory, _, _ = self.register.restore_bundle(parent)
        for store, artifact in (artifacts or {}).items():
            if store != artifact.surface.store_id:
                raise SourceFrameError("source store label mismatch")
            if store in inventory and inventory[store].surface.fingerprint != artifact.surface.fingerprint:
                raise SourceFrameError("immutable source store changed; use a new store id")
            if artifact.surface.model_id != cut.model_identity.get("model_id"):
                raise SourceFrameError("source artifact and StateCut model identities disagree")
            if artifact.surface.revision != cut.model_identity.get("revision"):
                raise SourceFrameError("source artifact and StateCut revisions disagree")
            inventory[store] = artifact
        if not inventory:
            raise SourceFrameError("root frame requires native source artifacts")
        exports, names = [], set()
        for binding in bindings:
            if binding.address.fingerprint in names:
                raise SourceFrameError("duplicate spectral export in one frame")
            names.add(binding.address.fingerprint)
            artifact = inventory.get(binding.store_id)
            if artifact is None:
                raise SourceFrameError("spectral export names a missing store")
            selected = binding.source_addresses or artifact.surface.addresses
            if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(artifact.surface.addresses):
                raise SourceFrameError("spectral export names invalid source records")
            exports.append({"address": asdict(binding.address),
                            "address_fingerprint": binding.address.fingerprint,
                            "store_id": binding.store_id,
                            "surface_fingerprint": artifact.surface.fingerprint,
                            "source_addresses": list(selected)})
        body = {"schema": SOURCE_FRAME_SCHEMA, "label": label, "parent_frame": parent,
                "statecut_ref": cut.ref, "model_identity": dict(cut.model_identity),
                "bindings": exports, "resolution": dict(resolution or {}),
                "evidence": dict(evidence or {})}
        body["fingerprint"] = hashlib.sha256(_json(body)).hexdigest()
        snapshot = self.register.capture(inventory, label=label, parent_id=parent)
        ref = self.register.forest.put_json_object(body, media_type="application/vnd.saturn.source-frame+json")
        node = self.register.forest.fork(snapshot.node_id, transition={"kind": "push-source-frame"},
                    diff_refs={"frame-stack/current": ref}, metadata={"kind": SOURCE_FRAME_SCHEMA, "label": label})
        self._checkout(node)
        return node

    def pop(self) -> str | None:
        if self.current is None:
            raise SourceFrameError("cannot pop an empty frame stack")
        parent = self.inspect()["parent_frame"]
        self._checkout(parent)
        return parent

    def checkout(self, node: str) -> dict[str, Any]:
        """Select a retained source scope. Native state restoration is explicit."""
        self._checkout(node)
        return self.inspect()

    def visible_bindings(self, node: str | None = None) -> list[dict[str, Any]]:
        visible: dict[str, dict[str, Any]] = {}
        for frame in self.chain(node):
            for binding in frame["bindings"]:
                visible.setdefault(binding["address_fingerprint"], {**binding, "owner_frame": frame["node_id"]})
        return list(visible.values())

    def resolve(
        self, query: SpectralQuery, *, node: str | None = None,
        policy: Any | None = None, query_state: Any | None = None,
        candidate_states: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve exact SAS names, or let a supplied SourceAddressPolicy propose.

Policy features are supplied, recipient-local states; no semantics are inferred
from the integer keys. Exact identity validation follows policy nomination.
"""
        node = node or self.current
        frame = self.inspect(node)
        candidates = self.visible_bindings(node)
        policy_receipt = None
        if policy is not None:
            import torch
            if query_state is None or candidate_states is None or query_state.shape[0] != 1:
                raise SourceFrameError("policy nomination requires one query and bound candidate features")
            if policy.model_identity != frame["model_identity"].get("model_id") or policy.tokenizer_identity != frame["model_identity"].get("tokenizer_sha256"):
                raise SourceFrameError("SourceAddressPolicy identity disagrees with frame")
            if not candidates:
                return {"abstained": True, "reason": "no-visible-source", "frame": node}
            keys = torch.stack([candidate_states[row["address_fingerprint"]] for row in candidates])[None].to(query_state.device)
            selection = policy.select(query_state, keys, torch.ones(keys.shape[:2], dtype=torch.bool, device=keys.device),
                                      torch.arange(len(candidates), device=keys.device)[None])
            policy_receipt = {"policy_fingerprint": hashlib.sha256(_json(policy.manifest())).hexdigest(),
                              "confidence": float(selection["confidence"][0].detach()),
                              "abstain_probability": float(selection["abstain_probability"][0].detach())}
            if bool(selection["abstained"][0]):
                return {"abstained": True, "reason": "source-address-policy-abstained", "frame": node, "policy": policy_receipt}
            # Nomination cannot hide a conflicting exact binding in the scope.
            exact_scope = self.resolve(query, node=node)
            if exact_scope["abstained"]:
                return {**exact_scope, "policy": policy_receipt}
            candidates = [candidates[int(selection["selected_candidate_indices"][0])]]
        bindings = []
        by_ref = {}
        for index, candidate in enumerate(candidates):
            address = _address(candidate["address"])
            if address.fingerprint != candidate["address_fingerprint"]:
                raise SourceFrameError("spectral address fingerprint changed")
            reference = f"source-frame:{node}:{index}"
            bindings.append(SpectralBinding(reference, "source-coalition", ((address.role_transform.role_id, address),), reference, address.frame))
            by_ref[reference] = candidate
        if query.exact_role_id is None:
            raise SourceFrameError("frame source reads require an explicit role")
        result = read_bindings(query, bindings,
                    SpectralReadKernelSpec("frame-source-exact-v1", metric="exact_identity"),
                    SupportResult(True, True, True, True, True, True, True, True, True))
        receipt = {"schema": "saturn-source-frame-resolution-v1", "frame": node,
                   "query": asdict(query), "read_trace": asdict(result.trace), "policy": policy_receipt,
                   "abstained": result.selected_binding is None, "reason": result.abstention_reason}
        if result.selected_binding is not None:
            receipt["selection"] = by_ref[result.selected_binding.payload_reference]
        return receipt

    def materialize_plan(self, resolution: Mapping[str, Any]):
        if resolution.get("abstained") or "selection" not in resolution:
            raise SourceFrameError("cannot materialize an abstained source read")
        node = resolution["frame"]
        expected = self.resolve(SpectralQuery(**resolution["query"]), node=node)
        if expected.get("selection") != resolution["selection"]:
            raise SourceFrameError("source selection does not match sealed frame scope")
        selected = resolution["selection"]
        artifacts, _, _ = self.register.restore_bundle(node)
        artifact = artifacts[selected["store_id"]]
        if artifact.surface.fingerprint != selected["surface_fingerprint"]:
            raise SourceFrameError("selected native source content changed")
        plan = compile_source_program_plan((artifact.surface,),
                tuple(SourceProgramSelection(store_id=selected["store_id"], address=address)
                      for address in selected["source_addresses"]),
                resolver="source-frame-sas-exact-v1", query=resolution["query"])
        return {selected["store_id"]: artifact}, plan

    def restore_statecut(self, node: str | None = None):
        """Return verified executable slots; the native adapter hydrates them."""
        return self.cuts.reopen(self.inspect(node)["statecut_ref"])

    def debug_view(self, node: str | None = None) -> dict[str, Any]:
        chain = self.chain(node)
        return {"schema": SOURCE_FRAME_SCHEMA, "active": self.current,
                "frames": chain, "visible_sources": self.visible_bindings(node),
                "scope_depth": len(chain), "native_restore": "explicit-statecut-adapter"}
