"""Content-addressed persistent forests of speculative model futures.

A future node stores ancestry, typed transition metadata, observations, and
references to changed payloads.  Unchanged model state is inherited by reference.
Promotion changes a small, hash-bound head pointer; rejected branches remain
immutable and queryable.  The store is intentionally model-family-neutral: tensor
chunking, KV paging, and Act serialization live behind opaque content references.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

FOREST_SCHEMA = "saturn-persistent-future-forest-v1"
NODE_SCHEMA = "saturn-persistent-future-node-v1"
OBJECT_REF_SCHEMA = "saturn-future-object-reference-v1"
OBSERVATION_SCHEMA = "saturn-future-observation-v1"
DECISION_SCHEMA = "saturn-future-decision-v1"
HEAD_SCHEMA = "saturn-future-head-v1"

Decision = Literal["promote", "prune", "retain"]
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CLOSURE_SNAPSHOT = ContextVar("mdb_forest_closure_snapshot", default=None)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_MEDIA_TYPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*/[A-Za-z0-9][A-Za-z0-9.+_-]*$")


class FutureForestError(RuntimeError):
    """Raised when forest identity, ancestry, or object custody fails closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_digest(value: Any, *, label: str) -> str:
    value = str(value)
    if _DIGEST_RE.fullmatch(value) is None:
        raise FutureForestError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: Any, *, label: str) -> str:
    value = str(value)
    if _NAME_RE.fullmatch(value) is None or ".." in value:
        raise FutureForestError(f"{label} has an invalid identifier")
    return value


def _require_media_type(value: Any) -> str:
    value = str(value)
    if _MEDIA_TYPE_RE.fullmatch(value) is None:
        raise FutureForestError("media_type must be a concrete type/subtype identifier")
    return value


def _jsonable(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FutureForestError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = _require_name(key, label=f"{label} key")
            result[name] = _jsonable(item, label=f"{label}.{name}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _jsonable(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise FutureForestError(f"{label} is not JSON-like: {type(value).__name__}")


class PersistentFutureForest:
    """No-overwrite content store and ancestry graph for speculative futures."""

    def __init__(self, root: Path, *, forest_id: str) -> None:
        self.root = Path(root)
        self.forest_id = _require_name(forest_id, label="forest_id")
        self.objects = self.root / "objects"
        self.nodes = self.root / "nodes"
        self.observations = self.root / "observations"
        self.decisions = self.root / "decisions"
        self.heads = self.root / "heads"
        for directory in (
            self.objects,
            self.nodes,
            self.observations,
            self.decisions,
            self.heads,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": FOREST_SCHEMA,
            "forest_id": self.forest_id,
            "storage": "content-addressed-parent-plus-diff",
        }
        self._publish_json(self.root / "manifest.json", manifest)

    @classmethod
    def reopen(cls, root: Path) -> PersistentFutureForest:
        path = Path(root) / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FutureForestError(f"cannot reopen future forest manifest: {path}") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema") != FOREST_SCHEMA:
            raise FutureForestError("future forest manifest schema changed")
        forest = cls(Path(root), forest_id=str(manifest.get("forest_id")))
        if forest._read_json(path) != dict(manifest):
            raise FutureForestError("future forest manifest changed during reopen")
        return forest

    @staticmethod
    def _publish_bytes(path: Path, payload: bytes) -> None:
        from .bench_publication import stage_write
        if stage_write(path, payload): return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise FutureForestError(f"immutable content path contains different bytes: {path}")
            return
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise FutureForestError(
                        f"concurrent immutable publication disagreed: {path}"
                    ) from None
            except OSError as exc:
                # Some mrun work volumes (notably NTFS-backed mounts) permit an
                # adjacent temporary file and atomic rename but reject hard
                # links with EPERM/EOPNOTSUPP.  Retain the no-overwrite contract
                # by serializing the fallback under a per-path advisory lock,
                # rechecking the destination, and atomically publishing the
                # already-fsynced temporary file within the same directory.
                unsupported = {
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                }
                if exc.errno not in unsupported:
                    raise
                lock_root = Path(tempfile.gettempdir()) / "saturn-future-forest-publish-locks"
                lock_root.mkdir(parents=True, exist_ok=True)
                lock_identity = hashlib.sha256(
                    str(path.resolve()).encode("utf-8")
                ).hexdigest()
                lock_path = lock_root / f"{lock_identity}.lock"
                with lock_path.open("a+b") as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    try:
                        if path.exists():
                            if path.read_bytes() != payload:
                                raise FutureForestError(
                                    f"concurrent immutable publication disagreed: {path}"
                                ) from None
                        else:
                            os.replace(temporary, path)
                    finally:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _publish_json(cls, path: Path, payload: Mapping[str, Any]) -> None:
        cls._publish_bytes(path, _canonical(payload) + b"\n")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FutureForestError(f"cannot read future forest object: {path}") from exc
        if not isinstance(payload, Mapping):
            raise FutureForestError(f"future forest JSON root is not an object: {path}")
        return dict(payload)

    @staticmethod
    @contextmanager
    def _exclusive_lock(path: Path) -> Any:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _sealed(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
        content = {"schema": schema, **dict(body)}
        return {**content, "content_sha256": _digest(content)}

    @staticmethod
    def _verify_seal(payload: Mapping[str, Any], *, schema: str, label: str) -> str:
        if payload.get("schema") != schema:
            raise FutureForestError(f"{label} schema changed")
        declared = _require_digest(payload.get("content_sha256"), label=f"{label} hash")
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if _digest(body) != declared:
            raise FutureForestError(f"{label} content hash changed")
        return declared

    def put_blob(
        self,
        payload: bytes,
        *,
        media_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store irreducible changed bytes once and return a typed reference."""

        if not isinstance(payload, bytes):
            raise FutureForestError("future object payload must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        object_path = self.objects / digest[:2] / digest[2:]
        self._publish_bytes(object_path, payload)
        return {
            "schema": OBJECT_REF_SCHEMA,
            "sha256": digest,
            "bytes": len(payload),
            "media_type": _require_media_type(media_type),
            "metadata": _jsonable(metadata or {}, label="object metadata"),
        }

    def put_json_object(
        self,
        value: Mapping[str, Any],
        *,
        media_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.put_blob(
            _canonical(_jsonable(value, label="JSON object")),
            media_type=media_type,
            metadata=metadata,
        )

    def get_blob(self, reference: Mapping[str, Any]) -> bytes:
        if reference.get("schema") != OBJECT_REF_SCHEMA:
            raise FutureForestError("future object reference schema changed")
        digest = _require_digest(reference.get("sha256"), label="object reference hash")
        path = self.objects / digest[:2] / digest[2:]
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise FutureForestError(f"future object is missing: {digest}") from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise FutureForestError(f"future object bytes changed: {digest}")
        if int(reference.get("bytes", -1)) != len(payload):
            raise FutureForestError(f"future object size changed: {digest}")
        return payload

    def _normalize_refs(
        self, refs: Mapping[str, Mapping[str, Any]], *, label: str
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for address, raw in refs.items():
            name = _require_name(address, label=f"{label} address")
            reference = dict(_jsonable(raw, label=f"{label}.{name}"))
            self.get_blob(reference)
            result[name] = reference
        return dict(sorted(result.items()))

    def create_root(
        self,
        *,
        state_refs: Mapping[str, Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> str:
        if self.list_nodes():
            raise FutureForestError("future forest already has a root")
        node = self._sealed(
            NODE_SCHEMA,
            {
                "forest_id": self.forest_id,
                "parents": [],
                "primary_parent": None,
                "transition": {"kind": "root"},
                "diff_refs": self._normalize_refs(state_refs, label="root state"),
                "metadata": _jsonable(metadata, label="root metadata"),
            },
        )
        node_id = str(node["content_sha256"])
        self._publish_json(self.nodes / f"{node_id}.json", node)
        self._write_head("accepted", node_id, prior=None, reason="forest-root")
        return node_id

    def fork(
        self,
        parent_id: str,
        *,
        transition: Mapping[str, Any],
        diff_refs: Mapping[str, Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
        secondary_parents: Sequence[str] = (),
    ) -> str:
        parent_id = self.verify_node(parent_id)["content_sha256"]
        parents = [str(parent_id)]
        for candidate in secondary_parents:
            resolved = self.verify_node(candidate)["content_sha256"]
            if resolved not in parents:
                parents.append(resolved)
        transition_payload = _jsonable(transition, label="transition")
        if not isinstance(transition_payload, Mapping) or not transition_payload.get("kind"):
            raise FutureForestError("future transition requires a kind")
        node = self._sealed(
            NODE_SCHEMA,
            {
                "forest_id": self.forest_id,
                "parents": parents,
                "primary_parent": parent_id,
                "transition": transition_payload,
                "diff_refs": self._normalize_refs(diff_refs, label="node diff"),
                "metadata": _jsonable(metadata or {}, label="node metadata"),
            },
        )
        node_id = str(node["content_sha256"])
        self._publish_json(self.nodes / f"{node_id}.json", node)
        return node_id

    def verify_node(self, node_id: str, *, _visiting: set[str] | None = None) -> dict[str, Any]:
        node_id = _require_digest(node_id, label="node_id")
        snapshot = _CLOSURE_SNAPSHOT.get()
        verified = snapshot[1] if snapshot is not None and snapshot[0] is self else None
        if verified is not None and node_id in verified:
            return verified[node_id]
        node = self._read_json(self.nodes / f"{node_id}.json")
        declared = self._verify_seal(node, schema=NODE_SCHEMA, label="future node")
        if declared != node_id or node.get("forest_id") != self.forest_id:
            raise FutureForestError("future node identity changed")
        visiting = set() if _visiting is None else _visiting
        if node_id in visiting:
            raise FutureForestError("future forest ancestry contains a cycle")
        visiting.add(node_id)
        parents = node.get("parents")
        if not isinstance(parents, list):
            raise FutureForestError("future node parents must be a list")
        primary = node.get("primary_parent")
        if not parents:
            if primary is not None or node.get("transition", {}).get("kind") != "root":
                raise FutureForestError("root future node contract changed")
        elif primary != parents[0]:
            raise FutureForestError("future node primary parent changed")
        for parent in parents:
            self.verify_node(str(parent), _visiting=visiting)
        refs = node.get("diff_refs")
        if not isinstance(refs, Mapping):
            raise FutureForestError("future node diff_refs must be an object")
        self._normalize_refs(refs, label="verified node diff")
        visiting.remove(node_id)
        if verified is not None:
            verified[node_id] = node
        return node

    def effective_state(self, node_id: str) -> dict[str, dict[str, Any]]:
        node = self.verify_node(node_id)
        snapshot = _CLOSURE_SNAPSHOT.get()
        states = snapshot[2] if snapshot is not None and snapshot[0] is self else None
        if states is not None and node_id in states:
            return dict(states[node_id])
        primary = node.get("primary_parent")
        state = {} if primary is None else self.effective_state(str(primary))
        state.update({key: dict(value) for key, value in node["diff_refs"].items()})
        if states is not None:
            states[node_id] = dict(state)
        return dict(sorted(state.items()))

    def observe(
        self,
        node_id: str,
        *,
        stage: str,
        scores: Mapping[str, float],
        evidence_refs: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        node_id = str(self.verify_node(node_id)["content_sha256"])
        score_payload: dict[str, float] = {}
        for name, score in scores.items():
            key = _require_name(name, label="score name")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise FutureForestError(f"score {key} must be numeric")
            value = float(score)
            if not math.isfinite(value):
                raise FutureForestError(f"score {key} must be finite")
            score_payload[key] = value
        refs = []
        for raw in evidence_refs:
            reference = dict(_jsonable(raw, label="evidence reference"))
            self.get_blob(reference)
            refs.append(reference)
        observation = self._sealed(
            OBSERVATION_SCHEMA,
            {
                "forest_id": self.forest_id,
                "node_id": node_id,
                "stage": _require_name(stage, label="observation stage"),
                "scores": dict(sorted(score_payload.items())),
                "evidence_refs": refs,
                "metadata": _jsonable(metadata or {}, label="observation metadata"),
            },
        )
        observation_id = str(observation["content_sha256"])
        self._publish_json(self.observations / f"{observation_id}.json", observation)
        return observation_id

    def verify_observation(self, observation_id: str) -> dict[str, Any]:
        observation_id = _require_digest(observation_id, label="observation_id")
        payload = self._read_json(self.observations / f"{observation_id}.json")
        declared = self._verify_seal(
            payload, schema=OBSERVATION_SCHEMA, label="future observation"
        )
        if declared != observation_id or payload.get("forest_id") != self.forest_id:
            raise FutureForestError("future observation identity changed")
        self.verify_node(str(payload.get("node_id")))
        for reference in payload.get("evidence_refs", ()):
            self.get_blob(reference)
        return payload

    def decide(
        self,
        node_id: str,
        *,
        decision: Decision,
        observation_ids: Sequence[str],
        reason: str,
    ) -> str:
        node_id = str(self.verify_node(node_id)["content_sha256"])
        if decision not in {"promote", "prune", "retain"}:
            raise FutureForestError(f"unsupported future decision {decision!r}")
        observations = []
        for observation_id in observation_ids:
            observation = self.verify_observation(observation_id)
            if observation.get("node_id") != node_id:
                raise FutureForestError("decision observation belongs to another node")
            observations.append(str(observation_id))
        before = self.head("accepted")
        if decision == "promote" and before not in self.ancestors(
            node_id, include_self=True
        ):
            raise FutureForestError(
                "promoted future does not descend from the accepted head"
            )
        after = node_id if decision == "promote" else before
        receipt = self._sealed(
            DECISION_SCHEMA,
            {
                "forest_id": self.forest_id,
                "node_id": node_id,
                "decision": decision,
                "observation_ids": observations,
                "reason": str(reason),
                "accepted_before": before,
                "accepted_after": after,
            },
        )
        decision_id = str(receipt["content_sha256"])
        self._publish_json(self.decisions / f"{decision_id}.json", receipt)
        if decision == "promote":
            self._write_head("accepted", after, prior=before, reason=decision_id)
        return decision_id

    def verify_decision(self, decision_id: str) -> dict[str, Any]:
        decision_id = _require_digest(decision_id, label="decision_id")
        payload = self._read_json(self.decisions / f"{decision_id}.json")
        declared = self._verify_seal(payload, schema=DECISION_SCHEMA, label="future decision")
        if declared != decision_id or payload.get("forest_id") != self.forest_id:
            raise FutureForestError("future decision identity changed")
        self.verify_node(str(payload.get("node_id")))
        for observation_id in payload.get("observation_ids", ()):
            self.verify_observation(str(observation_id))
        return payload

    def _write_head(self, name: str, node_id: str, *, prior: str | None, reason: str) -> None:
        name = _require_name(name, label="head name")
        node_id = str(self.verify_node(node_id)["content_sha256"])
        path = self.heads / f"{name}.json"
        with self._exclusive_lock(self.heads / f".{name}.lock"):
            generation = 0
            if path.exists():
                current = self._read_json(path)
                self._verify_seal(current, schema=HEAD_SCHEMA, label="future head")
                if current.get("node_id") != prior:
                    raise FutureForestError("future head compare-and-swap parent changed")
                generation = int(current.get("generation", -1)) + 1
            elif prior is not None:
                raise FutureForestError("future head has no prior generation")
            head = self._sealed(
                HEAD_SCHEMA,
                {
                    "forest_id": self.forest_id,
                    "name": name,
                    "node_id": node_id,
                    "generation": generation,
                    "prior_node_id": prior,
                    "reason": str(reason),
                },
            )
            payload = _canonical(head) + b"\n"
            from .bench_publication import stage_write
            if stage_write(path, payload, immutable=False): return
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)

    def head(self, name: str = "accepted") -> str:
        name = _require_name(name, label="head name")
        payload = self._read_json(self.heads / f"{name}.json")
        self._verify_seal(payload, schema=HEAD_SCHEMA, label="future head")
        if payload.get("forest_id") != self.forest_id or payload.get("name") != name:
            raise FutureForestError("future head binding changed")
        return str(self.verify_node(str(payload.get("node_id")))["content_sha256"])

    def rewind(self, node_id: str, *, reason: str) -> None:
        before = self.head("accepted")
        node_id = str(self.verify_node(node_id)["content_sha256"])
        if node_id not in self.ancestors(before, include_self=True):
            raise FutureForestError("rewind target is not an ancestor of the accepted head")
        self._write_head("accepted", node_id, prior=before, reason=reason)

    def ancestors(self, node_id: str, *, include_self: bool = False) -> set[str]:
        node = self.verify_node(node_id)
        result = {str(node_id)} if include_self else set()
        for parent in node["parents"]:
            result.add(str(parent))
            result.update(self.ancestors(str(parent), include_self=False))
        return result

    def cherry_pick(
        self,
        source_id: str,
        onto_id: str,
        *,
        addresses: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        source = self.effective_state(source_id)
        selected: dict[str, dict[str, Any]] = {}
        for address in addresses:
            name = _require_name(address, label="cherry-pick address")
            if name not in source:
                raise FutureForestError(f"source future has no address {name}")
            selected[name] = source[name]
        return self.fork(
            onto_id,
            transition={
                "kind": "cherry-pick",
                "source_node_id": source_id,
                "addresses": list(addresses),
            },
            diff_refs=selected,
            metadata=metadata,
            secondary_parents=(source_id,),
        )

    def three_way_merge(
        self,
        *,
        base_id: str,
        left_id: str,
        right_id: str,
        resolutions: Mapping[str, Literal["left", "right"]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        base = self.effective_state(base_id)
        left = self.effective_state(left_id)
        right = self.effective_state(right_id)
        left_ancestors = self.ancestors(left_id, include_self=True)
        right_ancestors = self.ancestors(right_id, include_self=True)
        if base_id not in left_ancestors or base_id not in right_ancestors:
            raise FutureForestError("three-way merge base is not an ancestor of both branches")
        choices = dict(resolutions or {})
        merged = dict(left)
        conflicts: list[str] = []
        for address in sorted(set(base) | set(left) | set(right)):
            base_ref = base.get(address)
            left_ref = left.get(address)
            right_ref = right.get(address)
            if left_ref == right_ref:
                chosen = left_ref
            elif left_ref == base_ref:
                chosen = right_ref
            elif right_ref == base_ref:
                chosen = left_ref
            else:
                resolution = choices.get(address)
                if resolution not in {"left", "right"}:
                    conflicts.append(address)
                    continue
                chosen = left_ref if resolution == "left" else right_ref
            if chosen is None:
                merged.pop(address, None)
            else:
                merged[address] = chosen
        if conflicts:
            raise FutureForestError(f"three-way merge has unresolved addresses: {conflicts}")
        delta = {address: ref for address, ref in merged.items() if left.get(address) != ref}
        return self.fork(
            left_id,
            transition={
                "kind": "three-way-merge",
                "base_node_id": base_id,
                "left_node_id": left_id,
                "right_node_id": right_id,
                "resolutions": choices,
            },
            diff_refs=delta,
            metadata=metadata,
            secondary_parents=(right_id,),
        )

    def list_nodes(self) -> list[str]:
        return sorted(path.stem for path in self.nodes.glob("*.json"))

    def verify_closure(self) -> dict[str, Any]:
        # The caller serializes writers. Verify each immutable node once in this
        # read snapshot rather than rereading every ancestor for every state.
        # Context-local caches never escape this call or survive a later audit.
        token = _CLOSURE_SNAPSHOT.set((self, {}, {}))
        try:
            return self._verify_closure()
        finally:
            _CLOSURE_SNAPSHOT.reset(token)

    def _verify_closure(self) -> dict[str, Any]:
        nodes = self.list_nodes()
        referenced_objects: set[str] = set()
        for node_id in nodes:
            node = self.verify_node(node_id)
            referenced_objects.update(
                str(reference["sha256"]) for reference in node["diff_refs"].values()
            )
        observations = sorted(path.stem for path in self.observations.glob("*.json"))
        for observation_id in observations:
            observation = self.verify_observation(observation_id)
            referenced_objects.update(
                str(reference["sha256"])
                for reference in observation["evidence_refs"]
            )
        decisions = sorted(path.stem for path in self.decisions.glob("*.json"))
        for decision_id in decisions:
            self.verify_decision(decision_id)
        object_paths = [path for path in self.objects.glob("*/*") if path.is_file()]
        physical_objects = {path.parent.name + path.name for path in object_paths}
        if physical_objects != referenced_objects:
            raise FutureForestError(
                "future forest object closure has missing or orphaned objects: "
                f"missing={sorted(referenced_objects - physical_objects)} "
                f"orphaned={sorted(physical_objects - referenced_objects)}"
            )
        logical_refs = sum(len(self.effective_state(node_id)) for node_id in nodes)
        physical_bytes = sum(path.stat().st_size for path in object_paths)
        return {
            "schema": "saturn-persistent-future-forest-closure-v1",
            "forest_id": self.forest_id,
            "nodes": len(nodes),
            "observations": len(observations),
            "decisions": len(decisions),
            "content_objects": len(object_paths),
            "logical_effective_state_references": logical_refs,
            "physical_content_bytes": physical_bytes,
            "accepted_head": self.head("accepted"),
            "all_dependency_closures_verified": True,
            "no_orphan_content_objects": True,
        }


__all__ = [
    "DECISION_SCHEMA",
    "FOREST_SCHEMA",
    "FutureForestError",
    "NODE_SCHEMA",
    "OBJECT_REF_SCHEMA",
    "OBSERVATION_SCHEMA",
    "PersistentFutureForest",
]
