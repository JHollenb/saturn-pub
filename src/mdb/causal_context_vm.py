"""Saturn VM context execution with durable cuts and shared KV residency.

The graph cell binds a JSON request. The native consumer always executes its
Model OS transaction, even if a caller reuses request-binding artifacts. Native
outputs are never replayed as cache hits. One service serializes device work
across sessions; the budget covers resident KV capacity, not model weights,
temporary staging, monitoring tensors, or the device allocator's reserved pool.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from mdb.ar.causal_context_qwen import CausalContextQwenRuntime
from mdb.causal_radix_store import CausalRadixStore
from mdb.circuit_incremental import CircuitBuildGraph, CircuitCell
from mdb.circuit_runtime import CircuitExecutionResult, CircuitRuntime, DirectoryArtifactStore
from mdb.model_os_causal_context import (
    CONTEXT_ADDRESS, NEXT_ADDRESS, RECEIPT_ADDRESS, TOKEN_ADDRESS,
    CausalContextQwenModelBackend, build_causal_context_package,
)
from mdb.model_os_contracts import content_digest
from mdb.model_os_kernel import ModelOSExecutionError, ModelOSKernel, ModelOSStateCut
from mdb.vm_cache import VMCacheEmitter, VMCacheEntry, VMCachePointer
from mdb.vm_payload_cache import PayloadRef


CHECKPOINT_SCHEMA = "saturn-causal-context-vm-checkpoint-v1"
CHECKPOINT_SCHEMA_V2 = "saturn-causal-context-vm-checkpoint-v2"
CHECKPOINT_SCHEMA_V3 = "saturn-causal-context-vm-checkpoint-v3"
REQUEST_SCHEMA = "saturn-causal-context-vm-request-v1"


class CausalContextVMError(ValueError):
    """A context request, durable closure, or residency admission is invalid."""


@dataclass
class _Session:
    session_id: str
    mount_sha256: str
    page_id: str
    extra_rows: int
    backend: CausalContextQwenModelBackend | None = None
    kernel: ModelOSKernel | None = None
    checkpoint_ref: str | None = None
    receipt_ref: str | None = None
    pinned: bool = False
    cuts: dict[str, str] = field(default_factory=dict)


class CausalContextExecutor:
    """Pure request-binding cell; the attached native consumer owns execution."""

    def __call__(self, *, cell, inputs, lease):
        request = inputs["external"]
        if (cell.executor_key != "causal-context" or not isinstance(request, Mapping)
                or request.get("schema") != REQUEST_SCHEMA):
            raise CausalContextVMError("invalid causal context cell request")
        return dict(request)


class _ContextConsumer:
    def __init__(self, vm: CausalContextVM, session: _Session, request: dict, verifier) -> None:
        self.vm, self.session, self.request, self.verifier = vm, session, request, verifier
        self.identity = content_digest({
            "consumer": "saturn-vm-causal-context-model-os-v1",
            "package": vm.package.fingerprint,
        })
        self._output: dict | None = None

    def __call__(self, outputs):
        if outputs != {"context": self.request}:
            raise CausalContextVMError("context graph changed the bound request")
        kernel, backend = self.session.kernel, self.session.backend
        if kernel is None or backend is None:
            raise CausalContextVMError("session is not resident")
        if kernel.current_statecut.fingerprint != self.request["parent_fingerprint"]:
            raise CausalContextVMError("context request has a stale parent")
        result = kernel.execute(
            updates={TOKEN_ADDRESS: self.request["token_id"]}, horizon=1,
            metadata={"vm_request_fingerprint": content_digest(self.request)},
            verifier=self.verifier,
        )
        receipt_ref = self.vm.artifacts.put(result.receipt.to_payload())
        self.session.receipt_ref = receipt_ref
        if result.status == "committed":
            # Instrument tensors are not retained in serving residency or bundles.
            backend.consume_private_output()
        checkpoint_ref = self.vm._checkpoint(self.session)
        state = backend.current_mount
        self._output = {
            "session_id": self.session.session_id,
            "status": result.status,
            "next_token_id": result.receipt.continuation["next_token_id"],
            "output_visible": result.status == "committed",
            "receipt_ref": receipt_ref,
            "receipt_fingerprint": result.receipt.fingerprint,
            "checkpoint_ref": checkpoint_ref,
            "parent_fingerprint": self.request["parent_fingerprint"],
            "statecut_fingerprint": kernel.current_statecut.fingerprint,
            "mount_sha256": state.mount_sha256,
            "page_id": state.page_id,
            "node_sha256s": list(state.node_sha256s),
            "logical_rows": state.length,
        }
        return self._output

    def vm_cache_records(self, output):
        """Called by CircuitRuntime after this exact native execution."""
        if self._output is None or output != self._output:
            raise CausalContextVMError("VM records require the completed native output")
        mount = VMCachePointer.from_artifact_ref(
            f"causal-mount://sha256/{output['mount_sha256']}", role="causal-context-mount",
            sha256=output["mount_sha256"],
            metadata={"hash_semantics": "canonical-mount-body", "page_id": output["page_id"]},
        )
        checkpoint = VMCachePointer.from_artifact_ref(
            output["checkpoint_ref"], role="durable-context-checkpoint",
        )
        receipt = VMCachePointer.from_artifact_ref(
            output["receipt_ref"], role="model-os-transaction-receipt",
        )
        entries = {
            "source": (VMCacheEntry("context-mount", pointers=(mount, checkpoint),
                                    metadata={"mode": "exact-history"}),),
            "address": (VMCacheEntry("context-address", pointers=(mount,), metadata={
                "address": CONTEXT_ADDRESS, "node_sha256s": output["node_sha256s"],
                "logical_rows": output["logical_rows"],
            }),),
            "operator": (VMCacheEntry("context-transaction", pointers=(receipt,), metadata={
                "numerical_contract": self.vm.package.model_abi.numerical_contract,
                "parent_fingerprint": output["parent_fingerprint"],
                "statecut_fingerprint": output["statecut_fingerprint"],
                "status": output["status"], "native_forward_units": 1,
            }),),
            "renderer": (VMCacheEntry("context-output", pointers=(receipt,), metadata={
                "next_token_id": output["next_token_id"],
                "output_visible": output["output_visible"],
            }),),
        }
        return {
            "entries": {plane: [row.to_dict() for row in rows] for plane, rows in entries.items()},
            "metadata": {
                "session_id": output["session_id"], "mount_sha256": output["mount_sha256"],
                "receipt_ref": output["receipt_ref"],
                "receipt_fingerprint": output["receipt_fingerprint"],
                "checkpoint_ref": output["checkpoint_ref"],
                "native_forward_units": 1, "residency": self.vm.residency(),
            },
            "output_pointer": receipt.to_dict(),
        }


class CausalContextVM:
    """One resident model, multiple transactional sessions, durable local backing.

    Use a persistent directory outside the mrun lease. ``import_mount`` copies
    its complete verified closure from an existing store. Committed steps and
    eviction write visible private K/V and metadata to immutable CAS objects.
    A checkpoint reference can be resumed by a fresh process with the same ABI.
    """

    def __init__(self, model: Any, root: str | Path, *, model_abi_sha256: str,
                 model_id: str, tokenizer_identity: str, resident_kv_bytes: int,
                 backend: str = "triton", backend_factory=CausalContextQwenModelBackend,
                 checkpoint_page_bytes: int | None = None) -> None:
        if isinstance(resident_kv_bytes, bool) or not isinstance(resident_kv_bytes, int) or resident_kv_bytes <= 0:
            raise CausalContextVMError("resident KV budget must be a positive integer")
        self.root = Path(root).expanduser().resolve()
        self.store = CausalRadixStore(self.root / "radix")
        self.runtime = CausalContextQwenRuntime(
            model, self.store, model_abi_sha256=model_abi_sha256, backend=backend,
        )
        if checkpoint_page_bytes is not None and (type(checkpoint_page_bytes) is not int or checkpoint_page_bytes <= 0):
            raise CausalContextVMError("checkpoint page bytes must be positive")
        self.checkpoint_page_rows = max(1, checkpoint_page_bytes//self.runtime.codec.row_bytes) if checkpoint_page_bytes else None
        self.checkpoint_stats = {"encoded_bytes": 0, "reused_bytes": 0, "encoded_pages": 0, "reused_pages": 0}
        self.package = build_causal_context_package(
            self.runtime, model_id=model_id, tokenizer_identity=tokenizer_identity,
        )
        self.artifacts = DirectoryArtifactStore(self.root / "artifacts")
        self.circuit = CircuitRuntime(
            {"causal-context": CausalContextExecutor()}, artifact_store=self.artifacts,
            vm_cache_emitter=VMCacheEmitter(self.root / "vm-cache"),
        )
        self.resident_kv_bytes = resident_kv_bytes
        self.backend_factory = backend_factory
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = RLock()
        self.evictions = 0
        self.memory_events: list[dict] = []

    def import_mount(self, source: CausalRadixStore, mount_sha256: str) -> str:
        with self._lock:
            return source.export_mount(mount_sha256, self.store)

    def _session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise CausalContextVMError("unknown context session") from exc

    def residency(self) -> dict[str, int]:
        with self._lock:
            ancestors, deltas = {}, {}
            resident = 0
            for session in self._sessions.values():
                if session.backend is None:
                    continue
                resident += 1
                for state in session.backend.retained_mounts():
                    for pair in state.ancestor:
                        for tensor in pair:
                            ancestors[tensor.data_ptr()] = tensor.numel() * tensor.element_size()
                    for pair in state.delta:
                        for tensor in pair:
                            deltas[tensor.data_ptr()] = tensor.numel() * tensor.element_size()
            return {
                "resident_sessions": resident, "known_sessions": len(self._sessions),
                "ancestor_bytes": sum(ancestors.values()),
                "private_capacity_bytes": sum(deltas.values()),
                "total_kv_bytes": sum(ancestors.values()) + sum(deltas.values()),
                "budget_bytes": self.resident_kv_bytes, "evictions": self.evictions,
            }

    def allocation_map(self) -> list[dict]:
        """Unique physical allocations with all retained cut/session owners."""
        with self._lock:
            allocations = {}
            for session in self._sessions.values():
                if session.backend is None:
                    continue
                for state in session.backend.retained_mounts():
                    for kind, layers in (("shared-ancestor", state.ancestor), ("private", state.delta)):
                        for layer, pair in enumerate(layers):
                            for component, tensor in zip(("key", "value"), pair):
                                row = allocations.setdefault(tensor.data_ptr(), {
                                    "allocation_id": f"resident:{tensor.data_ptr()}",
                                    "kind": kind, "layer": layer, "component": component,
                                    "capacity_bytes": tensor.numel()*tensor.element_size(),
                                    "resident": True, "owners": [], "visible_rows": []})
                                if session.session_id not in row["owners"]:
                                    row["owners"].append(session.session_id)
                                rows = state.ancestor_rows if kind == "shared-ancestor" else state.delta_rows
                                if rows not in row["visible_rows"]:
                                    row["visible_rows"].append(rows)
            return list(allocations.values())

    def _prune_ancestors(self) -> None:
        self.runtime.evict_unused_ancestors({
            state.node_sha256s[0] for session in self._sessions.values()
            if session.backend is not None for state in session.backend.retained_mounts()
        })

    def _admit(self, additional: Callable[[], int], *, protected: str) -> None:
        while self.residency()["total_kv_bytes"] + additional() > self.resident_kv_bytes:
            victim = next((s for s in self._sessions.values()
                           if s.session_id != protected and s.backend is not None and not s.pinned), None)
            if victim is None:
                raise CausalContextVMError("resident KV budget exhausted by protected or pinned cuts")
            self._checkpoint(victim)
            self.memory_events.append({"kind": "eviction", "session": victim.session_id,
                                       "checkpoint_ref": victim.checkpoint_ref})
            victim.kernel = victim.backend = None
            self.evictions += 1
            self._prune_ancestors()

    def _place(self, session: _Session, checkpoint: dict | None = None) -> None:
        read = self.store.read(session.mount_sha256, session.page_id,
                               model_abi_sha256=self.runtime.model_abi_sha256)
        if len(read.segments) not in {1, 2}:
            raise CausalContextVMError("context consumer requires an ancestor and optional delta")
        row_bytes = self.runtime.codec.row_bytes
        if any(len(segment) % row_bytes for segment in read.segments):
            raise CausalContextVMError("mount contains partial model K/V rows")
        initial_rows = len(read.segments[1]) // row_bytes if len(read.segments) == 2 else 0
        capacity = initial_rows + session.extra_rows
        if len(read.segments[0]) + capacity * row_bytes > self.resident_kv_bytes:
            raise CausalContextVMError("resident KV budget exhausted by one session")
        raw_delta = None
        if checkpoint is not None:
            if (checkpoint["ancestor_rows"] != len(read.segments[0]) // row_bytes
                    or checkpoint["delta_capacity"] != capacity
                    or not initial_rows <= checkpoint["delta_rows"] <= capacity):
                raise CausalContextVMError("checkpoint row geometry changed")
            raw_delta = self._delta_bytes(checkpoint, consumer="context-checkpoint-resume")
        ancestor_id = read.node_sha256s[0]
        def additional():
            return capacity * row_bytes + (
                0 if self.runtime.has_resident_ancestor(ancestor_id) else len(read.segments[0])
            )
        self._admit(additional, protected=session.session_id)
        try:
            state = self.runtime.mount(session.mount_sha256, session.page_id,
                                       extra_rows=session.extra_rows)
            if raw_delta is not None:
                restored = self.runtime.codec.decode_segment(raw_delta, self.runtime.device) if raw_delta else ()
                for placed, saved in zip(state.delta, restored):
                    for target, source in zip(placed, saved):
                        target[:, :, :checkpoint["delta_rows"], :].copy_(source)
                state.delta_rows = checkpoint["delta_rows"]
            initial_cut = None
            if checkpoint is not None:
                cut_body = dict(checkpoint["statecut"])
                fingerprint = cut_body.pop("content_sha256")
                schema = cut_body.pop("schema")
                initial_cut = ModelOSStateCut(**cut_body)
                if (schema != "saturn-model-os-state-cut-v1"
                        or initial_cut.fingerprint != fingerprint):
                    raise CausalContextVMError("checkpoint StateCut identity changed")
            session.backend = self.backend_factory(
                self.runtime, state, self.package, initial_cut=initial_cut,
            )
            session.kernel = ModelOSKernel.install(self.package, session.backend)
            self.memory_events.append({"kind": "hydration" if checkpoint else "mount",
                                       "session": session.session_id, "checkpoint_ref": session.checkpoint_ref,
                                       "allocated_private_bytes": capacity*row_bytes,
                                       "ancestor_rows": state.ancestor_rows})
        except Exception:
            session.backend = session.kernel = None
            self._prune_ancestors()
            raise

    def open_session(self, session_id: str, mount_sha256: str, page_id: str, *, extra_rows: int = 32,
                     pinned: bool = False) -> ModelOSStateCut:
        with self._lock:
            if not isinstance(session_id, str) or not session_id or session_id in self._sessions:
                raise CausalContextVMError("session id must be nonempty and unique")
            if isinstance(extra_rows, bool) or not isinstance(extra_rows, int) or extra_rows <= 0:
                raise CausalContextVMError("private continuation capacity must be positive")
            session = _Session(session_id, mount_sha256, page_id, extra_rows, pinned=pinned)
            self._place(session)
            self._sessions[session_id] = session
            try:
                self._checkpoint(session)
            except Exception:
                del self._sessions[session_id]
                session.kernel = session.backend = None
                self._prune_ancestors()
                raise
            return session.kernel.current_statecut

    def _checkpoint(self, session: _Session, cut: ModelOSStateCut | None = None) -> str:
        if cut is not None and cut.fingerprint in session.cuts:
            reference = session.cuts[cut.fingerprint]
            self._load_checkpoint(reference)
            return reference
        if session.backend is None:
            if cut is not None:
                raise CausalContextVMError("cut is not retained by this session")
            if session.checkpoint_ref is None:
                raise CausalContextVMError("session has no durable cut")
            return session.checkpoint_ref
        selected = session.backend.resolve_open_cut(cut or session.kernel.current_statecut)
        state = session.backend.mount_for_cut(selected)
        old = self._load_checkpoint(session.checkpoint_ref) if session.checkpoint_ref else None
        cached = session.cuts.get(selected.fingerprint)
        if cut is None and cached:
            body = self._load_checkpoint(cached)
            if body["receipt_ref"] == session.receipt_ref:
                session.checkpoint_ref = cached
                return cached
        delta = {}
        if self.checkpoint_page_rows:
            pages = []
            reusable = (old and old.get("schema") == CHECKPOINT_SCHEMA_V3
                and old["page_rows"] == self.checkpoint_page_rows
                and old["mount_sha256"] == state.mount_sha256 and old["page_id"] == state.page_id
                and old["delta_rows"] <= state.delta_rows
                and (old["statecut"]["handle"] == selected.parent_handle
                     or old["statecut"]["content_sha256"] == selected.fingerprint))
            previous = {p["start_row"]: p for p in old["delta_pages"]} if reusable else {}
            for start in range(0, state.delta_rows, self.checkpoint_page_rows):
                count = min(self.checkpoint_page_rows, state.delta_rows-start)
                page = previous.get(start)
                if page and page["rows"] == count:
                    pages.append(page)
                    self.checkpoint_stats["reused_pages"] += 1
                    self.checkpoint_stats["reused_bytes"] += count*self.runtime.codec.row_bytes
                    continue
                layers = tuple(tuple(t[:, :, start:start+count, :] for t in pair) for pair in state.delta)
                raw = self.runtime.codec.encode_layers(layers, count)
                ref = self.store.payloads.put_bytes(raw, role="causal-radix-segment")
                pages.append({"start_row": start, "rows": count, "payload_ref": ref.to_dict()})
                self.checkpoint_stats["encoded_pages"] += 1
                self.checkpoint_stats["encoded_bytes"] += len(raw)
            delta = {"delta_pages": pages, "page_rows": self.checkpoint_page_rows}
        else:
            raw = self.runtime.codec.encode_layers(state.delta, state.delta_rows)
        # A visible initial delta may be byte-identical to a published radix
        # segment. This CAS binds one descriptor per byte digest, so retain
        # the common row role and put checkpoint semantics in its manifest.
            ref = self.store.payloads.put_bytes(raw, role="causal-radix-segment")
            delta = {"delta_ref": ref.to_dict()}
        body = {
            "schema": CHECKPOINT_SCHEMA_V3 if self.checkpoint_page_rows else CHECKPOINT_SCHEMA_V2 if state.delta_rows == 0 else CHECKPOINT_SCHEMA,
            "model_abi_sha256": self.runtime.model_abi_sha256,
            "package_fingerprint": self.package.fingerprint,
            "mount_sha256": state.mount_sha256, "page_id": state.page_id,
            "ancestor_rows": state.ancestor_rows, "delta_rows": state.delta_rows,
            "delta_capacity": state.delta_capacity, "extra_rows": session.extra_rows,
            **delta, "statecut": selected.to_payload(),
            "receipt_ref": session.receipt_ref,
        }
        reference = self.artifacts.put(body)
        session.cuts[selected.fingerprint] = reference
        if cut is None:
            session.checkpoint_ref = reference
        return reference

    def checkpoint(self, session_id: str, *, cut: ModelOSStateCut | None = None) -> str:
        with self._lock:
            return self._checkpoint(self._session(session_id), cut)

    def _load_checkpoint(self, reference: str) -> dict:
        body = self.artifacts.get(reference)
        if (content_digest(body) != reference.removeprefix("cas://sha256/")
                or body.get("schema") not in {CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA_V2, CHECKPOINT_SCHEMA_V3}
                or body.get("model_abi_sha256") != self.runtime.model_abi_sha256
                or body.get("package_fingerprint") != self.package.fingerprint):
            raise CausalContextVMError("checkpoint identity or model ABI changed")
        for key in ("ancestor_rows", "delta_rows", "delta_capacity", "extra_rows"):
            minimum = 0 if key == "delta_rows" and body["schema"] in {CHECKPOINT_SCHEMA_V2, CHECKPOINT_SCHEMA_V3} else 1
            if isinstance(body.get(key), bool) or not isinstance(body.get(key), int) or body[key] < minimum:
                raise CausalContextVMError("checkpoint has invalid row geometry")
        if body["receipt_ref"] is not None:
            receipt = self.artifacts.get(body["receipt_ref"])
            if content_digest(receipt) != body["receipt_ref"].removeprefix("cas://sha256/"):
                raise CausalContextVMError("checkpoint transaction receipt changed")
        if body["schema"] == CHECKPOINT_SCHEMA_V3:
            if type(body.get("page_rows")) is not int or body["page_rows"] <= 0:
                raise CausalContextVMError("checkpoint page geometry changed")
            start = 0
            for page in body["delta_pages"]:
                ref = PayloadRef.from_dict(page["payload_ref"])
                if (type(page["start_row"]) is not int or page["start_row"] != start
                    or type(page["rows"]) is not int or not 0 < page["rows"] <= body["page_rows"]
                    or page["rows"] != min(body["page_rows"], body["delta_rows"]-start)
                    or ref.bytes != page["rows"]*self.runtime.codec.row_bytes):
                    raise CausalContextVMError("checkpoint page extent changed")
                start += page["rows"]
            if start != body["delta_rows"]: raise CausalContextVMError("checkpoint page closure incomplete")
        return body

    def _delta_bytes(self, body, *, consumer):
        refs = [p["payload_ref"] for p in body["delta_pages"]] if body["schema"] == CHECKPOINT_SCHEMA_V3 else [body["delta_ref"]]
        raw = b"".join(self.store.payloads.hydrate(PayloadRef.from_dict(ref), native_consumer=consumer).payload for ref in refs)
        if len(raw) != body["delta_rows"]*self.runtime.codec.row_bytes:
            raise CausalContextVMError("checkpoint private byte count changed")
        return raw

    def resume_session(self, session_id: str, checkpoint_ref: str, *, pinned: bool = False) -> ModelOSStateCut:
        with self._lock:
            if not isinstance(session_id, str) or not session_id or session_id in self._sessions:
                raise CausalContextVMError("session id must be nonempty and unique")
            body = self._load_checkpoint(checkpoint_ref)
            session = _Session(session_id, body["mount_sha256"], body["page_id"], body["extra_rows"],
                               checkpoint_ref=checkpoint_ref, receipt_ref=body["receipt_ref"], pinned=pinned)
            self._place(session, body)
            self._sessions[session_id] = session
            session.cuts[session.kernel.current_statecut.fingerprint] = checkpoint_ref
            return session.kernel.current_statecut

    def _resident(self, session: _Session) -> None:
        if session.backend is None:
            self._place(session, self._load_checkpoint(session.checkpoint_ref))
        self._sessions.move_to_end(session.session_id)

    def statecut(self, session_id: str) -> ModelOSStateCut:
        with self._lock:
            session = self._session(session_id)
            self._resident(session)
            return session.kernel.current_statecut

    def restore(self, session_id: str, cut: ModelOSStateCut) -> Mapping[str, Any]:
        with self._lock:
            session = self._session(session_id)
            self._resident(session)
            try:
                result = session.kernel.restore(cut)
            except ModelOSExecutionError:
                reference = session.cuts.get(cut.fingerprint)
                if reference is None:
                    raise CausalContextVMError("cut is not retained by this session")
                body = self._load_checkpoint(reference)
                session.kernel = session.backend = None
                self._prune_ancestors()
                self._place(session, body)
                result = {"status": "restored", "current_handle": session.kernel.current_statecut.handle,
                          "logical_rows": session.backend.current_mount.length}
            self._checkpoint(session)
            return result

    def execute(self, session_id: str, token_id: int, *, run_id: str, verifier=None) -> CircuitExecutionResult:
        with self._lock:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < self.runtime.model.config.vocab_size:
                raise CausalContextVMError("token must be an integer in the native vocabulary")
            if not isinstance(run_id, str) or not run_id.strip():
                raise CausalContextVMError("run id must be nonempty")
            session = self._session(session_id)
            self._resident(session)
            parent = session.kernel.current_statecut
            if session.backend.current_mount.delta_rows >= session.backend.current_mount.delta_capacity:
                raise CausalContextVMError("private continuation capacity exhausted")
            self._admit(lambda: session.backend.preview_allocation_bytes(parent), protected=session_id)
            request = {
                "schema": REQUEST_SCHEMA, "session_id": session_id, "token_id": token_id,
                "parent_fingerprint": parent.fingerprint, "run_id": run_id,
            }
            consumer = _ContextConsumer(self, session, request, verifier)
            graph = CircuitBuildGraph(
                graph_id="saturn-vm-causal-context-v1", skeleton_fingerprint=content_digest(REQUEST_SCHEMA),
                specimen_fingerprint=self.package.model_abi.fingerprint,
                cells=(CircuitCell("context", "context-request", executor_key="causal-context",
                                   reads=(CONTEXT_ADDRESS, TOKEN_ADDRESS),
                                   writes=(NEXT_ADDRESS, RECEIPT_ADDRESS)),), outputs=("context",),
                numerical_contract=self.package.model_abi.numerical_contract,
                native_consumer_identity=consumer.identity,
                native_consumer_contract="model-os-causal-context-one-token-v1",
                metadata={"native_forward_units": 1, "cell_work": "request-binding-only"},
            )
            old_checkpoint, old_receipt = session.checkpoint_ref, session.receipt_ref
            try:
                return self.circuit.full_build(
                    graph, cell_inputs={"context": request}, native_consumer=consumer,
                    native_consumer_identity=consumer.identity,
                    vm_cache_emission={
                        "run_id": run_id, "model_identity": {"model_id": self.package.model_abi.model_id},
                        "input_identity": {"request_fingerprint": content_digest(request)},
                        "abi_identity": {"fingerprint": self.package.model_abi.fingerprint},
                        "entries": {plane: () for plane in ("source", "address", "operator", "renderer")},
                    },
                )
            except Exception:
                # A durable-write or bundle failure cannot leave an unpublished
                # child selected. Immutable candidate evidence remains on disk.
                session.kernel.restore(parent)
                session.checkpoint_ref, session.receipt_ref = old_checkpoint, old_receipt
                raise

    def close_session(self, session_id: str) -> str:
        with self._lock:
            session = self._session(session_id)
            reference = self._checkpoint(session)
            del self._sessions[session_id]
            session.kernel = session.backend = None
            self._prune_ancestors()
            return reference

    def export_session(self, session_id: str, destination: str | Path) -> str:
        """Copy the selected cut, its mount, private bytes, and latest receipt."""
        with self._lock:
            reference = self._checkpoint(self._session(session_id))
            body = self._load_checkpoint(reference)
            target = Path(destination).expanduser().resolve()
            store = CausalRadixStore(target / "radix")
            self.store.export_mount(body["mount_sha256"], store)
            refs = [p["payload_ref"] for p in body["delta_pages"]] if body["schema"] == CHECKPOINT_SCHEMA_V3 else [body["delta_ref"]]
            for item in refs:
                ref = PayloadRef.from_dict(item)
                raw = self.store.payloads.hydrate(ref, native_consumer="context-session-export").payload
                if store.payloads.put_bytes(raw, role=ref.role).to_dict() != ref.to_dict():
                    raise CausalContextVMError("session export changed private byte identity")
            artifacts = DirectoryArtifactStore(target / "artifacts")
            if body["receipt_ref"] is not None:
                if artifacts.put(self.artifacts.get(body["receipt_ref"])) != body["receipt_ref"]:
                    raise CausalContextVMError("session export changed transaction identity")
            if artifacts.put(body) != reference:
                raise CausalContextVMError("session export changed checkpoint identity")
            return reference
