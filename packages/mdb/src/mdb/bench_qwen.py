"""Experimental single-writer open bench over native Qwen and causal-context VM.

The bench owns selection and durable references. The VM owns residency, the
backend owns tensors, and Model OS remains token admission/commit authority.
"""
from __future__ import annotations

import fcntl
import json
import shutil
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from mdb.ar.causal_context_qwen import QwenTokenExecution
from mdb.ar.program_record import ProgramRecorder
from mdb.bench_metrics import BenchMetrics
from mdb.bench_publication import BenchPublication, recover_locked, writer_path
from mdb.causal_context_vm import CausalContextVM
from mdb.circuit_runtime import DirectoryArtifactStore
from mdb.future_forest import PersistentFutureForest
from mdb.model_os_causal_context import TOKEN_ADDRESS, CausalContextQwenModelBackend
from mdb.model_os_contracts import content_digest
from mdb.model_os_kernel import ModelOSExecutionError
from mdb.vm_payload_cache import PayloadRef

BENCH_SCHEMA = "saturn-open-bench-v1"
BENCH_CUT_SCHEMA = "saturn-open-bench-cut-v1"


from .errors import BenchError


def tensor_digest(tensor: torch.Tensor) -> str:
    return sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            self._check()
            outer = self._serialization_depth == 0
            batch = (outer and method.__name__ in {"step", "generate"}
                     and self._optimized_capture())
            self._serialization_depth += 1
            try:
                with self.publication.transaction():
                    with self._command_publication() if batch else nullcontext():
                        result = method(self, *args, **kwargs)
            finally:
                self._serialization_depth -= 1
            if outer and method.__name__ in {"step", "generate", "restore", "fork", "checkout",
                                   "configure_cot_register", "commit_edit", "abort_edit", "close"}:
                self._render_cot_panel()
            return result
    return call


class BenchQwenBackend(CausalContextQwenModelBackend):
    """Layer-steppable provisional state behind the existing Model OS kernel."""

    execution: QwenTokenExecution | None = None
    prepared: Any = None
    last_logits: torch.Tensor | None = None

    def begin(self, kernel, token: int):
        if self.execution is not None:
            return self.execution
        plan = kernel.plan(horizon=1)
        candidate, preview = super().preview(kernel.current_statecut, plan, {TOKEN_ADDRESS: token})
        self.prepared = candidate, preview, kernel.current_statecut.fingerprint
        self.execution = self.runtime.begin_token(self._states[candidate.handle], token)
        return self.execution

    def preview(self, parent, plan, updates):
        if self.prepared is None:
            return super().preview(parent, plan, updates)
        candidate, preview, fingerprint = self.prepared
        if parent.fingerprint != fingerprint or updates != {TOKEN_ADDRESS: self.execution.token_id}:
            raise ModelOSExecutionError("prepared token belongs to another parent/input")
        if set(plan.writes) != {TOKEN_ADDRESS}:
            raise ModelOSExecutionError("prepared token requires the admitted write set")
        return candidate, preview

    def continue_execution(self, candidate, *, consumer, horizon):
        if self.execution is None:
            return super().continue_execution(candidate, consumer=consumer, horizon=horizon)
        if self.prepared[0].handle != candidate.handle or horizon != 1:
            raise ModelOSExecutionError("prepared execution identity changed")
        while self.execution.next_layer < self.runtime.codec.layers:
            self.runtime.advance_token(self.execution)
        logits = self.execution.logits
        if logits is None:
            logits = self.runtime.finish_token(self.execution)
        if not bool(torch.isfinite(logits).all()):
            raise ModelOSExecutionError("non-finite readout cannot commit a token")
        self.execution.state.delta_rows += 1
        self._outputs[candidate.handle] = logits
        self._traces[candidate.handle] = {"layers": list(range(self.runtime.codec.layers))}
        self._next_tokens[candidate.handle] = int(logits.argmax(-1).item())
        return {"accepted": True, "consumer_id": consumer.consumer_id,
                "next_token_id": self._next_tokens[candidate.handle],
                "logits_sha256": tensor_digest(logits),
                "logical_rows_after": self.execution.state.length,
                "native_layers_consumed": self.runtime.codec.layers}

    def consume_private_output(self):
        output = super().consume_private_output()
        self.last_logits = output[0]
        self.execution = self.prepared = None
        return output

    def abort(self, candidate, *, reason):
        result = super().abort(candidate, reason=reason)
        self.execution = self.prepared = None
        return result

    def cancel_execution(self):
        if self.prepared is not None:
            self.abort(self.prepared[0], reason="bench provisional token discarded")


@dataclass
class BenchBranch:
    name: str
    pending: int
    token_ids: list[int]
    prompt_length: int
    cut: str | None = None
    node: str | None = None
    symbols: dict[str, str] = field(default_factory=dict)
    intervention_lineage: list[str] = field(default_factory=list)


class QwenBenchSession:
    """One resident model, one cut registry, many independently owned futures.

    Use from_resident inside the caller's mrun worker. It performs no model
    load or scheduler submission. All mutations are serialized and capability
    scoped; references and summaries are the public control-plane currency.
    """

    @classmethod
    def from_resident(cls, model, tokenizer=None, *, root, model_id="local-qwen2",
                      revision="local", model_sha256=None, session_id="bench",
                      lane="virtual", attention_backend=None, resident_kv_bytes=64 << 20,
                      continuation_capacity=128, publication_mode="journal"):
        if (Path(root)/"selected.json").exists():
            raise BenchError("session already exists; use BenchSession.resume")
        return cls(model, tokenizer, root=root, model_id=model_id, revision=revision,
                   model_sha256=model_sha256, session_id=session_id, lane=lane,
                   attention_backend=attention_backend, resident_kv_bytes=resident_kv_bytes,
                   continuation_capacity=continuation_capacity, publication_mode=publication_mode)

    def __init__(self, model, tokenizer, *, root, model_id, revision, model_sha256,
                 session_id, lane, attention_backend, resident_kv_bytes, continuation_capacity,
                 publication_mode="journal", _writer=None):
        if lane not in {"virtual", "native"}:
            raise BenchError("lane must be virtual or native")
        if isinstance(continuation_capacity, bool) or not isinstance(continuation_capacity, int) or continuation_capacity <= 0:
            raise BenchError("continuation capacity must be positive")
        self.model, self.tokenizer = model, tokenizer
        self.family = str(getattr(model.config, "model_type", ""))
        if self.family not in {"qwen2", "qwen3"}:
            raise BenchError("native Qwen bench supports only qwen2/qwen3")
        if next(model.parameters()).dtype != torch.bfloat16:
            raise BenchError("Qwen stateful debugging requires BF16; use model.to(torch.bfloat16) before opening the session")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._writer = _writer or writer_path(self.root).open("a+")
        try:
            fcntl.flock(self._writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._writer.close()
            raise BenchError("another writer owns this session") from None
        try:
            self._initialize_resident(model, tokenizer, model_id=model_id, revision=revision,
                model_sha256=model_sha256, session_id=session_id, lane=lane, attention_backend=attention_backend,
                resident_kv_bytes=resident_kv_bytes, continuation_capacity=continuation_capacity,
                publication_mode=publication_mode)
        except BaseException:
            self._writer.close()
            raise

    def _initialize_resident(self, model, tokenizer, *, model_id, revision, model_sha256,
                             session_id, lane, attention_backend, resident_kv_bytes,
                             continuation_capacity, publication_mode):
        self.publication = BenchPublication(self.root, mode=publication_mode)
        self.metrics = BenchMetrics()
        self.session_id, self.lane = session_id, lane
        self.capacity = continuation_capacity
        if model_sha256 is None:
            digest = sha256()
            for name, parameter in model.named_parameters():
                digest.update(name.encode())
                digest.update(bytes.fromhex(tensor_digest(parameter)))
            model_sha256 = digest.hexdigest()
        tokenizer_hash = content_digest(tokenizer.get_vocab()) if tokenizer is not None else "token-ids-only"
        self.identity = {"model_id": model_id, "revision": revision, "weights_sha256": model_sha256,
                         "config_sha256": content_digest(json.loads(json.dumps(model.config.to_dict()))),
                         "tokenizer_sha256": tokenizer_hash, "dtype": str(next(model.parameters()).dtype)}
        self.model_abi = content_digest(self.identity)
        attention_backend = ("native-eager" if lane == "native" else
                             attention_backend or ("triton" if next(model.parameters()).is_cuda else "torch"))
        with self.publication.initializing():
            self.vm = CausalContextVM(model, self.root / "vm", model_abi_sha256=self.model_abi,
                model_id=model_id, tokenizer_identity=tokenizer_hash, resident_kv_bytes=resident_kv_bytes,
                backend=attention_backend, backend_factory=BenchQwenBackend, checkpoint_page_bytes=64 << 10)
            self.artifacts = DirectoryArtifactStore(self.root / "artifacts")
            self.forest = PersistentFutureForest(self.root / "forest", forest_id=session_id)
        record_path = self.root / "programs.json"
        self.recorder = (ProgramRecorder.open(record_path) if record_path.exists() else
                         ProgramRecorder(record_path, metadata={"session_id": session_id}))
        self.branches: dict[str, BenchBranch] = {}
        self.cuts: dict[str, dict] = {}
        self.active = "main"
        self.closed = False
        self._lock = RLock()
        self.publication.owner_lock = self._lock
        self._edit = None
        self._jacobian_pending = None
        self._serialization_depth = 0
        self._manifest_batch = False
        self._manifest_dirty = False
        self._weight_epoch = 0
        self._frozen_readers = None
        self._capture_execution = None
        self._capture_staged = []
        self._capture_logits = None
        self._capture_stats = {"staged_tensors_encoded": 0, "staged_tensors_reused": 0,
                               "logit_refs_reused": 0, "manifests_deferred": 0}
        self.events: list[dict] = []
        from mdb.source_frame_stack import SourceFrameStack
        with self.publication.initializing():
            self.source_frames = SourceFrameStack(self.root / "source-frames")
        self.safe_point = None
        self.last_logits = None
        self._mlp_memory = None  # (MLPMemory, Installation, path, install act ref) while live

    def _check(self):
        if self.closed:
            raise BenchError("bench session is closed")

    @property
    def branch(self):
        self._check()
        if self.active not in self.branches:
            raise BenchError("prefill or resume a prompt first")
        return self.branches[self.active]

    def _backend(self):
        if self.active not in self.vm._sessions:
            snapshot = self.artifacts.get(self.branch.cut)
            self.vm.resume_session(self.active, snapshot["vm_checkpoint"])
        self.vm._resident(self.vm._session(self.active))
        return self.vm._session(self.active).backend

    def capabilities(self):
        return {"schema": BENCH_SCHEMA, "session_id": self.session_id, "lane": self.lane,
            "family": self.family, "checkpoint_identity": self.identity["weights_sha256"],
            "model_identity": self.identity, "numerical_contract": self.vm.package.model_abi.numerical_contract,
            "boundaries": ["token", "layer"], "writable": ["residual"],
            "execution_stages": ["token", "embedding", "layer", "readout", "sample", "commit"],
            "mutable_closure": ["residual", "staged_kv", "pending_token", "committed_history", "cursor"],
            "operations": ["prefill", "step", "generate", "inspect", "checkpoint", "fork", "checkout",
                "restore", "replay", "compare", "export", "close", "discover", "bind_symbol", "stage_input", "replay_observation",
                "annotate_state", "list_state_annotations", "show_state_annotation", "state_lineage", "verify_state_annotations",
                *(["memory_read", "memory_write", "memory_diff", "memory_probe",
                   "measure_circuit", "replay_circuit", "pin_jacobian_lens",
                   "read_jacobian", "preview_jacobian_edit", "configure_cot_register",
                   "cot_register", "cot_graph", "cot_pointer"] if self.publication.mode == "journal" else [])],
            "hidden_memory": {"protocol": "saturn-bench-hidden-memory-v1",
                "available": self.publication.mode == "journal", "publication_mode_required": "journal",
                "components": ["key", "value"], "selectors": ["layers", "row_start", "row_stop", "components"],
                "write_modes": ["copy", "zero", "scale", "replace"],
                "values_default": True, "default_max_values": 16, "max_values": 256,
                "default_max_rows": 32, "max_rows": 256,
                "boundary": "committed-token", "authority": "model-os",
                "publication": "single-atomic-journal-selection"},
            "circuit_measurement": {"protocol": "saturn-bench-circuit-measurement-v1", "available": self.publication.mode == "journal",
                "publication_mode_required": "journal",
                "observable": "token_margin", "start_boundary": "committed-token", "memory_mediation": True,
                "semantic_reasoning_detector": False,
                "memory_policy": "typed row/depth/component successor-KV interchange; fixed recipient input"},
            "integrations": {"model_os": "token-admission-and-commit", "virtual_memory": "resident-and-durable",
                "hyperion": "context-bound-native-consumer-contrasts", "future_forest": "persistent",
                "program_recorder": "persistent", "source_frames": "attached" if self.source_frames else "available-on-attach",
                "family_factory": "qwen2-mamba-flux-sdxl"},
            "publication": {"mode": self.publication.mode, "schema": "saturn-bench-publication-v1",
                "requested_cuts_durable": True, "recovery": "sealed-pack-before-selected-pointer"}}

    def _manifest(self):
        return {"schema": BENCH_SCHEMA, "session_id": self.session_id, "identity": self.identity,
                "lane": self.lane, "attention_backend": self.vm.runtime.backend, "capacity": self.capacity,
                "resident_kv_bytes": self.vm.resident_kv_bytes, "active": self.active,
                "branches": {name: vars(branch) for name, branch in self.branches.items()},
                "cuts": self.cuts, "events": self.events, "vm_events": self.vm.memory_events,
                "capabilities": self.capabilities(), "publication_mode": self.publication.mode,
                **({"mlp_memory": {"digest": self._mlp_memory[1].digest, "path": self._mlp_memory[2],
                                   "act_ref": self._mlp_memory[3]}}
                   if self._mlp_memory is not None else {})}

    def _publish(self):
        if self._manifest_batch:
            self._manifest_dirty = True
            self._capture_stats["manifests_deferred"] += 1
            return None
        ref = self.artifacts.put(self._manifest())
        self.publication.select(ref)
        return ref

    def _optimized_capture(self):
        if self._frozen_readers is None or not self._frozen_readers.valid:
            return False
        from .bench_cot import _config
        config, _ = _config(self)
        return bool(config and config.get("performance", {}).get("capture") == "changed-state-v1")

    @contextmanager
    def _command_publication(self):
        previous = (deepcopy(self.branches), dict(self.cuts), list(self.events),
                    self.active, list(self.vm.memory_events))
        self._manifest_batch, self._manifest_dirty = True, False
        try:
            with self.publication.atomic_selection():
                yield
                self._manifest_batch = False
                if self._manifest_dirty:
                    self._publish()
        except BaseException:
            self._manifest_batch = False
            (self.branches, self.cuts, self.events,
             self.active, self.vm.memory_events) = previous
            self._restore(self.branch.cut)
            raise
        finally:
            self._manifest_batch = self._manifest_dirty = False

    def _invalidate_capture(self):
        self._capture_execution = None
        self._capture_staged = []
        self._capture_logits = None

    @contextmanager
    def weight_mutation(self):
        """Exclusive mutation boundary, including untracked .data/custom-kernel writes.

        Fast bindings remain invalid until explicit reconfiguration verifies
        the new lens/model. Ordinary mutable callers should retain strict mode.
        """
        with self._lock:
            self._check()
            self._weight_epoch += 1
            if self._frozen_readers is not None:
                self._frozen_readers.invalidate()
            self._invalidate_capture()
            yield self.model

    def _logits_ref(self, tensor):
        if tensor is None:
            return None
        if not self._optimized_capture():
            return self._tensor_ref(tensor)
        if self._capture_logits is not None and self._capture_logits[0] is tensor:
            self._capture_stats["logit_refs_reused"] += 1
            return self._capture_logits[1]
        reference = self._tensor_ref(tensor)
        self._capture_logits = tensor, reference
        return reference

    def _staged_refs(self, execution):
        if not self._optimized_capture():
            return [[self._tensor_ref(t[:, :, execution.state.delta_rows:execution.state.delta_rows+1])
                     for t in pair] for pair in execution.state.delta[:execution.next_layer]]
        if self._capture_execution is not execution:
            self._capture_execution, self._capture_staged = execution, []
        saved = self._capture_staged
        if len(saved) > execution.next_layer:
            del saved[execution.next_layer:]
        self._capture_stats["staged_tensors_reused"] += 2 * len(saved)
        for pair in execution.state.delta[len(saved):execution.next_layer]:
            saved.append([self._tensor_ref(t[:, :, execution.state.delta_rows:execution.state.delta_rows+1])
                          for t in pair])
            self._capture_stats["staged_tensors_encoded"] += 2
        return deepcopy(saved)

    def _state_annotation_store(self):
        from .state_annotations import StateAnnotationStore
        return StateAnnotationStore.from_session(self)

    @_serialized
    def annotate_state(self, label, *, cut_id=None, namespace="default", role, stage,
                       scientific_source_cut_ids=(), recipient_parent_cut_id=None,
                       tags=(), native_act_refs=(), consumer_refs=(), geometry=None, details=None,
                       note=None, producer=None):
        """Attach descriptive provenance to an immutable cut without changing it."""
        return self._state_annotation_store().annotate(
            label, cut_id=cut_id or self.branch.cut, namespace=namespace, role=role, stage=stage,
            scientific_source_cut_ids=scientific_source_cut_ids,
            recipient_parent_cut_id=recipient_parent_cut_id, tags=tags,
            native_act_refs=native_act_refs, consumer_refs=consumer_refs,
            geometry=geometry, details=details, note=note, producer=producer)

    @_serialized
    def list_state_annotations(self, **filters):
        return self._state_annotation_store().list(**filters)

    @_serialized
    def show_state_annotation(self, reference):
        return self._state_annotation_store().show(reference)

    @_serialized
    def state_lineage(self, cut_id=None):
        return self._state_annotation_store().lineage(cut_id or self.branch.cut)

    @_serialized
    def verify_state_annotations(self):
        return self._state_annotation_store().verify()

    @_serialized
    @torch.inference_mode()
    def prefill(self, prompt, *, branch="main"):
        with self._lock:
            self._check()
            if branch in self.branches:
                raise BenchError("branch already exists")
            if isinstance(prompt, str):
                if self.tokenizer is None:
                    raise BenchError("text prompt requires a tokenizer")
                ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                prompt_text = prompt
            else:
                ids = torch.as_tensor(prompt, dtype=torch.long).reshape(1, -1)
                prompt_text = json.dumps(ids.tolist())
            if not ids.numel() or ids.shape[1] > self.model.config.max_position_embeddings:
                raise BenchError("prompt length is empty or exceeds model context")
            minimum_kv_bytes = (ids.shape[1] + self.capacity) * self.vm.runtime.codec.row_bytes
            if minimum_kv_bytes > self.vm.resident_kv_bytes:
                raise BenchError("prompt and private capacity exceed the resident KV budget")
            ids = ids.to(self.vm.runtime.device)
            previous = self.model.config._attn_implementation
            try:
                self.model.config._attn_implementation = "eager"
                result = self.model(ids, use_cache=True)
            finally:
                self.model.config._attn_implementation = previous
            if not bool(torch.isfinite(result.logits[:, -1]).all()):
                raise BenchError("non-finite prompt readout cannot publish a session")
            rows = self.vm.runtime.codec.encode_cache_rows(result.past_key_values)
            mount = self.vm.store.publish(model_abi_sha256=self.model_abi, pages={branch: rows})
            self.vm.open_session(branch, mount, branch, extra_rows=self.capacity)
            pending = int(result.logits[:, -1].argmax(-1).item())
            # First generated token is emitted only by step(), not by prefill().
            self.branches[branch] = BenchBranch(branch, pending, ids[0].tolist(), ids.shape[1])
            previous_active, previous_logits = self.active, self.last_logits
            previous_source = self.source_frames.current
            self.active = branch
            self.last_logits = result.logits[:, -1].detach()
            self.recorder.add(prompts=prompt_text, program={"kind": "bench-prefill", "identity": self.identity,
                "mount": mount, "lane": self.lane}, label=f"prefill-{branch}")
            self.recorder.write()
            try:
                return self.checkpoint("prefill")
            except Exception:
                self.vm.close_session(branch)
                del self.branches[branch]
                self.active, self.last_logits = previous_active, previous_logits
                self.source_frames._checkout(previous_source)
                raise

    def _begin(self, supplied_token=None):
        backend = self._backend()
        if backend.execution is None:
            if len(self.branch.token_ids) >= self.model.config.max_position_embeddings:
                raise BenchError("model context capacity exhausted")
            session = self.vm._session(self.active)
            self.vm._admit(lambda: backend.preview_allocation_bytes(session.kernel.current_statecut), protected=self.active)
            backend.begin(session.kernel, self.branch.pending if supplied_token is None else supplied_token)
            session.pinned = True  # Provisional rows cannot be evicted.
        return backend

    @_serialized
    def stage_input(self, token_id):
        if self._edit or self._backend().execution is not None:
            raise BenchError("input selection requires a committed AR boundary")
        if type(token_id) is not int or not 0 <= token_id < self.model.config.vocab_size:
            raise BenchError("invalid supplied input token")
        parent = self.branch.cut
        try:
            backend = self._begin(token_id)
            self.branch.pending = token_id
            self.events.append({"kind": "supplied-input", "token_id": token_id, "parent_cut": parent,
                "input_authority": "caller-supplied", "autonomous": False,
                "candidate": backend.prepared[0].to_payload()})
            return self.checkpoint("supplied-input-embedding")
        except BaseException:
            self._restore(parent); raise

    def _point(self, boundary, index):
        if self.safe_point:
            self.safe_point(boundary, index, {"branch": self.active, "cursor": self.cursor()})

    def _native(self, callback, *args):
        try:
            with self.metrics.measure("native_transition"):
                return callback(*args)
        except BaseException:
            if self.branch.cut: self._restore(self.branch.cut)
            raise

    def cursor(self):
        backend = self._backend()
        execution = backend.execution
        return {"token": len(self.branch.token_ids) - self.branch.prompt_length,
                "phase": ("sampled" if execution.sampled_token is not None else "readout" if execution.logits is not None
                          else "layers") if execution else "token", "next_layer": execution.next_layer if execution else None,
                "pending_token": self.branch.pending}

    @_serialized
    @torch.inference_mode()
    def step(self, kind="token", count=1):
        with self._lock:
            self._check()
            if self._edit:
                raise BenchError("commit or abort the pending Act first")
            if kind not in {"token", "layer", "embedding", "readout", "sample", "commit"} or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise BenchError("step requires token/layer and positive count")
            output = []
            for _ in range(count):
                backend = self._begin()
                if kind == "embedding":
                    if backend.execution.next_layer != 0:
                        raise BenchError("embedding has already executed")
                    output.append(self.cursor())
                    continue
                if kind in {"readout", "sample"}:
                    execution = backend.execution
                    if execution.next_layer != self.vm.runtime.codec.layers:
                        raise BenchError("readout/sampling requires all decoder layers")
                    if execution.logits is None:
                        self._native(self.vm.runtime.finish_token, execution)
                    if kind == "sample":
                        execution.sampled_token = int(execution.logits.argmax(-1).item())
                    output.append(self.cursor())
                    continue
                if kind == "commit" and backend.execution.sampled_token is None:
                    raise BenchError("commit requires readout and sampling")
                self._point("layer", backend.execution.next_layer)
                if kind == "layer":
                    if backend.execution.next_layer >= self.vm.runtime.codec.layers:
                        raise BenchError("all layers complete; step token to publish")
                    self._native(self.vm.runtime.advance_token, backend.execution)
                    self._observe_cot_boundary()
                    output.append(self.cursor())
                    self._point("layer", backend.execution.next_layer)
                    continue
                while backend.execution.next_layer < self.vm.runtime.codec.layers:
                    self._native(self.vm.runtime.advance_token, backend.execution)
                    self._observe_cot_boundary()
                    self._point("layer", backend.execution.next_layer)
                parent = self.branch.cut
                emitted = self.branch.pending
                try:
                    with self.metrics.measure("model_os_transaction"):
                        result = self.vm.execute(self.active, emitted, run_id=f"bench-{content_digest(self.cursor())[:20]}")
                    self.last_logits = backend.last_logits
                    self.branch.token_ids.append(emitted)
                    self.branch.pending = result.output["next_token_id"]
                    self.vm._session(self.active).pinned = False
                    cut = self.checkpoint("token")
                    output.append({"token_id": emitted, "next_token_id": self.branch.pending, "cut": cut,
                        "receipt_ref": result.output["receipt_ref"], "bundle_ref": result.receipt.vm_cache_bundle_reference,
                        "logits_sha256": tensor_digest(self.last_logits)})
                except Exception:
                    if parent:
                        self._restore(parent)
                    raise
                self._point("token", self.cursor()["token"])
            return output[0] if count == 1 else output

    @_serialized
    def generate(self, max_new_tokens=32):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise BenchError("generation budget must be a nonnegative integer")
        return [self.step() for _ in range(max_new_tokens)]

    def _tensor_ref(self, value):
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return self.vm.store.payloads.put_bytes(raw, shape=tuple(value.shape), dtype=str(value.dtype),
                                              role="bench-tensor").to_dict()

    def _tensor_load(self, reference):
        ref = PayloadRef.from_dict(reference)
        raw = self.vm.store.payloads.hydrate(ref, native_consumer="bench-resume").payload
        dtype = getattr(torch, ref.dtype.removeprefix("torch."))
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(ref.shape).to(self.vm.runtime.device)

    @_serialized
    def checkpoint(self, label=None):
        with self._lock:
            backend = self._backend()
            branch = self.branch
            with self.metrics.measure("checkpoint_encode"):
                vm_checkpoint = self.vm.checkpoint(self.active)
            source_parent = self.source_frames.current
            source_view = self._source_view(vm_checkpoint, label or "cut")
            snapshot = {"schema": BENCH_CUT_SCHEMA, "identity": self.identity, "lane": self.lane,
                "numerical_contract": self.vm.package.model_abi.fingerprint, "branch": self.active,
                "label": label or "cut", "token_ids": list(branch.token_ids), "prompt_length": branch.prompt_length,
                "pending": branch.pending, "parent_cut": branch.cut,
                "vm_checkpoint": vm_checkpoint, "symbols": dict(branch.symbols),
                "intervention_lineage": list(branch.intervention_lineage),
                "source_frames": source_view,
                "logits": self._logits_ref(self.last_logits),
                "execution": None}
            if self._mlp_memory is not None:
                snapshot["mlp_memory"] = self._mlp_memory[1].digest
            if backend.execution:
                execution = backend.execution
                snapshot["execution"] = {"token_id": execution.token_id, "next_layer": execution.next_layer,
                    "hidden": self._tensor_ref(execution.hidden), "staged_kv": self._staged_refs(execution),
                    "logits": self._logits_ref(execution.logits),
                    "sampled_token": execution.sampled_token}
            ref = self.artifacts.put(snapshot)
            obj = self.forest.put_json_object({"cut": ref}, media_type="application/json")
            parent_node = branch.node or next((b.node for b in self.branches.values() if b.node), None)
            node = (self.forest.fork(parent_node, transition={"kind": snapshot["label"]}, diff_refs={"cut": obj})
                    if parent_node else self.forest.create_root(state_refs={"cut": obj}, metadata={"session": self.session_id}))
            previous = branch.cut, branch.node
            branch.cut, branch.node = ref, node
            self.cuts[ref] = {"label": snapshot["label"], "node": node, "branch": self.active,
                              "cursor": self.cursor(), "vm_checkpoint": snapshot["vm_checkpoint"]}
            try:
                self._publish()
            except Exception:
                branch.cut, branch.node = previous
                self.cuts.pop(ref, None)
                self.source_frames._checkout(source_parent)
                raise
            return ref

    def _source_view(self, vm_checkpoint, label):
        from mdb.ar.source_set import ArSourceSet
        from mdb.source_program_surface import artifact_from_ar_source_set
        stack = self.source_frames
        # Execution cuts share the prompt's lexical scope. They do not introduce
        # another nested source scope on every decoder layer/token.
        if stack.current and label != "prefill":
            stack.checkout(stack.chain()[-1]["node_id"])
        parent = stack.inspect()["statecut_ref"] if stack.current else None
        cursor = self.cursor()
        state = {"session_id": self.session_id, "branch": self.active, "cursor": cursor,
                 "vm_checkpoint": vm_checkpoint, "bench_parent_cut": self.branch.cut}
        live = stack.cuts.capture(cut_id=f"bench-{content_digest({'state': state, 'parent': parent})[:24]}", family=self.family,
            state_schema_id=BENCH_CUT_SCHEMA, model_identity=self.identity,
            boundary="layer" if cursor["next_layer"] is not None else "token",
            index=cursor["next_layer"] if cursor["next_layer"] is not None else cursor["token"],
            total=self.vm.runtime.codec.layers if cursor["next_layer"] is not None else self.capacity,
            state=state, parent_cut=parent)
        artifacts = {}
        if label == "prefill":
            mount = self._backend().current_mount
            source = ArSourceSet(model_id=self.identity["model_id"], revision=self.identity["revision"],
                source_token_ids=tuple(self.branch.token_ids),
                layers=tuple(tuple(t.detach().cpu().contiguous() for t in pair) for pair in mount.ancestor),
                admitted=tuple(True for _ in self.branch.token_ids))
            store_id = f"prompt-{content_digest(self.branch.token_ids)[:16]}"
            if self._mlp_memory is not None:  # same prompt, different effective model
                store_id += f"-mlp-{self.mlp_memory_digest[:16]}"
            artifacts[store_id] = artifact_from_ar_source_set(source, store_id=store_id)
        stack.push(label=label, statecut_ref=live.ref, artifacts=artifacts,
                   evidence={"session_id": self.session_id, "vm_checkpoint": vm_checkpoint})
        return {"active": stack.current, "statecut_ref": live.ref}

    def _restore(self, ref, *, branch=None):
        self._invalidate_capture()
        snapshot = self.artifacts.get(ref)
        if (snapshot.get("schema") != BENCH_CUT_SCHEMA or snapshot["identity"] != self.identity
                or snapshot["lane"] != self.lane or snapshot["numerical_contract"] != self.vm.package.model_abi.fingerprint):
            raise BenchError("checkpoint model, numerical lane or schema is incompatible")
        if snapshot.get("mlp_memory") != self.mlp_memory_digest:
            raise BenchError("checkpoint was cut under a different MLP memory overlay; "
                             "install or remove that overlay first")
        tokens, pending, prompt_length = snapshot["token_ids"], snapshot["pending"], snapshot["prompt_length"]
        if (type(pending) is not int or not 0 <= pending < self.model.config.vocab_size
                or type(prompt_length) is not int or not 0 < prompt_length <= len(tokens)
                or len(tokens) > self.model.config.max_position_embeddings
                or any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in tokens)):
            raise BenchError("checkpoint token accounting is incompatible")
        checkpoint = self.vm.artifacts.get(snapshot["vm_checkpoint"])
        if checkpoint["ancestor_rows"] + checkpoint["delta_rows"] != len(tokens):
            raise BenchError("checkpoint KV visibility differs from committed token history")
        saved = snapshot["execution"]
        if saved is not None:
            layer = saved["next_layer"]
            if (type(layer) is not int or not 0 <= layer <= self.vm.runtime.codec.layers
                    or saved["token_id"] != pending or len(saved["staged_kv"]) != layer
                    or (saved.get("logits") is not None and layer != self.vm.runtime.codec.layers)
                    or (saved.get("sampled_token") is not None and (saved.get("logits") is None
                        or type(saved["sampled_token"]) is not int
                        or not 0 <= saved["sampled_token"] < self.model.config.vocab_size))):
                raise BenchError("checkpoint provisional execution position is incompatible")
            expected = (1, self.vm.runtime.codec.groups, 1, self.vm.runtime.codec.head_dim)
            refs = [(saved["hidden"], (1, 1, self.model.config.hidden_size))]
            for pair in saved["staged_kv"]:
                if len(pair) != 2:
                    raise BenchError("checkpoint staged KV pair is incomplete")
                refs.extend((reference, expected) for reference in pair)
            if saved.get("logits"):
                refs.append((saved["logits"], (1, self.model.config.vocab_size)))
            for reference, shape in refs:
                payload = PayloadRef.from_dict(reference)
                if tuple(payload.shape) != shape or payload.dtype != str(self.vm.runtime.codec.dtype):
                    raise BenchError("checkpoint provisional tensor geometry is incompatible")
        name = branch or self.active
        if name in self.vm._sessions:
            if self.vm._session(name).backend:
                self.vm._session(name).backend.cancel_execution()
            self.vm.close_session(name)
        self.vm.resume_session(name, snapshot["vm_checkpoint"])
        self.branches[name] = BenchBranch(name, snapshot["pending"], list(snapshot["token_ids"]),
            snapshot["prompt_length"], ref, self.cuts.get(ref, {}).get("node"), dict(snapshot["symbols"]),
            list(snapshot.get("intervention_lineage", [])))
        self.active = name
        if snapshot.get("source_frames"):
            self.source_frames.checkout(snapshot["source_frames"]["active"])
        self.last_logits = self._tensor_load(snapshot["logits"]) if snapshot["logits"] else None
        if snapshot["execution"]:
            backend = self._begin()
            execution = backend.execution
            execution.hidden = self._tensor_load(saved["hidden"])
            execution.next_layer = saved["next_layer"]
            execution.logits = self._tensor_load(saved["logits"]) if saved.get("logits") else None
            execution.sampled_token = saved.get("sampled_token")
            for pair, refs in zip(execution.state.delta, saved["staged_kv"]):
                for tensor, reference in zip(pair, refs):
                    tensor[:, :, execution.state.delta_rows:execution.state.delta_rows+1].copy_(self._tensor_load(reference))
        return {"cut_id": ref, "cursor": self.cursor(), "session_id": self.session_id}

    @_serialized
    def restore(self, ref):
        with self._lock:
            if self._edit:
                raise BenchError("commit or abort the pending Act first")
            previous = self.branch.cut
            try:
                result = self._restore(ref)
                self._publish()
                return result
            except Exception:
                if previous:
                    self._restore(previous)
                raise

    @_serialized
    def fork(self, name, *, cut=None):
        with self._lock:
            if not name or name in self.branches or self._edit:
                raise BenchError("fork needs a unique name and no pending Act")
            ref = cut or self.checkpoint("fork-parent")
            previous_name, previous_cut = self.active, self.branch.cut
            self._suspend()
            try:
                self._restore(ref, branch=name)
                self.events.append({"kind": "fork", "branch": name, "parent": ref})
                return self.checkpoint("fork")
            except Exception:
                if name in self.vm._sessions:
                    self.vm.close_session(name)
                self.branches.pop(name, None)
                self.active = previous_name
                self._restore(previous_cut)
                raise

    @_serialized
    def checkout(self, name):
        with self._lock:
            if self._edit or name not in self.branches:
                raise BenchError("unknown branch or pending Act")
            # Persist provisional state before changing the selected branch.
            self.checkpoint("checkout-parent")
            self._suspend()
            self.active = name
            return self.restore(self.branches[name].cut)

    def _suspend(self):
        """A persisted layer cut is sufficient to evict an inactive provisional token."""
        session = self.vm._session(self.active)
        if session.backend:
            session.backend.cancel_execution()
        session.pinned = False

    @_serialized
    def replay(self, ref, max_new_tokens=1):
        self.restore(ref)
        return self.generate(max_new_tokens)

    def compare(self, left, right):
        a, b = self.artifacts.get(left), self.artifacts.get(right)
        if a["identity"] != b["identity"] or a["lane"] != b["lane"]:
            raise BenchError("compare requires compatible cuts; cross-lane assays are explicit")
        return {"left": left, "right": right, "same_tokens": a["token_ids"] == b["token_ids"],
                "same_logits": a["logits"] == b["logits"], "same_execution": a["execution"] == b["execution"],
                "left_tokens": a["token_ids"], "right_tokens": b["token_ids"]}

    def inspect(self, path=None):
        with self._lock:
            if path == "performance":
                return {**self.metrics.inspect(), "publication": dict(self.publication.stats),
                        "payload_stats": self.vm.store.payloads.stats.to_dict(), "checkpoint_pages": dict(self.vm.checkpoint_stats),
                        "capture": dict(self._capture_stats),
                        "reader_binding": dict(self._frozen_readers.stats) if self._frozen_readers else {"mode": "strict"},
                        "register_index": dict(self._cot_index.stats) if hasattr(self, "_cot_index") else {}}
            if path == "memory":
                residency = self.vm.residency()
                pages = {}
                checkpoints = {}
                for name in self.branches:
                    session = self.vm._sessions.get(name)
                    checkpoint = (session.checkpoint_ref if session else
                                  self.artifacts.get(self.branches[name].cut)["vm_checkpoint"])
                    checkpoints[name] = checkpoint
                    if not checkpoint: continue
                    body = self.vm.artifacts.get(checkpoint)
                    for page in body.get("delta_pages", []):
                        ref = PayloadRef.from_dict(page["payload_ref"])
                        row = pages.setdefault(ref.sha256, {"sha256": ref.sha256, "bytes": ref.bytes,
                            "owners": [], "backed": True, "allocation_kind": "durable-kv-row-block"})
                        row["owners"].append({"branch": name, "start_row": page["start_row"], "rows": page["rows"]})
                return {**residency, "weights_bytes": sum(p.numel()*p.element_size() for p in self.model.parameters()),
                    "weights_in_kv_budget": False, "cuts": len(self.cuts),
                    "branches": {name: {"cut": b.cut, "logical_rows": len(b.token_ids),
                        "resident": name in self.vm._sessions and self.vm._session(name).backend is not None,
                        "checkpoint": checkpoints[name]} for name, b in self.branches.items()},
                    "row_bytes": self.vm.runtime.codec.row_bytes,
                    "visible_bytes": sum(len(b.token_ids)*self.vm.runtime.codec.row_bytes for b in self.branches.values()),
                    "unique_allocations": self.vm.allocation_map(),
                    "workspace": {"cuda_allocated_bytes": torch.cuda.memory_allocated(self.vm.runtime.device)
                                  if self.vm.runtime.device.type == "cuda" else None,
                                  "includes_weights_and_kv": True},
                    "payload_stats": self.vm.store.payloads.stats.to_dict(), "events": list(self.events),
                    "vm_events": list(self.vm.memory_events), "backing_pages": list(pages.values()),
                    "backing_page_rows": self.vm.checkpoint_page_rows,
                    "backing_unique_bytes": sum(p["bytes"] for p in pages.values()),
                    "publication": dict(self.publication.stats)}
            if path in {"symbols", "circuits"}:
                from .bench_discovery import binding_status
                bindings = {name: {"report": self.artifacts.get(ref), "status":
                    binding_status(self.artifacts.get(ref)["symbol"]["context"], self.artifacts.get(ref)["symbol"]["context"].get("source_tokens_sha256")
                    == content_digest(self.branch.token_ids) and self.artifacts.get(ref)["symbol"]["context"].get("intervention_lineage", [])
                    == self.branch.intervention_lineage),
                    "qualification": self.artifacts.get(ref)["symbol"]["context"].get("qualification", {"quality_status": "candidate-trend", "semantic_binding": False})}
                    for name, ref in self.branch.symbols.items()}
                if path == "symbols":
                    bindings["_structural"] = {"status": "unmeasured-structure", "semantic_binding": False,
                        "names": ["qwen.embedding", *[f"qwen.layer.{i}.residual" for i in range(self.vm.runtime.codec.layers)], "qwen.readout"],
                        "runtime_addresses": ["residual", "logits", "kv_cache"], "cursor": self.cursor()}
                return bindings
            if path in {None, "status"}:
                result = {"session_id": self.session_id, "branch": self.active, "cursor": self.cursor(),
                          "cut_id": self.branch.cut, "capabilities": self.capabilities()}
                if any(event.get("kind") == "cot-register-configured" for event in self.events):
                    result["cot"] = self.cot_register(limit=3)
                return result
            if path == "cot":
                return self.cot_register()
            if path == "cot.graph":
                return self.cot_graph()
            if path == "frames":
                return self.source_frames.debug_view() if self.source_frames else {"attached": False}
            if path == "measurements":
                from .bench_circuit_measurement import inspect_measurements
                return inspect_measurements(self)
            if path == "cursor":
                return self.cursor()
            if path == "token_ids":
                return {"value": list(self.branch.token_ids)}
            value = self.resolve_value(path)
            return {"path": path, "shape": list(value.shape), "dtype": str(value.dtype),
                    "device": str(value.device), "sha256": tensor_digest(value),
                    "stats": {"norm": float(value.float().norm()), "mean": float(value.float().mean())}}

    def resolve_value(self, path):
        backend = self._backend()
        if path == "residual" and backend.execution:
            return backend.execution.hidden.detach().clone()
        if path == "logits" and self.last_logits is not None:
            value = backend.execution.logits if backend.execution and backend.execution.logits is not None else self.last_logits
            return value.detach().clone()
        if path.startswith("kv_cache.layer_"):
            parts = path.split(".")
            index = int(parts[1].removeprefix("layer_"))
            component = {"key": 0, "value": 1}.get(parts[2] if len(parts) > 2 else "")
            if component is None or not 0 <= index < self.vm.runtime.codec.layers:
                raise BenchError("unknown K/V address")
            mount = backend.current_mount
            if len(parts) > 3 and parts[3] == "ancestor":
                return mount.ancestor[index][component].detach().clone()
            if len(parts) > 3 and parts[3] == "private":
                return mount.delta[index][component][:, :, :mount.delta_rows].detach().clone()
            return torch.cat((mount.ancestor[index][component], mount.delta[index][component][:, :, :mount.delta_rows]), dim=2)
        raise BenchError("address is not inspectable at this boundary")

    @_serialized
    def preview_edit(self, path, value, *, mode="replace"):
        with self._lock:
            if self._edit or path != "residual" or not self._backend().execution or self._backend().execution.logits is not None:
                raise BenchError("residual edit requires a layer boundary and no pending Act")
            old = self.resolve_value(path)
            candidate = torch.as_tensor(value, dtype=old.dtype, device=old.device)
            if candidate.numel() == 1:
                candidate = candidate.expand_as(old)
            if candidate.shape != old.shape or not torch.isfinite(candidate).all():
                raise BenchError("residual edit shape or finite-value contract failed")
            if mode == "add":
                candidate = old + candidate
            elif mode == "scale":
                candidate = old * candidate
            elif mode != "replace":
                raise BenchError("unsupported edit mode")
            candidate = candidate.detach().clone()
            if not torch.isfinite(candidate).all():
                raise BenchError("residual edit produced non-finite values")
            parent = self.checkpoint("act-parent")
            act_id = content_digest({"parent": parent, "path": path, "mode": mode, "payload": tensor_digest(candidate)})
            from mdb.bench_admission import residual_transaction
            kernel, updates = residual_transaction(self, parent, candidate)
            plan = kernel.plan(horizon=0)
            staged, _ = kernel.backend.preview(kernel.current_statecut, plan, updates)
            kernel.backend.prepared = staged
            self._edit = act_id, parent, candidate, kernel, updates
            return {"act_id": act_id, "parent_cut": parent, "candidate_sha256": tensor_digest(candidate),
                    "plan": plan.to_payload()}

    @_serialized
    def commit_edit(self, act_id):
        with self._lock:
            if not self._edit or self._edit[0] != act_id:
                raise BenchError("pending Act identity changed")
            _, parent, candidate, kernel, updates = self._edit
            previous_events, previous_cuts = deepcopy(self.events), deepcopy(self.cuts)
            jacobian = self._jacobian_pending
            self._jacobian_pending = None
            self._edit = None
            try:
                receipt = kernel.execute(updates=updates, horizon=0).receipt
                receipt_ref = self.artifacts.put(receipt.to_payload())
                self.branch.intervention_lineage.append(act_id)
                event = {"kind": "residual-act", "parent": parent, "act_id": act_id,
                         "receipt_ref": receipt_ref}
                if jacobian is not None and jacobian["act_id"] == act_id:
                    event["jacobian_candidate_ref"] = jacobian["candidate_ref"]
                self.events.append(event)
                cut = self.checkpoint("act-committed")
                result = {"act_id": act_id, "cut_id": cut, "receipt_ref": receipt_ref}
                if "jacobian_candidate_ref" in event:
                    result["jacobian_candidate_ref"] = event["jacobian_candidate_ref"]
                return result
            except Exception:
                self.events, self.cuts = previous_events, previous_cuts
                self._restore(parent)
                raise

    @_serialized
    def abort_edit(self, act_id):
        if not self._edit or self._edit[0] != act_id:
            raise BenchError("pending Act identity changed")
        _, parent, _, kernel, updates = self._edit
        previous_events, previous_cuts = deepcopy(self.events), deepcopy(self.cuts)
        jacobian = self._jacobian_pending
        self._edit = None
        self._jacobian_pending = None
        try:
            receipt = kernel.execute(updates=updates, horizon=0, verifier=lambda _: False).receipt
            event = {"kind": "act-rejected", "act_id": act_id, "parent": parent,
                     "receipt_ref": self.artifacts.put(receipt.to_payload())}
            if jacobian is not None and jacobian["act_id"] == act_id:
                event["jacobian_candidate_ref"] = jacobian["candidate_ref"]
            self.events.append(event)
            return self.restore(parent)
        except Exception:
            self.events, self.cuts = previous_events, previous_cuts
            self._restore(parent)
            raise

    @_serialized
    def bind_symbol(self, report, *, name=None):
        from mdb.hyperion import verify_hyperion_report
        verify_hyperion_report(report)
        symbol = report["symbol"]
        context = symbol["context"]
        if (symbol["model_id"] != self.identity["model_id"] or context.get("bench_model_abi") != self.model_abi
                or context.get("lane") != self.lane
                or context.get("source_tokens_sha256") != content_digest(self.branch.token_ids)):
            raise BenchError("symbol is outside its measured model/context/numerical lane")
        if context.get("intervention_lineage", []) != self.branch.intervention_lineage:
            raise BenchError("symbol intervention lineage is incompatible")
        name = name or symbol["logical_symbol"]
        ref = self.artifacts.put(report)
        previous = self.branch.symbols.get(name)
        self.branch.symbols[name] = ref
        try:
            self._publish()
        except Exception:
            if previous is None:
                self.branch.symbols.pop(name)
            else:
                self.branch.symbols[name] = previous
            raise
        return ref

    @_serialized
    def discover(self, spec):
        from mdb.bench_symbols import discover
        return discover(self, spec)

    @_serialized
    def replay_observation(self, evidence_ref):
        from mdb.bench_symbols import replay_observation
        return replay_observation(self, evidence_ref)

    @_serialized
    def memory_read(self, spec=None):
        from .bench_hidden_memory import memory_read
        return memory_read(self, spec)

    @_serialized
    def memory_write(self, spec):
        from .bench_hidden_memory import memory_write
        return memory_write(self, spec)

    @_serialized
    def memory_diff(self, spec):
        from .bench_hidden_memory import memory_diff
        return memory_diff(self, spec)

    @_serialized
    def memory_probe(self, spec):
        from .bench_memory_probe import atomic_probe
        return atomic_probe(self, spec)

    @property
    def mlp_memory_digest(self):
        return None if self._mlp_memory is None else self._mlp_memory[1].digest

    @_serialized
    def mlp_memory_install(self, memory):
        from .bench_mlp_memory import install
        return install(self, memory)

    @_serialized
    def mlp_memory_remove(self):
        from .bench_mlp_memory import remove
        return remove(self)

    @_serialized
    def measure_circuit(self, spec):
        from .bench_circuit_measurement import atomic_measurement, measure_circuit
        return atomic_measurement(self, measure_circuit, spec)

    @_serialized
    def pin_jacobian_lens(self, manifest, transport):
        from .bench_jacobian import atomic_sidecar, pin_lens
        return atomic_sidecar(self, pin_lens, manifest, transport)

    @_serialized
    def read_jacobian(self, lens_ref, *, token_subset=None, top_k=10, logical_address=None):
        from .bench_jacobian import atomic_sidecar, read
        options = {"token_subset": token_subset, "top_k": top_k, "logical_address": logical_address}
        return atomic_sidecar(self, read, lens_ref, options)

    @_serialized
    def preview_jacobian_edit(self, observation_ref, spec):
        from .bench_jacobian import atomic_sidecar, preview
        return atomic_sidecar(self, preview, observation_ref, spec)

    @_serialized
    def configure_cot_register(self, lens_refs=None, *, top_k=5, token_subset=None, enabled=True, optimized=False):
        """Attach a live read-only register to compatible native layer boundaries."""
        from .bench_cot import configure
        from .bench_jacobian import atomic_sidecar
        return atomic_sidecar(self, lambda bench: configure(
            bench, lens_refs, top_k=top_k, token_subset=token_subset, enabled=enabled, optimized=optimized))

    def _observe_cot_boundary(self):
        from .bench_cot import observe_boundary
        observe_boundary(self)

    def _render_cot_panel(self):
        from .bench_cot import _config, write_live_panel
        try:
            if _config(self)[0] is None:
                return
            with self.metrics.measure("cot_render"):
                write_live_panel(self)
            if self._optimized_capture():
                from .bench_pointer import emit
                with self.metrics.measure("cot_pointer"):
                    self._cot_pointer = emit(self)
            self._cot_panel_error = None
        except Exception as exc:
            self._cot_panel_error = f"{type(exc).__name__}: {exc}"

    @_serialized
    def cot_register(self, *, limit=8):
        from .bench_cot import view
        result = view(self, limit=limit)
        if getattr(self, "_cot_panel_error", None):
            result["panel_error"] = self._cot_panel_error
        return result

    @_serialized
    def cot_graph(self, *, limit=128):
        from .bench_cot import graph
        return graph(self, limit=limit)

    def cot_pointer(self, destination=None):
        from .bench_pointer import emit
        with self._lock:
            self._check()
            return emit(self, destination)

    @_serialized
    def replay_circuit(self, evidence_ref):
        from .bench_circuit_measurement import atomic_measurement, replay_circuit
        return atomic_measurement(self, replay_circuit, evidence_ref)

    def _circuit_memory_act(self, donor_cut, selector=None):
        """Install one typed donor KV selection at a committed successor cut."""
        if self._edit or self._backend().execution is not None or self.cursor()["phase"] != "token":
            raise BenchError("successor KV interchange requires a committed Qwen token boundary")
        donor = self.artifacts.get(donor_cut)
        if (donor.get("schema") != BENCH_CUT_SCHEMA
                or donor.get("identity") != self.identity
                or donor.get("lane") != self.lane
                or donor.get("numerical_contract") != self.vm.package.model_abi.fingerprint
                or donor.get("execution") is not None):
            raise BenchError("incompatible Qwen successor KV donor closure")
        if (donor.get("token_ids") != self.branch.token_ids
                or donor.get("prompt_length") != self.branch.prompt_length):
            raise BenchError("Qwen successor KV interchange requires identical committed history")
        parent = self.branch.cut
        previous_lineage = list(self.branch.intervention_lineage)
        previous_events = list(self.events)
        from .bench_qwen_memory import transaction
        kernel, updates, plan = transaction(self, parent_cut=parent, donor_cut=donor_cut,
                                            donor=donor, selector=selector)
        try:
            result = kernel.execute(updates=updates, horizon=0,
                metadata={"session_id": self.session_id, "branch": self.active,
                          "parent_cut": parent, "donor_cut": donor_cut})
            if result.receipt.decision != "commit":
                raise BenchError("successor KV Act did not commit")
            receipt_ref = self.artifacts.put(result.receipt.to_payload())
            self.branch.intervention_lineage.append(plan["act_id"])
            self.events.append({"kind": "successor-kv-act", "parent": parent,
                "donor_cut": donor_cut, "act_id": plan["act_id"], "receipt_ref": receipt_ref,
                "selector": plan["selector"], "pending_token_policy": plan["pending_token_policy"]})
            cut = self.checkpoint("successor-kv-committed")
            return {**plan, "cut_id": cut, "receipt_ref": receipt_ref,
                    "parent_cut": parent, "donor_cut": donor_cut,
                    "plan": {"writes": ["successor_kv"], "authority": "model-os",
                             "selector": plan["selector"]}}
        except BaseException:
            self.branch.intervention_lineage = previous_lineage
            self.events = previous_events
            self._restore(parent)
            raise

    @_serialized
    def attach_source_frames(self, stack):
        if stack.inspect()["model_identity"] != self.identity or not stack.root.resolve().is_relative_to(self.root):
            raise BenchError("source frames must have the session identity and durable backing inside the session root")
        self.source_frames = stack
        self._publish()

    @_serialized
    def export(self, destination):
        with self._lock:
            if self._edit:
                raise BenchError("commit or abort the pending Act before export")
            self.checkpoint("export")
            target = Path(destination).expanduser().resolve()
            if target == self.root or self.root in target.parents:
                raise BenchError("export needs an independent destination")
            if target.exists():
                raise BenchError("export destination already exists")
            shutil.copytree(self.root, target)
            return {"root": str(target), "cut": self.branch.cut, "manifest": json.loads((target/"selected.json").read_text())["manifest"]}

    @classmethod
    def resume(cls, model, tokenizer=None, *, root, model_sha256=None, _writer=None, _manifest=None,
               lazy_branches=False):
        root = Path(root)
        if _writer is None: _writer, selected = recover_locked(root)
        else: selected = json.loads((root/"selected.json").read_text())
        try: manifest = _manifest or DirectoryArtifactStore(root/"artifacts").get(selected["manifest"])
        except BaseException: _writer.close(); raise
        result = cls(model, tokenizer, root=root, model_id=manifest["identity"]["model_id"],
            revision=manifest["identity"]["revision"], model_sha256=model_sha256,
            session_id=manifest["session_id"], lane=manifest["lane"], attention_backend=manifest["attention_backend"],
            resident_kv_bytes=manifest["resident_kv_bytes"], continuation_capacity=manifest["capacity"],
            publication_mode=manifest.get("publication_mode", "journal"), _writer=_writer)
        if result.identity != manifest["identity"]:
            result._writer.close()
            raise BenchError("resume model/tokenizer identity mismatch")
        result.cuts, result.events = manifest["cuts"], manifest["events"]
        result.vm.memory_events = list(manifest.get("vm_events", []))
        for name, branch in manifest["branches"].items():
            result.branches[name] = BenchBranch(**branch)
            if not lazy_branches:
                snapshot = result.artifacts.get(branch["cut"])
                result.vm.resume_session(name, snapshot["vm_checkpoint"])
        result.active = manifest["active"]
        if manifest.get("mlp_memory"):
            from .bench_mlp_memory import reinstall
            reinstall(result, manifest["mlp_memory"])
        with result.publication.transaction(): result._restore(result.branches[result.active].cut)
        return result

    def close(self):
        with self._lock:
            if self.closed:
                return
            with self.publication.transaction():
                if self._edit:
                    self.abort_edit(self._edit[0])
                if self.branches:
                    self.checkpoint("close")
                for name in list(self.vm._sessions):
                    self.vm._session(name).backend.cancel_execution() if self.vm._session(name).backend else None
                    self.vm.close_session(name)
                if self._mlp_memory is not None:  # the resident model is caller owned
                    self._mlp_memory[1].remove()
                self.closed = True
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
