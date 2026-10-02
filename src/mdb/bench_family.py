"""Family-neutral control plane; native adapters own all execution state.

Every transition is admitted by Model OS before its native callback, then
published as a durable cut. Publication failure restores the complete parent.
"""
from dataclasses import dataclass, field
from functools import wraps
from hashlib import sha256
from copy import deepcopy
from pathlib import Path
from threading import RLock
import fcntl
import json
import os
import shutil
import tempfile
import torch

from .bench_qwen import BenchError, tensor_digest
from .bench_pages import BenchPages, tensors
from .circuit_runtime import DirectoryArtifactStore
from .future_forest import PersistentFutureForest
from .ar.program_record import ProgramRecorder
from .source_frame_stack import SourceFrameStack
from .model_os_contracts import (ModelABI, StateSchema, StateSlot, StateSurface,
    ModelPackage, ModelProgram, ProgramInstruction, ConsumerContract, content_digest)
from .model_os_kernel import InMemoryModelBackend, ModelOSKernel
from .vm_cache import VMCacheEmitter, VMCacheEntry, VMCachePointer
from .bench_publication import BenchPublication, recover_locked, writer_path
from .bench_metrics import BenchMetrics

SCHEMA = "saturn-open-bench-v3"
CUT_SCHEMA = "saturn-open-bench-cut-v3"
ADDRESS = "bench.runtime"


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            if self.closed: raise BenchError("bench session is closed")
            with self.publication.transaction():
                return method(self, *args, **kwargs)
    return call


@dataclass
class FamilyBranch:
    name: str
    state: dict | None = None
    cut: str | None = None
    node: str | None = None
    symbols: dict = field(default_factory=dict)
    intervention_lineage: list = field(default_factory=list)


class NativeTransitionBackend(InMemoryModelBackend):
    def preview(self, parent, plan, updates):
        if set(updates) != {self.address} or updates[self.address] != self.command:
            raise BenchError("native transition command differs from admitted command")
        # Kernel.plan and scope validation precede preview. This native callback
        # runs once, and its actual tensor references define the candidate cut.
        self.state, self.output = self.callback()
        with self.owner.metrics.measure("validation"):
            self.owner._validate(self.state)
        with self.owner.metrics.measure("encode"):
            self.encoded = self.owner.pages.encode(self.state, previous_state=self.owner.state,
                previous_encoded=self.owner._encoded_state,
                readonly_ids=self.owner.adapter.readonly_ids(self.owner.state, self.command.get("kind"))
                    if self.owner.state is not None and hasattr(self.owner.adapter, "readonly_ids") else frozenset())
        return super().preview(parent, plan, {self.address: self.encoded})

    def continue_execution(self, candidate, *, consumer, horizon):
        super().continue_execution(candidate, consumer=consumer, horizon=horizon)
        return {"accepted": True, "execution_mode": "family-native-transition",
                "consumer_id": consumer.consumer_id, "horizon": horizon,
                "state_sha256": content_digest(self.encoded), "output": self.output}


class FamilyBenchSession:
    @classmethod
    def from_resident(cls, model, tokenizer=None, *, root, family, **kwargs):
        if (Path(root)/"selected.json").exists(): raise BenchError("session exists; use resume")
        return cls(model, tokenizer, root=root, family=family, **kwargs)

    def __init__(self, model, tokenizer=None, *, root, family, model_id=None, revision="local",
                 model_sha256=None, session_id="bench", lane="virtual", resident_kv_bytes=64 << 20,
                 resident_state_bytes=None, continuation_capacity=128, attention_backend=None,
                 publication_mode="journal", _writer=None, _adapter=None,
                 _legacy_numerical_environment=None):
        if lane not in {"native", "virtual"}: raise BenchError("unsupported numerical lane")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._writer = _writer or writer_path(self.root).open("a+")
        try: fcntl.flock(self._writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._writer.close(); raise BenchError("another writer owns this session") from None
        self.closed, self._lock, self._edit = False, RLock(), None
        self._encoded_state = None
        self._owns_adapter = _adapter is None
        try:
            self.publication = BenchPublication(self.root, mode=publication_mode)
            self.publication.owner_lock = self._lock
            self.metrics = BenchMetrics()
            if _adapter is not None:
                if _adapter.family != family or _adapter.model is not model: raise BenchError("incompatible shared native adapter")
                self.adapter = _adapter
            elif family == "mamba":
                from .bench_mamba import MambaBenchAdapter
                self.adapter = MambaBenchAdapter(model, tokenizer)
            elif family in {"flux", "sdxl"}:
                from .bench_diffusion import DiffusionBenchAdapter
                self.adapter = DiffusionBenchAdapter(model, family=family)
            else: raise BenchError("unsupported family adapter")
            self.model, self.tokenizer, self.family = model, tokenizer, family
            self.session_id, self.lane, self.capacity = session_id, lane, continuation_capacity
            components = {"model": model} if hasattr(model, "named_parameters") else model.components
            self.parameters = [(f"{name}.{key}", value) for name, component in components.items()
                if hasattr(component, "named_parameters") for key, value in component.named_parameters()]
            if model_sha256 is None:
                digest = sha256()
                for name, value in self.parameters:
                    digest.update(name.encode()); digest.update(bytes.fromhex(tensor_digest(value)))
                model_sha256 = digest.hexdigest()
            config = dict(model.config) if isinstance(model.config, dict) else model.config.to_dict()
            self.identity = {"model_id": model_id or f"local-{family}", "revision": revision,
                "family": family, "weights_sha256": model_sha256,
                "config_sha256": content_digest(json.loads(json.dumps(config, default=str))),
                "tokenizer_sha256": content_digest(tokenizer.get_vocab()) if tokenizer else "pipeline-owned-or-token-ids",
                "dtype": str(self.parameters[0][1].dtype), "device": str(self.adapter.device),
                "runtime_sha256": content_digest(self.adapter.runtime_identity)}
            cuda_lane = self.adapter.device.type == "cuda"
            self.identity["numerical_environment"] = {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "matmul_precision": torch.get_float32_matmul_precision(),
                "cpu_threads": torch.get_num_threads() if not cuda_lane else None,
                "cudnn_deterministic": torch.backends.cudnn.deterministic if cuda_lane else None,
                "cudnn_benchmark": torch.backends.cudnn.benchmark if cuda_lane else None,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32 if cuda_lane else None,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32 if cuda_lane else None,
                "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if cuda_lane else None,
                "cuda_version": torch.version.cuda if cuda_lane else None,
                "cudnn_version": torch.backends.cudnn.version() if cuda_lane else None,
                "cuda_device": torch.cuda.get_device_name(self.adapter.device) if cuda_lane else None}
            self.legacy_numerical_environment = deepcopy(_legacy_numerical_environment)
            if self.legacy_numerical_environment is not None:
                if family != "mamba" or self.identity["numerical_environment"] != self.legacy_numerical_environment:
                    raise BenchError("legacy numerical environment does not match the resident lane")
                # Retain the old ABI/CAS identity; the explicit qualification is
                # bound separately in the enclosing manifest for future resumes.
                del self.identity["numerical_environment"]
            self.model_abi = content_digest(self.identity)
            with self.publication.initializing():
                self.pages = BenchPages(self.root/"payloads", device=self.adapter.device,
                                       budget=resident_state_bytes or resident_kv_bytes)
                self.artifacts = DirectoryArtifactStore(self.root/"artifacts")
                self.forest = PersistentFutureForest(self.root/"forest", forest_id=session_id)
                self.source_frames = SourceFrameStack(self.root/"source-frames")
                self.emitter = VMCacheEmitter(self.root/"vm-cache")
            path = self.root/"programs.json"
            self.recorder = ProgramRecorder.open(path) if path.exists() else ProgramRecorder(path, metadata={"session_id": session_id})
            self.active, self.branches, self.cuts, self.events = "main", {}, {}, []
            self.state, self.safe_point, self.source_root = None, None, None
        except Exception:
            self._writer.close(); raise

    @property
    def branch(self):
        if self.active not in self.branches: raise BenchError("prepare or prefill a prompt first")
        return self.branches[self.active]

    @property
    def last_logits(self):
        return self.adapter.value(self.state, "logits") if self.family == "mamba" and self.state else None

    def cursor(self): return self.adapter.cursor(self.state) if self.state is not None else {"phase": "unprepared"}

    def capabilities(self):
        return {"schema": SCHEMA, "state_schema_id": CUT_SCHEMA, "session_id": self.session_id, "family": self.family, "lane": self.lane,
                "checkpoint_identity": self.identity["weights_sha256"], "model_identity": self.identity,
                "numerical_contract": self.adapter.contract, "execution_stages": list(self.adapter.stages),
                "boundaries": list(self.adapter.stages), "writable": list(self.adapter.writable),
                "mutable_closure": list(self.adapter.closure), "default_step": self.adapter.default_step,
                "operations": ["prefill" if self.family == "mamba" else "prepare", "step", "generate", "inspect",
                    "checkpoint", "fork", "checkout", "restore", "replay", "compare", "export", "close", "discover", "bind_symbol", "replay_observation",
                    *(["stage_input"] if self.family == "mamba" else []),
                    *(["measure_circuit", "replay_circuit"] if self.family == "mamba" and self.publication.mode == "journal" else [])],
                "circuit_measurement": {"protocol": "saturn-bench-circuit-measurement-v1", "available": self.family == "mamba" and self.publication.mode == "journal",
                    "publication_mode_required": "journal",
                    "observable": "token_margin", "start_boundary": "committed-token", "memory_mediation": self.family == "mamba",
                    "semantic_reasoning_detector": False, "memory_policy": "complete recurrent pages and flags; fixed recipient input"
                        if self.family == "mamba" else "not-qualified"},
                "integrations": {"model_os": "native-transition-admission", "virtual_memory": "durable-state-pages",
                    "future_forest": "persistent", "program_recorder": "persistent", "source_frames": "attached",
                    "hyperion": "context-bound-discovery-and-held-consumers"},
                "publication": {"mode": self.publication.mode, "schema": "saturn-bench-publication-v1",
                    "requested_cuts_durable": True, "recovery": "sealed-pack-before-selected-pointer"}}

    def _validate(self, state):
        self.adapter.validate(state)
        if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in tensors(state)):
            raise BenchError("non-finite native state cannot publish")
        self.pages.admit(state)

    def _package(self, address=ADDRESS):
        schema = StateSchema(name=f"bench-{self.family}-closure", version=2,
            slots=(StateSlot(address=address, value_type="tensor-tree-reference", owner=self.family,
                read_scopes=("model-os",), write_scopes=("model-os",), branchable=True, persistable=True),))
        consumer = ConsumerContract(consumer_id=f"bench.{self.family}.native", reads=(address,),
            output_kind="native-boundary-receipt", continuation_tokens=0, native=True)
        abi = ModelABI(model_id=self.identity["model_id"], revision=self.identity["revision"],
            family=self.family, backend="native", state_schema=schema.fingerprint,
            device=str(self.adapter.device), dtype=self.identity["dtype"],
            numerical_contract=self.adapter.contract, checkpoint_sha256=self.identity["weights_sha256"],
            runtime_fingerprint=self.identity["runtime_sha256"], native_consumers=(consumer.consumer_id,))
        program = ModelProgram(program_id=f"bench.{self.family}.transition", version=2,
            instructions=(ProgramInstruction(instruction_id="transition", op="Write", reads=(address,),
                writes=(address,), outputs=(address,), authority="model-os"),), portable=False, terminal_claim=False)
        return ModelPackage(package_id=f"bench.{self.family}", version=2, model_abi=abi, state_schema=schema,
            state_surface=StateSurface.from_schema(schema, surface_id="bench-runtime", owner=self.family),
            program=program, consumers=(consumer,), permissions={"subjects": {"model-os": {
                "read": [address], "write": [address]}}, "authorities": ["model-os"]})

    def _transaction(self, command, callback, *, verifier=None):
        address = "bench."+command["path"] if command.get("op") == "typed-edit" else ADDRESS
        package = self._package(address)
        backend = NativeTransitionBackend(model_abi=package.model_abi, state_schema=package.state_schema,
            initial_state={address: self._encoded_state})
        backend.address = address
        backend.owner, backend.command, backend.callback = self, command, callback
        kernel = ModelOSKernel.install(package, backend)
        with self.metrics.measure("model_os_transaction"):
            result = kernel.execute(updates={address: command}, horizon=0, verifier=verifier,
                metadata={"session_id": self.session_id, "branch": self.active,
                          "parent_cut": self.branches.get(self.active).cut if self.active in self.branches else None})
        receipt = self.artifacts.put(result.receipt.to_payload())
        if result.receipt.decision == "commit":
            self.state, self._encoded_state = backend.state, backend.encoded
        return backend.output, receipt, result.receipt

    def _manifest(self):
        manifest = {"schema": SCHEMA, "session_id": self.session_id, "identity": self.identity, "family": self.family,
            "lane": self.lane, "capacity": self.capacity, "resident_state_bytes": self.pages.budget,
            "numerical_contract": self.adapter.contract, "active": self.active,
            "branches": {name: vars(b) for name, b in self.branches.items()}, "cuts": self.cuts,
            "events": self.events, "memory_events": self.pages.events, "capabilities": self.capabilities(),
            "source_root": self.source_root, "publication_mode": self.publication.mode}
        if self.legacy_numerical_environment is not None:
            manifest["legacy_numerical_environment"] = self.legacy_numerical_environment
        return manifest

    def _publish(self):
        ref = self.artifacts.put(self._manifest())
        self.publication.select(ref)
        return ref

    @serialized
    def prefill(self, prompt, *, branch="main"):
        if self.family != "mamba": raise BenchError("diffusion uses prepare")
        return self._initialize(branch, lambda: self.adapter.prefill(prompt), {"prompt": str(prompt)})

    @serialized
    def prepare(self, prompt, *, branch="main", seed=0, height=512, width=512, num_inference_steps=4, guidance_scale=1.0):
        if self.family == "mamba": raise BenchError("autoregressive models use prefill")
        arguments = dict(prompt=prompt, seed=seed, height=height, width=width,
                         num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        return self._initialize(branch, lambda: self.adapter.prepare(**arguments), arguments)

    def _initialize(self, name, callback, command):
        if self.branches: raise BenchError("initial prompt already prepared; fork its cut")
        self.active = name; self.branches[name] = FamilyBranch(name)
        try:
            _, receipt, _ = self._transaction({"op": "prepare", **command}, lambda: (callback(), {"prepared": True}))
            self.recorder.add(prompts=command.get("prompt", ""), program={"kind": "bench-prepare",
                "family": self.family, "identity": self.identity}, label=name); self.recorder.write()
            return self.checkpoint("prepare", receipt_ref=receipt)
        except Exception:
            self.branches.clear(); self.state = self._encoded_state = None; raise

    @serialized
    @torch.inference_mode()
    def step(self, kind=None, count=1):
        if self._edit: raise BenchError("commit or abort the pending Act first")
        kind = kind or self.adapter.default_step
        if kind not in self.adapter.stages or type(count) is not int or count < 1: raise BenchError("invalid step kind/count")
        results = []
        for _ in range(count):
            parent = self.branch.cut
            encoded = self.branch.state
            try:
                def execute():
                    with self.metrics.measure("decode_and_private_copy"):
                        if hasattr(self.adapter, "private_state"):
                            private = self.adapter.private_state(self.state, kind)
                        else: private = self.pages.decode(encoded, private=True)
                    with self.metrics.measure("native_transition"):
                        output = self.adapter.advance(private, kind)
                    return private, output
                output, receipt, _ = self._transaction({"op": "step", "kind": kind}, execute)
                cut = self.checkpoint(kind, receipt_ref=receipt)
                row = {**output, "cut": cut, "receipt_ref": receipt, "cursor": self.cursor()}
                if self.last_logits is not None: row["logits_sha256"] = tensor_digest(self.last_logits)
                results.append(row)
                if self.safe_point:
                    c = self.cursor(); self.safe_point("layer" if c.get("next_layer") is not None else kind,
                        c.get("next_layer") if c.get("next_layer") is not None else c.get("token", c.get("step", 0)),
                        {"branch": self.active, "cursor": c})
            except Exception:
                self._restore(parent); raise
        return results[0] if count == 1 else results

    @serialized
    def generate(self, max_new_tokens=None):
        if self.family == "mamba":
            count = 32 if max_new_tokens is None else max_new_tokens
            if type(count) is not int or count < 0: raise BenchError("invalid token budget")
            return [self.step("token") for _ in range(count)]
        if max_new_tokens is not None: raise BenchError("diffusion generation uses its prepared schedule")
        rows = []
        while self.cursor()["phase"] != "complete": rows.append(self.step())
        return rows

    @serialized
    def stage_input(self, token_id):
        """Admit a caller-supplied AR token without publishing token history.

        Scientific query suffixes are explicit inputs, separate from autonomous
        generation. The resulting embedding cut can be intervened on and replayed.
        """
        if self.family != "mamba" or self._edit or self.state["execution"] is not None:
            raise BenchError("input selection requires a committed AR boundary")
        if type(token_id) is not int or not 0 <= token_id < self.model.config.vocab_size:
            raise BenchError("invalid supplied input token")
        parent = self.branch.cut
        def execute():
            private = self.adapter.private_state(self.state, "embedding")
            private["pending"] = token_id
            return private, self.adapter.advance(private, "embedding")
        try:
            _, receipt, _ = self._transaction({"op": "stage-input", "token_id": token_id,
                "input_authority": "caller-supplied", "autonomous": False}, execute)
            self.events.append({"kind": "supplied-input", "token_id": token_id, "parent_cut": parent})
            return self.checkpoint("supplied-input-embedding", receipt_ref=receipt)
        except BaseException:
            self._restore(parent); raise

    @serialized
    def checkpoint(self, label=None, *, receipt_ref=None):
        branch = self.branch
        if self._encoded_state is None:
            with self.metrics.measure("encode"):
                self._encoded_state = self.pages.encode(self.state)
        encoded = self._encoded_state
        snapshot = {"schema": CUT_SCHEMA, "family": self.family, "identity": self.identity,
            "lane": self.lane, "numerical_contract": self.adapter.contract, "mutable_closure": list(self.adapter.closure),
            "parent_cut": branch.cut, "branch": self.active, "label": label or "cut", "state": encoded,
            "cursor": self.cursor(), "symbols": branch.symbols, "intervention_lineage": branch.intervention_lineage,
            "receipt_ref": receipt_ref}
        ref = self.artifacts.put(snapshot)
        pointer = VMCachePointer(kind="artifact_ref", ref=ref, role="native-family-closure")
        bundle = self.emitter.emit(run_id=content_digest({"cut": ref}), parent_identity={"cut": branch.cut or "prepare"},
            graph_fingerprint=self.model_abi, model_identity=self.identity,
            input_identity={"context": content_digest(encoded)}, abi_identity={"contract": self.adapter.contract},
            entries={plane: [VMCacheEntry(entry_id=plane, pointers=(pointer,), role=f"{self.family}-{plane}")]
                     for plane in ("source", "address", "operator", "renderer")},
            metadata={"session_id": self.session_id, "cut": ref, "receipt_ref": receipt_ref})
        obj = self.forest.put_json_object({"cut": ref}, media_type="application/json")
        node = self.forest.fork(branch.node, transition={"kind": label or "cut"}, diff_refs={"cut": obj}) if branch.node else self.forest.create_root(state_refs={"cut": obj}, metadata={"session_id": self.session_id})
        previous = branch.state, branch.cut, branch.node
        old_registry = self.cuts.get(ref)
        old_source = self.source_frames.current
        old_root = self.source_root
        branch.state, branch.cut, branch.node = encoded, ref, node
        self.cuts[ref] = {"label": label, "branch": self.active, "node": node, "cursor": self.cursor(),
                          "bundle_ref": bundle.reference}
        try:
            c = self.cursor()
            # Execution history is in the cut registry and Future Forest. A
            # token/block transition does not introduce a new lexical scope.
            # Every live source view inherits the one immutable prompt root.
            if self.source_root: self.source_frames.checkout(self.source_root)
            source_parent = self.source_frames.inspect()["statecut_ref"] if self.source_frames.current else None
            live = self.source_frames.cuts.capture(cut_id=f"bench-{content_digest({'cut': ref, 'session': self.session_id})[:24]}",
                family=self.family, state_schema_id=CUT_SCHEMA, model_identity=self.identity,
                boundary=c["phase"], index=c.get("token", c.get("step", 0)), total=c.get("total_steps", self.capacity),
                state={"session_id": self.session_id, "bench_cut": ref, "cursor": c}, parent_cut=source_parent)
            artifacts = self._source_artifacts() if source_parent is None else {}
            self.source_frames.push(label=label or "cut", statecut_ref=live.ref,
                                    artifacts=artifacts,
                                    evidence={"session_id": self.session_id, "bench_cut": ref})
            if self.source_root is None: self.source_root = self.source_frames.current
            self.cuts[ref]["source_frame"] = self.source_frames.current
            self.pages.admit(self.state); self._publish()
        except Exception:
            branch.state, branch.cut, branch.node = previous
            if old_registry is None: self.cuts.pop(ref, None)
            else: self.cuts[ref] = old_registry
            self.source_root = old_root
            self.source_frames._checkout(old_source); raise
        return ref

    def _source_artifacts(self):
        if self.family == "mamba":
            from .ar.mamba_state_program import MambaStateProgram, MambaLayerState
            from .source_program_surface import artifact_from_mamba_state_program
            program = MambaStateProgram(model_id=self.identity["model_id"], revision=self.identity["revision"],
                source_token_ids=tuple(self.state["token_ids"]), pending_token_id=self.state["pending"],
                layers=tuple(MambaLayerState(p["conv"], p["ssm"]) for p in self.state["pages"]))
            artifact = artifact_from_mamba_state_program(program, store_id="prompt-state")
        else:
            from .source_program_surface import (SourceProgramSurface, SourceProgramArtifact,
                SourceRecordDescriptor, _component, build_raw_workspace_artifact)
            values = {f"conditioner/{i}": t.detach().cpu().contiguous()
                      for i, t in enumerate(tensors(self.state["checkpoint"])) if t.ndim > 0}
            records = tuple(SourceRecordDescriptor(address=key,
                components=(_component("native-source", i, key, t),)) for i, (key, t) in enumerate(values.items()))
            surface = SourceProgramSurface(store_id="prompt-state", family=self.family,
                model_id=self.identity["model_id"], revision=self.identity["revision"],
                source_abi=self.adapter.contract, native_consumer=f"bench.{self.family}.native",
                source_fingerprint=content_digest(self.pages.encode(self.state["checkpoint"])), records=records)
            encoded, reference, manifest = build_raw_workspace_artifact(values,
                scope={"kind": "source-program", "family": self.family,
                       "source_program_surface_fingerprint": surface.fingerprint})
            artifact = SourceProgramArtifact(surface, encoded, reference, manifest)
        return {"prompt-state": artifact}

    def _restore(self, ref, *, branch=None):
        cut = self.artifacts.get(ref)
        if (cut.get("schema") not in {CUT_SCHEMA, "saturn-open-bench-cut-v2"} or cut["identity"] != self.identity or cut["family"] != self.family
            or cut["lane"] != self.lane or cut["numerical_contract"] != self.adapter.contract
            or cut["mutable_closure"] != list(self.adapter.closure)):
            raise BenchError("incompatible family/checkpoint/numerical lane/mutable closure")
        state = self.pages.decode(cut["state"])
        self._validate(state)
        name = branch or self.active
        self.branches[name] = FamilyBranch(name, cut["state"], ref, self.cuts.get(ref, {}).get("node"),
                                          dict(cut["symbols"]), list(cut["intervention_lineage"]))
        self.active, self.state = name, state
        self._encoded_state = cut["state"]
        if hasattr(self.adapter, "restore_external"): self.adapter.restore_external(state)
        source = self.cuts.get(ref, {}).get("source_frame")
        if source: self.source_frames.checkout(source)
        return {"cut_id": ref, "session_id": self.session_id, "cursor": self.cursor()}

    @serialized
    def restore(self, ref):
        if self._edit: raise BenchError("pending Act")
        previous = self.branch.cut
        try:
            result = self._restore(ref); self._publish(); return result
        except Exception:
            self._restore(previous); raise

    @serialized
    def fork(self, name, *, cut=None):
        if not name or name in self.branches or self._edit: raise BenchError("unique branch and no pending Act required")
        ref = cut or self.checkpoint("fork-parent")
        old_name, old_cut = self.active, self.branch.cut
        try:
            self.state = None; self._restore(ref, branch=name)
            self.events.append({"kind": "fork", "parent": ref, "branch": name})
            return self.checkpoint("fork")
        except Exception:
            self.branches.pop(name, None); self.active = old_name; self._restore(old_cut); raise

    @serialized
    def checkout(self, name):
        if name not in self.branches or self._edit: raise BenchError("unknown branch or pending Act")
        self.checkpoint("checkout-parent")
        old_name, old_cut = self.active, self.branch.cut
        try:
            self.state = None; self.active = name
            return self.restore(self.branch.cut)
        except Exception:
            self.active = old_name; self._restore(old_cut); raise

    @serialized
    def replay(self, ref, max_new_tokens=None):
        self.restore(ref)
        return self.generate(max_new_tokens)

    @serialized
    def compare(self, left, right):
        a, b = self.artifacts.get(left), self.artifacts.get(right)
        if any(a[k] != b[k] for k in ("identity", "lane", "family", "numerical_contract")):
            raise BenchError("compare requires compatible cuts")
        return {"left": left, "right": right, "same_state": a["state"] == b["state"],
                "same_cursor": a["cursor"] == b["cursor"]}

    @serialized
    def resolve_value(self, path): return self.adapter.value(self.state, path).detach().clone()

    @serialized
    def inspect(self, path=None):
        if path in {None, "status"}:
            return {"session_id": self.session_id, "branch": self.active, "cut_id": self.branch.cut,
                    "cursor": self.cursor(), "capabilities": self.capabilities()}
        if path == "cursor": return self.cursor()
        if path == "performance":
            return {**self.metrics.inspect(), "publication": dict(self.publication.stats),
                    "pages": dict(getattr(self.pages, "stats", {}))}
        if path == "frames": return self.source_frames.debug_view()
        if path == "measurements":
            from .bench_circuit_measurement import inspect_measurements
            return inspect_measurements(self)
        if path == "token_ids" and self.family == "mamba": return {"value": self.state["token_ids"][:]}
        if path == "memory":
            result = self.pages.inspect({n: b.state for n, b in self.branches.items()}, self.state)
            return {**result, "cuts": len(self.cuts), "weights_bytes": sum(p.numel()*p.element_size() for _, p in self.parameters),
                "weights_in_state_budget": False, "workspace": {"cuda_allocated_bytes":
                    torch.cuda.memory_allocated(self.adapter.device) if self.adapter.device.type == "cuda" else None,
                    "includes_weights_state_and_workspace": True}}
        if path in {"symbols", "circuits"}:
            from .bench_discovery import binding_status
            result = {name: {"report": self.artifacts.get(ref), "status": binding_status(self.artifacts.get(ref)["symbol"]["context"],
                self.artifacts.get(ref)["symbol"]["context"].get("binding_context", self.artifacts.get(ref)["symbol"]["context"]) == self.symbol_context()),
                "qualification": self.artifacts.get(ref)["symbol"]["context"].get("qualification", {"quality_status": "candidate-trend", "semantic_binding": False})}
                for name, ref in self.branch.symbols.items()}
            if path == "symbols": result["_structural"] = {"status": "unmeasured-structure", "family": self.family,
                "names": list(self.adapter.stages), "runtime_addresses": list(self.adapter.writable), "cursor": self.cursor()}
            return result
        value = self.resolve_value(path)
        return {"path": path, "shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_digest(value),
                "stats": {"norm": float(value.float().norm()), "mean": float(value.float().mean())}}

    @serialized
    def preview_edit(self, path, value, *, mode="replace"):
        if self._edit: raise BenchError("pending Act")
        old = self.resolve_value(path)
        candidate = torch.as_tensor(value, dtype=old.dtype, device=old.device)
        if candidate.numel() == 1: candidate = candidate.expand_as(old)
        if candidate.shape != old.shape or not bool(torch.isfinite(candidate).all()): raise BenchError("typed edit shape/finite contract")
        candidate = old+candidate if mode == "add" else old*candidate if mode == "scale" else candidate if mode == "replace" else None
        if candidate is None or not bool(torch.isfinite(candidate).all()): raise BenchError("invalid edit mode or result")
        parent = self.checkpoint("act-parent")
        # This Act replaces one address. Private wrappers isolate assignment;
        # unchanged leaves remain read-only inputs to subsequent native steps.
        from .bench_pages import private_tree
        private = private_tree(self.state, readonly_ids={id(t) for t in tensors(self.state)})
        self.adapter.edit(private, path, candidate.clone()); self._validate(private)
        payload = self.pages.encode(private, previous_state=self.state, previous_encoded=self._encoded_state,
            readonly_ids={id(t) for t in tensors(self.state)})
        act_id = content_digest({"parent": parent, "path": path, "payload": payload, "mode": mode})
        self._edit = {"act_id": act_id, "parent": parent, "path": path, "payload": payload, "mode": mode}
        return {"act_id": act_id, "parent_cut": parent, "candidate_sha256": content_digest(payload),
                "plan": {"writes": [path], "family": self.family, "authority": "model-os"}}

    @serialized
    def commit_edit(self, act_id):
        if not self._edit or act_id != self._edit["act_id"]: raise BenchError("Act identity mismatch")
        edit, self._edit = self._edit, None
        try:
            _, receipt, _ = self._transaction({"op": "typed-edit", "path": edit["path"], "act_id": act_id},
                lambda: (self.pages.decode(edit["payload"]), {"act_id": act_id, "path": edit["path"]}))
            self.branch.intervention_lineage.append(act_id)
            cut = self.checkpoint("act-committed", receipt_ref=receipt)
            return {"act_id": act_id, "cut_id": cut, "receipt_ref": receipt}
        except Exception:
            self._restore(edit["parent"]); raise

    @serialized
    def abort_edit(self, act_id):
        if not self._edit or act_id != self._edit["act_id"]: raise BenchError("Act identity mismatch")
        edit, self._edit = self._edit, None
        _, receipt, _ = self._transaction({"op": "abort-edit", "act_id": act_id},
            lambda: (self.pages.decode(edit["payload"]), {"act_id": act_id}), verifier=lambda _: False)
        self.events.append({"kind": "act-rejected", "act_id": act_id, "receipt_ref": receipt})
        return self.restore(edit["parent"])

    def symbol_context(self):
        return {"model_abi": self.model_abi, "family": self.family, "lane": self.lane,
                "context": content_digest(self.branch.state), "intervention_lineage": self.branch.intervention_lineage[:]}

    @serialized
    def bind_symbol(self, report, *, name=None):
        from .hyperion import verify_hyperion_report
        verify_hyperion_report(report)
        context = report["symbol"]["context"]
        if context.get("binding_context", context) != self.symbol_context(): raise BenchError("symbol outside measured context")
        name = name or report["symbol"]["logical_symbol"]
        ref = self.artifacts.put(report)
        before = dict(self.branch.symbols); self.branch.symbols[name] = ref
        try: self._publish()
        except Exception: self.branch.symbols = before; raise
        return ref

    @serialized
    def discover(self, spec):
        from .bench_family_symbols import discover
        return discover(self, spec)

    @serialized
    def replay_observation(self, evidence_ref):
        from .bench_family_symbols import replay_observation
        return replay_observation(self, evidence_ref)

    @serialized
    def measure_circuit(self, spec):
        from .bench_circuit_measurement import atomic_measurement, measure_circuit
        return atomic_measurement(self, measure_circuit, spec)

    @serialized
    def replay_circuit(self, evidence_ref):
        from .bench_circuit_measurement import atomic_measurement, replay_circuit
        return atomic_measurement(self, replay_circuit, evidence_ref)

    def _circuit_memory_act(self, donor_cut):
        """Internal, complete Mamba memory interchange through Model OS.

        The recipient retains its output, pending token and history. A fixed
        supplied future input is required by the measurement protocol. The
        adapter validates the donor's complete native cache, including flags.
        """
        if self.family != "mamba" or self._edit or self.state["execution"] is not None:
            raise BenchError("successor memory interchange requires committed Mamba state")
        cut = self.artifacts.get(donor_cut)
        if (cut.get("schema") not in {CUT_SCHEMA, "saturn-open-bench-cut-v2"}
            or cut["identity"] != self.identity or cut["family"] != self.family or cut["lane"] != self.lane
            or cut["numerical_contract"] != self.adapter.contract or cut["mutable_closure"] != list(self.adapter.closure)):
            raise BenchError("incompatible successor memory closure")
        donor = self.pages.decode(cut["state"])
        self._validate(donor)
        self.adapter.validate_memory_transplant(self.state, donor)
        parent = self.checkpoint("successor-memory-parent")
        act_id = content_digest({"parent": parent, "donor_cut": donor_cut,
            "writes": "complete-successor-memory", "recipient_policy": "preserve-history-pending-logits"})
        try:
            _, receipt, decision = self._transaction({"op": "typed-edit", "path": "successor_memory",
                "act_id": act_id, "donor_cut": donor_cut,
                "recipient_policy": "preserve-history-pending-logits"},
                lambda: (self.adapter.transplant_memory(self.state, donor), {"act_id": act_id,
                    "writes": ["conv_state", "ssm_state", "cache_flags"], "donor_cut": donor_cut}))
            if decision.decision != "commit": raise BenchError("successor memory Act did not commit")
            self.branch.intervention_lineage.append(act_id)
            result = self.checkpoint("successor-memory-committed", receipt_ref=receipt)
            return {"act_id": act_id, "cut_id": result, "receipt_ref": receipt,
                "parent_cut": parent, "donor_cut": donor_cut,
                "plan": {"writes": ["conv_state", "ssm_state", "cache_flags"], "authority": "model-os"}}
        except BaseException:
            self._restore(parent)
            raise

    @serialized
    def formed_consumer(self):
        if self.family != "mamba": raise BenchError("formed recurrent consumer is Mamba-specific")
        value, receipt = self.adapter.formed_consumer(self.state, model_abi=self.model_abi, identity=self.identity)
        ref = self.artifacts.put(receipt)
        return {"value": value, "receipt_ref": ref, "session_id": self.session_id, "cut_id": self.branch.cut}

    @serialized
    def export(self, destination):
        self.checkpoint("export")
        target = Path(destination).expanduser().resolve()
        if target.exists() or self.root == target or self.root in target.parents: raise BenchError("independent absent destination required")
        shutil.copytree(self.root, target)
        return {"root": str(target), "cut": self.branch.cut, "manifest": json.loads((target/"selected.json").read_text())["manifest"]}

    @classmethod
    def resume(cls, model, tokenizer=None, *, root, family, model_sha256=None, _writer=None,
               _manifest=None, _adapter=None, legacy_numerical_environment=None):
        root = Path(root)
        if _writer is None:
            _writer, selected = recover_locked(root)
        else: selected = json.loads((root/"selected.json").read_text())
        try: manifest = _manifest or DirectoryArtifactStore(root/"artifacts").get(selected["manifest"])
        except BaseException: _writer.close(); raise
        if manifest.get("schema") not in {SCHEMA, "saturn-open-bench-v2"} or manifest["family"] != family:
            _writer.close(); raise BenchError("incompatible session family/schema")
        saved_qualification = manifest.get("legacy_numerical_environment")
        if saved_qualification is not None and legacy_numerical_environment is not None and saved_qualification != legacy_numerical_environment:
            _writer.close(); raise BenchError("legacy numerical qualification changed")
        qualification = legacy_numerical_environment if legacy_numerical_environment is not None else saved_qualification
        if qualification is not None and (family != "mamba" or "numerical_environment" in manifest["identity"]):
            _writer.close(); raise BenchError("legacy qualification requires an older Mamba identity")
        result = cls(model, tokenizer, root=root, family=family, model_id=manifest["identity"]["model_id"],
            revision=manifest["identity"]["revision"], model_sha256=model_sha256,
            session_id=manifest["session_id"], lane=manifest["lane"], resident_state_bytes=manifest["resident_state_bytes"],
            continuation_capacity=manifest["capacity"], publication_mode=manifest.get("publication_mode", "journal"),
            _writer=_writer, _adapter=_adapter, _legacy_numerical_environment=qualification)
        try:
            if result.identity != manifest["identity"]:
                changed = {k: {"expected": manifest["identity"].get(k), "actual": result.identity.get(k)}
                           for k in result.identity if result.identity[k] != manifest["identity"].get(k)}
                raise BenchError(f"resident checkpoint/runtime identity mismatch: {changed}")
            result.branches = {n: FamilyBranch(**b) for n, b in manifest["branches"].items()}
            result.cuts, result.events, result.active = manifest["cuts"], manifest["events"], manifest["active"]
            result.pages.events = manifest.get("memory_events", [])
            with result.publication.transaction(): result._restore(result.branch.cut)
            result.source_root = manifest.get("source_root") or result.source_frames.chain()[-1]["node_id"]
            return result
        except Exception: result._writer.close(); raise

    def close(self):
        with self._lock:
            if self.closed: return
            with self.publication.transaction():
                if self._edit: self.abort_edit(self._edit["act_id"])
                if self.branches: self.checkpoint("close")
                if self._owns_adapter and hasattr(self.adapter, "close"): self.adapter.close()
                self.state = None; self.pages.pool.clear(); self.closed = True
            self._writer.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
