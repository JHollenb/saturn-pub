"""Bounded native assays with Model OS admission and durable tensor cuts.

This is the research-adapter seam, not an additional BenchSession family. The
caller declares the complete consumer input, writable addresses, validation and
numerical identity. Models remain caller-owned; this module never acquires a
model or submits work. Outputs are summarized by default.
"""
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import uuid4

import torch

from .bench_pages import BenchPages, tensors
from .bench_publication import BenchPublication
from .circuit_runtime import DirectoryArtifactStore
from .model_os_contracts import (
    ConsumerContract,
    ModelABI,
    ModelPackage,
    ModelProgram,
    ProgramInstruction,
    StateSchema,
    StateSlot,
    StateSurface,
    content_digest,
)
from .model_os_kernel import InMemoryModelBackend, ModelOSKernel

SCHEMA = "mdb-native-research-session-v1"


def tensor_sha256(tensor):
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return sha256(value.numpy().tobytes()).hexdigest()


class _ResearchBackend(InMemoryModelBackend):
    def continue_execution(self, candidate, *, consumer, horizon):
        super().continue_execution(candidate, consumer=consumer, horizon=horizon)
        state = self._states[candidate.handle]
        # No mutable native storage is shared with the parent or its siblings.
        private = self.owner.pages.decode(state, private=True)
        self.owner._validate(private)
        self.owner.pages.admit(private)
        self.output, observation = self.owner.consumer(private)
        if not isinstance(observation, dict):
            raise TypeError("native consumer observation must be a dictionary")
        self.owner.pages.admit()
        return {"accepted": True, "execution_mode": "declared-native-research-consumer",
                "consumer_id": consumer.consumer_id, "horizon": horizon,
                "observation": observation}


class NativeResearchSession:
    """A serialized finite native consumer with tensor-addressed common parents.

    ``consumer(state)`` returns ``(private_output, JSON observation)``. It must
    consume only the declared state plus the frozen model/instrument identified
    in ``identity``. A cut captures its inputs; it does not claim to checkpoint
    arbitrary hidden state inside an external instrument or a loaded model.
    """
    def __init__(self, *, root, identity, initial_state, writable, consumer,
                 validator=None, device="cpu", resident_bytes=64 << 20):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        required = {"model_id", "weights_sha256", "family", "numerical_contract"}
        if not required.issubset(identity):
            raise ValueError("native research identity requires " + str(sorted(required)))
        self.session_id = "research-" + uuid4().hex
        self.closed, self._lock = False, RLock()
        self.consumer, self.validator = consumer, validator
        self.addresses = tuple(sorted(initial_state))
        self.writable = tuple(sorted(writable))
        if not set(self.writable).issubset(self.addresses):
            raise ValueError("writable addresses must be in the declared closure")
        self.publication = BenchPublication(self.root, mode="journal")
        with self.publication.initializing():
            self.pages = BenchPages(self.root / "pages", device=device, budget=resident_bytes)
            self.artifacts = DirectoryArtifactStore(self.root / "artifacts")
        with self.publication.transaction():
            self._initialize(initial_state, device)

    def _initialize(self, initial_state, device):
        identity = self.identity
        self._validate(initial_state)
        self.pages.admit(initial_state)
        encoded = self.pages.encode(initial_state)
        schema = StateSchema(name=SCHEMA, version=1, slots=tuple(StateSlot(
            address=address, value_type="native-research-tensor-reference", owner="native-adapter",
            read_scopes=("model-os",), write_scopes=("model-os",) if address in self.writable else (),
            branchable=True, persistable=True,
        ) for address in self.addresses))
        consumer_id = "mdb.research." + identity["family"]
        abi = ModelABI(model_id=identity["model_id"], revision=identity.get("revision", "local"),
            family=identity["family"], backend="native-research-adapter",
            state_schema=schema.fingerprint, device=str(device), dtype=identity.get("dtype", "declared"),
            numerical_contract=identity["numerical_contract"], checkpoint_sha256=identity["weights_sha256"],
            runtime_fingerprint=content_digest(identity), native_consumers=(consumer_id,))
        native = ConsumerContract(consumer_id=consumer_id, reads=self.addresses,
            output_kind="native-assay-observation", native=True, continuation_tokens=0)
        program = ModelProgram(program_id=consumer_id, version=1, instructions=(ProgramInstruction(
            instruction_id="bounded-intervention", op="Write", reads=self.addresses,
            writes=self.writable, outputs=self.writable, authority="model-os"),),
            portable=False, terminal_claim=False)
        package = ModelPackage(package_id=consumer_id, version=1, model_abi=abi, state_schema=schema,
            state_surface=StateSurface.from_schema(schema, surface_id="research-inputs", owner="native-adapter"),
            program=program, consumers=(native,), permissions={"subjects": {"model-os": {
                "read": list(self.addresses), "write": list(self.writable)}}, "authorities": ["model-os"]})
        self.backend = _ResearchBackend(model_abi=abi, state_schema=schema, initial_state=encoded)
        self.backend.owner = self
        self.kernel = ModelOSKernel.install(package, self.backend)
        self.cuts = {"parent": self.backend.current_statecut}
        self.receipts, self.evidence = [], []
        self._publish()

    def _validate(self, state):
        if set(state) != set(self.addresses):
            raise ValueError("native input closure changed addresses")
        for tensor in tensors(state):
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("non-finite native input")
        if self.validator:
            self.validator(state)

    def _require_open(self):
        if self.closed:
            raise RuntimeError("native research session is closed")

    def _publish(self):
        manifest = {"schema": SCHEMA, "session_id": self.session_id, "identity": self.identity,
            "addresses": self.addresses, "writable": self.writable,
            "cuts": {name: {"cut": cut.to_payload(), "state": self.backend._states[cut.handle]}
                     for name, cut in self.cuts.items()},
            "receipts": self.receipts, "evidence": self.evidence,
            "limitations": ["declared native consumer input cuts", "external instrument internal state is not checkpointed",
                            "interactive BenchSession family qualification is separate"]}
        reference = self.artifacts.put(manifest)
        self.publication.select(reference)
        return reference

    def capture(self, name, state, *, parent="parent", metadata=None):
        with self._lock, self.publication.transaction():
            self._require_open()
            if name in self.cuts:
                raise ValueError("cut name already exists")
            self._validate(state)
            self.pages.admit(state)
            encoded = self.pages.encode(state)
            cut = self.backend._new_cut(parent_handle=self.cuts[parent].handle, state=encoded,
                status="captured", metadata=metadata or {})
            self.cuts[name] = cut
            try:
                return self._publish()
            except Exception:
                del self.cuts[name]
                raise

    def run(self, name, *, parent="parent", updates=None, commit=True, metadata=None):
        with self._lock, self.publication.transaction():
            self._require_open()
            if name in self.cuts:
                raise ValueError("cut name already exists")
            updates = updates or {}
            denied = set(updates) - set(self.writable)
            if denied:
                raise ValueError("read-only or undeclared native addresses: " + str(sorted(denied)))
            parent_cut = self.cuts[parent]
            encoded = deepcopy(self.backend._states[parent_cut.handle])
            # Validate the whole proposed closure before any consumer runs.
            private = self.pages.decode(encoded, private=True)
            private.update(updates)
            self._validate(private)
            self.pages.admit(private)
            encoded_updates = {address: self.pages.encode(private[address]) for address in updates}
            del private
            current = self.kernel.current_statecut
            self.kernel.restore(parent_cut)
            result = self.kernel.execute(parent=parent_cut, updates=encoded_updates,
                metadata={"session_id": self.session_id, "arm": name, **(metadata or {})},
                verifier=lambda _: {"accepted": bool(commit), "reason": "retain observation; explicit branch decision"})
            cut = self.backend._cuts[result.candidate.handle]
            self.cuts[name] = cut
            receipt = self.artifacts.put(result.receipt.to_payload())
            self.receipts.append(receipt)
            try:
                manifest = self._publish()
            except Exception:
                self.kernel.restore(current)
                del self.cuts[name]
                self.receipts.pop()
                raise
            return self.backend.output, {"session_id": self.session_id, "cut": cut.to_payload(),
                "receipt": receipt, "manifest": manifest,
                "observation": result.receipt.continuation["observation"]}

    def restore(self, name):
        with self._lock:
            self._require_open()
            return self.kernel.restore(self.cuts[name])

    def inspect_memory(self):
        with self._lock:
            self._require_open()
            return {**self.pages.inspect({name: self.backend._states[cut.handle]
                                         for name, cut in self.cuts.items()}, None),
                    "publication_stats": dict(self.publication.stats)}

    def close(self):
        with self._lock:
            if not self.closed:
                with self.publication.transaction():
                    self._publish()
                self.pages.pool.clear()
                self.pages.pool_refs.clear()
                self.closed = True
