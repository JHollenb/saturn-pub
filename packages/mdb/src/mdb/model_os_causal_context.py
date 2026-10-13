"""Model OS transaction bridge for exact-history Qwen causal-context mounts.

The portable package exposes identities, permissions, and token decisions. The
radix bytes and resident tensors stay private to the native backend. Every
preview forks only its private delta; commit publishes that branch and abort
discards it without mutating its parent or shared ancestor.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any

import torch

from mdb.ar.causal_context_qwen import CausalContextQwenMount, CausalContextQwenRuntime
from mdb.model_os_contracts import (
    ConsumerContract,
    ContinuationContract,
    ModelABI,
    ModelPackage,
    ModelProgram,
    ProgramInstruction,
    StateSchema,
    StateSlot,
    StateSurface,
    content_digest,
)
from mdb.model_os_kernel import ModelOSExecutionError, ModelOSStateCut


CONTEXT_ADDRESS = "native.causal-context-mount"
TOKEN_ADDRESS = "decoder.input-token"
NEXT_ADDRESS = "native.next-token"
RECEIPT_ADDRESS = "native.causal-context-receipt"
CONSUMER_ID = "saturn.qwen2.causal-context-direct-read"
SCHEMA = "saturn-model-os-causal-context-qwen-v1"


def build_causal_context_package(
    runtime: CausalContextQwenRuntime,
    *,
    model_id: str,
    tokenizer_identity: str,
) -> ModelPackage:
    family = runtime.codec.model_type
    consumer_id = CONSUMER_ID if family == "qwen2" else "saturn.qwen3.causal-context-direct-read"
    schema = SCHEMA if family == "qwen2" else "saturn-model-os-causal-context-qwen3-v1"
    state_schema = StateSchema(
        name=schema,
        version=1,
        slots=(
            StateSlot(
                address=CONTEXT_ADDRESS, value_type="verified-radix-mount",
                owner="causal-context-runtime", dtype="metadata-reference",
                read_scopes=("model-os", consumer_id), write_scopes=(consumer_id,),
                lifetime="session", branchable=True, persistable=True,
                freshness="native-resident", lineage="parent-radix-root",
                metadata={"tensor_payload": "private", "mode": "exact-history"},
            ),
            StateSlot(
                address=TOKEN_ADDRESS, value_type="token-id", owner="model-os",
                dtype="int64", read_scopes=("model-os", consumer_id),
                write_scopes=("model-os",), lifetime="token", branchable=True,
                persistable=False, freshness="candidate", lineage="selected-input",
            ),
            StateSlot(
                address=NEXT_ADDRESS, value_type="token-id", owner=consumer_id,
                dtype="int64", read_scopes=("model-os",),
                write_scopes=(consumer_id,), lifetime="token", branchable=True,
                persistable=False, freshness="native-produced",
                lineage="native-consumer-output",
            ),
            StateSlot(
                address=RECEIPT_ADDRESS, value_type="statecut-receipt", owner=consumer_id,
                dtype="metadata", read_scopes=("model-os",),
                write_scopes=(consumer_id,), lifetime="frame", branchable=True,
                persistable=False, freshness="native-produced",
                lineage="native-transaction", metadata={"tensor_payload": "forbidden"},
            ),
        ),
    )
    abi = ModelABI(
        model_id=model_id,
        revision=runtime.model_abi_sha256,
        family=family, backend="causal-context-direct-read",
        state_schema=state_schema.fingerprint,
        tokenizer_identity=tokenizer_identity,
        kv_cache_schema="saturn-causal-radix-physical-v1",
        device=str(runtime.device), dtype=str(runtime.codec.dtype),
        numerical_contract=(f"{family}-bf16-hf-eager-v1" if runtime.backend == "native-eager"
                            else "qwen2-bf16-segmented-two-pass-v1"),
        checkpoint_sha256=runtime.model_abi_sha256,
        graph_identity=f"{family}-one-token-full-layer-v1",
        native_consumers=(consumer_id,),
        runtime_fingerprint=content_digest({
            "consumer": consumer_id,
            "backend": runtime.backend,
            "row_bytes": runtime.codec.row_bytes,
        }),
        source_fingerprint=runtime.model_abi_sha256,
        capabilities=("exact-history-radix-mount", "shared-ancestor-kv", "private-delta-fork"),
    )
    consumer = ConsumerContract(
        consumer_id=consumer_id,
        reads=(CONTEXT_ADDRESS, TOKEN_ADDRESS),
        output_kind="native-next-token", continuation_tokens=1,
        replay_contract="candidate", native=True,
        metadata={"runtime": "CausalContextQwenRuntime", "mode": "exact-history"},
    )
    continuation = ContinuationContract(
        continuation_id="causal-context-qwen-one-token-v1",
        consumer_id=consumer_id,
        input_state=(CONTEXT_ADDRESS, TOKEN_ADDRESS),
        output_state=(NEXT_ADDRESS, RECEIPT_ADDRESS),
        horizon=1, replay_contract="candidate", exact=False,
        metadata={"input": "one selected token", "output": "full frozen LM head"},
    )
    program = ModelProgram(
        program_id="model-os.qwen.causal-context-continuation.v1", version=1,
        instructions=(
            ProgramInstruction(
                instruction_id="read-verified-context", op="Read",
                reads=(CONTEXT_ADDRESS,), outputs=(CONTEXT_ADDRESS,),
                authority="model-os", replay_contract="candidate",
            ),
            ProgramInstruction(
                instruction_id="bind-input-token", op="Write",
                reads=(CONTEXT_ADDRESS,), writes=(TOKEN_ADDRESS,),
                outputs=(TOKEN_ADDRESS,), authority="model-os",
                dose_limit=1.0, replay_contract="candidate",
            ),
            ProgramInstruction(
                instruction_id="continue-direct-read", op="Continue",
                reads=(CONTEXT_ADDRESS, TOKEN_ADDRESS),
                outputs=(NEXT_ADDRESS, RECEIPT_ADDRESS),
                native_consumer=consumer_id,
                continuation=continuation.continuation_id,
                authority="model-os", dose_limit=1.0,
                replay_contract="candidate",
            ),
        ),
        portable=False,
        source_symbols=(
            "saturn.causal_radix_store.CausalRadixStore",
            "saturn.ar.causal_context_qwen.CausalContextQwenRuntime",
        ),
        claim_level="controlled-mechanics", terminal_claim=False,
    )
    return ModelPackage(
        package_id=f"model-os.{family}.causal-context-direct-read", version=1,
        model_abi=abi, state_schema=state_schema,
        state_surface=StateSurface.from_schema(
            state_schema, surface_id="qwen-causal-context-state", owner="model-os+saturn",
        ),
        program=program, consumers=(consumer,), continuations=(continuation,),
        permissions={
            "subjects": {"model-os": {
                "read": [CONTEXT_ADDRESS, TOKEN_ADDRESS, NEXT_ADDRESS, RECEIPT_ADDRESS],
                "write": [TOKEN_ADDRESS],
            }},
            "authorities": ["model-os"],
        },
        resource_limits={"horizon": 1, "persisted_tensor_bytes": 0},
        evidence_refs=("saturn-causal-radix-physical-v1", schema),
        terminal_claim=False,
    )


class CausalContextQwenModelBackend:
    """One branch of the shared physical mount behind Model OS StateCuts."""

    def __init__(
        self,
        runtime: CausalContextQwenRuntime,
        initial_mount: CausalContextQwenMount,
        package: ModelPackage,
        *,
        initial_cut: ModelOSStateCut | None = None,
    ) -> None:
        if initial_mount.model_abi_sha256 != runtime.model_abi_sha256:
            raise ModelOSExecutionError("initial mount belongs to another model ABI")
        self.runtime = runtime
        self.model_abi = package.model_abi
        self.state_schema = package.state_schema
        if len(package.consumers) != 1:
            raise ModelOSExecutionError("causal-context package needs one native consumer")
        self.consumer_id = package.consumers[0].consumer_id
        self._cuts: dict[str, ModelOSStateCut] = {}
        self._states: dict[str, CausalContextQwenMount] = {}
        self._tokens: dict[str, int] = {}
        self._next_tokens: dict[str, int] = {}
        self._outputs: dict[str, torch.Tensor] = {}
        self._traces: dict[str, dict[str, Any]] = {}
        self._counter = 0
        if initial_cut is None:
            genesis = self._new_cut(initial_mount, parent=None, status="committed", token=None)
        else:
            if (initial_cut.model_abi_fingerprint != self.model_abi.fingerprint
                    or initial_cut.state_schema_fingerprint != self.state_schema.fingerprint
                    or initial_cut.status not in {"committed", "captured"}
                    or initial_cut.metadata.get("logical_rows") != initial_mount.length
                    or initial_cut.metadata.get("mount_sha256") != initial_mount.mount_sha256
                    or initial_cut.metadata.get("page_id") != initial_mount.page_id):
                raise ModelOSExecutionError("durable cut does not describe the restored mount")
            try:
                self._counter = int(initial_cut.handle.rsplit("/", 1)[1].split("-", 1)[0])
            except (ValueError, IndexError) as exc:
                raise ModelOSExecutionError("durable cut has an invalid context handle") from exc
            genesis = initial_cut
            self._cuts[genesis.handle] = genesis
            self._states[genesis.handle] = initial_mount
        self._current = genesis.handle

    def _new_cut(
        self, state: CausalContextQwenMount, *, parent: str | None,
        status: str, token: int | None,
    ) -> ModelOSStateCut:
        self._counter += 1
        fingerprint = content_digest({
            "abi": self.model_abi.fingerprint,
            "mount": state.mount_sha256,
            "page": state.page_id,
            "nodes": state.node_sha256s,
            "length": state.length,
            "parent": parent,
            "token": token,
        })
        handle = f"model-os://causal-context/{self._counter:08d}-{fingerprint[:24]}"
        cut = ModelOSStateCut(
            handle=handle, model_abi_fingerprint=self.model_abi.fingerprint,
            state_schema_fingerprint=self.state_schema.fingerprint,
            state_fingerprint=fingerprint, parent_handle=parent, status=status,
            metadata={"mount_sha256": state.mount_sha256, "page_id": state.page_id,
                      "node_sha256s": list(state.node_sha256s), "logical_rows": state.length},
        )
        self._cuts[handle] = cut
        self._states[handle] = state
        return cut

    def _require(self, cut: ModelOSStateCut) -> ModelOSStateCut:
        stored = self._cuts.get(cut.handle)
        if stored is None or stored.model_abi_fingerprint != self.model_abi.fingerprint:
            raise ModelOSExecutionError("unknown or incompatible causal-context cut")
        return stored

    @property
    def current_statecut(self) -> ModelOSStateCut:
        return self._cuts[self._current]

    @property
    def current_mount(self) -> CausalContextQwenMount:
        return self._states[self._current]

    @property
    def latest_private_output(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._outputs[self._current], self._traces[self._current]

    def consume_private_output(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Move instrument tensors to the caller; receipts keep only metadata."""
        return self._outputs.pop(self._current), self._traces.pop(self._current)

    def retained_private_delta_bytes(self) -> int:
        """Count live private cut payloads, including cuts kept for rollback."""
        tensors = {
            tensor.data_ptr(): tensor
            for state in self._states.values()
            for pair in state.delta for tensor in pair
        }
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())

    def retained_mounts(self) -> tuple[CausalContextQwenMount, ...]:
        """Resident cut closure, including rollback futures, for KV admission."""
        return tuple(self._states.values())

    def mount_for_cut(self, cut: ModelOSStateCut) -> CausalContextQwenMount:
        stored = self.resolve_open_cut(cut)
        return self._states[stored.handle]

    def resolve_open_cut(self, cut: ModelOSStateCut) -> ModelOSStateCut:
        """Resolve backend-owned metadata before writing a durable reference."""
        stored = self._require(cut)
        if stored.status not in {"committed", "captured"}:
            raise ModelOSExecutionError("durable checkpoint requires an open context cut")
        return stored

    def preview_allocation_bytes(self, parent: ModelOSStateCut) -> int:
        source = self.mount_for_cut(parent)
        return source.physical_kv_bytes if any(
            other.delta is source.delta and other.delta_rows > source.delta_rows
            for other in self._states.values()
        ) else 0

    def capture_statecut(self, parent: ModelOSStateCut | None, *, metadata) -> ModelOSStateCut:
        base = self._require(parent or self.current_statecut)
        if base.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot capture an invalid context cut")
        return self._new_cut(self._states[base.handle], parent=base.handle,
                             status="captured", token=None)

    def preview(self, parent: ModelOSStateCut, plan, updates):
        base = self._require(parent)
        if base.status not in {"committed", "captured"}:
            raise ModelOSExecutionError("preview requires an open parent")
        if set(updates) != {TOKEN_ADDRESS} or TOKEN_ADDRESS not in plan.writes:
            raise ModelOSExecutionError("preview requires one declared input token")
        token = updates[TOKEN_ADDRESS]
        if isinstance(token, bool) or not isinstance(token, int):
            raise ModelOSExecutionError("input token must be an integer")
        if any(cut.status == "candidate" for cut in self._cuts.values()):
            raise ModelOSExecutionError("another candidate owns the private append slot")
        source = self._states[base.handle]
        # Older cuts may share the same buffer as later committed futures. A
        # new branch from such a cut must copy its private suffix before it
        # overwrites a row visible to a retained future.
        has_later_shared_future = any(
            other.delta is source.delta and other.delta_rows > source.delta_rows
            for handle, other in self._states.items()
            if handle != base.handle and self._cuts[handle].status != "restored"
        )
        fork = source.fork() if has_later_shared_future else source.view()
        candidate = self._new_cut(fork, parent=base.handle, status="candidate", token=token)
        self._tokens[candidate.handle] = token
        return candidate, {"accepted": True, "mount_sha256": fork.mount_sha256,
                           "page_id": fork.page_id, "logical_rows": fork.length,
                           "node_sha256s": list(fork.node_sha256s)}

    def continue_execution(self, candidate: ModelOSStateCut, *, consumer, horizon):
        stored = self._require(candidate)
        if stored.status != "candidate" or consumer.consumer_id != self.consumer_id or horizon > 1:
            raise ModelOSExecutionError("candidate or consumer contract changed")
        logits, trace = self.runtime.step(
            self._states[stored.handle], self._tokens[stored.handle], monitor=True,
        )
        self._outputs[stored.handle] = logits
        self._traces[stored.handle] = trace
        raw = logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        next_token = int(logits.argmax(dim=-1).item())
        self._next_tokens[stored.handle] = next_token
        return {"accepted": True, "consumer_id": self.consumer_id,
                "next_token_id": next_token,
                "logits_sha256": sha256(raw).hexdigest(),
                "logical_rows_after": self._states[stored.handle].length,
                "native_layers_consumed": len(trace["layers"])}

    def commit(self, candidate: ModelOSStateCut):
        stored = self._require(candidate)
        if stored.status != "candidate" or stored.handle not in self._outputs:
            raise ModelOSExecutionError("candidate has no native continuation")
        rows = self._states[stored.handle].length
        self._cuts[stored.handle] = replace(
            stored, status="committed", metadata={**stored.metadata, "logical_rows": rows},
            state_fingerprint=content_digest({
                "candidate": stored.state_fingerprint, "logical_rows": rows,
                "next_token_id": self._next_tokens[stored.handle],
            }),
        )
        previous = self._current
        self._current = stored.handle
        self._outputs.pop(previous, None)
        self._traces.pop(previous, None)
        return {"status": "committed", "current_handle": stored.handle,
                "logical_rows": self._states[stored.handle].length, "output_visible": True}

    def abort(self, candidate: ModelOSStateCut, *, reason: str):
        stored = self._require(candidate)
        if stored.status not in {"candidate", "captured"}:
            raise ModelOSExecutionError("only provisional cuts can abort")
        self._cuts[stored.handle] = replace(stored, status="restored")
        self._states.pop(stored.handle, None)
        self._outputs.pop(stored.handle, None)
        self._traces.pop(stored.handle, None)
        self._next_tokens.pop(stored.handle, None)
        return {"status": "restored", "restored_to": stored.parent_handle,
                "reason": reason, "output_visible": False}

    def restore(self, target: ModelOSStateCut):
        stored = self._require(target)
        if stored.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot restore an invalid cut")
        self._current = stored.handle
        return {"status": "restored", "current_handle": stored.handle,
                "logical_rows": self._states[stored.handle].length}

    def read_state(self, cut: ModelOSStateCut, address: str):
        stored = self._require(cut)
        self.state_schema.slot(address)
        if address == CONTEXT_ADDRESS:
            return dict(stored.metadata)
        if address == TOKEN_ADDRESS:
            return self._tokens.get(stored.handle)
        if address == NEXT_ADDRESS:
            return self._next_tokens.get(stored.handle)
        if address == RECEIPT_ADDRESS:
            return {"state_fingerprint": stored.state_fingerprint,
                    "status": stored.status}
        return None
