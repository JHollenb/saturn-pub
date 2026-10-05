"""Model OS transaction bridge for Mamba recurrent-state virtual memory.

This is the Mamba analogue of ``model_os_causal_context`` (the Qwen causal-context
bridge).  A page is a per-layer ``(C_l, H_l)`` recurrent-state cell.  The portable
package exposes identities, permissions and the page-install surface; the state
tensors stay private to :class:`MambaStateRuntime`.  Every page operation is one
Model OS lifecycle: ``preview`` forks the parent program and applies the install;
``continue_execution`` runs the native consumer (the stock forward and frozen LM
head) over the session's declared suffix; ``commit`` publishes the branch and
``abort`` discards it without mutating the parent.

Install modes follow the reading rule (§2.3 rule 6), and each is labeled:

- ``read-formed-state`` — read/ablation at the formed-circuit cut (a state page swap);
- ``residual-write`` — the write is offered through the residual at the source
  position and the model forms its own state;
- ``forced-formed-install`` — a direct formed-state install, kept but labeled as a
  different object than a residual write;
- ``noop`` — a zero-change install (the exact no-op gate).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from mdb.ar.mamba_state_program import MambaStateProgram
from mdb.ar.mamba_state_runtime import (
    MambaStateRuntime,
    logits_sha256,
    page_swapped_program,
)
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


PAGE_ADDRESS = "native.mamba-state-page"
INSTALL_ADDRESS = "decoder.page-install"
NEXT_ADDRESS = "native.next-token"
READOUT_ADDRESS = "native.consumer-readout"
RECEIPT_ADDRESS = "native.mamba-state-receipt"
CONSUMER_ID = "saturn.mamba.recurrent-state-direct-read"
SCHEMA = "saturn-model-os-mamba-recurrent-state-v1"

OP_READ = "read-formed-state"
OP_RESIDUAL_WRITE = "residual-write"
OP_FORCED_INSTALL = "forced-formed-install"
OP_NOOP = "noop"
_MODES = {OP_READ, OP_RESIDUAL_WRITE, OP_FORCED_INSTALL, OP_NOOP}

# read/ablation reads the formed state; residual writes go through the residual.
_READ_MODES = {OP_READ, OP_NOOP}
_FORMED_INSTALL_MODES = {OP_READ, OP_FORCED_INSTALL, OP_NOOP}


def build_mamba_state_package(
    runtime: MambaStateRuntime,
    *,
    model_id: str,
    tokenizer_identity: str,
) -> ModelPackage:
    state_schema = StateSchema(
        name=SCHEMA,
        version=1,
        slots=(
            StateSlot(
                address=PAGE_ADDRESS, value_type="mamba-recurrent-state-program",
                owner="mamba-state-runtime", dtype="metadata-reference",
                read_scopes=("model-os", CONSUMER_ID), write_scopes=(CONSUMER_ID,),
                lifetime="session", branchable=True, persistable=True,
                freshness="native-resident", lineage="parent-state-program",
                metadata={"tensor_payload": "private", "page": "per-layer-(C_l,H_l)"},
            ),
            StateSlot(
                address=INSTALL_ADDRESS, value_type="page-install-spec", owner="model-os",
                dtype="json", read_scopes=("model-os", CONSUMER_ID),
                write_scopes=("model-os",), lifetime="frame", branchable=True,
                persistable=False, freshness="candidate", lineage="selected-install",
                metadata={"modes": sorted(_MODES)},
            ),
            StateSlot(
                address=NEXT_ADDRESS, value_type="token-id", owner=CONSUMER_ID,
                dtype="int64", read_scopes=("model-os",), write_scopes=(CONSUMER_ID,),
                lifetime="frame", branchable=True, persistable=False,
                freshness="native-produced", lineage="native-consumer-output",
            ),
            StateSlot(
                address=READOUT_ADDRESS, value_type="consumer-log-probs", owner=CONSUMER_ID,
                dtype="metadata", read_scopes=("model-os",), write_scopes=(CONSUMER_ID,),
                lifetime="frame", branchable=True, persistable=False,
                freshness="native-produced", lineage="native-consumer-output",
                metadata={"tensor_payload": "forbidden"},
            ),
            StateSlot(
                address=RECEIPT_ADDRESS, value_type="statecut-receipt", owner=CONSUMER_ID,
                dtype="metadata", read_scopes=("model-os",), write_scopes=(CONSUMER_ID,),
                lifetime="frame", branchable=True, persistable=False,
                freshness="native-produced", lineage="native-transaction",
                metadata={"tensor_payload": "forbidden"},
            ),
        ),
    )
    abi = ModelABI(
        model_id=model_id, revision=runtime.model_abi_sha256,
        family="mamba", backend="recurrent-state-direct-read",
        state_schema=state_schema.fingerprint, tokenizer_identity=tokenizer_identity,
        kv_cache_schema="saturn-mamba-state-program-v1",
        device=str(runtime.device), dtype=str(runtime.dtype),
        numerical_contract="mamba-slow-path-staged-suffix-v1",
        checkpoint_sha256=runtime.model_abi_sha256,
        graph_identity="mamba-recurrent-state-full-layer-v1",
        native_consumers=(CONSUMER_ID,),
        runtime_fingerprint=content_digest({
            "consumer": CONSUMER_ID, "execution_abi": runtime.execution_abi_sha256,
            "n_layers": runtime.n_layers,
        }),
        source_fingerprint=runtime.model_abi_sha256,
        capabilities=("per-layer-state-page", "formed-state-read", "residual-source-write"),
    )
    consumer = ConsumerContract(
        consumer_id=CONSUMER_ID, reads=(PAGE_ADDRESS, INSTALL_ADDRESS),
        output_kind="native-next-token", continuation_tokens=1,
        replay_contract="candidate", native=True,
        metadata={"runtime": "MambaStateRuntime", "mode": "staged-suffix-forward"},
    )
    continuation = ContinuationContract(
        continuation_id="mamba-recurrent-state-one-readout-v1", consumer_id=CONSUMER_ID,
        input_state=(PAGE_ADDRESS, INSTALL_ADDRESS),
        output_state=(NEXT_ADDRESS, READOUT_ADDRESS, RECEIPT_ADDRESS),
        horizon=1, replay_contract="candidate", exact=False,
        metadata={"input": "one page install", "output": "frozen LM head over the declared suffix"},
    )
    program = ModelProgram(
        program_id="model-os.mamba.recurrent-state-continuation.v1", version=1,
        instructions=(
            ProgramInstruction(
                instruction_id="read-state-page", op="Read",
                reads=(PAGE_ADDRESS,), outputs=(PAGE_ADDRESS,),
                authority="model-os", replay_contract="candidate",
            ),
            ProgramInstruction(
                instruction_id="bind-page-install", op="Write",
                reads=(PAGE_ADDRESS,), writes=(INSTALL_ADDRESS,),
                outputs=(INSTALL_ADDRESS,), authority="model-os",
                dose_limit=1.0, replay_contract="candidate",
            ),
            ProgramInstruction(
                instruction_id="continue-native-readout", op="Continue",
                reads=(PAGE_ADDRESS, INSTALL_ADDRESS),
                outputs=(NEXT_ADDRESS, READOUT_ADDRESS, RECEIPT_ADDRESS),
                native_consumer=CONSUMER_ID,
                continuation=continuation.continuation_id,
                authority="model-os", dose_limit=1.0, replay_contract="candidate",
            ),
        ),
        portable=False,
        source_symbols=(
            "saturn.ar.mamba_state_program.MambaStateProgram",
            "saturn.ar.mamba_state_runtime.MambaStateRuntime",
        ),
        claim_level="controlled-mechanics", terminal_claim=False,
    )
    return ModelPackage(
        package_id="model-os.mamba.recurrent-state-direct-read", version=1,
        model_abi=abi, state_schema=state_schema,
        state_surface=StateSurface.from_schema(
            state_schema, surface_id="mamba-recurrent-state-surface", owner="model-os+saturn",
        ),
        program=program, consumers=(consumer,), continuations=(continuation,),
        permissions={
            "subjects": {"model-os": {
                "read": [PAGE_ADDRESS, INSTALL_ADDRESS, NEXT_ADDRESS, READOUT_ADDRESS, RECEIPT_ADDRESS],
                "write": [INSTALL_ADDRESS],
            }},
            "authorities": ["model-os"],
        },
        resource_limits={"horizon": 1, "persisted_tensor_bytes": 0},
        evidence_refs=("saturn-mamba-state-program-v1", SCHEMA),
        terminal_claim=False,
    )


class MambaStateModelBackend:
    """One branch of a mounted Mamba state program behind Model OS StateCuts."""

    def __init__(
        self,
        runtime: MambaStateRuntime,
        base_program: MambaStateProgram,
        package: ModelPackage,
        *,
        suffix: Sequence[int],
    ) -> None:
        self.runtime = runtime
        self.model_abi = package.model_abi
        self.state_schema = package.state_schema
        self.suffix = tuple(int(t) for t in suffix)
        if not self.suffix:
            raise ModelOSExecutionError("a mounted Mamba session needs a non-empty readout suffix")
        # Donor programs and residual-write deltas, keyed by program fingerprint.
        self._programs: dict[str, MambaStateProgram] = {}
        self._residual: dict[str, tuple[Sequence[int], Sequence[int], Mapping[int, torch.Tensor]]] = {}
        self._cuts: dict[str, ModelOSStateCut] = {}
        self._effective: dict[str, MambaStateProgram] = {}
        self._install: dict[str, dict[str, Any]] = {}
        self._readouts: dict[str, torch.Tensor] = {}
        self._next: dict[str, int] = {}
        self._counter = 0
        self.register_program(base_program, role="base")
        self._base_fingerprint = base_program.manifest()["fingerprint"]
        genesis = self._new_cut(
            base_program, install={"mode": OP_NOOP, "layers": [], "donor_fingerprint": None, "role": "base"},
            parent=None, status="committed",
        )
        self._readouts[genesis.handle] = self.runtime.stage(base_program, self.suffix)
        self._next[genesis.handle] = int(self._readouts[genesis.handle].argmax().item())
        self._current = genesis.handle
        self._genesis = genesis.handle

    # -- program / residual registry (tensors stay backend-private) -------
    def register_program(self, program: MambaStateProgram, *, role: str) -> str:
        fingerprint = program.manifest()["fingerprint"]
        self._programs[fingerprint] = program
        return fingerprint

    def register_residual_write(
        self,
        donor_fingerprint: str,
        *,
        base_ids: Sequence[int],
        donor_ids: Sequence[int],
        positions: Sequence[int],
        deltas: Mapping[int, torch.Tensor],
    ) -> None:
        self._residual[donor_fingerprint] = (tuple(base_ids), tuple(donor_ids), dict(deltas))

    def program_for_cut(self, cut: ModelOSStateCut) -> MambaStateProgram:
        return self._effective[self._require(cut).handle]

    def readout_tensor(self, cut: ModelOSStateCut) -> torch.Tensor:
        return self._readouts[self._require(cut).handle]

    # -- StateCut bookkeeping (mirrors the Qwen backend) ------------------
    def _new_cut(
        self, effective: MambaStateProgram, *, install: Mapping[str, Any],
        parent: str | None, status: str,
    ) -> ModelOSStateCut:
        self._counter += 1
        fingerprint = content_digest({
            "abi": self.model_abi.fingerprint,
            "program": effective.manifest()["fingerprint"],
            "install": dict(install), "parent": parent,
        })
        handle = f"model-os://mamba-state/{self._counter:08d}-{fingerprint[:24]}"
        cut = ModelOSStateCut(
            handle=handle, model_abi_fingerprint=self.model_abi.fingerprint,
            state_schema_fingerprint=self.state_schema.fingerprint,
            state_fingerprint=fingerprint, parent_handle=parent, status=status,
            metadata={
                "program_fingerprint": effective.manifest()["fingerprint"],
                "mode": install["mode"], "layers": list(install.get("layers") or []),
                "op_label": install["mode"], "n_layers": len(effective.layers),
            },
        )
        self._cuts[handle] = cut
        self._effective[handle] = effective
        self._install[handle] = dict(install)
        return cut

    def _require(self, cut: ModelOSStateCut) -> ModelOSStateCut:
        stored = self._cuts.get(cut.handle)
        if stored is None or stored.model_abi_fingerprint != self.model_abi.fingerprint:
            raise ModelOSExecutionError("unknown or incompatible Mamba state cut")
        return stored

    @property
    def current_statecut(self) -> ModelOSStateCut:
        return self._cuts[self._current]

    @property
    def genesis_handle(self) -> str:
        return self._genesis

    def consume_private_output(self, cut: ModelOSStateCut | None = None) -> torch.Tensor:
        handle = self._require(cut).handle if cut is not None else self._current
        return self._readouts.get(handle)

    # -- ModelBackend protocol --------------------------------------------
    def capture_statecut(self, parent: ModelOSStateCut | None, *, metadata) -> ModelOSStateCut:
        base = self._require(parent or self.current_statecut)
        if base.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot capture an invalid Mamba state cut")
        cut = self._new_cut(self._effective[base.handle], install=self._install[base.handle],
                            parent=base.handle, status="captured")
        self._readouts[cut.handle] = self._readouts.get(base.handle)
        self._next[cut.handle] = self._next.get(base.handle, -1)
        return cut

    def preview(self, parent: ModelOSStateCut, plan, updates):
        base = self._require(parent)
        if base.status not in {"committed", "captured"}:
            raise ModelOSExecutionError("preview requires an open parent")
        if set(updates) != {INSTALL_ADDRESS} or INSTALL_ADDRESS not in plan.writes:
            raise ModelOSExecutionError("preview requires one declared page install")
        spec = dict(updates[INSTALL_ADDRESS])
        mode = spec.get("mode")
        if mode not in _MODES:
            raise ModelOSExecutionError(f"unknown page-install mode {mode!r}")
        layers = [int(x) for x in (spec.get("layers") or [])]
        base_program = self._effective[base.handle]
        if mode in _FORMED_INSTALL_MODES:
            donor_fp = spec.get("donor_fingerprint")
            donor = base_program if (mode == OP_NOOP or donor_fp is None) else self._programs.get(donor_fp)
            if donor is None:
                raise ModelOSExecutionError("page install references an unregistered donor program")
            effective = page_swapped_program(base_program, donor, layers)
        else:  # residual-write: the formed program is unchanged; the write is in the forward
            donor_fp = spec.get("donor_fingerprint")
            if donor_fp not in self._residual:
                raise ModelOSExecutionError("residual write references an unregistered donor")
            effective = base_program
        candidate = self._new_cut(effective, install={**spec, "mode": mode, "layers": layers},
                                  parent=base.handle, status="candidate")
        return candidate, {"accepted": True, "mode": mode, "layers": layers,
                           "program_fingerprint": effective.manifest()["fingerprint"]}

    def continue_execution(self, candidate: ModelOSStateCut, *, consumer, horizon):
        stored = self._require(candidate)
        if stored.status != "candidate" or consumer.consumer_id != CONSUMER_ID or horizon > 1:
            raise ModelOSExecutionError("candidate or consumer contract changed")
        install = self._install[stored.handle]
        mode = install["mode"]
        if mode in _READ_MODES or mode == OP_FORCED_INSTALL:
            log_probs = self.runtime.stage(self._effective[stored.handle], self.suffix)
        else:  # residual-write
            base_ids, _donor_ids, deltas = self._residual[install["donor_fingerprint"]]
            positions = [int(p) for p in install["positions"]]
            log_probs = self.runtime.residual_write_stage(
                base_ids, positions, deltas, [install["layers"]],
            )[0]
        self._readouts[stored.handle] = log_probs
        next_token = int(log_probs.argmax().item())
        self._next[stored.handle] = next_token
        return {
            "accepted": True, "consumer_id": CONSUMER_ID, "mode": mode,
            "op_label": mode, "formed_state_read": mode in _READ_MODES,
            "residual_write": mode == OP_RESIDUAL_WRITE,
            "forced_formed_install": mode == OP_FORCED_INSTALL,
            "next_token_id": next_token, "logits_sha256": logits_sha256(log_probs),
            "suffix_len": len(self.suffix), "layers": list(install["layers"]),
        }

    def commit(self, candidate: ModelOSStateCut):
        stored = self._require(candidate)
        if stored.status != "candidate" or stored.handle not in self._readouts:
            raise ModelOSExecutionError("candidate has no native continuation")
        self._cuts[stored.handle] = replace(stored, status="committed")
        self._current = stored.handle
        return {"status": "committed", "current_handle": stored.handle,
                "next_token_id": self._next[stored.handle], "output_visible": True}

    def abort(self, candidate: ModelOSStateCut, *, reason: str):
        stored = self._require(candidate)
        if stored.status not in {"candidate", "captured"}:
            raise ModelOSExecutionError("only provisional cuts can abort")
        self._cuts[stored.handle] = replace(stored, status="restored")
        self._effective.pop(stored.handle, None)
        self._readouts.pop(stored.handle, None)
        self._next.pop(stored.handle, None)
        return {"status": "restored", "restored_to": stored.parent_handle,
                "reason": reason, "output_visible": False}

    def restore(self, target: ModelOSStateCut):
        stored = self._require(target)
        if stored.status in {"restored", "aborted"}:
            raise ModelOSExecutionError("cannot restore an invalid cut")
        self._current = stored.handle
        return {"status": "restored", "current_handle": stored.handle}

    def read_state(self, cut: ModelOSStateCut, address: str):
        stored = self._require(cut)
        self.state_schema.slot(address)
        if address == PAGE_ADDRESS:
            program = self._effective[stored.handle]
            manifest = program.manifest()
            return {"program_fingerprint": manifest["fingerprint"], "n_layers": len(program.layers),
                    "mode": self._install[stored.handle]["mode"]}
        if address == INSTALL_ADDRESS:
            return dict(self._install[stored.handle])
        if address == NEXT_ADDRESS:
            return self._next.get(stored.handle)
        if address == READOUT_ADDRESS:
            tensor = self._readouts.get(stored.handle)
            return None if tensor is None else {"logits_sha256": logits_sha256(tensor)}
        if address == RECEIPT_ADDRESS:
            return {"state_fingerprint": stored.state_fingerprint, "status": stored.status}
        return None
