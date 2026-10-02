"""Saturn VM — Mamba recurrent-state model virtual memory.

The Mamba analogue of :class:`CausalContextVM`.  One resident model; each session
installs a :class:`MambaStateModelBackend` behind a :class:`ModelOSKernel`, so every
page operation (mount / fork / swap / diff / unmount / sweep / clock) is one audited
Model OS transaction with a portable evidence receipt — the same kernel and contracts
the Qwen causal-context runtime uses.

Pages are the per-layer ``(C_l, H_l)`` cells of a :class:`MambaStateProgram`.  The
read/write discipline of §2.3 rule 6 is exposed as typed, labeled operations:

- ``read-formed-state``: read/ablation at the formed recurrent-state cut (a page swap);
- ``residual-write``: the write is offered through the residual at the source position;
- ``forced-formed-install``: a direct formed-state install, kept but labeled as such.

The exact validity gates (no-op, uninstall, replay parity) are first-class and are
proven through the kernel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import torch

from mdb.ar.mamba_state_program import MambaStateProgram
from mdb.ar.mamba_state_runtime import (
    MambaStateRuntime,
    diff_programs,
    logits_sha256,
    page_swapped_program,
)
from mdb.model_os_mamba_state import (
    INSTALL_ADDRESS,
    OP_FORCED_INSTALL,
    OP_NOOP,
    OP_READ,
    OP_RESIDUAL_WRITE,
    MambaStateModelBackend,
    build_mamba_state_package,
)
from mdb.model_os_kernel import ModelOSKernel, ModelOSStateCut


class MambaStateVMError(ValueError):
    """A Mamba VM mount, page op, or gate request is invalid."""


@dataclass
class _Session:
    session_id: str
    kernel: ModelOSKernel
    backend: MambaStateModelBackend
    base_program: MambaStateProgram
    base_cut: ModelOSStateCut
    suffix: tuple[int, ...]
    span_ids: tuple[int, ...]
    cut_pos: int
    positions: tuple[int, ...]
    readout: dict[str, Any] | None = None
    pinned: bool = False
    donors: dict[str, MambaStateProgram] = field(default_factory=dict)


def _readout(log_probs: torch.Tensor, readout: Mapping[str, Any]) -> dict[str, Any]:
    """Matches ``worker_sweep.readout`` exactly (token-id vocab, 4-dp rounding)."""

    lp_b = float(log_probs[readout["answer_base"]])
    dl = {int(d): float(log_probs[int(d)]) for d in [readout["answer_target"], *readout["distractors"]]}
    top_d = max(dl, key=dl.get)
    names = {int(readout["answer_base"]): lp_b, **dl}
    return {"lp_base_answer": round(lp_b, 4), "margin": round(lp_b - dl[top_d], 4),
            "top1": int(max(names, key=names.get))}


def _margin_t(log_probs: torch.Tensor, readout: Mapping[str, Any]) -> float:
    return float(log_probs[int(readout["answer_target"])] - log_probs[int(readout["answer_base"])])


class MambaStateVM:
    """One resident Mamba model, many transactional state-page sessions."""

    def __init__(self, model: Any, *, model_abi_sha256: str, model_id: str,
                 tokenizer_identity: str) -> None:
        self.runtime = MambaStateRuntime(model, model_abi_sha256=model_abi_sha256)
        self.package = build_mamba_state_package(
            self.runtime, model_id=model_id, tokenizer_identity=tokenizer_identity,
        )
        self.model_id = model_id
        self.revision = model_abi_sha256
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()

    # -- capture (export) -------------------------------------------------
    def capture(self, token_ids: Sequence[int], *, cut_pos: int, pending_token_id: int) -> MambaStateProgram:
        return self.runtime.capture(
            token_ids, cut_pos=cut_pos, pending_token_id=pending_token_id,
            model_id=self.model_id, revision=self.revision,
        )

    def _session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise MambaStateVMError("unknown Mamba state session") from exc

    # -- mount ------------------------------------------------------------
    def mount(self, session_id: str, program: MambaStateProgram, *, suffix: Sequence[int],
              span_ids: Sequence[int] | None = None, cut_pos: int | None = None,
              positions: Sequence[int] | None = None, readout: Mapping[str, Any] | None = None,
              pinned: bool = False) -> dict[str, Any]:
        with self._lock:
            if not session_id or session_id in self._sessions:
                raise MambaStateVMError("session id must be nonempty and unique")
            backend = MambaStateModelBackend(self.runtime, program, self.package, suffix=suffix)
            kernel = ModelOSKernel.install(self.package, backend)
            base_cut = kernel.current_statecut
            session = _Session(
                session_id=session_id, kernel=kernel, backend=backend, base_program=program,
                base_cut=base_cut, suffix=tuple(int(t) for t in suffix),
                span_ids=tuple(int(t) for t in (span_ids or program.source_token_ids)),
                cut_pos=int(cut_pos if cut_pos is not None else len(program.source_token_ids) - 1),
                positions=tuple(int(p) for p in (positions or ())),
                readout=dict(readout) if readout is not None else None, pinned=pinned,
            )
            self._sessions[session_id] = session
            base_tensor = backend.readout_tensor(base_cut)
            return {
                "op": "mount", "session_id": session_id,
                "statecut": base_cut.to_payload(),
                "program_fingerprint": program.manifest()["fingerprint"],
                "n_layers": len(program.layers), "suffix_len": len(session.suffix),
                "next_token_id": int(base_tensor.argmax().item()),
                "base_logits_sha256": logits_sha256(base_tensor),
                "readout": _readout(base_tensor, session.readout) if session.readout else None,
                "model_os_consumer": self.package.model_abi.native_consumers[0],
            }

    # -- internal measurement transaction ---------------------------------
    def _measure(self, session: _Session, spec: Mapping[str, Any], *, run_id: str,
                 verifier=None) -> tuple[Any, torch.Tensor]:
        result = session.kernel.execute(
            updates={INSTALL_ADDRESS: dict(spec)}, parent=session.base_cut, horizon=1,
            metadata={"run_id": run_id, "op_label": spec["mode"]}, verifier=verifier,
        )
        candidate = result.candidate
        tensor = session.backend.readout_tensor(candidate) if result.status == "committed" else None
        session.kernel.restore(session.base_cut)  # keep the session anchored on its mounted base
        return result, tensor

    # -- fork -------------------------------------------------------------
    def fork(self, session_id: str, new_session_id: str, *, pinned: bool = False) -> dict[str, Any]:
        with self._lock:
            parent = self._session(session_id)
            clone = parent.base_program.relocated()  # non-aliasing deep copy
            child = self.mount(
                new_session_id, clone, suffix=parent.suffix, span_ids=parent.span_ids,
                cut_pos=parent.cut_pos, positions=parent.positions, readout=parent.readout,
                pinned=pinned,
            )
            parent_tensor = parent.backend.readout_tensor(parent.base_cut)
            child_tensor = self._session(new_session_id).backend.readout_tensor(
                self._session(new_session_id).base_cut
            )
            fork_noop_max_abs = float((parent_tensor - child_tensor).abs().max())
            child["op"] = "fork"
            child["forked_from"] = session_id
            child["fork_noop_max_abs"] = fork_noop_max_abs
            child["fork_exact"] = fork_noop_max_abs == 0.0
            return child

    # -- swap (single page op) --------------------------------------------
    def swap(self, session_id: str, *, donor: MambaStateProgram, layers: Sequence[int],
             mode: str = OP_READ, run_id: str = "vm-swap", verifier=None) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            donor_fp = session.backend.register_program(donor, role="donor")
            layers = [int(x) for x in layers]
            spec: dict[str, Any] = {"mode": mode, "layers": layers, "donor_fingerprint": donor_fp}
            if mode == OP_RESIDUAL_WRITE:
                if not session.positions:
                    raise MambaStateVMError("residual-write swap needs declared carrier positions")
                donor_span = _donor_span(donor, session)
                deltas = self.runtime.block_output_residual_deltas(
                    session.span_ids, donor_span, session.positions,
                )
                session.backend.register_residual_write(
                    donor_fp, base_ids=session.span_ids, donor_ids=donor_span,
                    positions=session.positions, deltas=deltas,
                )
                spec["positions"] = list(session.positions)
            result, tensor = self._measure(session, spec, run_id=run_id, verifier=verifier)
            base_tensor = session.backend.readout_tensor(session.base_cut)
            payload = {
                "op": "swap", "session_id": session_id, "mode": mode, "op_label": mode,
                "formed_state_read": mode in (OP_READ, OP_NOOP),
                "residual_write": mode == OP_RESIDUAL_WRITE,
                "forced_formed_install": mode == OP_FORCED_INSTALL,
                "layers": layers, "status": result.status,
                "receipt": result.receipt.to_payload(),
                "receipt_fingerprint": result.receipt.fingerprint,
            }
            if tensor is not None and session.readout is not None:
                base_r = _readout(base_tensor, session.readout)
                cand_r = _readout(tensor, session.readout)
                payload["readout"] = cand_r
                payload["base_readout"] = base_r
                payload["necessity_delta"] = round(cand_r["lp_base_answer"] - base_r["lp_base_answer"], 4)
                payload["logits_max_abs_vs_base"] = float((tensor - base_tensor).abs().max())
            return payload

    # -- diff -------------------------------------------------------------
    def diff(self, left: MambaStateProgram, right: MambaStateProgram) -> dict[str, Any]:
        result = diff_programs(left, right)
        return {"op": "diff", **result.to_payload(),
                "left_fingerprint": left.manifest()["fingerprint"],
                "right_fingerprint": right.manifest()["fingerprint"]}

    # -- unmount ----------------------------------------------------------
    def unmount(self, session_id: str, *, check_gates: bool = True) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            program = session.base_program
            del self._sessions[session_id]
            payload = {"op": "unmount", "session_id": session_id,
                       "program_fingerprint": program.manifest()["fingerprint"],
                       "durable": True}
            if check_gates:
                remount_lp = self.runtime.stage(program, session.suffix)
                base_lp = session.backend.readout_tensor(session.base_cut)
                payload["uninstall_max_abs"] = float((remount_lp - base_lp).abs().max())
                payload["uninstall_exact"] = payload["uninstall_max_abs"] == 0.0
                if len(session.span_ids) >= 2:
                    native_lp = self.runtime.native_full(session.span_ids)
                    payload["replay_max_abs"] = float((base_lp - native_lp).abs().max())
                    payload["replay_argmax_match"] = bool(
                        int(base_lp.argmax()) == int(native_lp.argmax())
                    )
            return payload

    # -- exact validity gates (audited through the kernel) ----------------
    def gates(self, session_id: str, *, run_id: str = "vm-gates") -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            base_lp = session.backend.readout_tensor(session.base_cut)
            _noop_result, noop_lp = self._measure(
                session, {"mode": OP_NOOP, "layers": [], "donor_fingerprint": None}, run_id=f"{run_id}-noop",
            )
            noop_max_abs = float((noop_lp - base_lp).abs().max())
            # uninstall: re-stage the mounted base program outside the kernel branch
            uninstall_lp = self.runtime.stage(session.base_program, session.suffix)
            uninstall_max_abs = float((uninstall_lp - base_lp).abs().max())
            gate = {"op": "gates", "session_id": session_id,
                    "noop_max_abs": noop_max_abs, "uninstall_max_abs": uninstall_max_abs}
            if len(session.span_ids) >= 2:
                native_lp = self.runtime.native_full(session.span_ids)
                gate["replay_max_abs"] = float((base_lp - native_lp).abs().max())
                gate["replay_argmax_match"] = bool(int(base_lp.argmax()) == int(native_lp.argmax()))
            gate["noop_exact"] = noop_max_abs == 0.0
            gate["uninstall_exact"] = uninstall_max_abs == 0.0
            return gate

    # -- sweep (per-layer page sweep) -------------------------------------
    def sweep(self, session_id: str, *, donor: MambaStateProgram, mode: str = OP_READ,
              run_id: str = "vm-sweep", chunk: int = 32, audit_layer: int | None = None,
              target: MambaStateProgram | None = None) -> dict[str, Any]:
        """Per-layer page sweep across all layers.

        ``read`` mode measures per-layer necessity (generic-state swap at one layer);
        ``residual-write`` mode measures per-layer authorship through the residual.
        Returns the per-layer profile plus the kernel-proven exact gates and a
        spot audit that the batched runtime path equals one audited kernel swap.
        """
        with self._lock:
            session = self._session(session_id)
            if session.readout is None:
                raise MambaStateVMError("sweep needs a mounted session with a readout vocabulary")
            base = session.base_program
            n = len(base.layers)
            base_lp = session.backend.readout_tensor(session.base_cut)
            clean = _readout(base_lp, session.readout)
            gate = self.gates(session_id, run_id=f"{run_id}-gate")

            if mode == OP_READ:
                programs = [page_swapped_program(base, donor, [li]) for li in range(n)]
                sweep_lp = self.runtime.stage_batched(programs, session.suffix, chunk=chunk)
                per_layer = [round(_readout(sweep_lp[li], session.readout)["lp_base_answer"]
                                   - clean["lp_base_answer"], 4) for li in range(n)]
                profile_key = "necessity_per_layer"
            elif mode == OP_RESIDUAL_WRITE:
                if not session.positions:
                    raise MambaStateVMError("residual-write sweep needs declared carrier positions")
                donor_span = _donor_span(target or donor, session)
                deltas = self.runtime.block_output_residual_deltas(
                    session.span_ids, donor_span, session.positions,
                )
                pc = self._pc_shift(session, target or donor)
                rows = [[li] for li in range(n)]
                rw_lp = self.runtime.residual_write_stage(
                    session.span_ids, session.positions, deltas, rows, chunk=chunk,
                )
                per_layer = [round((_margin_t(rw_lp[li], session.readout) - _margin_t(base_lp, session.readout))
                                   / pc, 4) if pc else None for li in range(n)]
                profile_key = "resid_write_per_layer"
            else:
                raise MambaStateVMError("sweep mode must be read-formed-state or residual-write")

            abs_vals = [abs(x) for x in per_layer if x is not None]
            max_val = max(abs_vals) if abs_vals else None
            max_layer = int([abs(x) if x is not None else -1 for x in per_layer].index(max_val)) if abs_vals else None

            payload = {
                "op": "sweep", "session_id": session_id, "mode": mode, "op_label": mode,
                "n_layers": n, profile_key: per_layer, "gates": gate,
                "max_single": {"value": max_val, "layer": max_layer},
            }
            # spot audit: batched path == one audited kernel swap at the dominant layer
            if mode == OP_READ and audit_layer is None:
                audit_layer = max_layer
            if mode == OP_READ and audit_layer is not None:
                audited = self.swap(session_id, donor=donor, layers=[audit_layer],
                                    mode=OP_READ, run_id=f"{run_id}-audit-L{audit_layer}")
                payload["audit"] = {
                    "layer": audit_layer,
                    "kernel_necessity_delta": audited.get("necessity_delta"),
                    "batched_necessity_delta": per_layer[audit_layer],
                    "max_abs_batched_vs_kernel": abs(
                        (audited.get("necessity_delta") or 0.0) - (per_layer[audit_layer] or 0.0)
                    ),
                }
            return payload

    def _pc_shift(self, session: _Session, target: MambaStateProgram) -> float:
        base_lp = session.backend.readout_tensor(session.base_cut)
        tgt_lp = self.runtime.stage(target, session.suffix)
        return _margin_t(tgt_lp, session.readout) - _margin_t(base_lp, session.readout)

    # -- clock (per-layer retention + write magnitude) --------------------
    def clock(self, session_id: str, *, reference: MambaStateProgram | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            if len(session.span_ids) < 2:
                raise MambaStateVMError("clock needs a mounted session with a declared token span")
            result = self.runtime.layer_clock(session.span_ids, session.cut_pos)
            result = {"op": "clock", "session_id": session_id, **result}
            if reference is not None:
                d = diff_programs(session.base_program, reference)
                result["write_vs_reference"] = {
                    "reference_fingerprint": reference.manifest()["fingerprint"],
                    "delta_H_l2": list(d.recurrent_l2), "delta_C_l2": list(d.conv_l2),
                }
            return result


def _donor_span(donor: MambaStateProgram, session: _Session) -> tuple[int, ...]:
    """Reconstruct the donor's full token span aligned to the base span length."""

    span = tuple(donor.source_token_ids) + tuple(session.span_ids[len(donor.source_token_ids):])
    if len(span) != len(session.span_ids):
        raise MambaStateVMError("donor prefix length does not align with the session span")
    return span


# ---------------------------------------------------------------------------
# Task + op helpers (used by the mrun workers and the CLI worker).  These take a
# constructed VM plus a tokenizer, so they stay import-safe on the laptop.
# ---------------------------------------------------------------------------

def build_task(vm: MambaStateVM, tok: Any, object_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Tokenize an E3b-style prompt object and capture base/generic/target programs.

    Mirrors ``worker_sweep.run_object`` setup so VM sweeps reproduce the recorded
    profiles: a single carrier cut, a shared suffix, and token-id readout vocab.
    """

    def tid(word: str) -> int:
        enc = tok(" " + word, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise MambaStateVMError(f"multi-token word: {word!r} -> {enc}")
        return int(enc[0])

    prompts = object_spec["prompts"]
    carriers = object_spec["carriers"]
    ids = {k: tok(p, return_tensors="pt")["input_ids"][0].tolist() for k, p in prompts.items()}
    length = len(ids["base"])
    if any(len(v) != length for v in ids.values()):
        raise MambaStateVMError("prompt length mismatch")

    def carrier_positions(kind: str) -> list[int]:
        toks = tok.convert_ids_to_tokens(ids[kind])
        pos, used = [], set()
        for word in carriers[kind]:
            hit = None
            for i, t in enumerate(toks):
                if i in used:
                    continue
                if t.lstrip("Ġ▁").lower() == word.lower():
                    hit = i
                    break
            if hit is None:
                raise MambaStateVMError(f"carrier {word!r} not found in {kind}")
            used.add(hit)
            pos.append(hit)
        return pos

    positions = carrier_positions("base")
    for kind in ("generic", "target"):
        if carrier_positions(kind) != positions:
            raise MambaStateVMError(f"carrier positions differ in {kind}")
    cut_pos = max(positions)
    if cut_pos + 1 >= length:
        raise MambaStateVMError("no suffix after carrier")
    suffix = [int(x) for x in ids["base"][cut_pos + 1:]]
    readout = {
        "answer_base": tid(object_spec["answer_base"]),
        "answer_target": tid(object_spec["answer_target"]),
        "distractors": [tid(w) for w in object_spec["distractors"]],
    }
    programs = {}
    for kind in ("base", "generic", "target"):
        programs[kind] = vm.capture(ids[kind], cut_pos=cut_pos, pending_token_id=ids[kind][cut_pos + 1]).relocated(device="cpu")
    return {
        "ids": ids, "positions": positions, "cut_pos": cut_pos, "suffix": suffix,
        "readout": readout, "programs": programs,
        "span_ids": tuple(int(x) for x in ids["base"]),
        "target_span_ids": tuple(int(x) for x in ids["target"]),
    }
