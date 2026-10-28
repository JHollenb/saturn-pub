"""MDB control over the public Saturn Session and debugger engines.

No model forward, cache implementation, symbol resolver or observer is copied
here. The adapter translates the MDB protocol and retains public native cuts.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import replace
from uuid import uuid4

from saturn_pub.cli import Debugger
from saturn_pub.core import Act, Session
from saturn_pub.symbols import SymbolBinding, SymbolTable
from saturn_pub.values import describe

from .debugger_adapters import AdapterCapabilities, CallbackDebuggerAdapter
from .debugger_protocol import DebuggerAddressSpec


class SaturnPubDebuggerAdapter(CallbackDebuggerAdapter):
    """Wrap a caller-owned public Session; optionally retain it in a Project."""

    def __init__(self, session, *, project=None, session_id=None, max_steps=1000, resume=False):
        if not isinstance(session, Session):
            raise TypeError("expected a saturn_pub.Session")
        if type(max_steps) is not int or not 1 <= max_steps <= 100000:
            raise ValueError("max_steps must be in 1..100000")
        self.public = Debugger(session)
        self.project = project
        self.session_id = session_id or "public-" + uuid4().hex
        self.max_steps = max_steps
        self.worker = self.bench = None
        self._pending = None
        self._cuts = {}
        self._unsaved = {}
        self._extra_receipts = []
        self._watch_history = []
        self._watch_errors = []
        self._transition_index = 0
        self._project_revision = 0
        self.identity = {
            "model_id": session.adapter.model_identity,
            "execution": dict(session.adapter.execution),
        }
        if project and self.session_id in {r["session_id"] for r in project.sessions()}:
            if not resume:
                raise ValueError("project session already exists; use Project.resume")
            row = project.session(self.session_id)
            if session.capture(retain=False).fingerprint != row["selected"]:
                raise ValueError("project session changed while reopening")
            self._project_revision = row["revision"]
            info = json.loads(row["debug_info"])
            self.public.symbols = SymbolTable(
                tuple(SymbolBinding.from_dict(v) for v in info.get("symbols", []))
            )
            self.public.symbol_context = info.get("symbol_context", "")
            self._transition_index = info.get("transition_index", 0)
            self._watch_history = info.get("watch_events", [])
            self._watch_errors = info.get("watch_errors", [])
            for identifier in project.cut_ids(self.session_id):
                self._cuts[identifier] = None
        frame = session.inspect()
        slots = frame.surface["slots"]
        caps = AdapterCapabilities(
            family=str(frame.execution_point["family"]),
            boundaries=("transition",),
            inspectable_slots=tuple(
                dict.fromkeys([*frame.slots, "cursor", "symbols", "watch_events"])
            ),
            writable_slots=tuple(s["name"] for s in slots if s["writable"]),
            mutable_closure=tuple(s["name"] for s in slots),
            exact_replay=session.adapter.numerical_contract().mode == "same-program-exact",
            state_schema_id=frame.surface["state_schema"],
            backend="saturn-pub",
            consumer=frame.surface["consumer"],
            checkpoint_requires_pause=True,
            boundary_catalog=(
                {"kind": "transition", "index_semantics": "retained debugger transition index"},
            ),
        )
        super().__init__(
            caps,
            status=self._status,
            inspect=self._inspect,
            resolve_value=self._resolve,
            checkpoint=self._checkpoint,
            restore=self._restore,
            preview_act=self._preview,
            commit_act=self._commit,
            abort_act=self._abort,
            replay=self._replay,
            step=self._advance,
            continue_run=lambda: self._advance(self.max_steps),
            diff=self._diff,
            fork_branch=self._fork,
            checkout=self._checkout,
            public_command=self.command,
        )
        self._retain(session.capture(retain=False))
        self._persist()

    @property
    def session(self):
        return self.public.session

    @property
    def symbols(self):
        return self.public.symbols

    @symbols.setter
    def symbols(self, value):
        if not isinstance(value, SymbolTable):
            raise TypeError("symbols must be a public SymbolTable")
        self.public.symbols = value

    @property
    def symbol_context(self):
        return self.public.symbol_context

    @symbol_context.setter
    def symbol_context(self, value):
        self.public.symbol_context = value

    def _retain(self, cut):
        if cut.fingerprint not in self._cuts:
            self._unsaved[cut.fingerprint] = cut
        self._cuts[cut.fingerprint] = cut
        self.public.cuts[cut.fingerprint] = cut
        return cut

    def _cut(self, identifier):
        cut = self._cuts.get(identifier)
        if cut is None and identifier in self._cuts and self.project:
            device = str(getattr(self.session.adapter, "device", "cpu"))
            cut = self.project.load_cut(self.session_id, identifier, device=device)
            self._cuts[identifier] = cut
            self.public.cuts[identifier] = cut
        if cut is None:
            raise ValueError("unknown retained public cut")
        return cut

    def _persist(self):
        current = self._retain(self.session.capture(retain=False))
        if self.project:
            self._project_revision = self.project.save_session(
                self.session_id,
                self.session,
                self._unsaved.values(),
                debug_info={
                    "symbols": [b.to_dict() for b in self.symbols.bindings],
                    "symbol_context": self.symbol_context,
                    "transition_index": self._transition_index,
                    "watch_events": [
                        *self._watch_history,
                        *[e.to_dict() for e in self.public.watch_events],
                    ],
                    "watch_errors": [*self._watch_errors, *self.public.watch_errors],
                },
                receipts=self._extra_receipts,
                expected_revision=self._project_revision,
            )
            self._unsaved.clear()
            # LocalStore owns historical payloads. Keep references, not one
            # complete GPU cache per visited boundary, in the control plane.
            self._cuts = dict.fromkeys(self._cuts)
            self.public.cuts.clear()
            if len(self.public.watch_events) > 1:
                self._watch_history.extend(e.to_dict() for e in self.public.watch_events[:-1])
                self.public.watch_events[:] = self.public.watch_events[-1:]
        return current

    def cut_info(self):
        cut = self._retain(self.session.capture(retain=False))
        return {
            "cut_id": cut.fingerprint,
            "parent_cut": cut.parent,
            "branch": self.public.active,
            "cursor": dict(cut.execution_point or {}),
        }

    def _status(self):
        return {
            **self.cut_info(),
            "session_id": self.session_id,
            "native_format": "saturn-pub-statecut-v2",
        }

    def validate_read(self, path):
        if path and path.startswith("symbol:"):
            self.public._read_observed(self.session, path)
            return
        return super().validate_read(path)

    def address_specs(self):
        # Native permissions can change at a boundary (e.g. hidden is absent
        # before embedding). Live declarations also revoke initially writable
        # ports after restore; Session.apply remains the mutation authority.
        return tuple(
            DebuggerAddressSpec(
                name=f"public_slot_{index}",
                kind="native-port",
                path_template=slot["name"].replace("{", "{{").replace("}", "}}"),
                writable=slot["writable"],
                invalidates=tuple(slot["invalidates"]),
                boundaries=("transition",),
            )
            for index, slot in enumerate(self.session.inspect().surface["slots"])
        )

    def _resolve(self, path):
        if path == "cursor":
            return self.session.inspect().execution_point
        if path.startswith("cursor."):
            return self.session.inspect().execution_point[path.split(".", 1)[1]]
        return self.public._read_observed(self.session, path)

    def _inspect(self, path=None):
        if path is None:
            frame = self.session.inspect()
            return {
                **self._status(),
                "boundary": frame.boundary,
                "slots": dict(frame.slots),
                "surface": dict(frame.surface),
            }
        if path == "symbols":
            return self.public.execute("info symbols")
        if path == "watch_events":
            return {
                "events": [*self._watch_history, *[e.to_dict() for e in self.public.watch_events]],
                "errors": [*self._watch_errors, *self.public.watch_errors],
            }
        return {"path": path, "value": describe(self._resolve(path))}

    def _checkpoint(self, name=None):
        self._persist()
        return {**self.cut_info(), "name": name}

    def _restore(self, identifier):
        if self._pending:
            raise ValueError("cannot restore with a pending intervention")
        self._persist()
        self.public.cuts[identifier] = self._cut(identifier)
        try:
            receipt = self.public.execute(shlex.join(["restore", identifier]))
        finally:
            self._persist()
            self._pause_here()
        return {**self.cut_info(), "receipt": receipt, "exact": True}

    def _advance(self, count=1):
        if self._pending:
            raise ValueError("commit or abort the pending intervention first")
        if type(count) is not int or not 1 <= count <= self.max_steps:
            raise ValueError("transition count exceeds the configured budget")
        self._persist()
        if self.worker:
            self.worker._paused = False
            self.worker._pause_reason = None
        completed, result = 0, {}
        try:
            for _ in range(count):
                self._retain(self.session.capture(retain=False))
                previous_receipts = len(self.session.receipts)
                try:
                    result = self.public.advance(1)
                finally:
                    if len(self.session.receipts) > previous_receipts:
                        completed += 1
                        self._transition_index += 1
                    # Failed watches must retain the completed native result.
                    for cut in self.public.cuts.values():
                        self._retain(cut)
                    self._persist()
                if self.worker:
                    if result.get("stop") == "watchpoint":
                        self.worker._pending_pause = True
                    state = self.safe_point()
                    if state["state"] == "paused":
                        break
                elif result.get("stop") == "watchpoint":
                    break
        finally:
            self._pause_here()
        return {
            **self.cut_info(),
            "steps": completed,
            "stop": result.get("stop"),
            "events": result.get("events", []),
        }

    def safe_point(self):
        return self.worker.safe_point(
            "transition",
            self._transition_index,
            {**dict(self.session.inspect().execution_point), "branch": self.public.active},
        )

    def _pause_here(self):
        if self.worker:
            self.worker._pending_pause = True
            self.safe_point()

    def bind_debugger(self, controller):
        from .local_debugger import LocalDebugTransport

        self.worker = controller
        controller._remember_cut(self.cut_info())
        return LocalDebugTransport(controller, self)

    def optional_operations(self):
        return (*super().optional_operations(), "public_command", "fork_branch", "checkout")

    def _preview(self, spec):
        if self._pending:
            raise ValueError("an intervention is already pending")
        parent = self._persist()
        if parent.fingerprint != spec.parent_cut:
            raise ValueError("intervention parent is stale")
        current = self.session.read(spec.path)
        value = spec.value
        if hasattr(current, "shape"):
            import torch

            value = torch.as_tensor(value, dtype=current.dtype, device=current.device)
            if value.ndim == 0:
                value = torch.ones_like(current) * value
        if spec.sign == "scale":
            act = Act.replace(spec.path, current * value)
        elif spec.sign == "add":
            act = Act.add(spec.path, value)
        elif spec.sign == "replace":
            act = Act.replace(spec.path, value)
        else:
            raise ValueError("public interventions support replace, add and scale")
        slot = next(s for s in self.session.inspect().surface["slots"] if s["name"] == spec.path)
        act = replace(act, invalidates=tuple(slot["invalidates"]))
        branch = self.session.fork(parent)
        try:
            # Fork capture has its own native lineage identity. Retain the
            # actual Act parent, not only the pre-fork cut and edited result.
            self._retain(branch.capture(retain=False))
            branch.apply(act)
            candidate = self._retain(branch.capture(retain=False))
            self._extra_receipts.extend(branch.receipts)
            self._persist()
            self._pending = (spec.act_id, branch)
            return {
                "act_id": spec.act_id,
                "parent_cut": parent.fingerprint,
                "candidate_cut": candidate.fingerprint,
            }
        finally:
            self.session.capture(retain=False)  # Reactivate the parent's external closure.

    def _finish(self, act_id, *, commit):
        if self._pending is None or self._pending[0] != act_id:
            raise ValueError("unknown pending intervention")
        branch = self._pending[1]
        parent = self._persist()
        receipt = self.session.commit(branch) if commit else self.session.abort(branch)
        try:
            self.public.observer.sync(self.session, self.public.active)
        except (ValueError, KeyError) as error:
            self.public.watch_errors.append(
                {
                    "branch": self.public.active,
                    "cut": self.session.capture(retain=False).fingerprint,
                    "receipt": receipt.fingerprint,
                    "errors": [{"reason": str(error), "stage": "baseline-after-intervention"}],
                }
            )
        try:
            self._persist()
        except Exception:
            self.session.restore(parent)
            raise
        self._pending = None
        self._pause_here()
        return {**self.cut_info(), "act_id": act_id, "receipt": receipt.to_dict()}

    def _commit(self, act_id):
        return self._finish(act_id, commit=True)

    def _abort(self, act_id):
        return self._finish(act_id, commit=False)

    def _fork(self, name):
        if self._pending:
            raise ValueError("cannot fork with a pending intervention")
        self._persist()
        self.public.execute(shlex.join(["fork", name]))
        return self._checkout(name)

    def _checkout(self, name):
        if self._pending:
            raise ValueError("cannot checkout with a pending intervention")
        self.public.execute(shlex.join(["use", name]))
        self._persist()
        self._pause_here()
        return self.cut_info()

    def _diff(self, path, left, right):
        a = self.session.fork(self._cut(left))
        b = self.session.fork(self._cut(right))
        try:
            return {
                "path": path,
                "left": describe(a.read(path)),
                "right": describe(b.read(path)),
                "equal": describe(a.read(path)) == describe(b.read(path)),
            }
        finally:
            self.session.capture(retain=False)

    def _replay(self, identifier):
        if self._pending:
            raise ValueError("cannot replay with a pending intervention")
        if not self.capabilities.exact_replay:
            raise ValueError("this public execution contract does not claim exact replay")
        self._persist()
        cut = self._cut(identifier)
        reference = self.session.replay(cut)
        repeated = self.session.replay(cut)
        first = reference.capture(retain=False)
        second = repeated.capture(retain=False)
        if first.fingerprint != second.fingerprint:
            raise ValueError("same-program suffix replay differed")
        self._retain(second)
        self._extra_receipts.extend([*reference.receipts, *repeated.receipts])
        self._persist()
        return {
            **self.cut_info(),
            "replayed_cut": identifier,
            "result_cut": second.fingerprint,
            "exact": True,
            "steps": 1,
            "authority": "same-program-repeated-suffix",
        }

    def command(self, line):
        args = shlex.split(line)
        if args and args[0] == "project":
            if self.project is None:
                raise ValueError("project navigation requires an attached project")
            from .project_navigation import command

            self._persist()
            return command(self.project, self.session_id, args[1:])
        if not args or args[0] not in {
            "info",
            "resolve",
            "backtrace",
            "read",
            "watch",
            "watchpoints",
            "unwatch",
        }:
            raise ValueError(
                "use MDB motion/edit commands; public commands expose symbols, backtraces and watches"
            )
        result = self.public.execute(line)
        self._persist()
        return result if isinstance(result, dict) else {"value": result}


def main(argv=None):
    from .bench import BenchSession
    from .debugger_cli import DebugShell
    from .local_cli import token_ids
    from .project import Project

    parser = argparse.ArgumentParser(description="Debug a Saturn-pub decoder in an MDB project")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", metavar="FAMILY", help="random CPU decoder; no download")
    from pathlib import Path

    source.add_argument("--local", type=Path, help="explicit local HF checkpoint")
    parser.add_argument("--project", required=True)
    parser.add_argument("--session-id", default="main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--token-ids", type=token_ids)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--max-steps", type=int, default=1000)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--command")
    commands.add_argument("--script", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.token_ids is not None:
        parser.error("resume uses retained input; omit --token-ids")
    if args.tiny:
        if args.device != "cpu" or args.dtype != "fp32":
            parser.error("tiny decoders use CPU FP32")
        from saturn_pub.adapters.decoder import DecoderAdapter

        native = DecoderAdapter.tiny(args.tiny, seed=args.seed)
    else:
        import torch
        from saturn_pub.adapters import load

        if not args.local.is_dir() or not (args.local / "config.json").is_file():
            parser.error("--local requires a checkpoint directory containing config.json")
        native = load(
            args.local,
            device=args.device,
            local_files_only=True,
            trust_remote_code=False,
            dtype={"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
                args.dtype
            ],
        )
    project = Project(args.project)
    adapter = (
        project.resume(args.session_id, native, max_steps=args.max_steps)
        if args.resume
        else SaturnPubDebuggerAdapter(
            native.session(args.token_ids or [1, 2, 3]),
            project=project,
            session_id=args.session_id,
            max_steps=args.max_steps,
        )
    )
    owner = BenchSession.from_backend(adapter, identity=adapter.identity)
    with owner:
        shell = DebugShell(owner.debugger(), json_output=args.json)
        if args.command:
            return 0 if shell.execute(args.command) else 1
        if args.script:
            return 0 if shell.run_script(args.script)["ok"] else 1
        return shell.repl()
