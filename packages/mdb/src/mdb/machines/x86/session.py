"""The debugger verbs on a binary: one definition each (proposal §3), Python API.

``attach(program, input)`` runs the binary once and returns a ``BinRun`` whose cursor starts at the
initial state. The run is recorded (rr-style): forward and reverse motion move a cursor over the
recorded trace, and an edit (``set``) or a ``hold`` re-executes natively from the start with the
edits applied at their ticks, producing a new trace. ``pos`` = instructions executed so far; the
state at the cursor is the state after tick ``pos - 1`` and the next instruction is ``ticks[pos]``.

Navigation: step / next / reverse_step / finish / seek in any clock domain (``domains``; ``define``
adds one), cont / reverse_cont to breakpoints and watches. Data: print / x / reads. Edits: set,
fork, compare, hold, checkpoint / restart. Predictions: whatif (sealed, then native, with the
verdicts ``path-diverged`` / ``fault`` / ``nonterminating`` and the paired (frozen-selection
effect, total effect)), forecast / verify (exact equality), sweep (a batched patch grid on lanes).
Recovery: symbols / circuit (``method``).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mdb.core.records import declare_kind, seal_record

from .. import FAULT, NONTERMINATING, PATH_DIVERGED, ClockDomain
from . import address as xa
from . import index as xi
from .binary import FLAG0, FLAGS, GPRS, Binary, site_name
from .lanes import Lanes, reference
from .trace import Trace, default_input

declare_kind("machine-forecast", required=("schema", "target", "predicted", "sealed_at"))
declare_kind("machine-verify", required=("schema", "forecast", "target", "predicted", "native", "verdict"))
declare_kind("machine-whatif", required=("schema", "edit", "consumer", "prediction", "sealed_prediction"))
declare_kind("machine-whatif-result", required=("schema", "whatif", "verdict", "total_effect", "frozen_effect"))


def attach(program, input: bytes | None = None, *, device="cpu", donors: list | None = None,
           cap: int | None = None) -> BinRun:
    """Run ``program`` (a reference-store directory or a ``Binary``) on ``input`` and stop at tick 0."""
    b = program if isinstance(program, Binary) else Binary.load(Path(program))
    run = BinRun(b, input if input is not None else default_input(b), device=device, donors=donors or [], cap=cap)
    run.runs["main"] = run
    return run


@dataclass
class Stop:
    reason: str            # boundary | breakpoint N | watch N | start | end | <machine stop kind>
    pos: int
    pc: int
    text: str
    uop: str | None = None

    def __str__(self) -> str:
        u = f" [{self.uop}]" if self.uop else ""
        return f"{self.reason}: pos {self.pos}  {self.pc:#x}  {self.text}{u}"


@dataclass
class _Point:
    id: int
    kind: str              # break | watch
    spec: Any
    mode: str = "change"
    pred: Callable | None = None
    enabled: bool = True
    hits: int = 0


@dataclass
class BinRun:
    binary: Binary
    input: bytes | None
    device: str = "cpu"
    donors: list = field(default_factory=list)
    cap: int | None = None
    name: str = "main"

    def __post_init__(self):
        self.edits: list[tuple] = []          # (at, site, byte value)
        self.held: tuple | None = None        # (reference run, what)
        self.pos, self.uop_i = 0, 0
        self.points: dict[int, _Point] = {}
        self.checkpoints: dict[str, tuple] = {}
        self.forecasts: list[dict] = []
        self.runs: dict[str, BinRun] = {}
        self.doms: dict[str, ClockDomain] = {}
        self._execute()
        self._ladder()

    # ------------------------------------------------------------------ execution
    def _execute(self) -> None:
        cap = self.cap or (4 * self.trace.T + 64 if hasattr(self, "trace") else 20000)
        lanes = Lanes(self.binary, [self.input], device=self.device, cap=cap)
        for at, site, val in self.edits:
            lanes.patch(at, site, val)
        if self.held is not None:
            ref, _what = self.held
            lanes.hold_to(reference(self.binary, ref.result), [0], [True])
        res = lanes.run(record="full")
        self.result = res
        self.trace = Trace(self.binary, res, 0, self.input)

    def _ladder(self) -> None:
        b = self.binary
        heads = {lp.head for lp in b.loops}
        for d in (ClockDomain("run", lambda tk: False),
                  ClockDomain("call", lambda tk: tk.kind in ("call", "ret"), "run"),
                  ClockDomain("loop", lambda tk: tk.next_pc in heads, "call"),
                  ClockDomain("block", lambda tk: tk.kind not in ("op",) or tk.next_pc in b.leaders, "loop"),
                  ClockDomain("instruction", lambda tk: True, "block")):
            self.define(d)
        for h in sorted(heads):
            self.define(ClockDomain(f"loop:{h:#x}", lambda tk, h=h: tk.next_pc == h, "call"))

    # ------------------------------------------------------------------ clock
    def domains(self) -> list[ClockDomain]:
        return [*self.doms.values(), ClockDomain("uop", lambda tk: True, "instruction")]

    def define(self, domain: ClockDomain) -> None:
        """Add a clock domain: any predicate over a tick (``Tick``: t, pc, next_pc, kind, depth, reads,
        writes, k, index). ``step 1 <name>`` then runs to the next tick where it holds."""
        self.doms[domain.name] = domain

    def phases(self, pc: int | None = None) -> list[str]:
        b = self.binary
        r = b.row_of(self.pc if pc is None else pc)
        g = lambda f: b.g(r, f)
        out = ["address"] if g("mbase") >= 0 or g("midx") >= 0 or g("mw") else []
        out += ["load"] if g("mload") else []
        out += ["alu"] + (["flags"] if g("fmask") else [])
        out += ["writeback"] if g("dk") == 1 or g("d2") >= 0 or g("spk") else []
        out += ["store"] if g("dk") == 2 else []
        return out + ["pc"]

    @property
    def pc(self) -> int:
        tr = self.trace
        return tr.ticks[self.pos].pc if self.pos < tr.T else -1

    def where(self) -> Stop:
        return self._stop("here")

    def _stop(self, reason: str) -> Stop:
        tr = self.trace
        if self.pos >= tr.T:
            return Stop(reason if reason not in ("boundary", "here") else f"end ({tr.stop})", self.pos, -1, "")
        tk = tr.ticks[self.pos]
        ph = self.phases()
        return Stop(reason, self.pos, tk.pc, f"{tk.text}   ; {xi.text(tk.index)}",
                    ph[self.uop_i] if self.uop_i else None)

    def step(self, n: int = 1, unit: str = "instruction") -> Stop:
        """Run n units of a clock domain forward (``unit="uop"``: phases inside one instruction)."""
        if unit == "uop":
            for _ in range(n):
                if self.uop_i + 1 < len(self.phases()):
                    self.uop_i += 1
                elif self.pos < self.trace.T:
                    self.pos, self.uop_i = self.pos + 1, 0
            return self._stop("boundary")
        return self._forward(n, self.doms[unit], depth_cap=None)

    def next(self, n: int = 1, unit: str = "instruction") -> Stop:
        """Like step, counting only units that end at or above the starting call depth (steps over calls)."""
        return self._forward(n, self.doms[unit], depth_cap=self._depth())

    def _depth(self) -> int:
        return self.trace.ticks[self.pos - 1].depth if self.pos else 0

    def _forward(self, n: int, d: ClockDomain, depth_cap) -> Stop:
        tr, seen = self.trace, 0
        self.uop_i = 0
        while seen < n:
            if self.pos >= tr.T:
                return self._stop("end")
            tk = tr.ticks[self.pos]
            self.pos += 1
            seen += bool(d.boundary(tk)) and (depth_cap is None or tk.depth <= depth_cap)
        return self._stop("boundary")

    def finish(self) -> Stop:
        """Run until the current frame returns."""
        base = self._depth()
        tr = self.trace
        while self.pos < tr.T:
            tk = tr.ticks[self.pos]
            self.pos += 1
            if tk.kind == "ret" and tk.depth < base:
                return self._stop("boundary")
        return self._stop("end")

    def reverse_step(self, n: int = 1, unit: str = "instruction") -> Stop:
        """Back to the end of the n-th previous unit."""
        d, seen, i = self.doms[unit], 0, self.pos - 2
        self.uop_i = 0
        while i >= 0 and seen < n:
            if d.boundary(self.trace.ticks[i]):
                seen += 1
                if seen == n:
                    break
            i -= 1
        self.pos = max(i + 1, 0)
        return self._stop("boundary")

    def seek(self, pos: int) -> Stop:
        self.pos, self.uop_i = max(0, min(pos, self.trace.T)), 0
        return self._stop("boundary")

    # ------------------------------------------------------------------ addresses and values
    def addr(self, text: str) -> xa.Addr:
        """Parse an address; a bare site (``rax``, ``mem[rbp-0x8:8]``, ``flags.ZF``) is at the cursor."""
        try:
            return xa.parse(text)
        except xa.AddressError:
            t = self.pos - 1
            return xa.Addr(None, ("step", t) if t >= 0 else ("init",), xa.parse_site(text))

    def _run_of(self, a: xa.Addr) -> BinRun:
        if a.origin in (None, "bin", self.name):
            return self
        if a.origin not in self.runs:
            raise xa.AddressError(f"no run named {a.origin!r} (runs: {sorted(self.runs)})")
        return self.runs[a.origin]

    def read(self, text: str) -> dict:
        """``print``: the value of an address (exact bits, authority ``measured``)."""
        a = self.addr(text)
        run = self._run_of(a)
        t = xa.resolve_tick(a, run.trace)
        sites, lo, nbits = xa.sites(a, run.trace, t)
        v = 0
        for i, s in enumerate(sites):
            v |= run.trace.value(s, t) << (8 * i if s < FLAG0 or s >= FLAG0 + 6 else 0)
        v = (v >> lo) & ((1 << nbits) - 1)
        return {"address": a.text(), "tick": t, "value": v, "hex": f"{v:#x}", "bits": nbits, "authority": "measured",
                "sites": [site_name(s) for s in sites]}

    def print(self, text: str) -> str:
        r = self.read(text)
        return f"{r['address']} = {r['hex']} ({r['value']}) [{r['authority']}]"

    def x(self, text: str, count: int = 8, width: int = 8) -> list[int]:
        """Examine ``count`` words of ``width`` bytes starting at a memory address."""
        a = self.addr(text)
        run = self._run_of(a)
        t = xa.resolve_tick(a, run.trace)
        start = xa.sites(a, run.trace, t)[0][0]
        raw = run.trace.mem(t, start, count * width)
        return [int.from_bytes(raw[i * width:(i + 1) * width], "little") for i in range(count)]

    def _state(self, pos: int) -> dict:
        """Names a break/watch predicate sees at position pos (state after pos ticks)."""
        tr, t = self.trace, pos - 1
        env = {g: tr.reg(t, i) for i, g in enumerate(GPRS)}
        env.update({f: tr.flag(t, i) for i, f in enumerate(FLAGS)})
        nxt = tr.ticks[pos] if pos < tr.T else None
        env.update(pc=nxt.pc if nxt else -1, k=nxt.k if nxt else -1, t=pos,
                   mem=lambda a, n=8: int.from_bytes(tr.mem(t, a, n), "little"),
                   text=nxt.text if nxt else "")
        return env

    # ------------------------------------------------------------------ breakpoints, watches
    def break_at(self, spec) -> int:
        """Break before an instruction: a pc (int or hex), ``pc#k`` (execution index instance),
        ``i<t>``, or a predicate (callable on the state dict, or an expression string over
        rax..r15, CF..OF, pc, k, t, mem(addr, n), text)."""
        if isinstance(spec, str) and "#" in spec and spec.startswith("0x"):
            pc, k = spec.split("#")
            spec = ("inst", int(pc, 16), int(k))
        elif isinstance(spec, str) and spec.startswith("i") and spec[1:].isdigit():
            spec = ("step", int(spec[1:]))
        elif isinstance(spec, str) and spec.startswith("0x") and all(ch in "0123456789abcdefx" for ch in spec.lower()):
            spec = ("pc", int(spec, 16))
        elif isinstance(spec, int):
            spec = ("pc", spec)
        elif isinstance(spec, str):
            code = compile(spec, "<break>", "eval")
            spec = ("pred", lambda env, c=code: eval(c, {"__builtins__": {}}, env))  # noqa: S307
        elif callable(spec):
            spec = ("pred", spec)
        pid = len(self.points) + 1
        self.points[pid] = _Point(pid, "break", spec)
        return pid

    def watch(self, text: str, mode: str = "change", pred: str | Callable | None = None) -> int:
        """Watch a site (resolved at the cursor; memory by its address): stop after a tick that changes
        it (``change``), or at ticks where ``pred(v)`` holds (``level``) or becomes true (``edge``)."""
        a = self.addr(text)
        t = xa.resolve_tick(a, self.trace)
        sites, lo, nbits = xa.sites(a, self.trace, t)
        if isinstance(pred, str):
            code = compile(pred, "<watch>", "eval")
            pred = lambda v, c=code: eval(c, {"__builtins__": {}}, {"v": v})  # noqa: E731, S307
        pid = len(self.points) + 1
        self.points[pid] = _Point(pid, "watch", (sites, lo, nbits, a.site), mode, pred)
        return pid

    def delete(self, pid: int | None = None) -> None:
        self.points.clear() if pid is None else self.points.pop(pid, None)

    def _site_value(self, spec, t: int) -> int:
        sites, lo, nbits, _ = spec
        v = 0
        for i, s in enumerate(sites):
            v |= self.trace.value(s, t) << (8 * i if s < FLAG0 or s >= FLAG0 + 6 else 0)
        return (v >> lo) & ((1 << nbits) - 1)

    def _hit(self, p: _Point, pos: int) -> bool:
        tr = self.trace
        if p.kind == "break":
            if pos >= tr.T:
                return False
            kind = p.spec[0]
            tk = tr.ticks[pos]
            if kind == "pc":
                return tk.pc == p.spec[1]
            if kind == "inst":
                return tk.pc == p.spec[1] and tk.k == p.spec[2]
            if kind == "step":
                return pos == p.spec[1]
            return bool(p.spec[1](self._state(pos)))
        if pos == 0:
            return False
        now, before = self._site_value(p.spec, pos - 1), self._site_value(p.spec, pos - 2)
        if p.mode == "change":
            touched = set(p.spec[0]) & {s for s, _ in tr.ticks[pos - 1].writes}
            return bool(touched) and now != before
        if p.mode == "level":
            return bool(p.pred(now))
        return bool(p.pred(now)) and not bool(p.pred(before))

    def cont(self, n: int = 1) -> Stop:
        """Run to the n-th next breakpoint or watch hit (or the end of the run)."""
        return self._scan(n, +1)

    def reverse_cont(self, n: int = 1) -> Stop:
        return self._scan(n, -1)

    def _scan(self, n: int, direction: int) -> Stop:
        tr, seen = self.trace, 0
        self.uop_i = 0
        pos = self.pos
        while True:
            pos += direction
            if pos < 0 or pos > tr.T:
                self.pos = max(0, min(pos, tr.T))
                return self._stop("start" if pos < 0 else "end")
            hits = [p for p in self.points.values() if p.enabled and self._hit(p, pos)]
            if hits:
                seen += 1
                if seen == n:
                    self.pos = pos
                    for p in hits:
                        p.hits += 1
                    tag = ", ".join(f"{p.kind} {p.id}" for p in hits)
                    if any(p.kind == "watch" for p in hits):
                        w = next(p for p in hits if p.kind == "watch")
                        tag += (f" (old {self._site_value(w.spec, pos - 2):#x} new "
                                f"{self._site_value(w.spec, pos - 1):#x}, written at {tr.ticks[pos - 1].pc:#x})")
                    return self._stop(tag)

    # ------------------------------------------------------------------ checkpoints, edits, forks
    def checkpoint(self, name: str) -> dict:
        self.checkpoints[name] = (self.pos, list(self.edits), self.held)
        return {"checkpoint": name, "pos": self.pos, "edits": len(self.edits)}

    def restart(self, name: str) -> Stop:
        pos, edits, held = self.checkpoints[name]
        if edits != self.edits or held != self.held:
            self.edits, self.held = list(edits), held
            self._execute()
        return self.seek(pos)

    def set(self, text: str, value) -> dict:
        """Poke a site at the cursor and re-execute natively from there. ``value`` is an int or another
        run's address (``donor:i12.rax``). Returns how the new run differs from the old one."""
        a = self.addr(text)
        if isinstance(value, str):
            value = self.read(value)["value"]
        t = self.pos - 1
        sites, lo, nbits = xa.sites(a, self.trace, t)
        if lo or nbits % 8 and nbits != 1:
            raise xa.AddressError("set takes whole bytes or one flag")
        if not 0 <= int(value) < 1 << nbits:
            raise ValueError(f"{int(value):#x} does not fit {nbits} bits")
        old_trace = self.trace
        for i, s in enumerate(sites):
            self.edits.append((self.pos, s, (int(value) >> (8 * i)) & 0xFF if nbits != 1 else int(value) & 1))
        self._execute()
        return self._delta(old_trace, self.trace, a.text(), value)

    def _delta(self, old: Trace, new: Trace, what: str, value) -> dict:
        T = min(old.T, new.T)
        ctrl = next((t for t in range(T) if old.pc[t] != new.pc[t]), None if old.T == new.T else T)
        addr = next((t for t in range(T) if old.ea[t] != new.ea[t] and old.pc[t] == new.pc[t]
                     and self.binary.g(self.binary.row_of(int(old.pc[t])), "mload")
                     | self.binary.g(self.binary.row_of(int(old.pc[t])), "mstore")), None)
        y_changed = (old.y, old.exit_code) != (new.y, new.exit_code) or new.stop != "exit"
        verdict = verdict_of(new.stop, ctrl is not None or addr is not None, y_changed)
        return {"edit": what, "value": f"{int(value):#x}", "at_pos": self.pos, "verdict": verdict,
                "path_diverged_at": ctrl, "address_diverged_at": addr, "stop": new.stop,
                "y_changed": y_changed, "y": new.y.hex(), "exit_code": new.exit_code}

    def fork(self, name: str) -> BinRun:
        other = copy.copy(self)
        other.name, other.edits, other.points = name, list(self.edits), {}
        other.checkpoints, other.forecasts = {}, []
        self.runs[name] = other
        other.runs = self.runs
        return other

    def compare(self, a: str = "main", b: str | None = None) -> dict:
        ra, rb = self.runs[a], self.runs[b] if b else self
        d = ra._delta(ra.trace, rb.trace, f"{a} vs {rb.name}", 0)
        return {"a": a, "b": rb.name, **{k: v for k, v in d.items() if k not in ("edit", "value", "at_pos")}}

    def hold(self, ref: str | None = "main", what: tuple = ("ctrl", "addr")) -> dict:
        """Freeze this run's selection reads (branch outcomes and effective addresses) to run ``ref``'s
        and re-execute; ``hold(None)`` releases it."""
        self.held = None if ref is None else (self.runs[ref] if ref != self.name else _Frozen(self.result), what)
        self._execute()
        return {"hold": ref, "what": list(what), "T": self.trace.T, "stop": self.trace.stop}

    # ------------------------------------------------------------------ Program protocol
    def sites(self) -> list[int]:
        return sorted({c.site for c in self.trace.cells})

    def snapshot(self) -> tuple:
        return (self.pos, list(self.edits), self.held)

    def restore(self, snap: tuple) -> None:
        self.checkpoints["_snap"] = snap
        self.restart("_snap")

    def poke(self, site: str, value: int) -> None:
        self.set(site, value)

    def reads_of(self, t: int) -> list:
        return self.trace.reads_of(t)

    def writes_of(self, t: int) -> list:
        return self.trace.writes_of(t)

    # ------------------------------------------------------------------ reads
    def reads(self, text: str) -> list[dict]:
        """Instruction instances that read the cell holding ``text`` at its tick, typed data/addr/ctrl
        (mass is 1: one exact read)."""
        a = self.addr(text)
        t = xa.resolve_tick(a, self.trace)
        by = {}
        for s in xa.sites(a, self.trace, t)[0]:
            c = self.trace.cell_at(s, t)
            if c is None:
                continue
            for rt, edge in c.readers:
                tk = self.trace.ticks[rt]
                r = by.setdefault((rt, edge), {"reader": f"{tk.pc:#x}#{tk.k}", "tick": rt, "text": tk.text,
                                               "edge": edge, "mass": 1, "cells": [], "sites": []})
                r["cells"].append(self.trace.describe(c))
                r["sites"].append(site_name(s))
        return sorted(by.values(), key=lambda r: (r["tick"], r["edge"]))

    # ------------------------------------------------------------------ predictions
    def _identity(self):
        b = self.binary
        return (f"x86:{b.name}:{b.image_sha256[:12]}", {"kind": "x86-64-binary", "program": b.name,
                                                        "image_sha256": b.image_sha256}, {"backend": "torchx86"})

    def forecast(self, text: str, value: int) -> dict:
        """Seal a claim about a later occurrence's exact value (verdict by ``verify``)."""
        a = self.addr(text)
        sid, ident, execu = self._identity()
        body = {"schema": "mdb-machine-forecast-v1", "target": a.text(),
                "predicted": {"hex": f"{int(value):#x}", "authority": "supplied"},
                "sealed_at": {"pos": f"{self.pos}", "run": self.name}}
        rec = seal_record(session_id=sid, kind="machine-forecast", model_identity=ident, execution=execu,
                          authority="supplied", body=body)
        self.forecasts.append({"record": rec, "addr": a, "value": int(value), "done": False})
        return rec

    def verify(self) -> list[dict]:
        """Check open forecasts whose target tick the cursor has passed: exact equality."""
        out = []
        sid, ident, execu = self._identity()
        for f in self.forecasts:
            if f["done"]:
                continue
            try:
                t = xa.resolve_tick(f["addr"], self.trace)
            except xa.AddressError:
                continue
            if t >= self.pos:
                continue
            native = self.read(f["addr"].text())["value"]
            verdict = "confirmed" if native == f["value"] else "not confirmed"
            body = {"schema": "mdb-machine-verify-v1", "forecast": f["record"]["id"], "target": f["addr"].text(),
                    "predicted": f"{f['value']:#x}", "native": f"{native:#x}", "verdict": verdict}
            rec = seal_record(session_id=sid, kind="machine-verify", model_identity=ident, execution=execu,
                              authority="measured", body=body)
            f["done"] = True
            out.append({"target": f["addr"].text(), "verdict": verdict, "native": native, "predicted": f["value"],
                        "record": rec["id"]})
        return out

    def taint(self, sites: list[int], t: int) -> set[int]:
        """Output byte positions of y reachable from ``sites`` after tick t through any typed read
        (data, addr or ctrl; the structural upper bound, P3 of the PREREG)."""
        tr = self.trace
        tainted, out, ctrl = set(sites), set(), False
        for tk in tr.ticks[t + 1:]:
            hit = ctrl or any(s in tainted for s, _e in tk.reads)
            ctrl = ctrl or any(e == "ctrl" and s in tainted for s, e in tk.reads)   # taints everything after
            if tk.kind == "syscall":
                out_bytes = [s for s, e in tk.reads if e == "data" and s >= 0x1000]
                if out_bytes:
                    rsi = tr.reg(tk.t - 1, 6)
                    moved = ctrl or any(e == "addr" and s in tainted for s, e in tk.reads)
                    out |= {s - rsi for s in out_bytes if moved or s in tainted}
                elif ctrl or 8 * 7 in tainted:
                    out.add(-1)                      # the exit code
            for s, _slot in tk.writes:
                (tainted.add if hit else tainted.discard)(s)
        return out

    def whatif(self, text: str, value) -> dict:
        """Seal a predicted effect on y (taint reach), then measure natively: one free lane and one lane
        whose selection reads are held to this run. The session is left exactly where it was."""
        a = self.addr(text)
        if isinstance(value, str):
            value = self.read(value)["value"]
        t = self.pos - 1
        sites, lo, nbits = xa.sites(a, self.trace, t)
        reach = self.taint(sites, t)
        sid, ident, execu = self._identity()
        pred_body = {"schema": "mdb-machine-whatif-v1", "edit": {"address": a.text(), "value": f"{int(value):#x}"},
                     "consumer": "y = last write bytes + exit code",
                     "prediction": {"method": "taint reach over typed reads (structural upper bound)",
                                    "effect": "possible" if reach else "none",
                                    "y_bytes": ",".join(str(i) for i in sorted(reach))[:200]},
                     "sealed_prediction": "sealed before the native lanes run"}
        sealed = seal_record(session_id=sid, kind="machine-whatif", model_identity=ident, execution=execu,
                             authority="derived-local", body=pred_body)
        lanes = Lanes(self.binary, [self.input, self.input], device=self.device, cap=4 * self.trace.T + 64)
        for at, s, v in self.edits:
            lanes.patch(at, s, v)
        for i, s in enumerate(sites):
            lanes.patch(self.pos, s, (int(value) >> (8 * i)) & 0xFF if nbits != 1 else int(value) & 1)
        lanes.hold_to(reference(self.binary, self.result), [0, 0], [False, True])
        r = lanes.run()
        base = (self.trace.y, self.trace.exit_code)
        total = (r.y[0], int(r.exit_code[0])) != base or r.stop_kind(0) != "exit"
        frozen = (r.y[1], int(r.exit_code[1])) != base or r.stop_kind(1) != "exit"
        diverged = int(r.ctrl_at[0]) >= 0 or int(r.addr_at[0]) >= 0
        verdict = verdict_of(r.stop_kind(0), diverged, total)
        body = {"schema": "mdb-machine-whatif-result-v1", "whatif": sealed["id"], "verdict": verdict,
                "total_effect": "effect" if total else "no effect",
                "frozen_effect": "effect" if frozen else "no effect",
                "path_diverged_at": f"{int(r.ctrl_at[0])}", "address_diverged_at": f"{int(r.addr_at[0])}",
                "selection_moved_under_hold_at": f"{int(r.moved_at[1])}",
                "stop": r.stop_kind(0), "y": r.y[0].hex(), "y_frozen": r.y[1].hex(),
                "bound_holds": "false" if total and not reach else "true"}
        res = seal_record(session_id=sid, kind="machine-whatif-result", model_identity=ident, execution=execu,
                          authority="measured", body=body)
        return {**body, "prediction": pred_body["prediction"], "records": [sealed["id"], res["id"]]}

    # ------------------------------------------------------------------ batched verbs
    def sweep(self, cells: list[int] | None = None, **kw) -> dict:
        """Patch every listed cell (default: every cell) from the donors under the rule grid, on lanes
        (``grid.run_grid``): one row per patch."""
        from .grid import run_grid  # noqa: PLC0415

        return run_grid(self.binary, self.input, self.donors, cells=cells, device=self.device, **kw)

    def symbols(self, **kw):
        """Recover the symbol table by intervention over the donors (``method.recover``)."""
        from .method import recover  # noqa: PLC0415

        if not self.donors:
            raise ValueError("symbols needs donor inputs (attach(..., donors=[...]))")
        return recover(self.binary, self.input, self.donors, device=self.device, **kw)


@dataclass
class _Frozen:
    """A run's recorded result kept as a hold reference after the run itself changes."""
    result: Any


def verdict_of(stop: str, diverged: bool, y_changed: bool) -> str:
    if stop in ("step_limit",):
        return NONTERMINATING
    if stop not in ("exit",):
        return FAULT
    if diverged:
        return PATH_DIVERGED
    return "effect" if y_changed else "no effect"
