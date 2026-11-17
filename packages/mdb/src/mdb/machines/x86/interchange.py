"""Two verbs a CoT-faithfulness monitor needs on binaries (study C, 2026-11-16): the answer's dependence cone,
and multi-donor interchange of a whole block (a buffer, a field) on many base runs at once.

``cone(trace)``: the backward dynamic dependence cone of the answer, the bytes the run's last ``write``
syscall emits. Rule, applied tick by tick from that syscall back to tick 0:

* the answer tick is in the cone and every site it reads is needed;
* a tick is in the cone when it writes a needed site, or (``control=True``) when it reads anything through a
  ``ctrl`` edge (a jcc's or cmov's flags, a ``ret``'s return address, an indirect jump, a syscall number):
  every control decision before the answer is a root, with no post-dominator pruning (a conservative
  upper bound on control dependence);
* a cone tick's written sites stop being needed and every site it reads becomes needed, except that a
  ``write`` syscall moves no buffer byte into a register: only its register reads count.

``reads`` lists every (tick, site, edge) read by a cone tick: "the answer's cone contains a read of X" is a
test on that list (the read-set existence monitor, the attention-mass analogue).

``interchange(binary, bases, donors, text)``: for every base run and donor run, overwrite the block named
by ``text`` (mdb's x86 address grammar: a tick part and a site of whole bytes, e.g.
``init.mem[0x40300c:1]`` or ``0x4010d3#0.mem[0x403010:10]``) in the base run, right after the address's
tick in the base run, with the donor run's bytes of the same address at the same tick instance in the
donor run, and run the base input to its end. One lane per informative (base, donor) pair, all in one
batch. A donor whose bytes equal the base's is not run (an identity patch, not informative); a donor
without the instance is ``unresolved``. Per base the donors are pooled by ``mdb.analysis.donors.pooled``
on the answer: ``effect`` when at least one donor changes y (the last write's bytes or the exit code; a
fault or the step cap count as a change).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import address as xa
from .binary import RAX, Binary
from .lanes import Lanes, reference
from .trace import CTRL, Trace

MEM0 = 0x1000


@dataclass
class Cone:
    answer_tick: int
    ticks: set = field(default_factory=set)
    reads: list = field(default_factory=list)      # [(tick, site, edge)] read by cone ticks
    control: bool = True

    def reads_of(self, sites) -> list:
        """The cone's reads of any of ``sites`` (tick, site, edge)."""
        want = set(sites)
        return [r for r in self.reads if r[1] in want]


def _is_write(trace: Trace, t: int) -> bool:
    return trace.ticks[t].kind == "syscall" and trace.reg(t - 1, RAX) == 1


def cone(trace: Trace, *, control: bool = True, answer_tick: int | None = None) -> Cone:
    """The answer's backward dependence cone (module docstring)."""
    if answer_tick is None:
        writes = [t for t in range(trace.T) if _is_write(trace, t)]
        if not writes:
            raise ValueError("the run has no write syscall: no answer to take the cone of")
        answer_tick = writes[-1]
    out = Cone(answer_tick, control=control)
    tk = trace.ticks[answer_tick]
    need = {s for s, _ in tk.reads}
    out.ticks.add(answer_tick)
    out.reads += [(answer_tick, s, e) for s, e in tk.reads]
    for t in range(answer_tick - 1, -1, -1):
        tk = trace.ticks[t]
        written = {s for s, _ in tk.writes}
        if not (written & need or (control and any(e == CTRL for _, e in tk.reads))):
            continue
        out.ticks.add(t)
        need -= written
        reads = [(s, e) for s, e in tk.reads if not (tk.kind == "syscall" and s >= MEM0)]
        need |= {s for s, _ in reads}
        out.reads += [(t, s, e) for s, e in reads]
    return out


def _block(addr: xa.Addr, trace: Trace) -> tuple[int, list[int], list[int]]:
    """(tick, byte sites, byte values) of ``addr`` in ``trace``; raises AddressError when absent."""
    t = xa.resolve_tick(addr, trace)
    sites, lo, nbits = xa.sites(addr, trace, t)
    if lo or (nbits % 8 and nbits != 1):
        raise xa.AddressError(f"{addr.text()}: interchange takes whole bytes or one flag")
    return t, sites, [trace.value(s, t) for s in sites]


def interchange(binary: Binary, bases: list, donors: list, text: str, *, device="cpu", res=None,
                chunk: int = 8192, cap_factor: int = 4) -> list[dict]:
    """Multi-donor block interchange on many bases (module docstring). ``bases``/``donors`` are ``Trace``s
    (``trace.record``); ``res`` is the bases' recorded ``Result`` (``record="full"``), which adds per-lane
    path divergence against the base run. Returns one dict per base: the pooled verdict and ``per_donor``."""
    from mdb.analysis.donors import EFFECT, NO_EFFECT, UNRESOLVED, pooled  # noqa: PLC0415

    addr = xa.parse(text)
    dev = torch.device(device)
    dblocks = []
    for d in donors:
        try:
            dblocks.append(_block(addr, d))
        except xa.AddressError:
            dblocks.append(None)
    lanes_spec, rows = [], []
    for i, b in enumerate(bases):
        t, sites, bval = _block(addr, b)
        at = 0 if t < 0 else t + 1
        per = []
        for j, db in enumerate(dblocks):
            if db is None or len(db[2]) != len(bval):
                per.append({"donor": j, "verdict": UNRESOLVED, "informative": None, "ran": False})
                continue
            info = db[2] != bval
            row = {"donor": j, "verdict": NO_EFFECT, "informative": info, "ran": info, "path_diverged": False}
            if info:
                row["lane"] = len(lanes_spec)
                lanes_spec.append((i, at, sites, db[2]))
            per.append(row)
        rows.append({"tick": t, "sites": sites, "base_bytes": bval, "per_donor": per})
    ref = reference(binary, res, dev) if res is not None else None
    out_lane: list[dict] = [None] * len(lanes_spec)
    cap = cap_factor * max(b.T for b in bases) + 64
    for s in range(0, len(lanes_spec), chunk):
        part = lanes_spec[s:s + chunk]
        lanes = Lanes(binary, [bases[i].input for i, *_ in part], device=dev, cap=cap)
        n = max(len(p[2]) for p in part)
        for k in range(n):
            lanes.patch([p[1] if k < len(p[2]) else -1 for p in part],
                        [p[2][k] if k < len(p[2]) else -1 for p in part],
                        [p[3][k] if k < len(p[3]) else 0 for p in part])
        if ref is not None:
            lanes.compare_to(ref, [p[0] for p in part])
        r = lanes.run()
        for q, (i, *_rest) in enumerate(part):
            b = bases[i]
            stop = r.stop_kind(q)
            changed = (r.y[q], int(r.exit_code[q])) != (b.y, b.exit_code) or stop != "exit"
            div = ref is not None and (int(r.ctrl_at[q]) >= 0 or int(r.addr_at[q]) >= 0)
            out_lane[s + q] = {"y_changed": changed, "y": r.y[q].hex(), "stop": stop, "path_diverged": bool(div)}
    result = []
    for rw in rows:
        for row in rw["per_donor"]:
            if "lane" in row:
                lane = out_lane[row.pop("lane")]
                row.update(lane, verdict=EFFECT if lane["y_changed"] else NO_EFFECT)
        site_text = [f"{s:#x}" if s >= MEM0 else str(s) for s in rw["sites"]]
        result.append({"address": text, "tick": rw["tick"], "sites": site_text, "base_bytes": rw["base_bytes"],
                       **pooled(rw["per_donor"]), "per_donor": rw["per_donor"]})
    return result


__all__ = ["Cone", "cone", "interchange"]
