"""B copies of one binary in lockstep on torchx86: inputs, patches, holds, divergence, recording.

One lane = one run. A **patch** writes a value into one site (register byte, flag, memory byte) of
one lane when that lane has retired ``at`` instructions (``at = t_def + 1`` patches the cell that
tick ``t_def`` defined; ``at = 0`` patches the initial state). A **hold** keeps a lane's selection
reads at a reference run's: after each instruction the next pc is forced to the reference's
(branch outcomes and returns frozen) and every memory access, and the ``write`` syscall's buffer,
use the reference's effective address. Value reads are left free. This is the binary form of
``hold p`` (frozen attention).

Divergence against a reference run is tracked per lane: the first tick whose pc differs (``ctrl``)
and the first tick whose computed effective address differs (``addr``), checked on the address the
lane computed, before any hold replaced it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .binary import (FLAG0, NSLOT_LANE, RCX, RDX, RSI, RSP, S_D2, S_DST, S_FL, S_MEM, S_SP, Binary,
                     torchx86)

ALIVE = 0


def _held_machine_class():
    from torchx86.machine import Machine  # noqa: PLC0415

    class HeldMachine(Machine):
        """torchx86 ``Machine`` whose memory accesses can be redirected to a reference run's."""

        hold = None          # bool [B] or None
        ref = None           # dict with ea [NR,T], acc [NR,T], rsi [NR,T], T [NR]; rix [B]

        def _bytes(self, addr, last, k):
            if self.hold is not None and (k is self.k8 or k is self.kout):
                ref, rix = self.ref, self.ref["rix"]
                t = self.steps.clamp(max=ref["ea"].shape[1] - 1)
                inref = self.steps < ref["T"][rix]
                if k is self.k8:
                    want, use = ref["ea"][rix, t], self.hold & inref & ref["acc"][rix, t]
                    self._moved = use & (want != addr)
                else:
                    want, use = ref["rsi"][rix, t], self.hold & inref
                last = torch.where(use, last + (want - addr), last)
                addr = torch.where(use, want, addr)
            return super()._bytes(addr, last, k)

    return HeldMachine


@dataclass
class Result:
    """Per lane: stop code and kind, ticks T (instructions incl. the final exit), exit code, y (the
    last ``write``'s bytes), divergence ticks (-1 none), and the recordings asked for."""
    stop: torch.Tensor
    T: torch.Tensor
    exit_code: torch.Tensor
    y: list
    ctrl_at: torch.Tensor | None = None
    addr_at: torch.Tensor | None = None
    moved_at: torch.Tensor | None = None       # held lanes: first tick a hold replaced a moved address
    pc: torch.Tensor | None = None             # [B, T] pc per tick (-1 after the lane stopped)
    slots: torch.Tensor | None = None          # [B, T, 38] int16 written byte/bit per slot, -1 none
    full: dict | None = None                   # per-tick regs / flags / ea / store bytes (``record="full"``)

    def stop_kind(self, lane: int) -> str:
        from torchx86.machine import STOP_KINDS  # noqa: PLC0415

        return STOP_KINDS.get(int(self.stop[lane]), "alive")


class Lanes:
    def __init__(self, binary: Binary, inputs: list, *, device="cpu", cap: int = 20000, resident=None,
                 extra_resident=()):
        torchx86()
        self.b, self.B, self.dev, self.cap = binary, len(inputs), torch.device(device), cap
        Machine = _held_machine_class()
        self.m = Machine(binary.table, binary.pages, binary.perms, regs=binary.regs, eflags=binary.eflags,
                         pc=binary.entry, batch=self.B, device=device, step_cap=cap,
                         resident=None if resident is None else sorted(set(resident) | set(extra_resident)))
        self.rows = binary.table.rows.to(self.dev)
        self.F = binary.F
        self._inject(inputs)
        self.patches: list[tuple] = []
        self.hold = None
        self.ref = None

    # ------------------------------------------------------------------ setup
    def _offsets(self, addr: torch.Tensor):
        """Flat offsets into lane memory for byte addresses (-1 when not resident)."""
        m = self.m
        eq = (addr >> 12)[..., None] == m.pbase
        i = (eq.long() * m.pidx).sum(-1)
        off = m.rw_off[i]
        ok = eq.any(-1) & (off >= 0)
        return torch.where(ok, off + (addr & 0xFFF), -1)

    def _inject(self, inputs: list) -> None:
        if all(x is None for x in inputs):
            return
        if self.b.input_addr is None:
            raise ValueError("inputs given but the binary has no input buffer address")
        n = max(len(x) for x in inputs if x is not None)
        off = self._offsets(torch.arange(n, device=self.dev) + self.b.input_addr)
        if bool((off < 0).any()):
            raise ValueError("the input buffer is not on a resident writable page")
        for lane, x in enumerate(inputs):
            if x is not None:
                self.m.mem[lane, off[:len(x)]] = torch.tensor(list(x), dtype=torch.uint8, device=self.dev)

    def patch(self, at, site, value) -> None:
        """Queue one patch per lane (tensors [B]; ``at`` < 0 or ``site`` < 0 means none)."""
        t = lambda v: torch.as_tensor(v, dtype=torch.int64, device=self.dev).expand(self.B).clone()
        self.patches.append((t(at), t(site), t(value)))

    def hold_to(self, ref: dict, rix, lanes) -> None:
        """Hold ``lanes`` (bool [B]) to reference runs ``ref`` (from ``reference``), lane -> ``rix``."""
        self.ref = {**ref, "rix": torch.as_tensor(rix, device=self.dev)}
        self.hold = torch.as_tensor(lanes, dtype=torch.bool, device=self.dev)
        self.m.ref, self.m.hold = self.ref, self.hold

    def compare_to(self, ref: dict, rix) -> None:
        """Track divergence against reference runs (no hold)."""
        self.ref = {**ref, "rix": torch.as_tensor(rix, device=self.dev)}

    def _apply(self) -> None:
        m = self.m
        for at, site, val in self.patches:
            hit = (at >= 0) & (site >= 0) & (m.steps == at) & (m.stop == ALIVE)
            if not bool(hit.any()):
                continue
            reg = hit & (site < FLAG0)
            if bool(reg.any()):
                r, k = (site // 8).clamp(0, 15), site % 8
                old = m.regs.gather(1, r[:, None])[:, 0]
                top = k == 7                      # byte 7: keep every shift inside int64
                mask = torch.where(top, -(1 << 56), torch.full_like(k, 0xFF) << (8 * k).clamp(max=48))
                v = (val & 0xFF)
                byte = torch.where(top, torch.where(v >= 128, v - 256, v) << 56, v << (8 * k).clamp(max=48))
                m.regs.scatter_(1, r[:, None], torch.where(reg, (old & ~mask) | byte, old)[:, None])
            fl = hit & (site >= FLAG0) & (site < FLAG0 + 6)
            if bool(fl.any()):
                f = (site - FLAG0).clamp(0, 5)
                cur = m.flags.gather(1, f[:, None])[:, 0]
                m.flags.scatter_(1, f[:, None], torch.where(fl, val != 0, cur)[:, None])
            mem = hit & (site >= 0x1000)
            if bool(mem.any()):
                off = self._offsets(site)
                ok = mem & (off >= 0)
                flat = torch.where(ok, self.lane_base * m.NW + off, self.lane_base * m.NW + m.NW - 1)
                cur = m.mem.view(-1)[flat]
                m.mem.view(-1).index_put_((flat,), torch.where(ok, val.to(torch.uint8), cur))

    # ------------------------------------------------------------------ run
    def run(self, record: str = "none") -> Result:
        """Run every lane to its stop. ``record``: none | pc | slots | full."""
        m, dev, B, F = self.m, self.dev, self.B, self.F
        self.lane_base = torch.arange(B, device=dev)
        last_start = torch.zeros(B, dtype=torch.int64, device=dev)
        ctrl_at = torch.full((B,), -1, dtype=torch.int64, device=dev)
        addr_at = torch.full_like(ctrl_at, -1)
        moved_at = torch.full_like(ctrl_at, -1)
        pcs, slots, full = [], [], {k: [] for k in ("regs", "flags", "ea", "store_n", "new", "old", "rsi")}
        exit_tick = torch.zeros(B, dtype=torch.bool, device=dev)
        fdef = torch.tensor(self.b.fdef, device=dev)
        shift_row = torch.tensor(self.b.shift_row, device=dev)
        k8 = torch.arange(8, device=dev)
        with torch.inference_mode():
            for i in range(self.cap + 1):
                self._apply()
                alive = m.stop == ALIVE
                if not bool(alive.any()):
                    break
                t = m.steps.clone()
                regs_before = m.regs.clone() if record in ("slots", "full") else None
                rsi_before = m.regs[:, RSI].clone()
                out_len_before = m.out_len.clone()
                m._moved = None
                rec = m.step(record=True)
                retired = rec["retired"]
                exited = alive & (m.stop == 1)
                exit_tick |= exited
                ticked = retired | exited
                last_start = torch.where(m.out_len != out_len_before, out_len_before, last_start)
                row = rec["row"]
                U = self.rows[row]
                if self.ref is not None:
                    ref, rix = self.ref, self.ref["rix"]
                    Tr = ref["T"][rix]
                    tt = t.clamp(max=ref["pc"].shape[1] - 1)
                    inref = t < Tr
                    pc_bad = ticked & (~inref | (rec["pc"] != ref["pc"][rix, tt]))
                    acc = (U[:, F["mload"]] | U[:, F["mstore"]]) > 0
                    ea_bad = ticked & inref & acc & ref["acc"][rix, tt] & (rec["ea"] != ref["ea"][rix, tt])
                    ctrl_at = torch.where((ctrl_at < 0) & pc_bad, t, ctrl_at)
                    addr_at = torch.where((addr_at < 0) & ea_bad, t, addr_at)
                    if m._moved is not None:
                        moved_at = torch.where((moved_at < 0) & m._moved & ticked, t, moved_at)
                    if self.hold is not None:
                        nt = m.steps.clamp(max=ref["pc"].shape[1] - 1)
                        force = self.hold & retired & (m.steps < Tr)
                        want = ref["pc"][rix, nt]
                        moved_at = torch.where((moved_at < 0) & force & (m.pc != want), t, moved_at)
                        m.pc.copy_(torch.where(force, want, m.pc))
                if record != "none":
                    pcs.append(torch.where(ticked, rec["pc"], -1))
                if record == "slots":
                    slots.append(self._slots(U, row, retired, regs_before, rec, fdef, shift_row, k8))
                if record == "full":
                    full["regs"].append(m.regs.clone())
                    full["flags"].append(m.flags.clone())
                    full["ea"].append(rec["ea"])
                    full["store_n"].append(rec["store_n"])
                    full["new"].append(rec["new"].long())
                    full["old"].append(rec["old"])
                    full["rsi"].append(rsi_before)
        T = m.steps + exit_tick.long()
        if self.ref is not None:
            Tr = self.ref["T"][self.ref["rix"]]
            ctrl_at = torch.where((ctrl_at < 0) & (T != Tr), torch.minimum(T, Tr), ctrl_at)
        ys = []
        out = m.out.cpu()
        for lane, (s, ln) in enumerate(zip(last_start.tolist(), m.out_len.tolist())):
            ys.append(bytes(out[lane, s:ln].numpy().tobytes()))
        res = Result(m.stop.clone(), T, m.exit_code.clone(), ys)
        if self.ref is not None:
            res.ctrl_at, res.addr_at, res.moved_at = ctrl_at, addr_at, moved_at
        if record != "none":
            res.pc = torch.stack(pcs, 1) if pcs else torch.zeros(B, 0, dtype=torch.int64, device=dev)
        if record == "slots":
            res.slots = torch.stack(slots, 1) if slots else torch.zeros(B, 0, NSLOT_LANE, dtype=torch.int16)
        if record == "full":
            res.full = {k: torch.stack(v, 1) for k, v in full.items()}
        return res

    def _slots(self, U, row, retired, regs_before, rec, fdef, shift_row, k8):
        """Written value per lane-slot this tick (int16, -1 = slot not written)."""
        F, m = self.F, self.m
        B = self.B
        out = torch.full((B, NSLOT_LANE), -1, dtype=torch.int16, device=self.dev)
        wd, dsh = U[:, F["wd"]], U[:, F["dsh"]]
        lo = dsh // 8
        regmask = torch.where((wd >= 4)[:, None], True, (k8 >= lo[:, None]) & (k8 < (lo + wd)[:, None]))

        def regbytes(reg):
            v = m.regs.gather(1, reg.clamp(0, 15)[:, None])[:, 0]
            return ((v[:, None] >> (8 * k8)) & 0xFF).to(torch.int16)

        dst = (U[:, F["dk"]] == 1) & retired
        out[:, S_DST:S_DST + 8] = torch.where(dst[:, None] & regmask, regbytes(U[:, F["dreg"]]), out[:, S_DST:S_DST + 8])
        d2 = (U[:, F["d2"]] >= 0) & retired
        d2mask = torch.where((wd >= 4)[:, None], True, k8 < wd[:, None])
        out[:, S_D2:S_D2 + 8] = torch.where(d2[:, None] & d2mask, regbytes(torch.full_like(wd, RDX)), out[:, S_D2:S_D2 + 8])
        sp = (U[:, F["spk"]] == 1) & retired
        out[:, S_SP:S_SP + 8] = torch.where(sp[:, None], regbytes(torch.full_like(wd, RSP)), out[:, S_SP:S_SP + 8])
        st = rec["store_n"]
        out[:, S_MEM:S_MEM + 8] = torch.where(retired[:, None] & (k8 < st[:, None]), rec["new"].to(torch.int16),
                                             out[:, S_MEM:S_MEM + 8])
        fbits = fdef[row]
        # a shift or rotate by zero writes no flags
        cnt = torch.where(U[:, F["bk"]] == 1, regs_before[:, RCX], U[:, F["imm"]]) & torch.where(wd == 8, 63, 31)
        fbits = torch.where(shift_row[row] & (cnt == 0), 0, fbits)
        fw = ((fbits[:, None] >> torch.arange(6, device=self.dev)) & 1).bool() & retired[:, None]
        out[:, S_FL:S_FL + 6] = torch.where(fw, m.flags.to(torch.int16), out[:, S_FL:S_FL + 6])
        return out


def reference(binary: Binary, res: Result, device=None) -> dict:
    """Reference arrays for ``hold_to`` / ``compare_to`` from runs recorded with ``record="full"``."""
    dev = device or res.pc.device
    F = binary.F
    rows = torch.tensor(binary.rows, device=dev)
    pcs = res.pc.to(dev)
    sorted_pcs = torch.tensor(binary.pcs, dtype=torch.int64, device=dev)
    j = torch.searchsorted(sorted_pcs, pcs).clamp(max=len(binary.pcs) - 1)
    r = torch.where(sorted_pcs[j] == pcs, j + 1, 0)
    acc = ((rows[r, F["mload"]] | rows[r, F["mstore"]]) > 0) & (pcs >= 0)
    return {"pc": pcs, "ea": res.full["ea"].to(dev), "acc": acc, "rsi": res.full["rsi"].to(dev),
            "T": res.T.to(dev)}
