"""One recorded run of a binary: ticks, typed read/write sets, cells, execution indices.

Tick ``t`` executes ``pc[t]``; ``regs[t]``/``flags[t]`` are the state after it (binctr's convention;
the final ``exit`` tick carries the state before it). ``t = -1`` is the initial state.

A **cell** is (site, [def, kill)): a maximal interval in which a site byte (or flag bit) is not
rewritten (advisor memo §1.1). Cells come from write events; the initial state contributes cells
(``def = -1``) for the input buffer and for every site read before it is written. A write of an
architecturally undefined flag ends the flag's cell and starts none (PREREG: undefined flags are
excluded from every cell). Each cell carries its **alignment key**: (execution-index key of the
defining tick, write slot), or (``init``, site) for initial cells, the name by which the same cell
is found in another run.

Read edge types follow the advisor memo §1.1, one rule per operand: base/index registers of a
memory access are ``addr`` (``lea``: ``data``); flags tested by jcc/cmov are ``ctrl``; the bytes a
``ret`` or ``jmp [m]`` reads and a ``jmp reg`` target are ``ctrl``; implicit RSP of push/pop/call/
ret (RBP of leave) is ``addr`` and ``data``; the ``write`` syscall reads RAX ``ctrl``, RDI ``data``,
RSI/RDX ``addr`` and each output byte ``data``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import index as xi
from .binary import (FLAG0, FLAGS, NSLOT, RAX, RBP, RCX, RDI, RDX, RSI, RSP, R11, S_D2, S_DST, S_FL, S_MEM,
                     S_SP, S_SYS, SLOT_ROLE, Binary, site_bits, site_name)
from .lanes import Lanes, Result

DATA, ADDR, CTRL = "data", "addr", "ctrl"
INIT = -1
COPY_MNEMONICS = {"mov", "movabs", "movzx", "movsx", "movsxd", "push", "pop", "leave", "cdqe", "cwde", "cbw"}


@dataclass
class Cell:
    id: int
    site: int
    def_: int                 # defining tick, -1 = initial state
    kill: int                 # first tick that rewrites the site (T if never)
    slot: int                 # write slot (binary.SLOT_ROLE), -1 for initial cells
    pc: int                   # defining pc, -1 for initial cells
    value: int
    key: tuple                # alignment key: (exec key, slot) or ("init", site)
    readers: list = field(default_factory=list)   # [(tick, edge)]

    @property
    def bits(self) -> int:
        return site_bits(self.site)

    @property
    def weight(self) -> int:
        return self.bits * (self.kill - self.def_)


@dataclass
class Tick:
    t: int
    pc: int
    kind: str
    depth: int                # call depth after the tick
    text: str
    k: int                    # this is the k-th execution of pc
    key: int                  # execution-index key
    index: tuple              # readable execution index
    reads: list               # [(site, edge)]
    writes: list              # [(site, slot)]
    next_pc: int = -1


def _reg_sites(reg: int, lo: int, n: int) -> list[int]:
    return [8 * reg + b for b in range(lo, min(lo + n, 8))]


class Trace:
    def __init__(self, binary: Binary, res: Result, lane: int, input_bytes: bytes | None, *, key_row=None):
        self.b = b = binary
        self.input = input_bytes
        T = int(res.T[lane])
        f = {k: v[lane, :T].cpu().numpy() for k, v in res.full.items()}
        self.T, self.stop, self.exit_code, self.y = T, res.stop_kind(lane), int(res.exit_code[lane]), res.y[lane]
        self.pc = res.pc[lane, :T].cpu().numpy().astype(np.int64)
        self.regs, self.flags = f["regs"].astype(np.int64), f["flags"].astype(np.int64)
        self.ea, self.store_n, self.new = f["ea"], f["store_n"], f["new"]
        self.regs0 = np.array([int(v) - (1 << 64) if int(v) >> 63 else int(v) for v in b.regs], np.int64)
        self.flags0 = np.array([(b.eflags >> bit) & 1 for bit in (0, 2, 4, 6, 7, 11)], np.int64)
        if key_row is None:
            key_row = xi.keys(b, torch.tensor(self.pc)[None], torch.ones(1, T, dtype=torch.bool))[0]
        self.keys = [int(k) for k in key_row[:T].tolist()]
        self._build()

    # ------------------------------------------------------------------ state
    def reg(self, t: int, r: int) -> int:
        v = int(self.regs0[r] if t < 0 else self.regs[t, r])
        return v & ((1 << 64) - 1)

    def flag(self, t: int, f: int) -> int:
        return int(self.flags0[f] if t < 0 else self.flags[t, f])

    def mem(self, t: int, addr: int, n: int = 1) -> bytes:
        """Memory after tick t (t = -1: the initial image after input injection)."""
        out = bytearray()
        for a in range(addr, addr + n):
            page = self.b.pages.get(a & ~0xFFF)
            v = page[a & 0xFFF] if page is not None else None
            if self.input is not None and self.b.input_addr is not None and 0 <= a - self.b.input_addr < len(self.input):
                v = self.input[a - self.b.input_addr]
            for (tt, aa, _old, new) in self._writes_by_addr.get(a, ()):
                if tt <= t:
                    v = new
            if v is None:
                raise ValueError(f"address {a:#x} is unmapped")
            out.append(int(v))
        return bytes(out)

    def value(self, site: int, t: int) -> int:
        if site < FLAG0:
            return (self.reg(t, site // 8) >> (8 * (site % 8))) & 0xFF
        if site < FLAG0 + 6:
            return self.flag(t, site - FLAG0)
        try:
            return self.mem(t, site)[0]
        except ValueError:
            return -1                                     # unmapped (a faulting access)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        b = self.b
        self.ticks: list[Tick] = []
        self.cells: list[Cell] = []
        self.cur: dict[int, int] = {}
        self.by_key: dict[tuple, int] = {}
        self.mem_writes: list[tuple] = []
        self._writes_by_addr: dict[int, list] = {}
        undefined: set[int] = set()
        if self.input is not None and b.input_addr is not None:
            for i in range(len(self.input)):
                self._new_cell(b.input_addr + i, INIT, -1, -1, ("init", b.input_addr + i), self.input[i])
        readable = xi.readable(b, self.pc)
        inst = xi.instances(self.pc)
        depth = 0
        for t in range(self.T):
            pc = int(self.pc[t])
            r = b.row_of(pc)
            kind = b.kind[r]
            reads, writes = self._accesses(t, r, kind, undefined)
            for site, edge in reads:
                if site not in self.cur:
                    self._new_cell(site, INIT, -1, -1, ("init", site), self.value(site, -1))
                self.cells[self.cur[site]].readers.append((t, edge))
            for site, slot in writes:
                if site in self.cur:
                    self.cells[self.cur.pop(site)].kill = t
                if slot is None:                              # undefined flag: ends the cell, starts none
                    continue
                if S_MEM <= slot < S_MEM + 8:
                    w = (t, site, int(self.mem(t - 1, site)[0]), int(self.new[t, slot - S_MEM]))
                    self.mem_writes.append(w)
                    self._writes_by_addr.setdefault(site, []).append(w)
                self._new_cell(site, t, slot, pc, (self.keys[t], slot), self._written(t, site, slot))
            depth += 1 if kind == "call" else -1 if kind == "ret" else 0
            self.ticks.append(Tick(t, pc, kind, depth, b.table.text[r], inst[t], self.keys[t], readable[t],
                                   reads, [(s, sl) for s, sl in writes if sl is not None]))
        for c in self.cells:
            if c.kill < 0:
                c.kill = self.T
        for t, tk in enumerate(self.ticks):
            tk.next_pc = self.ticks[t + 1].pc if t + 1 < self.T else -1
        dup = len(self.by_key) != len(self.cells)
        self.duplicate_keys = dup

    def _new_cell(self, site, t, slot, pc, key, value) -> None:
        c = Cell(len(self.cells), site, t, -1, slot, pc, int(value), key)
        self.cells.append(c)
        self.cur[site] = c.id
        self.by_key[key] = c.id

    def _written(self, t: int, site: int, slot: int) -> int:
        if S_MEM <= slot < S_MEM + 8:
            return int(self.new[t, slot - S_MEM])
        return self.value(site, t)

    def _accesses(self, t: int, r: int, kind: str, undefined: set) -> tuple[list, list]:
        """Typed read set and write set (site, slot; slot None = undefined flag) of tick t."""
        b = self.b
        g = lambda f: int(b.rows[r, b.F[f]])
        rb = lambda reg: self.reg(t - 1, reg)
        reads, writes = [], []
        if kind == "stop":
            return reads, writes
        if kind == "syscall":
            nr = rb(RAX)
            reads.append((8 * RAX, CTRL))
            if nr == 1:
                reads += [(8 * RDI, DATA), *[(s, ADDR) for s in _reg_sites(RSI, 0, 8)],
                          *[(s, ADDR) for s in _reg_sites(RDX, 0, 8)]]
                reads += [(a, DATA) for a in range(rb(RSI), rb(RSI) + rb(RDX))]
                for slot0, reg in ((S_SYS, RAX), (S_SYS + 8, RCX), (S_SYS + 16, R11)):
                    writes += [(8 * reg + k, slot0 + k) for k in range(8)]
            elif nr in (60, 231):
                reads.append((8 * RDI, DATA))
            return reads, writes
        cls_lea = b.table.mnemonic[r] == "lea"
        ea = int(self.ea[t]) & ((1 << 64) - 1)
        access = (g("mload") or g("mstore")) and not cls_lea
        for reg in (g("mbase"), g("midx")):
            if reg >= 0 and (access or cls_lea):
                reads += [(s, DATA if cls_lea else ADDR) for s in _reg_sites(reg, 0, 8)]
        wd, ws = g("wd"), g("ws")
        pck2 = g("pck") == 2
        if g("ak") == 1:
            reads += [(s, DATA) for s in _reg_sites(g("areg"), g("ash") // 8, wd)]
        elif g("ak") == 2:
            reads += [(a, DATA) for a in range(ea, ea + wd)]
        if g("bk") == 1:
            reads += [(s, CTRL if pck2 else DATA) for s in _reg_sites(g("breg"), g("bsh") // 8, ws)]
        elif g("bk") == 2:
            reads += [(a, CTRL if pck2 else DATA) for a in range(ea, ea + ws)]
        if g("spk"):
            reads += [(s, DATA) for s in _reg_sites(g("spbase"), 0, 8)]
        mn = b.table.mnemonic[r]
        if mn in ("div", "idiv"):
            reads += [(s, DATA) for s in _reg_sites(RDX, 0, wd)]
        if mn == "sbb":
            reads.append((FLAG0, DATA))
        if g("cc") < 16:
            edge = DATA if mn.startswith("set") else CTRL
            reads += [(FLAG0 + FLAGS.index(f), edge) for f in xi_cc(g("cc"))]
        reads = [(s, e) for s, e in reads if not (FLAG0 <= s < FLAG0 + 6 and (s - FLAG0) in undefined)]
        # writes
        if g("dk") == 1:
            lo = g("dsh") // 8
            ks = range(8) if wd >= 4 else range(lo, lo + wd)
            writes += [(8 * g("dreg") + k, S_DST + k) for k in ks]
        elif g("dk") == 2:
            writes += [(ea + k, S_MEM + k) for k in range(int(self.store_n[t]))]
        if g("d2") >= 0:
            writes += [(8 * RDX + k, S_D2 + k) for k in (range(8) if wd >= 4 else range(wd))]
        if g("spk"):
            writes += [(8 * RSP + k, S_SP + k) for k in range(8)]
        fw, fd = int(b.fwrite[r]), int(b.fdef[r])
        if b.shift_row[r]:
            c = (rb(RCX) & 0xFF if g("bk") == 1 else g("imm")) & (63 if wd == 8 else 31)
            if c == 0:
                fw = fd = 0
        for f in range(6):
            if fw >> f & 1:
                if fd >> f & 1:
                    writes.append((FLAG0 + f, S_FL + f))
                    undefined.discard(f)
                else:
                    writes.append((FLAG0 + f, None))
                    undefined.add(f)
        return reads, writes

    # ------------------------------------------------------------------ queries
    def cell_at(self, site: int, t: int) -> Cell | None:
        """The cell holding ``site`` after tick t (def <= t < kill)."""
        best = None
        for c in self.cells:
            if c.site == site and c.def_ <= t < c.kill:
                best = c
        return best

    def reads_of(self, t: int) -> list:
        return [(s, site_bits(s), e) for s, e in self.ticks[t].reads]

    def writes_of(self, t: int) -> list:
        return [(s, SLOT_ROLE[sl], self.value(s, t) if sl < S_MEM or sl >= S_MEM + 8 else int(self.new[t, sl - S_MEM]))
                for s, sl in self.ticks[t].writes]

    def webs(self) -> list[int]:
        """Structural copy webs (this backend's own, from the trace): cell id -> root cell id. Copy
        edges: mov/movzx/movsx/movsxd/movabs/push/pop/leave/cdqe-family, low bytes source -> dest."""
        parent = list(range(len(self.cells)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        b = self.b
        live: dict[int, int] = {c.site: c.id for c in self.cells if c.def_ == INIT}
        defs: dict[int, list] = {}
        for c in self.cells:
            if c.def_ >= 0:
                defs.setdefault(c.def_, []).append(c)
        for t, tick in enumerate(self.ticks):
            r = b.row_of(tick.pc)
            mn = b.table.mnemonic[r]
            snapshot = dict(live)
            for c in defs.get(t, ()):
                live[c.site] = c.id
            if mn not in COPY_MNEMONICS or b.kind[r] != "op":
                continue
            g = lambda f: int(b.rows[r, b.F[f]])
            ea = int(self.ea[t]) & ((1 << 64) - 1)
            if g("bk") == 1:
                src = _reg_sites(g("breg"), g("bsh") // 8, g("ws"))
            elif g("bk") == 2:
                src = list(range(ea, ea + g("ws")))
            else:
                continue
            dst = [c for c in defs.get(t, ()) if c.slot < S_FL and not (S_SP <= c.slot < S_SP + 8)]
            dst.sort(key=lambda c: c.slot)
            for k, c in enumerate(dst[:len(src)]):
                sid = snapshot.get(src[k])
                if sid is not None:
                    parent[find(c.id)] = find(sid)
        return [find(i) for i in range(len(self.cells))]

    def web_labels(self) -> list:
        """Per cell: its structural web's label = the alignment key of the web's earliest cell."""
        root = self.webs()
        return [self.cells[r].key for r in root]

    def describe(self, c: Cell) -> str:
        if c.def_ == INIT:
            return f"init.{site_name(c.site)}"
        tick = self.ticks[c.def_]
        return f"{c.pc:#x}#{tick.k}.{self.b.operand_text(c.pc, c.slot)}"


def xi_cc(cc: int) -> list[str]:
    from .binary import CC_FLAGS  # noqa: PLC0415

    return CC_FLAGS[cc].split()


def record(binary: Binary, inputs: list, *, device="cpu", cap: int = 20000, resident=None) -> tuple[list, Result]:
    """Run ``inputs`` (bytes or None for the image's own) as lanes and return one Trace per lane."""
    lanes = Lanes(binary, inputs, device=device, cap=cap, resident=resident)
    res = lanes.run(record="full")
    valid = res.pc >= 0
    keys = xi.keys(binary, res.pc, valid)
    return [Trace(binary, res, i, inp if inp is not None else default_input(binary), key_row=keys[i])
            for i, inp in enumerate(inputs)], res


def default_input(binary: Binary) -> bytes | None:
    if binary.input_addr is None or not binary.input_size:
        return None
    page = binary.pages[binary.input_addr & ~0xFFF]
    o = binary.input_addr & 0xFFF
    return bytes(page[o:o + binary.input_size])


__all__ = ["Cell", "Tick", "Trace", "record", "default_input", "NSLOT"]
