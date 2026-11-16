"""Static facts of one x86-64 binary: decoded micro-op table, CFG, dominator loops, sites, operands.

The executor is torchx86 (isa-bench ``models/torchx86``), imported by path, never copied:
``MDB_TORCHX86_SRC`` (its ``src`` directory) and ``MDB_ISA_BENCH_SRC`` (isa-bench ``src``, for
``isa_bench.state``) are put on ``sys.path`` when the modules are not importable already.

Sites are integers, one per byte (the binctr web-key convention): register byte ``8*gpr + byte``
(0..127), flag ``128 + f`` (CF PF AF ZF SF OF), memory byte = its address (>= 0x1000).
Write slots are the fixed per-tick positions a write can land in (``SLOTS``): the destination
register's 8 bytes, the second destination (RDX of mul/div), RSP, the stored memory bytes, the six
flags, and the three registers a ``write`` syscall returns in.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GPRS = ("rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
FLAGS = ("CF", "PF", "AF", "ZF", "SF", "OF")
RAX, RCX, RDX, RSP, RBP, RSI, RDI, R11 = 0, 1, 2, 4, 5, 6, 7, 11
FLAG0 = 128
# write slots: 0-7 dst reg bytes, 8-15 d2 (RDX), 16-23 RSP, 24-31 stored bytes, 32-37 flags,
# 38-61 syscall returns (RAX, RCX, R11); lanes record slots 0..37 only
S_DST, S_D2, S_SP, S_MEM, S_FL, S_SYS = 0, 8, 16, 24, 32, 38
NSLOT_LANE, NSLOT = 38, 62
SLOT_ROLE = (["dst"] * 8 + ["d2"] * 8 + ["sp"] * 8 + ["mem"] * 8 + [f"flags.{f}" for f in FLAGS]
             + ["sys.rax"] * 8 + ["sys.rcx"] * 8 + ["sys.r11"] * 8)
KINDS = ("op", "call", "ret", "jmp", "jcc", "syscall", "stop")
# condition code -> flags it reads (torchx86 cc numbering: o no b ae e ne be a s ns p np l ge le g)
CC_FLAGS = {0: "OF", 1: "OF", 2: "CF", 3: "CF", 4: "ZF", 5: "ZF", 6: "CF ZF", 7: "CF ZF", 8: "SF", 9: "SF",
            10: "PF", 11: "PF", 12: "SF OF", 13: "SF OF", 14: "ZF SF OF", 15: "ZF SF OF"}


def _import_paths() -> None:
    for env in ("MDB_TORCHX86_SRC", "MDB_ISA_BENCH_SRC"):
        p = os.environ.get(env)
        if p and p not in sys.path:
            sys.path.insert(0, p)


def torchx86():
    """The torchx86 package (by path, see the module docstring)."""
    _import_paths()
    try:
        import torchx86 as tx  # noqa: PLC0415
        from torchx86 import decode, machine  # noqa: F401, PLC0415
    except ImportError as why:
        raise ImportError("mdb.machines.x86 needs torchx86 (isa-bench models/torchx86/src) and capstone: "
                          "set MDB_TORCHX86_SRC and MDB_ISA_BENCH_SRC") from why
    return tx


def site_name(site: int) -> str:
    if site < FLAG0:
        return f"{GPRS[site // 8]}[{8 * (site % 8)}:{8 * (site % 8) + 8}]"
    if site < FLAG0 + 6:
        return f"flags.{FLAGS[site - FLAG0]}"
    return f"mem[{site:#x}:1]"


def site_bits(site: int) -> int:
    return 1 if FLAG0 <= site < FLAG0 + 6 else 8


@dataclass
class Loop:
    id: int
    head: int
    body: frozenset
    function: int


@dataclass
class Binary:
    name: str
    table: object                      # torchx86 Table
    pages: dict
    perms: dict
    regs: list                         # initial GPRs (uint64 values)
    eflags: int
    entry: int
    input_addr: int | None = None
    input_size: int = 0
    image_sha256: str = ""
    # derived in __post_init__
    rows: np.ndarray = field(init=False, repr=False)
    pcs: list = field(init=False, repr=False)
    kind: list = field(init=False, repr=False)
    loops: list = field(init=False, repr=False)
    functions: dict = field(init=False, repr=False)

    def __post_init__(self):
        from torchx86.decode import FIELDS  # noqa: PLC0415

        self.F = {n: i for i, n in enumerate(FIELDS)}
        self.rows = self.table.rows.cpu().numpy()
        self.pcs = [int(p) for p in self.table.pcs.tolist()]
        self.row_index = {pc: i + 1 for i, pc in enumerate(self.pcs)}
        self.kind = [self._kind(r) for r in range(len(self.rows))]
        self._flag_tables()
        self._cfg()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, program_dir, *, input_symbol: str = "gt_input", name: str | None = None) -> Binary:
        """A program directory of the isa-bench reference store: ``mem.bin`` (image), ``trace.npz``
        (native row 0 = initial state) and optionally ``prog.elf`` (symbol table only, for the input
        buffer address and size; debug sections are never read)."""
        torchx86()
        from isa_bench.state import load_image  # noqa: PLC0415
        from torchx86.decode import decode  # noqa: PLC0415

        d = Path(program_dir)
        pages, perms = load_image(d / "mem.bin")
        row = [int(v) for v in np.load(d / "trace.npz")["steps"][0]]
        addr, size = None, 0
        if (d / "prog.elf").exists():
            addr, size = elf_symbol(d / "prog.elf", input_symbol)
        table = decode(pages, perms, row[0])
        sha = hashlib.sha256((d / "mem.bin").read_bytes()).hexdigest()
        return cls(name or d.name, table, pages, perms, row[1:17], row[17] & ~0x100, row[0], addr, size, sha)

    @classmethod
    def from_image(cls, pages: dict, perms: dict, *, regs, eflags: int, entry: int, input_addr=None,
                   input_size: int = 0, name: str = "image") -> Binary:
        torchx86()
        from torchx86.decode import decode  # noqa: PLC0415

        sha = hashlib.sha256(b"".join(bytes(pages[p]) for p in sorted(pages))).hexdigest()
        return cls(name, decode(pages, perms, entry), pages, perms, list(regs), eflags, entry, input_addr,
                   input_size, sha)

    # ------------------------------------------------------------------ rows
    def g(self, r: int, f: str) -> int:
        return int(self.rows[r, self.F[f]])

    def row_of(self, pc: int) -> int:
        return self.row_index.get(int(pc), 0)

    def text(self, pc: int) -> str:
        return self.table.text[self.row_of(pc)]

    def mnemonic(self, pc: int) -> str:
        return self.table.mnemonic[self.row_of(pc)]

    def _kind(self, r: int) -> str:
        if r == 0 or self.g(r, "stop"):
            return "stop"
        mn = self.table.mnemonic[r]
        if mn in ("call", "ret", "jmp", "syscall"):
            return mn
        return "jcc" if mn.startswith("j") else "op"

    def _flag_tables(self) -> None:
        """Per row: flag bits written (fmask), and the subset that is architecturally defined."""
        from isa_bench.state import flag_effects  # noqa: PLC0415

        R = len(self.rows)
        self.fwrite = np.zeros(R, np.int64)
        self.fdef = np.zeros(R, np.int64)
        for r in range(1, R):
            fm = self.g(r, "fmask")
            eff = flag_effects(self.table.mnemonic[r])
            defined = set(FLAGS) if eff is None else set(eff[0])
            self.fwrite[r] = fm
            self.fdef[r] = fm & sum(1 << i for i, f in enumerate(FLAGS) if f in defined)
        from torchx86.decode import CLASSES  # noqa: PLC0415

        cls = self.rows[:, self.F["cls"]]
        self.shift_row = (cls >= CLASSES.index("shl")) & (cls <= CLASSES.index("ror")) & (self.rows[:, self.F["fmask"]] != 0)

    # ------------------------------------------------------------------ CFG, dominators, loops
    def successors(self, pc: int) -> list[int]:
        r = self.row_of(pc)
        k = self.kind[r]
        nxt, tgt, pck = self.g(r, "next"), self.g(r, "target"), self.g(r, "pck")
        if k in ("stop", "ret") or (k == "jmp" and pck == 2):
            return []
        if k == "jmp":
            return [tgt]
        if k == "jcc":
            return [nxt, tgt]
        return [nxt]                    # op, syscall, call (intraprocedural fallthrough)

    def _cfg(self) -> None:
        entries = {self.entry} | {self.g(r, "target") for r in range(1, len(self.rows)) if self.kind[r] == "call"}
        self.functions, self.loops = {}, []
        heads: dict[int, set] = {}
        func_of_head = {}
        for e in sorted(entries):
            if self.row_of(e) == 0:
                continue
            nodes, work = set(), [e]
            while work:
                n = work.pop()
                if n in nodes or self.row_of(n) == 0:
                    continue
                nodes.add(n)
                work.extend(self.successors(n))
            self.functions[e] = nodes
            preds = {n: [] for n in nodes}
            for n in nodes:
                for s in self.successors(n):
                    if s in nodes:
                        preds[s].append(n)
            dom = {n: set(nodes) for n in nodes}
            dom[e] = {e}
            changed = True
            while changed:
                changed = False
                for n in sorted(nodes):
                    if n == e:
                        continue
                    ps = [dom[p] for p in preds[n]]
                    new = (set.intersection(*ps) if ps else set()) | {n}
                    if new != dom[n]:
                        dom[n], changed = new, True
            for n in nodes:
                for h in self.successors(n):
                    if h in nodes and h in dom[n]:          # back edge n -> h
                        body, work = {h}, [n]
                        while work:
                            x = work.pop()
                            if x not in body:
                                body.add(x)
                                work.extend(preds[x])
                        heads.setdefault(h, set()).update(body)
                        func_of_head[h] = e
        R = len(self.rows)
        self.head_row = np.full(R, -1, np.int64)
        for i, h in enumerate(sorted(heads)):
            self.loops.append(Loop(i, h, frozenset(heads[h]), func_of_head[h]))
            self.head_row[self.row_of(h)] = i
        self.inbody = np.zeros((max(len(self.loops), 1), R), bool)
        for lp in self.loops:
            for pc in lp.body:
                self.inbody[lp.id, self.row_of(pc)] = True
        self.leaders = {self.entry} | set(entries)
        for r in range(1, R):
            if self.kind[r] in ("jcc", "jmp", "call", "ret", "syscall"):
                self.leaders.add(self.g(r, "next"))
                if self.g(r, "pck") == 1:
                    self.leaders.add(self.g(r, "target"))

    # ------------------------------------------------------------------ operands
    def operand_text(self, pc: int, slot: int) -> str:
        """The written operand of a slot at pc, in the address grammar (``rax[0:8]``, ``mem[rbp-0x8:8]``)."""
        role = SLOT_ROLE[slot]
        k = slot % 8
        if role == "dst":
            r = self.row_of(pc)
            if self.g(r, "dk") == 2:
                return f"mem[{self.mem_operand(pc)}]"
            return f"{GPRS[self.g(r, 'dreg')]}[{8 * k}:{8 * k + 8}]"
        if role == "d2":
            return f"rdx[{8 * k}:{8 * k + 8}]"
        if role == "sp":
            return f"rsp[{8 * k}:{8 * k + 8}]"
        if role == "mem":
            return f"mem[{self.mem_operand(pc)}+{k}:1]" if k else f"mem[{self.mem_operand(pc)}:1]"
        if role.startswith("flags."):
            return role
        reg = role.split(".")[1]
        return f"{reg}[{8 * k}:{8 * k + 8}]"

    def mem_operand(self, pc: int) -> str:
        r = self.row_of(pc)
        b, i, s, d = self.g(r, "mbase"), self.g(r, "midx"), self.g(r, "mscale"), self.g(r, "mdisp")
        parts = ([GPRS[b]] if b >= 0 else []) + ([f"{GPRS[i]}*{s}"] if i >= 0 else [])
        txt = "+".join(parts)
        if d or not parts:
            txt = f"{txt}{'-' if d < 0 else '+'}{abs(d):#x}" if parts else f"{d & ((1 << 64) - 1):#x}"
        return txt


def elf_symbol(path, name: str) -> tuple[int | None, int]:
    """(address, size) of a symbol from the ELF ``.symtab`` (no debug section is read)."""
    try:
        from elftools.elf.elffile import ELFFile  # noqa: PLC0415
    except ImportError:
        return None, 0
    with open(path, "rb") as fh:
        sec = ELFFile(fh).get_section_by_name(".symtab")
        for sym in (sec.get_symbol_by_name(name) or []) if sec is not None else []:
            return int(sym["st_value"]), int(sym["st_size"])
    return None, 0
