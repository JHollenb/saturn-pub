"""Address grammar for binaries (proposal §2), a sibling of mdb's decoder grammar (``core.address``).

::

    [origin:] TICK . SITE
    TICK := i<t>                      state after tick t (instruction index, from 0)
          | init                      the initial state (after input injection)
          | <pc>#<k>                  after the k-th (0-based) execution of pc
          | <pc>@<calls>/L<head>:<i>  after the instance with this execution index
                                      (``index.text``: call sites, then loop head:iteration)
    SITE := <reg>[[lo:hi]]            GPR bit range (rax, eax, al, ah, r8d, ...); default the whole register
          | flags.<F>                 CF PF AF ZF SF OF
          | mem[<expr>:<n>]           n bytes at expr; expr = int | reg(+|-)disp, evaluated at the tick

``origin`` names another run (a fork, a donor) or ``bin`` for the current binary. Examples:
``i1042.rax[0:8]``, ``i1042.flags.CF``, ``i1042.mem[0x601040:8]``, ``0x401020#3.mem[rbp-0x10:8]``,
``donor:0x40103e@0x401007/L0x401028:2.rdx``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .binary import FLAG0, FLAGS, GPRS

_SUB = {}
for _i, _r in enumerate(GPRS):
    _SUB[_r] = (_i, 0, 64)
    if _i < 8:
        _SUB["e" + _r[1:]] = (_i, 0, 32)
        _SUB[_r[1:]] = (_i, 0, 16)
        _SUB[{"rax": "al", "rcx": "cl", "rdx": "dl", "rbx": "bl", "rsp": "spl", "rbp": "bpl", "rsi": "sil",
              "rdi": "dil"}[_r]] = (_i, 0, 8)
        if _i < 4:
            _SUB[_r[1] + "h"] = (_i, 8, 16)
    else:
        _SUB.update({_r + "d": (_i, 0, 32), _r + "w": (_i, 0, 16), _r + "b": (_i, 0, 8)})
REGS = _SUB

_ORIGIN = re.compile(r"^([A-Za-z_][\w\-]*):")
_TICK = re.compile(r"^(?:i(?P<t>\d+)|(?P<init>init)|(?P<pc>0x[0-9a-fA-F]+)(?:#(?P<k>\d+)|@(?P<ctx>[0-9a-fA-FxL:/]+)))\.")
_REG = re.compile(r"^(?P<reg>[a-z][a-z0-9]*)(?:\[(?P<lo>\d+):(?P<hi>\d+)\])?$")
_MEM = re.compile(r"^mem\[(?P<expr>[^:\]]+):(?P<n>\d+)\]$")
_EXPR = re.compile(r"^(?:(?P<reg>[a-z][a-z0-9]*)(?:(?P<sign>[+-])(?P<d>0x[0-9a-fA-F]+|\d+))?|(?P<abs>0x[0-9a-fA-F]+|\d+))$")


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Addr:
    origin: str | None
    tick: tuple          # ("step", t) | ("init",) | ("inst", pc, k) | ("exec", pc, calls, loops)
    site: tuple          # ("reg", r, lo, hi) | ("flag", f) | ("mem", expr, n); expr = (reg|None, disp)

    def text(self) -> str:
        kind = self.tick[0]
        t = {"step": lambda: f"i{self.tick[1]}", "init": lambda: "init",
             "inst": lambda: f"{self.tick[1]:#x}#{self.tick[2]}"}.get(kind)
        tick = t() if t else _exec_text(self.tick)
        s = self.site
        if s[0] == "reg":
            site = f"{GPRS[s[1]]}[{s[2]}:{s[3]}]"
        elif s[0] == "flag":
            site = f"flags.{FLAGS[s[1]]}"
        else:
            reg, d = s[1]
            e = f"{reg}{'-' if d < 0 else '+'}{abs(d):#x}" if reg else f"{d:#x}"
            site = f"mem[{e}:{s[2]}]"
        return (f"{self.origin}:" if self.origin else "") + f"{tick}.{site}"


def _exec_text(tick) -> str:
    _, pc, calls, loops = tick
    parts = [f"{c:#x}" for c in calls] + [f"L{h:#x}:{i}" for h, i in loops]
    return f"{pc:#x}@" + "/".join(parts)


def parse(text: str) -> Addr:
    s = text.strip()
    origin = None
    m = _ORIGIN.match(s)
    if m:
        origin, s = m.group(1), s[m.end():]
    m = _TICK.match(s)
    if not m:
        raise AddressError(f"{text!r}: expected i<t>., init., <pc>#<k>. or <pc>@<index>. before the site")
    if m.group("t") is not None:
        tick = ("step", int(m.group("t")))
    elif m.group("init"):
        tick = ("init",)
    elif m.group("k") is not None:
        tick = ("inst", int(m.group("pc"), 16), int(m.group("k")))
    else:
        calls, loops = [], []
        for part in m.group("ctx").split("/"):
            if part.startswith("L"):
                h, _, i = part[1:].partition(":")
                loops.append((int(h, 16), int(i)))
            elif part:
                calls.append(int(part, 16))
        tick = ("exec", int(m.group("pc"), 16), tuple(calls), tuple(loops))
    return Addr(origin, tick, parse_site(s[m.end():], text))


def parse_site(s: str, text: str | None = None) -> tuple:
    text = text or s
    if s.startswith("flags."):
        f = s[6:]
        if f not in FLAGS:
            raise AddressError(f"{text!r}: flag must be one of {FLAGS}")
        return ("flag", FLAGS.index(f))
    m = _MEM.match(s)
    if m:
        e = _EXPR.match(m.group("expr").replace(" ", ""))
        if not e:
            raise AddressError(f"{text!r}: memory expression is int or reg+-disp")
        if e.group("abs"):
            expr = (None, int(e.group("abs"), 0))
        else:
            if e.group("reg") not in GPRS:
                raise AddressError(f"{text!r}: base register must be a 64-bit GPR")
            d = int(e.group("d"), 0) if e.group("d") else 0
            expr = (e.group("reg"), -d if e.group("sign") == "-" else d)
        return ("mem", expr, int(m.group("n")))
    m = _REG.match(s)
    if m and m.group("reg") in REGS:
        r, lo, hi = REGS[m.group("reg")]
        if m.group("lo") is not None:
            a, b = int(m.group("lo")), int(m.group("hi"))
            if not 0 <= a < b <= hi - lo:
                raise AddressError(f"{text!r}: bit range outside the register")
            lo, hi = lo + a, lo + b
        return ("reg", r, lo, hi)
    raise AddressError(f"{text!r}: unknown site {s!r}")


def resolve_tick(addr: Addr, trace) -> int:
    """The tick index of ``addr`` in ``trace`` (-1 for ``init``)."""
    kind = addr.tick[0]
    if kind == "init":
        return -1
    if kind == "step":
        t = addr.tick[1]
        if not 0 <= t < trace.T:
            raise AddressError(f"tick {t} outside the run (T={trace.T})")
        return t
    for tk in trace.ticks:
        if kind == "inst" and tk.pc == addr.tick[1] and tk.k == addr.tick[2]:
            return tk.t
        if kind == "exec" and tk.index == (addr.tick[2], addr.tick[3], addr.tick[1]):
            return tk.t
    raise AddressError(f"{addr.text()}: no such instance in this run")


def sites(addr: Addr, trace, t: int) -> tuple[list[int], int, int]:
    """(byte/flag sites, low bit inside the first site, number of bits) of addr's site at tick t."""
    s = addr.site
    if s[0] == "flag":
        return [FLAG0 + s[1]], 0, 1
    if s[0] == "reg":
        r, lo, hi = s[1:]
        return [8 * r + b for b in range(lo // 8, (hi + 7) // 8)], lo % 8, hi - lo
    reg, d = s[1]
    base = trace.reg(t, GPRS.index(reg)) if reg else 0
    a = (base + d) & ((1 << 64) - 1)
    return list(range(a, a + s[2])), 0, 8 * s[2]


__all__ = ["Addr", "AddressError", "parse", "parse_site", "resolve_tick", "sites", "REGS"]
