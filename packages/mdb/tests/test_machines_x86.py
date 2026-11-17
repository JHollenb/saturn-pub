"""mdb.machines.x86 on a hand-assembled binary (no reference data). Needs torch, capstone and torchx86
(isa-bench ``models/torchx86/src``, found through MDB_TORCHX86_SRC / MDB_ISA_BENCH_SRC or a sibling
``isa-bench`` checkout)."""
import os
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("capstone")
for env, rel in (("MDB_TORCHX86_SRC", "models/torchx86/src"), ("MDB_ISA_BENCH_SRC", "src")):
    if env not in os.environ:
        for root in (Path.home() / "domains/isa-bench", *Path(__file__).resolve().parents):
            cand = (root if root.name == "isa-bench" else root / "isa-bench") / rel
            if cand.exists():
                os.environ[env] = str(cand)
                break
try:
    from mdb.machines.x86 import Binary, torchx86
    torchx86()
except ImportError:
    pytest.skip("torchx86 not found", allow_module_level=True)

from mdb.machines import FAULT, PATH_DIVERGED  # noqa: E402
from mdb.machines.x86 import address as xa  # noqa: E402
from mdb.machines.x86 import index as xi  # noqa: E402
from mdb.machines.x86.grid import run_grid  # noqa: E402
from mdb.machines.x86.method import recover  # noqa: E402
from mdb.machines.x86.records import circuit_map_record, symbol_table_records, table_text  # noqa: E402
from mdb.machines.x86.session import attach  # noqa: E402

CODE, DATA, STACK = 0x401000, 0x600000, 0x7FFF0000
# a = in[0]; b = in[8]; [rsp-16] = a; 4x (a += b); call f (a += zx(in[0] byte)); if a >= b: a ^= [rsp-16]
# out[0x600010] = a; write(1, out, 8); exit(0)
PROG = bytes.fromhex(
    "488b042500006000"        # 401000 mov rax, [0x600000]
    "488b1c2508006000"        # 401008 mov rbx, [0x600008]
    "48894424f0"              # 401010 mov [rsp-0x10], rax (below the call's push)
    "b904000000"              # 401015 mov ecx, 4
    "4801d8"                  # 40101a L: add rax, rbx
    "ffc9"                    # 40101d dec ecx
    "75f9"                    # 40101f jne L
    "e834000000"              # 401021 call f (0x40105a)
    "4839d8"                  # 401026 cmp rax, rbx
    "7208"                    # 401029 jb skip
    "488b5424f0"              # 40102b mov rdx, [rsp-0x10]
    "4831d0"                  # 401030 xor rax, rdx
    "4889042510006000"        # 401033 skip: mov [0x600010], rax
    "b801000000bf01000000be10006000ba08000000"  # mov eax,1; mov edi,1; mov esi,0x600010; mov edx,8
    "0f05"                    # 40104f syscall (write)
    "b83c00000031ff0f05"      # mov eax,60; xor edi,edi; syscall
    "0fb6342500006000"        # 40105a f: movzx esi, byte [0x600000]
    "4801f0c3")               # add rax, rsi; ret


def binary():
    pages = {CODE: bytearray(PROG.ljust(4096, b"\xf4")), DATA: bytearray(4096), STACK: bytearray(4096)}
    perms = {CODE: "r-xp", DATA: "rw-p", STACK: "rw-p"}
    regs = [0] * 16
    regs[4] = STACK + 0x800
    return Binary.from_image(pages, perms, regs=regs, eflags=0x202, entry=CODE, input_addr=DATA, input_size=16,
                             name="tiny")


def inp(a, b):
    return a.to_bytes(8, "little") + b.to_bytes(8, "little")


def donors(n=12, seed=0):
    rng = random.Random(seed)
    return [inp(rng.getrandbits(62), rng.getrandbits(62)) for _ in range(n)]


@pytest.fixture(scope="module")
def tiny():
    return binary()


def expected(a, b):
    M = (1 << 64) - 1
    x = a
    for _ in range(4):
        x = (x + b) & M
    x = (x + (a & 0xFF)) & M
    if x >= b:
        x ^= a
    return x


def test_static_facts(tiny):
    assert [lp.head for lp in tiny.loops] == [0x40101A]                # dominator loop
    assert set(tiny.loops[0].body) == {0x40101A, 0x40101D, 0x40101F}
    assert tiny.kind[tiny.row_of(0x401021)] == "call"


def test_address_grammar():
    a = xa.parse("i1042.rax[0:8]")
    assert a.tick == ("step", 1042) and a.site == ("reg", 0, 0, 8)
    assert xa.parse("i3.flags.CF").site == ("flag", 0)
    assert xa.parse("i3.mem[0x601040:8]").site == ("mem", (None, 0x601040), 8)
    a = xa.parse("0x401020#3.mem[rbp-0x10:8]")
    assert a.tick == ("inst", 0x401020, 3) and a.site == ("mem", ("rbp", -16), 8)
    a = xa.parse("donor:0x40103e@0x401007/L0x401028:2.ah")
    assert a.origin == "donor" and a.tick == ("exec", 0x40103E, (0x401007,), ((0x401028, 2),))
    assert a.site == ("reg", 0, 8, 16)
    assert xa.parse(a.text()) == a
    assert xa.parse("bin:init.eax[0:8]").site == ("reg", 0, 0, 8)
    with pytest.raises(xa.AddressError):
        xa.parse("i3.xmm0")


def test_run_matches_python_and_exec_index(tiny):
    run = attach(tiny, inp(5, 3))
    tr = run.trace
    assert tr.stop == "exit" and int.from_bytes(tr.y, "little") == expected(5, 3)
    rd = xi.readable(tiny, tr.pc)
    assert [xi.key_of(tiny, r) for r in rd] == tr.keys
    loop = [tk for tk in tr.ticks if tk.pc == 0x40101D]
    assert [tk.index[1] for tk in loop] == [((0x40101A, i),) for i in range(4)]
    f = next(tk for tk in tr.ticks if tk.pc == 0x40105A)
    assert f.index[0] == (0x401021,) and f.index[1] == ()            # call string, loop left


def test_typed_reads_and_cells(tiny):
    run = attach(tiny, inp(5, 3))
    tr = run.trace
    jb = next(tk for tk in tr.ticks if tk.pc == 0x401029)
    assert jb.reads == [(128, "ctrl")]                                 # CF tested by jb
    store = next(tk for tk in tr.ticks if tk.pc == 0x401010)
    assert (8 * 4, "addr") in store.reads and (0, "data") in store.reads
    ret = next(tk for tk in tr.ticks if tk.kind == "ret")
    assert {e for s, e in ret.reads if s >= 0x1000} == {"ctrl"}
    assert not tr.duplicate_keys
    r = run.reads("0x401010#0.mem[rsp-0x10:1]")
    assert r and r[0]["reader"] == "0x40102b#0" and r[0]["edge"] == "data"


def test_navigation_domains_and_breaks(tiny):
    run = attach(tiny, inp(5, 3))
    st = run.step(1, "loop:0x40101a")
    assert st.pc == 0x40101A and st.pos == 4
    st = run.step(2, "loop")
    assert st.pc == 0x40101A and st.pos == 10
    st = run.reverse_step(1, "loop")
    assert st.pos == 7
    run.seek(16)
    st = run.next(1, "instruction")                                     # steps over the call
    assert st.pc == 0x401026 and st.pos == 20
    run.seek(17)
    assert run.pc == 0x40105A
    st = run.finish()
    assert st.pc == 0x401026
    st = run.step(1, "uop")
    assert st.uop == "flags"
    run.seek(0)
    run.break_at("0x401029")
    run.break_at("0x40101a#2")
    assert run.cont().pos == 10 and run.pc == 0x40101A
    assert run.cont().pc == 0x401029
    assert run.reverse_cont().pc == 0x40101A
    run.delete()
    run.seek(0)
    run.break_at("rcx == 2 and pc == 0x40101a")
    assert run.cont().pc == 0x40101A and run.read("rcx")["value"] == 2
    run.delete()
    run.seek(0)
    run.watch("mem[0x600010:8]")
    st = run.cont()
    assert "watch" in st.reason and run.x("mem[0x600010:8]", 1)[0] == expected(5, 3)
    assert run.read("i0.rax")["value"] == 5 and run.read("init.rsp")["value"] == STACK + 0x800


def test_set_path_diverged_hold_whatif(tiny):
    run = attach(tiny, inp(5, 3), donors=donors())
    run.seek(20)                                                         # before cmp rax, rbx
    base_y = run.trace.y
    w = run.whatif("rbx", 1 << 40)                                       # flips jb: the xor is skipped
    assert w["verdict"] == PATH_DIVERGED and w["total_effect"] == "effect"
    # rbx only feeds the compare: with the branch held, y is unchanged (the paired column)
    assert w["frozen_effect"] == "no effect" and int(w["selection_moved_under_hold_at"]) == 21
    assert run.trace.y == base_y and run.pos == 20                       # session unchanged
    w = run.whatif("rcx", 7)                                             # dead: loop is over
    assert w["verdict"] == "no effect" and w["prediction"]["effect"] == "none"
    out = run.set("rbx", 1 << 40)
    assert out["verdict"] == PATH_DIVERGED and out["path_diverged_at"] == 22
    run.checkpoint("edited")
    run.fork("alt")
    assert run.compare("main", "alt")["verdict"] == "no effect"
    run2 = attach(tiny, inp(5, 3))
    run2.seek(19)                                                        # before ret
    assert run2.set("mem[rsp:8]", 0xDEAD)["verdict"] == FAULT
    run3 = attach(tiny, inp(5, 3))
    rf = run3.forecast("0x401033#0.mem[0x600010:8]", 0x1234)
    ok = run3.forecast("0x401033#0.mem[0x600010:8]", expected(5, 3))
    assert run3.verify() == []                                           # target not reached yet
    run3.seek(run3.trace.T)
    v = run3.verify()
    assert rf["kind"] == "machine-forecast" and ok["body"]["predicted"]["hex"] == hex(expected(5, 3))
    assert [x["verdict"] for x in v] == ["not confirmed", "confirmed"]


def test_program_protocol(tiny):
    from mdb.machines import Program
    from mdb.machines.x86 import attach as top_attach

    run = top_attach(tiny, inp(5, 3))
    names = [m for m in vars(Program) if not m.startswith("_")]
    assert len(names) == 11 and all(callable(getattr(run, m)) for m in names)
    snap = run.snapshot()
    run.seek(20)
    run.poke("rbx", 1 << 40)
    assert run.trace.y != attach(tiny, inp(5, 3)).trace.y
    run.restore(snap)
    assert run.pos == 0 and int.from_bytes(run.trace.y, "little") == expected(5, 3)
    assert run.reads_of(0) and run.writes_of(0)[0][1] and 0x600000 in run.sites()


def test_hold_freezes_branch(tiny):
    run = attach(tiny, inp(5, 3))
    alt = run.fork("alt")
    alt.seek(20)
    alt.set("rbx", 1 << 40)                                              # flips jb
    taken = [tk.pc for tk in alt.trace.ticks]
    alt.hold("main")
    assert [tk.pc for tk in alt.trace.ticks] == [tk.pc for tk in run.trace.ticks] != taken


def test_symbols_records_and_grid(tiny):
    a, b = 5, 3
    ds = donors(16)
    rec = recover(tiny, inp(a, b), ds, log=lambda *_: None)
    tr = rec.base
    names = {tr.describe(c): k for c, k in zip(tr.cells, rec.binding) if k >= 0}
    # rax -> [rsp-16] -> rdx is one copy web (bytes 0..7 each)
    assert names["0x401010#0.mem[rsp-0x10:1]"] == names["0x40102b#0.rdx[0:8]"] == names["0x401000#0.rax[0:8]"]
    assert names["0x401026#0.flags.CF"] not in (names["0x401000#0.rax[0:8]"],)
    dead = {tr.describe(c) for c, lv in zip(tr.cells, rec.live) if not lv}
    assert "0x401026#0.flags.ZF" in dead and "0x401015#0.rcx[0:8]" in dead   # never read / loop counter fixed
    recs = symbol_table_records(rec)
    assert recs[0]["kind"] == "symbol-table" and recs[0]["body"]["page"]["total"] == rec.n_bindings
    cm = circuit_map_record(rec)
    assert cm["body"]["schema"] == "mdb-circuit-map-v1" and any(e["dst"] == "y" for e in cm["body"]["edges"])
    assert "symbol table" in table_text(rec)
    assert rec.b1 is not None and len(rec.b2) == len(tr.cells)
    live = [i for i, lv in enumerate(rec.live) if lv][:6]
    g = run_grid(tiny, inp(a, b), ds[:4], cells=live, log=lambda *_: None)
    col = g["columns"]
    assert len(col["verdict"]) == len(live) * (2 + 4 * 4)
    assert set(col["rule"]) == {"zero", "unaligned", "aligned"}
    ex = (col["rule"] == "aligned") & (col["align"] == "exec") & (col["status"] == "run")
    assert ex.any() and (col["base_web"][ex] == col["donor_web"][ex]).all()


def test_method_v2_words_and_dead_copies(tiny):
    from mdb.machines.x86.binary import FLAG0
    from mdb.machines.x86.method_v2 import lessons

    rec = recover(tiny, inp(5, 3), donors(16), log=lambda *_: None)
    frozen = rec.binding.copy()
    v2 = lessons(rec)
    assert (rec.binding == frozen).all()                       # the frozen method is untouched
    cells, b = rec.base.cells, v2.binding
    for i, c in enumerate(cells):                              # status follows liveness
        if b[i] >= 0:
            assert v2.status[b[i]] == ("live" if rec.live[i] else "dead")
    # same op instance, adjacent bytes of one register, same liveness -> one wide binding
    pairs = [(i, j) for i, c in enumerate(cells) for j, d in enumerate(cells)
             if c.def_ >= 0 and c.def_ == d.def_ and d.site == c.site + 1 and c.site < FLAG0
             and c.site // 8 == d.site // 8 and rec.live[i] == rec.live[j] and b[i] >= 0 and b[j] >= 0]
    assert pairs and all(b[i] == b[j] for i, j in pairs)
    names = {rec.base.describe(c): k for c, k in zip(cells, b)}
    # here byte 0 of the loaded a is live and bytes 1..7 are not: two bindings, not eight
    assert names["0x401000#0.rax[0:8]"] != names["0x401000#0.rax[8:16]"]
    assert len({names[f"0x401000#0.rax[{8 * k}:{8 * k + 8}]"] for k in range(1, 8)}) == 1
    # every not-live member of a copy web of >= 2 cells is bound, as dead
    size = {r: list(v2.web).count(r) for r in set(v2.web)}
    dead_web = [i for i in range(len(cells)) if not rec.live[i] and size[v2.web[i]] >= 2]
    assert dead_web and all(b[i] >= 0 and v2.status[b[i]] == "dead" for i in dead_web)
    # each lesson alone
    only_words, only_dead = lessons(rec, dead_copies=False), lessons(rec, words=False)
    assert all(s == "live" for s in only_words.status)
    assert only_dead.n_bindings - sum(s == "dead" for s in only_dead.status) == rec.n_bindings


# ---------------------------------------------------------------- cone + multi-donor block interchange (study C)
def _paths(seed=3, n=40):
    """inputs split by the final compare: xor path (x >= b, reads [rsp-0x10]) and skip path"""
    rng = random.Random(seed)
    xs, sk = [], []
    for _ in range(n):
        a, b = rng.getrandbits(62), rng.getrandbits(63)
        (xs if _xor_taken(a, b) else sk).append((a, b))
    return xs, sk


def _xor_taken(a, b):
    M = (1 << 64) - 1
    x = a
    for _ in range(4):
        x = (x + b) & M
    return (x + (a & 0xFF)) & M >= b


def test_cone_follows_the_taken_path(tiny):
    xs, sk = _paths()
    assert xs and sk
    for (a, b), slot_read in [(xs[0], True), (sk[0], False)]:
        run = attach(tiny, inp(a, b))
        c = run.cone()
        slot = 0x401010
        t_store = next(tk.t for tk in run.trace.ticks if tk.pc == slot)
        rsp16 = [s for s, _ in run.trace.ticks[t_store].writes if s >= 0x1000]
        assert len(rsp16) == 8
        assert bool(c.reads_of(rsp16)) == slot_read          # the stack copy of a is used only on the xor path
        assert c.reads_of(range(0x600000, 0x600010))          # both inputs feed y on every path
        assert c.answer_tick == next(tk.t for tk in run.trace.ticks if tk.pc == 0x40104F)
        # without control roots the jb is not in the cone; the data chain still reaches both inputs
        nc = run.cone(control=False)
        jb = next(tk.t for tk in run.trace.ticks if tk.pc == 0x401029)
        assert jb in c.ticks and jb not in nc.ticks
        assert nc.ticks <= c.ticks


def test_interchange_pools_donors_and_matches_python(tiny):
    a, b = 0x1234, 0x10
    run = attach(tiny, inp(a, b))
    ds = donors(12, seed=5)
    r = run.interchange("init.mem[0x600000:8]", ds)            # interchange a, the first input word
    want = [expected(int.from_bytes(d[:8], "little"), b) != expected(a, b) for d in ds]
    got = [row.get("y_changed", False) for row in r["per_donor"]]
    assert got == want
    assert r["verdict"] == "effect" and r["n_effect"] == sum(want) and r["n_informative"] == 12
    assert r["n_donors"] == 12 and "12 informative" in r["label"]
    # the base itself as the donor: an identity patch, not run, uninformative
    same = run.interchange("init.mem[0x600000:8]", [inp(a, b)])
    assert same["verdict"] == "no effect" and same["n_informative"] == 0
    assert "not informative" in same["label"] and not same["per_donor"][0]["ran"]
    # a block written mid-run, at an instance: the stack copy of a, after its store (0x401010#0)
    r2 = run.interchange("0x401010#0.mem[rsp-0x10:8]", ds[:4])
    xor_path = _xor_taken(a, b)
    assert r2["verdict"] == ("effect" if xor_path else "no effect")
    for row, d in zip(r2["per_donor"], ds[:4]):
        da = int.from_bytes(d[:8], "little")
        if xor_path:
            x = expected(a, b) ^ a ^ da                         # y = x ^ (patched copy)
            assert int.from_bytes(bytes.fromhex(row["y"]), "little") == x


def test_interchange_batched_over_bases(tiny):
    from mdb.machines.x86.interchange import interchange
    from mdb.machines.x86.trace import record

    bases_in = [inp(7, 3), inp(1 << 40, 5), inp(99, 1 << 50)]
    ds = donors(6, seed=11)
    bases, res = record(tiny, bases_in)
    dtr, _ = record(tiny, ds)
    out = interchange(tiny, bases, dtr, "init.mem[0x600008:8]", res=res)  # interchange b
    for (bi, o) in zip(bases_in, out):
        a, b = int.from_bytes(bi[:8], "little"), int.from_bytes(bi[8:], "little")
        want = [expected(a, int.from_bytes(d[8:], "little")) != expected(a, b) for d in ds]
        assert [row["y_changed"] for row in o["per_donor"]] == want
        assert o["n_effect"] == sum(want)
