"""Demo transcript on bsort-O2 (the experiment's demo, not an mdb example): step by loop iteration, watch
the array, break on the compare, patch a cell and see ``path-diverged``, print the recovered symbol table.

    python demo.py --ref REF --containers CDIR [--device cuda] --out result/demo.txt
"""
from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parents[2] / "mdb/src"):
    if p.exists():
        sys.path.insert(0, str(p))

from worker import inputs_of  # noqa: E402

ARR = 0x403000  # gt_input (ELF symbol table)


def demo(ref: Path, cdir: Path, device: str, symbols: bool = True) -> None:
    from mdb.machines.x86 import Binary
    from mdb.machines.x86.records import symbol_table_records, table_text
    from mdb.machines.x86.session import attach

    def say(cmd, out=None):
        print(f"(mdb) {cmd}")
        if out is not None:
            print(out if isinstance(out, str) else repr(out))

    b = Binary.load(ref / "programs" / "bsort-O2")
    ins, _ = inputs_of(cdir, "bsort-O2")
    run = attach(b, ins[0], device=device, donors=ins[1:])
    say("attach bin:bsort-O2 input=default", f"T={run.trace.T} stop={run.trace.stop} "
        f"loops={[hex(lp.head) for lp in b.loops if lp.head < 0x402000]}")
    say("x/8gx gt_input", " ".join(f"{v:#x}" for v in run.x(f"mem[{ARR:#x}:8]", 8)))
    say("info domains", ", ".join(d.name for d in run.domains()))
    for _ in range(3):
        say("step 1 unit=loop:0x401030", run.step(1, "loop:0x401030"))
        say("print rax", run.print("rax"))
    say("step 1 unit=loop:0x401028", run.step(1, "loop:0x401028"))
    say("print rsi", run.print("rsi"))
    say("reverse-step 1 unit=loop:0x401030", run.reverse_step(1, "loop:0x401030"))
    say("seek 0", run.seek(0))
    w = run.watch(f"mem[{ARR:#x}:8]")
    say(f"watch mem[{ARR:#x}:8]  (change)", f"watch {w}")
    for _ in range(2):
        say("continue", run.cont())
        say("x/2gx gt_input", " ".join(f"{v:#x}" for v in run.x(f"mem[{ARR:#x}:8]", 2)))
    run.delete()
    say("seek 0", run.seek(0))
    bp = run.break_at("0x40103e#5")
    say("break 0x40103e#5   (6th execution of cmp rcx, rdx)", f"breakpoint {bp}")
    say("continue", run.cont())
    say("print rcx", run.print("rcx"))
    say("print rdx", run.print("rdx"))
    say("reads rcx", "\n".join(f"  {r['reader']} {r['text']}  [{r['edge']}, {len(r['sites'])} bytes]"
                              for r in run.reads("rcx")))
    run.delete()
    flip = run.read("rdx")["value"]       # rcx := rdx flips jae (rcx < rdx before: the swap ran)
    say(f"whatif rcx = {flip:#x}", {k: run.whatif("rcx", flip)[k] for k in
                                     ("verdict", "total_effect", "frozen_effect", "path_diverged_at",
                                      "selection_moved_under_hold_at", "prediction")})
    say("fork orig", run.fork("orig").name)
    out = run.set("rcx", flip)
    say(f"set rcx = {flip:#x}", out)
    say("compare orig", run.compare("orig"))
    say("hold orig  (branch outcomes + addresses frozen to the unedited run)", run.hold("orig"))
    say("compare orig", {k: v for k, v in run.compare("orig").items() if k in ("verdict", "path_diverged_at", "y")})
    say("hold off", run.hold(None))
    if not symbols:
        return
    say("symbols  (run-1 method, 32 donors)")
    base = attach(b, ins[0], device=device, donors=ins[1:])
    rec = base.symbols(log=lambda *_: None)
    print(table_text(rec, limit=40))
    recs = symbol_table_records(rec)
    say("records", f"{len(recs)} sealed mdb-symbol-table-v1 page(s); first id {recs[0]['id']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--containers", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-symbols", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("result/demo.txt"))
    a = ap.parse_args(argv)
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo(a.ref, a.containers, a.device, not a.no_symbols)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(buf.getvalue())
    print(buf.getvalue(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
