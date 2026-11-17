"""Beast worker for study C (saturn/experiments/2026-11-16-cot-latent-binaries/C/PREREG.md): CoT-faithfulness
monitors on the cot suite, per input, through mdb's x86 machine on torchx86 GPU lanes.

    python worker.py --ref REF --programs faithful-O0 ... --device cuda --out result

Per program: the 64 base inputs and 16 donor inputs are ``reference.inputs_for(name, 80)`` (bases 0..63, donors
64..79), recorded as lanes in one batch. For every base input:

- B-read: ``interchange.cone`` of the answer (control roots on; and off, as a diagnostic); the number of the
  cone's reads of the hint byte (gt_input+12) and of cot_buf's 10 bytes.
- M-causal: ``interchange.interchange`` of the hint byte at ``init`` and of cot_buf right after the chain's
  write syscall (``<pc>#0``), over the 16 donors; per donor y_changed / informative / path_diverged, pooled.
  M-causal-1 is read off donor 0 (and every single donor) by the scorer.
- B-tie: the top-two gap of the four evidence scores, read from cot_buf at the chain's write.

The labels are not read here (``score.py`` computes them from ``reference.py``); the worker writes only what
the monitors see plus the lane checks (y == reference answer, chain bytes == reference chain).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE / "programs/cot", HERE.parents[2] / "mdb/src"):
    if p.exists():
        sys.path.insert(0, str(p))

import torch  # noqa: E402

N_BASE, N_DONOR, COT_LEN, HINT_OFF = 64, 16, 10, 12


def chain_tick(tr) -> tuple[int, int]:
    """(tick, pc) of the first write syscall (the chain's)."""
    from mdb.machines.x86.interchange import _is_write

    t = next(t for t in range(tr.T) if _is_write(tr, t))
    return t, tr.ticks[t].pc


def gap(cot: bytes) -> int:
    s = sorted((int.from_bytes(cot[2 * o:2 * o + 2], "little") for o in range(4)), reverse=True)
    return s[0] - s[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--programs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    import reference
    from mdb.machines.x86 import Binary
    from mdb.machines.x86.binary import elf_symbol
    from mdb.machines.x86.interchange import cone, interchange
    from mdb.machines.x86.trace import record

    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"torch {torch.__version__} device {dev} {torch.cuda.get_device_name() if dev == 'cuda' else ''}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for prog in a.programs:
        t0 = time.time()
        name = prog.split("-")[0]
        pdir = a.ref / "programs" / prog
        b = Binary.load(pdir)
        cot_addr, cot_size = elf_symbol(pdir / "prog.elf", "cot_buf")
        assert cot_size == COT_LEN, cot_size
        hint_addr = b.input_addr + HINT_OFF
        ins = reference.inputs_for(name, N_BASE + N_DONOR)
        traces, res = record(b, ins, device=dev)
        bases, donors = traces[:N_BASE], traces[N_BASE:]
        t_rec = time.time() - t0
        ticks = [chain_tick(tr) for tr in traces]
        pcs = {pc for _, pc in ticks}
        assert len(pcs) == 1, f"chain write at several pcs {pcs}"
        cpc = pcs.pop()
        hint_text = f"init.mem[{hint_addr:#x}:1]"
        chain_text = f"{cpc:#x}#0.mem[{cot_addr:#x}:{COT_LEN}]"
        lane_ok = []
        for tr, inp, (tc, _) in zip(traces, ins, ticks):
            want = reference.expected(name, inp)["stdout_hex"]
            lane_ok.append(tr.stop == "exit" and tr.exit_code == 0 and tr.y.hex() == want[-2:]
                           and tr.mem(tc, cot_addr, COT_LEN).hex() == want[:-2])
        t1 = time.time()
        mh = interchange(b, bases, donors, hint_text, device=dev, res=res)
        mc = interchange(b, bases, donors, chain_text, device=dev, res=res)
        t_int = time.time() - t1
        items = []
        hint_sites, cot_sites = [hint_addr], list(range(cot_addr, cot_addr + COT_LEN))
        t2 = time.time()
        for i, tr in enumerate(bases):
            c, nc = cone(tr), cone(tr, control=False)
            tc = ticks[i][0]
            cot = tr.mem(tc, cot_addr, COT_LEN)
            hr, cr = c.reads_of(hint_sites), c.reads_of(cot_sites)
            items.append({
                "i": i, "input_hex": ins[i].hex(), "T": tr.T, "answer": tr.y.hex(), "lane_ok": lane_ok[i],
                "bread_hint_n": len(hr), "bread_chain_n": len(cr),
                "bread_hint_edges": sorted({e for _, _, e in hr}), "bread_chain_edges": sorted({e for _, _, e in cr}),
                "bread_hint_n_nocontrol": len(nc.reads_of(hint_sites)),
                "bread_chain_n_nocontrol": len(nc.reads_of(cot_sites)),
                "cone_ticks": len(c.ticks), "gap": gap(cot),
                "mc_hint": {k: mh[i][k] for k in ("verdict", "n_effect", "n_informative", "n_donors", "label",
                                                  "n_path_diverged")} | {
                    "y_changed": [bool(r.get("y_changed", False)) for r in mh[i]["per_donor"]],
                    "informative": [r["informative"] for r in mh[i]["per_donor"]],
                    "path_diverged": [bool(r.get("path_diverged", False)) for r in mh[i]["per_donor"]]},
                "mc_chain": {k: mc[i][k] for k in ("verdict", "n_effect", "n_informative", "n_donors", "label",
                                                   "n_path_diverged")} | {
                    "y_changed": [bool(r.get("y_changed", False)) for r in mc[i]["per_donor"]],
                    "informative": [r["informative"] for r in mc[i]["per_donor"]],
                    "path_diverged": [bool(r.get("path_diverged", False)) for r in mc[i]["per_donor"]]}})
        t_cone = time.time() - t2
        summary = {"program": prog, "hint": hint_text, "chain": chain_text, "cot_buf": hex(cot_addr),
                   "lanes_equal_reference": f"{sum(lane_ok)}/{len(lane_ok)}",
                   "T_min": min(tr.T for tr in traces), "T_max": max(tr.T for tr in traces),
                   "seconds": {"record": round(t_rec, 1), "interchange": round(t_int, 1), "cone": round(t_cone, 1)},
                   "device": dev, "torch": torch.__version__}
        (a.out / f"{prog}.json").write_text(json.dumps({"summary": summary, "items": items}))
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
