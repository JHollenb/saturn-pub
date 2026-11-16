"""Record size and write/read time on a tiny CPU model, runnable on the v2 (main) and v3 trees.

usage: python bench_records.py OUTDIR
Writes OUTDIR/sink.jsonl (every record sealed), OUTDIR/project/, OUTDIR/bench.json.
"""
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
for p in (OUT / "sink.jsonl",):
    p.unlink(missing_ok=True)

import mdb  # noqa: E402
from mdb.core import custody, index  # noqa: E402
from mdb.core import records as R  # noqa: E402
from mdb.analysis import rows  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

import shutil  # noqa: E402
shutil.rmtree(OUT / "project", ignore_errors=True)
sink = custody.records_to(OUT / "sink.jsonl")

# --- prompt set: 32 recall-shaped items over the tiny vocabulary
rng = random.Random(7)
items = []
for i in range(32):
    n = rng.randint(4, 7)
    toks = [rng.randint(1, 31) for _ in range(n)]
    items.append({"name": f"i{i:02d}", "tokens": toks, "roles": {"subject": 1},
                  "vars": {"answer": 4, "foil": 9}, "pair": f"i{(i + 1) % 32:02d}"})
(OUT / "set.jsonl").write_text("\n".join(json.dumps(x) for x in items) + "\n")

dbg = open_tiny("qwen2", "bf16", project=OUT / "project")
t0 = time.perf_counter()
each = dbg.each(str(OUT / "set.jsonl"), "whatif L1.A[h1] *= 0 --consumer M[$answer,$foil]")
t_each = time.perf_counter() - t0

# --- a stepped row trail: row table over 28 rows, sealed as a trace record (payload in store)
torch.manual_seed(0)
dbg32 = open_tiny("qwen2", "fp32", project=OUT / "project")
IDS = [(i * 7) % 31 + 1 for i in range(28)]
run = dbg32.run(IDS)
trace = rows.row_table(run, {"sink": [0], "question": list(range(1, 6)),
                             "hint": list(range(6, 10)), "template": list(range(10, 14))},
                       letters=[20, 21, 22, 23], X="B", R="C", arms={"hint": list(range(6, 10))})
trace_record = trace.seal(session_id=dbg32.session_id, run=run, store=dbg32.project)
sink.close()


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


recs = sink.records()
by_kind = {}
for r in recs:
    by_kind.setdefault(r["kind"], []).append(len(R.stored_bytes(r)))

fields = ("session_id", "kind", "model_identity", "execution", "authority", "body", "pointers")
REPS = 20
t_seal = sum(timed(lambda r=r: R.seal_record(**{k: r[k] for k in fields}), REPS) for r in recs)
t_verify = sum(timed(lambda r=r: R.verify_record(r), REPS) for r in recs)
t_stored = sum(timed(lambda r=r: R.stored_bytes(r), REPS) for r in recs)
t_sink_read = timed(lambda: custody.Sink(OUT / "sink.jsonl").records(), 5)


def write_sink():
    p = OUT / "w.jsonl"
    p.unlink(missing_ok=True)
    s = custody.Sink(p)
    for r in recs:
        s.write(r)


t_sink_write = timed(write_sink, 5)
t_list = timed(lambda: index.list_records(dbg.project, dbg.session_id), 5)
payload = trace_record["body"]["payload"]
out = {
    "schema": recs[0]["schema"], "n_records": len(recs),
    "each_wall_s": round(t_each, 3), "each_stored": each.summary.get("stored"),
    "bytes_by_kind": {k: {"n": len(v), "total": sum(v), "max": max(v)} for k, v in
                      sorted(by_kind.items())},
    "bytes_total": sum(sum(v) for v in by_kind.values()),
    "sink_file_bytes": (OUT / "sink.jsonl").stat().st_size,
    "trace_record_bytes": len(R.stored_bytes(trace_record)),
    "trace_payload_bytes": payload.get("bytes"),
    "ms_seal_all": round(1e3 * t_seal, 2), "ms_verify_all": round(1e3 * t_verify, 2),
    "ms_stored_bytes_all": round(1e3 * t_stored, 2),
    "ms_sink_write_all": round(1e3 * t_sink_write, 2),
    "ms_sink_read_verify_all": round(1e3 * t_sink_read, 2),
    "ms_index_list_records_session": round(1e3 * t_list, 2),
    "n_index_records_session": len(index.list_records(dbg.project, dbg.session_id)),
    "ids": [r["id"] for r in recs],
}
(OUT / "bench.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "ids"}, indent=1))
