"""A 28 x 32 grid of labelled numbers in one record: as num dicts (v2 and v3) and as a column (v3).
usage: python bench_grid.py"""
import json
import statistics
import time

from mdb.core import records as R

R.declare_kind("bench-grid", plain_numeric=())
EXE = {"dtype": "bfloat16", "device": "cpu"}
grid = [[((i * 32 + j) * 0.618033988749895) % 1 - 0.5 for j in range(32)] for i in range(28)]


def timed(fn, reps=20):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return round(1e3 * statistics.median(out), 3)


def try_seal(body):
    try:
        r = R.seal_record(session_id="s", kind="bench-grid", model_identity="m", execution=EXE,
                          authority="derived-exact", body=body)
        return r, None
    except R.RecordError as e:
        return None, str(e)[:120]


out = {"schema": R.SCHEMA}
for rows in (14, 28):
    body = {f"L{i:02d}": [R.num(v, "derived-exact", proof="P15") for v in grid[i]]
            for i in range(rows)}
    r, err = try_seal(body)
    key = f"num_dicts_{rows}x32"
    out[key] = ({"bytes": len(R.stored_bytes(r)), "ms_seal": timed(lambda: try_seal(body)),
                 "ms_verify": timed(lambda: R.verify_record(r))} if r else {"refused": err})
if hasattr(R, "nums"):
    body = {"grid": R.nums(grid, "derived-exact", proof="P15")}
    r, err = try_seal(body)
    line = R.stored_form(r)
    out["column_28x32"] = {"bytes": len(R.stored_bytes(r)), "ms_seal": timed(lambda: try_seal(body)),
                           "ms_verify": timed(lambda: R.verify_record(r)),
                           "ms_load_line": timed(lambda: R.load_record(line, execution=EXE))}
print(json.dumps(out, indent=1))
