"""Compare main vs branch outputs: every bench digest field and example 04's records."""
import json
from pathlib import Path

TIMING = {"s", "ms_per_row", "until_s", "held_bytes_s", "row_s", "peak_vram_mb", "peak_rss_mb",
          "tree", "calls"}


def results(tree):
    out = []
    for line in Path(f"bench_{tree}.out").read_text().splitlines():
        if line.startswith("RESULT "):
            out.append(json.loads(line[7:]))
    return out


main, branch = results("main"), results("branch")
key = lambda r: (r["case"], r.get("T"), r.get("lookahead"))  # noqa: E731
bm = {key(r): r for r in branch}
ok = True
for r in main:
    b = bm.get(key(r))
    if b is None:
        print("MISSING on branch", key(r)); ok = False; continue
    same = {k: r[k] == b[k] for k in r if k not in TIMING}
    print("CASE", key(r), "identical" if all(same.values()) else f"DIFF {same}",
          "| main", {k: r[k] for k in ("s", "ms_per_row", "until_s", "held_bytes_s", "row_s", "peak_vram_mb", "peak_rss_mb") if k in r},
          "| branch", {k: b[k] for k in ("s", "ms_per_row", "until_s", "held_bytes_s", "row_s", "peak_vram_mb", "peak_rss_mb") if k in b})
    ok &= all(same.values())
for r in branch:
    if r.get("lookahead") == "auto":
        none = next((x for x in branch if x["case"] == "ports" and x["lookahead"] == "none"), None)
        print("AUTO vs NONE reads/trail identical:", none is not None and r["reads"] == none["reads"]
              and r["trail"] == none["trail"], "| auto", r["row_s"], r["calls"])
for name in ("records.jsonl",):
    a, b = Path("ex04_main/out") / name, Path("ex04_branch/out") / name
    if a.exists() and b.exists():
        same = a.read_bytes() == b.read_bytes()
        ids = lambda p: [json.loads(x).get("id") or json.loads(x).get("record") for x in p.read_text().splitlines() if x.strip()]  # noqa: E731
        print("EX04 records byte-identical:", same, "| ids main", ids(a), "| ids branch", ids(b))
        ok &= same
for tree in ("main", "branch"):
    p = Path(f"ex04_{tree}/run.out")
    if p.exists():
        lines = [x for x in p.read_text().splitlines() if x.startswith("RESULT_JSON")]
        if lines:
            r = json.loads(lines[-1][12:])
            r.pop("seconds", None); r.pop("peak_vram_mb", None)
            Path(f"ex04_{tree}.result").write_text(json.dumps(r, sort_keys=True))
if Path("ex04_main.result").exists() and Path("ex04_branch.result").exists():
    same = Path("ex04_main.result").read_text() == Path("ex04_branch.result").read_text()
    print("EX04 RESULT_JSON (minus seconds/peak_vram) identical:", same); ok &= same
print("ALL_IDENTICAL", ok)
