"""Release-gate 0.2.0 probe for example 03: do batched f-port arms work now?

Runs the unmodified example (composer_neurons.py) with one wrapper: after each stepped
``Branches.run(names)`` the same arms are measured with ``Branches.batched(names, recheck=...)``
(default band, then recheck='all' on the first item) and compared arm by arm with the stepped
rows. The stepped rows (and each arm's stepped result) are restored, so the example's own numbers
and checks are unchanged by the probe. Prints PROBE03 lines and a PROBE03_JSON summary.
Not a validation run: the example's numbers in this job are not the release numbers.
"""
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

from mdb.analysis import branches as B

OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out")
_orig_run = B.Branches.run
SUMMARY = {"items": [], "errors": []}


def _run(self, names=None, *, mode="stepped", **kw):
    rows = _orig_run(self, names, mode=mode, **kw)
    if mode != "stepped":
        return rows
    chosen = list(self.arms) if names is None else list(names)
    stepped = {r["name"]: r for r in rows}
    saved = {n: self.arm(n).result for n in chosen}
    item = {"branch": self.name if hasattr(self, "name") else None, "arms": {}}
    try:
        for recheck in (["band", "all"] if not SUMMARY["items"] else ["band"]):
            b = self.batched(chosen, recheck=recheck)
            for r in b:
                s = stepped[r["name"]]
                d = {"stepped": s["dM"]["value"], "batched": r["dM"]["value"],
                     "diff": r["dM"]["value"] - s["dM"]["value"],
                     "floor": (r.get("floor") or {}).get("value") if isinstance(r.get("floor"), dict) else r.get("floor"),
                     "recheck": r.get("recheck"),
                     "kernel_delta": (r.get("kernel_delta") or {}).get("value")}
                item["arms"].setdefault(recheck, {})[r["name"]] = d
                print(f"PROBE03 {item['branch']} recheck={recheck} {r['name']:11s} stepped {d['stepped']:+.4f} "
                      f"batched {d['batched']:+.4f} diff {d['diff']:+.4f} recheck_done "
                      f"{(d['recheck'] or {}).get('done')} ({(d['recheck'] or {}).get('why')})", flush=True)
    except Exception as e:  # noqa: BLE001 - the probe records any refusal
        SUMMARY["errors"].append({"branch": item["branch"], "error": f"{type(e).__name__}: {e}",
                                  "tb": traceback.format_exc()[-1500:]})
        print(f"PROBE03 ERROR {item['branch']}: {type(e).__name__}: {e}", flush=True)
    finally:
        for n, res in saved.items():
            self.arm(n).result = res
        self.restore()
    SUMMARY["items"].append(item)
    return rows


B.Branches.run = _run
example = Path(sys.argv[1]).resolve()
sys.argv = [str(example)]
try:
    runpy.run_path(str(example), run_name="__main__")
finally:
    diffs = [abs(a["diff"]) for it in SUMMARY["items"] for a in it["arms"].get("band", {}).values()]
    SUMMARY["n_items"] = len(SUMMARY["items"])
    SUMMARY["n_arms"] = len(diffs)
    SUMMARY["max_abs_batched_minus_stepped"] = max(diffs) if diffs else None
    (OUT / "probe03.json").write_text(json.dumps(SUMMARY, default=str, indent=1))
    print("PROBE03_JSON " + json.dumps({k: SUMMARY[k] for k in ("n_items", "n_arms",
                                                                 "max_abs_batched_minus_stepped")}
                                       | {"n_errors": len(SUMMARY["errors"]),
                                          "first_error": SUMMARY["errors"][0]["error"] if SUMMARY["errors"] else None}))
