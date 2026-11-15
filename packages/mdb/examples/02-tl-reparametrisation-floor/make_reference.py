"""Cut the example's data from the showcase (light Python, no model): ``items.json`` (the 24
items per model this example runs, plus their donors) and ``reference.json`` (the showcase's raw
exp6 margins on those items, jobs job-c5a45f9c2811 Qwen3 / job-80cc14782825 Qwen2.5).

    python3 make_reference.py   # from this directory, in the repo checkout
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOW = HERE.parents[1] / "results" / "showcase-20261114"
panel = json.load(open(SHOW / "data" / "panel.json"))
by_id = {i["id"]: i for i in panel["items"]}
base = sorted((i for i in panel["items"] if i["block"] == "base"), key=lambda i: i["id"])
subset = base[::2]                                    # every second base item: 6 per family
keep = {i["id"] for i in subset} | {i["donor"] for i in subset}
fields = ("id", "family", "block", "prompt", "ids", "span", "answer", "donor")
items = [{k: by_id[i][k] for k in fields} for i in sorted(keep)]
json.dump({"source": "results/showcase-20261114/data/panel.json (results/scale-20261114 panel)",
           "subset": "every second of the 48 base items sorted by id (6 per family); donors "
                     "included only as donors",
           "run": [i["id"] for i in subset], "items": items},
          open(HERE / "items.json", "w"), indent=0)

KEYS = ("crow", "donor_crow", "a", "b", "mdb", "max_abs_logit", "U32", "P32", "U16", "U16_b1",
        "P16")
ref = {"source": "results/showcase-20261114/exp6/out/exp6-<model>.jsonl (MEASURED)",
       "jobs": {"Qwen3-0.6B-Base": "job-c5a45f9c2811", "Qwen2.5-0.5B": "job-80cc14782825"},
       "arms": "[base, zero head h at L_c (h = 0..H-1), at L_c+1, V interchange]; margins "
               "M = logit[a] - logit[b] (mdb: stepped native; U16/P16: TL batched; U16_b1: TL "
               "batch 1; U32/P32: fp32 CPU base only)",
       "models": {}}
want = {i["id"] for i in subset}
for model in ("Qwen3-0.6B-Base", "Qwen2.5-0.5B"):
    rows = [json.loads(line) for line in open(SHOW / "exp6" / "out" / f"exp6-{model}.jsonl")]
    ref["models"][model] = {r["id"]: {k: r[k] for k in KEYS} for r in rows if r["id"] in want}
    assert len(ref["models"][model]) == len(want)
json.dump(ref, open(HERE / "reference.json", "w"), separators=(",", ":"))
print(len(subset), "items,", len(items), "with donors")
