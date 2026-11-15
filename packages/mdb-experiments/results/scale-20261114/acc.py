"""Accuracy addendum (answer argmax only): python acc.py capture-*.json > prereg_scale_accuracy.json"""
import json
import sys

P = json.load(open("panel.json"))
out = {"note": "answer argmax == item answer, from Job C capture logs only; written before any readout or arm is opened",
       "models": {}}
for path in sys.argv[1:]:
    d = json.load(open(path))
    m = out["models"].setdefault(d["model"], {"files": [], "items": {}})
    m["files"].append(path)
    m["items"].update(d["items"])
for name, m in out["models"].items():
    fam = {}
    for it in P["items"]:
        r = m["items"].get(it["id"])
        if r is None:
            continue
        f = fam.setdefault(it["family"], [0, 0])
        f[0] += r["argmax"] == it["answer"]
        f[1] += 1
    d3 = [fam.get("d3-lm", [0, 0])[k] + fam.get("d3-per", [0, 0])[k] for k in (0, 1)]
    m["accuracy"] = {**fam, "d3_pooled": d3}
    m["n_items"] = len(m["items"])
    del m["items"]
json.dump(out, sys.stdout, indent=1)
