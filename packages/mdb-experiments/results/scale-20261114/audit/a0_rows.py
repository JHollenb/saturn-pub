"""Audit: carrier row per item (argmin dM over span rows at the audited model carrier), my own code,
compared with the builder's row_at_item. Writes audit/carriers.json for the Beast CPU job."""
import json
import torch
A = json.load(open("audit/a1_carriers.json"))
P = json.load(open("panel.json"))
SW = {"Qwen2.5-0.5B": ["sw-Qwen2.5-0.5B"], "Qwen3-0.6B-Base": ["sw-Qwen3-0.6B-Base"], "Qwen2.5-1.5B": ["sw-Qwen2.5-1.5B"],
      "Qwen3-1.7B-Base": ["sw-Qwen3-1.7B-Base"], "Qwen2.5-3B": ["sw-Qwen2.5-3B-a", "sw-Qwen2.5-3B-b"], "Qwen3-4B-Base": ["sw-Qwen3-4B-Base-a", "sw-Qwen3-4B-Base-b"]}
out = {}
for m, ps in SW.items():
    sw = {}
    for p in ps:
        sw.update(torch.load(f"audit/data/{p}.pt", weights_only=False))
    LC, KC = A[m]["L_c"], A[m]["kv_c"]
    rows = {}
    for it in P["items"]:
        s = sw[it["id"]]
        best = None
        for j in range(len(s["dM"])):
            x, k, r = s["cells"][j].tolist()
            if x == LC and k == KC and (best is None or float(s["dM"][j]) < best[0]):
                best = (float(s["dM"][j]), r)
        rows[it["id"]] = best[1]
    b = json.load(open(f"out/score-{m}.json"))["carriers"]["row_at_item"]
    print(m, "rows agree with builder:", sum(rows[i] == b[i] for i in rows), "/", len(rows))
    out[m] = {"L_c": LC, "kv_c": KC, "row_at_item": rows}
json.dump(out, open("audit/carriers.json", "w"))
