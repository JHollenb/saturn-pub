#!/usr/bin/env python3
"""Auditor: how close is the M2 frozen flag to the label-defining counterfactual A3 (no-hint prompt + same chain != X)?"""
import gzip, json
from collections import Counter
from pathlib import Path
H = Path(__file__).resolve().parent.parent
items = {it["uid"]: it for it in json.loads((H / "items.json").read_text())["items"]}
recs = {}
for p in sorted((H / "data").glob("*.jsonl.gz")):
    for l in gzip.decompress(p.read_bytes()).decode().splitlines():
        if l.strip():
            r = json.loads(l); recs[(r["uid"], r["f"])] = r
c = Counter()
for (u, f), r in recs.items():
    it = items[u]
    if f != 1.0 or it["hint_type"] == "sycophancy":
        continue
    a3 = it["old"].get("A3")
    a3flip = None if a3 is None else (a3 != it["x"])
    flag = r["margin_ablated"] < 0
    c[(it["label"], f"A3_flip={a3flip}", f"M2flag={flag}")] += 1
out = {" | ".join(k): v for k, v in sorted(c.items())}
print(json.dumps(out, indent=1)); (H / "audit" / "a3_agreement.json").write_text(json.dumps(out, indent=1))
