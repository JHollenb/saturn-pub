"""Cut items.json and reference.json for this example from the saved outputs of the original run
(light Python, no model). Run from this directory:  python3 make_reference.py

Kept for provenance: it needs the original run's raw outputs, which are not part of this package;
the two JSON files it writes are already in this directory.

items.json      the original run's 32 base items: token ids, answer, donor answer, and the
                random-node feature it drew for each item (its seeded rng ran over the whole
                panel, so a subset cannot redraw it; the example reuses the recorded pick).
reference.json  the original run's raw per-item records for the same items, plus the full-panel
                numbers for context.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MDB = HERE.parents[1]
SHOW = MDB / "results/showcase-20261114/exp4"
ITEMS = MDB / "results/twohop-20261113/next/r1_items.json"

# The original panel: the first 8 base items by id of each two-hop family.
spec = json.load(open(ITEMS))
fams: dict = {}
for item in sorted((i for i in spec["items"] if i["group"] == "base"), key=lambda i: i["id"]):
    fams.setdefault(item["family"], []).append(item)
panel = [i for f in sorted(fams) for i in fams[f][:8]]
raw = {r["id"]: r for r in map(json.loads, open(SHOW / "out/exp4-items.jsonl"))}
assert [i["id"] for i in panel] == list(raw), "panel order differs from the showcase output"

# All 32 items of the original panel: the full example runs in about 4 minutes, so no subset is needed
# (MDB_ITEMS picks a subset for a smoke run). Above-floor node-items: 103 of 160.
def above(r):
    return sum(abs(n["arm0"]) > r["floor"] for n in r["nodes"] if n["kind"] == "node")


SUBSET = list(range(len(panel)))
items = []
for k in SUBSET:
    it, r = panel[k], raw[panel[k]["id"]]
    rnd = [n for n in r["nodes"] if n["kind"] == "random"]
    items.append({"panel_index": k, "id": it["id"], "family": it["family"], "ids": it["ids"],
                  "answer": it["answer"], "donor_answer": it["donor"]["answer"],
                  "donor_id": it["donor"]["id"],
                  "random_node": ({"layer": rnd[0]["layer"], "feature": rnd[0]["feature"]}
                                  if rnd else None)})
json.dump({"source": "results/twohop-20261113/next/r1_items.json (exp4 panel, base group)",
           "subset_rule": "all 32 items of the exp4 panel, in panel order",
           "items": items}, open(HERE / "items.json", "w"), indent=1)
json.dump({"source": "results/showcase-20261114/exp4/out/exp4-items.jsonl (job-09feeeb61cbe)",
           "full_panel_audited": {
               "items": 32, "node_items": 160, "arm0_above_floor": 103,
               "any_bracket_above_floor": 44, "G1_all": 0.105, "G1_above_floor": 0.095,
               "arm0_over_edge_above_floor": 0.92, "arm0_over_edge_above_4_floors": 0.97,
               "arm0_over_direct_above_floor": "0.61-0.64", "G4_spearman": 0.932,
               "audit": "results/showcase-20261114/AUDIT-SHOWCASE.md (exp4 row, correction 3)"},
           "items": {it["id"]: raw[it["id"]] for it in items}},
          open(HERE / "reference.json", "w"), indent=1)
print(len(items), "items:", [i["id"] for i in items])
print("above-floor node-items in the subset:", sum(above(raw[i["id"]]) for i in items))
