"""Build the fixed 128-item scale panel (memo §2) and its frozen candidate sets.

python panel.py TOKENIZER_DIR > panel.json

Rule (deterministic, no model outputs used):
  families and counts: currency 12, person 12, landmark 12, city 12 (base two-hop, capital answer);
  d2 14, d3-lm 14, d3-per 14, d4-work 14 (depth, language answer); dist-city 12, obsc-city 12 (hard).
  Every item is eligible. Selection walks the
  family's bridges in sorted order, round robin, taking the lowest-id item per bridge per round, so
  countries are spread. Donor = the panel item of the same family with another bridge and another
  answer token minimising (|span length diff|, |n diff|, id); donor rows are aligned to the item's
  span from the span end (cells with no aligned donor row are not swept).
  dist items also carry the distractor entity, its rows and its country (gen_ext.py: the distractor
  is the next base item of the family with another bridge).
Candidate sets: b1_sets.json sets, extended by every panel item's country/answer token not present.
"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
base = json.load(open("candidates.json"))["items"]
ext = json.load(open("candidates_ext.json"))["items"]
S = json.load(open("b1_sets.json"))
POOL = {}
for i in base + ext:
    POOL.setdefault(i["family"], []).append(i)
COUNTS = {"currency": 12, "person": 12, "landmark": 12, "city": 12,
          "d2": 14, "d3-lm": 14, "d3-per": 14, "d4-work": 14, "dist-city": 12, "obsc-city": 12}
BLOCK = {"currency": "base", "person": "base", "landmark": "base", "city": "base",
         "d2": "depth", "d3-lm": "depth", "d3-per": "depth", "d4-work": "depth",
         "dist-city": "hard", "obsc-city": "hard"}


def donor_of(item, pool):
    c = [d for d in pool if d["bridge"] != item["bridge"] and d["answer"] != item["answer"]]
    key = lambda d: (abs(len(d["span"]) - len(item["span"])), abs(d["n"] - item["n"]), d["id"])
    return min(c, key=key)["id"] if c else None


def dist_info(item):
    city = {i["descriptor"]: i for i in base if i["family"] == "city"}
    pre = "My friend just came back from a trip to "
    p = item["prompt"]
    k = p.index(pre) + len(pre)
    y = p[k:p.index(". ", k)]
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == item["ids"], item["id"]
    rows = [r for r, (a, b) in enumerate(enc["offset_mapping"]) if a < k + len(y) and b > k]
    d = city[y]
    return {"entity": y, "rows": rows, "country": d["bridge"], "country_token": d["country_token"],
            "answer": d["answer"]}


panel = []
for fam, n in COUNTS.items():
    pool = sorted(POOL[fam], key=lambda i: i["id"])
    elig = pool
    bridges = sorted({i["bridge"] for i in elig})
    by_b = {b: [i for i in elig if i["bridge"] == b] for b in bridges}
    chosen, rnd = [], 0
    while len(chosen) < n and any(len(v) > rnd for v in by_b.values()):
        for b in bridges:
            if len(chosen) < n and len(by_b[b]) > rnd:
                chosen.append(by_b[b][rnd])
        rnd += 1
    assert len(chosen) == n, (fam, len(chosen))
    for i in chosen:
        assert tok(i["prompt"], add_special_tokens=False)["input_ids"] == i["ids"], i["id"]
        rec = {k: i[k] for k in ("id", "family", "descriptor", "bridge", "answer", "country_token",
                                 "prompt", "ids", "n", "span", "row")}
        rec["block"] = BLOCK[fam]
        rec["answer_kind"] = "language" if BLOCK[fam] == "depth" else "capital"
        rec["capital_token"] = i.get("capital_token") or (i["answer"] if rec["answer_kind"] == "capital" else None)
        rec["donor"] = donor_of(i, chosen)
        if fam == "dist-city":
            rec["distractor"] = dist_info(i)
        panel.append(rec)
by_id = {i["id"]: i for fam in POOL.values() for i in fam}
donors = {}
for p in panel:
    d = by_id[p["donor"]]
    donors[d["id"]] = {k: d[k] for k in ("id", "family", "bridge", "answer", "country_token", "prompt",
                                         "ids", "n", "span", "row")}
sets = {k: list(v) for k, v in S["sets"].items()}
for p in panel + list(donors.values()):
    if p["country_token"] not in sets["countries"]:
        sets["countries"].append(p["country_token"])
    if p.get("answer") and BLOCK.get(p["family"]) != "depth" and p["answer"] not in sets["cities"]:
        sets["cities"].append(p["answer"])
    if BLOCK.get(p["family"]) == "depth" and p["answer"] not in sets["languages"]:
        sets["languages"].append(p["answer"])
for p in panel:
    if "distractor" in p and p["distractor"]["country_token"] not in sets["countries"]:
        sets["countries"].append(p["distractor"]["country_token"])
first_sub = {k: [tok.decode([t]) for t in v] for k, v in sets.items()}
json.dump({"n": len(panel), "counts": COUNTS, "items": panel, "donors": donors, "sets": sets,
           "set_strings": first_sub, "persons": S["people"], "creator": S["creator"]}, sys.stdout, indent=1)
