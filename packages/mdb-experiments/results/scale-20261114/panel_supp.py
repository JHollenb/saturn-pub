"""Supplement panel (prereg addendum): d5 / d6 chains and native-name cities, built only from items the
second reader verified (fact_pools_verified.json verdict pass). python panel_supp.py TOKENIZER_DIR > panel_supp.json

Families: d5 (pass, no shortcut, 28), d6 (pass, no shortcut, all <= 28), d5-short (pass, shortcut flagged,
16, a control where skipping a hop still answers), native-city (pass, native name differs from English,
28) and english-city (the same 28 cities by English name, paired). Depth prompts use the depth few-shot
prefix and a language answer; city prompts the two-hop prefix and a capital answer. Span = the descriptor
phrase rows (depth) or the city name rows. Selection: sorted ids, round robin over countries. Donor rule
as panel.py. Items whose country, language or capital first token is not in the frozen sets are skipped
and listed.
"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
P = json.load(open("panel.json"))
SETS = P["sets"]
ext = json.load(open("candidates_ext.json"))
base = json.load(open("candidates.json"))
draft = json.load(open("fact_pools_draft.json"))
ver = {i["id"]: i for i in json.load(open("fact_pools_verified.json"))["items"]}
CAP_OF = {}
for i in base["items"]:
    CAP_OF.setdefault(i["country_token"], i["answer"])
for i in ext["items"]:
    if i.get("capital_token"):
        CAP_OF.setdefault(i["country_token"], i["capital_token"])


def first(word):
    return tok(" " + word, add_special_tokens=False)["input_ids"][0]


def rows_of(prompt, sub, start_at):
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    k = prompt.index(sub, start_at)
    rows = [r for r, (a, b) in enumerate(enc["offset_mapping"]) if a < k + len(sub) and b > k]
    return enc["input_ids"], rows


skipped, pools = [], {"d5": [], "d6": [], "d5-short": [], "native-city": [], "english-city": []}
for pool in ("d5", "d6"):
    for n, d in enumerate(draft[pool]):
        vid = f"{pool}-{n:03d}"
        v = ver[vid]
        if v["verdict"] != "pass":
            continue
        fam = pool + ("-short" if v["shortcut"].get("flag") else "")
        if fam == "d6-short":
            continue
        ct, lt = first(d["country"]), first(d["language"])
        cap = first(d["capital"])
        if ct not in SETS["countries"] or lt not in SETS["languages"]:
            skipped.append([vid, "country or language not in the frozen sets"])
            continue
        prompt = ext["prefix_depth"] + d["prompt"]
        ids, rows = rows_of(prompt, d["descriptor"], len(ext["prefix_depth"]))
        pools[fam].append({"id": f"{fam}:{vid}", "family": fam, "descriptor": d["descriptor"], "bridge": d["country"],
                           "answer": lt, "country_token": ct, "capital_token": cap if cap in SETS["cities"] else None,
                           "prompt": prompt, "ids": ids, "n": len(ids), "span": rows, "row": rows[-1],
                           "block": "depth", "answer_kind": "language", "verified": vid})
for n, d in enumerate(draft["native_city"]):
    vid = f"native_city-{n:03d}"
    if ver[vid]["verdict"] != "pass" or not d.get("differs_from_english"):
        continue
    ct = first(d["country"])
    capt = CAP_OF.get(ct)
    if ct not in SETS["countries"] or capt is None or capt not in SETS["cities"]:
        skipped.append([vid, "country or its capital not in the frozen sets"])
        continue
    for fam, name in (("native-city", d["native"]), ("english-city", d["english"])):
        prompt = base["prefix_two"] + f"The capital of the country containing the city of {name} is"
        ids, rows = rows_of(prompt, name, len(base["prefix_two"]))
        pools[fam].append({"id": f"{fam}:{vid}", "family": fam, "descriptor": name, "bridge": d["country"],
                           "answer": capt, "country_token": ct, "capital_token": capt, "prompt": prompt, "ids": ids,
                           "n": len(ids), "span": rows, "row": rows[-1], "block": "supp-city",
                           "answer_kind": "capital", "verified": vid})
COUNTS = {"d5": 28, "d6": 28, "d5-short": 16, "native-city": 28}


def pick(pool, n):
    bridges = sorted({i["bridge"] for i in pool})
    by_b = {b: [i for i in pool if i["bridge"] == b] for b in bridges}
    out, rnd = [], 0
    while len(out) < n and any(len(v) > rnd for v in by_b.values()):
        for b in bridges:
            if len(out) < n and len(by_b[b]) > rnd:
                out.append(by_b[b][rnd])
        rnd += 1
    return out


items = []
for fam, n in COUNTS.items():
    items += pick(sorted(pools[fam], key=lambda i: i["id"]), n)
nat = {i["verified"] for i in items if i["family"] == "native-city"}
items += [i for i in sorted(pools["english-city"], key=lambda i: i["id"]) if i["verified"] in nat]
for it in items:
    fam = [j for j in items if j["family"] == it["family"] and j["bridge"] != it["bridge"] and j["answer"] != it["answer"]]
    key = lambda d: (abs(len(d["span"]) - len(it["span"])), abs(d["n"] - it["n"]), d["id"])
    it["donor"] = min(fam, key=key)["id"] if fam else None
json.dump({"n": len(items), "counts": {f: sum(i["family"] == f for i in items) for f in pools}, "items": items,
           "sets": SETS, "skipped": skipped}, sys.stdout, indent=1)
