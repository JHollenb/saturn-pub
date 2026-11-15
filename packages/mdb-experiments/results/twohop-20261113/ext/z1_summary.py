"""Z1 verdicts (prereg_chain.json) from z1-*.jsonl: python3 z1_summary.py FILE.jsonl; writes z1-summary.json."""
import json
import statistics as st
import sys
from collections import Counter

rows = [json.loads(line) for line in open(sys.argv[1])]


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


def readable(get, name):
    return rate(bool(get(r)[name]["names_truth"]) for r in rows)


view = lambda r: r["table"]["bridge"]["content"]  # noqa: E731
out = {"n": len(rows)}
for s in ("countries", "all", "capitals", "entities"):
    out[f"view_readable@1({s})"] = readable(view, s)
out["view_all_best_kind"] = dict(Counter(view(r)["all"]["kind"] for r in rows))
out["capital_in_cell"] = rate(view(r)["capitals"]["names_truth"] for r in rows)
out["entity_in_cell (all-set best is an entity)"] = rate(view(r)["all"]["kind"] == "entity" for r in rows)
for s in ("countries", "all"):
    out[f"B_mass({s})"] = readable(lambda r: r["B_mass"], s)
    out[f"B_answer({s})"] = readable(lambda r: r["B_answer"], s)
    for x in ("L20", "L24"):
        out[f"B_lens_{x}({s})"] = readable(lambda r, x=x: r["B_lens"][x], s)
    out[f"B_lens_better({s})"] = max(out[f"B_lens_L20({s})"], out[f"B_lens_L24({s})"],
                                     key=lambda v: v[0])
out["B_mass_site_is_bridge"] = rate(r["B_mass"]["site"] == [r["r0"], 20, 6] for r in rows)
td = [r["T_diff"] for r in rows]
out["T_diff"] = {"content_moves": rate(t["content_moves"] for t in td),
                 "answer_moves": rate(t["answer_moves"] for t in td),
                 "together": rate(t["together"] for t in td),
                 "2x2 (content, answer)": {f"{a},{b}": sum(1 for t in td if t["content_moves"] == a
                                                           and t["answer_moves"] == b)
                                           for a in (True, False) for b in (True, False)}}
tc = [r["T_city"] for r in rows if "T_city" in r]
if tc:
    out["T_city"] = {"n": len(tc), "country_unchanged": rate(t["country_unchanged"] for t in tc),
                     "content_moves(entity)": rate(t["content_moves"] for t in tc),
                     "answer_moves": rate(t["answer_moves"] for t in tc),
                     "answer_is_donor_city": rate(t["answer_is_donor_city_token"] for t in tc),
                     "together": rate(t["together"] for t in tc),
                     "items": [{k: t[k] for k in ("donor", "argmax_text", "content_entity_was",
                                                  "content_entity", "content_country", "together")}
                               for t in tc]}
fa = []
for r in rows:
    rest = r["table"]["r0rest"]
    if rest["content"]["countries"]["names_truth"]:
        strip = rest["strip"]["countries"]
        fa.append(strip.index("C") if "C" in strip else None)
out["r0rest_readable(countries)"] = rate(r["table"]["r0rest"]["content"]["countries"]["names_truth"]
                                         for r in rows)
out["first_appearance_le_20"] = rate(x is not None and x <= 20 for x in fa)
out["first_appearance_median"] = st.median(x for x in fa if x is not None) if any(
    x is not None for x in fa) else None
shares, checks = [], []
for r in rows:
    h = r["hop_check"]
    checks.append(abs(h["native_delta"] - h["job2_GT.V.d0_delta"]))
    if h["native_delta"]:
        shares.append(1 - h["direct_delta"] / h["native_delta"])
out["hop_indirect_share_median"] = st.median(shares)
out["hop_indirect_share_quartiles"] = st.quantiles(shares, n=4)
out["hop_native_vs_job2_max_abs_diff"] = max(checks)
out["hop_direct_median"] = st.median(r["hop_check"]["direct_delta"] for r in rows)
out["hop_native_median"] = st.median(r["hop_check"]["native_delta"] for r in rows)
fam = {}
for f in sorted({r["family"] for r in rows}):
    rs = [r for r in rows if r["family"] == f]
    fam[f] = {"countries": rate(view(r)["countries"]["names_truth"] for r in rs),
              "all": rate(view(r)["all"]["names_truth"] for r in rs),
              "lens24": rate(r["B_lens"]["L24"]["countries"]["names_truth"] for r in rs),
              "T_diff_together": rate(r["T_diff"]["together"] for r in rs)}
out["by_family"] = fam
v_c, v_a = out["view_readable@1(countries)"][2], out["view_readable@1(all)"][2]
lc, la = out["B_lens_better(countries)"][2], out["B_lens_better(all)"][2]
out["verdicts"] = {
    "Z1 view(countries)>=0.80": v_c >= 0.80,
    "Z2 view(all)>=0.60": v_a >= 0.60,
    "Z3 lens<=view-0.10 (countries) and <=view-0.20 (all)": lc <= v_c - 0.10 and la <= v_a - 0.20,
    "Z4 B_answer(countries)<=0.50": out["B_answer(countries)"][2] <= 0.50,
    "Z5 B_mass(countries)<=view-0.10": out["B_mass(countries)"][2] <= v_c - 0.10,
    "Z6 T_diff content>=0.85 and together>=0.60": out["T_diff"]["content_moves"][2] >= 0.85
    and out["T_diff"]["together"][2] >= 0.60,
    "Z7 T_city country>=0.80, entity>=0.50, together>=0.60": (
        tc and out["T_city"]["country_unchanged"][2] >= 0.80
        and out["T_city"]["content_moves(entity)"][2] >= 0.50
        and out["T_city"]["together"][2] >= 0.60) if len(tc) >= 8 else "n<8: reported",
    "Z8 first_appearance<=20 on >=0.70": (out["first_appearance_le_20"][2] or 0) >= 0.70,
    "Z9 indirect share median in [0.3,0.9]": 0.3 <= out["hop_indirect_share_median"] <= 0.9,
    "Z10 capital_in_cell<=0.30": out["capital_in_cell"][2] <= 0.30,
    "F_read (view(all)<=0.35)": v_a <= 0.35,
    "F_lens (lens>=view on countries or all)": lc >= v_c or la >= v_a,
    "F_lens_memo (lens>=view+0.10)": lc >= v_c + 0.10 or la >= v_a + 0.10,
    "F_causal (together<0.60 on T_diff, or T_city n>=8)": out["T_diff"]["together"][2] < 0.60 or (
        len(tc) >= 8 and out["T_city"]["together"][2] < 0.60),
}
json.dump(out, open(sys.argv[1].replace(".jsonl", "-summary.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
