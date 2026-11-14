"""Score exp6 (prereg exp6/prereg.json).  python score.py ITEMS_JSONL [ITEMS_JSONL ...] OUT_JSON"""
import json
import math
import sys


def sp(x):
    x = abs(float(x))
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x else 2.0 ** -133


out = {}
for path in sys.argv[1:-1]:
    R = [json.loads(line) for line in open(path)]
    model = path.split("exp6-")[1].replace(".jsonl", "")
    r1 = sum(1 for r in R if abs(r["P32"][0] - r["U32"][0]) <= 1e-4)
    r1_max = max(abs(r["P32"][0] - r["U32"][0]) for r in R)
    base_ok = sum(1 for r in R if abs(r["P16"][0] - r["mdb"][0]) <= 2 * sp(r["max_abs_logit"]))
    pe, sp_units, flips_eff, n_eff, flips_small, n_small = 0, [], 0, 0, 0, 0
    tot = 0
    for r in R:
        s = sp(r["max_abs_logit"])
        fl = 2 * s
        for k in range(1, len(r["mdb"])):
            nat = r["mdb"][k] - r["mdb"][0]
            pro = r["P16"][k] - r["P16"][0]
            unp = r["U16"][k] - r["U16"][0]
            tot += 1
            pe += abs(pro - nat) <= fl
            sp_units.append(abs(pro - unp) / s)
            if abs(nat) > fl:
                n_eff += 1
                flips_eff += (pro * nat < 0) or abs(pro) <= fl
            else:
                n_small += 1
                flips_small += (pro * nat < 0)
    sp_units.sort()
    p95 = sp_units[int(0.95 * (len(sp_units) - 1))]
    kern = sorted(abs((r["U16"][k] - r["U16"][0]) - (r["U16_b1"][k] - r["U16_b1"][0])) / sp(r["max_abs_logit"])
                  for r in R for k in range(1, len(r["mdb"])))
    parity = sum(1 for r in R if r["U16_b1"][0] == r["mdb"][0])
    out[model] = {
        "items": len(R),
        "R1": {"fp32_base_within_1e-4": f"{r1}/{len(R)}", "max_abs": r1_max, "holds": r1 == len(R)},
        "R2": {"base_within_floor": f"{base_ok}/{len(R)}", "patch_within_floor": f"{pe}/{tot}",
               "p95_processed_minus_unprocessed_spacings": p95,
               "holds": base_ok / len(R) >= 0.9 and pe / tot >= 0.9},
        "R3": {"effects_flipped_or_lost": f"{flips_eff}/{n_eff}", "below_floor_sign_changed": f"{flips_small}/{n_small}",
               "holds": n_eff > 0 and flips_eff / n_eff <= 0.05 and (n_small == 0 or flips_small / n_small >= 0.3)},
        "R4": {"fp32_items_with_pattern_change_ge_0.1": sum(1 for r in R if r["pattern_heads_ge_0.1"]["fp32"]),
               "bf16_items_with_pattern_change_ge_0.1": sum(1 for r in R if r["pattern_heads_ge_0.1"]["bf16"]),
               "max_pattern_diff_fp32": max(r["pattern_maxdiff"]["fp32"] for r in R),
               "max_pattern_diff_bf16": max(r["pattern_maxdiff"]["bf16"] for r in R),
               "holds": all(not r["pattern_heads_ge_0.1"]["fp32"] for r in R)},
        "nulls": {"U16_batch1_base_bitwise_eq_mdb": f"{parity}/{len(R)}",
                  "kernel_path_p95_spacings": kern[int(0.95 * (len(kern) - 1))],
                  "kernel_path_max_spacings": kern[-1]},
        "falsifier_fires": p95 > 4,
        "claims_judged_native_effect": sum(1 for r in R for j in r["judged"] if j["native_effect"]),
    }
json.dump(out, open(sys.argv[-1], "w"), indent=1)
print(json.dumps(out, indent=1))
