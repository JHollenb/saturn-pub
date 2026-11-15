"""Audit: replica / first-order ratios from the sealed Job F outputs (out/fc-*.json, .pt) against the
native A1 sweep (audit/data/sw-*.pt), my own join; per base family."""
import json, statistics as st, torch
P = json.load(open("panel.json")); fam = {i["id"]: i["family"] for i in P["items"]}
FC = {"Qwen2.5-0.5B": ["fc-Qwen2.5-0.5B"], "Qwen3-0.6B-Base": ["fc-Qwen3-0.6B-Base"], "Qwen2.5-1.5B": ["fc-Qwen2.5-1.5B"],
      "Qwen3-1.7B-Base": ["fc-Qwen3-1.7B-Base"], "Qwen2.5-3B": ["fc-Qwen2.5-3B-a", "fc-Qwen2.5-3B-b"], "Qwen3-4B-Base": ["fc-Qwen3-4B-Base-a", "fc-Qwen3-4B-Base-b"]}
SW = {m: [p.replace("fc-", "sw-") for p in v] for m, v in FC.items()}
out = {}
for m in FC:
    fj, ft, sw = {}, {}, {}
    for p in FC[m]:
        fj.update(json.load(open(f"out/{p}.json"))); ft.update(torch.load(f"out/{p}.pt", weights_only=False))
    for p in SW[m]:
        sw.update(torch.load(f"audit/data/{p}.pt", weights_only=False))
    r = {}
    for f in ("currency", "person", "landmark", "city"):
        rep, fo_c, rel = [], [], []
        for iid in [i for i in fj if fam[i] == f]:
            s = sw[iid]
            idx = {tuple(c): n for n, c in enumerate(s["cells"].tolist())}
            for a in fj[iid]["arms"]:
                nat = float(s["dM"][idx[tuple(a["cell"])]])
                if nat:
                    rep.append(a["replica_dM"] / nat)
                    rel.append(abs(a["replica_dM"] - nat))
            j = int(s["dM"].argmin())
            if float(s["dM"][j]):
                fo_c.append(float(ft[iid]["fo"][j]) / float(s["dM"][j]))
        r[f] = {"replica_over_native_med": round(st.median(rep), 3), "replica_abs_err_med": round(st.median(rel), 3),
                "replica_ratio_range": [round(min(rep), 2), round(max(rep), 2)], "fo_over_native_at_carrier_med": round(st.median(fo_c), 3)}
    r["null_err_max"] = round(max(v["replica_null_err"] for v in fj.values()), 3)
    out[m] = r
    print(m, json.dumps(r))
json.dump(out, open("audit/a4_forecast.json", "w"), indent=1)
