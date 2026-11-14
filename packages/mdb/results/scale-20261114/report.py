"""Markdown tables for tracker entry 10 from out/score-*.json and out/supscore-*.json (no new numbers).
python report.py > report.md"""
import json
from pathlib import Path

ORDER = ["Qwen2.5-0.5B", "Qwen3-0.6B-Base", "Qwen2.5-1.5B", "Qwen3-1.7B-Base", "Qwen2.5-3B", "Qwen3-4B-Base"]
BASE = ["currency", "person", "landmark", "city"]
S = {m: json.load(open(f"out/score-{m}.json")) for m in ORDER if Path(f"out/score-{m}.json").exists()}
U = {m: json.load(open(f"out/supscore-{m}.json")) for m in ORDER if Path(f"out/supscore-{m}.json").exists()}


def fr(x):
    return f"{x[0]}/{x[1]}" if x and x[1] else "–"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def rng(vals):
    v = [x for x in vals if x is not None]
    return "–" if not v else (f"{min(v):.2f}" if min(v) == max(v) else f"{min(v):.2f}–{max(v):.2f}")


def add(fs):
    return [sum(f[0] for f in fs), sum(f[1] for f in fs)]


print("### (a) Carrier per model\n")
print("| Model | Layers | Carrier (L, KV) | L/depth | Tail | Mode over correct base | Carrier names country, correct base | Write window ≤5 layers, all KV (sealed) | Same, carrier KV only (post hoc) |")
print("|---|---|---|---|---|---|---|---|---|")
for m, d in S.items():
    M, F = d["model"], d["families"]
    print(f"| {m} | {M['layers']} | L{M['L_c']} KV{M['kv_c']} | {M['frac_depth']:.2f} | {M['tail']} | {fr(M['mode'])} | "
          f"{fr(add([F[f]['names_country_correct'] for f in BASE]))} | {fr(add([F[f]['write_window_le5_correct'] for f in BASE]))} | "
          f"{fr(add([F[f]['write_window_kv_le5_correct_posthoc'] for f in BASE]))} |")
print("\n### (b) Accuracy, hop budget, readable errors\n")
print("| Model | d2 | d3-lm | d3-per | d4 | d5 (supp) | d6 (supp) | Slack AUC d3+d4 | Median L_first, correct d2 / d3-lm / d3-per / d4 (chance) | Readable-error share of wrong items, d4 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m, d in S.items():
    F, M = d["families"], d["model"]
    u = U.get(m, {}).get("families", {})
    lf = " / ".join(f"{F[f]['L_first']['med_correct']} ({F[f]['L_first']['med_fake_correct']})" for f in ("d2", "d3-lm", "d3-per", "d4-work"))
    print(f"| {m} | {fr([F['d2']['correct'], F['d2']['n']])} | {fr([F['d3-lm']['correct'], F['d3-lm']['n']])} | "
          f"{fr([F['d3-per']['correct'], F['d3-per']['n']])} | {fr([F['d4-work']['correct'], F['d4-work']['n']])} | "
          f"{fr([u['d5']['correct'], u['d5']['n']]) if 'd5' in u else '–'} | {fr([u['d6']['correct'], u['d6']['n']]) if 'd6' in u else '–'} | "
          f"{f2(M['B2_auc_slack_d3d4'])} | {lf} | {fr(F['d4-work']['readable_error_share_wrong'])} |")
print("\n### (b) Repair and downstream necessity\n")
print("| Model | Back-patch repairs, wrong d3/d4, country late | Same, country never | B6 median margin drop, language (depth) | B6, capital (base) | Projected language names truth, depth | Shuffled-country chance |")
print("|---|---|---|---|---|---|---|")
for m, d in S.items():
    F, M = d["families"], d["model"]
    late = add([F[f]["backpatch"]["late"] for f in ("d3-lm", "d3-per", "d4-work")])
    never = add([F[f]["backpatch"]["never"] for f in ("d3-lm", "d3-per", "d4-work")])
    print(f"| {m} | {fr(late)} | {fr(never)} | {f2(M['B6']['language_drop_median'])} | {f2(M['B6']['capital_drop_median'])} | "
          f"{fr(M['projected_language_hits'])} | {fr(M['projected_language_shuffled_hits'])} |")
print("\n### (c) Hop-2 split of the carrier transplant (correct base items)\n")
print("| Model | Tail | MLP share, dose 0.5 median (IQR) | MLP share, dose 1.0 median (IQR) | Direct share, dose 1.0 | Re-read, dose 1.0 | Median dose move of the MLP share |")
print("|---|---|---|---|---|---|---|")
for m, d in S.items():
    M = d["model"]
    sp = M["split"]
    g = lambda dose, k: sp.get(dose, {}).get(k, {})
    iq = lambda x: "" if not x.get("iqr") else f" ({x['iqr'][0]:.2f}–{x['iqr'][1]:.2f})"
    print(f"| {m} | {M['tail']} | {f2(g('0.5', 'mlp').get('median'))}{iq(g('0.5', 'mlp'))} | {f2(g('1.0', 'mlp').get('median'))}{iq(g('1.0', 'mlp'))} | "
          f"{f2(g('1.0', 'direct').get('median'))} | {f2(g('1.0', 'reread').get('median'))} | {f2(M['split_dose_move_median'])} |")
print("\n### (d) Concentration and family stability (correct items, per family)\n")
print("| Model | Carrier agreement: currency / person / landmark / city | Top cell over V band, median by base family | Top-two share | Top two adjacent, same row |")
print("|---|---|---|---|---|")
for m, d in S.items():
    F = d["families"]
    print(f"| {m} | {' / '.join(fr(F[f]['carrier_agree_correct']) for f in BASE)} | "
          f"{' / '.join(f2(F[f]['concentration_correct']['median']) for f in BASE)} | {rng([F[f]['top2_share_correct'] for f in BASE])} | "
          f"{fr(add([F[f]['top2_adjacent_same_row'] for f in BASE]))} |")
print("\n### (e) Hard prompts: where the error is\n")
print("| Model | dist acc | dist wrong: true carrier names the true country | dist wrong: distractor cell names its country | dist wrong: reader mass on distractor > true | dist wrong: repair by true cell | obsc acc | obsc wrong: readable error |")
print("|---|---|---|---|---|---|---|---|")
for m, d in S.items():
    F = d["families"]
    dw = F["dist-city"].get("dist_wrong", {})
    print(f"| {m} | {fr([F['dist-city']['correct'], F['dist-city']['n']])} | {fr(dw.get('true_carrier_names_true'))} | "
          f"{fr(dw.get('distractor_names_its'))} | {fr(dw.get('mass_distractor_gt_true'))} | {fr(dw.get('repair'))} | "
          f"{fr([F['obsc-city']['correct'], F['obsc-city']['n']])} | {fr(F['obsc-city']['readable_error_share_wrong'])} |")
print("\n### Forecasts (per base family: range over the four families)\n")
print("| Model | First-order Spearman vs native | Shuffled 95th pct | Passive proposal hit@1 | First-order / native at the carrier | Replica re-forward / native | Replica null error, max logit |")
print("|---|---|---|---|---|---|---|")
for m, d in S.items():
    F = d["families"]
    print(f"| {m} | {rng([F[f]['forecast']['fo']['spearman_med'] for f in BASE])} | {rng([F[f]['forecast']['fo']['shuffle95_med'] for f in BASE])} | "
          f"{fr(add([F[f]['forecast']['prop']['hit1'] for f in BASE]))} | {rng([F[f]['forecast']['fo_over_native']['median'] for f in BASE])} | "
          f"{rng([F[f]['forecast']['replica_over_native']['median'] for f in BASE])} | {rng([F[f]['forecast']['replica_null_err_max'] for f in BASE])} |")
print("\n### Co-readouts at the carrier (capital, all base items; language, all depth items)\n")
print("| Model | Capital raw | Capital, country projected out | Lexical null (country direction alone) | Capital counts (projected and not lexical) | Language raw | Language projected | Language lexical null | Language counts |")
print("|---|---|---|---|---|---|---|---|---|")
for m, d in S.items():
    F = d["families"]
    c = lambda fams, sk, k: fr(add([F[f]["co"][sk][k] for f in fams if sk in F[f]["co"]]))
    dep = ("d2", "d3-lm", "d3-per", "d4-work")
    print(f"| {m} | {c(BASE, 'cities', 'raw')} | {c(BASE, 'cities', 'projected')} | {c(BASE, 'cities', 'lexical_null')} | {c(BASE, 'cities', 'counts')} | "
          f"{c(dep, 'languages', 'raw')} | {c(dep, 'languages', 'projected')} | {c(dep, 'languages', 'lexical_null')} | {c(dep, 'languages', 'counts')} |")
print("\n### Site nulls (all items of the base families; truth = the item's country)\n")
print("| Model | Carrier | Fixed last span row | Mass row | Random span row (expectation) | Shuffled cell | Cross-item | Lens at L_c | Lens at L_c+4 |")
print("|---|---|---|---|---|---|---|---|---|")
for m, d in S.items():
    F, M = d["families"], d["model"]
    n = lambda k: fr(add([F[f]["nulls"][k] for f in BASE]))
    print(f"| {m} | {n('carrier_names')} | {n('fixed_last_span')} | {n('mass_row')} | {sum(F[f]['nulls']['random_span_row_expect'] for f in BASE):.1f}/{sum(F[f]['n'] for f in BASE)} | "
          f"{n('shuffled_cell')} | {n('cross_item')} | {n('lens_L' + str(M['L_c']))} | {n('lens_L' + str(M['L_c'] + 4)) if M['L_c'] + 4 < M['layers'] else '–'} |")
print("\n### Closure witness (base families)\n")
print("| Model | Same-country mean cell keeps the answer | Donor-country mean cell moves the answer to the donor's |")
print("|---|---|---|")
for m, d in S.items():
    F = d["families"]
    print(f"| {m} | {fr(add([F[f]['closure']['same_keeps'] for f in BASE]))} | {fr(add([F[f]['closure']['donor_moves'] for f in BASE]))} |")
if U:
    print("\n### Supplement: d5/d6 and native-name cities (carrier frozen from the main panel)\n")
    print("| Model | d5 | d5-short | d6 | native-city | english-city | Readable-error share of wrong items: d5 / d6 / native | Carrier names country, correct: native / english | Median L_first, correct d5 / d6 (chance) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m, d in U.items():
        F = d["families"]
        a = lambda f: fr([F[f]["correct"], F[f]["n"]])
        print(f"| {m} | {a('d5')} | {a('d5-short')} | {a('d6')} | {a('native-city')} | {a('english-city')} | "
              f"{fr(F['d5']['readable_error_share_wrong'])} / {fr(F['d6']['readable_error_share_wrong'])} / {fr(F['native-city']['readable_error_share_wrong'])} | "
              f"{fr(F['native-city']['names_country_correct'])} / {fr(F['english-city']['names_country_correct'])} | "
              f"{F['d5']['L_first']['med_correct']} ({F['d5']['L_first']['med_fake_correct']}) / {F['d6']['L_first']['med_correct']} ({F['d6']['L_first']['med_fake_correct']}) |")
