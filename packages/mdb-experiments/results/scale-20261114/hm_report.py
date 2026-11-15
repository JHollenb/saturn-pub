"""Tables for the entry-10 'Hop-math arms' block from out/hma-*.json (no new numbers)."""
import json
from pathlib import Path

M = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"]
REF = {"Qwen3-0.6B-Base": "17/48", "Qwen3-1.7B-Base": "30/48", "Qwen3-4B-Base": "15/48"}
D = {m: json.load(open(f"out/hma-{m}.json")) for m in M if Path(f"out/hma-{m}.json").exists()}


def fr(x):
    return f"{x[0]}/{x[1]}" if x and x[1] else "–"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


print("| Model | Donor-moves, native (entry 10) / replica | Readable stratum: cell repair | band repair | Right-operand stratum: cell | band | Excluded (no correct same-country item) |")
print("|---|---|---|---|---|---|---|")
for m, d in D.items():
    o = d["oprep"]
    print(f"| {m} | {REF[m]} / {fr(o['replica_donor_moves_base'])} | {fr(o['readable.all']['cell'])} | {fr(o['readable.all']['band'])} | "
          f"{fr(o['right_operand.all']['cell'])} | {fr(o['right_operand.all']['band'])} | {o['excluded']} |")
print("\nBy depth (readable / right-operand, cell arm):\n")
print("| Model | d3 | d4 | d5 |")
print("|---|---|---|---|")
for m, d in D.items():
    o = d["oprep"]
    print(f"| {m} | " + " | ".join(f"{fr(o[f'readable.{f}']['cell'])} / {fr(o[f'right_operand.{f}']['cell'])}" for f in ("d3", "d4", "d5")) + " |")
print("\n| Model | Arm 1 move: language / capital | Direct share 1.0: language / capital | MLP share: language / capital | Re-read: language / capital | MLP dose ratio: language / capital | Arm 3 margin ratio: language / capital | Arm 4 complement names: language / capital | Arm 4 complement margin move: language / capital | Any 2x language layer (arm 2) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m, d in D.items():
    b, l = d["h3"]["base"], d["h3"]["depth"]
    s = lambda t, k: t["a1_shares_1.0"][k]
    print(f"| {m} | {fr(l['a1_moves'])} / {fr(b['a1_moves'])} | {f2(s(l, 'direct'))} / {f2(s(b, 'direct'))} | {f2(s(l, 'mlp'))} / {f2(s(b, 'mlp'))} | "
          f"{f2(s(l, 'reread'))} / {f2(s(b, 'reread'))} | {f2(l['a1_dose_ratio']['mlp'])} / {f2(b['a1_dose_ratio']['mlp'])} | {f2(l['a3_ratio'])} / {f2(b['a3_ratio'])} | "
          f"{fr(l['a4_comp_names'])} / {fr(b['a4_comp_names'])} | {f2(l['a4_comp_move'])} / {f2(b['a4_comp_move'])} | "
          f"{'yes: ' + ', '.join(k for k, v in d['a2_asymmetry'].items() if v['lang_2x']) if d['a2_any_2x'] else 'no'} |")
print("\nPer-layer MLP term of the arm-1 ladder (median share of the native delta), language / capital:\n")
for m, d in D.items():
    lc = d["L_c"]
    l = d["h3"]["depth"]["a1_mlp_by_layer_1.0_median"]
    b = d["h3"]["base"]["a1_mlp_by_layer_1.0_median"]
    print(f"- {m}: " + ", ".join(f"L{lc + i} {x:+.2f}/{y:+.2f}" for i, (x, y) in enumerate(zip(l, b))))
