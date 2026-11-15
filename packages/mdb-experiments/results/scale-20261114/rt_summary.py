"""EXPLORATORY summary of out/rt-*.json (retrace counts by model size)."""
import json
import statistics as st
from pathlib import Path

ORDER = ["Qwen2.5-0.5B", "Qwen3-0.6B-Base", "Qwen2.5-1.5B", "Qwen3-1.7B-Base", "Qwen2.5-3B", "Qwen3-4B-Base"]


def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else None


def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x) if x else None


print("| Model | Layers | Carrier switches L0..L_c, mean (correct / wrong) | After first true country | Answer-lens switches all layers | Answer-lens switches ≥ L_c | Answer settles at (fraction of depth), median |")
print("|---|---|---|---|---|---|---|")
for m in ORDER:
    p = Path(f"out/rt-{m}.json")
    if not p.exists():
        continue
    d = json.load(open(p))
    R = list(d["items"].values())
    c = [r for r in R if r["correct"]]
    w = [r for r in R if not r["correct"]]
    g = lambda rs, k: mean([r[k] for r in rs])
    print(f"| {m} | {d['layers']} | {g(c, 'carrier_switches'):.1f} / {g(w, 'carrier_switches'):.1f} | "
          f"{g(c, 'carrier_switches_after_first_true'):.1f} / {g(w, 'carrier_switches_after_first_true'):.1f} | "
          f"{g(c, 'answer_switches'):.1f} / {g(w, 'answer_switches'):.1f} | {g(c, 'answer_switches_after_Lc'):.1f} / {g(w, 'answer_switches_after_Lc'):.1f} | "
          f"{med([r['answer_settle_frac'] for r in c]):.2f} / {med([r['answer_settle_frac'] for r in w]):.2f} |")
