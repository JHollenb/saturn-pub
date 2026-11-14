import json, math, sys
from mdb.analysis.stability import calibrate
D = "results/showcase-20261114/exp6/out"
def sp(x):
    x = abs(float(x)); return 2.0 ** (math.floor(math.log2(x)) - 7) if x else 2.0 ** -133
for model in ["Qwen3-0.6B-Base", "Qwen2.5-0.5B"]:
    R = [json.loads(l) for l in open(f"{D}/exp6-{model}.jsonl")]
    def pairs(ref, oth):
        return [(r[ref][k]-r[ref][0], r[oth][k]-r[oth][0], 2*sp(r["max_abs_logit"])) for r in R for k in range(1, len(r["mdb"]))]
    allp = []
    for ref, oth, name in [("mdb","U16","kernel"),("mdb","P16","total"),("U16","P16","processing"),("U16_b1","U16","batch-composition"),("mdb","U16_b1","stepped-vs-TL-b1")]:
        c = calibrate(pairs(ref, oth), model=model, paths=(ref, oth), source="exp6")
        print(model, name, f"k={c.k:.3f}", f"lost {c.lost}/{c.above_floor}", "rev", c.sign_reversed, "gained", c.gained, [(t["bin"], t["n"], t["lost"]) for t in c.table])
    c = calibrate(pairs("mdb","U16")+pairs("mdb","P16")+pairs("mdb","U16_b1"), model=model, paths=("mdb-stepped","tl-batched-unprocessed","tl-batched-processed","tl-batch1-unprocessed"), source="results/showcase-20261114/exp6/out")
    print(model, "COMBINED mdb-ref", f"k={c.k:.4f}", c.lost, c.above_floor)
    c.save(f"/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/sc/calib-{model}.json")
