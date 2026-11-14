import sys, torch; torch.set_num_threads(4)
from pathlib import Path
from mdb.engine.models import load_local
from mdb.ops.debugger import attach
from mdb.interop import circuit_tracer_graph as ctg
S = Path("/private/tmp/claude-501/-Users-jakeholl-domains/f9daae27-47d0-4118-8855-2c7a6f58eecf/scratchpad/sc")
g = ctg.read_graph(S / "tiny-graph-plain.pt")
tc = torch.load(S / "tiny-tc.pt")
dbg = attach(load_local(S / "tiny-qwen3", dtype="fp32", device="cpu"), None)
ids = [int(t) for t in g["input_tokens"]]
run = dbg.run(ids, prefill=0)
run.until(f"t{len(ids)-1}.logits")
exact = ctg.direct_effects(run, g, tc)
A = g["adjacency_matrix"].double()
named = ctg.nodes(g)
kinds = [n["kind"] for n in named]
print("all", ctg.compare(A, exact))
for sk in ("feature", "error", "token"):
    for dk in ("feature", "logit"):
        rows = [i for i,k in enumerate(kinds) if k == dk]; cols = [j for j,k in enumerate(kinds) if k == sk]
        sub_a, sub_e = A[rows][:, cols], exact[rows][:, cols]
        if (sub_a.abs() + sub_e.abs()).sum() == 0: print(sk, "->", dk, "empty"); continue
        c = ctg.compare(sub_a, sub_e)
        print(sk, "->", dk, c["edges"], "maxgap/maxedge %.3g" % c["max_gap_over_max_edge"]["value"], "r %.6f" % c["pearson_r"]["value"], "norm ratio %.6f" % c["norm_ratio"]["value"], "sign", c["sign_agree"])
e = ctg.graph_edges(g, min_abs=0)
print("edges", len(e), e[0]["edge"], e[0]["src"]["id"], e[0]["dst"]["id"], e[0]["weight"])
