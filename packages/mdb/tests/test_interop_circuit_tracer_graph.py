"""circuit-tracer graph adapter (SC8): node naming, edges as circuit_map edges, and mdb's exact
direct effects. Completeness on tiny Qwen3 fp32: with every MLP write a node (row 0 kept, b_dec
0) the direct effects into a logit node sum to its demeaned native logit. circuit-tracer itself
is not imported (it needs transformers <= 4.57.3); its agreement with ``direct_effects`` was
measured in a pinned venv (README, SC8)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.circuit_map import EDGE_KINDS  # noqa: E402
from mdb.interop import circuit_tracer_graph as ctg  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3, 7]
FEATURES = 8


def _setup():
    torch.manual_seed(0)
    run = open_tiny("qwen3", "fp32").run(PROMPT, prefill=0)
    last = len(PROMPT) - 1
    run.until(f"t{last}.logits")
    d, layers = run.adapter.hidden_size, run.adapter.layers
    tc = {"W_enc": [torch.randn(FEATURES, d) * d ** -0.5 for _ in range(layers)],
          "W_dec": [torch.randn(FEATURES, d) * 0.1 for _ in range(layers)],
          "b_enc": [torch.zeros(FEATURES) for _ in range(layers)],
          "b_dec": [torch.zeros(d) for _ in range(layers)]}
    active = []
    for layer in range(layers):
        for row in range(len(PROMPT)):
            n = run.read(f"t{row}.L{layer}.n").reshape(-1).double()
            pre = n @ tc["W_enc"][layer].double().T
            active += [[layer, row, f] for f in range(FEATURES) if pre[f] > 0]
    logits = run.read(f"t{last}.logits").reshape(-1).double()
    top = logits.topk(2).indices
    size = len(active) + (layers + 1) * len(PROMPT) + len(top)
    graph = {"adjacency_matrix": torch.zeros(size, size),
             "active_features": torch.tensor(active), "input_tokens": torch.tensor(PROMPT),
             "logit_token_ids": top, "n_layers": layers}
    return run, graph, tc, logits, top


def test_direct_effects_into_a_logit_sum_to_its_demeaned_native_value(tmp_path):
    run, graph, tc, logits, top = _setup()
    exact = ctg.direct_effects(run, graph, tc, zero_rows=0)
    named = ctg.nodes(graph)
    for k, token in enumerate(top.tolist()):
        row = len(named) - len(top) + k
        assert named[row]["kind"] == "logit" and named[row]["token"] == token
        demeaned = float(logits[token] - logits.mean())
        assert float(exact[row].sum()) == pytest.approx(demeaned, rel=1e-5, abs=1e-5)
    # no edge runs backwards: a feature reads only sources written before its layer
    for i, dst in enumerate(named):
        if dst["kind"] != "feature":
            continue
        for j, src in enumerate(named):
            if src["kind"] in ("feature", "error") and src["layer"] >= dst["layer"]:
                assert exact[i, j] == 0


def test_graph_edges_name_nodes_by_mdb_address_and_compare_reads_agreement(tmp_path):
    run, graph, tc, _, _ = _setup()
    exact = ctg.direct_effects(run, graph, tc)
    graph["adjacency_matrix"] = exact.float()
    path = tmp_path / "graph.pt"
    torch.save(graph, path)
    edges = ctg.graph_edges(path)
    assert len(edges) == int((exact.float() != 0).sum())
    assert {e["edge"].kind for e in edges} <= set(EDGE_KINDS)
    for e in edges[:50]:
        assert e["weight"]["authority"] == "supplied"
        if e["dst"]["kind"] == "logit":
            assert e["edge"].kind == "consume" and e["dst"]["address"].endswith("]")
        if e["src"]["kind"] == "feature":
            assert e["src"]["writes"] == f"t{e['src']['position']}.L{e['src']['layer']}.m"
    # row-0 errors are zero, as circuit-tracer's zero_positions makes them
    named = ctg.nodes(graph)
    zero = [j for j, n in enumerate(named) if n["kind"] == "error" and n["position"] == 0]
    assert exact[:, zero].abs().sum() == 0
    agree = ctg.compare(graph["adjacency_matrix"], exact)
    assert agree["pearson_r"]["value"] > 0.999999 and agree["sign_agree"] == agree["edges"]
    assert agree["max_gap_over_max_edge"]["value"] < 1e-6
