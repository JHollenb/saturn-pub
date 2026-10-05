"""Offline tests for the cross-layer-transcoder (CLT) path of the native edge test.

The flagship ran per-layer transcoders (one write layer per feature). A CLT feature reads
at layer ``L`` and writes to layers ``L .. n_layers-1``; ``decoder_writes`` returns one
``(write_layer, direction)`` pair per write layer, and the native edge test must *sum* those
per-layer decoder slices when it re-tests the feature on the real model. The shipped interop
was designed for this (``residual_hook`` and ``native_group_intervention`` already iterate the
write list), but the multi-write path was never exercised by a test. These tests pin it with a
tiny random Gemma-2 adapter and a synthetic multi-write decoder, checking the interop's drop
against an *independent* direct multi-layer HF forward hook (the same consistency oracle the
real-weight smoke uses), and that the sealed CLT bundle re-derives offline, torch-free.
"""

import subprocess
import sys
import types

import pytest
import torch

from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.interop.circuit_tracer import (
    SyntheticAttributionGraph,
    build_synthetic_graph,
    default_edge_rule,
    native_edge_test,
    native_group_intervention,
    select_edges,
    verify_edge_bundle,
)

PROMPT = [5, 7, 11, 3, 9]
LAST = len(PROMPT) - 1
TOL = {"rtol": 1e-4, "atol": 1e-5}


@pytest.fixture
def adapter():
    return DecoderAdapter.tiny("gemma2")


def _clt_decoder(adapter, d_transcoder=8, seed=1):
    """A synthetic CLT: a feature read at layer L writes to every layer L..n_layers-1.

    ``mats[(read_layer, write_layer)]`` holds a distinct [d_transcoder, hidden] matrix, so the
    per-write-layer slices are different and a bug that drops/duplicates a slice is caught.
    """
    g = torch.Generator().manual_seed(seed)
    n = adapter.layers
    mats = {
        (rl, wl): torch.randn(d_transcoder, adapter.hidden_size, generator=g) * 0.4
        for rl in range(n)
        for wl in range(rl, n)
    }

    def decoder_writes(read_layer, feature):
        return [(wl, mats[(read_layer, wl)][feature]) for wl in range(read_layer, n)]

    return decoder_writes, mats


def _direct_multilayer_drop(adapter, tokens, writes, activation, target, *, multiplier=0.0,
                            position=None):
    """Independent oracle: add ``(multiplier-1)*activation*direction`` at each write layer's
    resid_post (the layer module output) at ``position`` (default: last), one forward, grade the
    target-logprob drop. ``multiplier=0`` reproduces the single-feature zero ablation that
    ``_residual_hook_effect`` applies (change = ``-activation*direction``).
    """
    pos = (len(tokens) - 1) if position is None else position
    coeff = (float(multiplier) - 1.0) * float(activation)

    def _target_lp(logits_1d):
        return float(torch.log_softmax(logits_1d.float().reshape(-1), dim=-1)[target])

    def make(direction):
        vec = coeff * direction.to(torch.float32).reshape(-1)

        def hook(_m, _i, output):
            hidden = (output[0] if isinstance(output, tuple) else output).clone()
            hidden[:, pos, :] = hidden[:, pos, :] + vec
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        return hook

    with torch.inference_mode():
        clean = adapter.model(torch.tensor([list(tokens)])).logits[:, -1]
    handles = [adapter._layer_modules[wl].register_forward_hook(make(d)) for wl, d in writes]
    try:
        with torch.inference_mode():
            inter = adapter.model(torch.tensor([list(tokens)])).logits[:, -1]
    finally:
        for h in handles:
            h.remove()
    return _target_lp(clean) - _target_lp(inter)


def test_clt_multi_write_residual_hook_matches_direct_multilayer_hook(adapter):
    decoder_writes, _ = _clt_decoder(adapter)
    # a feature read at layer 0 writes to layers 0..n-1 (>1 write layer) -> cross-layer path.
    graph = build_synthetic_graph(
        n_layers=adapter.layers, input_tokens=PROMPT, features=[(0, LAST, 0, 2.0)]
    )
    with torch.inference_mode():
        target = int(adapter.model(torch.tensor([PROMPT])).logits[0, -1].argmax())
    table = native_edge_test(
        graph, adapter, prompt_tokens=PROMPT, target_token=target,
        decoder_writes=decoder_writes, top_k=5, prompt_text="clt-synthetic",
    )
    row = next(r.to_dict() for r in table.rows if r.node_kind == "feature")
    # a multi-write feature is typed cross-layer and tested with the position-general hook,
    # never the single-boundary carrier.
    assert row["feature_type"] == "cross-layer-transcoder"
    assert row["operator"] == "residual_hook" and row["covered"]
    assert tuple(row["write_layers"]) == tuple(range(adapter.layers))
    # the interop's summed multi-layer drop equals the independent direct multi-layer hook.
    oracle = _direct_multilayer_drop(
        adapter, PROMPT, decoder_writes(0, 0), activation=2.0, target=target, multiplier=0.0
    )
    assert abs(row["native_logit_delta"] - oracle) < 1e-4, (row["native_logit_delta"], oracle)


def test_clt_group_multi_write_matches_direct_hook(adapter):
    decoder_writes, _ = _clt_decoder(adapter)
    feats = [(0, LAST, 0, 2.0), (1, LAST, 1, 1.5)]
    with torch.inference_mode():
        target = int(adapter.model(torch.tensor([PROMPT])).logits[0, -1].argmax())
    members = [
        {"layer": L, "position": p, "feature": f, "activation": a,
         "writes": decoder_writes(L, f)}
        for (L, p, f, a) in feats
    ]
    for mult in (0.0, -2.0):
        grp = native_group_intervention(
            adapter, PROMPT, members, target, name=f"g{mult}", multiplier=mult,
        )
        # independent oracle: sum every member's every write-layer slice at that member's
        # position with coeff (mult-1)*activation.
        def _oracle():
            coeffs = {}
            for m in members:
                for wl, d in m["writes"]:
                    coeffs.setdefault((wl, m["position"]), []).append(
                        (float(mult) - 1.0) * float(m["activation"]) * d.to(torch.float32)
                    )

            def _tlp(lg):
                return float(torch.log_softmax(lg.float().reshape(-1), -1)[target])

            with torch.inference_mode():
                clean = adapter.model(torch.tensor([PROMPT])).logits[:, -1]

            def make(wl):
                adds = [(pos, v) for (w, pos), vs in coeffs.items() if w == wl for v in vs]

                def hook(_m, _i, out):
                    h = (out[0] if isinstance(out, tuple) else out).clone()
                    for pos, v in adds:
                        h[:, pos, :] = h[:, pos, :] + v.reshape(-1)
                    return (h, *out[1:]) if isinstance(out, tuple) else h

                return hook

            layers = sorted({w for (w, _p) in coeffs})
            handles = [adapter._layer_modules[wl].register_forward_hook(make(wl)) for wl in layers]
            try:
                with torch.inference_mode():
                    inter = adapter.model(torch.tensor([PROMPT])).logits[:, -1]
            finally:
                for h in handles:
                    h.remove()
            return _tlp(clean) - _tlp(inter)

        assert abs(grp["native_logit_delta"] - _oracle()) < 1e-4, grp["name"]


def test_select_edges_reads_activation_by_active_index_not_selection_order():
    """Regression for the dose bug: Graph.activation_values is aligned with active_features, not
    with selected_features. When the graph prunes some active features out of the selection, the
    i-th selected feature's activation is activation_values[selected_features[i]] -- indexing by
    the selection order i reads the wrong activation and mis-doses the native ablation.
    """
    n_layers, n_pos, n_logit = 2, len(PROMPT), 1
    # 4 active features with DISTINCT activations; select active indices 2 then 0 (non-identity).
    active = torch.tensor([[0, LAST, 0], [1, LAST, 1], [0, LAST, 2], [1, LAST, 3]], dtype=torch.long)
    activation_values = torch.tensor([10.0, 20.0, 30.0, 40.0])  # aligned with active_features
    selected = torch.tensor([2, 0], dtype=torch.long)
    n_feat = len(selected)
    n_err, n_embed = n_layers * n_pos, n_pos
    n_nodes = n_feat + n_err + n_embed + n_logit
    adj = torch.zeros(n_nodes, n_nodes)
    logit_row = n_nodes - n_logit
    adj[logit_row, 0] = 2.0  # selected[0] (active idx 2) outranks
    adj[logit_row, 1] = 1.0  # selected[1] (active idx 0)
    graph = SyntheticAttributionGraph(
        active_features=active, selected_features=selected, activation_values=activation_values,
        adjacency_matrix=adj, logit_probabilities=torch.ones(n_logit),
        input_tokens=torch.tensor(PROMPT, dtype=torch.long), n_pos=n_pos,
        cfg=types.SimpleNamespace(n_layers=n_layers),
    )
    sel = select_edges(graph, top_k=2)
    e0, e1 = sel.edges
    # top edge is selected[0] = active index 2 -> feature 2, activation_values[2] == 30 (NOT 10)
    assert (e0.feature, e0.activation) == (2, 30.0), (e0.feature, e0.activation)
    assert (e1.feature, e1.activation) == (0, 10.0), (e1.feature, e1.activation)


def test_clt_out_of_range_write_reported_uncovered(adapter):
    n = adapter.layers

    def decoder_writes(read_layer, feature):  # writes one layer PAST the model -> out of range
        return [(read_layer, torch.ones(adapter.hidden_size)),
                (n, torch.ones(adapter.hidden_size))]

    graph = build_synthetic_graph(
        n_layers=n, input_tokens=PROMPT, features=[(0, LAST, 0, 2.0)]
    )
    with torch.inference_mode():
        target = int(adapter.model(torch.tensor([PROMPT])).logits[0, -1].argmax())
    table = native_edge_test(
        graph, adapter, prompt_tokens=PROMPT, target_token=target,
        decoder_writes=decoder_writes, top_k=5,
    )
    row = next(r.to_dict() for r in table.rows if r.node_kind == "feature")
    assert not row["covered"] and row["verdict"] == "uncovered"
    assert "outside the model's layer range" in row["reason"]


_FRESH = """
import sys
from saturn_pub.interop.circuit_tracer import verify_edge_bundle
result = verify_edge_bundle(sys.argv[1])
assert "torch" not in sys.modules, "offline re-derivation must not import torch"
assert result["ok"], result
print("FRESH_OK", result["summary"]["n_covered"])
"""


def test_clt_seal_and_offline_rederivation(adapter, tmp_path):
    decoder_writes, _ = _clt_decoder(adapter)
    feats = [(0, LAST, 0, 2.0), (1, LAST, 1, 1.5)]
    graph = build_synthetic_graph(n_layers=adapter.layers, input_tokens=PROMPT, features=feats)
    with torch.inference_mode():
        target = int(adapter.model(torch.tensor([PROMPT])).logits[0, -1].argmax())
    members = [
        {"layer": L, "position": p, "feature": f, "activation": a,
         "writes": decoder_writes(L, f)}
        for (L, p, f, a) in feats
    ]
    rule = default_edge_rule()
    groups = [
        native_group_intervention(adapter, PROMPT, members, target, name="topk_zero_ablate",
                                   multiplier=0.0, graph_predicted_delta=3.0, decision_rule=rule),
        native_group_intervention(adapter, PROMPT, members, target, name="topk_steer_-2x",
                                   multiplier=-2.0, decision_rule=rule),
    ]
    table = native_edge_test(
        graph, adapter, prompt_tokens=PROMPT, target_token=target,
        decoder_writes=decoder_writes, top_k=5, decision_rule=rule, group_interventions=groups,
    )
    # every feature row is cross-layer typed (2-layer model, feature at layer 0 -> 2 writes)
    feat_rows = [r.to_dict() for r in table.rows if r.node_kind == "feature"]
    assert any(r["feature_type"] == "cross-layer-transcoder" for r in feat_rows)
    sealed = table.seal(tmp_path)
    assert sealed["fingerprint"]
    assert verify_edge_bundle(tmp_path)["ok"]
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH, str(tmp_path)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("FRESH_OK")
