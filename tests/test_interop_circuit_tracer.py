"""Offline tests for the circuit-tracer native edge test.

No circuit_tracer, no checkpoint download: a tiny random Gemma-2 adapter, a synthetic
attribution graph, and a synthetic transcoder decoder matrix. These check the operator
mechanics against HF's own kernels (carrier == resid_post[L], carrier == residual hook),
the frozen-rule verdicts, error-node accounting, and the sealed bundle's offline,
torch-free re-derivation in a fresh process.
"""

import json
import subprocess
import sys

import pytest
import torch

from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.interop.circuit_tracer import (
    build_synthetic_graph,
    default_edge_rule,
    native_edge_test,
    select_edges,
    verify_edge_bundle,
)

PROMPT = [5, 7, 11, 3, 9]
LAST = len(PROMPT) - 1
TOL = {"rtol": 1e-4, "atol": 1e-5}


@pytest.fixture
def adapter():
    return DecoderAdapter.tiny("gemma2")


def _decoder_matrix(adapter, d_transcoder=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    mats = {
        layer: torch.randn(d_transcoder, adapter.hidden_size, generator=g) * 0.5
        for layer in range(adapter.layers)
    }

    def decoder_writes(layer, feature):
        return [(layer, mats[layer][feature])]

    return decoder_writes, mats


def test_select_edges_ranks_by_influence_and_counts_error_nodes(adapter):
    decoder_writes, _ = _decoder_matrix(adapter)
    graph = build_synthetic_graph(
        n_layers=adapter.layers,
        input_tokens=PROMPT,
        features=[(0, LAST, 0, 2.0), (1, LAST, 1, 1.5), (0, 0, 2, 1.0)],
    )
    selection = select_edges(graph, top_k=10)
    # decreasing edge weights -> feature 0 ranks first
    assert [e.feature for e in selection.edges] == [0, 1, 2]
    assert selection.n_error_nodes == adapter.layers * len(PROMPT)
    assert 0.0 < selection.error_node_influence_share < 1.0


def test_carrier_boundary_equals_hf_resid_post(adapter):
    # Saturn's layer:(L+1) carrier (last position) is HF resid_post[L] = the transcoder's write
    # point (feature_output_hook = hook_mlp_out, which flows into resid_post). Capture each layer
    # output directly (output_hidden_states norms its last entry, so hooks are the clean oracle).
    tokens = torch.tensor([PROMPT])
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach().clone()

        return hook

    for layer in range(adapter.layers):
        handles.append(adapter._layer_modules[layer].register_forward_hook(make(layer)))
    try:
        with torch.inference_mode():
            adapter.model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    for layer in range(adapter.layers):
        session = adapter.session(PROMPT)
        session.continue_(1 + (layer + 1))
        assert session.inspect().boundary == f"layer:{layer + 1}"
        torch.testing.assert_close(session.read("hidden")[0, 0], captured[layer][0, -1], **TOL)


def test_native_edge_test_verdicts_and_error_nodes(adapter):
    decoder_writes, _ = _decoder_matrix(adapter)
    graph = build_synthetic_graph(
        n_layers=adapter.layers,
        input_tokens=PROMPT,
        features=[(0, LAST, 0, 2.0), (1, LAST, 1, 1.5), (0, 0, 2, 1.0)],
    )
    with torch.inference_mode():
        target = int(adapter.model(torch.tensor([PROMPT])).logits[0, -1].argmax())
    table = native_edge_test(
        graph,
        adapter,
        prompt_tokens=PROMPT,
        target_token=target,
        decoder_writes=decoder_writes,
        top_k=10,
        prompt_text="synthetic",
    )
    rows = {(_r["node_kind"], _r["layer"], _r["position"], _r["feature"]): _r
            for _r in (r.to_dict() for r in table.rows)}
    # last-position features go through the carrier + arbitrate path with a receipt
    carrier = rows[("feature", 0, LAST, 0)]
    assert carrier["operator"] == "carrier+arbitrate"
    assert carrier["covered"] and carrier["receipt"] is not None
    assert carrier["verdict"] in {"survives", "collapses", "inconclusive"}
    # carrier and residual-hook operators intervene at the same point -> same measured drop
    assert abs(carrier["native_logit_delta"] - carrier["measurements"]["residual_hook_crosscheck_drop"]) < 1e-4
    # an earlier-position feature is covered by the residual hook (any position)
    hook = rows[("feature", 0, 0, 2)]
    assert hook["operator"] == "residual_hook" and hook["covered"]
    # transcoder error nodes are reported, never dropped
    error = rows[("error", None, None, None)]
    assert error["verdict"] == "uncovered" and not error["covered"]
    assert error["measurements"]["n_error_nodes"] == adapter.layers * len(PROMPT)
    summary = table.summary()
    assert summary["n_feature_edges"] == 3 and summary["n_covered"] == 3
    assert summary["error_node_influence_share"] is not None


def test_carrier_ablation_matches_hf_forward_hook(adapter):
    # The Act subtracting act*W_dec at the carrier equals an HF forward hook subtracting the
    # same vector from that layer's residual output at the last position.
    decoder_writes, mats = _decoder_matrix(adapter)
    layer, feature, activation = 0, 0, 2.0
    direction = mats[layer][feature]
    session = adapter.session(PROMPT)
    session.continue_(1 + (layer + 1))
    from saturn_pub import Act

    candidate = session.fork(session.capture())
    candidate.apply(Act.add("hidden", direction.reshape(1, 1, -1), dose=-activation))
    adapter.generate(candidate, 1)
    carrier_logits = candidate.read("logits")[0]

    vec = (-activation) * direction

    def hook(_m, _i, output):
        hidden = (output[0] if isinstance(output, tuple) else output).clone()
        hidden[:, -1, :] = hidden[:, -1, :] + vec
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = adapter._layer_modules[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            hf_logits = adapter.model(torch.tensor([PROMPT])).logits[0, -1]
    finally:
        handle.remove()
    torch.testing.assert_close(carrier_logits, hf_logits, **TOL)


_FRESH = """
import sys, json
from saturn_pub.interop.circuit_tracer import verify_edge_bundle
result = verify_edge_bundle(sys.argv[1])
assert "torch" not in sys.modules, "offline re-derivation must not import torch"
assert result["ok"], result
print("FRESH_OK", result["summary"]["n_covered"])
"""


def test_seal_and_offline_fresh_process_rederivation(adapter, tmp_path):
    decoder_writes, _ = _decoder_matrix(adapter)
    graph = build_synthetic_graph(
        n_layers=adapter.layers,
        input_tokens=PROMPT,
        features=[(0, LAST, 0, 2.0), (1, LAST, 1, 1.5), (0, 0, 2, 1.0)],
    )
    with torch.inference_mode():
        target = int(adapter.model(torch.tensor([PROMPT])).logits[0, -1].argmax())
    table = native_edge_test(
        graph, adapter, prompt_tokens=PROMPT, target_token=target,
        decoder_writes=decoder_writes, top_k=10,
    )
    sealed = table.seal(tmp_path)
    assert sealed["fingerprint"]
    # in-process verify
    result = verify_edge_bundle(tmp_path)
    assert result["ok"], result
    # fresh process, offline, torch-free re-derivation
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH, str(tmp_path)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("FRESH_OK")


def test_verify_detects_tampered_verdict(adapter, tmp_path):
    decoder_writes, _ = _decoder_matrix(adapter)
    graph = build_synthetic_graph(
        n_layers=adapter.layers, input_tokens=PROMPT,
        features=[(0, LAST, 0, 2.0)],
    )
    with torch.inference_mode():
        target = int(adapter.model(torch.tensor([PROMPT])).logits[0, -1].argmax())
    table = native_edge_test(
        graph, adapter, prompt_tokens=PROMPT, target_token=target,
        decoder_writes=decoder_writes, top_k=5,
    )
    table.seal(tmp_path)
    body = json.loads((tmp_path / "edge_verdicts.json").read_text())
    for row in body["rows"]:
        if row["node_kind"] == "feature":
            row["verdict"] = "survives" if row["verdict"] != "survives" else "collapses"
    (tmp_path / "edge_verdicts.json").write_text(json.dumps(body))
    result = verify_edge_bundle(tmp_path)
    assert not result["ok"]


def test_import_is_torch_free():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import saturn_pub.interop.circuit_tracer as m; "
         "assert 'torch' not in sys.modules; assert 'circuit_tracer' not in sys.modules; "
         "print('OK')"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_default_rule_is_frozen():
    rule = default_edge_rule()
    assert rule.metric == "consumer_logprob_drop"
    assert rule.fingerprint == default_edge_rule().fingerprint
