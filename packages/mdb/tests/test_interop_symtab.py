"""Foreign names of symbol-table rows, measured (review 2026-11-14, 2.8 and 2.12).

The cell / row / read names ``interop.names`` attaches to every binding are checked by reading
the same value through TransformerLens (bridge, unprocessed) and nnsight, indexed exactly as
the name says, against mdb's cache cell and attention weight on tiny Qwen2/Qwen3: TL and
nnsight agree bitwise (two full forwards); mdb agrees within float32 rounding (its prefill
batch and stepped row round differently from a full forward, issue M10). The feature-binding
section's SAELens ids are checked by building the same dictionary in SAELens and comparing its
top-k code with the component activations. Skips unless transformer_lens 4.x, nnsight 0.8 and
sae_lens 6.x import and a Qwen tokenizer is cached (run in the scratch interop venv).
"""
from __future__ import annotations

import json
import warnings

import pytest

torch = pytest.importorskip("torch")
TOKENIZERS = {"qwen2": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3": "Qwen/Qwen3-0.6B"}
TOKENS = [1, 5, 9, 3, 7, 2]


def _tokenizer(family):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOKENIZERS[family], local_files_only=True)
    except OSError:
        pytest.skip(f"no cached tokenizer {TOKENIZERS[family]}")


def _index(tensor, text):
    """Apply an index text such as ``[:, 2, 0, :]`` (from a name table) to ``tensor``."""
    return eval("t" + text, {"__builtins__": {}}, {"t": tensor})  # noqa: S307 (test-only)


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_cell_and_read_names_index_the_same_values_in_tl_and_nnsight(family):
    pytest.importorskip("transformer_lens")
    pytest.importorskip("nnsight")
    import mdb
    from mdb.analysis import symbols
    from mdb.analysis.forecast import Margin
    from mdb.engine.models import tiny_model
    from mdb.interop import names, nn, tl

    warnings.filterwarnings("ignore")
    model = tiny_model(family, "fp32", layers=2)
    run = mdb.attach(model).run(TOKENS)
    run.until("t5.logits")
    symbols.declare(run, "cell", ["cache.L1.v[kv0,row=2]"])
    tab = symbols.table(run, consumer=Margin(5, 4, 9), foreign=True)
    ids = torch.tensor([TOKENS])
    _, cache = tl.bridge(model, _tokenizer(family)).run_with_cache(ids)
    nm = nn.wrap(model)
    dh = run.adapter.head_dim
    saved = {}
    with torch.no_grad(), nm.trace(ids):
        for layer in range(2):
            saved[layer] = nm.model.layers[layer].self_attn.v_proj.output.save()
            saved[("p", layer)] = nm.model.layers[layer].self_attn.output[1].save()
    checked = 0
    for layer in range(2):
        v = run.read_cache(layer, "v", staged=False)
        k = run.read_cache(layer, "k", staged=False)
        for g in range(run.adapter.num_kv_heads):
            for row in range(6):
                for port, mine in (("v", v[0, g, row]), ("k", k[0, g, row])):
                    found = names.cell_names(layer, g, row, port, head_dim=dh)
                    theirs = _index(cache[found["tl"]["hook"]], found["tl"]["index"])[0]
                    # mdb's prefill batch and stepped row round differently from a full
                    # forward (M10): within float32 rounding of the cell
                    assert float((theirs - mine).abs().max()) < 1e-6, (layer, g, row, port)
                    if port == "v":
                        nn_value = _index(saved[layer], found["nnsight"]["index"])[0]
                        assert torch.equal(nn_value, theirs)      # two full forwards: bitwise
                    checked += 1
        p = run.read(f"t5.L{layer}.p")
        for h in range(run.adapter.num_heads):
            found = names.read_names(5, layer, h)
            tl_p = _index(cache[found["tl"]["hook"]], found["tl"]["index"])[0]
            nn_p = _index(saved[("p", layer)], found["nnsight"]["index"])[0]
            assert float((tl_p - p[0, h, 0]).abs().max()) < 1e-6
            assert float((nn_p - p[0, h, 0]).abs().max()) < 1e-6
    assert checked == 2 * run.adapter.num_kv_heads * 6 * 2
    # the names the table attached are the ones checked above
    cell = next(b for b in tab.report["bindings"] if b["symbol"] == "cell")
    assert cell["foreign"]["cells"][0] == names.cell_names(1, 0, 2, "v", head_dim=dh)
    resid = names.residual_names(1, "h_out", row=3)
    assert resid["tl"]["hook"] == "blocks.1.hook_resid_post[:, 3]"


def test_feature_bindings_carry_saelens_ids_that_saelens_agrees_with(tmp_path):
    pytest.importorskip("sae_lens")
    from mdb.analysis import sae, symbols
    from mdb.analysis.forecast import Margin
    from mdb.ops.debugger import open_tiny

    debugger = open_tiny("qwen3", "fp32")
    width, F, K = int(debugger.model.config.hidden_size), 64, 6
    g = torch.Generator().manual_seed(3)
    for layer in range(debugger.adapter.layers):
        torch.save({"W_enc": torch.randn(F, width, generator=g),
                    "b_enc": torch.randn(F, generator=g) * 0.1,
                    "W_dec": torch.randn(width, F, generator=g) / 8,
                    "b_dec": torch.randn(width, generator=g) * 0.1},
                   tmp_path / f"layer{layer}.sae.pt")
    repo = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
    (tmp_path / "config.json").write_text(json.dumps({"k": K, "hook_point": "resid_post",
                                                      "hf_repo_id": repo}))
    basis = sae.open_set("s", str(tmp_path)).get(0)
    run = debugger.run([1, 5, 9, 3, 7], prefill=1)
    run.until("t4.logits")
    cb = symbols.Binding.component(run, "carrier", "cache.L1.v[kv0,row=2]", basis)
    mapping = {"qwen-scope-3-1.7b-base-w32k-l50/layer0": {"model": "qwen3-1.7b-base",
                                                          "source": "0-qwenscope-res-32k"}}
    tab = symbols.table(run, consumer=Margin(4, 4, 9), components=[cb], neuronpedia=mapping)
    from mdb.core.records import verify_record

    verify_record(tab.records[0])
    section = tab.report["components"][0]
    assert section["kind"] == "feature" and section["partition"]["holds"]
    feats = [c for c in section["components"] if "feature" in c]
    assert feats and all(c["saelens"]["release"] == "qwen-scope-3-1.7b-base-w32k-l50"
                         for c in feats)
    assert feats[0]["saelens"]["hook_name"] == "blocks.0.hook_resid_post"
    assert feats[0]["neuronpedia"]["url"].endswith(f"/{feats[0]['feature']}")
    # SAELens on the same dictionary: same feature ids, same activations
    h = run.read("t2.L1.h").reshape(1, -1).float()
    code = basis.to_saelens().encode(h)[0]
    active = {int(i): float(code[i]) for i in code.nonzero().reshape(-1)}
    for c in feats:
        assert c["feature"] in active
        assert c["activation"]["value"] == pytest.approx(active[c["feature"]], rel=1e-5,
                                                         abs=1e-5)


def _tiny_qwen3_run():
    import mdb
    from mdb.engine.models import tiny_model

    model = tiny_model("qwen3", "fp32", layers=2)
    run = mdb.attach(model).run(TOKENS, prefill=0)      # every row stepped: ports at each row
    run.until("t5.logits")
    return model, run


def test_pyvene_specs_collect_the_tensors_mdb_reads():
    """The pyvene spec on a binding row (value cell: head_value_output, unit h.pos; residual:
    model.layers[l].output / block_input, unit pos) collects mdb's value, with the Qwen3 shim."""
    pytest.importorskip("pyvene")
    from mdb.analysis import symbols
    from mdb.analysis.forecast import Margin
    from mdb.interop import names, pyvene_names

    warnings.filterwarnings("ignore")
    model, run = _tiny_qwen3_run()
    assert set(pyvene_names.qwen3_shim()) <= {"Qwen3Model", "Qwen3ForCausalLM",
                                              "Qwen3ForSequenceClassification"}
    assert pyvene_names.qwen3_shim() == []                          # idempotent
    dh = run.adapter.head_dim
    for layer in range(2):
        v = run.read_cache(layer, "v", staged=False)
        for g in range(run.adapter.num_kv_heads):
            for row in (1, 3, 5):
                address = f"cache.L{layer}.v[kv{g},row={row}]"
                got = pyvene_names.collect(model, TOKENS, address).reshape(-1)
                assert float((got - v[0, g, row]).abs().max()) < 1e-6, address
                row_spec = names.cell_names(layer, g, row, "v", head_dim=dh)["pyvene"]
                found = pyvene_names.spec(address)
                assert row_spec["component"] == found["component"]
                assert (row_spec["unit_locations"]["h"], row_spec["unit_locations"]["pos"]) == \
                    found["unit_locations"][0]
        for port in ("h_out", "h"):
            address = f"t3.L{layer}.{port}"
            got = pyvene_names.collect(model, TOKENS, address).reshape(-1)
            mine = run.read(address).reshape(-1)
            assert float((got - mine).abs().max()) < 1e-5, address
            assert names.residual_names(layer, port)["pyvene"]["component"] == \
                pyvene_names.spec(address)["component"]
    assert names.cell_names(0, 0, 1, "k", head_dim=dh)["pyvene"]["state"] == "no counterpart"
    tab = symbols.table(run, consumer=Margin(5, 4, 9), foreign=True)
    from mdb.core.records import verify_record

    verify_record(tab.records[0])


def test_circuit_tracer_node_ids_and_qwen3_locations_match_mdb():
    """mdb's ct_* ids equal circuit-tracer 0.5.0's own Node ids, and the modules its Qwen3
    mapping hooks (mlp.hook_in / mlp.hook_out / embed) hold mdb's n, m and emb values."""
    pytest.importorskip("nnsight")
    try:
        from mdb.interop import circuit_tracer_names as ctn

        ctn.import_circuit_tracer()
    except ImportError as error:
        pytest.skip(f"circuit_tracer does not import: {error}")
    from circuit_tracer.frontend.graph_models import Node

    from mdb.interop import names, nn

    assert names.ct_node(3, 7, 1234)["id"] == Node.feature_node(3, 7, 1234).node_id
    assert names.ct_error_node(3, 7)["id"] == Node.error_node(3, 7).node_id
    assert names.ct_token_node(7, 42)["id"] == Node.token_node(7, 42).node_id
    assert names.ct_logit_node(7, 42, 28)["id"] == Node.logit_node(7, 42, "x", 28).node_id
    warnings.filterwarnings("ignore")
    model, run = _tiny_qwen3_run()
    nm = nn.wrap(model)
    saved = {}
    with torch.no_grad(), nm.trace(torch.tensor([TOKENS])):
        order = [(0, "emb")] + [(layer, port) for layer in range(2) for port in ("n", "m")]
        for layer, port in order:                     # nnsight requests in execution order
            path, side = ctn.qwen3_locations(layer)[port]
            envoy = nm
            for part in path.split("."):
                match = part.split("[")
                envoy = getattr(envoy, match[0])
                if len(match) > 1:
                    envoy = envoy[int(match[1].rstrip("]"))]
            saved[(layer, port)] = getattr(envoy, side).save()
    checked = 0
    for (layer, port), value in saved.items():
        for row in (2, 5):
            address = f"t{row}.emb" if port == "emb" else f"t{row}.L{layer}.{port}"
            mine = run.read(address).reshape(-1)
            theirs = value[0, row].reshape(-1)
            assert float((theirs - mine).abs().max()) < 1e-5, address
            checked += 1
    assert checked == 5 * 2
    node = names.ct_node(1, 5, 9)
    assert node["reads"] == "t5.L1.n" and node["writes"] == "t5.L1.m"
