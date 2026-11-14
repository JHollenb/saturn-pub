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
