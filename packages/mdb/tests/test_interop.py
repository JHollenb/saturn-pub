"""mdb <-> TransformerLens / nnsight interop on tiny random CPU models.

The names test needs nothing; the rest skip unless ``transformer_lens`` (4.x) and ``nnsight``
(0.8) import and a Qwen tokenizer is in the local Hugging Face cache (TransformerBridge needs a
tokenizer object; token ids are passed directly, so its vocabulary does not matter).
"""
from __future__ import annotations

import warnings

import pytest

from mdb.interop import names


def test_name_table_round_trips():
    for spec in names.NAMES:
        address = f"L7.{spec.port}"
        if spec.tl is not None:
            assert names.from_tl(names.to_tl(address)) == address
            for alias in spec.tl_aliases:
                assert names.from_tl(alias.format(l=7)) == address
        else:
            with pytest.raises(names.NoName):
                names.to_tl(address)
        if spec.nn is not None:
            assert names.from_nnsight(names.to_nnsight(address)) == address
    for glob in names.GLOBAL_NAMES:
        assert names.from_tl(names.to_tl(glob)) == glob
        assert names.from_nnsight(names.to_nnsight(glob)) == glob
    assert names.to_tl("L22.A") == "blocks.22.attn.hook_z"
    assert names.to_nnsight("L22.A") == "model.model.layers[22].self_attn.o_proj.input"
    with pytest.raises(names.NoName):
        names.from_tl("blocks.3.ln1.hook_scale")


torch = pytest.importorskip("torch")
TOKENIZERS = {"qwen2": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3": "Qwen/Qwen3-0.6B"}


def _tools():
    pytest.importorskip("transformer_lens")
    pytest.importorskip("nnsight")


def _tokenizer(family):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOKENIZERS[family], local_files_only=True)
    except OSError:
        pytest.skip(f"no cached tokenizer {TOKENIZERS[family]}")


def _model(family, dtype, layers=2):
    from mdb.engine.models import tiny_model

    return tiny_model(family, dtype, layers=layers)


def _all_ports(model):
    from mdb.engine.ports import program_for

    program = program_for(model.config.model_type)
    return [f"L{l}.{p}" for l in range(model.config.num_hidden_layers) for p in program.names]


TOKENS = [[1, 5, 9, 3, 7, 2], [2, 6, 10, 4, 8, 11]]


@pytest.mark.parametrize("family,dtype", [("qwen2", "fp32"), ("qwen2", "bf16"),
                                          ("qwen3", "bf16")])
def test_parity_trace_tl_nnsight_bitwise(family, dtype):
    """Every mapped port: mdb.trace == TL bridge (unprocessed) == nnsight, bit for bit."""
    _tools()
    import mdb
    from mdb.interop import nn, tl

    warnings.filterwarnings("ignore")
    model = _model(family, dtype)
    ids = torch.tensor(TOKENS)
    ports = _all_ports(model)
    with mdb.trace(model, ports=ports) as t, torch.no_grad():
        reference = model(ids).logits
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)
    exported = tl.trace_cache(t, head_dim=head_dim)
    bridge = tl.bridge(model, _tokenizer(family))
    logits, cache = bridge.run_with_cache(ids)
    assert torch.equal(logits, reference)
    compared = 0
    for hook, value in exported.items():
        if hook.endswith("hook_attn_scores"):
            continue          # masks differ: see test_scores_mask_convention
        assert torch.equal(cache[hook], value), hook
        compared += 1
    assert compared >= 15 * model.config.num_hidden_layers
    nm = nn.wrap(model)
    readable = [a for a in ports if names.BY_PORT[a.split(".")[1]].nn is not None]
    got = nn.read(nm, TOKENS, readable, rows=[len(TOKENS[0]) - 1])
    last = len(TOKENS[0]) - 1
    for address in readable:
        port = address.split(".")[1]
        mine = names.to_mdb(_trace_layout(port), t[address], last)
        assert torch.equal(got[f"t{last}.{address}"], mine), address


def _trace_layout(port):
    spec = names.BY_PORT[port]
    return {"rot_q": "bh", "rot_k": "bh_k"}.get(spec.tl_layout, spec.nn_layout)


def test_scores_mask_convention():
    """mdb's s replica adds HF's additive mask (finfo.min); TL hook_attn_scores holds the same
    unmasked entries. Measured: equal on the causal part."""
    _tools()
    import mdb
    from mdb.interop import tl

    warnings.filterwarnings("ignore")
    model = _model("qwen2", "fp32")
    ids = torch.tensor(TOKENS)
    with mdb.trace(model, ports=["L1.s"]) as t, torch.no_grad():
        model(ids)
    _, cache = tl.bridge(model, _tokenizer("qwen2")).run_with_cache(ids)
    mine, theirs = t["L1.s"], cache["blocks.1.attn.hook_attn_scores"]
    causal = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).tril()
    assert torch.equal(mine[..., causal], theirs[..., causal])


def test_stepped_row_against_full_forward():
    """mdb's stepped run (prefill + one row) against a TL full forward, fp32: recorded, within
    float32 rounding (bitwise equality is not expected, issue M10)."""
    _tools()
    import mdb
    from mdb.interop import tl

    warnings.filterwarnings("ignore")
    model = _model("qwen2", "fp32")
    tokens = TOKENS[0]
    _, cache = tl.bridge(model, _tokenizer("qwen2")).run_with_cache(torch.tensor([tokens]))
    values = tl.cache_values(cache, [len(tokens) - 1])
    run = mdb.attach(model).run(tokens)
    run.until("logits")
    worst = 0.0
    for key, value in values.items():
        stepped = run.read(key)
        assert stepped.shape == value.shape, key
        worst = max(worst, float((stepped.double() - value.double()).abs().max()))
    assert worst < 1e-5


def _debugger(family="qwen2", dtype="fp32"):
    import mdb

    model = _model(family, dtype)
    return model, mdb.attach(model)


def test_tl_cache_import_as_origin_matches_named_run():
    """A cell transplant from a TL cache equals the same transplant from a live donor run."""
    _tools()
    from mdb.interop import tl, verify

    warnings.filterwarnings("ignore")
    model, dbg = _debugger()
    donor, tokens = TOKENS[1], TOKENS[0]
    _, cache = tl.bridge(model, _tokenizer("qwen2")).run_with_cache(torch.tensor([donor]))
    imported = tl.import_cache(dbg, "tlrome", cache, donor)
    assert imported.provenance["processed"] is False
    dbg.run(donor, name="rome")
    run = dbg.run(tokens)
    claims = [verify.Claim("cache.L1.v[kv0,row=2]", f"{o}:cache.L1.v[kv0,row=2]",
                           consumer="M[4,9]") for o in ("tlrome", "rome")]
    claims.append(verify.Claim("L1.A[h1]", "tlrome:t5.L1.A", consumer="M[4,9]"))
    rows = verify.verify(dbg, run, claims)
    assert all(r["state"] == "resolved" for r in rows), [r.get("reason") for r in rows]
    assert abs(rows[0]["native"] - rows[1]["native"]) < 1e-5
    assert rows[0]["report"]["edit"]["origin"]["via"] == "import"
    assert rows[2]["native"] != 0.0


def test_processed_import_refused():
    _tools()
    from mdb.ops.errors import Refused

    model, dbg = _debugger()
    with pytest.raises(Refused, match="weight-processed"):
        dbg.import_values("p", {"t1.L1.v": torch.zeros(1, 1, 8)}, tokens=[1, 2],
                          provenance={"processed": True})
    dbg.import_values("p", {"t1.L1.p": torch.zeros(1, 2, 1, 2)}, tokens=[1, 2],
                      provenance={"processed": True})


def test_batch_whatif_matches_mdb_whatif():
    """nnsight batched edits written as mdb addresses == mdb's native whatif per prompt."""
    _tools()
    from mdb.interop import nn, verify

    warnings.filterwarnings("ignore")
    model, dbg = _debugger()
    recipients = [TOKENS[0], [3, 1, 4, 1, 5, 9]]
    donors = [TOKENS[1], [2, 7, 1, 8, 2, 8]]
    consumers = [(4, 9), (4, 9)]
    arms = {"head": [nn.Edit("L1.A[h1]", 0.0, "scale")],
            "neuron": [nn.Edit("L0.f[i=3]", 0.0, "replace")],
            "vcell": [nn.Edit("cache.L1.v[kv0,row=2]", "donor")],
            "v0cell": [nn.Edit("cache.L0.v[kv0,row=2]", "donor")],
            "v0cell_seq": [nn.Edit("cache.L0.v[kv0,row=2]", "donor", scope="sequence")],
            "kcell": [nn.Edit("cache.L0.k[kv0,row=1]", "donor")],
            "vself": [nn.Edit("cache.L1.v[kv0,row=2]", "self", donor_row=1)]}
    batch = nn.batch_whatif(model, recipients, consumers, arms, donors=donors)
    for i, tokens in enumerate(recipients):
        dbg.run(donors[i], name="donor")
        run = dbg.run(tokens)
        rows = verify.verify(dbg, run, [
            verify.Claim("L1.A[h1]", 0.0, "scale", "M[4,9]"),
            verify.Claim("L0.f[i=3]", 0.0, "replace", "M[4,9]"),
            verify.Claim("cache.L1.v[kv0,row=2]", "donor:cache.L1.v[kv0,row=2]",
                         consumer="M[4,9]"),
            verify.Claim("cache.L0.k[kv0,row=1]", "donor:cache.L0.k[kv0,row=1]",
                         consumer="M[4,9]"),
            verify.Claim("cache.L1.v[kv0,row=2]", "cache.L1.v[kv0,row=1]", consumer="M[4,9]")])
        for arm, row in zip(("head", "neuron", "vcell", "kcell", "vself"), rows):
            assert row["state"] == "resolved", row
            assert abs(batch["arms"][arm]["delta"][i] - row["native"]) < 1e-5, (arm, i)
        assert abs(batch["base"][i] - rows[0]["base"]) < 1e-5
    # sequence scope lets rows 3..4 of layer 0 read the patched cell too, and layer 1 reads
    # their changed keys and values: a different experiment
    assert batch["arms"]["v0cell_seq"]["delta"] != batch["arms"]["v0cell"]["delta"]


def test_bridge_leaves_mdb_model_untouched_and_bridged_tree_refused():
    """TL rewrites the module tree it boots on; tl.bridge boots a weight-sharing clone."""
    _tools()
    import mdb
    from mdb.interop import tl
    from mdb.ops.errors import Refused

    warnings.filterwarnings("ignore")
    model = _model("qwen2", "fp32")
    before = [(n, type(m).__name__) for n, m in model.named_modules()]
    bridge = tl.bridge(model, _tokenizer("qwen2"))
    assert [(n, type(m).__name__) for n, m in model.named_modules()] == before
    shared = {p.data_ptr() for p in model.parameters()}
    assert shared <= {p.data_ptr() for p in bridge.parameters()}
    with mdb.trace(model, ports=["L1.p"]) as t, torch.no_grad():
        model(torch.tensor(TOKENS[:1]))
    assert t.info("L1.p")["state"] == "resolved"
    other = _model("qwen2", "fp32")
    tl.bridge(other, _tokenizer("qwen2"), in_place=True)
    with pytest.raises(Refused, match="TransformerLens bridge"):
        mdb.attach(other)


def test_batch_whatif_per_prompt_vectors_match_import_whatif():
    """A per-prompt vector added at L1.m (e.g. minus one transcoder feature's write) measured in
    an nnsight batch equals mdb's whatif with the same vector as an import value."""
    _tools()
    from mdb.interop import nn, verify

    warnings.filterwarnings("ignore")
    model, dbg = _debugger()
    recipients = [TOKENS[0], [3, 1, 4, 1, 5, 9]]
    vectors = torch.randn(2, 16, generator=torch.Generator().manual_seed(0)) * 0.3
    batch = nn.batch_whatif(model, recipients, [(4, 9), (4, 9)],
                            {"vec": [nn.Edit("L1.m", vectors, "add")]})
    for i, tokens in enumerate(recipients):
        dbg.import_values("vec", {"t5.L1.m": vectors[i].reshape(1, 1, 16)}, tokens=tokens,
                          provenance={"tool": "test"})
        run = dbg.run(tokens)
        [row] = verify.verify(dbg, run, [verify.Claim("L1.m", "vec:t5.L1.m", "add", "M[4,9]")])
        assert row["state"] == "resolved", row.get("reason")
        assert abs(batch["arms"]["vec"]["delta"][i] - row["native"]) < 1e-5
