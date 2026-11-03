import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mdb.compiled_state import (  # noqa: E402
    CompiledStateError,
    FeatureSet,
    PromptFamily,
    Symbol,
    visibility,
)
from mdb.compiled_state_runner import DecoderSplice, _DeviceMoments, device_features  # noqa: E402
from mdb.compiled_state_serving import (  # noqa: E402
    ZeroLayerChain,
    apply_feedback,
    chain_agreement,
    feedback_tables,
    serve,
)

PREFIX = [1, 2, 3]
VOCAB = 40


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def family():
    # template x0 x1 op, then the first generated token t1 is fed back; score at op and t1
    return PromptFamily(name="chain", prefix="<ids 1 2 3>", positions=("x0", "x1", "op", "t1"),
                        symbols=(Symbol("x0", 3, 0), Symbol("pair", 9, 1), Symbol("t1", VOCAB, 3)),
                        feature_sets={"exact": FeatureSet(("x0", "pair", "t1"))}, score_positions=(2, 3))


def stream(reps):
    x0, x1 = (a.ravel() for a in np.meshgrid(np.arange(3), np.arange(3), indexing="ij"))
    ids = np.stack([10 + x0, 10 + x1, np.full(9, 20)], 1)
    vals = np.stack([x0, 3 * x0 + x1, np.zeros(9, int)], 1)
    return np.tile(ids, (reps, 1)), np.tile(vals, (reps, 1))


FEEDBACK = {"t1": (0, torch.arange(VOCAB))}


def fitted_chain(run, fam, ids, vals):
    toks, finals = run.decode(ids, 2)
    tables = feedback_tables(fam, FEEDBACK, run.device)
    v = apply_feedback(torch.as_tensor(vals), toks, tables)
    vis = torch.as_tensor(visibility(fam), dtype=torch.float32)
    m = _DeviceMoments(fam, [2], 16, ["exact"])
    m.add(device_features(fam, v, vis), {2: finals}, v)
    cmap = m.fit().solve("exact", 2, 1e-6)
    return ZeroLayerChain(run, cmap, feedback=FEEDBACK), toks, cmap


def test_zero_layer_chain_reproduces_native_greedy_decode(model):
    run = DecoderSplice(model, PREFIX)
    fam = family()
    ids, vals = stream(1)
    chain, native, cmap = fitted_chain(run, fam, ids, vals)
    toks, margins = chain(vals)
    assert torch.equal(toks, native) and margins.shape == (9, 2) and bool((margins >= 0).all())
    full_vocab = ZeroLayerChain(run, cmap, feedback=FEEDBACK, vocab=range(VOCAB))
    assert torch.equal(full_vocab(vals)[0], native)
    fast, none = chain(vals, margins=False)
    assert torch.equal(fast, native) and none is None
    a = chain_agreement(chain, torch.as_tensor(vals), native, batch=4)
    assert a["exact_through"] == [1.0, 1.0] and a["step_agreement_teacher_forced"] == [1.0, 1.0]
    with pytest.raises(CompiledStateError, match="num_layers"):
        ZeroLayerChain(run, dataclasses.replace(cmap, layer=1), feedback=FEEDBACK)
    with pytest.raises(CompiledStateError, match="visible after"):
        feedback_tables(fam, {"pair": (0, torch.arange(VOCAB))}, run.device)


def test_serve_counts_native_forwards_and_audits_served_answers(model):
    run = DecoderSplice(model, PREFIX)
    fam = family()
    ids, vals = stream(3)
    audit, _ = run.decode(ids, 2)
    kw = dict(steps=2, feedback=FEEDBACK, warmup=9, refit_every=1000, batch=4, lam=1e-6, audit_tokens=audit, window=9)
    never = serve(run, fam, "exact", ids, vals, tau=None, **kw)
    assert never["native_prompts"] == 9 and never["forwards_per_1000_prompts"] == pytest.approx(1000 / 3)
    assert never["audit"]["predicted_rows"] == 18 and never["audit"]["predicted_exact"] == 1.0
    assert [w["native_prompts"] for w in never["audit"]["windows"]] == [9, 0, 0]
    always = serve(run, fam, "exact", ids, vals, tau=float("inf"), **{**kw, "refit_every": 9})
    assert always["native_prompts"] == 27 and always["forwards_per_1000_prompts"] == 1000.0
    assert always["audit"]["served_exact"] == 1.0 and always["refits"] == 2  # warmup fit + one after 12 refreshed rows
    with pytest.raises(CompiledStateError, match="template plus"):
        serve(run, fam, "exact", ids, vals, tau=None, **{**kw, "steps": 3})
