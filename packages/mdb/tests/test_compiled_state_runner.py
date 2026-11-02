import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mdb.compiled_state import (  # noqa: E402
    CompiledStateMap,
    FeatureSet,
    PromptFamily,
    Symbol,
    verify_maps,
    verify_report,
)
from mdb.compiled_state_runner import DecoderSplice, Rows, measure  # noqa: E402

PREFIX = [1, 2, 3]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def toy():
    return PromptFamily(name="toy", prefix="<ids 1 2 3>", positions=("x0", "x1", "op", "y"),
                        symbols=(Symbol("x0", 3, 0), Symbol("x1", 3, 1), Symbol("g", 2, 1)),
                        feature_sets={"digits": FeatureSet(("x0", "x1")),
                                      "digits|g": FeatureSet(("x0", "x1"), switch="g")},
                        score_positions=(2, 3))


def rows(reps=1):
    x0, x1 = (a.ravel() for a in np.meshgrid(np.arange(3), np.arange(3), indexing="ij"))
    ids = np.stack([10 + x0, 10 + x1, np.full(9, 20), np.full(9, 21)], 1)
    vals = np.stack([x0, x1, (x0 + x1 >= 2).astype(int)], 1)
    return Rows(np.repeat(ids, reps, 0), np.repeat(vals, reps, 0))


def test_capture_matches_full_forward_and_native_splice_is_identity(model):
    run = DecoderSplice(model, PREFIX)
    r = rows()
    logits, cap = run.forward(r.input_ids, capture=[0, 1, 2])
    full = model(input_ids=torch.as_tensor([PREFIX + list(x) for x in r.input_ids]), output_hidden_states=True)
    torch.testing.assert_close(logits, full.logits[:, len(PREFIX):], rtol=1e-5, atol=1e-5)
    for L in (0, 1):
        torch.testing.assert_close(cap[L], full.hidden_states[L][:, len(PREFIX):], rtol=1e-5, atol=1e-5)
    for L in (0, 1, 2):
        out, _ = run.forward(r.input_ids, splice=(L, cap[L]))
        assert torch.equal(out, logits)
    shifted, _ = run.forward(r.input_ids, splice=(1, cap[1] + 1.0))
    assert not torch.equal(shifted, logits)


def test_measure_reproduces_exact_embedding_layer_and_seals_maps(model, tmp_path):
    fam = toy()
    report, maps = measure(model, fam, PREFIX, rows(4), rows(), layers=[0, 1, 2], lam=1e-6, threshold=0.99,
                           batch=5, model_identity={"name": "tiny-qwen2"})
    verify_report(report)
    digits = report["feature_sets"]["digits"]
    # the residual entering layer 0 is the token embedding: exactly linear in the visible digits
    assert digits["layers"]["0"]["agreement"]["row_agreement"] == 1.0
    r2 = [p["r2"] for p in digits["layers"]["0"]["positions"]]
    assert r2[0] > 0.9999 and r2[1] > 0.9999 and r2[2] is None and r2[3] is None
    assert digits["horizon"]["horizon"] is not None and report["rows"] == {"fit": 36, "eval": 9, "r2": 9}
    assert report["feature_sets"]["digits|g"]["maps"]["1"]["switch"] == "g"
    m = maps[("digits|g", 1)]
    m.save(tmp_path)
    assert verify_maps(tmp_path)[0]["fingerprint"] == report["feature_sets"]["digits|g"]["maps"]["1"]["fingerprint"]
    back = CompiledStateMap.load(tmp_path, "digits|g", 1)
    np.testing.assert_array_equal(back.weights, m.weights)
