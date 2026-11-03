import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mdb.compiled_state import (  # noqa: E402
    CompiledStateMap,
    FeatureSet,
    PromptFamily,
    Symbol,
    design,
    verify_maps,
    verify_report,
    visibility,
)
from mdb.compiled_state_runner import (  # noqa: E402
    DecoderSplice,
    Rows,
    device_features,
    kv_rows,
    kv_unrows,
    measure,
    parity,
)

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
    assert report["parity"]["gate"]["passed"]
    assert report["noise_ceiling"]["row_agreement"] == 1.0 and report["noise_ceiling"]["n"] == 9
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


def full_logits(model, ids):
    return model(input_ids=torch.as_tensor([PREFIX + list(x) for x in np.asarray(ids)])).logits[:, len(PREFIX):]


def test_true_resume_reproduces_native_without_running_lower_layers(model):
    run = DecoderSplice(model, PREFIX)
    r = rows()
    x, cap, _ = run.native(r.input_ids, capture=[0, 1, 2])
    torch.testing.assert_close(run.readout(x), full_logits(model, r.input_ids), rtol=1e-5, atol=1e-5)
    _, hook_cap = run.forward(r.input_ids, capture=[0, 1])
    for L in (0, 1):
        torch.testing.assert_close(cap[L], hook_cap[L], rtol=1e-5, atol=1e-5)
    calls = []
    handle = run.layers[0].register_forward_hook(lambda *a: calls.append(1))
    try:
        for L in (1, 2):
            xr, _, _ = run.run(cap[L], L)
            assert torch.equal(xr, x)
    finally:
        handle.remove()
    assert calls == []
    torch.testing.assert_close(run.readout(cap[2]), run.readout(x))


def test_partial_run_with_native_kv_and_parity_gate(model):
    run = DecoderSplice(model, PREFIX)
    fam = toy()
    r = rows()
    x, cap, kvs = run.native(r.input_ids, capture=[0, 1], capture_kv=[0, 1])
    for L in (0, 1):
        kv = {layer: (kvs[layer][0][:, :, :2], kvs[layer][1][:, :, :2]) for layer in range(L, 2)}
        xp, _, _ = run.run(cap[L][:, 2:], L, positions=[2, 3], kv=kv, kv_positions=[0, 1])
        torch.testing.assert_close(xp, x[:, 2:], rtol=1e-5, atol=1e-5)
    with pytest.raises(Exception, match="both run and be injected"):
        run.run(cap[1][:, 1:], 1, positions=[1, 2, 3], kv={1: (kvs[1][0][:, :, :2], kvs[1][1][:, :, :2])},
                kv_positions=[0, 1])
    gates = parity(run, fam, r.input_ids, [0, 1, 2], partial_layers=[0, 1], tol=1e-4)
    assert gates["gate"]["passed"] and set(gates["kv_injection"]) == {"0", "1"}
    assert gates["resume"]["2"]["top1_agreement"] == 1.0


def test_decode_matches_greedy_and_teacher_forced_states(model):
    run = DecoderSplice(model, PREFIX)
    ids = rows().input_ids[:, :3]
    toks, finals = run.decode(ids, 3)
    seq = np.asarray(ids)
    for _ in range(3):
        nxt = full_logits(model, seq)[:, -1].argmax(-1).numpy()
        seq = np.concatenate([seq, nxt[:, None]], 1)
    np.testing.assert_array_equal(toks.numpy(), seq[:, 3:])
    x, _, _ = run.native(seq[:, :-1])
    torch.testing.assert_close(finals, x, rtol=1e-5, atol=1e-5)
    sub, _ = run.decode(ids, 2, vocab=[5, 7, 9])
    assert set(sub.unique().tolist()) <= {5, 7, 9}


def test_kv_rows_round_trip_and_device_features_match_design():
    k, v = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
    z = kv_rows(k, v)
    assert z.shape == (2, 4, 30)
    k2, v2 = kv_unrows(z, 3, 5)
    assert torch.equal(k2, k) and torch.equal(v2, v)
    fam = toy()
    vals = rows().values
    vis = torch.as_tensor(visibility(fam), dtype=torch.float32)
    want = torch.as_tensor(design(fam, vals))[:, None, :] * vis[None]
    assert torch.equal(device_features(fam, torch.as_tensor(vals), vis), want.float())


def test_measure_partial_arms_report_compiled_and_oracle_kv(model):
    report, maps = measure(model, toy(), PREFIX, rows(4), rows(), layers=[2], partial_layers=[0, 1],
                           feature_sets=["digits"], lam=1e-6, batch=5, parity_rows=3)
    verify_report(report)
    d = report["feature_sets"]["digits"]
    assert set(d["partial"]) == {"0", "1"} and set(d["kv_maps"]) == {"0", "1"}
    assert d["partial"]["1"]["running_positions"] == ["op", "y"]
    assert d["partial"]["1"]["injected_positions"] == ["x0", "x1"]
    # layer 0 is the embedding, so the compiled state there is exact and the native-K/V oracle is native
    assert d["partial"]["0"]["native_kv_oracle"]["row_agreement"] == 1.0
    assert maps[("digits", 1, "kv")].target == "kv"
    assert maps[("digits", 1, "kv")].weights.shape == (4, 9, 16)  # T, family width, 2 * kv heads * head dim
    assert report["parity"]["gate"]["passed"] and set(report["parity"]["kv_injection"]) == {"0", "1"}
    assert report["parity"]["rows"] == 3
