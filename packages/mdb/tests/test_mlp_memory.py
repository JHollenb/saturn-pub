import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchError, BenchSession
from mdb.bench_mlp_memory import verify_act
from mdb.mlp_memory import MLPMemory, MLPMemoryError, capture


def tiny(dtype=torch.float32, seed=117):
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attn_implementation="eager",
    )).eval().to(dtype)


PRESERVE = [[5, 6, 7, 8, 9], [10, 11, 12], [20, 21, 22, 23], [30, 31, 32, 33, 34, 35]]
EDITS = [
    {"id": "a", "ids": [1, 2, 3, 4], "position": 2, "donor_ids": [40, 41, 42, 43], "donor_position": 2},
    {"id": "b", "ids": [13, 14, 15], "position": 1, "donor_ids": [50, 51, 52], "donor_position": 1},
]


def logits(model, ids):
    with torch.inference_mode():
        return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].clone()


def test_fit_writes_donor_outputs_and_undo_is_exact():
    model = tiny()
    native = {tuple(p): logits(model, p) for p in PRESERVE + [e["ids"] for e in EDITS]}
    memory = MLPMemory.fit(model, EDITS, PRESERVE, layers=[0, 1], alpha=1e-6, edit_weight=30.0)
    donors = {e["id"]: capture(model, e["donor_ids"], [0], {})[1] for e in EDITS}
    with memory.install(model):
        with pytest.raises(MLPMemoryError, match="native model"):
            capture(model, [1, 2], [0], {})
        with pytest.raises(MLPMemoryError, match="already installed"):
            memory.install(model)
        for edit in EDITS:
            donor = donors[edit["id"]]
            outs = {}
            hooks = []
            for layer in (0,):
                def hook(module, inputs, output, layer=layer):
                    outs[layer] = output[0].detach().double()
                hooks.append(model.model.layers[layer].mlp.register_forward_hook(hook))
            logits(model, edit["ids"])
            for h in hooks:
                h.remove()
            got, want = outs[0][edit["position"]], donor[0][edit["donor_position"]]
            assert float((got - want).norm() / want.norm()) < 0.05
        drift = max(float((logits(model, p) - native[tuple(p)]).abs().max()) for p in PRESERVE)
        assert drift < 0.05
    for ids, value in native.items():
        assert torch.equal(logits(model, list(ids)), value)


def test_woodbury_insert_and_delete_are_exact():
    model = tiny()
    memory = MLPMemory.fit(model, EDITS[:1], PRESERVE, layers=[1], alpha=1e-3, edit_weight=10.0)
    base = memory.readouts[1]
    P0, C0, B0 = base.P.clone(), base.C.clone(), base.B.clone()
    memory.insert(model, EDITS[1:])
    rows = list(base.rows.values())
    Phi = torch.cat([r[0] for r in rows])
    T = torch.cat([r[1] for r in rows])
    w = torch.cat([r[2] for r in rows])
    gram = Phi.T @ (Phi * w.view(-1, 1)) + base.alpha * torch.eye(Phi.shape[1], dtype=torch.float64)
    ref = torch.linalg.solve(gram, Phi.T @ (T * w.view(-1, 1)))
    assert float((base.P @ base.C - ref).abs().max()) < 1e-8 * float(ref.abs().max()) + 1e-12
    assert set(memory.edits) == {"a", "b"}
    memory.delete("b")
    assert float((base.P - P0).abs().max()) < 1e-8 * float(P0.abs().max())
    assert float((base.C - C0).abs().max()) < 1e-10
    assert torch.allclose(base.B, B0, atol=1e-6, rtol=1e-5)
    with pytest.raises(MLPMemoryError):
        memory.delete("b")


def test_save_load_round_trip_and_digest(tmp_path):
    model = tiny()
    memory = MLPMemory.fit(model, EDITS, PRESERVE, layers=[0, 2], base_identity="w")
    digest = memory.save(tmp_path / "m.safetensors")
    loaded = MLPMemory.load(tmp_path / "m.safetensors", expected_digest=digest)
    assert loaded.digest == digest and loaded.layers == [0, 2]
    probe = [1, 2, 3, 4]
    with memory.install(model):
        a = logits(model, probe)
    with loaded.install(model):
        b = logits(model, probe)
    assert torch.equal(a, b)
    with pytest.raises(MLPMemoryError):
        MLPMemory.load(tmp_path / "m.safetensors", expected_digest="0" * 64)


def test_rejects_malformed_edits():
    model = tiny()
    with pytest.raises(MLPMemoryError, match="position"):
        MLPMemory.fit(model, [{**EDITS[0], "position": 9}], layers=[0])
    with pytest.raises(MLPMemoryError, match="exactly one"):
        MLPMemory.fit(model, [{"id": "x", "ids": [1, 2], "position": 0}], layers=[0])
    with pytest.raises(MLPMemoryError, match="distinct"):
        MLPMemory.fit(model, [EDITS[0], EDITS[0]], layers=[0])


def test_bench_install_remove_binds_cuts_and_resumes(tmp_path):
    model = tiny(torch.bfloat16)
    memory = MLPMemory.fit(model, EDITS, PRESERVE, layers=[0, 1], edit_weight=30.0)
    root = tmp_path / "bench"
    b = BenchSession.from_resident(model, root=root, continuation_capacity=8)
    parent = b.prefill([1, 2, 3, 4])
    native_logits = b.last_logits.clone()
    act = b.mlp_memory_install(memory)
    assert verify_act({k: v for k, v in act.items() if k not in {"act_ref", "cut_id"}})
    assert b.artifacts.get(act["cut_id"])["mlp_memory"] == memory.digest
    assert act["act_id"] in b.branch.intervention_lineage
    assert act["installed_probe_sha256"] != act["native_probe_sha256"]
    with pytest.raises(BenchError, match="different MLP memory overlay"):
        b.restore(parent)
    with pytest.raises(BenchError, match="remove the live"):
        b.mlp_memory_install(memory)
    b.prefill([1, 2, 3, 4], branch="same-prompt-under-overlay")
    b.prefill([5, 6, 7], branch="other-prompt")
    b.checkout("main")
    b.step("token")
    live_cut = b.checkpoint("live")
    b.close()
    assert not any(layer.mlp._forward_hooks for layer in model.model.layers)

    r = BenchSession.resume(model, root=root)
    assert r.mlp_memory_digest == memory.digest
    assert r.artifacts.get(r.branch.cut)["mlp_memory"] == memory.digest
    removed = r.mlp_memory_remove()
    assert removed["exact_undo"] and verify_act(
        {k: v for k, v in removed.items() if k not in {"act_ref", "cut_id"}})
    assert not any(layer.mlp._forward_hooks for layer in model.model.layers)
    with pytest.raises(BenchError, match="different MLP memory overlay"):
        r.restore(live_cut)
    r.restore(parent)
    assert torch.equal(r.last_logits, native_logits)
    with pytest.raises(BenchError, match="no MLP memory"):
        r.mlp_memory_remove()
    r.close()


def test_legacy_cuts_carry_no_overlay_field(tmp_path):
    b = BenchSession.from_resident(tiny(torch.bfloat16), root=tmp_path / "plain", continuation_capacity=8)
    cut = b.prefill([1, 2, 3])
    assert "mlp_memory" not in b.artifacts.get(cut)
    assert "mlp_memory" not in b._manifest()
    b.close()


def _grid():
    subjects = {f"s{i}": [40 + i] for i in range(6)}
    relations = {"r0": [1, 2], "r1": [3, 4], "r2": [5, 6, 7]}
    return [{"ids": rel + subj, "position": len(rel + subj) - 1, "subject": s, "relation": r}
            for s, subj in subjects.items() for r, rel in relations.items()]


def test_feature_maps_fit_round_trip_and_linear_identity_is_unchanged(tmp_path):
    from mdb.mlp_memory import QuadraticFeatures, fit_address_readers
    model = tiny()
    linear = MLPMemory.fit(model, EDITS, PRESERVE, layers=[0, 1])
    assert "feature_maps" not in linear.manifest()
    bound, stats = fit_address_readers(model, _grid(), [0, 1], k_subject=4, kind="bound")
    assert abs(sum(v for k, v in stats[0].items() if k in {"subject", "relation", "interaction"}) - 1) < 1e-5
    phasor, _ = fit_address_readers(model, _grid(), [0, 1], k_subject=4, kind="phasor", phasor_k=16)
    quad = {layer: QuadraticFeatures.random(32, 24, layer) for layer in (0, 1)}
    probe = [1, 2, 3, 4]
    for features in (bound, phasor, quad):
        memory = MLPMemory.fit(model, EDITS, PRESERVE, layers=[0, 1], features=features)
        assert memory.manifest()["feature_maps"]["0"]["kind"] in {"bound", "phasor", "quadratic"}
        digest = memory.save(tmp_path / f"{memory.digest}.safetensors")
        loaded = MLPMemory.load(tmp_path / f"{digest}.safetensors", expected_digest=digest)
        with memory.install(model):
            a = logits(model, probe)
        with loaded.install(model):
            b = logits(model, probe)
        assert torch.equal(a, b)
    with pytest.raises(MLPMemoryError, match="one prompt per"):
        fit_address_readers(model, _grid()[:-1], [0])


def test_delta_edits_and_attribute_delta():
    from mdb.mlp_memory import attribute_delta
    model = tiny()
    ids, pos = [1, 2, 3, 4], 3
    before = torch.softmax(logits(model, ids), -1)
    target = int(before.argmin())
    deltas, stats = attribute_delta(model, ids, pos, target, keep=[{"ids": [5, 6, 3, 4], "position": 3}],
                                    layers=[0, 1], steps=40, lr=0.1)
    assert stats["p_target"] > float(before[target]) * 2
    assert not any(p.grad is not None for p in model.parameters())
    edit = {"id": "a", "ids": ids, "position": pos, "deltas": deltas}
    memory = MLPMemory.fit(model, [edit], PRESERVE, layers=[0, 1], alpha=1e-6, edit_weight=30.0)
    assert "deltas_sha256" in memory.edits["a"]
    with memory.install(model):
        after = torch.softmax(logits(model, ids), -1)
    assert float(after[target]) > float(before[target]) * 1.5


def test_spectral_readers_abstain_off_subject_and_round_trip(tmp_path):
    from mdb.mlp_memory import AddressFeatures, address_posteriors, fit_spectral_readers
    model = tiny()
    labelled = [dict(item) for item in _grid()]
    labelled += [{"ids": [40 + i], "position": 0, "subject": f"s{i}", "relation": None} for i in range(6)]
    for mode in ("phasor", "unit", "onehot"):
        maps, diag, classes = fit_spectral_readers(model, labelled, [0, 1], k=32, mode=mode, steps=50)
        assert classes["roles"][-1] == "<none>" and diag[0]["train_acc_subject"] > 0.5
        fmap = maps[1]
        assert isinstance(fmap, AddressFeatures)
        memory = MLPMemory.fit(model, EDITS, PRESERVE, layers=[0, 1], features=maps)
        digest = memory.save(tmp_path / f"{mode}.safetensors")
        loaded = MLPMemory.load(tmp_path / f"{mode}.safetensors", expected_digest=digest)
        with memory.install(model):
            a = logits(model, [1, 2, 3, 4])
        with loaded.install(model):
            b = logits(model, [1, 2, 3, 4])
        assert torch.equal(a, b)
    ins, _ = capture(model, [1, 2, 40], [1], {})
    ps, pr = address_posteriors(maps[1], ins[1].float())
    assert torch.allclose(ps.sum(-1), torch.ones(3, dtype=ps.dtype), atol=1e-5)
