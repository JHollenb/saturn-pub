import pytest
import torch
from test_mlp_memory import tiny

from mdb.bench import BenchError, BenchSession
from mdb.memit import (
    MEMITHyperParams,
    MEMITPatch,
    alphaedit_solve,
    layer_second_moments,
    load_overlay,
    memit,
    subject_last_index,
)
from mdb.mlp_memory import MLPMemoryError


class CharTok:
    """Char-level tokenizer over a 64-id vocabulary (test only)."""
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [1 + (ord(c) % 63) for c in text]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i) + 31) for i in ids)

    def get_vocab(self):
        return {chr(i + 31): i for i in range(64)}


TOK = CharTok()
HP = MEMITHyperParams(layers=(0, 1), v_num_grad_steps=15, mom2_update_weight=1.0)
TEMPLATES = ["{}", "ab. {}", "xyz. {}"]
REQS = [{"id": "f", "prompt": "cap of {} is", "subject": "Fr", "target_new": "Q"},
        {"id": "j", "prompt": "cap of {} is", "subject": "Jp", "target_new": "W"}]


def last_logits(model, text):
    with torch.inference_mode():
        return model(input_ids=torch.tensor([TOK(text).input_ids]), use_cache=False).logits[0, -1].float()


def test_memit_edits_and_exact_undo(tmp_path):
    model = tiny()
    texts = ["the quick brown fox jumps over the lazy dog " * 20] * 4
    C, n = layer_second_moments(model, TOK, texts, [0, 1], n_tokens=2000, max_len=64, batch=4)
    assert n >= 1000 and C[0].shape == (64, 64)
    assert subject_last_index(TOK, "cap of {} is", "Fr") == len("cap of Fr") - 1
    native = {r["id"]: last_logits(model, r["prompt"].format(r["subject"])) for r in REQS}
    patch, diag = memit(model, TOK, REQS, C, HP, context_templates=TEMPLATES, base_identity="w")
    assert set(patch.deltas) == {0, 1} and patch.deltas[0].shape == (32, 64)
    assert all(d["nll_final"] < d["losses"][0] for d in diag["requests"].values())
    with patch.install(model):
        for r in REQS:
            lp = torch.log_softmax(last_logits(model, r["prompt"].format(r["subject"])), -1)
            lp0 = torch.log_softmax(native[r["id"]], -1)
            t = TOK(r["target_new"]).input_ids[0]
            assert lp[t] > lp0[t]
        with pytest.raises(MLPMemoryError):
            memit(model, TOK, REQS, C, HP, context_templates=TEMPLATES)
    for r in REQS:
        assert torch.equal(last_logits(model, r["prompt"].format(r["subject"])), native[r["id"]])
    digest = patch.save(tmp_path / "p.safetensors")
    loaded = load_overlay(tmp_path / "p.safetensors", expected_digest=digest)
    assert isinstance(loaded, MEMITPatch) and loaded.digest == digest


def test_memit_patch_through_bench_resume(tmp_path):
    model = tiny(torch.bfloat16)
    texts = ["some generic text for statistics " * 30] * 3
    C, _ = layer_second_moments(model, TOK, texts, [0, 1], n_tokens=1500, max_len=64)
    patch, _ = memit(model, TOK, REQS, C, HP, context_templates=TEMPLATES)
    root = tmp_path / "bench"
    b = BenchSession.from_resident(model, root=root, continuation_capacity=8)
    parent = b.prefill([1, 2, 3, 4])
    act = b.mlp_memory_install(patch)
    assert act["memory"]["method"].startswith("MEMIT")
    with pytest.raises(BenchError, match="different MLP memory overlay"):
        b.restore(parent)
    b.close()
    r = BenchSession.resume(model, root=root)
    assert r.mlp_memory_digest == patch.digest
    assert r.mlp_memory_remove()["exact_undo"]
    r.restore(parent)
    r.close()


def test_alphaedit_projects_update_off_preserved_keys_and_stacks(tmp_path):
    from mdb.memit import AlphaEditHyperParams, alphaedit, null_space_projection
    model = tiny()
    texts = ["the quick brown fox jumps over the lazy dog " * 20] * 4
    C, _ = layer_second_moments(model, TOK, texts, [0, 1], n_tokens=2000, max_len=64, batch=4)
    thr = float(torch.linalg.svdvals(C[0]).median())  # tiny model: threshold from its own spectrum
    P, dims = null_space_projection(C[0], thr)
    assert 0 < dims < 64 and torch.allclose(P @ P, P, atol=1e-4)
    hp = AlphaEditHyperParams(layers=(0, 1), v_num_grad_steps=15, nullspace_threshold=thr, L2=1e-2)
    native = {r["id"]: last_logits(model, r["prompt"].format(r["subject"])) for r in REQS}
    first, diag = alphaedit(model, TOK, REQS[:1], C, hp, context_templates=TEMPLATES)
    assert first.algorithm == "AlphaEdit" and first.keys[0].shape == (64, 1)
    # update rows live in the null space: dW @ (I - P) ~ 0 relative to dW
    for layer in (0, 1):
        Pl, _ = null_space_projection(C[layer], thr)
        dW = first.deltas[layer]
        assert float((dW @ (torch.eye(64) - Pl)).norm()) < 1e-3 * float(dW.norm()) + 1e-6
    both, _ = alphaedit(model, TOK, REQS[1:], C, hp, context_templates=TEMPLATES, base=first)
    assert [r["id"] for r in both.requests] == ["f", "j"] and both.keys[0].shape == (64, 2)
    gain = 0.0
    with both.install(model):
        for r in REQS:
            t = TOK(r["target_new"]).input_ids[0]
            lp = torch.log_softmax(last_logits(model, r["prompt"].format(r["subject"])), -1)
            gain += float(lp[t] - torch.log_softmax(native[r["id"]], -1)[t])
    assert gain > 0
    digest = both.save(tmp_path / "a.safetensors")
    loaded = load_overlay(tmp_path / "a.safetensors", expected_digest=digest)
    assert loaded.algorithm == "AlphaEdit" and torch.equal(loaded.keys[1], both.keys[1])
    seq, _ = memit(model, TOK, REQS[1:], C, HP, context_templates=TEMPLATES, base=memit(
        model, TOK, REQS[:1], C, HP, context_templates=TEMPLATES)[0])
    assert seq.algorithm == "MEMIT" and len(seq.requests) == 2 and not seq.keys


def test_alphaedit_woodbury_solve_matches_dense_reference():
    g = torch.Generator().manual_seed(0)
    d, n, m, d_model = 48, 5, 7, 16
    C = torch.randn(d, 3 * d, generator=g, dtype=torch.float64)
    U, S, _ = torch.linalg.svd(C @ C.T / (3 * d))
    U0 = U[:, S < S.median()]
    P = U0 @ U0.T
    K = torch.randn(d, n, generator=g, dtype=torch.float64)
    Kp = torch.randn(d, m, generator=g, dtype=torch.float64)
    R = torch.randn(d_model, n, generator=g, dtype=torch.float64)
    eye = torch.eye(d, dtype=torch.float64)
    for prev, Cp in ((None, torch.zeros(d, d, dtype=torch.float64)), (Kp, Kp @ Kp.T)):
        for l2 in (1.0, 0.1):
            dense = torch.linalg.solve(P @ (K @ K.T + Cp) + l2 * eye, P @ K @ R.T)
            assert torch.allclose(alphaedit_solve(P, K, R, prev, l2), dense, atol=1e-9, rtol=1e-9)
