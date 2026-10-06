"""Native query authority, reversibility and durable paired-memory evidence."""

from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchError, BenchSession
from mdb.bench_memory_probe import verify_artifact
from mdb.bench_verify import verify


@pytest.fixture
def bench(tmp_path):
    torch.manual_seed(207)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=64,
                attn_implementation="eager",
            )
        )
        .eval()
        .to(torch.bfloat16)
    )
    session = BenchSession.from_resident(model, root=tmp_path / "bench", continuation_capacity=8)
    session.prefill([1, 2, 3])
    yield session
    session.close()


def _spec(**changes):
    spec = {
        "intervention": {
            "mode": "zero",
            "selector": {"layers": [1], "row_start": 1, "row_stop": 2, "components": ["value"]},
        },
        "query_token_ids": [7],
        "continuation_tokens": 2,
        "observable": {"kind": "token_margin", "token_id": 9, "against_token_id": 11},
    }
    spec.update(changes)
    return spec


def test_fixed_query_pair_restores_parent_and_sibling(bench, tmp_path):
    parent = bench.branch.cut
    history = list(bench.branch.token_ids)
    bench.fork("sibling", cut=parent)
    bench.checkout("main")
    parent = bench.branch.cut
    sibling = bench.branches["sibling"].cut
    values = bench.resolve_value("kv_cache.layer_001.value")
    result = bench.memory_probe(_spec())
    report = result["report"]
    assert bench.branch.cut == parent == result["cut_id"]
    assert bench.branches["sibling"].cut == sibling
    assert torch.equal(values, bench.resolve_value("kv_cache.layer_001.value"))
    for role in ("native", "candidate", "reinstall"):
        arm = report[role]
        assert arm["closure"]["token_ids"] == history + [7] + arm["answer_token_ids"]
        assert arm["answer_token_ids"] == [
            row["pending_token_id"] for row in arm["answer_readouts"]
        ]
        assert arm["native_output_measured"]
    assert report["reinstall_effect"]["closure_exact"]
    assert report["effect"]["signed_effect"] == (
        report["candidate"]["first_readout"]["raw_logit_margin"]
        - report["native"]["first_readout"]["raw_logit_margin"]
    )
    assert verify(bench.root)["counts"]["memory_probes"] == 1
    exported = bench.export(tmp_path / "export")
    assert verify(exported["root"])["counts"]["memory_probes"] == 1


def test_without_margin_does_not_invent_semantic_score(bench):
    spec = _spec()
    spec.pop("observable")
    result = bench.memory_probe(spec)
    assert result["report"]["effect"]["signed_effect"] is None
    assert result["report"]["native"]["answer_text"] is None
    assert result["report"]["effect"]["native_output_measured"]


def test_invalid_budget_refuses_before_native_work(bench, monkeypatch):
    parent = bench.branch.cut

    def forbidden(*_args, **_kwargs):
        pytest.fail("a refused probe executed native work or rollback")

    monkeypatch.setattr(bench, "step", forbidden)
    monkeypatch.setattr(bench, "_restore", forbidden)
    with pytest.raises(BenchError, match="native budget"):
        bench.memory_probe(_spec(max_native_operations=1))
    assert bench.branch.cut == parent


def test_pending_act_refusal_preserves_provisional_state(bench):
    bench.step("embedding")
    preview = bench.preview_edit("residual", 0.9, mode="scale")
    hidden = bench.resolve_value("residual")
    cursor = bench.cursor()
    with pytest.raises(BenchError, match="committed token"):
        bench.memory_probe(_spec())
    assert bench._edit[0] == preview["act_id"]
    assert bench.cursor() == cursor
    assert torch.equal(hidden, bench.resolve_value("residual"))
    bench.abort_edit(preview["act_id"])


def test_resealed_effect_or_query_tampering_refused(bench):
    report = bench.memory_probe(_spec())["report"]
    manifest = bench._manifest()
    changed = deepcopy(report)
    changed["effect"]["signed_effect"] += 1.0
    with pytest.raises(ValueError, match="signed effect"):
        verify_artifact(changed, bench.artifacts, manifest, bench.vm.artifacts)
    changed = deepcopy(report)
    changed["candidate"]["query_token_ids"] = [8]
    with pytest.raises(ValueError, match="paired query"):
        verify_artifact(changed, bench.artifacts, manifest, bench.vm.artifacts)
    changed = deepcopy(report)
    changed["instrument_identity"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source custody"):
        verify_artifact(changed, bench.artifacts, manifest, bench.vm.artifacts)
    changed = deepcopy(report)
    readout = changed["native"]["first_readout"]
    cut = bench.artifacts.get(readout["cut_id"])
    logits = bench._tensor_load(cut["logits"]).float()[0]
    worst = int(logits.argmin())
    readout["top_tokens"][0].update(token_id=worst, raw_logit=float(logits[worst]))
    with pytest.raises(ValueError, match="top-k ranking"):
        verify_artifact(changed, bench.artifacts, manifest, bench.vm.artifacts)
