"""Causal discovery mechanics on a tiny, untrained native Qwen organism."""

from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchError, BenchSession
from mdb.bench_circuit_discovery import _ReadObserver


@pytest.fixture
def bench(tmp_path):
    torch.manual_seed(603)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attn_implementation="eager",
    )).eval().to(torch.bfloat16)
    session = BenchSession.from_resident(model, root=tmp_path / "bench", continuation_capacity=16,
                                         attention_backend="native-eager")
    session.prefill([1, 2, 3])
    yield session
    session.close()


def spec(**changes):
    result = {"plan": [{"kind": "input", "token_ids": [7]},
                       {"kind": "generate", "count": 1},
                       {"kind": "input", "token_ids": [8]},
                       {"kind": "generate", "count": 1}],
              "receiver_offsets": [0, 2], "candidates_per_receiver": 1,
              "scales": [0.0, 0.5], "observable": {
                  "kind": "token_margin", "token_id": 9, "against_token_id": 11}}
    result.update(changes)
    return result


def test_native_reads_head_local_edits_and_own_history_followup(bench):
    parent = bench.branch.cut
    before = bench.resolve_value("kv_cache.layer_000.value")
    bench.fork("sibling", cut=parent)
    bench.checkout("main")
    parent = bench.branch.cut
    sibling = bench.branches["sibling"].cut
    report = bench.discover_circuit(spec())["report"]
    assert bench.branch.cut == parent
    assert bench.branches["sibling"].cut == sibling
    assert torch.equal(before, bench.resolve_value("kv_cache.layer_000.value"))
    assert report["observer_control"]["effect"]["closure_exact"]
    assert len(report["receivers"]) == 2
    assert len(report["tests"]) == 4
    assert report["receivers"][0]["parent_cut"] != report["receivers"][1]["parent_cut"]
    for receiver in report["receivers"]:
        assert len(receiver["coverage"]) == 2
        assert all(row["attention_reconstruction_max_abs"] == 0 for row in receiver["coverage"])
        for candidate in receiver["candidates"]:
            assert candidate["reinstall"]["effect"]["closure_exact"]
            assert not candidate["vocabulary"]["semantic_cot_qualified"]
    for test in report["tests"]:
        assert test["local_prediction"]["query_and_keys_exact"]
        prediction = bench.artifacts.get(test["local_prediction"]["prediction_ref"])
        assert prediction["scale"] == test["scale"]
        assert prediction["candidate_id"] == test["candidate_id"]
        locality = test["head_locality"]
        assert locality["untouched_heads_before_sha256"] == locality["untouched_heads_after_sha256"]
        a = bench._tensor_load(locality["after_ref"])
        b = bench._tensor_load(locality["before_ref"])
        head = locality["selected_head"]
        assert torch.equal(a[:, head:head + 1], b[:, head:head + 1] * test["scale"])
        trajectory = test["candidate"]["trajectory"]
        for left, right in zip(trajectory, trajectory[1:]):
            if right["input_authority"] == "native-greedy":
                assert right["consumed_token_id"] == left["next_token_id"]
        followup = next(row for row in trajectory if row["offset"] == 2)
        assert followup["consumed_token_id"] == 8
        assert followup["input_authority"] == "caller-supplied"
    assert report["semantic_cot_qualified"] is False
    assert report["terminal_status"] == "not-assessed"
    assert len(bench.inspect("circuits.discovery")) == 1
    assert bench.capabilities()["circuit_discovery"]["available"]
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in bench.model.modules())
    from mdb.bench_verify import verify
    assert verify(bench.root)["counts"]["circuit_discovery_artifacts"] == 1


@pytest.mark.parametrize("changes", [
    {"max_native_operations": 1}, {"row_stop": 9}, {"receiver_offsets": [4]},
    {"scales": [float("nan")]}, {"plan": [{"kind": "generate", "count": True}]},
    {"layers": [0, 0]}, {"max_capture_bytes": 1}, {"unknown": True},
])
def test_invalid_spec_rejected_before_execution_or_restore(bench, monkeypatch, changes):
    parent = bench.branch.cut

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid discovery performed native work or rollback")

    monkeypatch.setattr(bench, "step", forbidden)
    monkeypatch.setattr(bench, "_restore", forbidden)
    with pytest.raises(BenchError):
        bench.discover_circuit(spec(**changes))
    assert bench.branch.cut == parent


def test_refused_pending_act_preserves_provisional_state(bench):
    bench.step("embedding")
    preview = bench.preview_edit("residual", 0.5, mode="scale")
    cursor = bench.cursor()
    hidden = bench.resolve_value("residual")
    with pytest.raises(BenchError, match="committed token"):
        bench.discover_circuit(spec())
    assert bench.cursor() == cursor
    assert torch.equal(hidden, bench.resolve_value("residual"))
    assert bench._edit[0] == preview["act_id"]
    bench.abort_edit(preview["act_id"])


def test_segmented_backend_is_not_advertised_as_the_qualified_read_program(bench, monkeypatch):
    monkeypatch.setattr(bench.vm.runtime, "backend", "torch")
    assert bench.capabilities()["circuit_discovery"]["available"] is False
    with pytest.raises(BenchError, match="native-eager"):
        bench.discover_circuit(spec())


def test_hook_failure_restores_parent_registry_and_removes_observer(bench, monkeypatch):
    parent, cuts, events = bench.branch.cut, deepcopy(bench.cuts), deepcopy(bench.events)
    original = bench.resolve_value("kv_cache.layer_000.value")

    def fail(*_args):
        raise RuntimeError("observer failure")

    monkeypatch.setattr(_ReadObserver, "_observe", fail)
    with pytest.raises(RuntimeError, match="observer failure"):
        bench.discover_circuit(spec())
    assert bench.branch.cut == parent
    assert bench.cuts == cuts and bench.events == events
    assert torch.equal(original, bench.resolve_value("kv_cache.layer_000.value"))
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in bench.model.modules())


def test_observational_rank_does_not_invent_task_semantics(bench):
    config = spec(scales=[1.0], receiver_offsets=[0], continuation_tokens=1)
    config.pop("observable")
    report = bench.discover_circuit(config)["report"]
    effect = report["tests"][0]["effect"]
    assert effect["closure_exact"]
    assert report["tests"][0]["local_prediction"]["error_max_abs"] == 0
    assert effect["steps"][0]["signed_margin_effect"] is None
    assert report["graph"]["semantic_cot_qualified"] is False
    assert all(edge["kind"] != "semantic-operation" for edge in report["graph"]["edges"])


def test_same_carried_operand_is_bound_to_new_queries_on_later_own_states(bench):
    report = bench.discover_circuit(spec(
        layers=[1], row_start=0, row_stop=1, candidates_per_receiver=2,
        scales=[1.0], continuation_tokens=1,
    ))["report"]
    first, later = report["receivers"]
    by_head = {c["kv_head"]: c for c in first["candidates"]}
    for candidate in later["candidates"]:
        earlier = by_head[candidate["kv_head"]]
        assert earlier["value_sha256"] == candidate["value_sha256"]
        assert earlier["key_sha256"] == candidate["key_sha256"]
        assert earlier["query_heads"][0]["query_sha256"] != candidate["query_heads"][0]["query_sha256"]
    carried = [edge for edge in report["graph"]["edges"] if edge["kind"] == "same-payload-observation"]
    assert len(carried) == 2
    assert all(edge["causal_dependency_tested"] is False for edge in carried)


def test_native_operation_bound_counts_witness_and_candidate_as_one_future(bench, monkeypatch):
    advances = 0
    native = bench.vm.runtime.advance_token

    def counted(execution):
        nonlocal advances
        advances += 1
        return native(execution)

    monkeypatch.setattr(bench.vm.runtime, "advance_token", counted)
    report = bench.discover_circuit(spec())["report"]
    assert advances == report["operation_upper_bound"] == 52
    assert all(edge["causal_dependency_tested"] is False for edge in report["graph"]["edges"]
               if edge["kind"] == "observed-native-read")
