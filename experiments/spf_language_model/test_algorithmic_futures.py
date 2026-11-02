"""Controlled mechanics fixtures for nonmonotonic common-parent futures."""

import json
import random
import sys
from pathlib import Path

import pytest
import torch

from saturn_pub.training import EvaluationScores, PromotionObjective, PromotionPolicy

sys.path.insert(0, str(Path(__file__).parent))
from algorithmic_futures import AlgorithmicFutures, FutureRequest


def fixture(tmp_path, *, policy=None, selection_split="feedback"):
    torch.manual_seed(8)
    random.seed(9)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    model.register_buffer("scratch", torch.ones(1), persistent=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    cursor = {"updates": 0, "sampler_position": 0}
    diagnostics = {"latest": torch.ones(2)}
    generator = torch.Generator().manual_seed(19)
    runtime = AlgorithmicFutures(
        model,
        optimizer,
        cursor,
        tmp_path,
        {"model_id": "toy"},
        mutable={
            "diagnostics": (
                lambda: diagnostics["latest"],
                lambda value: diagnostics.update(latest=value),
            )
        },
        generators={"sampler": generator},
        policy=policy,
        evaluation_contract={
            "selection_split": selection_split,
            "input_sha256": "a" * 64,
            "consumer": "controlled-nonmonotonic-fixture-v1",
        },
    )

    def train(index):
        offset = cursor["sampler_position"]
        draw = float(torch.rand((), generator=generator))
        optimizer.zero_grad()
        (-model(torch.ones(1, 1)).sum()).backward()
        optimizer.step()
        model.scratch.add_(1)
        diagnostics["latest"] = torch.tensor([index, draw])
        random.random()
        cursor["updates"] += 1
        cursor["sampler_position"] += 1
        return {"offset": offset, "draw": draw, "step": index + 1}

    def evaluate():
        updates = cursor["updates"]
        value = {0: 1.0, 16: 2.0, 32: 3.0, 64: 0.5}.get(updates, 2.0)
        model.eval()
        cursor["sampler_position"] = 900
        diagnostics["latest"] = torch.full((2,), 800.0)
        torch.rand(3)
        return EvaluationScores(
            validation={"bpb": value},
            autonomous_rollout={"diagnostic": -value},
            trace_endpoints={"updates": float(updates)},
        )

    return runtime, train, evaluate, cursor, model, optimizer, diagnostics


def test_same_parent_16_64_future_keeps_harmful_intermediate_and_commits_endpoint(tmp_path):
    runtime, train, evaluate, cursor, model, optimizer, _ = fixture(tmp_path)
    parent = runtime.capture_parent(evaluate)
    parent_fp = parent["archive"]["fingerprint"]
    rows = runtime.propose_futures(
        [
            FutureRequest("short16", 16, train, (16,)),
            FutureRequest("long64", 64, train, (16, 32, 64)),
        ],
        evaluate,
    )
    assert runtime.physical_attempts == 80
    assert runtime.evaluator_calls == 5  # one parent, short endpoint, two cuts + long endpoint
    assert runtime.boundary.capture().fingerprint == parent_fp
    assert cursor == {"updates": 0, "sampler_position": 0}
    assert not optimizer.state
    assert model.weight.grad is None
    short, long = rows["short16"], rows["long64"]
    assert short["common_parent_fingerprint"] == long["common_parent_fingerprint"] == parent_fp
    assert not short["receipt"]["decision"]["promoted"]
    assert long["receipt"]["decision"]["promoted"]
    assert [r["evaluation"]["validation"]["bpb"] for r in long["observations"]] == [2.0, 3.0, 0.5]
    assert [r["selection_authority"] for r in long["observations"]] == [False, False, True]
    for step in range(1, 17):
        short_update = json.loads(
            (tmp_path / "parent-000" / "short16" / f"update-{step:03d}.json").read_text()
        )
        long_update = json.loads(
            (tmp_path / "parent-000" / "long64" / f"update-{step:03d}.json").read_text()
        )
        assert short_update == long_update  # same sampler position and named RNG draw
    assert short["endpoint_archive"] and Path(short["endpoint_archive"]["path"]).is_file()
    assert long["endpoint_archive"] and Path(long["endpoint_archive"]["path"]).is_file()
    assert runtime.best_future() == "long64"
    commit = runtime.commit_best()
    assert commit["verified_exact"] and commit["continued_future"] == "long64"
    assert commit["post_state_fingerprint"] == long["endpoint_archive"]["fingerprint"]
    assert cursor == {"updates": 64, "sampler_position": 64}
    assert model.training
    assert optimizer.state and model.weight.grad is not None
    assert runtime.physical_attempts == 80


def test_continue_control_uses_explicit_updates_objective_not_quality_gate(tmp_path):
    policy = PromotionPolicy(
        "lexicographic", (PromotionObjective("trace_endpoints", "updates", "maximize"),)
    )
    runtime, train, evaluate, cursor, _, _, _ = fixture(tmp_path, policy=policy)
    runtime.capture_parent(evaluate)
    rows = runtime.propose_futures([FutureRequest("inspect16", 16, train, (8, 16))], evaluate)
    assert rows["inspect16"]["observations"][-1]["evaluation"]["validation"]["bpb"] == 2.0
    assert runtime.commit("inspect16")["continued_future"] == "inspect16"
    assert cursor["updates"] == 16


def test_parent_restore_and_broken_sibling_keep_completed_raw_steps(tmp_path):
    runtime, train, evaluate, cursor, _, _, _ = fixture(tmp_path)
    parent = runtime.capture_parent(evaluate)

    def broken(index):
        metadata = train(index)
        if index == 3:
            raise RuntimeError("controlled future interruption")
        return metadata

    rows = runtime.propose_futures(
        [
            FutureRequest("broken", 16, broken, (8, 16)),
            FutureRequest("other", 16, train, (16,)),
        ],
        evaluate,
    )
    assert rows["broken"]["status"] == "error"
    assert rows["broken"]["partial_archives"]
    assert rows["broken"]["receipt"]["rollback"]["verified_exact"]
    assert (tmp_path / "parent-000" / "broken" / "update-003.json").is_file()
    assert rows["other"]["status"] == "complete"
    assert runtime.physical_attempts == 20
    assert runtime.commit(None)["post_state_fingerprint"] == parent["archive"]["fingerprint"]
    assert cursor["updates"] == 0


def test_endpoint_evaluator_mutation_is_refused_and_parent_restored(tmp_path):
    runtime, train, evaluate, _, model, _, _ = fixture(tmp_path)
    corrupt = {"enabled": False}

    def corrupting_evaluate():
        if corrupt["enabled"]:
            with torch.no_grad():
                model.weight.add_(100)
        return evaluate()

    parent = runtime.capture_parent(corrupting_evaluate)
    corrupt["enabled"] = True
    rows = runtime.propose_futures([FutureRequest("corrupt", 16, train, ())], corrupting_evaluate)
    assert rows["corrupt"]["status"] == "error"
    assert rows["corrupt"]["endpoint_archive"]
    assert runtime.boundary.capture().fingerprint == parent["archive"]["fingerprint"]
    assert runtime.best_future() is None


def test_common_parent_evaluation_binding_refuses_different_scorer_or_contract(tmp_path):
    runtime, train, evaluate, _, _, _, _ = fixture(tmp_path)
    runtime.capture_parent(evaluate)
    with pytest.raises(ValueError, match="binding changed"):
        runtime.propose_futures([FutureRequest("other", 16, train)], lambda: evaluate())
    runtime.evaluation_contract["input_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="binding changed"):
        runtime.propose_futures([FutureRequest("other", 16, train)], evaluate)


@pytest.mark.parametrize("split", ["sealed_test", "sealed_audit", "cold_audit", "heldout", "test"])
def test_cold_evaluation_cannot_select_training_futures(tmp_path, split):
    with pytest.raises(ValueError, match="held-out evaluation"):
        fixture(tmp_path, selection_split=split)
