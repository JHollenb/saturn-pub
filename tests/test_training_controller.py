"""Exact-state and decision-custody tests for bounded training intervals."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import saturn_pub.training._checkpoint as checkpoint_controller
from saturn_pub.training._checkpoint import (
    DecisionReceiptError,
    DeterministicCheckpointController,
    EvaluationMutationError,
    EvaluationScores,
    ImmutableDirectoryReceiptStore,
    PromotionObjective,
    PromotionPolicy,
    TrainingCheckpointError,
    TrainingIntervalError,
    TrainingStateBoundary,
    TrainingStateDriftError,
    TrainingStateSnapshot,
)


class _TinyStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)
        self.register_buffer("running_probe", torch.tensor([3.0]), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class _TiedStudent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 3)
        self.readout = torch.nn.Linear(3, 4, bias=False)
        self.readout.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.readout(self.embedding(token_ids))


def _policy(kind: str = "lexicographic") -> PromotionPolicy:
    return PromotionPolicy(
        kind,  # type: ignore[arg-type]
        (
            PromotionObjective("validation", "loss", "minimize"),
            PromotionObjective("autonomous_rollout", "exact_trajectory_rate", "maximize"),
            PromotionObjective("trace_endpoints", "endpoint_match_rate", "maximize"),
        ),
    )


def _scores(
    *, loss: float, rollout: float, trace: float, specimen: str = "sealed-validation"
) -> EvaluationScores:
    return EvaluationScores(
        validation={"loss": loss},
        autonomous_rollout={"exact_trajectory_rate": rollout},
        trace_endpoints={"endpoint_match_rate": trace},
        evidence={
            "specimen": specimen,
            "validation_split_used_for_updates": False,
            "teacher_execution": False,
        },
    )


def _controller(
    *,
    policy: PromotionPolicy | None = None,
    receipt_store: ImmutableDirectoryReceiptStore | None = None,
) -> tuple[
    DeterministicCheckpointController,
    _TinyStudent,
    torch.optim.Optimizer,
    dict[str, int],
    torch.Generator,
]:
    model = _TinyStudent()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    cursor = {"update": 0, "epoch": 0}
    generator = torch.Generator(device="cpu").manual_seed(404)

    def restore_cursor(value: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(value)

    controller = DeterministicCheckpointController(
        controller_id="tiny-student-controller",
        model=model,
        optimizer=optimizer,
        policy=policy or _policy(),
        cursor_capture=lambda: dict(cursor),
        cursor_restore=restore_cursor,
        max_interval_steps=4,
        torch_generators={"row-schedule": generator},
        receipt_store=receipt_store,
    )
    return controller, model, optimizer, cursor, generator


def _optimizer_update(
    model: _TinyStudent,
    optimizer: torch.optim.Optimizer,
    cursor: dict[str, int],
    generator: torch.Generator,
    steps: int,
) -> dict[str, Any]:
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        noise = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(()))
            + float(torch.rand((), generator=generator))
        )
        output = model(torch.tensor([[1.0, -1.0]])).sum()
        loss = (output - noise).square()
        loss.backward()
        optimizer.step()
        model.running_probe.add_(1.0)
        cursor["update"] += 1
        losses.append(float(loss.detach()))
    return {"optimizer_updates": steps, "last_loss": losses[-1]}


def _rng_draw(generator: torch.Generator) -> tuple[float, float, float, float]:
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )


def _reset_seeds() -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)


def test_rng_capture_eagerly_stabilizes_available_cuda_inventory(monkeypatch) -> None:
    initialized = False
    state = torch.tensor([17, 23], dtype=torch.uint8)

    def get_rng_state_all() -> list[torch.Tensor]:
        nonlocal initialized
        initialized = True
        return [state.clone()]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", get_rng_state_all)

    captured = checkpoint_controller._capture_rng({})

    assert initialized is True
    assert len(captured["torch_cuda"]) == 1
    assert torch.equal(captured["torch_cuda"][0], state)


def test_policy_can_require_all_objectives_and_never_scalarizes() -> None:
    with pytest.raises(ValueError, match="must score validation"):
        PromotionPolicy(
            "lexicographic",
            (PromotionObjective("validation", "loss", "minimize"),),
            require_all_score_sources=True,
        )
    with pytest.raises(ValueError, match="must be unique"):
        PromotionPolicy(
            "pareto",
            (
                PromotionObjective("validation", "loss", "minimize"),
                PromotionObjective("validation", "loss", "minimize"),
                PromotionObjective("autonomous_rollout", "exact", "maximize"),
                PromotionObjective("trace_endpoints", "match", "maximize"),
            ),
        )

    baseline = _scores(loss=1.0, rollout=0.5, trace=0.5)
    primary_better = _scores(loss=0.9, rollout=0.0, trace=0.0)
    lexicographic = _policy("lexicographic").compare(baseline, primary_better)
    assert lexicographic["promoted"] is True
    assert lexicographic["decisive_objective_index"] == 0
    assert lexicographic["scalarized_score_used"] is False

    pareto = _policy("pareto")
    dominating = pareto.compare(baseline, _scores(loss=0.9, rollout=0.6, trace=0.5))
    tradeoff = pareto.compare(baseline, _scores(loss=0.9, rollout=0.4, trace=0.6))
    assert dominating["promoted"] is True
    assert dominating["relation"] == "pareto-dominates-baseline"
    assert tradeoff["promoted"] is False
    assert tradeoff["relation"] == "pareto-tradeoff"


def test_regression_restores_model_optimizer_rng_cursor_gradients_and_modes(
    tmp_path: Path,
) -> None:
    _reset_seeds()
    store = ImmutableDirectoryReceiptStore(tmp_path / "receipts")
    controller, model, optimizer, cursor, generator = _controller(receipt_store=store)
    score = {"loss": 1.0, "rollout": 0.5, "trace": 0.5}

    def evaluator() -> EvaluationScores:
        model.eval()
        cursor["update"] += 99
        _rng_draw(generator)
        return _scores(
            loss=score["loss"],
            rollout=score["rollout"],
            trace=score["trace"],
        )

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    generator_state = generator.get_state()
    expected_next_draw = _rng_draw(generator)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    generator.set_state(generator_state)

    baseline_receipt = controller.establish_baseline(evaluator)
    baseline_fingerprint = controller.accepted_state_fingerprint
    baseline_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    assert model.training is True
    assert cursor == {"update": 0, "epoch": 0}
    assert optimizer.state == {}

    score.update(loss=2.0, rollout=0.2, trace=0.1)
    receipt = controller.run_interval(
        interval_id="regression-0001",
        steps=2,
        train_step=lambda _step: _optimizer_update(model, optimizer, cursor, generator, 1),
        evaluator=evaluator,
    )

    payload = receipt.to_dict()
    assert payload["decision"] == {
        "action": "restore-prior",
        "promoted": False,
        "reason": "lexicographic-regression",
    }
    assert payload["rollback"] == {"required": True, "verified_exact": True}
    assert payload["candidate"]["state_fingerprint"] != baseline_fingerprint
    assert payload["train_metadata"]["step_callback_invocations"] == 2
    assert len(payload["train_metadata"]["step_receipts"]) == 2
    assert payload["post_state_fingerprint"] == baseline_fingerprint
    assert controller.current_state_fingerprint() == baseline_fingerprint
    assert controller.accepted_state_fingerprint == baseline_fingerprint
    assert cursor == {"update": 0, "epoch": 0}
    assert optimizer.state == {}
    assert model.running_probe.item() == 3.0
    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(parameter, baseline_parameters[name])
        for name, parameter in model.named_parameters()
    )
    assert _rng_draw(generator) == expected_next_draw
    assert receipt.to_dict()["previous_receipt_fingerprint"] == baseline_receipt.fingerprint
    assert controller.ledger.verify() is True
    assert len(tuple((tmp_path / "receipts").glob("*.json"))) == 2

    # Idempotent publication is allowed; a byte-changing overwrite is not.
    destination = store.put(receipt)
    destination.write_bytes(b"tampered")
    with pytest.raises(DecisionReceiptError, match="different bytes"):
        store.put(receipt)


def test_policy_free_boundary_round_trips_a_safe_snapshot_payload() -> None:
    _reset_seeds()
    model = _TinyStudent()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    cursor = {"update": 0}
    generator = torch.Generator(device="cpu").manual_seed(404)

    def restore_cursor(value: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(value)

    boundary = TrainingStateBoundary(
        model=model,
        optimizer=optimizer,
        cursor_capture=lambda: dict(cursor),
        cursor_restore=restore_cursor,
        torch_generators={"row-schedule": generator},
    )
    snapshot = boundary.capture()
    restored = TrainingStateSnapshot.from_state_dict(snapshot.to_state_dict())
    with torch.no_grad():
        model.projection.weight.add_(10.0)
    cursor["update"] = 7

    assert boundary.restore(restored) == snapshot.fingerprint
    assert boundary.current_state_fingerprint() == snapshot.fingerprint
    assert cursor == {"update": 0}


def test_snapshot_payload_rejects_tampered_tensor_before_restore() -> None:
    _reset_seeds()
    controller, model, _optimizer, _cursor, _generator = _controller()
    snapshot = controller.capture()
    payload = snapshot.to_state_dict()
    payload["model_state"]["projection.weight"].add_(1.0)

    with pytest.raises(TrainingCheckpointError, match="fingerprint changed"):
        TrainingStateSnapshot.from_state_dict(payload)
    assert torch.equal(model.projection.weight, payload["model_state"]["projection.weight"] - 1.0)


def test_promotion_becomes_the_only_rollback_target_for_later_intervals() -> None:
    _reset_seeds()
    controller, model, optimizer, cursor, generator = _controller()
    score = {"loss": 1.0, "rollout": 0.4, "trace": 0.4}

    def evaluator() -> EvaluationScores:
        _rng_draw(generator)
        return _scores(
            loss=score["loss"],
            rollout=score["rollout"],
            trace=score["trace"],
        )

    controller.establish_baseline(evaluator)
    original = controller.accepted_state_fingerprint
    original_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    score.update(loss=0.8, rollout=0.6, trace=0.7)
    promoted = controller.run_interval(
        interval_id="improvement-0001",
        steps=1,
        train_step=lambda _step: _optimizer_update(model, optimizer, cursor, generator, 1),
        evaluator=evaluator,
    )
    promoted_fingerprint = controller.accepted_state_fingerprint
    promoted_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    promoted_cursor = dict(cursor)
    assert promoted.to_dict()["decision"]["promoted"] is True
    assert promoted_fingerprint != original
    assert cursor["update"] == 1
    assert optimizer.state
    assert any(
        not torch.equal(parameter, original_parameters[name])
        for name, parameter in model.named_parameters()
    )

    score.update(loss=1.2, rollout=1.0, trace=1.0)
    rejected = controller.run_interval(
        interval_id="regression-0002",
        steps=1,
        train_step=lambda _step: _optimizer_update(model, optimizer, cursor, generator, 1),
        evaluator=evaluator,
    )
    assert rejected.to_dict()["decision"]["promoted"] is False
    assert controller.current_state_fingerprint() == promoted_fingerprint
    assert controller.accepted_state_fingerprint == promoted_fingerprint
    assert cursor == promoted_cursor
    assert all(
        torch.equal(parameter, promoted_parameters[name])
        for name, parameter in model.named_parameters()
    )
    assert optimizer.state
    assert controller.ledger.verify() is True
    assert len(controller.ledger.receipts) == 3


def test_pareto_tradeoff_is_restored_even_when_one_metric_improves() -> None:
    _reset_seeds()
    controller, model, optimizer, cursor, generator = _controller(policy=_policy("pareto"))
    score = {"loss": 1.0, "rollout": 0.5, "trace": 0.5}

    def evaluator() -> EvaluationScores:
        return _scores(
            loss=score["loss"],
            rollout=score["rollout"],
            trace=score["trace"],
        )

    controller.establish_baseline(evaluator)
    baseline = controller.accepted_state_fingerprint
    score.update(loss=0.8, rollout=0.4, trace=0.7)
    receipt = controller.run_interval(
        interval_id="pareto-tradeoff",
        steps=1,
        train_step=lambda _step: _optimizer_update(model, optimizer, cursor, generator, 1),
        evaluator=evaluator,
    )
    payload = receipt.to_dict()
    assert payload["comparison"]["relation"] == "pareto-tradeoff"
    assert payload["decision"]["promoted"] is False
    assert controller.current_state_fingerprint() == baseline


def test_training_exception_is_rolled_back_and_receipted() -> None:
    _reset_seeds()
    controller, model, optimizer, cursor, generator = _controller()
    controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    baseline = controller.accepted_state_fingerprint

    def fail_after_update(_step: int) -> None:
        _optimizer_update(model, optimizer, cursor, generator, 1)
        raise RuntimeError("synthetic optimizer failure")

    with pytest.raises(TrainingIntervalError) as caught:
        controller.run_interval(
            interval_id="failed-update",
            steps=1,
            train_step=fail_after_update,
            evaluator=lambda: _scores(loss=0.5, rollout=0.8, trace=0.8),
        )
    payload = caught.value.receipt.to_dict()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert payload["kind"] == "interval-error"
    assert payload["decision"]["action"] == "restore-prior-after-error"
    assert payload["decision"]["error_type"] == "RuntimeError"
    assert payload["rollback"]["verified_exact"] is True
    assert controller.current_state_fingerprint() == baseline
    assert len(controller.ledger.receipts) == 2


def test_evaluator_cannot_smuggle_a_model_mutation_into_promotion() -> None:
    _reset_seeds()
    controller, model, optimizer, cursor, generator = _controller()
    score = {"mutate": False}

    def evaluator() -> EvaluationScores:
        if score["mutate"]:
            with torch.no_grad():
                model.projection.weight.add_(100.0)
        return _scores(loss=0.5, rollout=0.8, trace=0.8)

    controller.establish_baseline(evaluator)
    baseline = controller.accepted_state_fingerprint
    score["mutate"] = True
    with pytest.raises(TrainingIntervalError) as caught:
        controller.run_interval(
            interval_id="mutating-evaluator",
            steps=1,
            train_step=lambda _step: _optimizer_update(model, optimizer, cursor, generator, 1),
            evaluator=evaluator,
        )
    assert isinstance(caught.value.__cause__, EvaluationMutationError)
    assert controller.current_state_fingerprint() == baseline
    assert caught.value.receipt.to_dict()["rollback"]["verified_exact"] is True


def test_live_drift_is_restored_before_an_interval_can_run() -> None:
    _reset_seeds()
    controller, model, _optimizer, _cursor, _generator = _controller()
    controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    baseline = controller.accepted_state_fingerprint
    with torch.no_grad():
        model.projection.bias.add_(1.0)
    called = False

    def train(_step: int) -> None:
        nonlocal called
        called = True

    with pytest.raises(TrainingStateDriftError, match="prior restored"):
        controller.run_interval(
            interval_id="drifted",
            steps=1,
            train_step=train,
            evaluator=lambda: _scores(loss=0.5, rollout=0.8, trace=0.8),
        )
    assert called is False
    assert controller.current_state_fingerprint() == baseline
    assert len(controller.ledger.receipts) == 1


def test_optimizer_topology_and_defaults_are_inside_the_rollback_boundary() -> None:
    _reset_seeds()
    controller, model, optimizer, _cursor, _generator = _controller()
    controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    baseline = controller.accepted_state_fingerprint
    original_parameters = tuple(optimizer.param_groups[0]["params"])
    original_default_lr = optimizer.defaults["lr"]

    def corrupt_optimizer_topology(_step: int) -> dict[str, bool]:
        optimizer.param_groups[0]["params"] = list(reversed(optimizer.param_groups[0]["params"]))
        optimizer.defaults["lr"] = 999.0
        return {"deliberate_topology_probe": True}

    receipt = controller.run_interval(
        interval_id="optimizer-topology-regression",
        steps=1,
        train_step=corrupt_optimizer_topology,
        evaluator=lambda: _scores(loss=2.0, rollout=0.2, trace=0.2),
    )
    assert receipt.to_dict()["decision"]["promoted"] is False
    assert controller.current_state_fingerprint() == baseline
    assert tuple(optimizer.param_groups[0]["params"]) == original_parameters
    assert optimizer.defaults["lr"] == original_default_lr

    external = torch.nn.Parameter(torch.ones(()))
    external_optimizer = torch.optim.SGD([external], lr=0.1)
    with pytest.raises(TrainingCheckpointError, match="outside the model"):
        DeterministicCheckpointController(
            controller_id="external-optimizer",
            model=model,
            optimizer=external_optimizer,
            policy=_policy(),
            cursor_capture=lambda: 0,
            cursor_restore=lambda _value: None,
            max_interval_steps=1,
        )


def test_controller_owns_the_exact_number_of_bounded_step_callbacks() -> None:
    _reset_seeds()
    controller, _model, _optimizer, _cursor, _generator = _controller()
    controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    observed = []

    def step(step_index: int) -> dict[str, int]:
        observed.append(step_index)
        return {"step_index": step_index}

    receipt = controller.run_interval(
        interval_id="bounded-four",
        steps=4,
        train_step=step,
        evaluator=lambda: _scores(loss=1.0, rollout=0.5, trace=0.5),
    )
    assert observed == [0, 1, 2, 3]
    metadata = receipt.to_dict()["train_metadata"]
    assert metadata["step_callback_invocations"] == 4
    assert metadata["step_receipts"] == [
        {"step_index": 0},
        {"step_index": 1},
        {"step_index": 2},
        {"step_index": 3},
    ]


def test_tied_state_dict_entries_share_one_physical_snapshot_payload() -> None:
    _reset_seeds()
    model = _TiedStudent()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    cursor = {"update": 0}

    def restore_cursor(value: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(value)

    controller = DeterministicCheckpointController(
        controller_id="tied-student-controller",
        model=model,
        optimizer=optimizer,
        policy=_policy(),
        cursor_capture=lambda: dict(cursor),
        cursor_restore=restore_cursor,
        max_interval_steps=1,
    )
    baseline = controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    state = baseline.to_dict()["candidate"]["state"]
    assert state["model_state_entries"] == 2
    assert state["tensor_inventory"]["model"] == {
        "tensors": 1,
        "elements": 12,
        "bytes": 48,
    }
    initial_weight = model.embedding.weight.detach().clone()

    def step(_step: int) -> None:
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([0, 1])).sum().backward()
        optimizer.step()
        cursor["update"] += 1

    receipt = controller.run_interval(
        interval_id="tied-regression",
        steps=1,
        train_step=step,
        evaluator=lambda: _scores(loss=2.0, rollout=0.2, trace=0.2),
    )
    assert receipt.to_dict()["decision"]["promoted"] is False
    assert torch.equal(model.embedding.weight, initial_weight)
    assert model.embedding.weight is model.readout.weight


def test_rejected_candidate_needs_four_full_fingerprint_passes_and_no_clone() -> None:
    _reset_seeds()
    controller, _model, _optimizer, _cursor, _generator = _controller()
    controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    original_fingerprint = controller.current_state_fingerprint
    calls = 0

    def counted_fingerprint() -> str:
        nonlocal calls
        calls += 1
        return original_fingerprint()

    controller.current_state_fingerprint = counted_fingerprint  # type: ignore[method-assign]
    receipt = controller.run_interval(
        interval_id="four-pass-rejection",
        steps=1,
        train_step=lambda _step: None,
        evaluator=lambda: _scores(loss=1.0, rollout=0.5, trace=0.5),
    )
    payload = receipt.to_dict()
    assert calls == 4
    assert payload["decision"]["promoted"] is False
    assert payload["candidate"]["checkpoint"] is None


def test_receipt_store_failure_rolls_back_an_uncommitted_promotion() -> None:
    class _FailSecondReceipt:
        def __init__(self) -> None:
            self.calls = 0

        def put(self, _receipt: Any) -> None:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("durable receipt unavailable")

    _reset_seeds()
    sink = _FailSecondReceipt()
    controller, model, optimizer, cursor, generator = _controller()
    controller.receipt_store = sink
    controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    baseline = controller.accepted_state_fingerprint
    with pytest.raises(RuntimeError, match="durable receipt unavailable"):
        controller.run_interval(
            interval_id="unreceipted-promotion",
            steps=1,
            train_step=lambda _step: _optimizer_update(model, optimizer, cursor, generator, 1),
            evaluator=lambda: _scores(loss=0.5, rollout=0.8, trace=0.8),
        )
    assert controller.current_state_fingerprint() == baseline
    assert controller.accepted_state_fingerprint == baseline
    assert len(controller.ledger.receipts) == 1


def test_receipts_are_deterministic_for_identical_state_and_decisions() -> None:
    def run_once() -> tuple[str, str]:
        _reset_seeds()
        controller, model, optimizer, cursor, generator = _controller()
        baseline = controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
        interval = controller.run_interval(
            interval_id="deterministic-regression",
            steps=1,
            train_step=lambda _step: _optimizer_update(model, optimizer, cursor, generator, 1),
            evaluator=lambda: _scores(loss=2.0, rollout=0.2, trace=0.2),
        )
        return baseline.fingerprint, interval.fingerprint

    assert run_once() == run_once()


def test_score_and_interval_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        _scores(loss=float("nan"), rollout=0.5, trace=0.5)
    _reset_seeds()
    controller, _model, _optimizer, _cursor, _generator = _controller()
    with pytest.raises(ValueError, match="at least one score"):
        controller.establish_baseline(
            lambda: {
                "validation": {},
                "autonomous_rollout": {"exact_trajectory_rate": 0.5},
                "trace_endpoints": {"endpoint_match_rate": 0.5},
            }
        )
    controller.establish_baseline(lambda: _scores(loss=1.0, rollout=0.5, trace=0.5))
    with pytest.raises(ValueError, match="between 1 and 4"):
        controller.run_interval(
            interval_id="too-long",
            steps=5,
            train_step=lambda _step: None,
            evaluator=lambda: _scores(loss=1.0, rollout=0.5, trace=0.5),
        )
    receipt = controller.run_interval(
        interval_id="one-use",
        steps=1,
        train_step=lambda _step: None,
        evaluator=lambda: _scores(loss=1.0, rollout=0.5, trace=0.5),
    )
    payload = json.loads(receipt.canonical_bytes)
    assert payload["fingerprint"] == receipt.fingerprint
    with pytest.raises(ValueError, match="unique"):
        controller.run_interval(
            interval_id="one-use",
            steps=1,
            train_step=lambda _step: None,
            evaluator=lambda: _scores(loss=1.0, rollout=0.5, trace=0.5),
        )
