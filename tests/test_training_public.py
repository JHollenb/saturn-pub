import copy

import torch

from saturn_pub.training import (
    EvaluationScores,
    PromotionObjective,
    PromotionPolicy,
    TrainingTransaction,
)


def test_registered_hook_cache_restores_with_rejected_update():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    cursor = {"position": 0}
    cache = {"carrier": torch.tensor([1.0])}

    def restore(target, value):
        target.clear()
        target.update(value)

    controller = TrainingTransaction(
        controller_id="registered-closure",
        model=model,
        optimizer=optimizer,
        policy=PromotionPolicy(
            "lexicographic", (PromotionObjective("validation", "loss", "minimize"),)
        ),
        max_interval_steps=1,
        cursor_capture=lambda: cursor,
        cursor_restore=lambda value: restore(cursor, value),
        mutable={"hook-cache": (lambda: cache, lambda value: restore(cache, value))},
    )

    def scores():
        return EvaluationScores(
            validation={"loss": float(cursor["position"])},
            autonomous_rollout={"value": 0.0},
            trace_endpoints={"norm": float(cache["carrier"].norm())},
        )

    controller.establish_baseline(scores)
    accepted = controller.accepted_state_fingerprint
    weights = copy.deepcopy(model.state_dict())

    def update(_):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        cursor["position"] += 1
        cache["carrier"] += 10

    receipt = controller.run_interval(
        interval_id="rejected", steps=1, train_step=update, evaluator=scores
    )
    assert receipt.to_dict()["rollback"]["verified_exact"]
    assert controller.current_state_fingerprint() == accepted
    assert cursor == {"position": 0}
    assert torch.equal(cache["carrier"], torch.tensor([1.0]))
    assert all(torch.equal(model.state_dict()[key], value) for key, value in weights.items())
