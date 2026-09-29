"""Train a small carrier writer through frozen Qwen readout; reject a harmful future.

The desired token is supplied synthetic feedback, not learned natural-language semantics.
The two fresh context probes at the end never enter optimizer or continuation selection.
"""

import json
from pathlib import Path

import torch

from saturn_pub import Act
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.training import (
    EvaluationScores,
    ImmutableDirectoryReceiptStore,
    PromotionObjective,
    PromotionPolicy,
    TrainingTransaction,
)

adapter = QwenAdapter.tiny()
adapter.model.requires_grad_(False)
session = adapter.session([5, 7, 11])
session.continue_(adapter.layers + 1)
hidden = session.read("hidden").clone()  # leave inference tensor mode for gradient consumers
writer = torch.nn.Linear(32, 32)
torch.nn.init.zeros_(writer.weight)
torch.nn.init.zeros_(writer.bias)
optimizer = torch.optim.SGD(writer.parameters(), lr=0.1)
cursor = {"updates": 0}
cache = {"last_loss": None}
target = torch.tensor([3])


def loss():
    changed = hidden + writer(hidden)
    logits = adapter.model.lm_head(adapter.model.model.norm(changed))[:, -1]
    return torch.nn.functional.cross_entropy(logits, target)


@torch.no_grad()
def evaluate():
    delta = writer(hidden)
    branch = session.fork()
    branch.apply(Act.add("hidden", delta))
    adapter.generate(branch, tokens=2)
    return EvaluationScores(
        validation={"loss": float(loss())},
        autonomous_rollout={
            "first_token_correct": float(branch.read("tokens")[0, -2] == target[0])
        },
        trace_endpoints={"write_norm": float(delta.norm())},
        evidence={"feedback": "supplied-token-3", "heldout_used_for_selection": False},
    )


def restore_cursor(value):
    cursor.clear()
    cursor.update(value)


controller = TrainingTransaction(
    controller_id="offline-carrier-writer",
    model=writer,
    optimizer=optimizer,
    policy=PromotionPolicy(
        "lexicographic", (PromotionObjective("validation", "loss", "minimize"),)
    ),
    cursor_capture=lambda: cursor,
    cursor_restore=restore_cursor,
    mutable={"cache": (lambda: cache, lambda value: (cache.clear(), cache.update(value)))},
    max_interval_steps=2,
    receipt_store=ImmutableDirectoryReceiptStore(Path("outputs/training/receipts")),
)
controller.establish_baseline(evaluate)


def update(sign):
    def step(_):
        optimizer.zero_grad(set_to_none=True)
        objective = loss()
        (objective * sign).backward()
        optimizer.step()
        cursor["updates"] += 1
        cache["last_loss"] = float(objective.detach())
        return {"feedback_loss": cache["last_loss"], "direction": sign}

    return step


accepted = controller.run_interval(
    interval_id="helpful", steps=2, train_step=update(1), evaluator=evaluate
)
rejected = controller.run_interval(
    interval_id="harmful", steps=1, train_step=update(-1), evaluator=evaluate
)
assert accepted.to_dict()["decision"]["promoted"]
assert not rejected.to_dict()["decision"]["promoted"]
assert rejected.to_dict()["rollback"]["verified_exact"]
heldout = []
with torch.no_grad():
    for tokens in ([13, 17, 19], [23, 29, 31]):
        branch = adapter.session(tokens)
        branch.continue_(adapter.layers + 1)
        branch.apply(Act.add("hidden", writer(branch.read("hidden"))))
        adapter.generate(branch, tokens=2)
        heldout.append(branch.read("tokens").tolist())
report = {
    "accepted": accepted.to_dict(),
    "rejected": rejected.to_dict(),
    "heldout_context_probes": heldout,
    "supplied_feedback_token": 3,
}
Path("outputs/training/report.json").write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {
            "helpful": accepted.to_dict()["decision"],
            "harmful": rejected.to_dict()["decision"],
            "restored": rejected.to_dict()["rollback"],
            "heldout": heldout,
        },
        indent=2,
    )
)
