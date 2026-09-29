"""Reuse declared native normalization while always executing the final readout.

Random CPU Qwen mechanics, not a performance or language-capability benchmark.
"""

import json
from pathlib import Path

import torch

from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.dependencies import Cell, DependencyGraph

adapter = QwenAdapter.tiny()
adapter.model.requires_grad_(False)
session = adapter.session([5, 7, 11])
session.continue_(adapter.layers + 1)
hidden = session.read("hidden")
# The resident weights are immutable; include their identity in implementation versions.
graph = DependencyGraph(
    (
        Cell(
            "normalized",
            ("hidden",),
            lambda values: adapter.model.model.norm(values["hidden"]),
            version=f"{adapter.model_identity}:native-normalization-v1",
        ),
        Cell("label", ("context",), lambda values: values["context"], version="label-v1"),
    )
)
consumer_calls = 0


def consumer(values):
    global consumer_calls
    consumer_calls += 1
    return adapter.model.lm_head(values["normalized"])[:, -1]


rows = []
with torch.inference_mode():
    baseline, trace = graph.run({"hidden": hidden, "context": "original"}, consumer)
    rows.append({"run": "first", **trace})
    repeated, trace = graph.run({"hidden": hidden, "context": "original"}, consumer)
    rows.append({"run": "unchanged", **trace})
    assert torch.equal(baseline, repeated)
    relabelled, trace = graph.run({"hidden": hidden, "context": "new-label"}, consumer)
    rows.append({"run": "changed-label", **trace})
    assert torch.equal(baseline, relabelled)
    changed, trace = graph.run({"hidden": hidden + 0.1, "context": "new-label"}, consumer)
    rows.append({"run": "changed-carrier", **trace})
assert consumer_calls == 4
report = {
    "rows": rows,
    "consumer_calls": consumer_calls,
    "hidden_dirty_closure": sorted(graph.dirty({"hidden"})),
    "changed_carrier_max_logit_difference": float((changed - baseline).abs().max()),
    "claim_boundary": "Declared instance-local reuse; no speedup or semantic claim.",
}
output = Path("outputs/dependency-reuse")
output.mkdir(parents=True, exist_ok=True)
(output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
