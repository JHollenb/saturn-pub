"""Measure a write, bind its replay to the exact recipient, and refuse unsupported use."""

import json
from pathlib import Path

import torch

from saturn_pub import Act
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.investigation import Arm, Investigation
from saturn_pub.program import Instruction, Program
from saturn_pub.values import describe

adapter = QwenAdapter.tiny()
session = adapter.session([5, 7, 11])
session.continue_(adapter.layers + 1)
act = Act.add("hidden", torch.ones_like(session.read("hidden")) * 0.05)
plan = Investigation(
    "What does this carrier write change at native readout?",
    (Arm("native", role="native"), Arm("candidate", (act,))),
    steps=1,
)
panel = plan.run(
    session,
    lambda branch: {
        "next_token": int(branch.read("tokens")[0, -1]),
        "logit_max": float(branch.read("logits").max()),
    },
)
program = Program.compile(
    session, panel, "candidate", (Instruction(0, act),), context="known-development-context"
)
replayed = program.run(session, context="known-development-context")
candidate = next(row for row in panel["rows"] if row["name"] == "candidate")
exact = describe(replayed.capture().payload) == candidate["output"]
assert exact
refusals = {}
for name, target, context in (
    ("wrong-context", session, "other-context"),
    ("different-parent", adapter.session([13, 17, 19]), "known-development-context"),
):
    try:
        program.run(target, context=context)
    except ValueError as error:
        refusals[name] = str(error)
    else:
        raise RuntimeError("unsupported recipient/context was admitted")
report = {
    "panel": panel,
    "measured_payload_replayed_exactly": exact,
    "unsupported_use_refusals": refusals,
    "claim_boundary": "Random native CPU model; one exact parent/context, no learned portability.",
}
output = Path("outputs/measured-program")
output.mkdir(parents=True, exist_ok=True)
(output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({key: value for key, value in report.items() if key != "panel"}, indent=2))
