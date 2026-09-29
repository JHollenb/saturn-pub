"""Dose response plus hostile controls; compile one measured intervention program."""

import json
from pathlib import Path

import torch

from saturn_pub import Act
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.investigation import Arm, Investigation
from saturn_pub.program import Instruction, Program

adapter = QwenAdapter.tiny()
session = adapter.session([5, 7, 11])
session.continue_(adapter.layers + 1)  # stop before the frozen native readout
carrier = session.read("hidden")
payload = torch.ones_like(carrier) * 0.1
plan = Investigation.doses(
    "How does a final carrier write affect native readout?",
    "hidden",
    payload,
    [-1, 0, 0.25, 1],
    steps=1,
)
generator = torch.Generator().manual_seed(101)
random = torch.randn(carrier.shape, generator=generator)
random *= payload.norm() / random.norm()
plan = Investigation(
    plan.question,
    plan.arms
    + (
        Arm("wrong-source", (Act.add("hidden", -carrier),), "wrong-source"),
        Arm("norm-matched-random", (Act.add("hidden", random),), "matched-random"),
    ),
    1,
)
panel = plan.run(
    session,
    lambda branch: {
        "next_token": int(branch.read("tokens")[0, -1]),
        "logit_max": float(branch.read("logits").max()),
    },
)
chosen = plan.arms[4]
program = Program.compile(
    session,
    panel,
    chosen.name,
    tuple(Instruction(0, act) for act in chosen.acts),
    context="offline-synthetic-carrier",
)
replay = program.run(session, context="offline-synthetic-carrier")
panel["program_replay_token"] = int(replay.read("tokens")[0, -1])
Path("outputs/panel").mkdir(parents=True, exist_ok=True)
Path("outputs/panel/report.json").write_text(json.dumps(panel, indent=2))
print(
    json.dumps(
        {
            "rows": [
                {k: v for k, v in row.items() if k in ("name", "status", "metrics")}
                for row in panel["rows"]
            ],
            "terminal_status": panel["terminal_status"],
        },
        indent=2,
    )
)
