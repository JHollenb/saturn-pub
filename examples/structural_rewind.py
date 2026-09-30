"""Snapshot and rewind parameter-free structural state through the training boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from saturn_pub.training import TrainingStateBoundary


def main() -> None:
    output = Path("outputs/structural-rewind")
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "graph": {"start": "read", "read": "emit"},
        "rules": {"read": "memory.answer", "emit": "output"},
        "memory": {"answer": 42},
        "cursor": "start",
    }

    def capture():
        return copy.deepcopy(state)

    def restore(value):
        state.clear()
        state.update(copy.deepcopy(value))

    boundary = TrainingStateBoundary(
        model=torch.nn.Module(),
        optimizer=None,
        cursor_capture=capture,
        cursor_restore=restore,
    )
    parent_state = capture()
    parent = boundary.capture()
    state["graph"]["read"] = "validate"
    state["graph"]["validate"] = "emit"
    state["rules"]["validate"] = "answer >= 0"
    state["memory"]["validated"] = True
    state["cursor"] = "validate"
    candidate = boundary.capture()

    restored_parent = boundary.restore(parent)
    assert restored_parent == parent.fingerprint
    assert state == parent_state
    reopened_candidate = boundary.restore(candidate)
    assert reopened_candidate == candidate.fingerprint
    assert state["graph"]["read"] == "validate" and state["memory"]["validated"]

    # The payload round trip models a fresh-process boundary without relying on an optimizer.
    cold = type(candidate).from_state_dict(candidate.to_state_dict())
    state["graph"].clear()
    state["memory"].clear()
    cold_restored = boundary.restore(cold)
    assert cold_restored == candidate.fingerprint
    report = {
        "specimen": "empty torch.nn.Module; no parameters and no optimizer",
        "parent": parent.fingerprint,
        "candidate": candidate.fingerprint,
        "parent_restore_exact": restored_parent == parent.fingerprint,
        "candidate_reopened_exact": reopened_candidate == candidate.fingerprint,
        "cold_payload_restore_exact": cold_restored == candidate.fingerprint,
        "snapshot_state": copy.deepcopy(state),
        "authority": "structural rollback mechanics; no learned growth or utility claim",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
