import copy

import pytest
import torch

from saturn_pub.training import TrainingStateBoundary
from saturn_pub.training._checkpoint import TrainingCheckpointError, TrainingStateSnapshot


def test_parameter_free_structural_cut_cold_restore():
    state = {
        "graph": {"start": "end"},
        "rules": {"read": "advance"},
        "memory": {"answer": 42},
        "cursor": "start",
    }

    def restore(value):
        state.clear()
        state.update(copy.deepcopy(value))

    boundary = TrainingStateBoundary(
        model=torch.nn.Module(),
        optimizer=None,
        cursor_capture=lambda: copy.deepcopy(state),
        cursor_restore=restore,
    )
    parent = boundary.capture()
    payload = parent.to_state_dict()
    cold = TrainingStateSnapshot.from_state_dict(payload)
    state["graph"]["start"] = "wrong"
    state["memory"].clear()
    state["cursor"] = "end"
    assert boundary.restore(cold) == parent.fingerprint
    assert state["memory"]["answer"] == 42 and state["cursor"] == "start"
    malformed = copy.deepcopy(payload)
    malformed["optimizer_state"] = {"state": {"x": 1}, "param_groups": []}
    with pytest.raises(TrainingCheckpointError, match="optimizer-free"):
        TrainingStateSnapshot.from_state_dict(malformed)
