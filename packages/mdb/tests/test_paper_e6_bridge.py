import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments/time_circuits"))
from e6_bridge import FeatureSessions, graph_reference, logits_distance


class TinyInstrument:
    def __init__(self):
        self.calls = 0

    def feature_intervention(self, prompt, interventions, *, freeze_attention=True):
        self.calls += 1
        output = torch.zeros(1, 3, 8)
        for layer, position, feature, value in interventions:
            output[0, position, feature] += value * (layer + 1)
        return output, None


def test_record_based_graph_targets_serialize_via_token_ids():
    graph = SimpleNamespace(logit_targets=[SimpleNamespace(vocab_idx=7)],
        logit_token_ids=torch.tensor([7]), input_tokens=torch.tensor([1, 2, 3]),
        active_features=torch.tensor([[0, 1, 4], [2, 0, 1]]), selected_features=torch.tensor([1]))
    reference = graph_reference(graph, identity={}, prompt="prompt",
                                replacement=.7, completeness=.9, influence=[.3])
    assert reference["logit_target_token_ids"] == [7]
    assert reference["feature_coordinates"] == [[2, 0, 1]]


def test_prefix_difference_and_consumer_difference_are_distinct():
    left = torch.zeros(1, 3, 8)
    right = left.clone()
    right[0, 0, 0] = .25
    distance = logits_distance(left, right)
    assert not distance["all_logits_exact"]
    assert distance["last_logits_exact"]


def test_native_feature_addresses_parity_replay_and_rejection(tmp_path):
    native = TinyInstrument()
    owner = FeatureSessions(root=tmp_path,
        identity={"model_id": "tiny", "weights_sha256": "b" * 64, "family": "tiny-tracer",
                  "numerical_contract": "tiny-cpu"},
        tokenizer=lambda _: {"input_ids": [1, 2, 3]}, layers=4, features=8)
    owner.attach(native)
    try:
        output, _ = native.feature_intervention("prompt", [(2, 1, 4, -0.5)], freeze_attention=False)
        assert output[0, 1, 4] == -1.5
        assert owner.records[0]["native_logits_exact"]
        assert owner.records[0]["intervention_roundtrip_exact"]
        calls = native.calls
        with pytest.raises(ValueError, match="nonnegative integers"):
            native.feature_intervention("prompt", [(1.5, 0, 4, 1)])
        with pytest.raises(ValueError, match="geometry"):
            native.feature_intervention("prompt", [(4, 0, 4, 1)])
        assert native.calls == calls
        summaries = owner.finish()
        assert summaries[0]["replay_first_exact"]
        assert all(replay["exact"] for replay in summaries[0]["replays"])
    finally:
        owner.detach()
