from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from mdb.pass_causal_state_diff import (
    PASS_CAUSAL_STATE_DIFF_SCHEMA,
    PassCausalStateDiffError,
    compose_pass_causal_state_diff,
)


def _four_dimensional(rows: list[list[float]], offset: float = 0) -> list:
    return [[[[value + offset for value in row] for row in rows]]]


def _cut(checkpoint_id: str, rows: list[list[float]]) -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "model_identity": {"model_id": "tiny", "revision": "abc"},
        "slots": {
            "kv.000.key": _four_dimensional(rows),
            "kv.000.value": _four_dimensional(rows, 100),
            "kv.001.key": _four_dimensional(rows, 200),
            "kv.001.value": _four_dimensional(rows, 300),
            "attention_mask": [[1] * len(rows)],
            "position_ids": [list(range(len(rows)))],
            "token_ids": [[int(row[0]) for row in rows]],
            "pending_token_id": 9 if checkpoint_id == "pass-n" else 10,
        },
    }


def _positions() -> tuple[list[dict], list[dict]]:
    parent = [
        {
            "logical_position": f"prefix:{index}",
            "physical_position": index,
            "role": "immutable_context",
            "input_authority": "caller-supplied-prefix",
        }
        for index in range(2)
    ]
    child = [
        {**parent[0]},
        {
            "logical_position": "turn:0:question:0",
            "physical_position": 1,
            "role": "appended_question",
            "input_authority": "caller-supplied-question",
        },
        {
            "logical_position": "turn:0:answer:0",
            "physical_position": 2,
            "role": "appended_model_answer",
            "input_authority": "native-model-generated",
        },
        {**parent[1], "physical_position": 3},
    ]
    return parent, child


def _compose(**overrides) -> dict:
    parent_positions, child_positions = _positions()
    arguments = {
        "parent_statecut": _cut("pass-n", [[1, 2], [3, 4]]),
        "child_statecut": _cut("pass-n-plus-one", [[1, 2], [30, 31], [40, 41], [3, 4]]),
        "pass_index": 2,
        "native_consumer": {"kind": "unchanged-native-suffix", "consumer_id": "qwen"},
        "parent_positions": parent_positions,
        "child_positions": child_positions,
        "common_parent_statecut_id": "common-parent",
        "followup_query": {
            "query_id": "followup-7",
            "input_sha256": "a" * 64,
            "format_sha256": "b" * 64,
            "input_authority": "caller-supplied-identical-query",
        },
    }
    arguments.update(overrides)
    return compose_pass_causal_state_diff(**arguments)


def test_json_statecuts_align_unequal_caches_by_logical_position() -> None:
    report = _compose()
    assert report["schema"] == PASS_CAUSAL_STATE_DIFF_SCHEMA
    assert report["pass_transition"] == {"from_pass": 2, "to_pass": 3}
    old = report["state"]["old_rows"]
    assert old["all_rows_equal"] is True
    assert [
        (row["parent_physical_position"], row["child_physical_position"])
        for row in old["rows"]
    ] == [(0, 0), (1, 3)]
    appended = report["state"]["appended_rows"]
    assert appended["appended_question"][0]["logical_position"] == "turn:0:question:0"
    assert appended["appended_model_answer"][0]["logical_position"] == "turn:0:answer:0"
    assert report["status"]["causal_verdict"] == "not-inferred-from-diffs-alone"
    json.dumps(report, allow_nan=False)


def test_changed_old_row_is_only_an_observation() -> None:
    child = _cut("pass-n-plus-one", [[1, 2], [30, 31], [40, 41], [3, 99]])
    report = _compose(child_statecut=child)
    changed = next(
        row
        for row in report["state"]["old_rows"]["rows"]
        if row["logical_position"] == "prefix:1"
    )
    assert changed["all_layers_equal"] is False
    assert report["status"]["instrument_status"] == "observational-diff"


def test_logical_attention_capture_and_readout_planes() -> None:
    report = _compose(
        parent_capture={
            "0": {
                "query_residual": [1.0, 0.0],
                "attention_alpha": {
                    "values": [0.75, 0.25],
                    "logical_positions": ["prefix:0", "prefix:1"],
                },
                "mlp_gate": [0.1, 0.2],
                "norm_scale": [1.0],
            }
        },
        child_capture={
            "0": {
                "query_residual": [0.0, 1.0],
                "attention_alpha": {
                    "values": [0.20, 0.10, 0.15, 0.55],
                    "logical_positions": [
                        "prefix:0",
                        "turn:0:question:0",
                        "turn:0:answer:0",
                        "prefix:1",
                    ],
                },
                "mlp_gate": [0.2, 0.4],
                "norm_scale": [2.0],
            }
        },
        parent_readout={
            "raw_logits": [1.0, 2.0, 0.0],
            "target_logit": 2.0,
            "competitor_logit": 1.0,
            "target_logprob": -0.4,
            "generated_output": [7, 8],
        },
        child_readout={
            "raw_logits": [3.0, 1.0, 0.0],
            "target_logit": 1.0,
            "competitor_logit": 3.0,
            "target_logprob": -2.4,
            "generated_output": [9],
        },
    )
    layer = report["captures"]["layers"][0]
    assert layer["metrics"]["query_residual"]["delta_l2"] == pytest.approx(2**0.5)
    attention = layer["metrics"]["attention_alpha"]
    assert attention["common_logical_positions"] == ["prefix:0", "prefix:1"]
    assert attention["child_only_sum"] == pytest.approx(0.25)
    assert attention["delta_l2"] == pytest.approx((0.55**2 + 0.30**2) ** 0.5)
    assert report["native_readout"]["raw_margin"]["delta"] == -3.0
    assert report["native_readout"]["target_logprob"]["delta"] == pytest.approx(-2.0)


class _FakeTensor:
    def __init__(self, values):
        self.values = values
        self.dtype = "torch.float32"
        self.shape = self._shape(values)

    @classmethod
    def _shape(cls, value):
        return (len(value), *cls._shape(value[0])) if isinstance(value, list) and value else ()

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def tolist(self):
        return self.values


def test_duck_typed_native_statecut_needs_no_saturn_import() -> None:
    raw_parent = _cut("pass-n", [[1, 2], [3, 4]])
    raw_child = _cut("pass-n-plus-one", [[1, 2], [30, 31], [40, 41], [3, 4]])

    def native(raw):
        slots = {
            key: _FakeTensor(value) if isinstance(value, list) else value
            for key, value in raw["slots"].items()
        }
        return SimpleNamespace(
            checkpoint_id=raw["checkpoint_id"],
            model_identity=MappingProxyType(raw["model_identity"]),
            slots=MappingProxyType(slots),
        )

    report = _compose(parent_statecut=native(raw_parent), child_statecut=native(raw_child))
    assert report["binding"]["parent_statecut_id"] == "pass-n"


def test_links_existing_receipts_without_replaying() -> None:
    report = _compose(
        intervention_receipts=[
            {
                "receipt_id": "delete-answer",
                "comparison_role": "append_deletion",
                "parent_statecut_id": "common-parent",
            }
        ],
        restore_receipts=[{"content_sha256": "c" * 64, "control_kind": "exact_reinstall"}],
        continuation_receipts=[{"uri": "ar://continuation/one"}],
    )
    links = report["receipt_links"]
    assert links["interventions"][0]["comparison_role"] == "append_deletion"
    assert links["restores"][0]["control_kind"] == "exact_reinstall"
    assert links["continuations"][0]["uri"] == "ar://continuation/one"
    assert "does not replay" in links["authority"]


def test_rejects_identity_mismatch_and_ambiguous_position_roles() -> None:
    child = _cut("pass-n-plus-one", [[1, 2], [30, 31], [40, 41], [3, 4]])
    child["model_identity"] = {"model_id": "other"}
    with pytest.raises(PassCausalStateDiffError, match="different model identities"):
        _compose(child_statecut=child)
    parent_positions, _child_positions = _positions()
    parent_positions[0]["role"] = "appended_question"
    with pytest.raises(PassCausalStateDiffError, match="immutable_context"):
        _compose(parent_positions=parent_positions)


def test_module_imports_when_upstream_and_model_packages_are_blocked() -> None:
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'saturn','torch','numpy','transformers','mrun'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
import mdb.pass_causal_state_diff as module
assert module.PASS_CAUSAL_STATE_DIFF_SCHEMA == 'saturn-pass-causal-state-diff-v1'
assert not any(name == 'saturn' or name.startswith('saturn.') for name in sys.modules)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_sample_export_is_directly_consumable_by_temporal_viewer() -> None:
    from mdb.temporal_causal_viewer import normalize_evidence

    path = Path(__file__).parents[1] / "examples" / "pass-causal-state-diff.sample.json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == PASS_CAUSAL_STATE_DIFF_SCHEMA
    assert payload["status"]["causal_verdict"] == "not-inferred-from-diffs-alone"
    bundle = normalize_evidence([payload], sources=["passdiff-example"])
    assert bundle["binding"]["parent_cuts"] == ["cut-common-parent"]
    assert bundle["events"]
    assert {event["evidence_class"] for event in bundle["events"]} == {"observation"}
