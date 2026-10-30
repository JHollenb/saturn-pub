"""Mechanics for anonymous, sealed native forward traces."""

from __future__ import annotations

import copy
import json

import pytest

from mdb.forward_symbols import (
    FORWARD_SYMBOLS_SCHEMA,
    ForwardSymbolsError,
    ForwardSymbolTable,
    main,
    seal_trace,
    verify_trace,
)


def _trace():
    return {
        "schema": FORWARD_SYMBOLS_SCHEMA,
        "model_identity": {"model_id": "toy", "weights_sha256": "a" * 64},
        "execution": {"backend": "cpu", "dtype": "float32"},
        "input_identity": {"input_sha256": "b" * 64},
        "values": {
            "v000": {
                "shape": [2], "dtype": "torch.float32", "sha256": "1" * 64,
                "origin": "input", "producer": None, "preview": [1.0, 2.0],
            },
            "v001": {
                "shape": [2], "dtype": "torch.float32", "sha256": "2" * 64,
                "origin": "parameter", "producer": None, "preview": [0.5, -0.5],
            },
            "v002": {
                "shape": [2], "dtype": "torch.float32", "sha256": "3" * 64,
                "origin": "operation-output", "producer": 0, "preview": [1.5, 1.5],
            },
            "v003": {
                "shape": [], "dtype": "torch.int64", "sha256": "4" * 64,
                "origin": "operation-output", "producer": 1, "preview": [0],
            },
        },
        "events": [
            {
                "sequence": 0,
                "operator": "aten.add.Tensor",
                "module_path": "layers.0.projection",
                "inputs": ["v000", "v001"],
                "outputs": ["v002"],
                "arguments": {"self": {"$value": "v000"}, "other": {"$value": "v001"}},
                "result": {"$value": "v002"},
                "clock": {"forward_call": 0, "operation": 0},
            },
            {
                "sequence": 1,
                "operator": "aten.argmax.default",
                "module_path": "readout",
                "inputs": ["v002"],
                "outputs": ["v003"],
                "arguments": {"self": {"$value": "v002"}, "dim": 0},
                "result": {"$value": "v003"},
                "clock": {"forward_call": 0, "operation": 1},
            },
        ],
        "outputs": {"class_id": {"$value": "v003"}},
        "capture_status": "complete",
        "semantic_labels_supplied": False,
        "metadata": {"capture": "fixture", "purpose": "schema mechanics"},
    }


def test_seal_verify_and_step_view_preserve_physical_provenance():
    trace = seal_trace(_trace())
    verified = verify_trace(trace)
    assert verified["passed"] is True
    assert verified["semantic_labels_supplied"] is False

    table = ForwardSymbolTable(trace)
    first = table.at(0)
    assert first["defs"] == ["v002"]
    assert first["uses"] == ["v000", "v001"]
    assert set(first["visible_values"]) == {"v000", "v001", "v002"}
    assert first["outputs"] == {"class_id": {"$value": "v003"}}
    assert "aten.add.Tensor" in table.format_step(0)
    assert "v002" in table.format_step(0)

    second = table.inspect("v003")
    assert second["defs"] == [1]
    assert second["uses"] == []
    assert second["event_ancestry"] == [0, 1]
    assert table.inspect("v002")["uses"] == [1]


def test_trace_tampering_and_forward_references_are_rejected():
    trace = seal_trace(_trace())
    changed = copy.deepcopy(trace)
    changed["events"][0]["module_path"] = "layers.1.projection"
    with pytest.raises(ForwardSymbolsError, match="fingerprint"):
        verify_trace(changed)

    invalid = _trace()
    invalid["events"][0]["inputs"] = ["v003"]
    invalid["events"][0]["arguments"] = {"self": {"$value": "v003"}}
    with pytest.raises(ForwardSymbolsError, match="before its native producer"):
        seal_trace(invalid)


def test_duplicate_or_labeled_values_and_semantic_names_are_rejected():
    duplicate = _trace()
    duplicate["events"][0]["outputs"] = ["v002", "v002"]
    with pytest.raises(ForwardSymbolsError, match="duplicate value ID"):
        seal_trace(duplicate)

    named = _trace()
    named["values"]["projection_output"] = named["values"].pop("v002")
    with pytest.raises(ForwardSymbolsError, match="not anonymous"):
        seal_trace(named)

    labeled = _trace()
    labeled["semantic_labels_supplied"] = True
    with pytest.raises(ForwardSymbolsError, match="supplied semantic labels"):
        seal_trace(labeled)

    labeled_metadata = _trace()
    labeled_metadata["metadata"]["symbol"] = "answer"
    with pytest.raises(ForwardSymbolsError, match="semantic labeling"):
        seal_trace(labeled_metadata)


def test_step_requires_a_captured_native_operation():
    table = ForwardSymbolTable(seal_trace(_trace()))
    with pytest.raises(ForwardSymbolsError, match="captured event"):
        table.at(-1)
    with pytest.raises(ForwardSymbolsError, match="unknown value"):
        table.inspect("answer")


def test_readable_cli_keeps_mathematical_expression_and_source_provenance(tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(seal_trace(_trace())))
    assert main([str(path), "--readable"]) == 0
    rendered = capsys.readouterr().out
    assert "v002 = (v000 + v001)" in rendered
    assert "source-name" in rendered
    assert main([str(path), "--readable", "--output-path", "/class_id", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["target_values"] == ["v003"]


def test_native_result_tags_prims_and_literal_roots_are_recordable():
    trace = _trace()
    trace["values"]["v004"] = {
        "shape": [], "dtype": "torch.float32", "sha256": "5" * 64,
        "origin": "literal", "producer": None, "preview": ["inf"],
    }
    trace["values"]["v005"] = {
        "shape": [], "dtype": "torch.float32", "sha256": "6" * 64,
        "origin": "operation-output", "producer": 2, "preview": [3.0],
    }
    trace["events"].append({
        "sequence": 2, "operator": "aten.lift_fresh.default", "module_path": "<root>",
        "inputs": ["v004"],
        "outputs": ["v005"],
        "arguments": {"args": {"$tuple": [{"$value": "v004"}]},
                       "kwargs": {"layout": {"$torch": "strided"}, "scale": {"$float": "-inf"}}},
        "result": {"$value": "v005"}, "clock": {"forward_call": 1, "operation": 0},
    })
    trace["metadata"] = {
        "observer": "native-TorchDispatchMode", "semantic_interpretation": "unassigned",
        "replay_roots": "inputs-parameters-native-literals-only",
    }
    sealed = seal_trace(trace)
    table = ForwardSymbolTable(sealed)
    assert table.inspect("v004")["producer_event"] is None
    assert table.at(2)["defs"] == ["v005"]
    assert "aten.lift_fresh.default" in table.format_step(2)


def test_event_result_must_match_written_values_and_literal_is_restricted():
    mismatch = _trace()
    mismatch["events"][0]["result"] = {"$value": "v003"}
    with pytest.raises(ForwardSymbolsError, match="result references"):
        seal_trace(mismatch)

    literal = _trace()
    literal["values"]["v001"]["origin"] = "literal"
    literal["events"][0]["arguments"]["other"] = {"$value": "v001"}
    with pytest.raises(ForwardSymbolsError, match="literal roots"):
        seal_trace(literal)


def test_cli_inspects_one_value_as_text_or_json(tmp_path, capsys):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(seal_trace(_trace())), encoding="utf-8")

    assert main([str(trace_path), "--value", "v003"]) == 0
    rendered = capsys.readouterr().out
    assert "v003: operation-output" in rendered
    assert "aten.argmax.default" in rendered
    assert "event ancestry: [0, 1]" in rendered

    assert main([str(trace_path), "--value", "v003", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["value_id"] == "v003"
    assert result["producer_event"]["operator"] == "aten.argmax.default"
