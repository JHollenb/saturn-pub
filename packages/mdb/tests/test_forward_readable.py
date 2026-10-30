"""Framework-free mechanics for readable views of sealed physical traces."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mdb.forward_readable import (
    FORWARD_ANNOTATIONS_SCHEMA,
    ForwardReadableError,
    ForwardReadableProgram,
)
from mdb.forward_symbols import FORWARD_SYMBOLS_SCHEMA, seal_trace


def _value(index, *, origin="operation-output", producer=None, shape=(1,)):
    return {
        "shape": list(shape), "dtype": "torch.float32", "sha256": f"{index + 1:064x}",
        "origin": origin, "producer": producer, "preview": [],
    }


def _event(sequence, operator, module_path, args, output, *, kwargs=None):
    kwargs = kwargs or {}
    references = []

    def visit(value):
        if isinstance(value, dict) and set(value) == {"$value"}:
            if value["$value"] not in references:
                references.append(value["$value"])
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for argument in args:
        visit(argument)
    for argument in kwargs.values():
        visit(argument)
    return {
        "sequence": sequence, "operator": operator, "module_path": module_path,
        "inputs": references, "outputs": [output],
        "arguments": {"args": {"$tuple": args}, "kwargs": kwargs},
        "result": {"$value": output},
        "clock": {"forward_call": 0, "operation": sequence},
    }


def _trace():
    values = {
        "v000": _value(0, origin="input", producer=None),
        "v001": _value(1, origin="parameter", producer=None, shape=()),
        "v002": _value(2, origin="input", producer=None, shape=(1, 1)),
        "v003": _value(3, origin="parameter", producer=None, shape=()),
        "v004": _value(4, producer=0),
        "v005": _value(5, producer=1),
        "v006": _value(6, producer=2, shape=()),
        "v007": _value(7, producer=3, shape=()),
        "v008": _value(8, producer=4),
        "v009": _value(9, producer=5),
        "v010": _value(10, producer=6),
        "v011": _value(11, producer=7, shape=(1, 1)),
        "v012": _value(12, producer=8),
        "v013": _value(13, producer=9),
    }
    events = [
        _event(0, "aten.softplus.default", "mixer.time", [{"$value": "v000"}], "v004"),
        _event(1, "aten.unsqueeze.default", "mixer.time", [{"$value": "v004"}, 0], "v005"),
        _event(2, "aten.exp.default", "mixer.time", [{"$value": "v001"}], "v006"),
        _event(3, "aten.neg.default", "mixer.time", [{"$value": "v006"}], "v007"),
        _event(4, "aten.mul.Tensor", "mixer.time", [{"$value": "v005"}, {"$value": "v007"}], "v008"),
        _event(5, "aten.exp.default", "mixer.time", [{"$value": "v008"}], "v009"),
        _event(6, "aten.softmax.int", "mixer.read", [{"$value": "v000"}, -1], "v010"),
        _event(7, "aten.mm.default", "mixer.read", [{"$value": "v010"}, {"$value": "v002"}], "v011"),
        _event(8, "aten.softplus.default", "mixer.time", [{"$value": "v000"}], "v012",
               kwargs={"beta": 2.0, "threshold": 10.0}),
        _event(9, "aten.gelu.default", "mixer.time", [{"$value": "v000"}], "v013",
               kwargs={"approximate": "tanh"}),
    ]
    return seal_trace({
        "schema": FORWARD_SYMBOLS_SCHEMA,
        "model_identity": {"model_id": "small-synthetic"},
        "execution": {"backend": "metadata-only"},
        "input_identity": {"case": "readable-fixture"},
        "values": values, "events": events,
        "outputs": {"decay": {"$value": "v009"}, "weighted": {"$value": "v011"}},
        "capture_status": "complete", "semantic_labels_supplied": False,
        "metadata": {"fixture": "sealed synthetic trace"},
    })


def test_values_are_navigable_by_module_producer_and_use_and_sliced_from_outputs():
    program = ForwardReadableProgram(_trace())
    assert len(program.list_values()) == 14
    assert {row["value_id"] for row in program.list_values(scope="mixer.time")} == {
        "v000", "v001", "v004", "v005", "v006", "v007", "v008", "v009", "v012", "v013",
    }
    assert [row["value_id"] for row in program.list_values(producer=4)] == ["v008"]
    assert {row["value_id"] for row in program.list_values(use=4)} == {"v005", "v007"}

    decay = program.backward_slice(output_path="/decay")
    assert [row["sequence"] for row in decay["events"]] == list(range(6))
    assert {row["value_id"] for row in decay["values"]} == {
        "v000", "v001", "v004", "v005", "v006", "v007", "v008", "v009",
    }
    assert program.value("v009")["output_paths"] == ["/decay"]
    assert program.value("v009")["output_path_provenance"] == [
        {"path": "/decay", "provenance": "source-name"},
    ]
    assert program.value("v009")["provenance"] == "structural-observation"


def test_renderer_shows_operator_formulas_exact_patterns_and_source_scope():
    program = ForwardReadableProgram(_trace())
    assert program.event(0)["expression"] == "softplus(v000)"
    assert program.event(1)["expression"] == "unsqueeze(v004, 0)"
    assert program.event(8)["expression"] == "softplus(v000, beta=2.0, threshold=10.0)"
    assert program.event(9)["expression"] == 'gelu(v000, approximate="tanh")'
    patterns = program.list_patterns()
    decay = next(item for item in patterns if item["kind"] == "exp-of-softplus-scaled-negative-exp")
    assert decay["expression"] == "exp(unsqueeze(v004, 0) * (-exp(v001)))"
    weighted = next(item for item in patterns if item["kind"] == "softmax-weighted-matrix-product")
    assert weighted["weight_value"] == "v010"
    assert weighted["other_value"] == "v002"
    assert weighted["provenance"] == "structural-observation"

    rendered = program.render()
    assert "module mixer.time [source-name]" in rendered
    assert "v009 = exp(v008)" in rendered
    assert "output_ancestry=/decay" in rendered
    assert "softmax-weighted-matrix-product" in rendered
    assert "carry" not in rendered.lower()
    assert "ADD" not in rendered


def test_scaled_negative_exp_pattern_does_not_drop_a_nonunit_coefficient():
    body = copy.deepcopy(_trace())
    body.pop("fingerprint")
    body["events"][3] = _event(
        3, "aten.mul.Scalar", "mixer.time", [{"$value": "v006"}, -2], "v007"
    )
    program = ForwardReadableProgram(seal_trace(body))

    assert program.event(3)["expression"] == "(v006 * -2)"
    assert not any(
        item["kind"] == "exp-of-softplus-scaled-negative-exp"
        for item in program.list_patterns()
    )


def test_external_annotations_are_scoped_evidence_refs_and_fingerprint_bound():
    trace = _trace()
    annotation = {
        "schema": FORWARD_ANNOTATIONS_SCHEMA,
        "trace_fingerprint": trace["fingerprint"],
        "annotations": [{
            "id": "finding-7", "scope": {"kind": "output", "path": "/decay"},
            "claim": "A separate intervention study reports a causal effect.",
            "provenance": "causal-finding", "evidence_refs": ["study-row:7"],
        }],
    }
    program = ForwardReadableProgram(trace, annotation)
    assert program.value("v009")["annotations"][0]["id"] == "finding-7"
    assert "evidence_refs=['study-row:7']" in program.render(output_path="/decay")

    wrong_fingerprint = copy.deepcopy(annotation)
    wrong_fingerprint["trace_fingerprint"] = "f" * 64
    with pytest.raises(ForwardReadableError, match="different trace fingerprint"):
        ForwardReadableProgram(trace, wrong_fingerprint)

    wrong_scope = copy.deepcopy(annotation)
    wrong_scope["annotations"][0]["scope"] = {"kind": "value", "id": "v999"}
    with pytest.raises(ForwardReadableError, match="unknown or malformed scope"):
        ForwardReadableProgram(trace, wrong_scope)


def test_slice_and_value_navigation_reject_unknown_targets():
    program = ForwardReadableProgram(_trace())
    with pytest.raises(ForwardReadableError, match="unknown value ID"):
        program.value("v999")
    with pytest.raises(ForwardReadableError, match="unknown tensor output path"):
        program.backward_slice(output_path="/missing")


def test_readable_module_does_not_import_torch():
    package_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, ("src", env.get("PYTHONPATH", ""))))
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import mdb.forward_readable; assert 'torch' not in sys.modules"],
        cwd=package_root, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
