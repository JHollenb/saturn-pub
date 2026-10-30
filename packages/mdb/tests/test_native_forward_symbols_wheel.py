"""Framework-free qualification of trace-bound readable atlas tables."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "qualification" / "native_forward_symbols_wheel.py"
_SPEC = importlib.util.spec_from_file_location("native_forward_symbols_wheel_source", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)


class _Receipt:
    @staticmethod
    def make(**body):
        return SimpleNamespace(fingerprint=hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest())


def _branch_receipt():
    body = {"trace_fingerprint": "trace-seal", "parent_fingerprint": "parent-seal",
            "native_prefix": {"exact": True}}
    receipt = {**body, "fingerprint": _Receipt.make(**body).fingerprint}
    raw = json.dumps(receipt).encode()
    branch = {"receipt": {"sha256": hashlib.sha256(raw).hexdigest()},
              "recipient_trace_fingerprint": "trace-seal", "parent_fingerprint": "parent-seal"}
    return raw, branch, receipt


def test_receipt_verifies_public_seal_without_duplicate_descriptor_fingerprint():
    raw, branch, receipt = _branch_receipt()
    assert qualification.verify_branch_receipt(raw, branch, {"trace-seal"}, _Receipt) == receipt


@pytest.mark.parametrize("mutation", ["descriptor-seal", "receipt-seal", "parent", "trace"])
def test_receipt_rejects_mismatched_identity_even_without_duplicate_seal(mutation):
    raw, branch, receipt = _branch_receipt()
    if mutation == "descriptor-seal":
        branch["receipt"]["fingerprint"] = "wrong"
    elif mutation == "receipt-seal":
        receipt["fingerprint"] = "wrong"
        raw = json.dumps(receipt).encode()
        branch["receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    elif mutation == "parent":
        branch["parent_fingerprint"] = "wrong"
    else:
        branch["recipient_trace_fingerprint"] = "wrong"
    with pytest.raises(RuntimeError, match="receipt identity differs"):
        qualification.verify_branch_receipt(raw, branch, {"trace-seal"}, _Receipt)


def _metadata_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _component_document():
    input_port = {"kind": "tensor", "sha256": "input-bytes", "shape": [1, 2, 3]}
    output_port = {"kind": "tensor", "sha256": "output-bytes", "shape": [1, 2, 3]}
    before_ports = {"tokens": input_port, "scan_output": {"kind": "none"}}
    after_ports = {"tokens": input_port, "scan_output": output_port}

    def cut(ports, parent):
        manifest = {"model_identity": "compiled-model", "parent": parent,
                    "payload": {"kind": "mapping", "items": [
                        {"key": {"value": key}, "value": value} for key, value in ports.items()]}}
        return {"manifest": manifest, "fingerprint": _metadata_digest(manifest)}

    before = cut(before_ports, None)
    after = cut(after_ports, before["fingerprint"])
    body = {"operation": "continue", "parent": before["fingerprint"],
            "result": after["fingerprint"], "model_identity": "compiled-model",
            "execution": {"call_id": "call-0"}}
    receipt = {**body, "fingerprint": _Receipt.make(**body).fingerprint}
    return {"schema": "isa-compiled-scan-public-component-calls-v1", "calls": [{
        "call_id": "call-0", "receipt_fingerprint": receipt["fingerprint"],
        "component_call": {"call_id": "call-0", "receipt": receipt,
            "before_cut": before, "after_cut": after,
            "before_frame": {"slots": before_ports}, "after_frame": {"slots": after_ports},
            "advance_count": 1, "model_identity": "compiled-model", "execution": body["execution"],
            "input_sha256": "input-bytes", "output_sha256": "output-bytes"}}]}


def test_component_receipt_verifies_sealed_cuts_and_actual_ports():
    raw = json.dumps(_component_document()).encode()
    assert qualification.verify_component_calls(
        raw, {"sha256": hashlib.sha256(raw).hexdigest()}, _Receipt, _metadata_digest,
        expected_count=1) == 1


@pytest.mark.parametrize("mutation", ["receipt", "cut", "output-port", "duplicate-call"])
def test_component_receipt_rejects_changed_seals_ports_and_panel(mutation):
    document = _component_document()
    component = document["calls"][0]["component_call"]
    if mutation == "receipt":
        component["receipt"]["fingerprint"] = "wrong"
    elif mutation == "cut":
        component["after_cut"]["fingerprint"] = "wrong"
    elif mutation == "output-port":
        component["output_sha256"] = "wrong"
    else:
        document["calls"].append(document["calls"][0])
    raw = json.dumps(document).encode()
    with pytest.raises(RuntimeError, match="compiled component"):
        qualification.verify_component_calls(
            raw, {"sha256": hashlib.sha256(raw).hexdigest()}, _Receipt, _metadata_digest,
            expected_count=1)


def _stream_observation():
    roots = {name: {"sha256": "root-" + name} for name in ("window_bits", "register_bits", "flags")}
    values = [0.25, -0.0, 13.5]
    head = {"dtype": "torch.float32", "shape": [1, 3], "values": [values],
            "sha256": hashlib.sha256(struct.pack("<fff", *values)).hexdigest()}
    heads = {name: dict(head) for name in ("regs", "flags", "dest", "src", "width", "length",
                                          "syscall", "imm", "store", "store_addr", "store_value")}
    inputs = {"root." + name: {"kind": "tensor", **value} for name, value in roots.items()}
    before_ports = {**inputs, **{"output." + name: {"kind": "none"} for name in heads}}
    after_ports = {**inputs, **{"output." + name: {"kind": "tensor", "sha256": head["sha256"]}
                              for name in heads}}

    def cut(ports, parent):
        manifest = {"model_identity": "stream-model", "parent": parent,
                    "payload": {"kind": "mapping", "items": [
                        {"key": {"value": key}, "value": value} for key, value in ports.items()]}}
        return {"manifest": manifest, "fingerprint": _metadata_digest(manifest)}

    before = cut(before_ports, None)
    after = cut(after_ports, before["fingerprint"])
    body = {"operation": "continue", "parent": before["fingerprint"], "result": after["fingerprint"],
            "model_identity": "stream-model", "execution": {"call_id": "call-0"}}
    return {"call_id": "call-0", "context": {"input_roots": roots}, "output_heads": heads,
            "public_call": {"call_id": "call-0", "advance_count": 1,
                            "model_identity": "stream-model", "execution": body["execution"],
                            "before_cut": before, "after_cut": after,
                            "before_frame": {"slots": before_ports}, "after_frame": {"slots": after_ports},
                            "transition_spec": {"reads": list(inputs),
                                                "writes": ["output." + name for name in heads]},
                            "receipt": {**body, "fingerprint": _Receipt.make(**body).fingerprint}}}


def test_stream_call_joins_all_roots_and_exact_float32_head_bytes():
    observation = _stream_observation()
    assert qualification.verify_stream_call(observation, _Receipt, _metadata_digest)


@pytest.mark.parametrize("mutation", ["head-value", "root", "missing-head", "transition", "receipt"])
def test_stream_call_rejects_detached_values_roots_and_receipts(mutation):
    observation = _stream_observation()
    if mutation == "head-value":
        observation["output_heads"]["regs"]["values"] = [[1.0, 2.0, 3.0]]
    elif mutation == "root":
        observation["context"]["input_roots"]["flags"]["sha256"] = "wrong"
    elif mutation == "missing-head":
        del observation["output_heads"]["store_value"]
    elif mutation == "transition":
        observation["public_call"]["transition_spec"]["writes"] = ["output.regs"]
    else:
        observation["public_call"]["receipt"]["fingerprint"] = "wrong"
    with pytest.raises(RuntimeError, match="stream"):
        qualification.verify_stream_call(observation, _Receipt, _metadata_digest)


def _write(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class _Symbols:
    @staticmethod
    def verify_trace(trace):
        assert trace["schema"] == "fake-trace"
        return {"valid": True}


class _Readable:
    class ForwardReadableProgram:
        def __init__(self, trace, annotations):
            assert annotations["trace_fingerprint"] == trace["fingerprint"]
            self.fingerprint = trace["fingerprint"]

        def render(self):
            return "Readable full table\n"


class _CLI:
    def __init__(self):
        self.argv = None

    def main(self, argv):
        self.argv = argv
        print("Readable full table")
        return 0


def _atlas(tmp_path: Path):
    root = tmp_path / "atlas"
    trace_path = tmp_path / "captured-trace.json"
    trace = {"schema": "fake-trace", "fingerprint": "trace-seal"}
    trace_raw = json.dumps(trace).encode()
    trace_hash = _write(trace_path, trace_raw)
    annotations_raw = json.dumps({"trace_fingerprint": "trace-seal"}).encode()
    table_raw = b"Readable full table\n"
    annotations_hash = _write(root / "case-0" / "annotations.json", annotations_raw)
    table_hash = _write(root / "case-0" / "table.txt", table_raw)
    source_path = tmp_path / "source.py"
    source_hash = _write(source_path, b"source snapshot\n")
    input_path = tmp_path / "input.json"
    input_hash = _write(input_path, b"input snapshot\n")
    (root / "output-manifest.json").write_text(json.dumps({
        "files": [
            {"path": "case-0/annotations.json", "sha256": annotations_hash, "bytes": len(annotations_raw)},
            {"path": "case-0/table.txt", "sha256": table_hash, "bytes": len(table_raw)},
        ]
    }))
    (root / "references.json").write_text(json.dumps({
        "inputs": [{"path": str(input_path), "sha256": input_hash}],
        "sources": [{"path": str(source_path), "sha256": source_hash}],
    }))
    (root / "manifest.json").write_text(json.dumps({
        "cases": [{
            "case_id": "case-0",
            "trace": {"path": str(trace_path), "sha256": trace_hash},
            "trace_fingerprint": "trace-seal",
            "annotations": "case-0/annotations.json",
            "table": "case-0/table.txt",
        }],
        "networks": [{"name": "optional metadata"}],
    }))
    return root


def test_mlp_atlas_verifies_trace_bound_full_cli_table(tmp_path):
    root = _atlas(tmp_path)
    cli = _CLI()

    assert qualification.verify_mlp_atlas(root, cli, _Readable, _Symbols) == 1
    assert cli.argv is not None
    assert cli.argv[2:] == ["--readable", "--annotations", str(root / "case-0" / "annotations.json")]


def test_qualification_cli_exposes_mlp_atlas_option(capsys):
    with pytest.raises(SystemExit) as exc_info:
        qualification.main(["--help"])

    assert exc_info.value.code == 0
    assert "--mlp-atlas" in capsys.readouterr().out


def test_atlas_verifies_frozen_source_snapshot_envelope(tmp_path):
    root = _atlas(tmp_path)
    path = root / "references.json"
    refs = json.loads(path.read_text())
    source = refs["sources"][0]
    refs["sources"] = [{"origin": "/historical/source.py", "snapshot": source,
                        "source_sha256": source["sha256"]}]
    path.write_text(json.dumps(refs))

    assert qualification.verify_mlp_atlas(root, _CLI(), _Readable, _Symbols) == 1


def test_atlas_rejects_source_digest_not_matching_snapshot(tmp_path):
    root = _atlas(tmp_path)
    path = root / "references.json"
    refs = json.loads(path.read_text())
    refs["sources"] = [{"snapshot": refs["sources"][0], "source_sha256": "0" * 64}]
    path.write_text(json.dumps(refs))

    with pytest.raises(RuntimeError, match="source digest differs"):
        qualification.verify_mlp_atlas(root, _CLI(), _Readable, _Symbols)


def test_mlp_atlas_rejects_changed_output_manifest_bytes(tmp_path):
    root = _atlas(tmp_path)
    (root / "case-0" / "table.txt").write_text("changed table\n")

    with pytest.raises(RuntimeError, match="output differs from its manifest"):
        qualification.verify_mlp_atlas(root, _CLI(), _Readable, _Symbols)


def test_mlp_atlas_rejects_annotation_bound_to_another_trace(tmp_path):
    root = _atlas(tmp_path)
    (root / "case-0" / "annotations.json").write_text(
        json.dumps({"trace_fingerprint": "other-trace"})
    )
    raw = (root / "case-0" / "annotations.json").read_bytes()
    manifest = json.loads((root / "output-manifest.json").read_text())
    manifest["files"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["files"][0]["bytes"] = len(raw)
    (root / "output-manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="not bound to the case trace"):
        qualification.verify_mlp_atlas(root, _CLI(), _Readable, _Symbols)


def test_mlp_atlas_rejects_paths_outside_atlas(tmp_path):
    root = _atlas(tmp_path)
    atlas = json.loads((root / "manifest.json").read_text())
    atlas["cases"][0]["table"] = "../outside.txt"
    (root / "manifest.json").write_text(json.dumps(atlas))

    with pytest.raises(RuntimeError, match="stay inside the atlas"):
        qualification.verify_mlp_atlas(root, _CLI(), _Readable, _Symbols)
