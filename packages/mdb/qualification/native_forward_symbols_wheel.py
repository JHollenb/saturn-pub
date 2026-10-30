"""Inspect collected native traces using installed wheels and no model framework."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

BLOCKED = {
    "torch", "numpy", "safetensors", "transformers", "diffusers", "jax", "tensorflow",
    "mlx", "saturn", "common", "common_harness", "manalysis", "mrun",
}


class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError("framework/private import forbidden: " + fullname)
        return None


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_branch_receipt(raw, branch, trace_fingerprints, receipt_type):
    """Verify the receipt itself; a duplicated descriptor seal is optional.

    Some native report schemas retain only the receipt's byte hash. That hash
    still binds the complete stored receipt, whose public fingerprint must be
    independently reconstructed. A supplied descriptor seal must also agree.
    """
    receipt = json.loads(raw)
    body = {key: value for key, value in receipt.items() if key != "fingerprint"}
    descriptor = branch["receipt"]
    if (sha256(raw) != descriptor["sha256"]
            or receipt_type.make(**body).fingerprint != receipt["fingerprint"]
            or ("fingerprint" in descriptor
                and receipt["fingerprint"] != descriptor["fingerprint"])
            or receipt["trace_fingerprint"] not in trace_fingerprints
            or receipt["trace_fingerprint"] != branch["recipient_trace_fingerprint"]
            or receipt["parent_fingerprint"] != branch["parent_fingerprint"]
            or receipt["native_prefix"]["exact"] is not True):
        raise RuntimeError("typed branch receipt identity differs from native observation")
    return receipt


def _cut_ports(manifest):
    payload = manifest.get("payload", {})
    if payload.get("kind") != "mapping":
        raise RuntimeError("compiled component cut payload is not a mapping descriptor")
    return {item["key"]["value"]: item["value"] for item in payload["items"]}


def verify_component_calls(raw, descriptor, receipt_type, digest_value, *, expected_count=16):
    """Verify public compiled-transition receipts and their immutable port cuts."""
    if sha256(raw) != descriptor["sha256"]:
        raise RuntimeError("compiled component call artifact changed")
    document = json.loads(raw)
    calls = document.get("calls", [])
    if (document.get("schema") != "isa-compiled-scan-public-component-calls-v1"
            or len(calls) != expected_count
            or len({row["call_id"] for row in calls}) != expected_count):
        raise RuntimeError("compiled component receipt panel is incomplete")
    for row in calls:
        component = row["component_call"]
        receipt = component["receipt"]
        before, after = component["before_cut"], component["after_cut"]
        body = {key: value for key, value in receipt.items() if key != "fingerprint"}
        before_ports, after_ports = _cut_ports(before["manifest"]), _cut_ports(after["manifest"])
        if (component["call_id"] != row["call_id"]
                or receipt_type.make(**body).fingerprint != receipt["fingerprint"]
                or receipt["fingerprint"] != row["receipt_fingerprint"]
                or digest_value(before["manifest"]) != before["fingerprint"]
                or digest_value(after["manifest"]) != after["fingerprint"]
                or receipt.get("operation") != "continue"
                or receipt.get("parent") != before["fingerprint"]
                or receipt.get("result") != after["fingerprint"]
                or after["manifest"].get("parent") != before["fingerprint"]
                or receipt.get("model_identity") != component["model_identity"]
                or before["manifest"].get("model_identity") != component["model_identity"]
                or after["manifest"].get("model_identity") != component["model_identity"]
                or receipt.get("execution") != component["execution"]
                or component.get("advance_count") != 1
                or before_ports.get("tokens") != after_ports.get("tokens")
                or component["before_frame"]["slots"] != before_ports
                or component["after_frame"]["slots"] != after_ports
                or before_ports["tokens"].get("sha256") != component["input_sha256"]
                or after_ports["scan_output"].get("sha256") != component["output_sha256"]):
            raise RuntimeError("compiled component receipt/cut/port identity differs")
    return len(calls)


def _float32_head(row):
    """Reconstruct the exact recorded FP32 bytes without a tensor framework."""
    def flatten(value, shape):
        if not shape:
            if isinstance(value, (list, dict, bool)) or not isinstance(value, (int, float)):
                raise RuntimeError("stream head scalar is invalid")
            if not math.isfinite(value):
                raise RuntimeError("stream head scalar is nonfinite")
            return [float(value)]
        if not isinstance(value, list) or len(value) != shape[0]:
            raise RuntimeError("stream head values differ from declared shape")
        return [item for part in value for item in flatten(part, shape[1:])]

    if row.get("dtype") != "torch.float32":
        raise RuntimeError("stream head must contain FP32 values")
    values = flatten(row["values"], row["shape"])
    raw = struct.pack("<" + "f" * len(values), *values)
    if sha256(raw) != row.get("sha256"):
        raise RuntimeError("stream head values differ from sealed bytes")
    return row["sha256"]


def verify_stream_call(observation, receipt_type, digest_value):
    """Join one whole-program receipt to its three roots and eleven heads."""
    component = observation["public_call"]
    roots = observation["context"]["input_roots"]
    heads = observation["output_heads"]
    expected_roots = {"window_bits", "register_bits", "flags"}
    expected_heads = {"regs", "flags", "dest", "src", "width", "length", "syscall", "imm",
                      "store", "store_addr", "store_value"}
    if set(roots) != expected_roots or set(heads) != expected_heads:
        raise RuntimeError("stream call root/head panel differs")
    before, after = component["before_cut"], component["after_cut"]
    before_ports, after_ports = _cut_ports(before["manifest"]), _cut_ports(after["manifest"])
    input_ports = {"root." + name for name in expected_roots}
    output_ports = {"output." + name for name in expected_heads}
    receipt = component["receipt"]
    body = {key: value for key, value in receipt.items() if key != "fingerprint"}
    if (component["call_id"] != observation["call_id"]
            or receipt_type.make(**body).fingerprint != receipt["fingerprint"]
            or digest_value(before["manifest"]) != before["fingerprint"]
            or digest_value(after["manifest"]) != after["fingerprint"]
            or receipt.get("operation") != "continue"
            or receipt.get("parent") != before["fingerprint"]
            or receipt.get("result") != after["fingerprint"]
            or after["manifest"].get("parent") != before["fingerprint"]
            or any(value.get("model_identity") != component["model_identity"]
                   for value in (receipt, before["manifest"], after["manifest"]))
            or receipt.get("execution") != component["execution"]
            or component["advance_count"] != 1
            or set(before_ports) != input_ports | output_ports
            or set(after_ports) != input_ports | output_ports
            or component["before_frame"]["slots"] != before_ports
            or component["after_frame"]["slots"] != after_ports
            or set(component["transition_spec"]["reads"]) != input_ports
            or set(component["transition_spec"]["writes"]) != output_ports):
        raise RuntimeError("stream call receipt/cut identity differs")
    for name, root in roots.items():
        port = "root." + name
        if before_ports[port] != after_ports[port] or before_ports[port].get("sha256") != root["sha256"]:
            raise RuntimeError("stream call read-only root differs")
    for name, head in heads.items():
        if after_ports["output." + name].get("sha256") != _float32_head(head):
            raise RuntimeError("stream call output port differs from native-consumer bytes")
    return receipt["fingerprint"]


def verify_stream_calls(base, report, receipt_type, digest_value, *, expected_count=56):
    seals = []
    call_ids = []
    for key, descriptor in report["artifacts"].items():
        if not key.startswith("call_observation_"):
            continue
        path = _atlas_file(base, descriptor["path"], "stream observation path")
        raw = path.read_bytes()
        if sha256(raw) != descriptor["sha256"]:
            raise RuntimeError("stream observation artifact changed")
        observation = json.loads(raw)
        if observation.get("schema") != "isa-stream-program-call-observation-v1":
            raise RuntimeError("stream observation schema differs")
        seals.append(verify_stream_call(observation, receipt_type, digest_value))
        call_ids.append(observation["call_id"])
    if len(seals) != expected_count or len(set(call_ids)) != expected_count:
        raise RuntimeError("stream public receipt panel is incomplete")
    return len(seals)


def verify_frontier_branch(base, branch, receipt_type, digest_value):
    """Verify an Act/continuation chain and its actual frontier port bindings."""
    descriptor = branch["observation"]
    raw = _atlas_file(base, descriptor["path"], "frontier observation").read_bytes()
    if sha256(raw) != descriptor["sha256"]:
        raise RuntimeError("frontier observation changed")
    observation = json.loads(raw)
    public = observation["public_branch"]
    cuts = [public[name] for name in ("before_cut", "patched_cut", "after_cut")]
    ports = [_cut_ports(cut["manifest"]) for cut in cuts]
    for cut in cuts:
        if digest_value(cut["manifest"]) != cut["fingerprint"]:
            raise RuntimeError("frontier cut fingerprint differs")
    for index, (name, descriptor_key) in enumerate((("act_receipt", "receipt"),
                                                   ("continuation_receipt", "continuation_receipt"))):
        receipt = public[name]
        ref = branch[descriptor_key]
        raw = _atlas_file(base, ref["path"], "frontier receipt").read_bytes()
        body = {key: value for key, value in receipt.items() if key != "fingerprint"}
        if (sha256(raw) != ref["sha256"] or json.loads(raw) != receipt
                or receipt_type.make(**body).fingerprint != receipt["fingerprint"]
                or receipt["parent"] != cuts[index]["fingerprint"]
                or receipt["result"] != cuts[index + 1]["fingerprint"]
                or cuts[index + 1]["manifest"]["parent"] != cuts[index]["fingerprint"]):
            raise RuntimeError("frontier receipt chain differs")
    if public["advance_count"] != 1 or public["spec"] != observation["spec"]:
        raise RuntimeError("frontier branch execution differs")
    for name in public["preserved_ports"]:
        if ports[0][name] != ports[1][name]:
            raise RuntimeError("frontier preserved recipient port changed")
    for name, value in public["applied_replacements"].items():
        if ports[1][name].get("sha256") != _float32_head(value):
            raise RuntimeError("frontier replacement differs from typed port")
    for name, value in observation["output_heads"].items():
        if ports[2]["output." + name].get("sha256") != _float32_head(value):
            raise RuntimeError("frontier suffix output differs from sealed port")
    return 2


def _atlas_file(root, value, description):
    """Resolve an atlas-owned relative path without allowing it to escape the atlas."""
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise RuntimeError(description + " must name a relative file")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(description + " must stay inside the atlas")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(description + " escapes the atlas")
    return path


def _verify_references(root, refs, description):
    if not isinstance(refs, list):
        raise RuntimeError(description + " must be a list")
    for ref in refs:
        if isinstance(ref, dict) and "snapshot" in ref:
            snapshot = ref["snapshot"]
            if (not isinstance(snapshot, dict)
                    or ref.get("source_sha256") != snapshot.get("sha256")):
                raise RuntimeError(description + " source digest differs from its snapshot")
            ref = snapshot
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
            raise RuntimeError(description + " entries must contain path and sha256")
        path = Path(ref["path"])
        if not path.is_absolute():
            path = root / path
        raw = path.read_bytes()
        if sha256(raw) != ref.get("sha256"):
            raise RuntimeError(description + " reference changed: " + str(path))
        if "bytes" in ref and len(raw) != ref["bytes"]:
            raise RuntimeError(description + " reference size changed: " + str(path))


def verify_mlp_atlas(atlas_path, cli, readable_module, symbols_module):
    """Verify a generic, trace-bound MLP atlas using only the installed MDB reader."""
    atlas_path = Path(atlas_path).resolve()
    if atlas_path.is_dir():
        root = atlas_path
        manifest_path = next(
            (candidate for candidate in (root / "atlas.json", root / "manifest.json") if candidate.is_file()),
            None,
        )
        if manifest_path is None:
            raise RuntimeError("MLP atlas directory has no atlas.json or manifest.json")
    else:
        root = atlas_path.parent
        manifest_path = atlas_path
    atlas = json.loads(manifest_path.read_text())
    output_manifest = json.loads((root / "output-manifest.json").read_text())
    output_paths = set()
    for ref in output_manifest["files"]:
        path = _atlas_file(root, ref.get("path"), "MLP atlas output path")
        raw = path.read_bytes()
        if sha256(raw) != ref.get("sha256") or len(raw) != ref.get("bytes"):
            raise RuntimeError("MLP atlas output differs from its manifest: " + str(path))
        output_paths.add(str(path.relative_to(root.resolve())))

    references = json.loads((root / "references.json").read_text())
    _verify_references(root, references["inputs"], "MLP atlas input")
    _verify_references(root, references["sources"], "MLP atlas source")

    verified = 0
    for case in atlas["cases"]:
        trace_ref = case["trace"]
        trace_path = Path(trace_ref["path"])
        if not trace_path.is_absolute():
            raise RuntimeError("MLP atlas trace path must be absolute")
        trace_raw = trace_path.read_bytes()
        if sha256(trace_raw) != trace_ref["sha256"]:
            raise RuntimeError("MLP atlas trace changed: " + str(trace_path))
        trace = json.loads(trace_raw)
        symbols_module.verify_trace(trace)
        trace_fingerprint = case["trace_fingerprint"]
        if trace.get("fingerprint") != trace_fingerprint:
            raise RuntimeError("MLP atlas trace fingerprint differs: " + case["case_id"])

        annotations_path = _atlas_file(root, case["annotations"], "MLP atlas annotations")
        table_path = _atlas_file(root, case["table"], "MLP atlas table")
        for path in (annotations_path, table_path):
            if str(path.relative_to(root.resolve())) not in output_paths:
                raise RuntimeError("MLP atlas case artifact is absent from output manifest: " + str(path))
        annotations = json.loads(annotations_path.read_text())
        if annotations.get("trace_fingerprint") != trace_fingerprint:
            raise RuntimeError("MLP atlas annotations are not bound to the case trace")
        reader = readable_module.ForwardReadableProgram(trace, annotations)
        if getattr(reader, "fingerprint", None) != trace_fingerprint:
            raise RuntimeError("MLP atlas reader resolved a different trace fingerprint")

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = cli.main([
                "forward-symbols", str(trace_path), "--readable", "--annotations", str(annotations_path)
            ])
        cli_text = stream.getvalue().rstrip("\n")
        expected_text = reader.render().rstrip("\n")
        saved_text = table_path.read_text().rstrip("\n")
        if status != 0 or cli_text != expected_text or cli_text != saved_text:
            raise RuntimeError("MLP atlas table differs from installed trace-bound reader: " + case["case_id"])
        verified += 1
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mdb-wheel", required=True, type=Path)
    parser.add_argument("--saturn-wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atlas", type=Path, help="also verify joined human-readable annotation views")
    parser.add_argument("--algebra-atlas", type=Path, help="verify the factorized algebra atlas and its output manifest")
    parser.add_argument("--mlp-atlas", type=Path, help="verify a generic trace-bound MLP atlas")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    report = json.loads((run_dir / "report.json").read_text())
    if any(name.split(".")[0] in BLOCKED for name in sys.modules):
        raise RuntimeError("start this proof in a clean isolated interpreter")
    sys.meta_path.insert(0, BlockRuntime())

    import saturn_pub

    import mdb.cli
    import mdb.forward_symbols

    modules = [mdb.cli, mdb.forward_symbols, saturn_pub]
    if report.get("phase") in {"compiled-scan", "stream-program", "stream-frontier", "clock-feedback"}:
        import saturn_pub.contracts
        import saturn_pub.core
        import saturn_pub.values

        modules.extend([saturn_pub.contracts, saturn_pub.core, saturn_pub.values])
    if "cases" in report or args.mlp_atlas:
        import mdb.forward_readable

        modules.append(mdb.forward_readable)
    if report.get("phase") in {"algebra", "lowering"}:
        import mdb.forward_algebra

        modules.append(mdb.forward_algebra)
    wheels = {"mdb": args.mdb_wheel.resolve(), "saturn_pub": args.saturn_wheel.resolve()}
    paths = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        if "site-packages" not in path.parts or not path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("module is not installed in this proof environment: " + str(path))
        member = module.__name__.replace(".", "/")
        member += "/__init__.py" if path.name == "__init__.py" else ".py"
        with zipfile.ZipFile(wheels[module.__name__.split(".")[0]]) as wheel:
            if wheel.read(member) != path.read_bytes():
                raise RuntimeError("installed module differs from executed wheel: " + member)
        paths[module.__name__] = {"path": str(path), "sha256": sha256(path.read_bytes())}

    rows = []
    result_name = {"mlp": "mlp-result", "route": "route-result", "reads": "reads-result",
                   "read-payload": "read-payload-result", "post-context": "post-context-result",
                   "compiled-scan": "compiled-scan-result", "stream-program": "stream-program-result",
                   "response-quotient": "response-quotient-result", "response-history": "response-history-result",
                   "stream-frontier": "stream-frontier-result", "clock-feedback": "clock-feedback-result",
                   "response-route": "response-route-result", "symbolic-loop": "symbolic-loop-result"}.get(
        report.get("phase"), "causal-result" if "cases" in report else "forward-result")
    base = run_dir / "extracted" / result_name
    for step in report.get("steps", report.get("cases", [])):
        path = base / step["trace"]["path"]
        raw = path.read_bytes()
        if sha256(raw) != step["trace"]["sha256"]:
            raise RuntimeError("trace differs from native report: " + str(path))
        trace = json.loads(raw)
        verified = mdb.forward_symbols.verify_trace(trace)
        if trace["fingerprint"] != step["trace"]["fingerprint"]:
            raise RuntimeError("trace seal differs from native report")
        table = mdb.forward_symbols.ForwardSymbolTable(trace)
        operation = len(trace["events"]) // 2
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = mdb.cli.main(["forward-symbols", str(path), "--step", str(operation), "--json"])
        if status != 0 or json.loads(stream.getvalue()) != table.at(operation):
            raise RuntimeError("installed CLI navigation differs from table")
        output_id = trace["outputs"]["regs"]["$value"]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = mdb.cli.main(["forward-symbols", str(path), "--value", output_id, "--json"])
        if status != 0 or json.loads(stream.getvalue()) != table.inspect(output_id):
            raise RuntimeError("installed CLI value navigation differs from table")
        readable = None
        if "cases" in report:
            reader = mdb.forward_readable.ForwardReadableProgram(trace)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = mdb.cli.main(["forward-symbols", str(path), "--readable", "--scope", "mixers.2"])
            if status != 0 or stream.getvalue().strip() != reader.render(scope="mixers.2").strip():
                raise RuntimeError("installed readable CLI differs from program view")
            readable = {"values": len(reader.list_values()), "patterns": len(reader.list_patterns()),
                        "render_sha256": sha256(stream.getvalue().encode())}
            if report.get("phase") in {"algebra", "lowering"}:
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    status = mdb.cli.main(["forward-symbols", str(path), "--algebra", "--json"])
                expected = mdb.forward_algebra.acquire_symbol_candidates(reader)
                if status != 0 or json.loads(stream.getvalue()) != expected:
                    raise RuntimeError("installed algebra CLI differs from graph-derived candidates")
                readable["algebra_candidates"] = len(expected["graph_candidates"])
        rows.append({"program_step": step.get("program_step", step.get("case")), "trace_sha256": sha256(raw),
                     "verified": verified, "operation_inspected": operation,
                     "output_value_inspected": output_id, "readable": readable})

    receipt_count = 0
    fingerprints = {row["trace"]["fingerprint"] for row in report.get("cases", [])}
    for branch in report.get("branches", []):
        if branch["status"] != "observed":
            continue
        if report.get("phase") == "stream-frontier":
            receipt_count += verify_frontier_branch(
                base, branch, saturn_pub.Receipt, saturn_pub.values.digest
            )
            continue
        path = base / branch["receipt"]["path"]
        raw = path.read_bytes()
        verify_branch_receipt(raw, branch, fingerprints, saturn_pub.Receipt)
        receipt_count += 1

    component_receipt_count = 0
    if report.get("phase") == "compiled-scan":
        descriptor = report["artifacts"]["public_component_calls"]
        path = _atlas_file(base, descriptor["path"], "compiled component call path")
        component_receipt_count = verify_component_calls(
            path.read_bytes(), descriptor, saturn_pub.Receipt, saturn_pub.values.digest
        )
    elif report.get("phase") in {"stream-program", "clock-feedback"}:
        component_receipt_count = verify_stream_calls(
            base, report, saturn_pub.Receipt, saturn_pub.values.digest,
            expected_count=28 if report.get("phase") == "clock-feedback" else 56,
        )

    atlas_entries = 0
    if args.atlas:
        atlas = json.loads(args.atlas.read_text())
        for entry in atlas["entries"]:
            refs = {key: entry[key] for key in ("trace", "annotations", "human_table")}
            for ref in refs.values():
                if sha256(Path(ref["path"]).read_bytes()) != ref["sha256"]:
                    raise RuntimeError("atlas referenced artifact changed: " + ref["path"])
            trace = json.loads(Path(refs["trace"]["path"]).read_text())
            annotations = json.loads(Path(refs["annotations"]["path"]).read_text())
            reader = mdb.forward_readable.ForwardReadableProgram(trace, annotations)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = mdb.cli.main(["forward-symbols", refs["trace"]["path"], "--readable",
                                       "--scope", "mixers.2", "--annotations", refs["annotations"]["path"]])
            if (status != 0 or stream.getvalue().strip() != reader.render(scope="mixers.2").strip()
                    or stream.getvalue().strip() != Path(refs["human_table"]["path"]).read_text().strip()
                    or atlas["name"] not in stream.getvalue()):
                raise RuntimeError("atlas display differs from trace-bound human-readable view")
            atlas_entries += 1

    algebra_atlas_tables = 0
    if args.algebra_atlas:
        root = args.algebra_atlas.resolve().parent
        atlas = json.loads(args.algebra_atlas.read_text())
        manifest = json.loads((root / "output-manifest.json").read_text())
        for ref in manifest["files"]:
            path = root / ref["path"]
            if sha256(path.read_bytes()) != ref["sha256"] or path.stat().st_size != ref["bytes"]:
                raise RuntimeError("algebra atlas output differs from its manifest")
        references = json.loads((root / "references.json").read_text())
        for ref in references["inputs"] + references["sources"]:
            if sha256(Path(ref["path"]).read_bytes()) != ref["sha256"]:
                raise RuntimeError("algebra atlas source reference changed")
        for case in atlas["cases"]:
            path = Path(case["trace"]["path"])
            if sha256(path.read_bytes()) != case["trace"]["sha256"]:
                raise RuntimeError("algebra atlas trace changed")
            trace = json.loads(path.read_text())
            if trace["fingerprint"] != case["trace_fingerprint"]:
                raise RuntimeError("algebra atlas trace fingerprint differs")
            annotations = root / case["annotations"]
            reader = mdb.forward_readable.ForwardReadableProgram(trace, json.loads(annotations.read_text()))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = mdb.cli.main(["forward-symbols", str(path), "--readable", "--annotations", str(annotations)])
            if (status != 0 or stream.getvalue().strip() != reader.render().strip()
                    or stream.getvalue().strip() != (root / case["table"]).read_text().strip()):
                raise RuntimeError("algebra atlas table differs from installed native reader")
            for role in ("G", "C", "post"):
                if atlas["working_inferences"][role]["name"] not in stream.getvalue():
                    raise RuntimeError("algebra working inference omitted from symbol table")
            algebra_atlas_tables += 1

    mlp_atlas_tables = 0
    if args.mlp_atlas:
        mlp_atlas_tables = verify_mlp_atlas(
            args.mlp_atlas, mdb.cli, mdb.forward_readable, mdb.forward_symbols
        )

    loaded = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
    if loaded:
        raise RuntimeError("forbidden runtime loaded: " + repr(loaded))
    result = {
        "schema": "mdb-native-forward-symbols-installed-proof-v1", "status": "valid",
        "scope": "metadata seals, graph closure, typed receipts and installed CLI navigation; no tensors loaded",
        "job_id": report["job_id"], "report_sha256": sha256((run_dir / "report.json").read_bytes()),
        "packages": {name: importlib.metadata.version(name) for name in ("mdb-runtime", "saturn-pub")},
        "wheel_sha256": {name: sha256(path.read_bytes()) for name, path in wheels.items()},
        "installed_modules": paths, "blocked_module_roots": sorted(BLOCKED),
        "frameworks_or_private_modules_loaded": loaded, "traces": rows,
        "verified_typed_branch_receipts": receipt_count,
        "verified_compiled_component_receipts": component_receipt_count,
        "atlas_entries_verified": atlas_entries,
        "algebra_atlas_tables_verified": algebra_atlas_tables,
        "mlp_atlas_tables_verified": mlp_atlas_tables,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "valid", "traces": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
