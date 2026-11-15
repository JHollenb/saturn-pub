"""Normalize allowlisted authenticated metadata; never open scientific outcomes.

The optional evaluator replay uses only the installed public MDB module. No
source-checkout fallback, tensor/weight read, model import or service is allowed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent
TRUSTED = {
    "turn14": dict(root=ARTIFACTS / "mdb-qwen-row-sign-20261106/compact-collection-01/extracted/binding-minimal-result",
        manifest="82032b7ad875735d14a3c3f2ac2cc7dbb9bc342b3ae9373f2a8fbe349f06906f"),
    "turn15": dict(root=ARTIFACTS / "mdb-qwen-cached-qkv-20261106/compact-collection-01/extracted/binding-minimal-result",
        manifest="c662757ab5f261b91f72006bd3de57291e1ff70e208c38e9e30631cb6c1a3ca7"),
}
ALLOWLIST = (
    "source-plan.json", "source-source.json", "source-runtime-contract.json", "source-source-frontier.json",
    "source-metadata/qwen_config.json", "source-metadata/qwen_parameter_manifest.json", "source-metadata/qwen_source_binding.json",
    "source-metadata/parent_files.json", "source-metadata/parent_seal.json", "source-metadata/selection_protocol.json",
    "catalog/provenance.json", "coverage.json",
)
DESCRIPTOR_KEYS = ("path", "sha256", "bytes", "shape", "stride", "logical_dtype", "stored_dtype", "content_sha256", "device_at_capture")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class MetadataReader:
    """One explicit read barrier; even hashing an unlisted file is forbidden."""

    def __init__(self, root, expected_manifest, label):
        self.root, self.label, self.reads = Path(root), label, []
        raw = self._read_bytes("files.json")
        require(hashlib.sha256(raw).hexdigest() == expected_manifest, "Trusted compact wrapper differs")
        self.manifest = json.loads(raw)
        require(isinstance(self.manifest, dict) and all(isinstance(v, str) and len(v) == 64 for v in self.manifest.values()), "Compact wrapper schema differs")

    def _read_bytes(self, name):
        require(name == "files.json" or name in ALLOWLIST, "Metadata read forbidden: " + name)
        path = self.root / name
        require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "Metadata symlink refused")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if name != "files.json":
            require(self.manifest.get(name) == sha, "Allowlisted metadata hash differs: " + name)
        self.reads.append(dict(packet=self.label, file=name, bytes=len(raw), sha256=sha))
        return raw

    def read(self, name):
        return json.loads(self._read_bytes(name))


def metadata(root, expected_manifest, label):
    reader = MetadataReader(root, expected_manifest, label)
    documents = {name: reader.read(name) for name in ALLOWLIST}
    plan, source, frontier = (documents[name] for name in ("source-plan.json", "source-source.json", "source-source-frontier.json"))
    protocol = documents["source-metadata/selection_protocol.json"]
    require(frontier["source_arrays"] == {name: dict(row, inherited=True, root="input_root") for name,row in protocol["source_arrays"].items()}, "Declared source-array metadata differs")
    require(source["job_id"] == plan["job_id"] and source["request_sha256"] == plan["request_sha256"], "Source/plan identity differs")
    for key in ("qwen_config", "qwen_parameter_manifest", "qwen_source_binding", "parent_files", "parent_seal", "selection_protocol"):
        name = "source-metadata/" + key + ".json"
        require(plan[key + "_sha256"] == reader.manifest[name] == source["metadata"][key]["sha256"], "Source metadata identity differs: " + key)
    original = documents["source-metadata/parent_files.json"]
    seal = documents["source-metadata/parent_seal.json"]
    require(frontier["parent_job"] == original["job_id"] and frontier["parent_request_sha256"] == original["request_sha256"], "Original parent identity differs")
    for name, row in protocol["source_arrays"].items():
        wanted = dict(bytes=row["bytes"], sha256=row["sha256"])
        require(original["files"].get(row["path"]) == seal["declarations"]["frontier"]["external_files"].get(row["path"]) == wanted, "Source not joined to original inventory/global seal")
        require(set(DESCRIPTOR_KEYS) <= set(row), "Incomplete source descriptor: " + name)
    return documents, reader.reads


def positive_ports(documents):
    """Availability is packet-local. No reconstruction, aliases or implicit join."""
    source = documents["source-source-frontier.json"]["source_arrays"]
    return {name: {key: copy.deepcopy(row[key]) for key in DESCRIPTOR_KEYS} for name,row in source.items()
            if name in {"h", "head", "R", "gate", "up", "product", "M", "Hnext", "q", "k", "v"}}


CONTEXT_NAMES = {"input_ids", "attention_mask", "position_ids", "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq"}
COUNT_KEYS = {"primary_endpoint_fields", "primary_response_fields", "overlapping_endpoint_windows", "overlapping_response_windows", "QKV_channel_fields", "QKV_source_sign_fields"}


def normalize(label, documents):
    plan, source, runtime, frontier, config, recipes, protocol = (documents[name] for name in (
        "source-plan.json", "source-source.json", "source-runtime-contract.json", "source-source-frontier.json",
        "source-metadata/qwen_config.json", "source-metadata/qwen_parameter_manifest.json", "source-metadata/selection_protocol.json"))
    raw = protocol["source_arrays"]
    ports = {name: copy.deepcopy(row) for name,row in raw.items() if name not in CONTEXT_NAMES}
    require(set(raw) - set(ports) == CONTEXT_NAMES, "Exact five source context descriptors required")
    kind = {"turn14": "rgu", "turn15": "qkv"}[label]
    expected = {"h", "head", "R", "M", "Hnext", *({"gate", "up", "product"} if kind == "rgu" else {"q", "k", "v"})}
    require(set(ports) == expected, "Packet-local port availability differs; no implicit union allowed")
    binding = {key: frontier[key] for key in ("parent_job", "parent_request_sha256", "parent_raw_manifest_sha256", "parent_global_seal_sha256", "source_case")}
    binding.update(producer_sha256=plan["helper_source_sha256"]["qwen_sign_shortcut_campaign.py"],
        qwen_config_sha256=plan["qwen_config_sha256"], qwen_source_binding_sha256=plan["qwen_source_binding_sha256"],
        parameter_manifest_sha256=plan["qwen_parameter_manifest_sha256"], parent_parameter_manifest_sha256=plan["parent_parameter_manifest_sha256"],
        selection_protocol_sha256=plan["selection_protocol_sha256"])
    packet = dict(schema="mdb-qwen-source-packet-v1", packet_id=label, kind=kind,
        frontier=copy.deepcopy(frontier.get("transaction_frontier", frontier.get("frontier"))), binding=binding,
        ports=ports, context={name: copy.deepcopy(raw[name]) for name in CONTEXT_NAMES},
        parameters={row["source"]["tensor"]: copy.deepcopy(row["source"]) for row in recipes["artifacts"]})
    require(len(packet["parameters"]) == 11 and set(packet["parameters"]) == set(source["selected_original_names"]), "Original eleven parameter names differ")
    profile = dict(torch=runtime["torch"], transformers=runtime["transformers"], device=raw["h"]["device_at_capture"], dtype=raw["h"]["logical_dtype"],
        native_source_sha256=runtime["sources"]["qwen_native_source_sha256"]["sha256"], sdpa_source_sha256=runtime["sources"]["sdpa_source_sha256"]["sha256"],
        geometry=dict(batch_size=raw["h"]["shape"][0], sequence_length=plan["sequence_rows"], hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"], num_attention_heads=config["num_attention_heads"], num_key_value_heads=config["num_key_value_heads"], head_dim=config["head_dim"], layer=plan["edit_index"]),
        attention=dict(mask=frontier["context"]["attention_mask"], is_causal=frontier["context"]["is_causal"], enable_gqa=frontier["context"]["enable_gqa"],
            dropout_p=config["attention_dropout"], scale=config["head_dim"] ** -0.5, use_cache=frontier["context"]["use_cache"], past_key_values=frontier["context"]["past_key_values"]),
        positions=dict(start=0, step=1, length=plan["sequence_rows"]), rope_buffer_policy=copy.deepcopy(plan["qwen_rope_buffer_policy"]), flags=copy.deepcopy(runtime["flags"]))
    rows = []
    for arm in protocol["arms"]:
        signs = arm.get("row_signs")
        if signs is None:  # turn14 declares ranges, not a retained sign vector.
            signs = [1] * plan["sequence_rows"]
            ranges = [arm["negative_rows"]] if "negative_rows" in arm else arm["negative_ranges"]
            for start, stop in ranges:
                signs[start:stop] = [-1] * (stop - start)
        windows = arm.get("windows")
        if windows is None:
            split = protocol["fixed_split"]
            windows = [[0, split], [split, plan["sequence_rows"]]]
        rows.append(dict(case_id=label + "." + arm["id"], packet_id=label, arm=arm["id"],
            intervention=dict(row_signs=signs, permitted=dict(kind="causal-all-attended", rows=plan["sequence_rows"]), windows=windows)))
    coverage = documents["coverage.json"]
    require(all(set(counts) <= COUNT_KEYS and all(type(value) is int and value >= 0 for value in counts.values()) for counts in coverage["arms"].values()), "Coverage must be counts only")
    occurrence = []
    for record in documents["catalog/provenance.json"]["records"]:
        require(set(record) == {"arm", "context", "fingerprint"}, "Unexpected occurrence metadata schema")
        occurrence.append(dict(arm=record["arm"], occurrence_id=record["fingerprint"], method_id="m0" if kind == "rgu" else "m1",
            binding=copy.deepcopy(record["context"]), coverage_counts=copy.deepcopy(coverage["arms"][record["arm"]]),
            evidence_refs=[label + ":catalog/provenance.json#" + record["fingerprint"], label + ":coverage.json#" + record["arm"]]))
    return packet, profile, rows, occurrence


def prepare(output):
    output = Path(output)
    require(not output.exists(), "Fresh packet directory required")
    output.mkdir(parents=True)
    reads, rows, index = [], [], []
    for label, trusted in TRUSTED.items():
        documents, log = metadata(trusted["root"], trusted["manifest"], label)
        packet, profile, cases, occurrences = normalize(label, documents)
        save(output / "packets" / (label + ".json"), packet)
        save(output / "profiles" / (label + ".json"), profile)
        reads.extend(log); rows.extend(cases); index.extend(dict(packet_id=label, **row) for row in occurrences)
    save(output / "cases.json", rows)
    save(output / "output-spec.json", dict(ports=["head", "R", "M", "Hnext"], response="edited-minus-own-positive", horizon=19, continuation=None))
    save(output / "policy.json", dict(order=["m0", "m1", "m2"], allow_full_functional_fallback=True))
    save(output / "coverage-occurrences.json", dict(schema="mdb-source-coverage-projection-v1", occurrences=index,
        scope="Immutable historical occurrence metadata and counts only. No error/pass/quality data; no relabeling old m0 evidence as primary m2 evidence."))
    save(output / "read-manifest.json", dict(schema="mdb-source-only-read-manifest-v1", reads=reads,
        trusted_wrappers={label: value["manifest"] for label,value in TRUSTED.items()}, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forbidden_files_opened=0, arrays_or_weights_read=0, evaluator_calls=0,
        profile_scope="Position start/step and dispatch scale/dropout derive from qualified source contract/config, not newly loaded tensor values. Selected kernels remain unprofiled."))
    save(output / "inputs-manifest.json", {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.rglob("*.json"))})
    return dict(output=str(output), packets=2, cases=len(rows), evaluator_calls=0, read_files=len(reads))


def controls(packet, profile, intervention, output_spec, policy):
    """Synthetic metadata changes; none contains a measured target value."""
    cases = []
    def add(name, p=None, e=None, i=None, o=None, policy_override=None):
        cases.append(dict(control=name, packet=p or copy.deepcopy(packet), profile=e or copy.deepcopy(profile),
            intervention=i or copy.deepcopy(intervention), output=o or copy.deepcopy(output_spec), policy=policy_override or copy.deepcopy(policy)))
    p = copy.deepcopy(packet); p["binding"]["producer_sha256"] = "0" * 64; add("changed-source", p=p)
    e = copy.deepcopy(profile); e["flags"]["cuda_allow_bf16_reduced_precision_reduction"] = False; add("changed-profile", e=e)
    e = copy.deepcopy(profile); e["attention"]["use_cache"] = True; e["attention"]["past_key_values"] = {"past_length": 1}; add("cache-added", e=e)
    p = copy.deepcopy(packet)
    for name in (("gate", "up") if packet["kind"] == "rgu" else ("q", "k", "v")):
        p["ports"].pop(name, None)
    add("missing-prediction-ports", p=p)
    p = copy.deepcopy(packet); p["ports"].pop("head"); add("missing-output-baseline", p=p)
    o = copy.deepcopy(output_spec); o["continuation"] = dict(consumer_id="unqualified-block19", input_port="hidden", horizon=20,
        source_identity="unbound", parameter_instance="unbound", execution_profile=copy.deepcopy(profile)); add("unqualified-continuation", o=o)
    add("fallback-disabled", policy_override=dict(order=policy["order"], allow_full_functional_fallback=False))
    return cases


def replay(inputs, output, expected_manifest, expected_module):
    import importlib
    import inspect

    inputs, output = Path(inputs), Path(output)
    manifest_raw = (inputs / "inputs-manifest.json").read_bytes()
    require(hashlib.sha256(manifest_raw).hexdigest() == expected_manifest, "Frozen packet input manifest differs")
    manifest = json.loads(manifest_raw)
    for name, sha in manifest.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and (inputs / name).suffix == ".json", "Unsafe packet member")
        require(hashlib.sha256((inputs / name).read_bytes()).hexdigest() == sha, "Frozen packet input changed")
    module = importlib.import_module("mdb.computation_applicability")
    source = Path(inspect.getfile(module)).resolve()
    require("site-packages" in source.parts and hashlib.sha256(source.read_bytes()).hexdigest() == expected_module, "Installed evaluator source differs")
    require(not output.exists(), "Fresh replay output required")
    output.mkdir(parents=True)
    load = lambda name: json.loads((inputs / name).read_text())
    spec, policy = load("output-spec.json"), load("policy.json")
    rows, occurrence_index, annotations = [], [], load("coverage-occurrences.json")["occurrences"]
    for case in load("cases.json"):
        packet, profile = load("packets/" + case["packet_id"] + ".json"), load("profiles/" + case["packet_id"] + ".json")
        decision = module.evaluate(packet, case["intervention"], profile, spec, policy)
        rows.append(dict(case=case, decision=decision))
        for item in annotations:
            if item["packet_id"] == case["packet_id"] and item["arm"] == case["arm"]:
                occurrence_index.append(dict(occurrence_id=item["occurrence_id"], decision_sha256=decision["decision_sha256"],
                    method_id=item["method_id"], account_kinds=list(item["coverage_counts"]), evidence_refs=item["evidence_refs"]))
    index = dict(schema="mdb-computation-coverage-index-v1", index_id=digest(annotations), occurrences=occurrence_index)
    control_rows = []
    for row in rows:
        case = row["case"]
        packet, profile = load("packets/" + case["packet_id"] + ".json"), load("profiles/" + case["packet_id"] + ".json")
        annotated = module.evaluate(packet, case["intervention"], profile, spec, policy, coverage_index=index)
        require(annotated["decision_sha256"] == row["decision"]["decision_sha256"], "Coverage index changed source-only decision")
        row["annotated"] = annotated
        for control in controls(packet, profile, case["intervention"], spec, policy):
            try:
                result = module.evaluate(control["packet"], control["intervention"], control["profile"], control["output"], control["policy"])
                control_rows.append(dict(case_id=case["case_id"], control=control["control"], decision=result))
            except (ValueError, TypeError) as exc:
                control_rows.append(dict(case_id=case["case_id"], control=control["control"], status="input-refused", reason=str(exc)))
    save(output / "plans.json", rows); save(output / "controls.json", control_rows); save(output / "coverage-index.json", index)
    save(output / "replay-receipt.json", dict(schema="mdb-source-only-replay-receipt-v1", installed_module=str(source), module_sha256=expected_module,
        inputs_manifest_sha256=expected_manifest, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        main_cases=len(rows), controls=len(control_rows), model_calls=0, arrays_or_weights_read=0,
        outcome_status="not-read-or-executed", scope="Metadata applicability replay only. Historical occurrence binding is a retrospective projection; method mismatch remains unmatched and never becomes a finite accuracy certificate."))
    return dict(output=str(output), main_cases=len(rows), controls=len(control_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("replay"); run.add_argument("--inputs", type=Path, required=True); run.add_argument("--out", type=Path, required=True)
    run.add_argument("--expected-input-manifest", required=True); run.add_argument("--expected-module-sha256", required=True)
    args = parser.parse_args()
    result = prepare(args.out) if args.mode == "prepare" else replay(args.inputs, args.out, args.expected_input_manifest, args.expected_module_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
