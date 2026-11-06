"""Opt-in source-only planning for one qualified Qwen block 18 interface.

This module reads no files, arrays, models or outcome metrics. A selected method
is a whole-component prospective candidate, never a finite accuracy certificate.
"""
from __future__ import annotations

import hashlib
import json

from .equation_comparison import applicability

__all__ = ["evaluate"]

SOURCE_BINDING = {
    "parent_job": "job-8c949b1c0278",
    "parent_request_sha256": "1a9b11e4bc739410dcd165b31cd0c0452aa5f0583e7731598bc1e2b8a87bdd5a",
    "parent_raw_manifest_sha256": "d573bffe5c0c3dfc3d1c353721f5696909f73b75fb6ff293aed2e905f4ec0c33",
    "parent_global_seal_sha256": "a6d61b11fa67bfa0eb3a50fee66f26584b452874a590c255163102362460e652",
    "source_case": "long-description",
    "producer_sha256": "83f9d459938ccb96b913def78862cffc869cb6f0b4404a7f116d0a46fed184af",
    "qwen_config_sha256": "214b4c29a0d975e9fddf9994a5673f22cb2c4c5750352f9227c2c3251ebeab40",
    "qwen_source_binding_sha256": "663707b6a30042ff89e90f05560766c23f49b084b33385be09028c78e15859c7",
    "parameter_manifest_sha256": "425502121b4091b10124c548c98127785420233b12e95c8f1d972a186026dccf",
    "parent_parameter_manifest_sha256": "d5060dd8137bf34568d3d9b0fda805c5979133b0c4a4538e23579196cb5d3374",
}
SUPPORTED_PROFILE = {
    "torch": "2.13.0+cu130", "transformers": "5.14.1", "device": "cuda:0", "dtype": "torch.bfloat16",
    "native_source_sha256": "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab",
    "sdpa_source_sha256": "b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46",
    "geometry": {"batch_size": 1, "sequence_length": 512, "hidden_size": 2560, "intermediate_size": 9728,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "layer": 18},
    "attention": {"mask": None, "is_causal": True, "enable_gqa": True, "dropout_p": 0.0,
        "scale": 128**-0.5, "use_cache": False, "past_key_values": None},
    "positions": {"start": 0, "step": 1, "length": 512},
    "rope_buffer_policy": {"init_device": "cpu", "init_dtype": "float32", "placement_dtype": "float32",
        "source": "original native CPU initializer; device-only placement"},
    "flags": {"deterministic_algorithms": True, "float32_matmul_precision": "highest", "cuda_allow_tf32": False,
        "cuda_allow_bf16_reduced_precision_reduction": True, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True, "num_threads": 4},
}
_FRONTIER = {"kind": "architectural", "origin": "inherited-functional-state", "scope": "conditional block18 hidden-state interface",
    "raw_input_prediction": False, "genuine_record_prefix_capture": False}
_PROTOCOLS = {"rgu": "d942001a79784ebb1bc5d5ed3f3946eaa03b4629f2fe13c7493afbaaf85b51e9",
    "qkv": "8c53ae9417ba40420e806cb634d59931fb0131f95d0329e5b34e078429503bff"}
_PORTS = {"rgu": {"h", "head", "R", "gate", "up", "product", "M", "Hnext"},
    "qkv": {"h", "head", "R", "q", "k", "v", "M", "Hnext"}}
_CONTEXT = {"input_ids", "attention_mask", "position_ids", "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq"}
_RECORD_DIGESTS = {
    "Hnext": "1ba73f0767be82c1e49699a7daff2b2498c9499fc484b459f0aadc1c594f6f57",
    "M": "6d5f5da93d8c37c6f54e1218a4ac8d982eab061c835a6db56eb6333c6c420ec7",
    "R": "4d408742b73b99fe482d075bf283dc3343e4ef05485bf3fc192f55c0c1d29222",
    "attention_mask": "68c182b28b1be19173b0fe9aaec972b7c77c1fac825ed82b37842a21e7e7e243",
    "gate": "3a4264d3b1e71c80f1f9ae9fc9006d0564125fa9c45f2d2611d1b5783753520f",
    "h": "333a4b0a8fd563e427cc7a3db806580b4b2c6062a25108b1316b5dd69a6d27e6",
    "head": "fb5ce50797289f03aa231bbc6335d546eaf7cfd18141d908fe51f8fe604b8fae",
    "input_ids": "e95d303609c91447a3db21cb6fa68e4692f74ccaf82b74563d1bf3f4a7148c80",
    "k": "2bbcf2df6f53d3432f946e5b98c01fe9121d225fbbe823db38e7c1eedba6fda4",
    "position_ids": "01ec56091c4e1a1f5e470c047915b5c13a81e0a007b595d83b54332c12ed2357",
    "product": "0fd66e4cfbd1727496ac0e0a574c2bfd9ae2b0ee85208803a03e15d04923d5f2",
    "q": "19424f483a85de5f19c39dca2e863e1e7749eafa71eb3b980ab75eafbf76dec0",
    "rope_buffer:model.rotary_emb.inv_freq": "1d149f184da6ddf585dce03661007c6d85b5100b8c8254527025df0b22e7c11c",
    "rope_buffer:model.rotary_emb.original_inv_freq": "424a04534a10c7766537be0983d6b7455ebfa400a03e639dfbbbda01293ac966",
    "up": "2ab9145edf0187b476c3ddae5e3001391af9aa198d80b0ef9212e5c0950b1e67",
    "v": "d1083478488303bd0c14cb1dffad80b6a26edff0c6072796fd5d0711a525adfb",
}
_PARAMETER_DIGESTS = {
    "input_layernorm": "0edd5d3154dc86f6b1d8e1a94b826b16b82c80a9d4a749b8ce1eb305e6258d05",
    "mlp.down_proj": "0ebfd5d76104633719ee3bb2bb7f8dd63a46b6f885ca70d8849ddd0e3fb34601",
    "mlp.gate_proj": "e38f29e9d6e6e174ed85a51397a3a380c9ba14d6720115af09c8d6afbb7e7fb2",
    "mlp.up_proj": "a4e02d0dd41b475bad49e54cd642fa3fc716bae277131024d2690562866e071a",
    "post_attention_layernorm": "7078bec421b1ca549b3fcf93355ba96f69ad2100be4ddd11a8b4fce2487be6dc",
    "self_attn.k_norm": "ecbdc5b79febfd8397f4b79e2055ab09cc4226a5a8a0e788b480046fc85c229f",
    "self_attn.k_proj": "fb18017302e08769b48bcf044df7354735762159d3cac5ca1d6f096dfc2948d7",
    "self_attn.o_proj": "166caf5531f4a56e9ea4dfde8c31a9fade398f8aeecb66b6b0847c4e4052d007",
    "self_attn.q_norm": "a91230dc7b2e283e411516b1e60f8e52de9fa7d024fe4c2943b526789361a1c1",
    "self_attn.q_proj": "73456ba65e6603f401841a504eb0208fc5957e2aef594c022ae4d7f45f2dde0b",
    "self_attn.v_proj": "7005599e34c80b2f2f847a9e6a7c9cf71d6bf724cbe8647683805f9565d7f2f4",
}
_PARAMETER_DIGESTS = {"model.layers.18." + name + ".weight": digest for name, digest in _PARAMETER_DIGESTS.items()}
_PARAMETERS = {"m0": {"mlp.down_proj"}, "m1": {"self_attn.o_proj", "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}}
_PARAMETERS = {method: {"model.layers.18." + name + ".weight" for name in names} for method, names in _PARAMETERS.items()}
_PARAMETERS["m2"] = set(_PARAMETER_DIGESTS)
_REQUIRED_PORTS = {"m0": {"h", "head", "R", "gate", "up", "M", "Hnext"},
    "m1": {"h", "head", "R", "q", "k", "v", "M", "Hnext"}, "m2": {"h", "head", "R", "M", "Hnext"}}
_OPERATIONS = {
    "m0": {"qwen_row_sign_shortcut.py": "940c3cfaafeaeb3f0df5746b92b6a8b0878255769c4f230b2a53bedfc0110c1c",
        "qwen_sign_shortcut.py": "8e59051a34bc8e89c9376fd29aec966df9cbe84560c81f86910504172caaa457"},
    "m1": {"qwen_cached_qkv.py": "95d356185b8bdd64479bc1e7937fa034aa24dc5f5b011b7075f0430e31c1bb21",
        "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a"},
    "m2": {"qwen_h18_state_history.py": "17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036",
        "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a"},
}
_SOURCE_FIELDS = {"schema", "packet_id", "kind", "frontier", "binding", "ports", "context", "parameters"}
_OUTCOME_FIELDS = {"target", "targets", "error", "errors", "pass", "passed", "accuracy", "outcome", "outcomes", "actual_outcome", "observations", "observed", "rms", "max_abs"}
# Public inspection dictionaries cannot alter the evaluator's frozen domain.
_PROFILE_JSON = json.dumps(SUPPORTED_PROFILE, sort_keys=True, separators=(",", ":"))
_BINDING_JSON = json.dumps(SOURCE_BINDING, sort_keys=True, separators=(",", ":"))


def _copy(value):
    try:
        result = json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("applicability inputs must be finite JSON metadata") from error
    def reject(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower() in _OUTCOME_FIELDS or key.lower().startswith(("error_", "target_", "outcome_", "pass_")):
                    raise ValueError("target/error/pass/outcome inputs are outside the source-only schema")
                reject(child)
        elif isinstance(item, list):
            for child in item:
                reject(child)
    reject(result)
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _checks(expected, actual, prefix):
    if not isinstance(actual, dict):
        return [{"field": prefix, "status": "unknown", "reason": "missing-object"}]
    conditions = [{"field": key, "op": "eq", "value": value} for key, value in expected.items()]
    results = applicability(conditions, actual)["conditions"]
    checks = []
    for row in results:
        key = row["condition"]["field"]
        status = {"matches": "matched", "unknown": "unknown", "does-not-match": "unsupported"}[row["status"]]
        if status == "matched" and _digest(actual[key]) != _digest(expected[key]):
            status = "unsupported"  # Keep JSON bool/int and exact nested schemas distinct.
        checks.append(dict(field=prefix + "." + key, status=status, reason="declared-domain-predicate"))
    checks.extend(dict(field=prefix + "." + key, status="unsupported", reason="undeclared-field") for key in sorted(actual.keys() - expected.keys()))
    return checks


def _status(checks):
    statuses = {check["status"] for check in checks}
    return "unsupported" if "unsupported" in statuses else "unknown" if "unknown" in statuses else "matched"


def _source(packet):
    if not isinstance(packet, dict):
        raise ValueError("source_packet must be an object")
    checks = _checks({"schema": "mdb-qwen-source-packet-v1"}, {key: packet[key] for key in ("schema",) if key in packet}, "source")
    checks.extend(dict(field="source." + key, status="unsupported", reason="undeclared-field") for key in sorted(packet.keys() - _SOURCE_FIELDS))
    kind = packet.get("kind") if isinstance(packet.get("kind"), str) else None
    checks.append(dict(field="source.kind", status="matched" if kind in _PORTS else "unknown" if kind is None else "unsupported", reason="declared-packet-kind"))
    expected_binding = json.loads(_BINDING_JSON)
    if kind in _PROTOCOLS:
        expected_binding["selection_protocol_sha256"] = _PROTOCOLS[kind]
    checks.extend(_checks(expected_binding, packet.get("binding"), "source.binding"))
    checks.extend(_checks(_FRONTIER, packet.get("frontier"), "source.frontier"))
    ports, context, parameters = (packet.get(name, {}) for name in ("ports", "context", "parameters"))
    for label, records, allowed, digests in (("ports", ports, _PORTS.get(kind, set()), _RECORD_DIGESTS),
            ("context", context, _CONTEXT, _RECORD_DIGESTS), ("parameters", parameters, set(_PARAMETER_DIGESTS), _PARAMETER_DIGESTS)):
        if not isinstance(records, dict):
            checks.append(dict(field="source." + label, status="unknown", reason="missing-record-map"))
            continue
        for name, record in records.items():
            valid = name in allowed and _digest(record) == digests.get(name)
            checks.append(dict(field=f"source.{label}.{name}", status="matched" if valid else "unsupported", reason="immutable-original-metadata-digest"))
    if isinstance(context, dict):
        checks.extend(dict(field="source.context." + name, status="unknown", reason="missing-context") for name in sorted(_CONTEXT - context.keys()))
    return dict(status=_status(checks), checks=checks), ports if isinstance(ports, dict) else {}, parameters if isinstance(parameters, dict) else {}


def _structural(intervention, profile):
    if not isinstance(intervention, dict) or set(intervention) != {"row_signs", "permitted", "windows"}:
        raise ValueError("intervention requires only row_signs/permitted/windows")
    signs, permitted, windows = (intervention[key] for key in ("row_signs", "permitted", "windows"))
    if not isinstance(signs, list) or len(signs) != 512 or any(type(sign) is not int or sign not in (-1, 1) for sign in signs):
        raise ValueError("one exact integer +/-1 sign for each of 512 rows required")
    if (not isinstance(windows, list) or not windows or any(not isinstance(window, list) or len(window) != 2
            or any(type(value) is not int for value in window) or not 0 <= window[0] < window[1] <= 512 for window in windows)
            or windows[0][0] != 0 or windows[-1][1] != 512 or any(left[1] != right[0] for left, right in zip(windows, windows[1:]))):
        raise ValueError("declared windows must partition all 512 query rows")
    attention = profile.get("attention", {}) if isinstance(profile, dict) else {}
    attention = attention if isinstance(attention, dict) else {}
    bound = _digest(permitted) == _digest({"kind": "causal-all-attended", "rows": 512}) and attention.get("mask", "unknown") is None
    bound = bound and attention.get("is_causal") is True and attention.get("use_cache") is False and attention.get("past_key_values", "unknown") is None
    if not bound:
        return dict(status="unknown", reason="permitted-mask-or-cache-not-bound", closed_rows_per_query=None, closed_rows=None,
            cross_sign_edges=None, windows=[dict(bounds=window, rows=window[1] - window[0], closed_rows=None) for window in windows])
    counts = {1: 0, -1: 0}; cross, closed = [], []
    for sign in signs:
        counts[sign] += 1
        cross.append(counts[-sign]); closed.append(counts[-sign] == 0)
    return dict(status="computed", reason="nonempty causal permitted prefix shares query sign", closed_rows_per_query=closed,
        closed_rows=sum(closed), cross_sign_edges=sum(cross), windows=[dict(bounds=[start, stop], rows=stop-start,
            closed_rows=sum(closed[start:stop]), cross_sign_edges=sum(cross[start:stop])) for start, stop in windows],
        scope="Structural sufficient ideal-arithmetic premise only; all query rows retained; no attention-mass or BF16-bit guarantee")


def _continuation(value):
    if value is None:
        return dict(status="not-requested", local_horizon=19, authority="local component only")
    fields = {"consumer_id", "input_port", "horizon", "source_identity", "parameter_instance", "execution_profile"}
    complete = isinstance(value, dict) and set(value) == fields and all(value[key] not in (None, "", {}) for key in fields)
    return dict(status="unqualified" if complete else "unknown", requested=value,
        reason="No continuation cell is qualified by this local evaluator", local_support_unchanged=True)


def _coverage(index, decision, method):
    base = dict(raw_bit_identity_implied=False, numerical_identity_implied=False, signed_zero_accounts="separate port-specific observations",
        outcomes_read=False, authority="retained-account metadata only; never a selector")
    if index is None:
        return dict(base, status="unqueried", occurrence_refs=[], account_kinds=[]), dict(status="unqueried", exact_key=decision, refs=[])
    if not isinstance(index, dict) or set(index) - {"schema", "index_id", "occurrences"}:
        raise ValueError("coverage_index requires only schema/index_id/occurrences metadata")
    if index.get("schema") != "mdb-computation-coverage-index-v1" or not isinstance(index.get("occurrences"), list):
        return dict(base, status="incomplete-index", occurrence_refs=[], account_kinds=[]), dict(status="unknown-index", exact_key=decision, refs=[])
    matches, incomplete = [], 0
    fields = {"occurrence_id", "decision_sha256", "method_id", "account_kinds", "evidence_refs"}
    for entry in index["occurrences"]:
        if (not isinstance(entry, dict) or set(entry) != fields or not isinstance(entry.get("occurrence_id"), str)
                or not isinstance(entry.get("decision_sha256"), str) or len(entry["decision_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in entry["decision_sha256"]) or entry.get("method_id") not in ("m0", "m1", "m2")
                or not isinstance(entry.get("account_kinds"), list) or not isinstance(entry.get("evidence_refs"), list)
                or any(not isinstance(value, str) or not value for key in ("account_kinds", "evidence_refs") for value in entry[key])):
            incomplete += 1
            continue
        if entry["decision_sha256"] == decision and entry["method_id"] == method and method is not None:
            matches.append(entry)
    refs = sorted({entry["occurrence_id"] for entry in matches})
    kinds = sorted({kind for entry in matches for kind in entry["account_kinds"]})
    return dict(base, status="recorded-exact-occurrence" if matches else "incomplete-index" if incomplete else "no-exact-occurrence",
        index_id=index.get("index_id"), occurrence_refs=refs, account_kinds=kinds, incomplete_entries=incomplete), dict(
            status="exact-occurrence" if matches else "no-exact-occurrence", exact_key=decision, refs=refs,
            evidence_refs=sorted({ref for entry in matches for ref in entry["evidence_refs"]}), authority="recorded occurrence only; no generalization")


def evaluate(source_packet, intervention, execution_profile, output_spec, policy, *, coverage_index=None):
    """Plan one whole block from declared source metadata, before coverage lookup.

    Supported source records are the two original inherited packets only. Missing
    availability is explicit; unknown source/profile/cache never implies a null.
    Results own their nested data. Outcomes are not an accepted input.
    """
    packet, intervention, profile, output, policy = (_copy(value) for value in (source_packet, intervention, execution_profile, output_spec, policy))
    if (not isinstance(policy, dict) or set(policy) != {"order", "allow_full_functional_fallback"} or not isinstance(policy["order"], list)
            or len(policy["order"]) != 3 or any(not isinstance(method, str) for method in policy["order"])
            or set(policy["order"]) != {"m0", "m1", "m2"} or type(policy["allow_full_functional_fallback"]) is not bool):
        raise ValueError("policy requires declared m0/m1/m2 order and explicit Boolean fallback availability")
    if not isinstance(output, dict) or set(output) - {"ports", "response", "horizon", "continuation"}:
        raise ValueError("output_spec contains undeclared fields")
    source, ports, parameters = _source(packet)
    profile_checks = _checks(json.loads(_PROFILE_JSON), profile, "profile")
    output_checks = _checks({"ports": ["head", "R", "M", "Hnext"], "response": "edited-minus-own-positive", "horizon": 19},
        {key: value for key, value in output.items() if key != "continuation"}, "output")
    structural = _structural(intervention, profile)
    checks = source["checks"] + profile_checks + output_checks
    if structural["status"] != "computed":
        checks.append(dict(field="intervention.permitted", status="unknown", reason=structural["reason"]))
    domain = _status(checks)
    methods = {}
    for method in ("m0", "m1", "m2"):
        missing_ports = sorted(_REQUIRED_PORTS[method] - ports.keys())
        missing_parameters = sorted(_PARAMETERS[method] - parameters.keys())
        structure = structural["status"] == "computed" and (method != "m0" or structural["closed_rows"] == 512)
        allowed = method != "m2" or policy["allow_full_functional_fallback"]
        eligible = domain == "matched" and structure and not missing_ports and not missing_parameters and allowed
        methods[method] = dict(domain_support=domain, structural_eligibility=structure, missing_ports=missing_ports,
            missing_parameters=missing_parameters, policy_allowed=allowed, eligible=eligible,
            finite_identity_implied=False, query_bounds=[0, 512], operation_sources=_OPERATIONS[method])
    selected = next((method for method in policy["order"] if methods[method]["eligible"]), None)
    selection = dict(method_id=selected, status="fallback" if selected == "m2" else "candidate" if selected else "abstain",
        query_bounds=[0, 512], query_policy="whole-component; no query slicing or silent port union",
        reason="declared priority among source/domain/structure-eligible methods" if selected else "explicit missing/unsupported prerequisites",
        fallback_plan=None if selected else dict(method_id="m2", status="unqualified-or-unavailable",
            required="same qualified block18 domain, complete h/baselines/context/original eleven recipes and explicit fallback policy"),
        policy_authority="declared computation order; no measured speed claim")
    binding = dict(source={key: value for key, value in packet.items() if key != "packet_id"}, execution_profile=profile,
        intervention=intervention, output_spec=output, frontier=_FRONTIER, horizon=19)
    continuation = _continuation(output.get("continuation"))
    body = dict(binding=binding, structural=structural, methods=methods, selection=selection, policy=policy,
        local_support=dict(status=domain, checks=checks, payload_validation="not performed; runner must verify payload/device/finite/custody",
            profile_authority="declared-source-bound; no fresh runtime or hardware attestation", selected_kernels="unprofiled", hardware_attestation="not supplied"),
        continuation_support=continuation)
    decision = _digest(body)
    # Coverage is copied and inspected only after the complete decision is fixed.
    coverage, occurrence = _coverage(_copy(coverage_index) if coverage_index is not None else None, decision, selected)
    result = dict(schema="mdb-qwen-source-applicability-plan-v1", packet_id=packet.get("packet_id"), **body, decision_sha256=decision,
        finite_evidence_coverage=coverage, occurrence_lookup=occurrence, actual_outcome=dict(status="not-read-or-executed"),
        authority="source-only prospective plan; no execution or scientific-null/finite-accuracy verdict")
    return json.loads(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
