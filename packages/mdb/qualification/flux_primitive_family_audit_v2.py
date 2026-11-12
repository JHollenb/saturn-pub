"""Independent saved-array audit of a compiled nonlinear primitive family.

Original SDPA/SwiGLU/projection products are already paid. This reader joins
their actual operands and returns, reconstructs full merge and finite BF16
endpoints with NumPy, and compares complete own-state futures and RGB. It never
hydrates parameters or calls a model module. Numeric/rank misses are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from functools import cache
from pathlib import Path

import numpy as np

SCHEMA = "mdb-flux-primitive-family-saved-audit-v1"
RESULT = "primitive-family-audit-result"
MODEL_ID = "a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0"
WRITER_SHA256 = "fb4832ce7a1083e7f14881ff70bc253f2e5f5533a5c0f9b04193989ac66fe9f0"
CAMPAIGN_SHA256 = "632928f776cd10959d56bffc2fa6c54811af73dfb0430854cc571b4c51676531"
HELPERS = {
    "flux_parallel_family_audit.py": "e83bffcf2e12c8e3e238457d7acde01ff7bff5a94601381b54c5853fa5ca338c",
    "flux_binding_family_audit.py": "705fc21e5ab6d1fd012ecb0ed6157ef4195323643b7f66e1781bb67b1a05cadd",
}
CORE = {"R", "A", "M", "gate", "projection_return", "out", "image_in", "text_in", "image_out", "text_out"}
CONSUMER_PORTS = {"image_in", "text_in", "A", "M", "image_out", "text_out", "projection_return", "gate"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


@cache
def utilities():
    for name, digest in HELPERS.items():
        require(sha(Path(__file__).with_name(name)) == digest, "saved-only dependency differs: " + name)
    return runpy.run_path(str(Path(__file__).with_name("flux_parallel_family_audit.py")))


def expected_counts(size):
    return dict(functional_source_blocks=2*size, native_source_blocks=2*size,
        functional_primitive_calls=4*size, native_primitive_calls=4*size,
        functional_hybrid_projections=4*size, native_hybrid_projections=4*size,
        functional_transitions=606*size, native_target_transitions=606*size, native_blocks=542*size,
        functional_vae=6*size, native_vae=6*size, acts=10*size, forecast_acts=5*size, native_acts=5*size, rewinds=size)


def source_ports(family):
    require(family in {"sdpa", "swiglu"}, "unsupported primitive family")
    return CORE | ({"q_consumed", "k_consumed", "v_consumed"} if family == "sdpa" else {"mlp_input"})


def raw_hash(value):
    return utilities()["content"](utilities()["bf_bits"](value))


def compare_return_arrays(publication, forecast, native):
    """Complete logical-byte and numeric account, including projection nullspace."""
    result = publication.compare(forecast, native)
    left, right = (publication.array(record, raw=True) for record in (forecast, native))
    require(left.shape == right.shape and left.dtype == right.dtype == np.uint16,
        "qualified BF16 return-array ABI differs")
    result["raw_bit_mismatch_elements"] = int(np.count_nonzero(left != right))
    a, b = (np.ascontiguousarray(value).view(np.uint8).reshape(-1) for value in (left, right))
    bits = 0
    for start in range(0, a.size, 1 << 20):
        bits += int(np.unpackbits(np.bitwise_xor(a[start:start+(1 << 20)], b[start:start+(1 << 20)])).sum(dtype=np.uint64))
    result.update(raw_bit_mismatch_bits=bits, compared_elements=int(left.size), compared_bits=int(a.size*8))
    return result


def construction_return_account(publication, forecast, native, actions):
    """Compare every retained component return without gating on its error."""
    result = {}
    for action in actions:
        if action["role"] == "natural-reference":
            continue
        aid = action["id"]
        records = [lane["constructions"][aid]["ports"] for lane in (forecast, native)]
        require(all(set(record) == {"primitive_return", "projection_return"} for record in records),
            "complete component return retention required")
        result[aid] = {name: compare_return_arrays(publication, records[0][name], records[1][name])
            for name in ("primitive_return", "projection_return")}
    return result


def save_native_primitive_interaction(publication, native, output, context, baseline, candidates, both):
    """Retain widened native J over every primitive feature and token row."""
    require(len(candidates) == 2, "two-factor interaction requires two single corners")
    ids = [both, *candidates, baseline]
    records = {aid: native["constructions"][aid]["ports"]["primitive_return"] for aid in ids}
    value = publication.array(records[both]).astype(np.float64)
    for aid in candidates:
        operand = publication.array(records[aid])
        require(operand.shape == value.shape, "primitive interaction corner shape differs")
        value -= operand
    operand = publication.array(records[baseline])
    require(operand.shape == value.shape, "primitive interaction baseline shape differs")
    value += operand
    token = hashlib.sha256(context.encode()).hexdigest()[:16]
    relative = "primitive-interactions/" + token + "--native-J.npy"
    path = Path(output) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    record = dict(path=relative, sha256=sha(path), bytes=path.stat().st_size,
        content_sha256=hashlib.sha256(memoryview(value).cast("B")).hexdigest(), shape=list(value.shape),
        stride=[item//value.itemsize for item in value.strides], logical_dtype="torch.float64",
        stored_dtype="float64", encoding="native-numpy-v1", device_at_capture="cpu")
    total, maximum, nonzero = 0., 0., 0
    flat = value.reshape(-1)
    for start in range(0, flat.size, 1 << 20):
        chunk = flat[start:start+(1 << 20)]
        total += float(np.sum(chunk*chunk, dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(chunk))))
        nonzero += int(np.count_nonzero(chunk))
    return dict(array=record, rms=float(np.sqrt(total/value.size)), l2=float(np.sqrt(total)), max_abs=maximum,
        nonzero_elements=nonzero, elements=int(value.size), source_root=str(publication.root), operands=records,
        arithmetic=dict(dtype="NumPy float64 after widening each original BF16 return", order=ids,
            expression="((both - single0) - single1) + baseline"),
        scope="Complete native primitive feature return over full text+image geometry; no rows selected; finite four-corner interaction, not semantic attribution")


def primitive_account(q, p, source_records, meta, primitive, action, family):
    """Join source factor bytes to actual full-shape primitive receipts."""
    u = utilities()
    require(meta["construction"]["family"] == family and meta["construction"]["action"] == action,
        "primitive construction/action differs")
    receipt = meta["construction"]["primitive_receipt"]
    require(receipt["calls"] == 1, "actual primitive call census differs")
    sources = dict(Q=q, P=p)
    if family == "sdpa":
        require(set(receipt["consumed"]) == {"query", "key", "value"}
            and receipt["flags"] == dict(attn_mask=None, dropout_p=0., is_causal=False, scale=None, enable_gqa=False),
            "actual SDPA profile differs")
        joins = {}
        for field, factor, name in (("query", "route", "q_consumed"), ("key", "route", "k_consumed"), ("value", "value", "v_consumed")):
            source = action["bindings"][factor]
            declared = source_records[source]["descriptors"][name]
            require(receipt["consumed"][field] == declared == meta["descriptors"][name], "consumed SDPA source/layout differs: " + name)
            require(raw_hash(sources[source][name]) == declared["content_sha256"], "SDPA factor array differs")
            joins[name] = dict(source=source, descriptor=declared)
        b, h, tokens, d = receipt["consumed"]["query"]["shape"]
        require(list(primitive.shape) == [b, tokens, h*d], "primitive attention feature shape differs")
        raw = primitive.reshape(b, tokens, h, d).transpose(0, 2, 1, 3)
        require(raw_hash(raw) == receipt["returned"]["content_sha256"]
            and receipt["returned"]["shape"] == [b, h, tokens, d], "SDPA return/flatten provenance differs")
    else:
        operand = q["mlp_input"].copy()
        half = operand.shape[-1] // 2
        require(2*half == operand.shape[-1] and receipt["slices"] == dict(gate=[0, half], value=[half, 2*half]), "SwiGLU half interpretation differs")
        for factor, start in (("gate", 0), ("value", half)):
            operand[..., start:start+half] = sources[action["bindings"][factor]]["mlp_input"][..., start:start+half]
        consumed = receipt["consumed"]
        require(set(consumed) == {"mlp_input"} and consumed["mlp_input"] == meta["descriptors"]["mlp_input"], "actual SwiGLU operand differs")
        declared = source_records["Q"]["descriptors"]["mlp_input"]
        require(all(consumed["mlp_input"][key] == declared[key] for key in ("shape", "stride", "dtype", "device"))
            and consumed["mlp_input"]["content_sha256"] == raw_hash(operand), "SwiGLU source slices/layout differ")
        require(raw_hash(primitive) == receipt["returned"]["content_sha256"], "SwiGLU native module return differs")
        joins = dict(gate=dict(source=action["bindings"]["gate"], columns=[0, half]),
            value=dict(source=action["bindings"]["value"], columns=[half, 2*half]), consumed=consumed["mlp_input"])
    target = "A" if family == "sdpa" else "M"
    copy = None
    if action["role"] in {"baseline", "component-reference"}:
        copy = u["pair_account"](primitive, sources["Q" if action["role"] == "baseline" else "P"][target])
    return dict(actual_consumed_sources=joins, all_source_copy=copy, primitive_products="original same-profile kernels already paid; no CPU primitive rerun")


def endpoint_account(q, p, primitive, projection, action, family, ntext):
    """Independent literal full concat, BF16 multiply, then BF16 add."""
    u = utilities()
    if action["role"] == "natural-reference":
        value = {"P": p, "Q": q}[action["source"]]["image_out"]
        return value.copy(), None, None
    r, a, m = (q[name].copy() for name in ("R", "A", "M"))
    r[:, ntext:] = p["R"][:, ntext:]
    (a if family == "sdpa" else m)[:, ntext:] = primitive[:, ntext:]
    merged = np.concatenate((a, m), -1)
    write = u["bf"](q["gate"] * projection)
    out = u["bf"](r + write)
    return out[:, ntext:], merged, (write, out)


def axis_account(values, source, recipient):
    """Natural P=0/Q=1 signed coordinate; no scalar is a mechanics gate."""
    u = utilities()
    p, q = (np.asarray(values[name], dtype=np.float64) for name in (source, recipient))
    axis = q-p
    denominator = float(np.sum(axis*axis))
    rows = {}
    for aid, value in values.items():
        delta = np.asarray(value, dtype=np.float64)-p
        coordinate = float(np.sum(delta*axis)/denominator) if denominator else None
        residual = None if coordinate is None else delta-coordinate*axis
        rows[aid] = dict(p=coordinate, source_l2=float(np.linalg.norm(delta)), source_rms=u["metric"](value, p)["rms"],
            recipient_l2=float(np.linalg.norm(np.asarray(value, dtype=np.float64)-q)), recipient_rms=u["metric"](value, q)["rms"],
            orthogonal_l2=None if residual is None else float(np.linalg.norm(residual)),
            orthogonal_rms=None if residual is None else float(np.sqrt(np.mean(residual*residual))))
    return dict(status="applicable" if denominator else "not-applicable-zero-native-axis", source=source, recipient=recipient,
        axis_l2=float(np.sqrt(denominator)), axis_rms=float(np.sqrt(np.mean(axis*axis))), rows=rows)


def transaction_account(pub, summary, plan):
    from mdb.computation_protocol import STAGES
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    tx = verify_transaction(pub.root / "forecast-transaction", expected_seal_sha256=summary["forecast_seal"],
        expected_event_head=summary["transaction"]["event_head"])
    tree = read(pub.member("forecast-transaction/forecasts/all-numeric-futures.json"))
    forecast = load_forecast_tree(pub.root / "forecast-transaction", tree["predictions"])
    for name in ("native_targets_at_seal", "native_source_blocks_at_seal", "native_primitive_calls_at_seal", "native_decodes_at_seal"):
        require(forecast[name] == 0, "native work before global seal")
    for name, record in forecast["external_files"].items():
        require(pub.files.get(name) == record, "sealed external forecast changed")
    timeline = [json.loads(line) for line in pub.member("timeline.jsonl").read_text().splitlines()]
    seals = [index for index, row in enumerate(timeline) if row["kind"] == "all_forecasts_sealed"]
    require(len(seals) == 1 and timeline[seals[0]]["seal"] == summary["forecast_seal"], "timeline seal differs")
    require(all(index > seals[0] for index, row in enumerate(timeline) if row["kind"].startswith("native_")), "native chronology precedes global seal")
    program = read(pub.member("program.json"))
    journal = [json.loads(line) for line in pub.member("program-receipts.jsonl").read_text().splitlines()]
    require(program["plan_sha256"] == plan["plan_sha256"] and program["seal_sha256"] == summary["forecast_seal"]
        and program["commands"] == journal and [row["operation"] for row in journal] == list(STAGES)
        and [row["number"] for row in journal] == list(range(10))
        and all(row["plan_sha256"] == plan["plan_sha256"] for row in journal), "actual generated program differs")
    require(journal[4]["result"] == dict(seal_sha256=summary["forecast_seal"], native_calls_before_seal=0), "program numeric barrier differs")
    executions = {path.stem for path in (pub.root / "forecast-transaction/observations").glob("*.json")}
    require(executions == {"native-sources", "native-endpoints", "native-continuations", "native-decodes"}, "transaction native phase census differs")
    return forecast, tx, program


def validate_config(config):
    return utilities()["validate_config"](config)


def run(config):
    config = validate_config(config)
    with utilities()["model_free"]() as torch:
        return _run(config, torch)


def _run(config, torch):
    from PIL import Image
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe, digest

    from mdb.computation_protocol import evaluate_rule, verify_plan
    from mdb.equation_comparison import compare_equations
    from mdb.forecast_transaction import verify_transaction

    u = utilities()
    pub = u["Publication"](config["native_root"], config["native_manifest_sha256"], config["native_job_id"], config["native_request_sha256"])
    root = pub.root
    config_plan, request, summary = (read(pub.member(name)) for name in ("plan.json", "request.json", "summary.json"))
    require(u["request_identity"](request) == request["request_sha256"] == config["native_request_sha256"], "producer request differs")
    require({key: value for key, value in config_plan.items() if key not in {"job_id", "request_sha256", "output"}} == request["config"], "producer config differs")
    require(summary["schema"] == "mdb-flux-primitive-family-campaign-v1" and summary["job_id"] == config["native_job_id"]
        and summary["request_sha256"] == config["native_request_sha256"], "producer summary differs")
    require(request["source_hashes"]["flux_primitive_family_writer.py"] == WRITER_SHA256
        and request["source_hashes"]["flux_primitive_family_campaign.py"] == CAMPAIGN_SHA256, "producer source qualification differs")
    plan = verify_plan(config_plan["compiled_protocol"])
    family, actions = config_plan["family"], plan["actions"]
    expected_ports = source_ports(family)
    by_id = {row["id"]: row for row in actions}
    require(len(actions) == 6 and config_plan["arms"] == list(by_id) and summary["protocol_sha256"] == plan["plan_sha256"], "finite generated actions differ")
    references = {row["source"]: row["id"] for row in actions if row["role"] == "natural-reference"}
    baseline = next(row["id"] for row in actions if row["role"] == "baseline")
    both = next(row["id"] for row in actions if row["role"] == "component-reference")
    candidates = plan["candidate_menu"]
    require(set(references) == {"P", "Q"} and len(candidates) == 2, "two-factor reference menu differs")
    assay = read(pub.member("family-rule.json"))
    require(assay == config_plan["family_rule"] and sha(root / "family-rule.json") == config_plan["family_rule_file_sha256"]
        and assay["protocol_sha256"] == plan["plan_sha256"], "frozen assay differs")
    contexts = {row["id"]: row for row in config_plan["contexts"]}
    require(1 <= len(contexts) <= 4 and config_plan["contexts"] == assay["fresh_contexts"], "declared context subset differs")
    expected = expected_counts(len(contexts))
    contract = read(pub.member("native-contract.json"))
    require(summary["counts"] == contract["counts"] == expected and summary["observed"] == contract["observed"], "actual physical census differs")
    require(all(not values.get("module_entries", 0) for lane, values in contract["observed"].items() if lane.startswith("forecast")), "functional lane entered native modules")
    for lane, count in (("target-sources", 2), ("target-endpoints", 4 if family == "sdpa" else 0), ("target-continuations", 540)):
        require(contract["observed"].get(lane, {}).get("SDPA", 0) == count*len(contexts), "actual SDPA phase census differs")
    forecast, transaction, program = transaction_account(pub, summary, plan)
    identity = read(pub.member("source-identity-qualification.json"))
    require(identity["model_identity"] == MODEL_ID and identity["identity_override"] is False
        and identity["original_full_vae_identity"] is True and identity["original_transformer_parameter_signature"] is True
        and identity["transformer_config"]["_name_or_path"] == str(Path(config_plan["flux_parameter_root"]) / "transformer"), "original model profile qualification differs")
    output = Path(config["output"])
    require(not output.resolve().is_relative_to(root) and not root.is_relative_to(output.resolve()), "audit/source overlap")
    output.mkdir(parents=True, exist_ok=False)
    store = LocalStore(root / "state-store")

    def cut(record):
        result = store.load(record["cut"], device="cpu")
        require(result.fingerprint == record["cut"] and result.parent == record["parent"]
            and result.model_identity == record["model_identity"] == MODEL_ID
            and dict(result.execution_point) == record["execution_point"], "CAS/cut binding differs")
        return result

    def join(value, record):
        actual = describe(value)
        require(actual["sha256"] == record["content_sha256"] and actual["shape"] == record["shape"]
            and actual["stride"] == record["stride"] and actual["dtype"] == record["logical_dtype"], "full array/CAS join differs")

    def numpy_tensor(value):
        return value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()

    roots, provenance = (read(pub.member(name)) for name in ("source-roots.json", "source-provenance.json"))
    require(set(roots) == set(contexts) and provenance["source"] == config_plan["fresh_source"]
        and provenance["family_rule_sha256"] == config_plan["family_rule_file_sha256"], "source publication lineage differs")
    source = provenance["source"]
    original = u["Publication"](source["output_root"], source["files_manifest_sha256"], source["job_id"], source["request_sha256"])
    for name, record in provenance["consumed_members"].items():
        require(original.files.get(name) == record, "consumed prefix member lineage differs")
    prefix_request, prefix_summary = (read(original.member(name)) for name in ("request.json", "summary.json"))
    require(u["request_identity"](prefix_request) == source["request_sha256"], "prefix request differs")
    prefix_rows = prefix_request["config"]["contexts"]
    prefix_expected = dict(contexts=len(prefix_rows), prompt_encodes=2*len(prefix_rows), noise_draws=len({row["seed"] for row in prefix_rows}),
        native_transitions=20*len(prefix_rows), native_blocks=18*len(prefix_rows), selected_block_calls=0, decoder_calls=0, images=0,
        input_cuts=2*len(prefix_rows), pipeline_loads=1, encoder_cuda_placements=1, transformer_cuda_placements=1)
    require(prefix_summary["schema"] in {"mdb-parallel-relation-prefix-v1", "mdb-primitive-family-prefix-v1"}
        and prefix_summary["counts"] == prefix_expected, "native input-only prefix census differs")
    require(sha(original.member("family-rule.json")) == provenance["source_policy_sha256"] == prefix_request["config"]["family_rule_file_sha256"]
        and read(original.root / "family-rule.json") == prefix_request["config"]["family_rule"], "prefix own policy differs")
    script = prefix_request["worker_script"]
    expected_worker = {"flux_parallel_family_sources.py": "7b7ad97535a55d7efb10ac302e040813dc4ad0403981ec1e3a011b13a3a6005e",
        "flux_primitive_family_sources.py": request["source_hashes"]["flux_primitive_family_sources.py"]}
    require(script in expected_worker and prefix_request["source_hashes"][script] == expected_worker[script], "executed prefix source differs")
    prefix_tx = verify_transaction(original.root / "source-transaction", expected_seal_sha256=prefix_summary["source_policy_seal"])
    original_roots = read(original.member("source-roots.json"))
    require(roots == {cid: original_roots[cid] for cid in contexts}, "selected original input roots differ")
    original_store = LocalStore(original.root / "state-store")
    writers = {lane: read(pub.member(lane + "-source-writers.json")) for lane in ("forecast", "native")}
    writer_checks, writer_layouts, writer_live, source_joins, endpoints, construction_checks = {}, {}, {}, {}, {}, {}
    construction_returns, primitive_interactions = {}, {}
    for cid, source_root in roots.items():
        require(source_root["context"] == contexts[cid], "source prompt/seed metadata differs")
        inputs = {role: cut(source_root["inputs"][role]) for role in ("P", "Q")}
        for role, value in inputs.items():
            prior = original_store.load(value.fingerprint, device="cpu")
            require(value.manifest() == prior.manifest() and (value.payload["step"], value.payload["phase"], value.payload["cursor"]) == (0, "single", 4), "source input frontier differs")
        for name in ("latent", "img_ids", "txt_ids", "timesteps", "sigmas"):
            require(describe(inputs["P"].payload[name]) == describe(inputs["Q"].payload[name]), "paired noise/IDs/schedule differ")
        writer_checks[cid], writer_layouts[cid], writer_live[cid], source_joins[cid] = {}, {}, {}, {}
        for role in ("P", "Q"):
            f, n = (writers[lane][cid]["sources"][role] for lane in ("forecast", "native"))
            writer_live[cid][role] = u["descriptor_comparison"](f["descriptors"], n["descriptors"])
            writer_layouts[cid][role] = {lane: u["port_account"](pub, writers[lane][cid]["sources"][role], expected_ports) for lane in writers}
            writer_checks[cid][role] = dict(comparisons={name: pub.compare(f["ports"][name], n["ports"][name]) for name in expected_ports}, finite_closure={})
            source_joins[cid][role] = {}
            for lane in writers:
                saved = writers[lane][cid]["sources"][role]
                after, profile = cut(saved["state"]), saved["profile"]
                require(profile["model_identity"] == MODEL_ID and profile["index"] == 4 and profile["point"] == dict(inputs[role].execution_point)
                    and profile["execution"] == dict(inputs[role].execution), "writer source profile differs")
                for name, value in dict(image=inputs[role].payload["image"], text=inputs[role].payload["text"],
                        single_mod=inputs[role].payload["context"]["single_mod"], rope_cos=inputs[role].payload["context"]["rope"][0], rope_sin=inputs[role].payload["context"]["rope"][1]).items():
                    descriptor = describe(value)
                    require(all(descriptor[key] == profile["interfaces"][name][target] for key, target in
                        (("sha256", "content_sha256"), ("shape", "shape"), ("stride", "stride"), ("dtype", "dtype"))), "source live operand differs")
                require((after.payload["step"], after.payload["phase"], after.payload["cursor"]) == (0, "single", 5), "source writer exit clock differs")
                for stream in ("image", "text"):
                    join(inputs[role].payload[stream], saved["ports"][stream + "_in"])
                    join(after.payload[stream], saved["ports"][stream + "_out"])
                values = {name: pub.array(value) for name, value in saved["ports"].items()}
                require(raw_hash(np.concatenate((values["text_in"], values["image_in"]), 1)) == saved["ports"]["R"]["content_sha256"]
                    and raw_hash(np.concatenate((values["text_out"], values["image_out"]), 1)) == saved["ports"]["out"]["content_sha256"], "source full stream concat differs")
                width = values["R"].shape[-1]
                mod = inputs[role].payload["context"]["single_mod"].reshape(1, 1, 3*width)
                require(describe(mod[..., 2*width:])["sha256"] == saved["ports"]["gate"]["content_sha256"], "source gate/modulation differs")
                writer_checks[cid][role]["finite_closure"][lane] = u["pair_account"](u["bf"](values["R"] + u["bf"](values["gate"] * values["projection_return"])), values["out"])
                require(describe({key: value for key, value in inputs[role].payload.items() if key not in {"image", "text", "cursor"}})
                    == describe({key: value for key, value in after.payload.items() if key not in {"image", "text", "cursor"}}), "source persistent closure changed")
                source_joins[cid][role][lane] = dict(input_cut=inputs[role].fingerprint, output_cut=after.fingerprint)
                del values, after
        endpoints[cid] = {aid: pub.compare(writers["forecast"][cid]["endpoints"][aid], writers["native"][cid]["endpoints"][aid]) for aid in by_id}
        construction_checks[cid] = {}
        for lane in writers:
            saved = writers[lane][cid]
            source_records = saved["sources"]
            q, p = ({name: pub.array(value) for name, value in source_records[role]["ports"].items()} for role in ("Q", "P"))
            require(raw_hash(q["gate"]) == raw_hash(p["gate"]), "same-clock P/Q gates differ")
            construction_checks[cid][lane] = {}
            for action in actions:
                aid, meta = action["id"], saved["constructions"][action["id"]]
                require(meta["construction"]["action"] == action, "stored action differs")
                primitive = projection = None
                if action["role"] != "natural-reference":
                    require(set(meta["ports"]) == {"primitive_return", "projection_return"}, "construction retention differs")
                    primitive, projection = (pub.array(meta["ports"][name]) for name in ("primitive_return", "projection_return"))
                    primitive_proof = primitive_account(q, p, source_records, meta, primitive, action, family)
                    require(all(meta["descriptors"][name]["content_sha256"] == meta["ports"][name]["content_sha256"] for name in meta["ports"]), "construction arrays/descriptors differ")
                else:
                    require(meta["ports"] == {} and meta["descriptors"] == {}, "natural reference must not run a component projection")
                    primitive_proof = dict(reference=action["source"])
                image, merge, intermediates = endpoint_account(q, p, primitive, projection, action, family, q["text_in"].shape[1])
                if merge is not None:
                    require(raw_hash(merge) == meta["descriptors"]["merge_input"]["content_sha256"], "full mixed projection operand differs")
                    intermediate_checks = {name: dict(hash_equal=raw_hash(value) == meta["descriptors"][name]["content_sha256"])
                        for name, value in zip(("write", "out"), intermediates, strict=True)}
                else:
                    intermediate_checks = {}
                construction_checks[cid][lane][aid] = dict(primitive=primitive_proof,
                    endpoint=u["pair_account"](image, pub.array(saved["endpoints"][aid])), intermediates=intermediate_checks)
            component_values = {aid: pub.array(saved["endpoints"][aid]).astype(np.float64) for aid in [baseline, *candidates, both]}
            mixed = component_values[both]-component_values[candidates[0]]-component_values[candidates[1]]+component_values[baseline]
            construction_checks[cid][lane]["mixed_remainder"] = u["metric"](mixed, np.zeros_like(mixed))
            del q, p, component_values, mixed
        construction_returns[cid] = construction_return_account(pub, writers["forecast"][cid], writers["native"][cid], actions)
        primitive_interactions[cid] = save_native_primitive_interaction(pub, writers["native"][cid], output,
            cid, baseline, candidates, both)

    index, consumers = (read(pub.member(name)) for name in ("state-index.json", "consumer-ports.json"))
    planned = {cid + "/" + aid for cid in roots for aid in by_id}
    require(set(index) == set(forecast["paths"]) == planned and set(consumers) == {"forecast", "native"}
        and all(set(rows) == planned for rows in consumers.values()), "case/consumer census differs")
    cases, images, act_checks, equation_reports, native_values, consumer_layouts, consumer_live = {}, {}, {}, {}, {}, {}, {}
    boundary_counts = dict(paired_rows=0, tensors=0, scalar_fields=0, hash_mismatches=0, layout_mismatches=0, value_mismatches=0, scalar_mismatches=0)

    def act(record, operand):
        receipt, after = record["receipt"], cut(record["after"])
        require(digest({key: value for key, value in receipt.items() if key != "fingerprint"}) == receipt["fingerprint"], "Act receipt changed")
        require(receipt["result"] == after.fingerprint and receipt["parent"] == after.parent
            and receipt["act"]["reads"] == receipt["act"]["writes"] == ["image"] and not receipt["act"]["invalidates"], "Act address/lineage differs")
        before = store.load(receipt["parent"], device="cpu")
        common = cut(record["common"])
        require(before.parent == common.fingerprint and describe(before.payload) == describe(common.payload), "fork/common closure differs")
        require(describe({key: value for key, value in before.payload.items() if key != "image"})
            == describe({key: value for key, value in after.payload.items() if key != "image"}), "image Act changed another slot")
        require(receipt["writes"] == describe({"image": after.payload["image"]})
            and receipt["act"]["preconditions"] == {"image": digest(describe(before.payload["image"]))}, "Act writes/preconditions differ")
        raw = pub.array(operand, raw=True)
        require(raw.flags.c_contiguous, "endpoint NPY reload is not contiguous")
        parameters = u["patch_parameter_account"](receipt, describe(before.payload["image"]), describe(after.payload["image"]), operand,
            [stride//raw.itemsize for stride in raw.strides])
        join(after.payload["image"], operand)
        return after, dict(parent=before.fingerprint, common=common.fingerprint, after=after.fingerprint, parameters=parameters)

    def row_join(payload, fields):
        if isinstance(payload, torch.Tensor):
            actual = describe(payload)
            require(all(actual[name] == fields["tensor"][name] for name in ("sha256", "shape", "dtype", "stride")), "boundary/final CAS differs")
            value = numpy_tensor(payload).astype(np.float64)
            require(fields["elements"] == value.size and fields["witness_values"] == value.reshape(-1)[fields["witness_indices"]].tolist(), "boundary/CAS samples differ")
        elif isinstance(payload, dict):
            require(set(payload) == set(fields), "boundary/CAS slot census differs")
            for name in payload:
                row_join(payload[name], fields[name])
        elif isinstance(payload, (tuple, list)):
            require(len(payload) == len(fields), "boundary sequence differs")
            for left, right in zip(payload, fields, strict=True):
                row_join(left, right)
        else:
            require(type(payload) is type(fields) and payload == fields, "boundary scalar differs")

    for key, case in index.items():
        cid, aid = case["context"], case["arm"]
        action = by_id[aid]
        require(key == cid + "/" + aid and case["action"] == action and case["forecast_final"] == forecast["paths"][key]["forecast_final"], "sealed case identity differs")
        branch = root / case["root"]
        rows = {lane: read(pub.member(case["root"] + "/" + lane + "-rows.json")) for lane in writers}
        require(all(len(value) == 102 and [row["number"] for row in value] == list(range(102)) for value in rows.values()), "102-boundary census differs")
        misses = []
        for number, (f, n) in enumerate(zip(rows["forecast"], rows["native"], strict=True)):
            if number:
                require(f["event"] == n["event"] == {name: rows["native"][number-1]["fields"][name] for name in ("step", "phase", "cursor")}, "boundary event clock differs")
            account = u["boundary_account"](f["fields"], n["fields"])
            boundary_counts["paired_rows"] += 1
            for name in ("tensors", "scalar_fields"):
                boundary_counts[name] += account[name]
            for name in ("hash_mismatches", "layout_mismatches", "value_mismatches", "scalar_mismatches"):
                boundary_counts[name] += len(account[name])
            if any(account[name] for name in ("hash_mismatches", "layout_mismatches", "value_mismatches", "scalar_mismatches")):
                misses.append(dict(number=number, **account))
        consumer_layouts[key] = {lane: u["port_account"](pub, consumers[lane][key], CONSUMER_PORTS) for lane in writers}
        consumer_live[key] = u["descriptor_comparison"](consumers["forecast"][key]["descriptors"], consumers["native"][key]["descriptors"])
        consumer_comparisons = {name: pub.compare(consumers["forecast"][key]["ports"][name], consumers["native"][key]["ports"][name]) for name in CONSUMER_PORTS}
        decoders = {lane: read(pub.member(case["root"] + "/" + filename)) for lane, filename in
            (("forecast", "decoder-forecast.json"), ("native", "decoder-native-reference.json"))}
        decoder_comparisons = {name: pub.compare(decoders["forecast"][name], decoders["native"][name], base_a=branch, base_b=branch)
            for name in ("decoded", "pixels", "decoder_input")}
        lane_values, act_checks[key] = {}, {}
        for lane in writers:
            entry = read(pub.member(case["root"] + "/" + lane + "-entry-act.json"))
            role = "P" if action["role"] == "natural-reference" and action["source"] == "P" else "Q"
            source_state = cut(writers[lane][cid]["sources"][role]["state"])
            if role == "P":
                require(entry["reference"] == "natural-P" and entry["receipt"] is None
                    and entry["source"] == writers[lane][cid]["sources"]["P"]["state"], "natural P reference differs")
                start = cut(entry["after"])
                require(describe(start.payload) == describe(source_state.payload), "natural P full-state closure differs")
                act_checks[key][lane] = dict(reference="natural-P", after=start.fingerprint, source=source_state.fingerprint)
            else:
                start, act_checks[key][lane] = act(entry, writers[lane][cid]["endpoints"][aid])
                common = cut(entry["common"])
                require(describe(common.payload) == describe(source_state.payload), "Q common/source closure differs")
                if action["role"] == "natural-reference":
                    require(describe(start.payload) == describe(common.payload), "natural Q full-state sham differs")
            row_join(start.payload, rows[lane][0]["fields"])
            for stream in ("image", "text"):
                join(start.payload[stream], consumers[lane][key]["ports"][stream + "_in"])
                port, tensor = consumers[lane][key]["ports"][stream + "_out"], rows[lane][1]["fields"][stream]["tensor"]
                require(port["content_sha256"] == tensor["sha256"] and port["shape"] == tensor["shape"]
                    and port["logical_dtype"] == tensor["dtype"] and port["stride"] == tensor["stride"], "first consumer output/boundary differs")
            final = cut(case[lane + "_final"])
            require((final.payload["step"], final.payload["phase"], final.payload["cursor"]) == (4, "project", 0), "final horizon differs")
            row_join(final.payload, rows[lane][-1]["fields"])
            lane_values[lane] = {name: pub.array(decoders[lane][name], branch) for name in ("decoded", "pixels")}
            lane_values[lane]["latent"] = numpy_tensor(final.payload["latent"])
            png_record = decoders[lane]["png"]
            png = pub.member((branch / png_record["path"]).relative_to(root).as_posix())
            require(sha(png) == png_record["sha256"], "PNG binding differs")
            with Image.open(png) as image:
                require(image.mode == "RGB" and np.array_equal(np.asarray(image), lane_values[lane]["pixels"]), "PNG/full RGB differ")
            target = output / "images" / (key.replace("/", "--") + "--" + lane + ".png")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(png.read_bytes())
            images[key + "/" + lane] = dict(path=str(target.relative_to(output)), sha256=sha(target), bytes=target.stat().st_size)
        decoder_comparisons["latent"] = u["pair_account"](lane_values["forecast"]["latent"], lane_values["native"]["latent"])
        native_values[key] = lane_values["native"]
        cases[key] = dict(decoder=decoder_comparisons, consumer_ports=consumer_comparisons, entry_endpoint=endpoints[cid][aid],
            boundary_misses=misses, png_byte_exact=decoders["forecast"]["png"]["sha256"] == decoders["native"]["png"]["sha256"])
        equation_reports[key] = dict(writer_exit=compare_equations(
            {"image.exit": pub.array(writers["forecast"][cid]["endpoints"][aid])},
            {"image.exit": pub.array(writers["native"][cid]["endpoints"][aid])}, max_elements=4_000_000),
            continuation=compare_equations(lane_values["forecast"], lane_values["native"], max_elements=4_000_000))
    observations, measurement_rows = {}, []
    for cid in roots:
        observations[cid] = {}
        for field in ("latent", "decoded", "pixels"):
            values = {aid: native_values[cid + "/" + aid][field] for aid in by_id}
            account = axis_account(values, references["P"], references["Q"])
            account["component_axis_diagnostic"] = axis_account(values, both, baseline)
            mixed = sum(sign*np.asarray(values[aid], dtype=np.float64) for sign, aid in ((1, both), (-1, candidates[0]), (-1, candidates[1]), (1, baseline)))
            account["mixed_remainder"] = u["metric"](mixed, np.zeros_like(mixed))
            account["candidate_signed_margin_first_minus_second"] = None if account["status"] != "applicable" else account["rows"][candidates[0]]["p"]-account["rows"][candidates[1]]["p"]
            observations[cid][field] = account
        measurement_rows.extend(dict(context=cid, action=aid, value=observations[cid]["latent"]["rows"][aid]["p"]) for aid in by_id)
    measurements = dict(plan_sha256=plan["plan_sha256"], model_identity=MODEL_ID, metric=plan["specification"]["metric"],
        contexts=config_plan["contexts"],
        provenance=dict(producer_request=config["native_request_sha256"], producer_job=config["native_job_id"],
            seal=summary["forecast_seal"], audit_input_manifest=config["native_manifest_sha256"]), rows=measurement_rows)
    evaluated = None
    if assay["phase"] == "fresh":
        evaluated = evaluate_rule(plan, assay["effect_rule"], measurements)
    misses = [dict(case=key, kind=kind, field=name) for key, case in cases.items()
        for kind in ("decoder", "consumer_ports") for name, value in case[kind].items() if not value["raw_bit_exact"]]
    return_misses = [dict(context=cid, action=aid, port=port,
        raw_bit_mismatch_elements=value["raw_bit_mismatch_elements"], raw_bit_mismatch_bits=value["raw_bit_mismatch_bits"])
        for cid, rows in construction_returns.items() for aid, ports in rows.items()
        for port, value in ports.items() if not value["raw_bit_exact"]]
    result = dict(schema=SCHEMA, native_job_id=config["native_job_id"], native_request_sha256=config["native_request_sha256"],
        native_manifest_sha256=config["native_manifest_sha256"], source_worker_sha256=CAMPAIGN_SHA256, family=family,
        protocol_sha256=plan["plan_sha256"], contexts=list(roots), arms=list(by_id), actions=actions,
        cases=cases, writer_checks=writer_checks, writer_layouts=writer_layouts, writer_live_descriptors=writer_live,
        source_joins=source_joins, endpoint_checks=endpoints, construction_checks=construction_checks,
        construction_return_comparisons=construction_returns, construction_return_bit_misses=return_misses,
        native_primitive_interactions=primitive_interactions,
        act_checks=act_checks, consumer_layouts=consumer_layouts, consumer_live_descriptors=consumer_live,
        boundary_counts=boundary_counts, observations=observations, images=images, equation_reports=equation_reports,
        transaction=transaction, program=program, source_transaction=prefix_tx, source_provenance=provenance,
        native_counts=expected, external_sealed_files=len(forecast["external_files"]), frozen_assay=assay, frozen_effect_evaluation=evaluated,
        published_files_verified=len(pub.files), prefix_files_verified=len(original.files), unique_arrays_read=len(pub.arrays),
        repeated_array_elements_read=pub.elements, numerical_bit_misses=misses,
        writer_bit_misses=[dict(context=cid, source=role, port=name) for cid, roles in writer_checks.items()
            for role, checks in roles.items() for name, value in checks["comparisons"].items() if not value["raw_bit_exact"]],
        endpoint_bit_misses=[dict(context=cid, action=aid) for cid, rows in endpoints.items() for aid, value in rows.items() if not value["raw_bit_exact"]],
        mechanics_status="verified", instrument_status="independent saved-array/CAS/receipt reader", terminal_status="unassessed",
        model_forwards=0, cuda_initialized=False, original_parameters_exported=False,
        limitations=["Source and exact finite forecasts use original parameters/shared kernels; the reader reruns no learned products or SDPA/SwiGLU targets.",
            "Natural P/Q define the primary signed axis; both-source primitive at held R_P/Q closure is a separate component reference.",
            "Full retained arrays and all live hashes/layouts are distinguished; hash-only formation ports cannot localize numeric misses.",
            "Boundary rows cover complete hashes/layouts/statistics and up to16 witness values, not full saved arrays at every transition.",
            "The supplied graph is hand-specified; factor algebra does not establish semantic meaning, quality, or cross-family metric comparability.",
            "Zero native axes remain explicitly inapplicable; numeric, signed-zero, rank and orthogonal discrepancies are separate observations."])
    save(output / "audit.json", result)
    save(output / "observations.json", observations)
    save(output / "measurements.json", measurements)
    for name in ("parameter-artifacts.json", "retrieve_weights.py"):
        (output / name).write_bytes(pub.member(name).read_bytes())
    return result


audit = run


def worker(config_path):
    config = read(config_path)
    outer = Path(os.environ.get("SATURN_DEBUG_REQUEST", "saturn_debug_request.json"))
    if outer.exists():
        request = read(outer)
        require(utilities()["request_identity"](request) == request["request_sha256"] and request["config"] == config, "audit request/config differs")
        for name, expected in request["source_hashes"].items():
            require(sha(Path(config_path).parent / name) == expected, "audit payload source differs")
    config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / RESULT)
    result = run(config)
    from mdb.job_support import request_sha256

    publication = dict(output_root=config["output"], audit_sha256=sha(Path(config["output"]) / "audit.json"),
        source_job=config["native_job_id"], config_sha256=request_sha256(read(config_path)))
    print("MDB_FAMILY_AUDIT_READY " + json.dumps(publication, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
