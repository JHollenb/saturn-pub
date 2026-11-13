"""Saved-only row-sign component accounts; support is diagnostic, not a gate.

No producer, generative helper, model, tokenizer or learned operation runs.
Whole fields, fixed windows, finite premises and source replay overlap and are
reported separately. Native component calls are not Qwen Session Acts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "b3d4ae8b0cbddf3c0477ace8644b19e25f42a5f820f2def9b06981da7cfd99d4"
ROW_HELPER_SHA256 = "940c3cfaafeaeb3f0df5746b92b6a8b0878255769c4f230b2a53bedfc0110c1c"
PROTOCOL_SHA256 = "d942001a79784ebb1bc5d5ed3f3946eaa03b4629f2fe13c7493afbaaf85b51e9"
PINS = {
    "audit_qwen_sign_shortcut.py": "f1e3d115e76215eebc920f0f62251bc5a9d7f9f0e66ad7fa3403090206f37682",
    "audit_flux_raw_to_image.py": "f8c1a39f9b86b9002e079db084887aece105304f58e28e1ae5f4fa146a0ae4bf",
    "audit_flux_functional_qwen_prefix.py": "2331126c9b3d918b1ac5376a6a13250c14806bdbcf151277f59320440e191c7a",
    "audit_flux_functional_pass.py": "43e163b7543ec225da007b6ecc87418c7c0654f464c36c8e7f1e5234642aeaa8",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
HELPERS = tuple(PINS)
PORTS = ("head", "R", "M", "Hnext")
ARMS = ("all-negative", "prefix-negative")
PATH_KEYS = ("qwen_config", "qwen_parameter_manifest", "parent_parameter_manifest", "qwen_source_binding", "retrieval_script",
    "selection_protocol", "parent_files", "parent_request", "parent_source", "parent_runtime", "parent_seal")
NATIVE_SHA256 = "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
SDPA_SHA256 = "b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46"
FRONTIER = dict(kind="architectural", origin="inherited-functional-state", scope="conditional block18 hidden-state interface",
    raw_input_prediction=False, genuine_record_prefix_capture=False)
CHANNELS = {"shortcut-minus-full-functional": ("shortcut", "functional"),
            "full-functional-minus-native": ("functional", "native"), "shortcut-minus-native": ("shortcut", "native")}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


@lru_cache(maxsize=1)
def base():
    for name, digest in PINS.items():
        require(sha(HERE / name) == digest, "frozen saved helper differs: " + name)
    return runpy.run_path(str(HERE / "audit_qwen_sign_shortcut.py"))


def regional(left, right, split):
    require(left.shape == right.shape and left.ndim in (3, 4) and left.shape[0] == 1 and 0 < split < left.shape[1], "canonical query geometry differs")
    def metric(a, b):
        value = base()["measurement"](a, b)
        value.update(orientation="supplied left-minus-right", denominator="RMS of supplied right operand",
                     operand_keys="Inherited predicted/native metric keys denote supplied left/right, respectively.")
        return value
    return dict(whole=metric(left, right),
        windows={"prefix": metric(left[:, :split], right[:, :split]), "suffix": metric(left[:, split:], right[:, split:])},
        bounds=dict(prefix=[0, split], suffix=[split, left.shape[1]]),
        scope="Canonical B,T,... coordinates; whole is primary and two windows partition it, not additional independent evidence.")


def descriptor(edited, positive):
    import numpy as np

    delta = np.asarray(edited, np.float64) - np.asarray(positive, np.float64)
    return dict(shape=list(delta.shape), elements=int(delta.size), dtype="float64", orientation="edited-minus-own-positive",
        sha256=hashlib.sha256(np.asarray(delta, dtype="<f8").tobytes(order="C")).hexdigest())


def support_account(signs, permitted, retained, split):
    import numpy as np

    require(signs.ndim == 1 and np.isin(signs, (-1, 1)).all() and permitted.dtype == np.dtype("bool")
        and permitted.shape == (signs.size, signs.size) and retained.dtype == np.dtype("bool") and retained.shape == signs.shape, "row-sign/permitted support schema differs")
    closed = permitted.any(-1) & ~(permitted & (signs[:, None] != signs[None, :])).any(-1)
    require(np.array_equal(closed, retained), "saved structural support diagnostic differs")
    return dict(rows=int(signs.size), closed_rows=int(closed.sum()), empty_rows=int((~permitted.any(-1)).sum()),
        windows={name: dict(rows=int(mask.size), closed_rows=int(mask.sum()), classification="closed" if mask.all() else "mixed-or-empty" if not mask.any() else "heterogeneous")
                 for name, mask in (("prefix", closed[:split]), ("suffix", closed[split:]))},
        scope="Derived permitted-key sign relation only; all rows are predicted and retained regardless of support or numerical error.")


def premise(positive, edited, signs, split):
    import numpy as np

    expected = positive * signs.reshape(1, -1, *([1] * (positive.ndim - 2)))
    result = regional(edited, expected, split)
    result["signed_zero"] = {name: dict(shared=int(np.count_nonzero((p == 0) & (e == 0))),
        defects=int(np.count_nonzero((p == 0) & (e == 0) & (np.signbit(p) != np.signbit(e)))))
        for name, p, e in (("whole", edited, expected), ("prefix", edited[:, :split], expected[:, :split]), ("suffix", edited[:, split:], expected[:, split:]))}
    result["orientation"] = "edited minus row-signed own positive; finite numerical/bit/signed-zero premise, never an applicability or accuracy gate"
    return result


def array_accounts(positive, arms, *, split):
    """Only saved tensors and pointwise reductions; no attention/projection."""
    import numpy as np

    from mdb_lanes.equation_comparison import compare_equations

    require(set(positive) == {"functional", "native"} and set(arms) == set(ARMS), "two own baselines/two arms required")
    reports, equations = {}, {}
    for arm, row in arms.items():
        signs, permitted = row["signs"], row["permitted"]
        expected_signs = -np.ones(signs.size, dtype=np.float32)
        if arm == "prefix-negative":
            expected_signs[split:] = 1
        require(np.array_equal(signs, expected_signs), "fixed row-sign intervention differs")
        support = support_account(signs, permitted, row["support_closed"], split)
        require(np.array_equal(permitted, np.tri(signs.size, dtype=bool)), "declared all-attended causal support differs")
        lanes = row["lanes"]
        require(set(lanes) == {"shortcut", "functional", "native"}, "three comparison lanes required")
        for lane, bank in lanes.items():
            require(set(PORTS).issubset(bank), "complete head/R/M/H field coverage differs")
            if lane != "native":
                require(set(row["response_hashes"][lane]) == set(PORTS) and all(row["response_hashes"][lane][name] == descriptor(bank[name], positive["functional"][name]) for name in PORTS), "pretarget signed response hash differs")
        channels, equations[arm] = {}, {}
        for channel, (left, right) in CHANNELS.items():
            lbase, rbase = positive["native" if left == "native" else "functional"], positive["native" if right == "native" else "functional"]
            fields = {name: dict(endpoint=regional(lanes[left][name], lanes[right][name], split),
                response_error=regional(lanes[left][name].astype(np.float64) - lbase[name], lanes[right][name].astype(np.float64) - rbase[name], split),
                left_effect=regional(lanes[left][name], lbase[name], split), right_effect=regional(lanes[right][name], rbase[name], split)) for name in PORTS}
            channels[channel] = dict(fields=fields, first_whole_bit_difference=next((name for name in PORTS if not fields[name]["endpoint"]["whole"]["bitwise_exact"]), None))
            equations[arm][channel] = compare_equations({name: lanes[left][name] for name in PORTS}, {name: lanes[right][name] for name in PORTS},
                predicted_baseline={name: lbase[name] for name in PORTS}, baseline={name: rbase[name] for name in PORTS}, max_elements=sum(lanes[left][name].size for name in PORTS))
        reports[arm] = dict(channels=channels, support=support,
            premises={lane: {name: premise(positive[lane][name], lanes[lane][name], signs, split) for name in ("R", "gate", "up")} for lane in ("functional", "native")},
            coverage=dict(primary_endpoint_fields=12, primary_response_fields=12, overlapping_endpoint_windows=24, overlapping_response_windows=24))
    baseline = {name: regional(positive["functional"][name], positive["native"][name], split) for name in PORTS}
    conditional = {}
    for name in ("head", "R"):
        field = reports["prefix-negative"]["channels"]["shortcut-minus-native"]["fields"][name]
        cheap, actual, error = (field[key]["windows"]["suffix"] for key in ("left_effect", "right_effect", "response_error"))
        conditions = dict(cheap_response_zero=cheap["changed_values"] == 0, baseline_matches_numeric=baseline[name]["windows"]["suffix"]["changed_values"] == 0,
                          actual_native_response_nonzero=actual["changed_values"] > 0)
        conditional[name] = dict(conditions=conditions, implied_response_relative_error=1. if all(conditions.values()) else None,
            measured_response_relative_error=error["relative_rms"], scope="Conditional suffix bookkeeping observation, never a success/miss gate or semantic inference.")
    return dict(arms=reports, source_baseline_errors=baseline, conditional_positive_suffix_response=conditional,
        residual_directions=dict(scalar_accounts="left-minus-right in the named channel", MDB_equations="observed-minus-predicted; right-minus-left"),
        scope="New same-H18 native baseline is separate from archived functional positive. Whole/window and premise/source accounts overlap. Numerical misses never reject or repair rows."), equations


def parameter_accounts(full_manifest, selected_manifest, config, before, after, parameter_root, *, edit_index):
    """Restore only the exact original block subset; no exported weight bank."""
    old = base()
    _, q, streamer = old["helpers"]()
    roles = q["parameter_roles"](full_manifest, config, edit_index + 1)
    original = {row["source"]["tensor"]: row for row in full_manifest["artifacts"] if row["source"]["tensor"].startswith(f"model.layers.{edit_index}.")}
    selected = {row["source"]["tensor"]: row for row in selected_manifest["artifacts"]}
    require(len(original) == len(selected_manifest["artifacts"]) == 11 and selected == original, "selected recipes are not the complete exact original block subset")
    require(set(before) == set(after) == set(original) and before == after, "live component parameter coverage/immutability differs")
    expected = {}
    for name, row in original.items():
        source = row["source"]
        require(before[name] == dict(shape=source["shape"], dtype="torch.bfloat16", sha256=source["payload_sha256"]), "live original component parameter identity differs")
        expected[name] = source["payload_sha256"]
    restored = streamer["restore_parameters"](selected_manifest, parameter_root, expected, expected_module_sha256=old["RETRIEVER_SHA256"])
    return dict(selected=restored, complete_source_recipe_count=roles["counts"]["all"], selected_original_names=sorted(selected),
        scope="All11 original block weights, full source-file/header/payload/NPY archive hashes; no model construction, learned call or persistent parameter bank.")


def catalog_accounts(output, report, equations, contexts):
    from mdb_lanes.computation_atlas import ComputationAtlas
    from mdb_lanes.equation_comparison import EquationCatalog

    catalog = EquationCatalog(output / "catalog/equations")
    records, effects = [], {}
    for arm, context in contexts.items():
        record = catalog.record(law_id="law.qwen.row-sign-conditional-bf16.v1", route_id="route.qwen.block18.ownpositive-RGU-to-signed-tail.v1",
            alias="cmp.qwen.row-sign." + arm, binding={key: context[key] for key in ("model_identity", "execution", "input_identity", "frontier", "horizon", "source_identity")},
            conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
            readers=["qwen.block18." + name for name in PORTS], report=equations[arm]["shortcut-minus-native"],
            evidence=dict(audit_sha256=sha(output / "audit.json"), status="conditional observation; no validated rule or terminal certification",
                scope="Support is diagnostic; mixed rows, nonzero errors and finite oddness defects retained. No symbolic BF16 identity or independent kernel."))
        require(len(catalog.find(applicable_to=context)) == 1 and not catalog.find(applicable_to={**context, "input_identity": "0" * 64}), "conditional catalog lookup differs")
        records.append(dict(arm=arm, fingerprint=record["fingerprint"], context=context))
        fields = report["arms"][arm]["channels"]["shortcut-minus-native"]["fields"]
        effects[record["fingerprint"]] = [dict(consumer_id="qwen.block18." + name, contrast_id=contrast,
            value=fields[name][term]["whole"]["signed_mean"], units="represented coordinate units",
            provenance=[dict(path="audit.json", sha256=sha(output / "audit.json"))]) for name in PORTS for contrast, term in
            (("native-edited-minus-native-baseline", "right_effect"), ("shortcut-minus-native-prediction-error", "endpoint"),
             ("shortcut-response-minus-native-response-error", "response_error"))]
    save(output / "catalog/provenance.json", dict(records=records, rule_status="observation; no validated rule", actual_CLI_calls=0, independent_occurrences=False))
    atlas = ComputationAtlas(output / "atlas")
    observations = atlas.ingest(output / "catalog/equations", [row["fingerprint"] for row in records],
        provenance=dict(run_id=report["job_id"], artifacts=[dict(path="audit.json", sha256=sha(output / "audit.json"))]), signed_effects=effects)
    save(output / "atlas/provenance.json", dict(observations=[row["fingerprint"] for row in observations],
        scope="Actual native edit responses and two prediction-error effects have distinct contrast IDs. Two source-bound observations, no family induction or validated rule."))


def catalog_contexts(plan, source, sources, protocol, report):
    """One context builder shared by production and installed-schema checks."""
    operand = report["census"]["actual_SDPA_receipts"][0]["QKV"][0]
    return {arm: dict(model_identity=plan["qwen_parameter_manifest_sha256"], source_identity=source["source_hashes"],
        execution=dict(backend=operand["device"].split(":")[0], operand_device=operand["device"], learned_dtype=operand["dtype"],
            runtime_identity=base()["digest"](sources["runtime"]), primitive_policy=sources["runtime"]["flags"], attention_dispatch=plan["attention_dispatch"]),
        input_identity=protocol["source_arrays"]["h"]["content_sha256"], frontier="inherited-own-functional-positive-H18-and-ports",
        horizon=plan["horizon"], arm=arm, fixed_split=plan["fixed_split"], support=report["arms"][arm]["support"]["windows"]) for arm in ARMS}


def wrapper(root, expected_manifest, expected_request):
    require(sha(root / "files.json") == expected_manifest, "raw wrapper checksum differs")
    value = read(root / "files.json")
    require(value["schema"] == "mdb-image-tail-files-v1" and value["request_sha256"] == expected_request, "raw wrapper/request differs")
    require(not any(path.is_symlink() for path in root.rglob("*")) and
        {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != root / "files.json"} == set(value["files"]), "complete raw file closure differs")
    for name, row in value["files"].items():
        path = base()["member"](root, name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "retained member checksum differs: " + name)
    require(value["expanded_bytes"] == sum(row["bytes"] for row in value["files"].values()), "raw byte census differs")
    return value


def inherited_accounts(parent_root, protocol):
    """Authenticate only the thirteen declared positive/source tensor members."""
    import numpy as np

    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    b = base()
    parent_root = Path(parent_root)
    require(sha(parent_root / "files.json") == protocol["parent_raw_manifest_sha256"], "accepted parent wrapper differs")
    manifest = read(parent_root / "files.json")
    require(manifest["job_id"] == protocol["parent_job"] and manifest["request_sha256"] == protocol["parent_request_sha256"], "accepted parent job/request differs")
    tx = parent_root / "transaction"
    for path in tx.rglob("*"):
        if path.is_file():
            row = manifest["files"][path.relative_to(parent_root).as_posix()]
            require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "inherited transaction member differs from original wrapper")
    verified = verify_transaction(tx, expected_seal_sha256=protocol["parent_global_seal_sha256"], max_bytes=64 << 20)
    forecast = b["helpers"]()[0]["plain"](load_forecast_tree(tx, read(tx / "forecasts/cohort.json")["predictions"]))
    require(verified["phase"] == "closed" and forecast["native_targets_at_seal"] == 0 and not read(tx / "seal.json")["violations"]
        and [read(path)["kind"] for path in sorted((tx / "events").glob("*.json"))] ==
        ["opened", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"], "inherited positive prediction seal/journal differs")
    records = protocol["source_arrays"]
    require(set(records) == {"h", "head", "R", "gate", "up", "product", "M", "Hnext", "input_ids", "attention_mask", "position_ids",
        "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq"}, "exact13 inherited array whitelist differs")
    for record in records.values():
        named = dict(bytes=record["bytes"], sha256=record["sha256"])
        require(manifest["files"][record["path"]] == forecast["external_files"][record["path"]] == named
            and not any(part in {"native", "minus", "shortcut"} for part in Path(record["path"]).parts), "inherited array is not its original presealed positive/source member")
    values = {name: b["load_array"](parent_root, record) for name, record in records.items()}
    require(np.all(values["attention_mask"] == 1) and np.array_equal(values["position_ids"], np.arange(values["h"].shape[1])[None]), "own all-attended positions/mask differ")
    require(same(values["rope_buffer:model.rotary_emb.inv_freq"], values["rope_buffer:model.rotary_emb.original_inv_freq"])
        and all(records[name]["logical_dtype"] == "torch.float32" for name in records if name.startswith("rope_buffer:")), "original FP32 RoPE buffers differ")
    return values, dict(job_id=manifest["job_id"], request_sha256=manifest["request_sha256"], manifest_sha256=sha(parent_root / "files.json"),
        forecast_seal=verified["seal_sha256"], arrays=len(records), bytes=sum(row["bytes"] for row in records.values()))


def same(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def port_bank(root, records, *, head_permutation=None):
    import numpy as np

    require(head_permutation in (None, [0, 2, 1, 3]), "undeclared head alignment operation")
    require(set(PORTS).issubset(records) and all(row["logical_dtype"] == "torch.bfloat16" for row in records.values()), "complete BF16 port records required")
    result = {name: base()["load_array"](root, row) for name, row in records.items()}
    if head_permutation is not None:
        require(result["head"].ndim == 4, "native raw head geometry differs")
        result["head"] = np.transpose(result["head"], head_permutation)
    return result


def saved_accounts(root, inherited, protocol):
    forecast, shortcut, native, responses, frontier = (read(root / name) for name in
        ("forecast.json", "shortcut.json", "native.json", "forecast-responses.json", "source-frontier.json"))
    require(set(forecast["arms"]) == set(shortcut["arms"]) == set(native["arms"]) == set(ARMS)
        and responses["orientation"] == "edited-minus-own-positive" and set(responses["arms"]) == set(ARMS), "two complete edited branch sidecars differ")
    require(frontier["genuine_qwen_act"] is False and frontier["kind"] == "inherited-own-functional-positive-ports"
        and set(frontier["source_arrays"]) == set(protocol["source_arrays"]), "inherited component frontier scope differs")
    for name, record in frontier["source_arrays"].items():
        require(record["inherited"] is True and record["root"] == "input_root"
            and {key: value for key, value in record.items() if key not in {"inherited", "root"}} == protocol["source_arrays"][name], "inherited source reference differs")
    require(forecast["positive"] == {name: frontier["source_arrays"][name] for name in ("h", "head", "R", "gate", "up", "product", "M", "Hnext")}, "archived positive baseline is not exact inherited refs")
    require(native["head_to_canonical"] == [0, 2, 1, 3], "actual native raw-head alignment differs")
    keep = (*PORTS, "gate", "up")
    positive = dict(functional={name: inherited[name] for name in keep},
        native=port_bank(root, {name: native["baseline"]["ports"][name] for name in keep}, head_permutation=native["head_to_canonical"]))
    arms = {}
    split = protocol["fixed_split"]
    for arm in ARMS:
        item = shortcut["arms"][arm]
        signs, permitted, closed = (base()["load_array"](root, item[key]) for key in ("row_signs", "permitted", "support_closed"))
        arms[arm] = dict(signs=signs, permitted=permitted, support_closed=closed, response_hashes=responses["arms"][arm], lanes=dict(
            shortcut=port_bank(root, {name: item["ports"][name] for name in keep}),
            functional=port_bank(root, {name: forecast["arms"][arm]["ports"][name] for name in keep}),
            native=port_bank(root, {name: native["arms"][arm]["ports"][name] for name in keep}, head_permutation=native["head_to_canonical"])))
        consumed = base()["load_array"](root, native["inputs"][arm])
        edited_h = inherited["h"] * signs.reshape(1, -1, 1)
        require(same(consumed, edited_h), "native component consumed wrong edited own H18")
        require(same(base()["load_array"](root, forecast["arms"][arm]["ports"]["h"]), edited_h), "full functional reference consumed wrong edited own H18")
        require(item["profile"]["head_query_axis"] == 1 and item["profile"]["positive_head"] == base()["operand_identity"](frontier["source_arrays"]["head"]), "shortcut head source/query basis differs")
        profile = item["profile"]["positive_profile"]
        require(item["profile"]["row_signs"] == base()["operand_identity"](item["row_signs"]) and item["profile"]["permitted_mask"] ==
            dict(shape=item["permitted"]["shape"], stride=item["permitted"]["stride"], dtype="torch.bool", device=item["permitted"]["device_at_capture"],
                content_sha256=item["permitted"]["content_sha256"]), "shortcut actual signs/permitted operands differ")
        for name in ("R", "gate", "up"):
            require(profile["operands"][name] == base()["operand_identity"](frontier["source_arrays"][name]), "shortcut consumed non-own positive operand")
        attention = profile["attention_profile"]
        require(attention["mask_identity"] == frontier["source_arrays"]["attention_mask"]["content_sha256"]
            and attention["positions_identity"] == frontier["source_arrays"]["position_ids"]["content_sha256"] and attention["query_rows"] == "all-attended"
            and attention["backend"] == "sdpa" and attention["gqa_policy"] == "enable_gqa", "shortcut inherited attention source differs")
        names = dict(Rhat="R", Ghat="gate", Uhat="up", headhat="head")
        require(all(row == base()["operand_identity"](item["ports"][names.get(name, name)]) for name, row in item["output_descriptors"].items())
            and set(item["output_descriptors"]) == {"Rhat", "Ghat", "Uhat", "headhat", "activation", "product", "M", "Hnext"}, "shortcut live/fullsaved output joins differ")
        require(item["counts"]["linear"] == 1 and all(item["counts"][key] == 0 for key in ("attention", "RMSNorm", "gate_projection", "up_projection", "module_forwards")), "shortcut actual primitive scope differs")
    require(same(base()["load_array"](root, native["inputs"]["baseline"]), inherited["h"]), "fresh native baseline did not consume source functional H18")
    context = native["context"]
    require(context == frontier["context"] and context["attention_mask"] is None and context["is_causal"] is True
        and context["enable_gqa"] is True and context["use_cache"] is False, "native component did not consume unchanged own regenerated context")
    require(set(native["inputs"]) == {"baseline", *ARMS} and same(base()["load_array"](root, context["position_ids"]), inherited["position_ids"]), "complete actual inputs/own position bytes differ")
    for name in ("cos", "sin"):
        value = base()["load_array"](root, context[name])
        require(context[name]["logical_dtype"] == "torch.bfloat16" and value.shape[:2] == inherited["h"].shape[:2], "saved regenerated RoPE geometry/dtype differs")
    report, equations = array_accounts(positive, arms, split=split)
    report["head_alignment"] = dict(actual_native="B,H,T,D", canonical="B,T,H,D", permutation=native["head_to_canonical"], scope="Declared view only; no head values recomputed.")
    report["source_context"] = context
    return report, equations


def native_accounts(root, summary, config):
    contract, native, calls = (read(root / name) for name in ("native-contract.json", "native.json", "native-sdpa.json"))
    expected = dict(functional_blocks=2, shortcut_down_projections=2, native_blocks=3, functional_SDPA=2, shortcut_SDPA=0, native_SDPA=3,
        functional_linear=14, shortcut_linear=2, native_linear=21, native_authorizations=3, forecast_module_attempts=0)
    require(contract["observed"] == summary["counts"] and all(summary["counts"][key] == value for key, value in expected.items())
        and summary["counts"]["native_module_entries"] == 45 and contract["expected"] == summary["contract"], "observed finite component/primitive census differs")
    entries, calls = contract["block_entries"], calls["calls"]
    require(len(entries) == len(calls) == 3, "three actual component/SDPA receipts required")
    identities = []
    for arm, entry, call in zip(("baseline", *ARMS), entries, calls, strict=True):
        require(entry["arm"] == call["arm"] == arm and entry["input"] == native["inputs"][arm] and entry["context"] == native["context"], "actual component input/context receipt differs")
        consumed = entry["consumed_context"]
        for name in ("position_ids", "cos", "sin"):
            row, live = native["context"][name], consumed[name]
            require({key: live[key] for key in ("shape", "stride", "dtype", "device", "content_sha256")} ==
                dict(shape=row["shape"], stride=row["stride"], dtype=row["logical_dtype"], device=row["device_at_capture"], content_sha256=row["content_sha256"]), "actual consumed context tensor differs from presealed bytes/layout")
        identities.append({name: {key: consumed[name][key] for key in ("version", "object_id", "data_ptr")} for name in ("position_ids", "cos", "sin")})
        require(consumed["attention_mask"] is None and consumed["past_key_values"] is None and consumed["use_cache"] is False
            and call["attention_mask"] is None and call["is_causal"] is True and call["enable_gqa"] is True and call["dropout_p"] == 0.
            and call["scale"] == config["head_dim"] ** -.5, "actual native delegated dispatch/cache differs")
        length = native["inputs"][arm]["shape"][1]
        shapes = [[1, heads, length, config["head_dim"]] for heads in (config["num_attention_heads"], config["num_key_value_heads"], config["num_key_value_heads"])]
        require([row["shape"] for row in call["QKV"]] == shapes and all(row["dtype"] == "torch.bfloat16" for row in call["QKV"]), "actual native unequal-width Q/K/V geometry differs")
    require(identities[0] == identities[1] == identities[2], "shared native context mutated or was replaced between component calls")
    return dict(observed=summary["counts"], declared=summary["contract"], actual_component_entries=3, actual_context_operand_occurrences=3, actual_context_tensor_joins=9,
        actual_SDPA_receipts=calls, scope="Actual primitive/module/consumer receipts under original pinned component; excluded full-model/LM/tokenizer operations are program-contract declarations, not independent profiled counters.")


def transaction(root, summary, inventory, *, max_bytes, source_identity):
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    tx, plain = root / "transaction", base()["helpers"]()[0]["plain"]
    verified = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"], expected_event_head=summary["transaction"]["event_head"], max_bytes=max_bytes)
    events = [read(path)["kind"] for path in sorted((tx / "events").glob("*.json"))]
    require(verified["phase"] == "closed" and events == ["opened", "frontier_published", "forecast_persisted", "globally_sealed",
        "native_execution_released", "observation_persisted", "closed"] and not read(tx / "seal.json")["violations"], "global transaction journal differs")
    prediction = plain(load_forecast_tree(tx, read(tx / "forecasts/cohort.json")["predictions"]))
    observation = plain(load_forecast_tree(tx, read(tx / "observations/native-verification.json")["value"]))
    captured = plain(load_forecast_tree(tx, read(tx / "frontiers/inherited-positive.json")["value"]))
    frontier = read(root / "source-frontier.json")
    require(captured == dict(source_arrays=frontier["source_arrays"], context=frontier["context"], scope=FRONTIER)
        and summary["frontier"] == frontier["transaction_frontier"] == FRONTIER and summary["genuine_qwen_act"] is False, "inherited architectural carrier scope differs")
    post = {"timeline.jsonl", "native.json", "native-contract.json", "native-sdpa.json", "parameter-identity-after.json", "summary.json", "request.json", "declared-config.json"}
    expected = {name: row for name, row in inventory["files"].items() if not name.startswith(("transaction/", "native/")) and name not in post}
    require(prediction["external_files"] == expected and prediction["inherited_sources"] == frontier["source_arrays"], "complete pretarget frontier includes future or omits source")
    require(prediction["response_hashes"] == read(root / "forecast-responses.json")["arms"] and prediction["native_targets_at_seal"] == 0
        and observation == dict(counts=summary["counts"], arms=["baseline", *ARMS]), "sealed responses/actual occurrence counts differ")
    require(all(value == (0 if key.startswith("native_") or key == "forecast_module_attempts" else summary["counts"][key])
        for key, value in prediction["counts"].items()) and set(prediction["counts"]) == set(summary["counts"]), "native work preceded global seal or forecast census differs")
    declaration = read(tx / "seal.json")["declarations"]
    require(declaration["model_identity"] == "original-qwen3-decoder-layer" and declaration["input_identity"] == frontier["source_arrays"]["h"]["content_sha256"]
        and declaration["source_identity"] == base()["digest"](source_identity) and declaration["frontier"] == dict(FRONTIER,
            conditional_frontier="own-functional-positive-H18-and-ports", external_files=expected, inherited_sources=frontier["source_arrays"]), "sealed source/input/declaration differs")
    journal = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    kinds = ["forecast_arm_completed"] * 2 + ["cohort_globally_sealed", "native_targets_released"] + ["native_component_authorized", "native_component_completed"] * 3
    require([row["kind"] for row in journal] == kinds and [row["number"] for row in journal] == list(range(10))
        and [row["native_module_entries"] for row in journal] == [0, 0, 0, 0, 0, 15, 15, 30, 30, 45]
        and [row["arm"] for row in journal[:2]] == list(ARMS) and [row["arm"] for row in journal[4:]] == [arm for arm in ("baseline", *ARMS) for _ in range(2)]
        and journal[2]["seal_sha256"] == journal[3]["seal_sha256"] == summary["forecast_seal"] and journal[2]["native_targets_at_seal"] == 0, "durable global seal/native-entry chronology differs")
    return dict(verified=verified, chronology=journal, external_files=len(expected), inherited_arrays=13, native_targets_at_seal=0)


def request_envelope(root, inventory, expected_request):
    b = base()
    plan, declared, request, source, summary = (read(root / name) for name in ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json"))
    require(b["identity"](request) == request["request_sha256"] == expected_request and declared == request["config"], "standard executed request/config identity differs")
    require(plan == dict(declared, output=str(Path(declared["output_scratch"]) / inventory["job_id"] / declared["output_name"]),
        request_source_hashes=request["source_hashes"], request_sha256=expected_request, job_id=inventory["job_id"]), "effective producer plan differs")
    require(source["source_hashes"] == request["source_hashes"] and source["job_id"] == summary["job_id"] == inventory["job_id"]
        and source["request_sha256"] == summary["request_sha256"] == expected_request, "actual job/request/source joins differ")
    return plan, request, source, summary


def authenticate(root, expected_manifest, expected_request):
    b = base()
    inventory = wrapper(root, expected_manifest, expected_request)
    plan, request, source, summary = request_envelope(root, inventory, expected_request)
    producer_helpers = dict(b["PRODUCER_HELPERS"], **{"qwen_sign_shortcut_campaign.py": b["WORKER_SHA256"], "qwen_row_sign_shortcut.py": ROW_HELPER_SHA256})
    require(request["source_hashes"]["qwen_row_sign_campaign.py"] == WORKER_SHA256 and source["source_hashes"] == request["source_hashes"]
        and source["helper_source_sha256"] == plan["helper_source_sha256"] == producer_helpers
        and all(request["source_hashes"][key] == value for key, value in producer_helpers.items()), "executed producer/helper source differs")
    require(source["job_id"] == summary["job_id"] == inventory["job_id"] and source["request_sha256"] == summary["request_sha256"] == expected_request and source["frontier"] == FRONTIER
        and plan["schema"] == summary["schema"] == "mdb-qwen-row-sign-campaign-v1", "actual job/request/schema differs")
    require((plan["edit_index"], plan["horizon"], plan["fixed_split"], plan["sequence_rows"]) == (18, 19, 256, 512)
        and plan["attention_dispatch"] == dict(backend="sdpa", gqa_policy="enable_gqa") and plan["qwen_rope_buffer_policy"] ==
        dict(init_device="cpu", init_dtype="float32", placement_dtype="float32", source="original native CPU initializer; device-only placement"), "fixed component/profile policy differs")
    require(set(source["metadata"]) == set(PATH_KEYS), "complete source metadata coverage differs")
    metadata = {}
    for key in PATH_KEYS:
        row = source["metadata"][key]
        require(row == dict(path="metadata/" + key + Path(plan[key]).suffix, sha256=plan[key + "_sha256"])
            and sha(b["member"](root, row["path"])) == row["sha256"], "retained source metadata differs: " + key)
        metadata[key] = root / row["path"]
    require(sha(metadata["selection_protocol"]) == PROTOCOL_SHA256 and sha(metadata["qwen_parameter_manifest"]) ==
        "425502121b4091b10124c548c98127785420233b12e95c8f1d972a186026dccf" and sha(metadata["retrieval_script"]) == b["RETRIEVER_SHA256"], "frozen protocol/original subset/retriever differs")
    protocol, parent_request, parent_files = (read(metadata[key]) for key in ("selection_protocol", "parent_request", "parent_files"))
    require(b["identity"](parent_request) == parent_request["request_sha256"] == protocol["parent_request_sha256"]
        and sha(metadata["parent_files"]) == protocol["parent_raw_manifest_sha256"] and sha(metadata["parent_seal"]) == protocol["parent_global_seal_sha256"]
        and plan["input_root"] == protocol["parent_raw_root"], "accepted parent identity/closure differs")
    parent_names = dict(parent_source="source.json", parent_runtime="runtime.json", parent_parameter_manifest="qwen-parameter-artifacts.json",
        qwen_config="qwen-config.json", qwen_source_binding="qwen-source-binding.json", retrieval_script="retrieve_qwen_weights.py")
    require(all(sha(metadata[key]) == parent_files["files"][name]["sha256"] for key, name in parent_names.items()), "retained original parent metadata lineage differs")
    return plan, source, summary, inventory, metadata, protocol


def source_accounts(root, plan, source, metadata):
    b = base()
    before, after = (read(root / name) for name in ("parameter-identity-before.json", "parameter-identity-after.json"))
    require(before == after and source["selected_original_names"] == sorted(before["payloads"]), "final resident parameter versions/payloads differ")
    qconfig = read(metadata["qwen_config"])
    require(tuple(qconfig[key] for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")) ==
        (2560, 9728, 36, 32, 8, 128) and qconfig.get("attention_bias", False) is False and qconfig.get("attention_dropout", 0.) == 0., "original component geometry/sign assumptions differ")
    parameters = parameter_accounts(read(metadata["parent_parameter_manifest"]), read(metadata["qwen_parameter_manifest"]), qconfig,
        before["payloads"], after["payloads"], plan["parameter_root"], edit_index=18)
    io, restored = source["parameter_recipe_io"], parameters["selected"]
    require(io["payload_reads"] == io["load_requests"] == 11 and io["cache_hits"] == 0 and io["parameter_payload_bytes"] == restored["payload_bytes"] == 201861632
        and io["checkpoint_files_verified_once"] == len(restored["checkpoint_files"]), "actual one-hydration recipe access differs")
    runtime = read(root / "runtime-contract.json")
    require(runtime["torch"].split("+")[0] == "2.13.0" and runtime["transformers"] == "5.14.1" and
        set(runtime["sources"]) == {"qwen_native_source_sha256", "sdpa_source_sha256"}, "original runtime identity differs")
    for key, expected in (("qwen_native_source_sha256", NATIVE_SHA256), ("sdpa_source_sha256", SDPA_SHA256)):
        row = runtime["sources"][key]
        require(plan[key] == row["sha256"] == expected and Path(row["path"]).stat().st_size == row["bytes"] and sha(row["path"]) == expected, "installed native runtime source bytes differ")
    shortcuts = read(root / "shortcut.json")["arms"]
    execution = {}
    for arm, item in shortcuts.items():
        execution[arm] = b["execution_accounts"](item["profile"]["positive_profile"]["execution"], runtime["flags"])
        require(item["profile"]["positive_profile"]["source_identity"] == dict(source["helper_source_sha256"], qwen_native=NATIVE_SHA256, sdpa_native=SDPA_SHA256)
            and item["profile"]["positive_profile"]["operands"]["down_weight"]["bf16_le_sha256"] == before["payloads"]["model.layers.18.mlp.down_proj.weight"]["sha256"], "actual shortcut original learned/source operand differs")
    return dict(parameters=parameters, parameter_recipe_io=io, final_resident_parameter_identity=True, runtime=runtime, execution=execution,
        source_scope="Eleven original block18 recipes; inherited own-functional positive state. No tokenizer, full model, weights export or raw-prefix prediction."), qconfig


def audit(root, output, *, expected_manifest, expected_request):
    root, output = Path(root), Path(output)
    plan, source, summary, inventory, metadata, protocol = authenticate(root, expected_manifest, expected_request)
    inherited, parent = inherited_accounts(plan["input_root"], protocol)
    tx = transaction(root, summary, inventory, max_bytes=plan["forecast_transaction_bytes"],
        source_identity=dict(source["helper_source_sha256"], qwen_native=NATIVE_SHA256, sdpa_native=SDPA_SHA256))
    sources, qconfig = source_accounts(root, plan, source, metadata)
    report, equations = saved_accounts(root, inherited, protocol)
    frontier = read(root / "source-frontier.json")
    require(all(frontier[key] == protocol[key] for key in ("parent_job", "parent_request_sha256", "parent_raw_manifest_sha256", "parent_global_seal_sha256", "source_case"))
        and frontier["input_root"] == plan["input_root"] and (frontier["edit_index"], frontier["horizon"], frontier["fixed_split"], frontier["sequence_rows"]) == (18, 19, 256, 512), "actual inherited frontier lineage/geometry differs")
    report.update(schema="mdb-qwen-row-sign-audit-v1", job_id=inventory["job_id"], request_sha256=expected_request, native_manifest_sha256=expected_manifest,
        worker_sha256=WORKER_SHA256, audit_sha256=sha(__file__), source=sources, parent=parent, transaction=tx,
        census=native_accounts(root, summary, qconfig), mechanics_status="authenticated", instrument_status="complete saved coordinate reductions",
        trend_status="observations; preserve all misses", terminal_status="not certified",
        limitations=["Conditional inherited functional H18 and positive ports, not a fresh raw-input forecast or genuine Qwen Act.",
            "Support-closed/mixed applicability is per fixed window; no whole-prefix-arm accuracy claim or numerical gate.",
            "Native head is explicitly transposed from B,H,T,D to B,T,H,D; actual SDPA receipts do not identify a CUDA kernel.",
            "Premises, source-baseline replay, windows and three channels are dependent accounts; finite algebra is not a BF16 regrouping proof."])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", report)
    contexts = catalog_contexts(plan, source, sources, protocol, report)
    catalog_accounts(output, report, equations, contexts)
    save(output / "coverage.json", dict(arms={arm: row["coverage"] for arm, row in report["arms"].items()},
        model_calls=0, learned_projections=0, tokenizer_calls=0, primary_and_windows_separate=True, dependent_accounts=True))
    return report, inventory


def external_evidence(root, inventory):
    """Both aliases are accepted by the unchanged standalone retriever CLI."""
    frontier = read(root / "source-frontier.json")
    return dict(host="beast", roots=dict(native=str(root), reader=frontier["input_root"]),
        manifests=dict(native={name: row["sha256"] for name, row in inventory["files"].items()},
            reader={row["path"]: row["sha256"] for row in frontier["source_arrays"].values()}),
        alias_roles=dict(native="This component campaign raw output", reader="Legacy retriever CLI alias for authenticated inherited positive/source arrays in the original parent root; not this audit output"),
        scope="Full arrays and original checkpoint remain near data. Compact scalar/catalog/atlas, original recipes/retriever references; no activation or weight bank.")


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(base()["identity"](request) == request["request_sha256"] and request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request/config differs")
    for name, value in request["source_hashes"].items():
        require(sha(name) == value, "staged saved-reader source differs")
    started, root = time.monotonic(), Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    report, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime-contract.json", "files.json", "source-frontier.json", "forecast-responses.json",
        "forecast.json", "shortcut.json", "native.json", "parameter-identity-before.json", "parameter-identity-after.json", "timeline.jsonl", "native-contract.json", "native-sdpa.json"):
        shutil.copyfile(root / name, output / ("source-" + name))
    shutil.copytree(root / "metadata", output / "source-metadata")
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", external_evidence(root, inventory))
    save(output / "request.json", request)
    save(output / "summary.json", dict(schema="mdb-qwen-row-sign-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        source_job=report["job_id"], arms=2, native_component_observations=3, catalog_records=2, atlas_observations=2, model_calls=0, seconds=time.monotonic() - started))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output),
        files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("root", "output", "expected-manifest", "expected-request"):
            parser.add_argument("--" + name, required=True)
        args = parser.parse_args()
        audit(args.root, args.output, expected_manifest=args.expected_manifest, expected_request=args.expected_request)
