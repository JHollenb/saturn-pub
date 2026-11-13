"""Saved-only direct E18-route Act and complete own-state continuation accounts.

Native Acts/advances are verified as genuine receipts. Functional trajectory
cuts remain component-derived. Differences are observations, not certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "a26c70c701c86da8e6da302667c96f67dc5d874ecaf38e3688022e4cfb0a1c62"
PINS = {
    "audit_flux_raw_mask_pair.py": "ef47290a0b3ed44987e2c4696d5674164607c0dd867dbf4a948bf02b49939be6",
    "audit_flux_raw_to_image.py": "f8c1a39f9b86b9002e079db084887aece105304f58e28e1ae5f4fa146a0ae4bf",
    "audit_flux_functional_qwen_prefix.py": "2331126c9b3d918b1ac5376a6a13250c14806bdbcf151277f59320440e191c7a",
    "audit_flux_functional_remaining.py": "81f77f5f8e6a24915095875f1373097616c03b62c4f4983a588f0bc1fd957ea2",
    "audit_flux_functional_pass.py": "43e163b7543ec225da007b6ecc87418c7c0654f464c36c8e7f1e5234642aeaa8",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
HELPERS = tuple(PINS)
CASES = ("sham", "e18-zero")
PROMPT = ("A cobalt blue cube to the right of a matte orange ring on a gray tabletop. "
          "Two separate objects, simple studio product photograph.")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def helpers():
    for name, digest in PINS.items():
        require(sha(HERE / name) == digest, "frozen saved account helper differs: " + name)
    pair = runpy.run_path(str(HERE / "audit_flux_raw_mask_pair.py"))
    return pair, *pair["helpers"]()


def sealed_receipt(body):
    from saturn_pub.values import canonical

    require(body["schema"] == "saturn-pub-receipt-v1" and hashlib.sha256(canonical({key: value for key, value in body.items() if key != "fingerprint"})).hexdigest() == body["fingerprint"],
        "genuine receipt content seal differs")
    require(body.get("status") != "refused", "refused receipt cannot count as a native advance or Act")
    return body


def load_act_cut(root, record):
    from saturn_pub.store import LocalStore

    require(record["store"] == "acts-store", "Act store declaration differs")
    cut = LocalStore(Path(root) / "acts-store").load(record["cut"])
    cut.verify()
    require(cut.fingerprint == record["cut"] and cut.parent == record["parent"] and cut.model_identity == record["model_identity"]
        and cut.boundary == record["boundary"] and dict(cut.execution_point or {}) == record["execution_point"], "actual Act cut/sidecar differs")
    require({key: cut.payload[key] for key in ("step", "phase", "cursor")} == record["clock"] == dict(step=0, phase="project", cursor=0),
        "Act is outside real step0/project entry")
    return cut


def act_accounts(root, acts, own_conditioning, *, columns):
    """Verify registered patch manifests and actual before/after bytes, no apply()."""
    import torch
    from saturn_pub.values import describe, digest

    reports, cuts = {}, {}
    lo, hi = columns
    for lane in ("forecast", "native"):
        reports[lane], cuts[lane] = {}, {}
        require(set(acts[lane]) == set(CASES), "four genuine Act branch coverage differs")
        common_fingerprint = None
        for case in CASES:
            row = acts[lane][case]
            require(row["lane"] == lane and row["branch"] == case and row["columns"] == columns
                and row["operation_id"] == "saturn.builtin.patch", "declared Act lane/branch/operation differs")
            common, before, after = (load_act_cut(root, row[key]) for key in ("common_parent", "before", "after"))
            common_fingerprint = common_fingerprint or common.fingerprint
            require(common.fingerprint == common_fingerprint and before.parent == common.fingerprint and after.parent == before.fingerprint,
                "genuine fork/patch typed lineage differs")
            require(common.model_identity == before.model_identity == after.model_identity
                and dict(common.execution) == dict(before.execution) == dict(after.execution), "actual public adapter identity differs within Act")
            parent, result = before.payload, after.payload
            surface = {slot["name"]: slot for slot in before.surface["slots"]}
            require(surface["conditioning"]["writable"] is True and surface["conditioning"]["invalidates"] == [], "Act surface does not authorize conditioning patch")
            require(describe(common.payload) == describe(parent), "branch did not fork common own-lane parent")
            own = parent["conditioning"]
            require(own.dtype == torch.bfloat16 and 0 <= lo < hi <= own.shape[-1]
                and describe(own) == describe(own_conditioning[lane]), "Act used another lane E or wrong BF16 input")
            mask = torch.zeros_like(own, dtype=torch.bool)
            mask[..., lo:hi] = True
            expected = own.clone()
            if case == "e18-zero":
                expected[mask] = 0
            require(describe(expected) == describe(result["conditioning"]), "actual conditioning patch bytes differ")
            require(describe({key: value for key, value in parent.items() if key != "conditioning"}) ==
                describe({key: value for key, value in result.items() if key != "conditioning"}), "Act changed noise/IDs/schedule/clock/collateral slot")
            receipt = sealed_receipt(row["receipt"])
            require(receipt["operation"] == "apply" and receipt["parent"] == before.fingerprint and receipt["result"] == after.fingerprint
                and receipt["model_identity"] == after.model_identity and receipt["execution"] == dict(after.execution)
                and receipt["execution_point"] == dict(after.execution_point), "actual Act receipt/cut join differs")
            manifest = receipt["act"]
            require(manifest["name"] == "patch" and manifest["reads"] == manifest["writes"] == ["conditioning"] and manifest["invalidates"] == []
                and manifest["preconditions"] == {"conditioning": digest(describe(own))}, "registered patch footprint/precondition differs")
            require(manifest["operation"]["id"] == "saturn.builtin.patch" and manifest["operation"]["version"] == "1"
                and manifest["operation"]["pure"] is True and manifest["operation"]["authority"] == "registered", "actual registered operation authority differs")
            parameters = dict(address="conditioning", value=own if case == "sham" else 0, mask=mask, mode="replace", dose=1)
            require(manifest["parameters"] == describe(parameters) and receipt["writes"] == describe({"conditioning": result["conditioning"]}),
                "whole exact-shape Bool mask/value/mode or actual writes differ")
            require(row["exact_shape_mask"] == list(mask.shape) and row["selected_values"] == int(mask.sum())
                and row["conditioning_before"] == describe(own) and row["conditioning_after"] == describe(result["conditioning"])
                and row["all_other_slots_unchanged"] is True, "Act observed operand sidecar differs")
            changed = int((own.view(torch.int16) != result["conditioning"].view(torch.int16)).sum())
            require(changed == row["changed_values"], "actual Act changed-bit census differs")
            reports[lane][case] = dict(receipt=receipt["fingerprint"], parent=before.fingerprint, result=after.fingerprint,
                model_identity=after.model_identity, execution=dict(after.execution), columns=columns, selected_values=int(mask.sum()), changed_values=changed,
                outside_E18_exact=True, all_other_slots_exact=True, positive_zero_bits=case == "e18-zero", own_lane_E_exact=True)
            cuts[lane][case] = after
    return reports, cuts


def state_loader(root, old, streamer, config, *, identity_cut=None):
    from saturn_pub.store import LocalStore

    forecast, _ = old["state_loader"](root, streamer, config)

    def load(record, lane):
        if lane == "forecast":
            payload = forecast(record, lane)
            if identity_cut is not None:
                manifest = read(Path(root) / "forecast-state-store/cuts" / (record["cut"] + ".json"))["manifest"]
                require(manifest["model_identity"] == identity_cut.model_identity and {key: value for key, value in manifest["execution"].items()
                    if key not in {"composition", "lane", "transition", "receipt_kind"}} == dict(identity_cut.execution), "component-derived cut changed original public adapter/model identity")
            return payload
        require(record["store"] == "native-state-store", "native state store differs")
        cut = LocalStore(Path(root) / record["store"]).load(record["cut"])
        cut.verify()
        require(cut.fingerprint == record["cut"] and cut.parent == record["parent"] and cut.model_identity == record["model_identity"]
            and cut.boundary == record["boundary"] and dict(cut.execution_point or {}) == record["execution_point"] == streamer["point"](record["clock"]),
            "genuine native typed cut/clock differs")
        require("receipt_kind" not in cut.execution and "composition" not in cut.execution, "native Session cut mislabeled component-derived")
        payload = cut.payload
        require({key: payload[key] for key in ("step", "phase", "cursor")} == record["clock"], "native actual payload clock differs")
        return payload
    return load


def receipt_accounts(root, acts, cuts, rows, *, transitions):
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe, digest

    emissions = [json.loads(line) for line in (Path(root) / "receipt-emissions.jsonl").read_text().splitlines()]
    require([row["occurrence"] for row in emissions] == list(range(4 + 2 * transitions)), "physical receipt occurrence sequence differs")
    expected = [(lane, case, "act") for lane in ("forecast", "native") for case in CASES]
    expected += [("native", case, "native") for case in CASES for _ in range(transitions)]
    require([(row["lane"], row["branch"], row["kind"]) for row in emissions] == expected, "actual Act/native receipt emission order differs")
    for row in emissions[:4]:
        require(sealed_receipt(row["receipt"]) == acts[row["lane"]][row["branch"]]["receipt"], "emitted Act receipt differs from actual Act record")
    report = []
    for case_index, case in enumerate(CASES):
        native_rows = rows[case]["native"]
        require(native_rows[0]["cut"] == cuts["native"][case].fingerprint, "native trajectory did not start from own actual post-Act cut")
        require(rows[case]["forecast"][0]["parent"] == cuts["forecast"][case].fingerprint, "functional origin is not a component-derived child of own post-Act cut")
        store = LocalStore(Path(root) / "cases" / case / "native-state-store")
        for number, emission in enumerate(emissions[4 + case_index * transitions:4 + (case_index + 1) * transitions]):
            body = sealed_receipt(emission["receipt"])
            before, after = (store.load(native_rows[index]["cut"]) for index in (number, number + 1))
            require(body["operation"] == "continue" and body["steps"] == 1 and len(body["transitions"]) == 1
                and body["parent"] == before.fingerprint and body["result"] == after.fingerprint and after.parent == before.fingerprint,
                "actual single native advance/cut lineage differs")
            require(body["model_identity"] == after.model_identity == before.model_identity == cuts["native"][case].model_identity
                and body["execution"] == dict(after.execution) == dict(before.execution) == dict(cuts["native"][case].execution),
                "native advance public adapter/model identity differs")
            transition = body["transitions"][0]
            require(transition["before"] == dict(before.execution_point) and transition["after"] == dict(after.execution_point)
                and native_rows[number + 1]["receipt"] == body["fingerprint"] and native_rows[number + 1]["receipt_parent"] == before.fingerprint,
                "native advance execution point/receipt sidecar differs")
            require(body["boundary"] == after.boundary and body["execution_point"] == dict(after.execution_point), "native receipt result boundary differs")
            operation = transition["native_operation"]
            slots = {slot["name"] for slot in before.surface["slots"]}
            require(operation["footprint"] == "declared" and set(operation["reads"] + operation["writes"] + operation["invalidates"]) <= slots,
                "native advance footprint contains undeclared slots")
            p, n = before.payload, after.payload
            for key in slots - set(operation["writes"] + operation["invalidates"]):
                require(describe(p[key]) == describe(n[key]), "actual native advance changed an undeclared slot")
            require(transition["occurrence"] == digest(dict(entry=transition["before"], operation=transition["structural_operation"],
                transaction_parent=before.fingerprint, transition_index=0, execution=body["execution"])), "native transition occurrence identity differs")
            report.append(dict(branch=case, number=number, fingerprint=body["fingerprint"], parent=before.fingerprint, result=after.fingerprint,
                operation=transition["structural_operation"], native_operation=transition["native_operation"]))
    return dict(actual_acts=4, actual_native_advances=2 * transitions, total_receipt_occurrences=len(emissions), advances=report)


def context_accounts(consumers, cuts, *, steps):
    from saturn_pub.values import describe

    require(len(consumers) == 2 * steps and {(row["branch"], row["logical_step"]) for row in consumers} == {(case, step) for case in CASES for step in range(steps)},
        "actual project context consumer occurrence coverage differs")
    for row in consumers:
        own = cuts["native"][row["branch"]]
        require(row["operand"] == describe(own.payload["conditioning"]) and row["source_act_cut"] == own.fingerprint
            and row["actual_forward_pre_hook"] is row["full_operand_exact"] is True, "actual context_embedder operand did not join own acted conditioning")
    return dict(actual_occurrences=len(consumers), consumers=consumers, scope="Source-recorded actual linear prehook full-operand descriptors, independently joined to archived Act result bytes; no new module invocation.")


def authenticate(root, expected_manifest, expected_request, old):
    require(sha(root / "files.json") == expected_manifest, "raw wrapper checksum differs")
    inventory = read(root / "files.json")
    require(inventory["schema"] == "mdb-image-tail-files-v1" and inventory["request_sha256"] == expected_request, "raw wrapper schema/request differs")
    members = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != root / "files.json"}
    require(members == set(inventory["files"]) and not any(path.is_symlink() for path in root.rglob("*")), "complete raw member inventory differs")
    for name, record in inventory["files"].items():
        path = old["member"](root, name)
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "retained evidence differs: " + name)
    require(inventory["expanded_bytes"] == sum(record["bytes"] for record in inventory["files"].values()), "raw byte census differs")
    plan, request, source, summary, contract = (read(root / name) for name in ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json"))
    require(identity(request) == request["request_sha256"] == expected_request and plan["request_sha256"] == expected_request
        and all(plan[key] == value for key, value in request["config"].items()), "executed request/effective plan differs")
    require(all(row["request_sha256"] == expected_request and row["job_id"] == inventory["job_id"] for row in (source, summary)), "actual job/request joins differ")
    require(plan["schema"] == "mdb-flux-e18-conditioning-plan-v1" and plan["prompt"] == PROMPT and plan["seed"] == 277
        and plan["branches"] == list(CASES) and plan["conditioning_columns"] == [2560, 5120], "fixed direct E18-route specimen differs")
    require(request["source_hashes"]["flux_e18_conditioning_act.py"] == source["worker_sha256"] == WORKER_SHA256, "executed producer source differs")
    require(source["helper_source_sha256"] == plan["helper_source_sha256"] and all(request["source_hashes"][key] == value for key, value in source["helper_source_sha256"].items()),
        "source-bound original helper graph differs")
    require(summary["counts"] == contract["counts"] and contract["forecast_module_entries"] == 0 and summary["forecast_seal"] == source["forecast_seal"],
        "actual counts/guard/seal sidecars differ")
    expected = dict(component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=2, functional_transitions=224,
        native_transitions=224, functional_decode_calls=2, native_vae_calls=4, native_qwen_calls=1, act_receipts=4, receipt_occurrences=228, guard_refusals=0)
    require(all(summary["counts"][key] == value for key, value in expected.items()) and contract["context_operand_occurrences"] == summary["context_operand_occurrences"] == 8,
        "actual bounded operation census differs")
    require(sha(root / "runtime-contract.json") == source["runtime_contract_sha256"], "actual runtime contract checksum differs")
    return plan, source, summary, contract, inventory


def native_member(name):
    return name.startswith(("qwen/native/", "frontend/native/")) or name == "frontend/native.json" or "/native-state-store/" in name \
        or "/decoder/native-own/" in name or "/decoder/native-reference/" in name \
        or Path(name).name in {"native-state-rows.json", "decoder-native-own.json", "decoder-native-reference.json"}


def pretarget_files(root, inventory_names, forecast_acts):
    post = {"native-qwen.json", "native-acts.json", "input-lane-native.json", "act-records.json", "context-consumers.json", "native-contract.json", "source.json", "summary.json"}
    ordinary = {name for name in inventory_names if name not in post | {"timeline.jsonl", "receipt-emissions.jsonl"}
        and not name.startswith("forecast-transaction/") and not native_member(name) and not name.startswith("acts-store/")}
    return ordinary | act_store_closure(root, forecast_acts)


def transaction(root, config, source, summary, inventory, old):
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    tx = root / "forecast-transaction"
    verified = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"]["seal_sha256"], expected_event_head=summary["transaction"]["event_head"],
        max_bytes=config["forecast_transaction_bytes"])
    require(verified["phase"] == "closed" and [read(path)["kind"] for path in sorted((tx / "events").glob("*.json"))] ==
        ["opened", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"]
        and not read(tx / "seal.json")["violations"], "persisted transaction journal is incomplete/violated")
    prediction = old["plain"](load_forecast_tree(tx, read(tx / "forecasts/e18-complete-composition.json")["predictions"]))
    observed = old["plain"](load_forecast_tree(tx, read(tx / "observations/native-own-E-branches.json")["value"]))
    require(prediction["native_targets_at_seal"] == 0 and observed == dict(native_transitions=summary["counts"]["native_transitions"],
        native_vae_calls=summary["counts"]["native_vae_calls"], native_act_receipts=2, context_operand_occurrences=summary["context_operand_occurrences"]),
        "cohort forecast-release/observation count differs")
    post = {"native-qwen.json", "native-acts.json", "input-lane-native.json", "act-records.json", "context-consumers.json", "native-contract.json", "source.json", "summary.json"}
    declared = prediction["external_files"]
    declarations = read(tx / "seal.json")["declarations"]
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source_keys = tuple(source["source_files"])
    require(declarations["model_identity"] == "flux2-klein-4b-original"
        and declarations["input_identity"] == hashlib.sha256(canonical(dict(prompt=PROMPT, seed=277, intervention=config["intervention"]))).hexdigest()
        and declarations["source_identity"] == hashlib.sha256(canonical({key: config[key + "_sha256"] for key in source_keys})).hexdigest(),
        "sealed model/input/source identity differs")
    require(declarations["frontier"] == dict(kind="raw-input", declared_intervention="conditioning-Act", prompt=PROMPT, seed=277,
        intervention=config["intervention"], source_inventory=declared), "sealed own raw-input/Act frontier differs")
    require(set(declared) == pretarget_files(root, inventory["files"], prediction["acts"]), "complete pretarget file frontier differs")
    require({name for name in declared if name.startswith("acts-store/")} == act_store_closure(root, prediction["acts"]),
        "forecast Act CAS closure is incomplete or includes native-only future data")
    require(all(not native_member(name) and name not in post for name in declared), "native future evidence present in sealed frontier")
    for name, row in declared.items():
        path = old["member"](root, name)
        require(row == inventory["files"][name] and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "sealed input/prediction dependency differs")
    require(prediction["acts"] == read(root / "forecast-acts.json") and prediction["responses"] == dict(path="forecast-responses.json", sha256=sha(root / "forecast-responses.json"), orientation="e18-zero-minus-sham"),
        "pretarget actual Acts/complete signed response commitment differs")
    return verified, prediction


def act_store_closure(root, records):
    names = set()
    for case in CASES:
        for role in ("common_parent", "before", "after"):
            cut_name = "acts-store/cuts/" + records[case][role]["cut"] + ".json"
            data = read(Path(root) / cut_name)
            names.add(cut_name)
            names.update("acts-store/pages/" + page + ".json" for page in data["slots"].values())
            names.update("acts-store/blobs/" + blob + ".safetensors" for blob in data["physical_blobs"])
    return names


def qwen_accounts(root, fq, nq, config, old, q, hidden_width):
    import numpy as np

    ids = {lane: old["load_array"](root, record["input_ids"]) for lane, record in (("forecast", fq), ("native", nq))}
    masks = {lane: old["load_array"](root, record["attention_mask"]) for lane, record in (("forecast", fq), ("native", nq))}
    require(old["bytes_equal"](ids["forecast"], ids["native"]) and old["bytes_equal"](masks["forecast"], masks["native"]), "own Qwen tokenized source input differs")
    mask = masks["forecast"]
    require(ids["forecast"].shape == mask.shape and np.all((mask == 0) | (mask == 1)) and len(fq["hidden_states"]) == len(nq["hidden_states"]) == 28
        and fq["selected_hidden_states"] == nq["selected_hidden_states"] == [9, 18, 27] and nq["executed_full_layers"] == 36, "interior Qwen native/functional coverage differs")
    require(ids["forecast"].shape == (1, config["max_length"]) and old["bytes_equal"](old["load_array"](root, fq["position_ids"]), np.arange(config["max_length"], dtype=np.int64)[None]),
        "actual tokenized input/position query-row geometry differs")
    fields, selected = {}, {lane: [] for lane in ("forecast", "native")}
    for index in range(28):
        require(fq["hidden_states"][index]["logical_dtype"] == nq["hidden_states"][index]["logical_dtype"] == "torch.bfloat16", "own Qwen hidden logical dtype differs")
        values = {lane: old["load_array"](root, row["hidden_states"][index]) for lane, row in (("forecast", fq), ("native", nq))}
        require(all(value.shape == (*mask.shape, hidden_width) for value in values.values()), "retained Qwen query-row geometry differs")
        fields[str(index)] = q["row_accounts"](values["forecast"], values["native"], mask)
        if index in (9, 18, 27):
            for lane in selected:
                selected[lane].append(values[lane])
    own = {}
    for lane, row in (("forecast", fq), ("native", nq)):
        own[lane] = old["load_array"](root, row["conditioning"])
        require(old["bytes_equal"](np.concatenate(selected[lane], axis=-1), own[lane]), "conditioning is not own interior selected states")
    fields["conditioning"] = q["row_accounts"](own["forecast"], own["native"], mask)
    first = next((name for name, row in fields.items() if not row["full"]["bitwise_exact"]), None)
    return dict(fields=fields, first_bit_discrepancy=first, scope="Own original Qwen27-layer functional prefix vs full36-layer native interior states. Final normalized output is reference-only."), own, mask


def parser_sources(root, source, metadata, old):
    binding = read(metadata["qwen_source_binding"])
    expected = {name for name in binding["source_files"] if name.startswith("tokenizer/") or name in {"text_encoder/config.json", "text_encoder/generation_config.json"}}
    require(set(source["tokenizer_files"]) == expected, "complete pinned parser/template source set differs")
    for name, row in source["tokenizer_files"].items():
        path = old["member"](root, row["retained"])
        require(path.stat().st_size == row["bytes"] == binding["source_files"][name]["bytes"]
            and sha(path) == row["sha256"] == binding["source_files"][name]["sha256"], "saved parser/template source checksum differs")
    return dict(source_files=source["tokenizer_files"], tokenizer_executions=0,
        scope="Authenticated parser implementation/template files and saved token IDs only; no retokenization or independent parser reconstruction.")


def frontend_accounts(root, fronts, cuts, old):
    reports = {}
    for key in ("raw_noise", "packed_noise", "img_ids", "txt_ids", "timesteps", "sigmas", "generator_before", "generator_after"):
        values = {lane: old["load_array"](root, fronts[lane][key]) for lane in ("forecast", "native")}
        require(old["bytes_equal"](values["forecast"], values["native"]), "independent same-seed frontend byte identity differs")
        reports[key] = old["metric"](values["forecast"], values["native"])
        if key not in ("raw_noise", "generator_before", "generator_after"):
            slot = "latent" if key == "packed_noise" else key
            for lane in values:
                for case in CASES:
                    require(old["bytes_equal"](old["field_to_numpy"](cuts[lane][case].payload[slot]), values[lane]), "actual Act cut did not retain own frontend slot")
    return dict(fields=reports, scope="Archived independently formed lane noise/IDs/schedule/generator bytes and Act-cut consumption joins; no new RNG/scheduler execution.")


def tensor_leaves(value, name=""):
    if hasattr(value, "detach"):
        yield name, value
    elif hasattr(value, "items"):
        for key, child in value.items():
            yield from tensor_leaves(child, name + "." + str(key) if name else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, name + "." + str(index))


def response_account(values, sealed, old):
    import numpy as np

    deltas = {lane: np.asarray(values[lane]["e18-zero"], np.float64) - np.asarray(values[lane]["sham"], np.float64) for lane in ("forecast", "native")}
    predicted = deltas["forecast"]
    expected = dict(shape=list(predicted.shape), elements=int(predicted.size), dtype="float64", orientation="e18-zero-minus-sham",
        sha256=hashlib.sha256(np.asarray(predicted, dtype="<f8").tobytes(order="C")).hexdigest())
    require(sealed == expected, "complete pretarget signed response hash differs")
    return dict(forecast={**old["metric"](values["forecast"]["e18-zero"], values["forecast"]["sham"]), "orientation": "e18-zero-minus-sham"},
        native={**old["metric"](values["native"]["e18-zero"], values["native"]["sham"]), "orientation": "e18-zero-minus-sham"}, signed_response_error=old["metric"](deltas["forecast"], deltas["native"]),
        prospective_full_hash_verified=True, sealed=sealed,
        scope="Direct-E18-route signed branch response; targets/collaterals/continuations share state and are dependent observations.")


def audit(root, output, *, expected_manifest, expected_request):
    import torch
    from saturn_pub.values import describe

    from mdb_lanes.equation_comparison import EquationCatalog

    root, output = Path(root), Path(output)
    pair, old, q, streamer, remaining = helpers()
    config, source, summary, contract, inventory = authenticate(root, expected_manifest, expected_request, old)
    require(all(source["helper_source_sha256"].get(key) == value for key, value in old["NUMERICAL_PINS"].items())
        and source["helper_source_sha256"].get("flux_raw_to_image.py") == old["WORKER_SHA256"], "frozen original numerical graph differs")
    metadata = old["source_metadata"](root, config, source)
    restoration = pair["parameters"](root, config, source, contract, metadata, streamer)
    runtime = old["runtime_audit"](config, read(root / "runtime-contract.json"))
    parser = parser_sources(root, source, metadata, old)
    verification, prediction = transaction(root, config, source, summary, inventory, old)
    acts = read(root / "act-records.json")
    require(acts["forecast"] == prediction["acts"] == read(root / "forecast-acts.json") and acts["native"] == read(root / "native-acts.json"), "four actual Act sidecars differ")
    fq, nq = (read(root / (lane + "-qwen.json")) for lane in ("forecast", "native"))
    qconfig, fconfig = read(metadata["qwen_config"]), read(metadata["flux_config"])
    qreport, own_np, mask = qwen_accounts(root, fq, nq, config, old, q, qconfig["hidden_size"])
    require(config["conditioning_columns"] == [qconfig["hidden_size"], 2 * qconfig["hidden_size"]], "direct middle interior-state slice differs")
    own = {lane: torch.from_numpy(value.copy()).to(torch.bfloat16) for lane, value in own_np.items()}
    act_report, cuts = act_accounts(root, acts, own, columns=config["conditioning_columns"])
    fronts = {}
    for lane in ("forecast", "native"):
        declared = read(root / ("input-lane-" + lane + ".json"))
        require(declared["branches"] == acts[lane] and declared["own_conditioning"] == describe(own[lane]), "own E/input-lane/Act declaration differs")
        fronts[lane] = declared["frontend"]
    frontend = frontend_accounts(root, fronts, cuts, old)
    rows, loaders, case_reports, finals, endpoints, decoders = {}, {}, {}, {}, {}, {}
    transitions = config["steps"] * (3 + fconfig["num_layers"] + fconfig["num_single_layers"])
    require(transitions == 112, "complete four-pass original transition geometry differs")
    for case in CASES:
        case_root = root / "cases" / case
        rows[case] = {lane: read(case_root / (lane + "-state-rows.json")) for lane in ("forecast", "native")}
        require(all(len(value) == transitions + 1 for value in rows[case].values()), "complete own-branch retained boundary coverage differs")
        load = state_loader(case_root, old, streamer, config, identity_cut=cuts["forecast"][case])
        loaders[case] = load
        origins = {lane: load(rows[case][lane][0], lane) for lane in ("forecast", "native")}
        for lane in origins:
            require(describe(origins[lane]) == describe(cuts[lane][case].payload), "trajectory initial state did not consume own actual post-Act payload")
        trajectory = old["trajectory_accounts"](rows[case]["forecast"][1:], rows[case]["native"][1:],
            {lane: rows[case][lane][0] for lane in origins}, load, joints=fconfig["num_layers"], singles=fconfig["num_single_layers"], steps=config["steps"])
        finals[case] = {lane: load(rows[case][lane][-1], lane) for lane in origins}
        decoders[case] = {lane: read(case_root / ("decoder-" + lane + ".json")) for lane in ("forecast", "native-own", "native-reference")}
        decoder, endpoints[case] = old["decoder_accounts"](case_root, config, decoders[case], finals[case], streamer, remaining, metadata)
        case_reports[case] = dict(denoiser=trajectory, decoder=decoder)
    receipts = receipt_accounts(root, acts, cuts, rows, transitions=transitions)
    consumers = context_accounts(read(root / "context-consumers.json"), cuts, steps=config["steps"])
    sealed = read(root / "forecast-responses.json")
    require(sealed["schema"] == "mdb-e18-signed-response-v1" and sealed["orientation"] == "e18-zero-minus-sham"
        and len(sealed["denoiser"]) == transitions + 1 and set(sealed["decoder"]) == {"decoded", "pixels"}, "complete response contract differs")
    responses = dict(denoiser=[], decoder={})
    for index, sealed_row in enumerate(sealed["denoiser"]):
        require(all(rows[case][lane][index]["clock"] == sealed_row["clock"] and rows[case][lane][index]["number"] == sealed_row["number"]
            for case in CASES for lane in ("forecast", "native")), "signed response boundary clocks differ")
        values = {lane: {case: dict(tensor_leaves(loaders[case](rows[case][lane][index], lane))) for case in CASES} for lane in ("forecast", "native")}
        field_set = set(sealed_row["fields"])
        require(all(set(values[lane][case]) == field_set for lane in values for case in CASES), "complete recursive response field coverage differs")
        fields = {}
        for name in sorted(field_set):
            numeric = {lane: {case: old["field_to_numpy"](values[lane][case][name]) for case in CASES} for lane in values}
            fields[name] = response_account(numeric, sealed_row["fields"][name], old)
            if name == "conditioning":
                lo, hi = config["conditioning_columns"]
                routes = {}
                for role, selected_slice in (("E9-collateral", slice(0, lo)), ("E18-target", slice(lo, hi)), ("E27-collateral", slice(hi, None))):
                    sub = {lane: {case: numeric[lane][case][..., selected_slice] for case in CASES} for lane in numeric}
                    routes[role] = {lane: {**old["metric"](sub[lane]["e18-zero"], sub[lane]["sham"]), "orientation": "e18-zero-minus-sham"} for lane in sub}
                fields[name]["declared_target_and_collateral_routes"] = routes
        responses["denoiser"].append(dict(number=sealed_row["number"], clock=sealed_row["clock"], fields=fields))
    for field in ("decoded", "pixels"):
        values = {lane: {case: endpoints[case]["forecast" if lane == "forecast" else "native-reference"][field] for case in CASES} for lane in ("forecast", "native")}
        responses["decoder"][field] = response_account(values, sealed["decoder"][field], old)
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    kinds = [row["kind"] for row in events]
    require(kinds.count("forecast_globally_sealed") == kinds.count("native_targets_released") == kinds.count("native_qwen_entry") == 1
        and kinds.index("forecast_globally_sealed") < kinds.index("native_targets_released") < kinds.index("native_qwen_entry"), "global pretarget chronology differs")
    seal_event = events[kinds.index("forecast_globally_sealed")]
    require(seal_event["native_module_entries"] == 0 and seal_event["forecast_act_receipts"] == 2 and seal_event["functional_transitions"] == 224
        and seal_event["seal_sha256"] == summary["forecast_seal"]["seal_sha256"], "actual pretarget guard/Act census differs")
    inputs = read(root / "input.json")
    require(inputs["prompt"] == PROMPT and inputs["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
        and inputs["attended_rows"] == int(mask.sum()) and inputs["padded_rows"] == int(mask.size - mask.sum()), "raw prompt/saved tokenized row joins differ")
    result = dict(schema="mdb-flux-e18-conditioning-saved-audit-v1", job_id=inventory["job_id"], request_sha256=expected_request,
        raw_manifest_sha256=expected_manifest, raw_members=len(inventory["files"]), raw_expanded_bytes=inventory["expanded_bytes"], source=source,
        parameters=restoration, runtime=runtime, parser=parser, transaction=verification, input=inputs, qwen=qreport, acts=act_report, receipt_account=receipts,
        context_consumers=consumers, frontend=frontend, cases=case_reports, signed_responses=responses, actual_counts=summary["counts"],
        public_runtime_provenance=dict(runtime_wheels=read(root / "request.json").get("runtime_wheels"),
            execution_environment=read(root / "request.json").get("execution_environment"),
            actual_adapter_execution=act_report["native"]["sham"]["execution"], actual_model_identity=act_report["native"]["sham"]["model_identity"],
            scope="Actual source request/runtime pins plus typed Act/native-cut execution identities. Functional children preserve this adapter identity without inventing native receipts or Session interchangeability."),
        mechanics_status="authenticated-genuine-Acts-and-own-state-continuations", trend_status="observed", terminal_status="unassessed",
        new_model_calls=0, new_tokenizer_calls=0, scope="Direct E18 conditioning route removal, not upstream Qwen17 MLP ablation. H27 retains its upstream history.",
        limitations=["Only four native apply receipts and224 native advance receipts are counted as genuine occurrences. Functional112 transitions perbranch are component-derived StateCuts.",
            "Own E_forecast and E_native are formed separately and patched within their own lane. Equality of values, if observed, does not prove crossinstallation or independence.",
            "Actual8 native context_embedder full-operand prehook descriptors join archived native Act result bytes; source descriptors are not a new kernel measurement.",
            "Raw prompt/tokenizer files and saved tokenized inputs are authenticated; no tokenizer executed by this reader. No independent parser/RNG/attention kernel claim.",
            "Targets, unchanged E9/E27/noise collateral and downstream continuations are reported perrow. No automatic family induction or scientific pass/fail threshold."])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    catalog = EquationCatalog(output / "catalog/equations")
    queries = []
    for case in CASES:
        for lane, occurrence in (("native-reference", "whole"), ("native-own", "same-latent-decoder")):
            context = dict(branch=case, occurrence=occurrence, prompt_sha256=inputs["prompt_sha256"], seed=277,
                clock=dict(step=0, phase="project", cursor=0), intervention=config["intervention"], native_act=acts["native"][case]["receipt"]["fingerprint"],
                forecast_act=acts["forecast"][case]["receipt"]["fingerprint"], forecast_seal=summary["forecast_seal"]["seal_sha256"], request_sha256=expected_request)
            record = catalog.record(law_id="law.flux.direct-E18-act.complete-program.v1", route_id="route.flux.original-shared-primitives.E18-conditioning.v1",
                alias="cmp.flux.E18." + case + "." + occurrence,
                binding=dict(model_identity=cuts["native"][case].model_identity, execution=dict(id="mdb-flux-e18-conditioning-act-v1"),
                    input_identity=acts["forecast"][case]["after"]["cut"], source_identity=expected_request, frontier="own raw-Qwen E then actual initial-project conditioning Act", horizon=dict(steps=4, decoder=occurrence)),
                report=old["catalog_comparison"](endpoints[case]["forecast"], endpoints[case][lane]),
                conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
                evidence=dict(audit_sha256=sha(output / "audit.json"), genuine_acts=True, independent_kernel=False, direct_route_only=True))
            exact = len(catalog.find(applicable_to=context))
            wrong = len(catalog.find(applicable_to={**context, "clock": dict(step=1, phase="project", cursor=0)}))
            require(exact == 1 and wrong == 0, "catalog context/wrong-clock abstention differs")
            queries.append(dict(fingerprint=record["fingerprint"], context=context, exact=exact, wrong_clock=wrong))
    save(output / "catalog/provenance.json", dict(records=4, queries=queries, automatic_family_induction=False,
        scope="Dependent exact-context endpoint accounts; direct E18 route removal does not erase E27 upstream history."))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(identity(request) == request["request_sha256"] and request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged saved-reader source differs")
    started = time.monotonic()
    root = Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json", "files.json", "runtime-contract.json", "input.json",
        "forecast-qwen.json", "native-qwen.json", "act-records.json", "forecast-acts.json", "native-acts.json", "context-consumers.json", "receipt-emissions.jsonl", "timeline.jsonl", "forecast-responses.json"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for key, row in result["source"]["source_files"].items():
        target = output / "source-metadata" / (key + "-" + Path(row["path"]).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row["path"], target)
    for case in CASES:
        for lane in ("forecast", "native-own", "native-reference"):
            record = read(root / "cases" / case / ("decoder-" + lane + ".json"))
            shutil.copyfile(root / "cases" / case / record["png"]["path"], output / (case + "-" + lane + ".png"))
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots=dict(native=str(root)), manifests=dict(native={name: row["sha256"] for name, row in inventory["files"].items()}),
        scope="Complete arrays/cuts remain near data; compact original6PNGs, source/recipe/retriever metadata and accounts only, no weight bank."))
    save(output / "request.json", request)
    save(output / "summary.json", dict(schema="mdb-flux-e18-conditioning-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
        source_job=result["job_id"], cases=2, genuine_acts=4, native_advances=224, records=4, model_calls=0, seconds=time.monotonic() - started))
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
