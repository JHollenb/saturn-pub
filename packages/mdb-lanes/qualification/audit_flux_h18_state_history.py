"""Saved-only H18 sign-history, genuine FLUX Acts and own-state accounts.

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
import types
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "3269566464b4ff94202b62910699d7b32e3c5a3530f7c5b9d9a2e8064b04790b"
HISTORY_SHA256 = "17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036"
PINS = {
    "audit_flux_e18_conditioning_act.py": "5e6a33d9f6616c2d360e58dcf3adf4e77b4cbdc6b1b1c5e4f1c10b2d341ed6a4",
    "audit_flux_raw_mask_pair.py": "ef47290a0b3ed44987e2c4696d5674164607c0dd867dbf4a948bf02b49939be6",
    "audit_flux_raw_to_image.py": "f8c1a39f9b86b9002e079db084887aece105304f58e28e1ae5f4fa146a0ae4bf",
    "audit_flux_functional_qwen_prefix.py": "2331126c9b3d918b1ac5376a6a13250c14806bdbcf151277f59320440e191c7a",
    "audit_flux_functional_remaining.py": "81f77f5f8e6a24915095875f1373097616c03b62c4f4983a588f0bc1fd957ea2",
    "audit_flux_functional_pass.py": "43e163b7543ec225da007b6ecc87418c7c0654f464c36c8e7f1e5234642aeaa8",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
HELPERS = tuple(PINS)
CASES = ("direct", "coherent")
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


@lru_cache(maxsize=1)
def e18_accounts():
    for name, digest in PINS.items():
        require(sha(HERE / name) == digest, "frozen saved account helper differs: " + name)
    return runpy.run_path(str(HERE / "audit_flux_e18_conditioning_act.py"))


def helpers():
    return e18_accounts()["helpers"]()


def generic(name, *args, **kwargs):
    """Reuse a pure saved account with explicitly rebound branch labels only."""
    original = e18_accounts()[name]
    function = types.FunctionType(original.__code__, {**original.__globals__, "CASES": CASES},
                                  original.__name__, original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    return function(*args, **kwargs)


def sealed_receipt(*args, **kwargs):
    return generic("sealed_receipt", *args, **kwargs)


def load_act_cut(*args, **kwargs):
    return generic("load_act_cut", *args, **kwargs)


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
                and describe(own) == describe(own_conditioning[lane]["base_conditioning"]), "Act used another lane E or wrong BF16 input")
            mask = torch.zeros_like(own, dtype=torch.bool)
            mask[..., lo:hi] = True
            value = own_conditioning[lane][case]
            expected = own.clone()
            expected[mask] = value[mask]
            require(describe(expected) == describe(value), "replacement changed held H9")
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
            parameters = dict(address="conditioning", value=value, mask=mask, mode="replace", dose=1)
            require(manifest["parameters"] == describe(parameters) and receipt["writes"] == describe({"conditioning": result["conditioning"]}),
                "whole exact-shape Bool mask/value/mode or actual writes differ")
            require(row["exact_shape_mask"] == list(mask.shape) and row["selected_values"] == int(mask.sum())
                and row["conditioning_before"] == describe(own) and row["conditioning_after"] == describe(result["conditioning"])
                and row["replacement"] == describe(value) and row["all_other_slots_unchanged"] is True, "Act observed operand sidecar differs")
            changed = int((own.view(torch.int16) != result["conditioning"].view(torch.int16)).sum())
            require(changed == row["changed_values"], "actual Act changed-bit census differs")
            reports[lane][case] = dict(receipt=receipt["fingerprint"], parent=before.fingerprint, result=after.fingerprint,
                model_identity=after.model_identity, execution=dict(after.execution), columns=columns, selected_values=int(mask.sum()), changed_values=changed,
                H9_exact=True, all_other_slots_exact=True, own_lane_replacement_exact=True)
            cuts[lane][case] = after
    return reports, cuts


def state_loader(*args, **kwargs):
    return generic("state_loader", *args, **kwargs)


def receipt_accounts(*args, **kwargs):
    return generic("receipt_accounts", *args, **kwargs)


def context_accounts(*args, **kwargs):
    return generic("context_accounts", *args, **kwargs)


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
    require(all(row["request_sha256"] == expected_request and row["job_id"] == inventory["job_id"] for row in (source, summary))
        and ("job_id" not in plan or plan["job_id"] == inventory["job_id"]), "actual job/request joins differ")
    require(plan["schema"] == "mdb-flux-h18-state-history-plan-v1" and plan["prompt"] == PROMPT and plan["seed"] == 277
        and plan["branches"] == list(CASES) and plan["conditioning_columns"] == [2560, 7680], "fixed H18 sign-history specimen differs")
    require(request["source_hashes"]["flux_h18_state_history.py"] == source["worker_sha256"] == WORKER_SHA256, "executed producer source differs")
    require(source["helper_source_sha256"] == plan["helper_source_sha256"] and all(request["source_hashes"][key] == value for key, value in source["helper_source_sha256"].items()),
        "source-bound original helper graph differs")
    require(summary["counts"] == contract["counts"] and contract["forecast_module_entries"] == 0 and summary["forecast_seal"] == source["forecast_seal"],
        "actual counts/guard/seal sidecars differ")
    expected = dict(component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=2, functional_transitions=224,
        native_transitions=224, functional_decode_calls=2, native_vae_calls=4, native_qwen_calls=1, act_receipts=4, receipt_occurrences=228, guard_refusals=0,
        qwen_functional_blocks=36, qwen_native_blocks=45, qwen_native_authorizations=10, qwen_functional_SDPA=36, qwen_native_SDPA=45)
    require(all(summary["counts"][key] == value for key, value in expected.items()) and contract["context_operand_occurrences"] == summary["context_operand_occurrences"] == 8,
        "actual bounded operation census differs")
    count_contract = read(root / "count-contract.json")
    require(count_contract == summary["contract"] and all(count_contract[key] == value for key, value in expected.items() if key in count_contract), "declared geometry/count contract differs")
    require(source["helper_source_sha256"]["qwen_h18_state_history.py"] == contract["history_helper_sha256"] == HISTORY_SHA256
        and contract["genuine_qwen_act"] is False, "new component suffix source/Act provenance differs")
    require(plan["history"]["edit_index"] == 18 and plan["history"]["horizon"] == 27
        and plan["history"]["selected"] == [9, 18, 27] and plan["history"]["genuine_qwen_act"] is False, "fixed H18 state history geometry differs")
    require(sha(root / "runtime-contract.json") == source["runtime_contract_sha256"], "actual runtime contract checksum differs")
    return plan, source, summary, contract, inventory


def native_member(name):
    return name.startswith(("qwen-history/native/", "frontend/native/")) or name == "frontend/native.json" or "/native-state-store/" in name \
        or "/decoder/native-own/" in name or "/decoder/native-reference/" in name \
        or Path(name).name in {"native-state-rows.json", "decoder-native-own.json", "decoder-native-reference.json"}


def pretarget_files(root, inventory_names, forecast_acts):
    post = {"native-qwen-history.json", "native-acts.json", "input-lane-native.json", "act-records.json", "context-consumers.json", "native-contract.json", "source.json", "summary.json"}
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
    prediction = old["plain"](load_forecast_tree(tx, read(tx / "forecasts/h18-complete-composition.json")["predictions"]))
    observed = old["plain"](load_forecast_tree(tx, read(tx / "observations/native-own-history-branches.json")["value"]))
    require(prediction["native_targets_at_seal"] == 0 and observed == dict(native_transitions=summary["counts"]["native_transitions"],
        native_vae_calls=summary["counts"]["native_vae_calls"], qwen_native_blocks=summary["counts"]["qwen_native_blocks"], native_act_receipts=2, context_operand_occurrences=summary["context_operand_occurrences"]),
        "cohort forecast-release/observation count differs")
    post = {"native-qwen-history.json", "native-acts.json", "input-lane-native.json", "act-records.json", "context-consumers.json", "native-contract.json", "source.json", "summary.json"}
    declared = prediction["external_files"]
    declarations = read(tx / "seal.json")["declarations"]
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source_keys = tuple(source["source_files"])
    require(declarations["model_identity"] == "flux2-klein-4b-original"
        and declarations["input_identity"] == hashlib.sha256(canonical(dict(prompt=PROMPT, seed=277, intervention=config["intervention"]))).hexdigest()
        and declarations["source_identity"] == hashlib.sha256(canonical({key: config[key + "_sha256"] for key in source_keys})).hexdigest(),
        "sealed model/input/source identity differs")
    require(declarations["frontier"] == dict(kind="raw-input", prompt=PROMPT, seed=277, history=config["history"],
        intervention=config["intervention"], source_inventory=declared), "sealed own raw-input/Act frontier differs")
    require(set(declared) == pretarget_files(root, inventory["files"], prediction["acts"]), "complete pretarget file frontier differs")
    require({name for name in declared if name.startswith("acts-store/")} == act_store_closure(root, prediction["acts"]),
        "forecast Act CAS closure is incomplete or includes native-only future data")
    require(all(not native_member(name) and name not in post for name in declared), "native future evidence present in sealed frontier")
    for name, row in declared.items():
        path = old["member"](root, name)
        require(row == inventory["files"][name] and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "sealed input/prediction dependency differs")
    require(prediction["acts"] == read(root / "forecast-acts.json") and prediction["responses"] == dict(path="forecast-responses.json", sha256=sha(root / "forecast-responses.json"), orientation="coherent-minus-direct"),
        "pretarget actual Acts/complete signed response commitment differs")
    return verified, prediction


def act_store_closure(*args, **kwargs):
    return generic("act_store_closure", *args, **kwargs)


def load_tree(root, value, old):
    if isinstance(value, dict):
        if set(value) == {"tensor"}:
            return old["load_array"](root, value["tensor"])
        return {key: load_tree(root, child, old) for key, child in value.items()}
    if isinstance(value, list):
        return [load_tree(root, child, old) for child in value]
    return value


def tree_equal(left, right, old):
    if hasattr(left, "shape"):
        return hasattr(right, "shape") and old["bytes_equal"](left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(left[k], right[k], old) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(tree_equal(a, b, old) for a, b in zip(left, right, strict=True))
    return left == right


def oddness(plus, minus):
    import numpy as np

    require(plus.shape == minus.shape, "oddness port shapes differ")
    delta = minus.astype(np.float64) + plus.astype(np.float64)
    expected = -plus
    # Every first-suffix port is an archived BF16 tensor decoded losslessly to
    # float32 by the pinned reader. Recover its original BF16 payload bits.
    a = (np.asarray(expected, dtype="<f4").view("<u4") >> 16).astype("<u2")
    b = (np.asarray(minus, dtype="<f4").view("<u4") >> 16).astype("<u2")
    return dict(elements=int(plus.size), numerical_changed=int(np.count_nonzero(delta)),
        max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta**2))),
        bit_changed_elements=int(np.count_nonzero(a != b)),
        differing_bytes=int(np.count_nonzero(a.view(np.uint8) != b.view(np.uint8))),
        shared_zero_elements=int(np.count_nonzero((plus == 0) & (minus == 0))),
        zero_signbit_mismatches=int(np.count_nonzero((plus == 0) & (minus == 0) & (np.signbit(minus) != np.signbit(expected)))),
        status="observation-only")


def writer_account(ports):
    import numpy as np

    result = dict(status="observation-only", extra_learned_projections=0,
        MLP_even_component_max_abs=float(np.abs(ports["plus"]["M"].astype(np.float64) + ports["minus"]["M"].astype(np.float64)).max()))
    for lane, row in ports.items():
        remainder = row["Hnext"].astype(np.float64) - row["R"].astype(np.float64) - row["M"].astype(np.float64)
        result[lane + "_finite_addition_remainder"] = dict(max_abs=float(np.abs(remainder).max()), rms=float(np.sqrt(np.mean(remainder**2))))
    result["oddness"] = {name: oddness(ports["plus"][name], ports["minus"][name]) for name in sorted(ports["plus"].keys() & ports["minus"].keys())}
    return result


def signed_descriptor(direct, coherent):
    import numpy as np

    delta = np.asarray(coherent, np.float64) - np.asarray(direct, np.float64)
    return dict(shape=list(delta.shape), elements=int(delta.size), dtype="float64", orientation="coherent-minus-direct",
                sha256=hashlib.sha256(np.asarray(delta, dtype="<f8").tobytes(order="C")).hexdigest())


def qwen_accounts(root, fq, nq, config, old, q, qconfig):
    import numpy as np

    selected = config["history"]["selected"]
    first, edit, horizon = selected
    full_layers, width = qconfig["num_hidden_layers"], qconfig["hidden_size"]
    require(config["history"]["edit_index"] == edit and config["history"]["horizon"] == horizon and 0 < first < edit < horizon <= full_layers,
            "H18 history indexing differs")
    records = dict(forecast=fq, native=nq)
    values, fields, ports = {}, {}, {}
    for lane, row in records.items():
        require(row["schema"] == "mdb-qwen-h18-history-record-v1" and row["lane"] == lane and row["genuine_qwen_act"] is False
            and row["selected"] == selected and row["parent_conditioning_concats"] == 1, "history source/Act declaration differs")
        require(all(record["logical_dtype"] == "torch.bfloat16" for family in (row["common"], row["minus"], row["conditioning"]) for record in family.values()), "history learned residual/conditioning dtype differs")
        require(all(record["logical_dtype"] == "torch.bfloat16" for sign in row["first_suffix_ports"].values() for record in sign.values()), "first suffix port dtype differs")
        common = {int(k): old["load_array"](root, record) for k, record in row["common"].items()}
        minus = {int(k): old["load_array"](root, record) for k, record in row["minus"].items()}
        require(set(common) == set(range((horizon if lane == "forecast" else full_layers) + 1))
            and set(minus) == set(range(edit, horizon + 1)), "complete own history coverage differs")
        ids, mask = (old["load_array"](root, row[key]) for key in ("input_ids", "attention_mask"))
        require(ids.shape == mask.shape == (1, config["max_length"]) and np.all((mask == 0) | (mask == 1)), "history token/mask geometry differs")
        require(all(v.shape == (*ids.shape, width) for v in list(common.values()) + list(minus.values())), "history residual geometry differs")
        require(old["bytes_equal"](-common[edit], minus[edit]), "edited H18 is not exact own negation")
        own = {name: old["load_array"](root, record) for name, record in row["conditioning"].items()}
        recipes = dict(base_conditioning=[common[first], common[edit], common[horizon]],
                       direct=[common[first], minus[edit], common[horizon]], coherent=[common[first], minus[edit], minus[horizon]])
        require(set(own) == set(recipes) and all(old["bytes_equal"](np.concatenate(parts, axis=-1), own[name]) for name, parts in recipes.items()),
                "conditioning is not own selected history")
        require(old["bytes_equal"](own["direct"][..., :2 * width], own["coherent"][..., :2 * width]), "held H9/negativeH18 differ across branches")
        require(row["conditioning_response"] == signed_descriptor(own["direct"], own["coherent"])
            and row["history_response"] == {str(k): signed_descriptor(common[k], minus[k]) for k in minus}, "complete Qwen history response commitment differs")
        ports[lane] = {sign: {name: old["load_array"](root, record) for name, record in port.items()} for sign, port in row["first_suffix_ports"].items()}
        require(set(ports[lane]) == {"plus", "minus"}, "first suffix sign coverage differs")
        for sign, port in ports[lane].items():
            require({"h", "norm", "projection", "R", "postnorm", "gate", "up", "activation", "product", "M", "Hnext"} <= port.keys(), "first suffix port coverage differs")
            origin, exit_value = (common[edit], common[edit + 1]) if sign == "plus" else (minus[edit], minus[edit + 1])
            require(old["bytes_equal"](port["h"], origin) and old["bytes_equal"](port["Hnext"], exit_value), "first suffix port/exit joins differ")
        values[lane] = dict(common=common, minus=minus, own=own, ids=ids, mask=mask)
    require(old["bytes_equal"](values["forecast"]["ids"], values["native"]["ids"]) and old["bytes_equal"](values["forecast"]["mask"], values["native"]["mask"]), "lane parser inputs differ")
    positions = np.arange(config["max_length"], dtype=np.int64)[None]
    require(old["bytes_equal"](old["load_array"](root, fq["position_ids"]), positions), "forecast position indexing differs")
    require(fq["mask_profile"]["has_padding"] is True and fq["mask_profile"]["shape"] == [1, 1, config["max_length"], config["max_length"]], "original padded forecast dispatch geometry differs")
    require(fq["profile"]["edit_index"] == edit and fq["profile"]["horizon"] == horizon and fq["profile"]["selected"] == selected
        and fq["profile"]["attention_dispatch"] == config["attention_dispatch"] and fq["profile"]["rope_buffer_policy"] == config["rope_buffer_policy"], "forecast finite profile differs")
    blocks = 2 * horizon - edit
    require(fq["counts"]["blocks"] == blocks and fq["counts"]["SDPA"] == blocks and fq["counts"]["linear"] == 7 * blocks
        and fq["counts"]["RMSNorm"] == 4 * blocks and fq["counts"]["native_module_forwards"] == 0, "functional executed group census differs")
    context = load_tree(root, nq["source_context"], old)
    require(context.get("past_key_values") is None and not context.get("use_cache", False) and old["bytes_equal"](context["position_ids"], positions), "native source context/cache/position differs")
    mask = values["native"]["mask"]
    expected_mask = np.tril(np.ones((config["max_length"], config["max_length"]), dtype=bool))[None, None] & mask[:, None, None, :].astype(bool)
    require(context["attention_mask"].dtype == bool and old["bytes_equal"](context["attention_mask"], expected_mask), "native actual padding/causal context differs")
    require(len(context["position_embeddings"]) == 2 and all(v.shape == (*mask.shape, qconfig["head_dim"]) for v in context["position_embeddings"]), "native rotary query geometry differs")
    require([row["index"] for row in nq["consumed"]] == list(range(edit, horizon)), "native suffix consumed occurrence indexing differs")
    for row in nq["consumed"]:
        require(old["bytes_equal"](old["load_array"](root, row["hidden"]), values["native"]["minus"][row["index"]])
            and row["context_source"] == "source_context" and tree_equal(load_tree(root, row["context"], old), context, old), "actual suffix did not consume own prior edited exit/context")
    expected_entries = [dict(lane="baseline", index=i) for i in range(full_layers)] + [dict(lane="suffix", index=i) for i in range(edit, horizon)]
    require(nq["block_entries"] == expected_entries and len(nq["sdpa_calls"]) == len(expected_entries)
        and [(r["lane"], r["index"]) for r in nq["sdpa_calls"]] == [(r["lane"], r["index"]) for r in expected_entries], "native actual full baseline/suffix SDPA census differs")
    for row in nq["sdpa_calls"]:
        require(row["is_causal"] is False and row["enable_gqa"] is False and row["mask_shape"] == [1, 1, config["max_length"], config["max_length"]]
            and all(v["shape"] == [1, qconfig["num_attention_heads"], config["max_length"], qconfig["head_dim"]] for v in row["QKV"]), "actual native SDPA argument geometry/flags differ")
    require(nq["counts"]["full_model_calls"] == 1 and nq["counts"]["baseline_blocks"] == full_layers and nq["counts"]["suffix_blocks"] == horizon - edit
        and nq["counts"]["SDPA"] == len(expected_entries) and nq["counts"]["native_module_entries"] == len(nq["module_entries"]), "native execution counter joins differ")
    for name in ("", "lm_head", "model.embed_tokens", "model.rotary_emb"):
        require(sum(row == dict(lane="baseline", name=name) for row in nq["module_entries"]) == 1, "full baseline LM/embedding/RoPE module occurrence differs")
    require(nq["baseline_logits_shape"] == [1, config["max_length"], qconfig["vocab_size"]], "full native baseline LM readout missing")
    for kind in ("common", "minus"):
        fields[kind] = {str(i): q["row_accounts"](values["forecast"][kind][i], values["native"][kind][i], mask) for i in values["forecast"][kind]}
    fields["conditioning"] = {name: q["row_accounts"](values["forecast"]["own"][name], values["native"]["own"][name], mask) for name in values["forecast"]["own"]}
    comparable = {}
    groups = qconfig["num_attention_heads"] // qconfig["num_key_value_heads"]
    for sign in ("plus", "minus"):
        aligned = dict(ports["forecast"][sign])
        aligned["q_consumed"] = aligned.pop("q")
        aligned["k_consumed"] = np.repeat(aligned.pop("k"), groups, axis=1)
        aligned["v_consumed"] = np.repeat(aligned.pop("v"), groups, axis=1)
        aligned["head_consumed"] = aligned.pop("head").transpose(0, 2, 1, 3)
        require(aligned.keys() == ports["native"][sign].keys(), "declared first-suffix port basis coverage differs")
        comparable[sign] = {name: old["metric"](aligned[name], ports["native"][sign][name]) for name in aligned}
    first_miss = next((f"{kind}/{index}" for kind in ("common", "minus", "conditioning") for index, report in fields[kind].items() if not report["full"]["bitwise_exact"]), None)
    return dict(fields=fields, first_bit_discrepancy=first_miss, first_suffix=comparable,
        oddness={lane: writer_account(port) for lane, port in ports.items()}, recorded_writer_diagnostic={lane: records[lane]["writer_diagnostic"] for lane in records},
        native_context=dict(consumed_occurrences=len(nq["consumed"]), exact_own_prior_exit=True, same_source_context=True, use_cache=False),
        first_suffix_first_bit_discrepancy={sign: next((name for name in ("h", "norm", "q_consumed", "k_consumed", "v_consumed", "head_consumed", "projection", "R", "postnorm", "gate", "up", "activation", "product", "M", "Hnext") if not comparable[sign][name]["bitwise_exact"]), None) for sign in comparable},
        native_reference_only_exits=list(range(horizon + 1, full_layers + 1)),
        baseline_LM_scope="Actual source-witnessed module entry and logits shape; logits values are not retained or independently forecast.",
        native_counts=nq["counts"], functional_counts=fq["counts"], genuine_qwen_act=False,
        port_alignment="Only declared repeated-KV and head transpose views align functional ports to actual native SDPA operands; no learned or attention operation executed.",
        source_history_response_orientation_note="Raw history_response labels are preserved. Their actual operands are minus[k]-common[k] (edited-minus-baseline); conditioning_response and downstream responses are coherent-minus-direct.",
        scope=f"Own{horizon}+edited{horizon-edit} forecast versus full{full_layers} native baseline+own{horizon-edit}-layer component suffix. Not a Qwen Act or an MLP-writer ablation. Oddness/SwiGLU/addition remainders are diagnostics, not new predictions."), {lane: v["own"] for lane, v in values.items()}, mask


def parser_sources(*args, **kwargs):
    return generic("parser_sources", *args, **kwargs)


def frontend_accounts(*args, **kwargs):
    return generic("frontend_accounts", *args, **kwargs)


def tensor_leaves(*args, **kwargs):
    return generic("tensor_leaves", *args, **kwargs)


def response_account(values, sealed, old):
    import numpy as np

    deltas = {lane: np.asarray(values[lane]["coherent"], np.float64) - np.asarray(values[lane]["direct"], np.float64) for lane in ("forecast", "native")}
    predicted = deltas["forecast"]
    expected = dict(shape=list(predicted.shape), elements=int(predicted.size), dtype="float64", orientation="coherent-minus-direct",
        sha256=hashlib.sha256(np.asarray(predicted, dtype="<f8").tobytes(order="C")).hexdigest())
    require(sealed == expected, "complete pretarget signed response hash differs")
    return dict(forecast={**old["metric"](values["forecast"]["coherent"], values["forecast"]["direct"]), "orientation": "coherent-minus-direct"},
        native={**old["metric"](values["native"]["coherent"], values["native"]["direct"]), "orientation": "coherent-minus-direct"}, signed_response_error=old["metric"](deltas["forecast"], deltas["native"]),
        prospective_full_hash_verified=True, sealed=sealed,
        scope="Coherent-minus-direct H27-history response conditional on negative H18; targets/collaterals/continuations share state and are dependent observations.")


def coverage_account(qreport, cases, responses):
    def count(rows):
        return dict(fields=len(rows), positions=sum(row["sample_count"] for row in rows),
                    bit_exact_fields=sum(bool(row["bitwise_exact"]) for row in rows))
    qrows = [row for family in qreport["fields"].values() for row in family.values()]
    primary_qwen = [row["full"] for row in qrows]
    regions = [row[name]["comparison"] for row in qrows for name in ("attended_query_rows", "padded_query_rows") if row[name]["comparison"] is not None]
    denoiser = [field for case in cases.values() for row in case["denoiser"]["boundaries"] for field in row["fields"].values()]
    ports = [row for sign in qreport["first_suffix"].values() for row in sign.values()]
    response = [field["signed_response_error"] for row in responses["denoiser"] for field in row["fields"].values()]
    return dict(primary_qwen_whole_fields=count(primary_qwen), primary_denoiser_whole_fields=count(denoiser),
                qwen_query_region_diagnostics=count(regions), aligned_first_suffix_port_comparisons=count(ports),
                signed_denoiser_response_fields=count(response), oddness="Separate per-lane diagnostic accounts; not included in whole-field comparison counts.",
                scope="Overlapping/alias/dependent fields and regions are explicitly separate; counts are coverage, not independent measurements or new transitions.")


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
    fq, nq = (read(root / (lane + "-qwen-history.json")) for lane in ("forecast", "native"))
    qconfig, fconfig = read(metadata["qwen_config"]), read(metadata["flux_config"])
    qreport, own_np, mask = qwen_accounts(root, fq, nq, config, old, q, qconfig)
    require(config["conditioning_columns"] == [qconfig["hidden_size"], 3 * qconfig["hidden_size"]], "direct middle interior-state slice differs")
    own = {lane: {name: torch.from_numpy(value.copy()).to(torch.bfloat16) for name, value in values.items()} for lane, values in own_np.items()}
    act_report, cuts = act_accounts(root, acts, own, columns=config["conditioning_columns"])
    fronts = {}
    for lane in ("forecast", "native"):
        declared = read(root / ("input-lane-" + lane + ".json"))
        require(declared["branches"] == acts[lane] and declared["own_conditioning"] == {name: describe(value) for name, value in own[lane].items()}, "own E/input-lane/Act declaration differs")
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
    require(sealed["schema"] == "mdb-h18-history-signed-response-v1" and sealed["orientation"] == "coherent-minus-direct"
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
                for role, selected_slice in (("H9-held", slice(0, lo)), ("negative-H18-held", slice(lo, 2 * lo)), ("H27-history-contrast", slice(2 * lo, hi))):
                    sub = {lane: {case: numeric[lane][case][..., selected_slice] for case in CASES} for lane in numeric}
                    routes[role] = {lane: {**old["metric"](sub[lane]["coherent"], sub[lane]["direct"]), "orientation": "coherent-minus-direct"} for lane in sub}
                fields[name]["declared_target_and_collateral_routes"] = routes
        responses["denoiser"].append(dict(number=sealed_row["number"], clock=sealed_row["clock"], fields=fields))
    for field in ("decoded", "pixels"):
        values = {lane: {case: endpoints[case]["forecast" if lane == "forecast" else "native-reference"][field] for case in CASES} for lane in ("forecast", "native")}
        responses["decoder"][field] = response_account(values, sealed["decoder"][field], old)
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    kinds = [row["kind"] for row in events]
    require(kinds.count("forecast_globally_sealed") == kinds.count("native_targets_released") == 1 and kinds.count("native_qwen_authorized") == 10
        and kinds.index("forecast_globally_sealed") < kinds.index("native_targets_released") < kinds.index("native_qwen_authorized"), "global pretarget chronology differs")
    authorizations = [row for row in events if row["kind"] == "native_qwen_authorized"]
    require([row["occurrence"] for row in authorizations] == list(range(1, 11))
        and all(row["forecast_seal"] == summary["forecast_seal"]["seal_sha256"] for row in authorizations), "actual native Qwen release/source-context chronology differs")
    seal_event = events[kinds.index("forecast_globally_sealed")]
    require(seal_event["native_module_entries"] == 0 and seal_event["forecast_act_receipts"] == 2 and seal_event["functional_transitions"] == 224
        and seal_event["qwen_functional_blocks"] == 36 and seal_event["seal_sha256"] == summary["forecast_seal"]["seal_sha256"], "actual pretarget guard/Act census differs")
    inputs = read(root / "input.json")
    require(inputs["prompt"] == PROMPT and inputs["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
        and inputs["attended_rows"] == int(mask.sum()) and inputs["padded_rows"] == int(mask.size - mask.sum()), "raw prompt/saved tokenized row joins differ")
    result = dict(schema="mdb-flux-h18-state-history-saved-audit-v1", job_id=inventory["job_id"], request_sha256=expected_request,
        raw_manifest_sha256=expected_manifest, raw_members=len(inventory["files"]), raw_expanded_bytes=inventory["expanded_bytes"], source=source,
        parameters=restoration, runtime=runtime, parser=parser, transaction=verification, input=inputs, qwen=qreport, acts=act_report, receipt_account=receipts,
        context_consumers=consumers, frontend=frontend, cases=case_reports, signed_responses=responses, actual_counts=summary["counts"],
        public_runtime_provenance=dict(runtime_wheels=read(root / "request.json").get("runtime_wheels"),
            execution_environment=read(root / "request.json").get("execution_environment"),
            actual_adapter_execution=act_report["native"]["direct"]["execution"], actual_model_identity=act_report["native"]["direct"]["model_identity"],
            scope="Actual source request/runtime pins plus typed Act/native-cut execution identities. Functional children preserve this adapter identity without inventing native receipts or Session interchangeability."),
        mechanics_status="authenticated-genuine-Acts-and-own-state-continuations", trend_status="observed", terminal_status="unassessed",
        new_model_calls=0, new_tokenizer_calls=0, scope="Negative H18 direct route versus regenerated H27 history in the edited recipient; state intervention, not an MLP-writer ablation or full factorial.",
        limitations=["Only four native apply receipts and224 native advance receipts are counted as genuine occurrences. Functional112 transitions perbranch are component-derived StateCuts.",
            "Own E_forecast and E_native are formed separately and patched within their own lane. Equality of values, if observed, does not prove crossinstallation or independence.",
            "Actual8 native context_embedder full-operand prehook descriptors join archived native Act result bytes; source descriptors are not a new kernel measurement.",
            "Qwen FP32 RoPE buffer policy is bound through pinned helper/source guards; actual cosine/sine and consumed context arrays are archived. No separately archived inverse-frequency buffer is asserted.",
            "Raw prompt/tokenizer files and saved tokenized inputs are authenticated; no tokenizer executed by this reader. No independent parser/RNG/attention kernel claim.",
            "Targets, held H9/negative-H18/noise and downstream H27/continuations are reported perrow. No automatic family induction or scientific pass/fail threshold."])
    result["coverage"] = coverage_account(qreport, case_reports, responses)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    catalog = EquationCatalog(output / "catalog/equations")
    queries = []
    for case in CASES:
        for lane, occurrence in (("native-reference", "whole"), ("native-own", "same-latent-decoder")):
            context = dict(branch=case, occurrence=occurrence, prompt_sha256=inputs["prompt_sha256"], seed=277, history=config["history"],
                clock=dict(step=0, phase="project", cursor=0), intervention=config["intervention"], native_act=acts["native"][case]["receipt"]["fingerprint"],
                forecast_act=acts["forecast"][case]["receipt"]["fingerprint"], forecast_seal=summary["forecast_seal"]["seal_sha256"], request_sha256=expected_request)
            record = catalog.record(law_id="law.flux.H18-state-history.complete-program.v1", route_id="route.flux.original-shared-primitives.H18-state-history.v1",
                alias="cmp.flux.H18-history." + case + "." + occurrence,
                binding=dict(model_identity=cuts["native"][case].model_identity, execution=dict(id="mdb-flux-h18-state-history-v1"),
                    input_identity=acts["forecast"][case]["after"]["cut"], source_identity=expected_request, frontier="own raw-Qwen E then actual initial-project conditioning Act", horizon=dict(steps=4, decoder=occurrence)),
                report=old["catalog_comparison"](endpoints[case]["forecast"], endpoints[case][lane]),
                conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
                evidence=dict(audit_sha256=sha(output / "audit.json"), genuine_flux_acts=True, genuine_qwen_act=False, independent_kernel=False, full_factorial=False))
            exact = len(catalog.find(applicable_to=context))
            wrong = len(catalog.find(applicable_to={**context, "clock": dict(step=1, phase="project", cursor=0)}))
            require(exact == 1 and wrong == 0, "catalog context/wrong-clock abstention differs")
            queries.append(dict(fingerprint=record["fingerprint"], context=context, exact=exact, wrong_clock=wrong))
    save(output / "catalog/provenance.json", dict(records=4, queries=queries, automatic_family_induction=False,
        scope="Dependent exact-context endpoint accounts of H27 history conditional on held negative H18; no full-factorial or semantic attribution."))
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
    for name in ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json", "files.json", "runtime-contract.json", "input.json", "count-contract.json",
        "forecast-qwen-history.json", "native-qwen-history.json", "act-records.json", "forecast-acts.json", "native-acts.json", "context-consumers.json", "receipt-emissions.jsonl", "timeline.jsonl", "forecast-responses.json"):
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
    save(output / "summary.json", dict(schema="mdb-flux-h18-state-history-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
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
