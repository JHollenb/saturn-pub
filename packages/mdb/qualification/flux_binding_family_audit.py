"""Saved-only independent reader for producer/continuation family assays.

Authenticates every published member and rechecks full arrays, boundary hashes,
source return closure, interventions and prospective route inequalities. It
loads no pretrained parameters and performs no learned forward.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import sys
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def metric(left, right):
    a, b = (np.asarray(value, dtype=np.float64) for value in (left, right))
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "finite comparison shape differs")
    delta = a-b
    ar, br = float(np.sqrt(np.mean(a*a))), float(np.sqrt(np.mean(b*b)))
    return dict(rms=float(np.sqrt(np.mean(delta*delta))), max_abs=float(np.max(np.abs(delta))),
        changed=int(np.count_nonzero(delta)), elements=a.size, candidate_rms=ar, reference_rms=br,
        cosine=float(np.mean(a*b)/(ar*br)) if ar and br else None,
        projection=float(np.mean(a*b)/(br*br)) if br else None)


def decode_bits(value):
    return (value.astype(np.uint32) << 16).view(np.float32)


def bf(value):
    x = np.asarray(value, dtype=np.float32)
    bits = x.view(np.uint32)
    return decode_bits(((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) >> 16).astype(np.uint16))


def audit(config):
    class NoModelFramework(importlib.abc.MetaPathFinder):
        roots = {"diffusers", "transformers", "tensorflow", "jax"}

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in self.roots:
                raise ImportError("Saved-only family audit refuses model frameworks")

    require(not NoModelFramework.roots.intersection(sys.modules), "saved-only process already contains a model framework")
    sys.meta_path.insert(0, NoModelFramework())
    import torch
    from saturn_pub.core import StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe, digest
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), "saved reader initialized CUDA")
    def reject_forward(_module, _args):
        raise RuntimeError("Saved-only family audit refuses every nn.Module forward")
    forward_guard = torch.nn.modules.module.register_module_forward_pre_hook(reject_forward)
    from mdb.equation_comparison import compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction
    root = Path(config["native_root"]).resolve()
    manifest_path = root / "files.json"
    require(sha(manifest_path) == config["native_manifest_sha256"], "publication inventory differs")
    manifest = read(manifest_path)
    require(manifest["request_sha256"] == config["native_request_sha256"] and manifest["job_id"] == config["native_job_id"], "publication owner differs")
    files = manifest["files"]
    for relative, record in files.items():
        path = root / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and not path.is_symlink()
            and path.resolve().is_relative_to(root), "unsafe inventoried member")
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "published member changed: " + relative)
    plan, request, summary = (read(root / name) for name in ("plan.json", "request.json", "summary.json"))
    require(summary["schema"] == "mdb-flux-binding-family-v1" and summary["request_sha256"] == config["native_request_sha256"], "summary identity differs")
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"], "executed request changed")
    require({k: v for k, v in plan.items() if k not in ("job_id", "request_sha256", "output")} == request["config"], "executed plan/config differs")
    transaction = verify_transaction(root / "forecast-transaction", expected_seal_sha256=summary["forecast_seal"], expected_event_head=summary["transaction"]["event_head"])
    descriptor = read(root / "forecast-transaction/forecasts/all-numeric-futures.json")["predictions"]
    forecast = load_forecast_tree(root / "forecast-transaction", descriptor)
    require(forecast["native_targets_at_seal"] == forecast["native_source_blocks_at_seal"] == 0, "targets before seal")
    external = forecast["external_files"]
    for relative, record in external.items():
        require(relative in files and record == files[relative], "sealed external file no longer matches publication")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    repeated_elements, unique_arrays, fields_compared = 0, set(), 0

    def array(base, record, *, raw=False):
        nonlocal repeated_elements
        path = base / record["path"]
        relative = path.relative_to(root).as_posix()
        require(relative in files and sha(path) == record["sha256"], "array descriptor differs")
        value = np.load(path, allow_pickle=False)
        require(list(value.shape) == record["shape"], "array shape differs")
        digest = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        if "content_sha256" in record:
            require(digest == record["content_sha256"], "array content differs")
        unique_arrays.add(relative)
        repeated_elements += value.size
        return value if raw else decode_bits(value) if record.get("encoding") == "bf16-bits-v1" else value

    def compare_array(base_a, a, base_b, b):
        nonlocal fields_compared
        raw_a, raw_b = array(base_a, a, raw=True), array(base_b, b, raw=True)
        require(raw_a.dtype == raw_b.dtype and raw_a.shape == raw_b.shape, "full array ABI differs")
        require(all(a.get(name) == b.get(name) for name in ("encoding", "logical_dtype", "stored_dtype")), "array interpretation ABI differs")
        fields_compared += 1
        return dict(raw_bit_exact=raw_a.tobytes() == raw_b.tobytes(), **metric(array(base_a, a), array(base_b, b)))

    boundary_counts = dict(tensors=0, scalar_fields=0, hash_mismatches=0, layout_mismatches=0)

    def boundary(a, b):
        if isinstance(a, dict) and "tensor" in a:
            require(isinstance(b, dict) and "tensor" in b, "boundary tensor shape differs")
            boundary_counts["tensors"] += 1
            left, right = a["tensor"], b["tensor"]
            boundary_counts["hash_mismatches"] += left["sha256"] != right["sha256"]
            boundary_counts["layout_mismatches"] += any(left[k] != right[k] for k in ("shape", "dtype", "stride"))
        elif isinstance(a, dict):
            require(isinstance(b, dict) and set(a) == set(b), "boundary field census differs")
            for name in a:
                boundary(a[name], b[name])
        elif isinstance(a, list):
            require(isinstance(b, list) and len(a) == len(b), "boundary sequence differs")
            for x, y in zip(a, b, strict=True):
                boundary(x, y)
        else:
            boundary_counts["scalar_fields"] += 1
            require(a == b, "boundary scalar state differs")

    source_roots = read(root / "source-roots.json")
    require(set(source_roots) == set(plan["contexts"]), "declared context census differs")
    writers = {lane: read(root / (lane + "-source-writers.json")) for lane in ("forecast", "native")}
    store = LocalStore(root / "state-store")
    writer_checks, endpoint_checks, recomputed_checks, source_joins, construction_checks = {}, {}, {}, {}, {}

    def numpy_tensor(value):
        return value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()

    def same_descriptor(value, record):
        descriptor = describe(value)
        require(descriptor["sha256"] == record["content_sha256"] and descriptor["shape"] == record["shape"]
            and descriptor["stride"] == record["stride"] and descriptor["dtype"] == record["logical_dtype"], "port/cut operand join differs")

    def legacy_ports(cid, arm, lane):
        prefix = f"writer/{cid}/{arm}/{lane}/"
        records = {}
        for name in files:
            if not name.startswith(prefix) or not name.endswith(".npy"):
                continue
            value = np.load(root / name, allow_pickle=False)
            records[Path(name).stem] = dict(path=name, sha256=files[name]["sha256"], shape=list(value.shape),
                encoding="bf16-bits-v1" if value.dtype == np.uint16 else "native-numpy-v1")
        return records

    for cid in source_roots:
        writer_checks[cid] = {}
        source_joins[cid] = {}
        inputs = {role: store.load(record["cut"]).payload for role, record in source_roots[cid]["inputs"].items()}
        require(describe({k: v for k, v in inputs["P"].items() if k != "text"}) == describe({k: v for k, v in inputs["Q"].items() if k != "text"}), "source inputs differ outside text")
        outside = sorted(set(range(inputs["Q"]["text"].shape[1])) - set(source_roots[cid]["support"]))
        require(torch.equal(inputs["P"]["text"][:, outside].contiguous().view(torch.uint8),
            inputs["Q"]["text"][:, outside].contiguous().view(torch.uint8)), "source text bits differ outside frozen support")
        for role in ("Q", "P"):
            fp, np_ = (writers[lane][cid]["sources"][role]["ports"] for lane in ("forecast", "native"))
            require(set(fp) == set(np_), "writer port census differs")
            comparisons = {name: compare_array(root, fp[name], root, np_[name]) for name in fp}
            ports = {name: array(root, record) for name, record in np_.items()}
            u = bf(ports["text_in"] + bf(ports["text_attention_gate"] * ports["text_attention_return_used"]))
            t = bf(u + bf(ports["text_mlp_gate"] * ports["text_mlp_return_used"]))
            writer_checks[cid][role] = dict(comparisons=comparisons,
                explicit_attention_landing=metric(u, ports["text_after_attention"]), explicit_mlp_landing=metric(t, ports["text_out"]))
            source_joins[cid][role] = {}
            for lane in ("forecast", "native"):
                source = writers[lane][cid]["sources"][role]
                cut = store.load(source["state"]["cut"])
                same_descriptor(inputs[role]["text"], source["ports"]["text_in"])
                same_descriptor(cut.payload["text"], source["ports"]["text_out"])
                require((cut.payload["step"], cut.payload["phase"], cut.payload["cursor"]) == (2, "joint", 1), "source writer clock differs")
                source_joins[cid][role][lane] = dict(input_cut=source_roots[cid]["inputs"][role]["cut"], output_cut=cut.fingerprint)
        endpoint_checks[cid] = {arm: compare_array(root, record, root, writers["native"][cid]["endpoints"][arm])
            for arm, record in writers["forecast"][cid]["endpoints"].items()}
        recomputed_checks[cid] = {}
        for arm in ("QPM", "PQM"):
            if arm not in plan["arms"]:
                continue
            ports = {lane: writers[lane][cid].get("recomputed", {}).get(arm) or legacy_ports(cid, arm, lane) for lane in ("forecast", "native")}
            require(ports["forecast"] and set(ports["forecast"]) == set(ports["native"]), "recomputed MLP port census differs")
            recomputed_checks[cid][arm] = {name: compare_array(root, ports["forecast"][name], root, ports["native"][name]) for name in ports["forecast"]}
        q, p = ({name: array(root, value) for name, value in writers["native"][cid]["sources"][role]["ports"].items()} for role in ("Q", "P"))
        require(np.array_equal(q["text_attention_gate"], p["text_attention_gate"]) and np.array_equal(q["text_mlp_gate"], p["text_mlp_gate"]), "crossed writers do not share gates")
        rows = source_roots[cid]["support"]
        construction_checks[cid] = {}
        for arm, record in writers["native"][cid]["endpoints"].items():
            expected = q["text_out"].copy()
            if arm == "whole_P":
                expected = p["text_out"].copy()
            elif arm == "complement_P":
                expected = p["text_out"].copy(); expected[:, rows] = q["text_out"][:, rows]
            elif arm == "wrong_address":
                expected[:, source_roots[cid]["wrong_rows"]] = p["text_out"][:, rows]
            elif arm in ("QPM", "PQM"):
                rec = writers["native"][cid].get("recomputed", {}).get(arm) or legacy_ports(cid, arm, "native")
                expected[:, rows] = array(root, rec["text_out"])[:, rows]
            else:
                values = {"Q": q, "P": p}
                u = bf(values[arm[0]]["text_in"] + bf(q["text_attention_gate"] * values[arm[1]]["text_attention_return_used"]))
                t = bf(u + bf(q["text_mlp_gate"] * values[arm[2]]["text_mlp_return_used"]))
                expected[:, rows] = t[:, rows]
            observed = array(root, record)
            require(np.array_equal(expected, observed), "independent endpoint construction differs: " + cid + "/" + arm)
            construction_checks[cid][arm] = metric(expected, observed)

    index = read(root / "state-index.json")
    consumers = read(root / "consumer-ports.json")
    expected_cases = {cid + "/" + arm for cid in source_roots for arm in plan["arms"]}
    require(set(index) == expected_cases and set(consumers) == {"forecast", "native"}
        and all(set(consumers[lane]) == expected_cases for lane in consumers), "case/consumer census differs")
    expected_ports = {"M_base", "R_attention", "gate_ff", "image_ln", "text_ln", "x"}
    require(all(set(case) == expected_ports for lane in consumers.values() for case in lane.values()), "consumer port census differs")
    cases, images, act_checks, known_checks, equation_reports, all_images = {}, {}, {}, {}, {}, {lane: {} for lane in ("forecast", "native")}

    def check_act(record, slot, operand, *, common=None):
        receipt = record["receipt"]
        require(digest({k: v for k, v in receipt.items() if k != "fingerprint"}) == receipt["fingerprint"], "Act receipt changed")
        after = store.load(record["after"]["cut"])
        require(receipt["result"] == after.fingerprint and receipt["parent"] == after.parent, "Act receipt/cut lineage differs")
        require(receipt["act"]["writes"] == receipt["act"]["reads"] == [slot] and not receipt["act"]["invalidates"], "Act address differs")
        require(receipt["writes"] == describe({slot: after.payload[slot]}), "Act writes differ from resulting cut")
        parameters = {item["key"]["value"]: item["value"] for item in receipt["act"]["parameters"]["items"]}
        require(parameters["value"] == describe(after.payload[slot]) and parameters["mode"]["value"] == "replace"
            and parameters["dose"]["value"] == 1, "Act parameters differ")
        same_descriptor(after.payload[slot], operand)
        parent_path = store.root / "cuts" / (receipt["parent"] + ".json")
        before = store.load(receipt["parent"]) if parent_path.exists() else None
        coverage = "stored-parent"
        legacy_smoke = config["native_request_sha256"] == "8020269f98247d16a308dc9bcc940bc3c41767e311330e1add286958cdaa7648"
        require(before is not None or legacy_smoke, "required pre-Act cut is absent")
        if before is None and common is not None:
            parent = store.load(common["cut"])
            before = StateCut(parent.model_identity, parent.execution, parent.boundary, parent.fingerprint,
                parent.payload, parent.execution_point, parent.surface)
            require(before.fingerprint == receipt["parent"], "reconstructed public fork parent differs")
            coverage = "reconstructed-immutable-common-parent"
        if before is not None:
            require(describe({k: v for k, v in before.payload.items() if k != slot}) == describe({k: v for k, v in after.payload.items() if k != slot}), "Act changed undeclared state")
            require(receipt["act"]["preconditions"] == {slot: digest(describe(before.payload[slot]))}, "Act precondition differs")
        else:
            coverage = "missing-parent-historical-smoke-only"
        return dict(slot=slot, parent=receipt["parent"], result=after.fingerprint, coverage=coverage)

    def payload_tree(value):
        if isinstance(value, torch.Tensor):
            return dict(tensor=describe(value))
        if isinstance(value, dict):
            return {k: payload_tree(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [payload_tree(v) for v in value]
        return value

    for key, record in index.items():
        branch = root / record["root"]
        fr, nr = (read(branch / (lane + "-rows.json")) for lane in ("forecast", "native"))
        require(len(fr) == len(nr) == 55, "full continuation boundary count differs")
        for f, n in zip(fr, nr, strict=True):
            require(f["number"] == n["number"], "boundary order differs")
            boundary(f["fields"], n["fields"])
        fd, nd = (read(branch / name) for name in ("decoder-forecast.json", "decoder-native-reference.json"))
        comparisons = {name: compare_array(branch, fd[name], branch, nd[name]) for name in ("decoded", "pixels", "decoder_input")}
        ports = {name: compare_array(root, consumers["forecast"][key][name], root, actual) for name, actual in consumers["native"][key].items()}
        cid, arm = record["context"], record["arm"]
        act_checks[key] = {}
        for lane in ("forecast", "native"):
            entry, reset = (read(branch / (lane + "-" + name + "-act.json")) for name in ("entry", "reset"))
            act_checks[key][lane] = dict(entry=check_act(entry, "text", writers[lane][cid]["endpoints"][arm], common=entry["common"]),
                reset=check_act(reset, "conditioning", source_roots[cid]["base_conditioning"]))
        if arm in ("QQQ", "PPP") and source_roots[cid].get("known"):
            known = source_roots[cid]["known"]["whole_Q" if arm == "QQQ" else "both_RW"]
            require(record["native_final"]["cut"] == known["cut"] and nd["png"]["sha256"] == known["png_sha256"], "historical native endpoint witness differs")
            known_checks[key] = dict(final_cut=known["cut"], png_sha256=known["png_sha256"], exact=True)
        for lane, decoder in (("forecast", fd), ("native", nd)):
            all_images[lane][key] = {name: array(branch, decoder[name]).astype(np.float64) for name in ("decoded", "pixels")}
            final = store.load(record[lane + "_final"]["cut"])
            boundary((fr if lane == "forecast" else nr)[-1]["fields"], payload_tree(final.payload))
            all_images[lane][key]["latent"] = numpy_tensor(final.payload["latent"]).astype(np.float64)
            png = branch / decoder["png"]["path"]
            relative = png.relative_to(root).as_posix()
            require(sha(png) == decoder["png"]["sha256"] and relative in files, "PNG image custody differs")
            target = output / "images" / (key.replace("/", "--") + "--" + lane + ".png")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(png.read_bytes())
            images[key + "/" + lane] = dict(path=str(target.relative_to(output)), **files[relative])
        cases[key] = dict(decoder=comparisons, consumer_ports=ports,
            entry_endpoint=endpoint_checks[record["context"]][record["arm"]])
        equation_reports[key] = dict(
            writer_exit=compare_equations(
                {"text.exit": array(root, writers["forecast"][cid]["endpoints"][arm])},
                {"text.exit": array(root, writers["native"][cid]["endpoints"][arm])}, max_elements=2_000_000),
            continuation=compare_equations(all_images["forecast"][key], all_images["native"][key], max_elements=2_000_000))

    observations = {}
    for cid in source_roots:
        observations[cid] = {}
        for field in ("latent", "decoded", "pixels"):
            y = {arm: all_images["native"][cid + "/" + arm][field] for arm in plan["arms"]}
            baseline, support = y["QQQ"], y["PPP"]
            responses = {arm: metric(value-baseline, support-baseline) for arm, value in y.items()}
            distances = {arm: dict(to_Q=metric(value, baseline)["rms"], to_S=metric(value, support)["rms"]) for arm, value in y.items()}
            value = dict(responses=responses, distances=distances)
            if plan["mode"] == "historical":
                corners = ["QQQ", "QQP", "QPQ", "QPP", "PQQ", "PQP", "PPQ", "PPP"]
                if set(corners).issubset(y):
                    interaction = sum(((-1)**(3-arm.count("P"))) * y[arm] for arm in corners)
                    value["triple_interaction"] = metric(interaction, support-baseline)
                else:
                    value["triple_interaction"] = {"status": "not-applicable", "reason": "Partial mechanics panel has no complete eight-corner lattice"}
            else:
                whole, complement = y["whole_P"], y["complement_P"]
                axis = whole-baseline
                s, c = metric(support-baseline, axis), metric(complement-baseline, axis)
                dq = metric(support, baseline)["rms"]
                dp = metric(support, whole)["rms"]
                cq = metric(complement, baseline)["rms"]
                cp = metric(complement, whole)["rms"]
                value["frozen_route_predictions"] = dict(support_closer_to_whole_than_Q=dp < dq,
                    complement_closer_to_Q_than_whole=cq < cp,
                    support_larger_positive_projection_than_complement=(s["projection"] is not None and c["projection"] is not None and s["projection"] > max(0., c["projection"])),
                    support_to_whole=dp, support_to_Q=dq, complement_to_Q=cq, complement_to_whole=cp,
                    support_projection=s["projection"], complement_projection=c["projection"], whole_Q_rms=metric(whole, baseline)["rms"])
                chosen = plan["family_rule"].get("component_arm")
                if chosen:
                    value["component_prediction"] = dict(arm=chosen, closer_to_support_than_Q=distances[chosen]["to_S"] < distances["QQQ"]["to_S"],
                        component_to_support=distances[chosen]["to_S"], Q_to_support=distances["QQQ"]["to_S"])
            observations[cid][field] = value
    result = dict(schema="mdb-flux-binding-family-saved-audit-v1", native_job_id=config["native_job_id"],
        native_request_sha256=config["native_request_sha256"], native_manifest_sha256=config["native_manifest_sha256"],
        source_worker_sha256=request["source_hashes"]["flux_binding_family_campaign.py"],
        mode=plan["mode"], contexts=list(source_roots), arms=plan["arms"], cases=cases,
        writer_checks=writer_checks, endpoint_checks=endpoint_checks, recomputed_checks=recomputed_checks,
        source_joins=source_joins, construction_checks=construction_checks, act_checks=act_checks, known_endpoint_checks=known_checks, boundary_counts=boundary_counts,
        observations=observations, images=images, equation_reports=equation_reports, transaction=transaction, external_sealed_files=len(external),
        published_files_verified=len(files), original_parameters_exported=False, model_forwards=0,
        full_array_fields_compared=fields_compared, unique_arrays_read=len(unique_arrays), repeated_array_elements_read=repeated_elements,
        numerical_bit_misses=[dict(case=key, field=name, kind=kind) for key, case in cases.items()
            for kind in ("decoder", "consumer_ports") for name, row in case[kind].items() if not row["raw_bit_exact"]],
        writer_bit_misses=[dict(context=cid, role=role, port=name) for cid, data in writer_checks.items()
            for role, checks in data.items() for name, row in checks["comparisons"].items() if not row["raw_bit_exact"]],
        endpoint_bit_misses=[dict(context=cid, arm=arm) for cid, data in endpoint_checks.items()
            for arm, row in data.items() if not row["raw_bit_exact"]],
        recomputed_bit_misses=[dict(context=cid, arm=arm, port=name) for cid, data in recomputed_checks.items()
            for arm, checks in data.items() for name, row in checks.items() if not row["raw_bit_exact"]],
        mechanics_status="verified", instrument_status="saved independent complete-array/hash reader", terminal_status="unassessed",
        limitations=["No new model forward; this audit qualifies saved numerical evidence, not independent kernels.",
            "Contexts and intervention arms are dependent conditions; positions/fields overlap.",
            "Route/effect predictions and exact numeric forecasts have separate scopes; retain every false predicate.",
            "No calibrated object-level semantic evaluator is used."])
    save(output / "audit.json", result)
    save(output / "observations.json", observations)
    for name in ("parameter-artifacts.json", "retrieve_weights.py"):
        (output / name).write_bytes((root / name).read_bytes())
    forward_guard.remove()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read(args.config)
    config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-family-audit-result")
    result = audit(config)
    from mdb.job_support import request_sha256
    publication = dict(output_root=config["output"], audit_sha256=sha(Path(config["output"]) / "audit.json"),
        source_job=config["native_job_id"], config_sha256=request_sha256(read(args.config)))
    print("MDB_FAMILY_AUDIT_READY " + json.dumps(publication, sort_keys=True), flush=True)
