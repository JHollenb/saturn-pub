"""Model-free audit of latent/future-conditioning carrier interventions.

Native projection arrays are observed consumer outputs, not independently
evaluated weights. Numerical and visual trends never determine mechanics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_bits(actual, expected):
    """Signed zeros are distinct when checking an exact native cast recipe."""
    a, b = np.asarray(actual), np.asarray(expected)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def distance(actual, reference):
    a, b = (np.asarray(value, dtype=np.float64) for value in (actual, reference))
    require(a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all(),
            "incompatible or nonfinite saved observations")
    delta = a - b
    return {"exact_base": exact_bits(actual, reference), "base_l2": float(np.linalg.norm(delta.ravel())),
            "base_mae": float(np.abs(delta).mean()), "signed_mean": float(delta.mean()),
            "max_abs": float(np.abs(delta).max()), "scope": "numerical geometry only, not semantic correctness"}


def conditioning_recipe(own, beta, quantize):
    require(type(beta) in {int, float} and math.isfinite(beta) and 0 <= beta <= 1, "invalid conditioning gain")
    own = np.asarray(own)
    require(own.dtype == np.float32 and own.size > 0 and np.isfinite(own).all(), "invalid conditioning source")
    return own.copy() if beta == 1 else quantize(np.float32(beta) * own)


def latent_recipe(z2, z0, sigma2, alpha, quantize):
    require(type(alpha) in {int, float} and math.isfinite(alpha) and 0 <= alpha <= 1
            and type(sigma2) in {int, float} and math.isfinite(sigma2) and sigma2 >= 0, "invalid latent recipe scalar")
    z2, z0 = np.asarray(z2), np.asarray(z0)
    require(z2.dtype == z0.dtype == np.float32 and z2.shape == z0.shape and z2.size > 0
            and np.isfinite(z2).all() and np.isfinite(z0).all(), "latent source geometry differs")
    anchor = np.float32(sigma2) * z0
    residual = z2 - anchor
    installed = z2.copy() if alpha == 1 else quantize(anchor + np.float32(alpha) * residual)
    return anchor, residual, installed


def rescue_origin(row, baseline, parent_point):
    """Bind a rescue to its own actually edited zero-C prefix, not native Z3."""
    cut = next(item["cut"] for item in baseline["schedulers"] if item["step"] == 3)
    require(row["rescue_from"] == baseline["id"] and baseline["rescue_from"] is None and baseline["beta"] == 0.
            and row["alpha"] == baseline["alpha"] and row["parent"] == cut
            and parent_point["logical_step"] == 3 and parent_point["phase"] == "project"
            and row["shared_prefix"] == {"row_id": baseline["id"], "cut": cut, "logical_step": 3, "adapter_steps_reused": 28},
            "rescue borrowed a different edited Z3 history")
    return cut


def analyze(root):
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
    run = h["source_run"](root, "flux-carriers-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data = run["data"]
    read = h["read"]
    source, summary = (read(data / name) for name in ("source.json", "summary.json"))
    store = typed["audit_store"](data / "typed-store")
    slot = lambda cut, key: routes["slot_array"](data / "typed-store", cut, key, typed)

    def fields(cut):
        return {item["key"]["value"]: item["value"] for item in store["cuts"][cut]["payload"]["items"]}

    def point(cut):
        return typed["unpack"](store["cuts"][cut]["execution_point"])

    def array(record):
        path = h["file_path"](data, record["path"])
        require(h["sha"](path.read_bytes()) == record["sha256"], "retained array file digest differs")
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == record["stored_shape"]
                and record["stored_dtype"] == "float32" and np.isfinite(value).all()
                and typed["matches"](value, record["descriptor"], bf16=record["descriptor"]["dtype"] == "torch.bfloat16"),
                "retained native/FP32 array ABI differs")
        return value

    return audit_native(run, source, summary, store, slot, fields, point, array, h, typed)


def audit_native(run, source, summary, store, slot, fields, point, array, h, typed):
    data, plan, request = run["data"], run["plan"], run["request"]
    read, sha = h["read"], h["sha"]
    observed = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = run["tree"].decode(read(run["transaction"] / "forecasts/carrier_hypotheses.json")["predictions"])
    require(plan == request["config"] and hypotheses == plan["hypotheses"]
            and observed["source_sha256"] == sha((data / "source.json").read_bytes())
            and observed["counts"] == summary["counts"] and observed["RGB_forecast"] is False, "sealed source/plan/counts differ")
    require(source["request_sha256"] == request["request_sha256"] == summary["request_sha256"]
            and summary["model_identity"] == source["model_identity"] and source["source_hashes"] == request["source_hashes"]
            and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"]
            and source["execution"] == source["prior_execution"], "source model/execution binding differs")
    files = {"flux_carriers_campaign.py", "flux_carriers.py", "flux_joint_self.py", "campaign_support.py",
             "prior-source.json", "prior-native-row.json", "prior-audit.json", "plan.json", "source-selection.json", "preflight.v1.json"}
    require({name for name in request["source_hashes"] if not name.startswith("input-store/")} == files,
            "ten staged source/metadata byte bindings differ")
    payload = run["root"] / "payload"
    selection = read(payload / "source-selection.json")
    prior, prior_audit, prior_row = (read(payload / name) for name in ("prior-source.json", "prior-audit.json", "prior-native-row.json"))
    require(source["source_selection"] == selection and selection["input_cuts"] == plan["input_cuts"]
            and selection["files"] == {key: value for key, value in request["source_hashes"].items() if key.startswith("input-store/")}
            and len(selection["files"]) == 20
            and sha((payload / "prior-source.json").read_bytes()) == selection["source_record"]["sha256"]
            and sha((payload / "prior-audit.json").read_bytes()) == selection["audit_sha256"]
            and sha((payload / "prior-native-row.json").read_bytes()) == selection["prior_native_row_sha256"]
            and prior_audit["mechanics_status"] == "valid" and prior_row["id"] == "native", "prior native custody differs")
    for key, selected in (("job_id", "prior_job_id"), ("archive_sha256", "archive_sha256"), ("request_sha256", "request_sha256")):
        require(prior_audit["source"][key] == selection[selected], "prior audited occurrence differs")
    require(prior["implementation_sources"] == source["prior_implementation_sources"]
            and prior["execution"] == source["execution"] and prior["model_identity"] == source["model_identity"]
            and prior["checkpoint_sha256"] == source["checkpoint_sha256"], "prior implementation/checkpoint differs")
    pipeline = [value for name, value in source["prior_implementation_sources"].items() if name.endswith("pipeline_flux2_klein.py")]
    require(len(pipeline) == 1 and pipeline[0] == "8a57a8a7f1c22fc0c5bc5d3fd1bd9bc76c17c1f4d6644cc73370651dd955358e",
            "native unit-noise source implementation differs")
    initial, parent = source["initial_cut"], source["parent_cut"]
    require(initial == plan["input_cuts"]["Z0"] == prior["initial_cut"]
            and parent == plan["input_cuts"]["Z2"]
            == next(item["cut"] for item in prior_row["schedulers"] if item["step"] == 2), "selected own source cuts differ")
    for cut, step in ((initial, 0), (parent, 2)):
        require(point(cut)["logical_step"] == step and point(cut)["phase"] == "project" and point(cut)["local_clock"] == 0
                and typed["unpack"](fields(cut)["step"]) == step and typed["unpack"](fields(cut)["cursor"]) == 0
                and all(typed["unpack"](fields(cut)[key]) is None for key in ("text", "image", "context", "noise_prediction")),
                "source is not a cleared project boundary")
    require(set(fields(initial)) == set(fields(parent)) and all(fields(initial)[key] == fields(parent)[key]
            for key in fields(initial) if key not in {"step", "latent"}), "source schedule/IDs/conditioning or nuisance state differs")
    for key in ("shape", "stride", "dtype"):
        require(fields(initial)["latent"][key] == fields(parent)["latent"][key], "own latent geometry differs")
    z0, z2, own_c = (array(source[key]) for key in ("initial_latent", "parent_latent", "own_conditioning"))
    require(exact_bits(z0, slot(initial, "latent")) and exact_bits(z2, slot(parent, "latent"))
            and exact_bits(own_c, slot(parent, "conditioning")), "source array/cut bytes differ")
    sigmas, timesteps = slot(parent, "sigmas"), slot(parent, "timesteps")
    require(sigmas.dtype == np.float32 and sigmas.shape == (5,) and float(sigmas[0]) == 1.
            and len(timesteps) == 4 and source["sigmas"] == sigmas.tolist() and source["timesteps"] == timesteps.tolist()
            and source["sigma_index"] == 2 and source["sigma2"] == float(sigmas[2]), "actual four-step sigma/timestep source differs")
    q = typed["quantize_bf16"]
    anchor, residual, _ = latent_recipe(z2, z0, source["sigma2"], 1., q)
    require(exact_bits(anchor, array(source["anchor"])) and exact_bits(residual, array(source["residual"])), "FP32 anchor/residual bytes differ")
    specs = {spec["id"]: spec for spec in plan["branches"]}
    rows = {row["id"]: row for row in observed["rows"]}
    bases = {name for name, spec in specs.items() if spec["rescue_from"] is None}
    require(len(rows) == len(observed["rows"]) == len(specs) == 11 and set(rows) == set(specs)
            and len(bases) == 9 and set(source["recipes"]) == bases and source["branches"] == plan["branches"], "planned branch inventory differs")
    expected = {}
    for name in bases:
        spec, recipe = specs[name], source["recipes"][name]
        _, _, latent = latent_recipe(z2, z0, source["sigma2"], spec["alpha"], q)
        conditioning = conditioning_recipe(own_c, spec["beta"], q)
        require(recipe["alpha"] == spec["alpha"] and recipe["beta"] == spec["beta"] and recipe["sigma_index"] == 2
                and recipe["sigma2"] == source["sigma2"] and recipe["native_latent_copy"] == (spec["alpha"] == 1.)
                and recipe["native_conditioning_copy"] == (spec["beta"] == 1.)
                and recipe["latent_stride"] == fields(parent)["latent"]["stride"]
                and recipe["conditioning_stride"] == fields(parent)["conditioning"]["stride"]
                and exact_bits(latent, array(recipe["latent"])) and exact_bits(conditioning, array(recipe["conditioning"])),
                "exact FP32/BF16 latent/conditioning recipe differs, including signed zeros")
        expected[name] = (latent, conditioning)
    receipts, edges = {}, {}
    for path in (data / "typed-store/receipts").glob("*.json"):
        receipt = read(path)
        h["public_receipt"](receipt)
        require(receipt["fingerprint"] == path.stem and receipt["model_identity"] == source["model_identity"]
                and receipt["execution"] == source["execution"], "public receipt source/execution differs")
        receipts[path.stem] = receipt
        if receipt["operation"] == "continue":
            require(receipt["parent"] not in edges or edges[receipt["parent"]] == receipt, "ambiguous native continuation edge")
            edges[receipt["parent"]] = receipt
    for cut, manifest in store["cuts"].items():
        require(manifest["model_identity"] == source["model_identity"] and manifest["execution"] == source["execution"]
                and all(fields(cut)[key] == fields(parent)[key] for key in fields(parent)
                        if key not in {"latent", "conditioning", "text", "image", "context", "noise_prediction", "step", "cursor", "phase"}),
                "model/execution/schedule/ID closure changed")
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    require([event["sequence"] for event in timeline] == list(range(len(timeline))), "timeline sequence differs")
    seals = [event for event in timeline if event["kind"] == "hypotheses_sealed"]
    starts = {event["row"]: event for event in timeline if event["kind"] == "branch_start"}
    completes = {event["row"]: event for event in timeline if event["kind"] == "branch_complete"}
    require(len(seals) == 1 and set(starts) == set(completes) == set(rows)
            and source["preseal_counts"] == seals[0]["counts"] and source["preseal_native_entries"] == seals[0]["native_entries_before_release"] == 0
            and all(seals[0]["counts"][key] == 0 for key in
                    ("adapter_steps", "acts", "project_reads", "prompt_encodings", "noise_draws", "images", "vae_calls"))
            and seals[0]["counts"]["model_loads"] == seals[0]["counts"]["physical_batch"] == 1
            and all(seals[0]["sequence"] < starts[name]["sequence"] < completes[name]["sequence"] for name in rows),
            "seal did not precede all native branch/image work")
    latents, pixels, paths, project_checks, contexts, preview_count = {}, {}, {}, {}, {}, 0
    step_count, act_count, read_count = 0, 0, 0

    def png(record):
        path = h["file_path"](data, record["path"])
        require(sha(path.read_bytes()) == record["sha256"], "native PNG digest differs")
        with Image.open(path) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "native RGB ABI differs")
            return np.asarray(image).copy()

    for name, row in rows.items():
        spec, rescue = specs[name], specs[name]["rescue_from"]
        require(read(data / "rows" / (name + ".json")) == row and all(row[key] == value for key, value in spec.items())
                and starts[name]["parent"] == row["parent"] and starts[name]["rescue_from"] == rescue, "durable row/declared branch differs")
        if rescue is None:
            require(row["parent"] == parent and row["recipe"] == source["recipes"][name]
                    and row["continuation_policy"] == "stored-C-dose-through-final", "base branch source/recipe differs")
            expected_l, expected_c = expected[name]
        else:
            baseline = rows[rescue]
            rescue_origin(row, baseline, point(row["parent"]))
            require(rescue in bases and specs[rescue]["beta"] == 0. and spec["alpha"] == specs[rescue]["alpha"]
                    and completes[rescue]["sequence"] < starts[name]["sequence"]
                    and row["continuation_policy"] == "stored-C-dose-until-Z3-then-own-C"
                    and exact_bits(array(row["recipe"]["conditioning"]), own_c), "rescue borrowed a different edited Z3 history")
            expected_l, expected_c = slot(row["parent"], "latent"), own_c
        actions, current = {}, row["parent"]
        require([action["slot"] for action in row["acts"]] == (["conditioning"] if rescue else [] if name == "native" else ["latent", "conditioning"]),
                "declared carrier Act inventory differs")
        for action in row["acts"]:
            receipt, key = action["receipt"], action["slot"]
            value = expected_l if key == "latent" else expected_c
            require(receipt == receipts[receipt["fingerprint"]] and receipt["operation"] == "apply"
                    and receipt["act"] == action["act"] and receipt["parent"] == action["parent"] == current
                    and receipt["result"] == action["post_act"] and action["receipt_id"] == receipt["fingerprint"]
                    and action["act"]["writes"] == [key] and action["act"]["operation"]["id"] == "saturn.builtin.replace",
                    "public Act binding differs")
            parameter = typed["unpack"](action["act"]["parameters"])["value"]
            require(fields(action["post_act"]) == {**fields(current), key: parameter}
                    and typed["unpack"](receipt["writes"])[key] == parameter
                    and point(current) == point(action["post_act"]) == action["point"]
                    and point(current)["logical_step"] == (3 if rescue else 2) and point(current)["phase"] == "project"
                    and exact_bits(slot(action["post_act"], key), value)
                    and fields(action["post_act"])[key]["stride"] == fields(parent)[key]["stride"], "typed carrier closure/cast/stride differs")
            require(current not in actions, "ambiguous carrier Act path")
            actions[current], current = action, action["post_act"]
        require(exact_bits(slot(current, "latent"), expected_l) and exact_bits(slot(current, "conditioning"), expected_c), "installed carrier inputs differ")
        start, visited, native_steps = row["parent"], set(), 0
        while start != row["final_cut"]:
            require(start not in visited and len(visited) < 80, "cyclic/unbounded carrier suffix")
            visited.add(start)
            if start in actions:
                start = actions[start]["post_act"]
            else:
                require(start in edges, "broken native carrier suffix")
                receipt = edges[start]
                require(receipt["steps"] == len(receipt["transitions"]) == 1, "native transition inventory differs")
                native_steps += 1
                start = receipt["result"]
        require(native_steps == (28 if rescue else 56) and set(actions) <= visited
                and point(start)["logical_step"] == 4 and point(start)["phase"] == "project", "incomplete/wrong carrier future")
        reads = row["project_reads"]
        require([item["logical_step"] for item in reads] == ([3] if rescue else [2, 3]), "native projection-read horizon differs")
        project_checks[name] = []
        for item in reads:
            before, after, receipt = item["pre_cut"], item["post_cut"], item["receipt"]
            clock = item["logical_step"]
            require(before in visited and after in visited | {start} and receipt == edges[before]
                    and receipt["result"] == after and point(before)["phase"] == "project" and point(before)["logical_step"] == clock
                    and point(after)["phase"] == "joint" and point(after)["index"] == 0 and point(after)["logical_step"] == clock
                    and fields(before)["conditioning"] == fields(after)["conditioning"]
                    and fields(before)["latent"] == fields(after)["latent"]
                    and exact_bits(slot(before, "conditioning"), expected_c)
                    and all(typed["unpack"](fields(before)[key]) is None for key in ("text", "image", "context", "noise_prediction")),
                    "actual projection read is not this branch's cleared carrier state")
            for key, cut in (("conditioning", before), ("latent", before), ("text", after), ("image", after)):
                require(exact_bits(array(item[key]), slot(cut, key)), "actual projection port/cut bytes differ")
            if clock in contexts:
                require(fields(after)["context"] == contexts[clock], "same-clock time/position context differs")
            contexts[clock] = fields(after)["context"]
            project_checks[name].append({"clock": clock, "pre_cut": before, "post_cut": after,
                "receipt": receipt["fingerprint"], "actual_ports_bit_exact": True, "time_context_exact_same_clock": True,
                "scope": "Native read/port provenance and typed bytes checked; projection weights not independently evaluated."})
        saved = row["schedulers"]
        require(len(saved) == 2 and {item["step"] for item in saved} == {3, 4}, "scheduler observation horizon differs")
        latents[name], scheduler_cuts = {}, {}
        for item in saved:
            clock, cut = item["step"], item["cut"]
            require(cut in visited | {start} and point(cut)["logical_step"] == clock and point(cut)["phase"] == "project"
                    and exact_bits(array(item["latent"]), slot(cut, "latent")), "scheduler array is outside its actual branch")
            latents[name][clock], scheduler_cuts[str(clock)] = array(item["latent"]), cut
        require(scheduler_cuts["4"] == row["final_cut"] and (not rescue or scheduler_cuts["3"] == row["parent"]), "final/shared scheduler cut differs")
        previews = row["previews"]
        require(set(previews) == ({"3"} if name in plan["preview_rows"] else set()), "preview inventory differs")
        for clock, item in previews.items():
            png(item)
            require(item["cut"] == scheduler_cuts[clock] and item["point"] == point(item["cut"]), "preview source cut differs")
            preview_count += 1
        require(exact_bits(slot(start, "conditioning"), expected_c) and row["final_conditioning"] == row["expected_conditioning"]
                and typed["matches"](expected_c, row["expected_conditioning"], bf16=True), "final conditioning policy differs")
        pixels[name] = png(row["image"])
        paths[name] = {"parent": row["parent"], "final": start, "native_steps": native_steps,
            "acts": len(actions), "rescue_from": rescue, "shared_prefix_steps": 28 if rescue else 0}
        step_count, act_count, read_count = step_count + native_steps, act_count + len(actions), read_count + len(reads)
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    require([item["sequence"] for item in emissions] == list(range(len(emissions)))
            and all(item["receipt"] in receipts and item["branch"] in rows for item in emissions), "receipt emission census differs")
    native_emitted = sum(receipts[item["receipt"]].get("steps", 0) for item in emissions if receipts[item["receipt"]]["operation"] == "continue")
    acts_emitted = sum(receipts[item["receipt"]]["operation"] == "apply" for item in emissions)
    counts = summary["counts"]
    require(step_count == native_emitted == counts["adapter_steps"] == 560 and act_count == acts_emitted == counts["acts"] == 18
            and read_count == counts["project_reads"] == 20 and len(emissions) == counts["receipt_emissions"] == 578
            and counts["final_images"] == 11 and counts["preview_images"] == preview_count == 4 and counts["images"] == 15
            and counts["vae_calls"] == 16 and counts["rescue_prefix_reuses"] == 2
            and counts["prompt_encodings"] == counts["noise_draws"] == 0
            and counts["model_loads"] == counts["physical_batch"] == 1, "physical work inventory differs")
    for name, row in rows.items():
        emitted = [receipts[item["receipt"]] for item in emissions if item["branch"] == name]
        require(sum(item.get("steps", 0) for item in emitted if item["operation"] == "continue") == paths[name]["native_steps"]
                and sum(item["operation"] == "apply" for item in emitted) == paths[name]["acts"], "branch-labelled physical work differs")
    parity = read(data / "preview-helper-final-parity.json")
    require(parity == observed["preview_helper_final_parity"] and parity["cut"] == rows["native"]["final_cut"]
            and parity["point"] == point(rows["native"]["final_cut"])
            and parity["extra_vae_calls"] == 1 and parity["extra_saved_images"] == 0
            and parity["adapter_png_sha256"] == rows["native"]["image"]["sha256"]
            and parity["adapter_pixel_sha256"] == hashlib.sha256(pixels["native"].tobytes()).hexdigest()
            and parity["pixel_exact"] == (parity["adapter_pixel_sha256"] == parity["helper_pixel_sha256"])
            and parity["png_exact"] == (parity["adapter_png_sha256"] == parity["helper_png_sha256"]),
            "native/helper parity observation provenance differs")

    def compare(name, baseline):
        return {"rgb": distance(pixels[name], pixels[baseline]),
                "scheduler_trajectory": {str(clock): distance(latents[name][clock], latents[baseline][clock]) for clock in (3, 4)}}

    comparisons = {name: compare(name, "native") for name in rows}
    rescues = {name: {"baseline": spec["rescue_from"], **compare(name, spec["rescue_from"])}
               for name, spec in specs.items() if spec["rescue_from"]}
    require(all(item["scheduler_trajectory"]["3"]["exact_base"] for item in rescues.values()), "rescue does not share exact altered Z3 latent")
    controls = {"native_png_equals_prior": rows["native"]["image"]["sha256"] == plan["expected_native_png_sha256"],
        "sham_png_equals_native": comparisons["sham"]["rgb"]["exact_base"],
        "sham_scheduler_arrays_exact": all(item["exact_base"] for item in comparisons["sham"]["scheduler_trajectory"].values()),
        "source_and_recipe_bytes_exact_including_signed_zero": True, "actual_projection_ports_typed_exact": True,
        "same_edited_Z3_rescue_prefixes": True, "preview_helper_final_pixel_exact": parity["pixel_exact"],
        "preview_helper_final_png_exact": parity["png_exact"], "native_final_VAE_helper_exact": parity["pixel_exact"] and parity["png_exact"]}
    require(observed["control_observations"] == summary["control_observations"]
            and all(observed["control_observations"][key] == controls[key] for key in
                    ("native_png_equals_prior", "sham_png_equals_native", "preview_helper_final_pixel_exact", "preview_helper_final_png_exact"))
            and observed["control_observations"]["native_expected_prior_png_sha256"] == plan["expected_native_png_sha256"],
            "worker control observations differ from saved bytes")
    replay_controls = {key: controls[key] for key in ("native_png_equals_prior", "sham_png_equals_native",
                                                    "sham_scheduler_arrays_exact", "native_final_VAE_helper_exact")}
    replay_qualified = all(replay_controls.values())
    return {"schema": "mdb-flux-carriers-audit-v1", "source": run["source"], "counts": counts,
        "mechanics_status": "valid", "instrument_status": "native-replay-qualified" if replay_qualified else "replay-unresolved",
        "replay_qualification": {"controls": replay_controls, "status": "qualified" if replay_qualified else "unresolved",
                                 "exact_replay_claim": "established for named controls" if replay_qualified else "not established; retain all rows and metrics"},
        "terminal_status": "unassessed",
        "native_paths": paths, "project_reads": project_checks, "comparisons": comparisons, "rescue_comparisons": rescues,
        "control_observations": controls, "sealed_hypotheses": hypotheses, "preseal_counts": seals[0]["counts"],
        "preview_helper_final_parity": parity,
        "preseal_seconds": seals[0]["seconds"], "typed_store_counts": store["counts"], "models_executed": False,
        "auditor_source_sha256": sha(Path(__file__).read_bytes()),
        "audit_helpers": {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
            ("summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")},
        "limits": "Observed own Z0/Z2; derived sigma2*Z0 anchor and artificial C gains, not proven memory erasure or absent semantics. Projection arrays are observed native reads, not independent weight evaluation. One prompt/seed and dependent futures; visual hypotheses componentwise and separately inspected. No numerical effect or semantic-quality gate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("mechanics_status", "counts", "control_observations")}))
