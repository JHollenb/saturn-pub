"""Independent saved-array audit of paired early-KV and late register rescues.

No model, attention kernel, projection or suffix equation is executed. Exact
local ABI/recipe checks protect mechanics; replay and scientific outcomes remain
separate named observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def distance(actual, reference):
    a, b = (np.asarray(value, dtype=np.float64) for value in (actual, reference))
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), "incompatible saved arrays")
    delta = a - b
    return {"exact_base": exact_bits(actual, reference), "base_l2": float(np.linalg.norm(delta.ravel())),
            "base_mae": float(np.abs(delta).mean()), "signed_mean": float(delta.mean()),
            "max_abs": float(np.abs(delta).max())}


def factorial_interaction(zero, image, text, both):
    arrays = [np.asarray(value, dtype=np.float64) for value in (zero, image, text, both)]
    require(all(value.shape == arrays[0].shape and value.size and np.isfinite(value).all() for value in arrays),
            "factorial cell geometry differs")
    value = arrays[3] - arrays[1] - arrays[2] + arrays[0]
    return {"formula": "Y_late.both - Y_late.image - Y_late.text + Y_zero; FP64 arithmetic on saved native cells",
            "shape": list(value.shape), "l2": float(np.linalg.norm(value.ravel())),
            "rms": float(np.sqrt(np.mean(value * value))), "max_abs": float(np.abs(value).max()),
            "signed_mean": float(value.mean()), "vector_float64_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "scope": "Exploratory end-to-end vector nonadditivity at matched input frontier; no semantic score or threshold."}


class Evidence:
    """Verified public custody plus typed cuts/receipts; no runtime execution."""

    def __init__(self, root):
        here = Path(__file__).resolve().parent
        self.h = runpy.run_path(str(here / "summarize_context_window.py"))
        self.typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
        self.routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
        self.run = self.h["source_run"](root, "flux-rescue-routes-result", max_archive_bytes=512 << 20,
                                         max_member_bytes=128 << 20)
        self.data = self.run["data"]
        self.store = self.typed["audit_store"](self.data / "typed-store")
        self.receipts, self.edges = {}, {}
        for path in (self.data / "typed-store/receipts").glob("*.json"):
            receipt = self.read(path)
            self.h["public_receipt"](receipt)
            require(receipt["fingerprint"] == path.stem, "receipt filename differs")
            self.receipts[path.stem] = receipt
            if receipt["operation"] == "continue":
                require(receipt["parent"] not in self.edges or self.edges[receipt["parent"]] == receipt,
                        "ambiguous native receipt graph")
                self.edges[receipt["parent"]] = receipt

    def read(self, path):
        return self.h["read"](path)

    def fields(self, cut):
        return {item["key"]["value"]: item["value"] for item in self.store["cuts"][cut]["payload"]["items"]}

    def point(self, cut):
        if cut in self.store["cuts"]:
            return self.typed["unpack"](self.store["cuts"][cut]["execution_point"])
        # Navigator receipts retain intermediate execution points even when the
        # intermediate full StateCut was not selected for storage.
        if not hasattr(self, "_receipt_points"):
            self._receipt_points = receipt_points(self.receipts.values())
        require(cut in self._receipt_points, "execution point absent from retained cut and receipt")
        return self._receipt_points[cut]

    def slot(self, cut, key):
        return self.routes["slot_array"](self.data / "typed-store", cut, key, self.typed)

    def array(self, record):
        path = self.h["file_path"](self.data, record["path"])
        require(self.h["sha"](path.read_bytes()) == record["sha256"], "array file hash differs")
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and record["stored_dtype"] == "float32"
                and list(value.shape) == record["stored_shape"] and value.size and np.isfinite(value).all()
                and self.typed["matches"](value, record["descriptor"],
                    bf16=record["descriptor"]["dtype"] == "torch.bfloat16"), "array dtype/shape/native descriptor differs")
        return value

    def png(self, record):
        path = self.h["file_path"](self.data, record["path"])
        require(self.h["sha"](path.read_bytes()) == record["sha256"], "PNG hash differs")
        with Image.open(path) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "PNG ABI differs")
            return np.asarray(image).copy()

    def act(self, action, parent, expected):
        receipt, key = action["receipt"], action["slot"]
        require(receipt == self.receipts[receipt["fingerprint"]] and receipt["operation"] == "apply"
                and receipt["act"] == action["act"] and action["receipt_id"] == receipt["fingerprint"]
                and receipt["parent"] == action["parent"] == parent and receipt["result"] == action["post_act"]
                and action["act"]["writes"] == [key] and key in {"image", "text"}
                and action["act"]["operation"]["id"] == "saturn.builtin.replace", "Act source/slot binding differs")
        parameter = self.typed["unpack"](action["act"]["parameters"])["value"]
        after = action["post_act"]
        require(self.fields(after) == {**self.fields(parent), key: parameter}
                and self.typed["unpack"](receipt["writes"])[key] == parameter
                and self.point(parent) == self.point(after) == action["point"]
                and exact_bits(self.slot(after, key), expected)
                and self.fields(after)[key]["stride"] == self.fields(parent)[key]["stride"], "Act value/stride/undeclared closure differs")
        return after

    def walk(self, parent, target, actions=(), *, limit=40):
        action_map = {item["parent"]: item for item in actions}
        require(len(action_map) == len(actions), "ambiguous Act chain")
        current, seen, used, native = parent, set(), [], 0
        while current != target:
            require(current not in seen and len(seen) < limit, "cyclic or excessive native suffix")
            seen.add(current)
            if current in action_map:
                action = action_map[current]
                used.append(action["receipt_id"])
                current = action["post_act"]
            else:
                require(current in self.edges, "missing native transition")
                receipt = self.edges[current]
                require(receipt["steps"] == len(receipt["transitions"]) == 1, "transition cardinality differs")
                used.append(receipt["fingerprint"])
                native += 1
                current = receipt["result"]
        require(set(action_map) <= seen, "Act outside declared suffix")
        return {"parent": parent, "final": target, "native_steps": native, "acts": len(actions), "receipts": used,
                "visited_cuts": sorted(seen | {target})}


def sequential_equation(ports, quantize):
    """Reconstruct declared rounded scalar-tensor arithmetic, not native blocks."""
    attention_write = quantize(ports["gate_attention"] * ports["attention_used"])
    mlp_write = quantize(ports["gate_mlp"] * ports["mlp_used"])
    u = quantize(ports["residual"] + attention_write)
    image = quantize(u + mlp_write)
    require(exact_bits(attention_write, ports["attention_write"]) and exact_bits(mlp_write, ports["mlp_write"])
            and exact_bits(u, ports["intermediate"]) and exact_bits(image, ports["image"]),
            "saved BF16 sequential writer equation differs")
    return {"intermediate_exact": True, "image_exact": True}


def receipt_points(receipts):
    points = {}
    for receipt in receipts:
        if receipt["operation"] != "continue":
            continue
        require(receipt["steps"] == len(receipt["transitions"]) == 1, "native point receipt cardinality differs")
        transition = receipt["transitions"][0]
        for cut, key in ((receipt["parent"], "before"), (receipt["result"], "after")):
            require(cut not in points or points[cut] == transition[key], "inconsistent receipt execution point")
            points[cut] = transition[key]
        require(receipt["execution_point"] == transition["after"], "receipt result execution point differs")
    return points


def conditioning_coverage(path, receipts, retained_fields, expected, point):
    """Check stored states where available and source-bound receipt footprints.

    Missing intermediate full states are not reconstructed. The declared
    footprints are checked against the unchanged, hash-bound native adapter;
    this is not independent low-level tracing of every numerical read.
    """
    retained = [cut for cut in path["visited_cuts"] if cut in retained_fields]
    require(all(retained_fields[cut]["conditioning"] == expected for cut in retained),
            "retained conditioning state differs")
    native = 0
    for rid in path["receipts"]:
        receipt = receipts[rid]
        if receipt["operation"] != "continue":
            continue
        transition = receipt["transitions"][0]
        before, after, operation = (transition[key] for key in ("before", "after", "native_operation"))
        require(before == point(receipt["parent"]) and after == point(receipt["result"])
                and before["logical_step"] == 3 and before["phase"] in {"joint", "single", "readout", "scheduler"}
                and after["logical_step"] in {3, 4} and operation["footprint"] == "declared"
                and operation["operation"] == before["next_operation"]
                and "conditioning" not in operation["reads"] and "conditioning" not in operation["writes"]
                and "conditioning" not in operation["invalidates"],
                "native suffix has a conditioning read/write, extra project or inconsistent receipt point")
        native += 1
    return {"retained_full_states_checked": len(retained),
            "intermediate_full_states_not_retained": len(path["visited_cuts"]) - len(retained),
            "native_transition_footprints_checked": native, "no_further_conditioning_projection": True,
            "declared_denoiser_numerical_conditioning_reads": 0,
            "scope": "Stored conditioning checked only at retained full cuts. Missing full states not reconstructed; intervening native points/footprints follow public receipts and byte-bound unchanged adapter source. Schema/custody may inspect C."}


def qkv_selection(candidate, observed, used, donor=None):
    n = candidate["text_rows"]
    require(type(n) is int and 0 < n < observed["q"].shape[1]
            and set(observed) == set(used) == {"q", "k", "v"}
            and all(value.ndim == 4 and value.shape == observed["q"].shape for value in [*observed.values(), *used.values()])
            and candidate["dispatch_calls"] == 1, "native QKV partition/dispatch ABI differs")
    selected = observed if donor is None else donor
    require(all(selected[key].shape == observed[key].shape for key in ("k", "v")), "donor QKV geometry differs")
    controls = {"q_exact_recipient": exact_bits(used["q"], observed["q"]),
        "image_k_exact_recipient": exact_bits(used["k"][:, n:], observed["k"][:, n:]),
        "image_v_exact_recipient": exact_bits(used["v"][:, n:], observed["v"][:, n:]),
        "text_k_exact_selected": exact_bits(used["k"][:, :n], selected["k"][:, :n]),
        "text_v_exact_selected": exact_bits(used["v"][:, :n], selected["v"][:, :n])}
    require(candidate["partition_controls"] == controls and all(controls.values()), "QKV recipe changed undeclared operands")
    return controls


def analyze(root):
    e = Evidence(root)
    run, data = e.run, e.data
    h, read, typed = e.h, e.read, e.typed
    plan, request = run["plan"], run["request"]
    source, summary = (read(data / name) for name in ("source.json", "summary.json"))
    prefixes = read(data / "prefixes.json")["prefixes"]
    candidates = read(data / "candidates.json")["candidates"]
    observed = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = run["tree"].decode(read(run["transaction"] / "forecasts/rescue_route_hypotheses.json")["predictions"])
    require(plan == request["config"] and hypotheses == plan["hypotheses"]
            and source["request_sha256"] == request["request_sha256"] == summary["request_sha256"]
            and source["source_hashes"] == request["source_hashes"] and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"]
            and source["branches"] == plan["branches"] and source["model_identity"] == summary["model_identity"],
            "sealed source/plan/model bindings differ")
    require(observed["source_sha256"] == h["sha"]((data / "source.json").read_bytes())
            and observed["counts"] == summary["counts"] and observed["prefixes"] == prefixes
            and observed["candidates"] == candidates and observed["RGB_forecast"] is False,
            "persisted source/count/prefix/candidate binding differs")
    source_files = {"flux_rescue_routes_campaign.py", "flux_rescue_routes.py", "flux_joint_self.py", "campaign_support.py",
                    "prior-source.json", "prior-zero-row.json", "prior-own-row.json", "prior-audit.json",
                    "plan.json", "source-selection.json", "preflight.v1.json"}
    require({key for key in request["source_hashes"] if not key.startswith("input-store/")} == source_files,
            "staged source byte inventory differs")
    payload = run["root"] / "payload"
    selection = read(payload / "source-selection.json")
    prior, prior_audit = (read(payload / name) for name in ("prior-source.json", "prior-audit.json"))
    require(source["source_selection"] == selection and selection["input_cuts"] == plan["input_cuts"] == source["input_cuts"]
            and selection["files"] == {key: value for key, value in request["source_hashes"].items() if key.startswith("input-store/")}
            and h["sha"]((payload / "prior-source.json").read_bytes()) == selection["source_record"]["sha256"]
            and h["sha"]((payload / "prior-audit.json").read_bytes()) == selection["audit_sha256"]
            and prior_audit["mechanics_status"] == "valid" and prior_audit["source"]["job_id"] == selection["prior_job_id"]
            and prior_audit["source"]["archive_sha256"] == selection["archive_sha256"]
            and prior_audit["source"]["request_sha256"] == selection["request_sha256"]
            and prior["model_identity"] == source["model_identity"] and prior["checkpoint_sha256"] == source["checkpoint_sha256"]
            and source["execution"] == prior["execution"], "prior occurrence custody differs")
    implementation = lambda mapping: {Path(name).name: digest for name, digest in mapping.items()}
    require(implementation(source["implementation_sources"]) == implementation(prior["implementation_sources"])
            and source["prior_implementation_sources"] == prior["implementation_sources"],
            "native implementation byte identities differ from selected prior occurrence")
    for name in ("zero", "own"):
        binding = selection["prior_rows"][name]
        require(h["sha"]((payload / binding["path"]).read_bytes()) == binding["sha256"], "prior row byte binding differs")
        oldrow = read(payload / binding["path"])
        require(source["input_cuts"][name] == next(item["post_cut"] for item in oldrow["project_reads"] if item["logical_step"] == 3)
                and oldrow["image"]["sha256"] == plan["expected_pngs"][name], "paired source is not the actual prior project read")
    zero, own = (source["input_cuts"][name] for name in ("zero", "own"))
    require(set(e.fields(zero)) == set(e.fields(own))
            and all(e.fields(zero)[key] == e.fields(own)[key] for key in e.fields(zero) if key not in {"conditioning", "text"})
            and e.point(zero) == e.point(own) and e.point(zero)["logical_step"] == 3
            and e.point(zero)["phase"] == "joint" and e.point(zero)["index"] == e.point(zero)["local_clock"] == 0,
            "paired joint0 parents differ outside declared C/text")
    require(np.count_nonzero(e.slot(zero, "text")) == 0 and np.count_nonzero(e.slot(zero, "conditioning")) == 0,
            "zero-source C/text is not the declared zero input")
    for name, cut in (("zero", zero), ("own", own)):
        for key in ("latent", "conditioning", "image", "text"):
            record = source["input_arrays"][name][key]
            require(exact_bits(e.array(record), e.slot(cut, key))
                    and record["native_stride"] == e.fields(cut)[key]["stride"], "paired saved source port/cut ABI differs")
    for cut, manifest in e.store["cuts"].items():
        require(manifest["model_identity"] == source["model_identity"] and manifest["execution"] == source["execution"]
                and all(e.fields(cut)[key] == e.fields(zero)[key] for key in e.fields(zero)
                    if key not in {"text", "image", "conditioning", "context", "latent", "noise_prediction", "step", "cursor", "phase"}),
                "typed model/execution/schedule/position closure differs")
    for receipt in e.receipts.values():
        require(receipt["model_identity"] == source["model_identity"] and receipt["execution"] == source["execution"],
                "receipt model/execution differs")
    retained_fields = {cut: e.fields(cut) for cut in e.store["cuts"]}
    public_points = receipt_points(e.receipts.values())
    require(all(cut not in e.store["cuts"] or e.point(cut) == value for cut, value in public_points.items()),
            "retained StateCut/receipt execution points differ")
    prefix_checks, prefix_receipts = {}, []
    require(set(prefixes) == {"zero", "own"}, "paired prefix inventory differs")
    for name, prefix in prefixes.items():
        start = source["input_cuts"][name]
        path = e.walk(start, prefix["single0"], limit=8)
        require(prefix["input_cut"] == start and path["native_steps"] == 5
                and e.point(prefix["joint0_post"])["phase"] == "joint" and e.point(prefix["joint0_post"])["index"] == 1
                and e.point(prefix["single0"])["phase"] == "single" and e.point(prefix["single0"])["index"] == 0
                and e.point(prefix["single0"])["logical_step"] == 3
                and prefix["joint0_post"] in path["visited_cuts"]
                and prefix["prefix_receipts"] == [e.receipts[fingerprint] for fingerprint in path["receipts"]],
                "paired five-joint prefix lineage differs")
        prefix_receipts.extend(path["receipts"])
        path["conditioning_coverage"] = conditioning_coverage(path, e.receipts, retained_fields,
            e.fields(start)["conditioning"], e.point)
        statistics = {}
        for key in ("joint0_post", "single0"):
            value = e.slot(prefix[key], "text").astype(np.float64)
            stats = prefix["observed_text_statistics"][key]
            require(all(np.isfinite(stats[field]) for field in ("l2_norm", "zero_fraction")), "nonfinite prefix text statistic")
            norm, fraction = float(np.linalg.norm(value.ravel())), float(np.mean(value == 0))
            statistics[key] = {"l2_norm": norm, "zero_fraction": fraction, "recorded": stats,
                "norm_reduction_difference": norm - stats["l2_norm"],
                "zero_fraction_difference": fraction - stats["zero_fraction"],
                "scope": "Observed native working text; history label does not imply numerical zero after joint blocks."}
        path["observed_text_statistics"] = statistics
        prefix_checks[name] = path
    require(set(candidates) == {"zero", "own", "kv"}, "three candidate calls differ")
    candidate_arrays, qkvs, candidate_checks = {}, {}, {}
    for name, candidate in candidates.items():
        parent = own if name == "own" else zero
        require(candidate["input_parent"] == parent and candidate["donor_parent"] == (own if name == "kv" else None),
                "candidate observed parent/donor differs")
        ports = {key: e.array(value) for key, value in candidate["ports"].items()}
        require(exact_bits(ports["residual"], e.slot(parent, "image"))
                and exact_bits(ports["input_text"], e.slot(parent, "text")), "candidate input port/cut differs")
        require(exact_bits(ports["attention_raw"], ports["attention_used"])
                and exact_bits(ports["mlp_raw"], ports["mlp_selected"])
                and exact_bits(ports["mlp_selected"], ports["mlp_used"]), "undeclared candidate output scaling/selection")
        candidate_checks[name] = sequential_equation(ports, typed["quantize_bf16"])
        observed_qkv = {key: e.array(value) for key, value in candidate["observed_qkv"].items()}
        used = {key: e.array(value) for key, value in candidate["used_qkv"].items()}
        candidate_arrays[name], qkvs[name] = ports, (observed_qkv, used)
        dispatch = e.array(candidate["attention_dispatch_output"])
        require(dispatch.shape == observed_qkv["q"].shape, "retained native attention dispatch output ABI differs")
    for name, candidate in candidates.items():
        candidate_checks[name]["operand_selection"] = qkv_selection(candidate, *qkvs[name], qkvs["own"][0] if name == "kv" else None)
    n = candidates["zero"]["text_rows"]
    candidate_controls = {"kv_observed_recipient_qkv_exact_zero": all(
        exact_bits(qkvs["kv"][0][key], qkvs["zero"][0][key]) for key in ("q", "k", "v"))}
    for name in ("zero", "own"):
        cut = prefixes[name]["joint0_post"]
        for key in ("image", "text"):
            candidate_controls[f"{name}_candidate_{key}_equals_native"] = exact_bits(candidate_arrays[name][key], e.slot(cut, key))
    candidate_controls["paired_image_qkv_exact"] = all(exact_bits(qkvs["zero"][0][key][:, n:], qkvs["own"][0][key][:, n:]) for key in ("q", "k", "v"))
    for key in ("attention_raw", "mlp_raw", "image"):
        candidate_controls["kv_" + key + "_equals_own"] = exact_bits(candidate_arrays["kv"][key], candidate_arrays["own"][key])
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    require([item["sequence"] for item in timeline] == list(range(len(timeline))), "timeline sequence differs")
    seals = [item for item in timeline if item["kind"] == "hypotheses_sealed"]
    require(len(seals) == 1 and source["preseal_counts"] == seals[0]["counts"]
            and source["preseal_native_entries"] == seals[0]["native_entries_before_release"] == 0
            and all(seals[0]["counts"][key] == 0 for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls", "prompt_encodings", "noise_draws")),
            "native work preceded hypothesis seal")
    require(all(item["sequence"] > seals[0]["sequence"] for item in timeline
                if item["kind"] in {"candidate_start", "candidate_complete", "branch_start", "branch_complete"}),
            "candidate or branch work preceded seal")
    starts = {item["row"]: item for item in timeline if item["kind"] == "branch_start"}
    completes = {item["row"]: item for item in timeline if item["kind"] == "branch_complete"}
    rows = {row["id"]: row for row in observed["rows"]}
    specs = {spec["id"]: spec for spec in plan["branches"]}
    require(len(rows) == len(observed["rows"]) == len(specs) == 9 and set(rows) == set(specs) == set(starts) == set(completes),
            "nine-row native inventory differs")
    paths, pixels, latents, act_total = {}, {}, {}, 0
    for name, row in rows.items():
        require(read(data / "rows" / (name + ".json")) == row and all(row[key] == value for key, value in specs[name].items())
                and seals[0]["sequence"] < starts[name]["sequence"] < completes[name]["sequence"], "row durability/timing/spec differs")
        early = name.startswith("early.")
        reference = "own" if name == "own" else "zero"
        parent = prefixes[reference]["single0"] if not early else prefixes["zero"]["joint0_post"]
        require(row["parent"] == parent and row["start_point"] == e.point(parent), "row borrowed a different prefix")
        if name == "sham":
            expected = {key: e.slot(parent, key) for key in ("image", "text")}
        elif name.startswith("late."):
            keys = {"late.image": ["image"], "late.text": ["text"], "late.both": ["image", "text"]}[name]
            expected = {key: e.slot(prefixes["own"]["single0"], key) for key in keys}
        elif early:
            donor = "own" if name == "early.coherent" else "kv"
            keys = ["image"] if name == "early.kv.image" else ["image", "text"]
            expected = {key: (e.slot(prefixes["own"]["joint0_post"], key) if donor == "own" else candidate_arrays[donor][key]) for key in keys}
        else:
            expected = {}
        require([action["slot"] for action in row["acts"]] == list(expected), "route Act slot inventory differs")
        current = parent
        for action in row["acts"]:
            current = e.act(action, current, expected[action["slot"]])
        path = e.walk(parent, row["final_cut"], row["acts"])
        require(path["native_steps"] == (26 if early else 22)
                and e.point(row["final_cut"])["logical_step"] == 4 and e.point(row["final_cut"])["phase"] == "project"
                and row["continuation_receipts"] == [e.receipts[rid] for rid in path["receipts"] if e.receipts[rid]["operation"] == "continue"]
                and row["prefix_source"] == reference
                and row["continuation_policy"] == ("own-stored-C" if reference == "own" else "zero-stored-C-no-more-projections"),
                "suffix horizon/conditioning/receipt lineage differs")
        path["conditioning_coverage"] = conditioning_coverage(path, e.receipts, retained_fields,
            e.fields(parent)["conditioning"], e.point)
        latents[name] = e.array(row["final_latent"])
        require(exact_bits(latents[name], e.slot(row["final_cut"], "latent")), "final scheduler array/cut differs")
        require(row["final_conditioning"] == row["expected_conditioning"]
                and typed["matches"](e.slot(parent, "conditioning"), row["expected_conditioning"], bf16=True),
                "declared final conditioning descriptor differs")
        pixels[name], paths[name] = e.png(row["image"]), path
        act_total += len(row["acts"])
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    allowed = set(prefix_receipts) | {rid for path in paths.values() for rid in path["receipts"]}
    require([item["sequence"] for item in emissions] == list(range(len(emissions)))
            and all(item["receipt"] in allowed for item in emissions), "receipt emission lineage differs")
    expected_emissions = [("prefix." + name, rid) for name in ("zero", "own") for rid in prefix_checks[name]["receipts"]]
    expected_emissions += [(spec["id"], rid) for spec in plan["branches"] for rid in paths[spec["id"]]["receipts"]]
    require([(item["branch"], item["receipt"]) for item in emissions] == expected_emissions,
            "branch-labelled emission order/census differs")
    emitted_native = sum(e.receipts[item["receipt"]].get("steps", 0) for item in emissions if e.receipts[item["receipt"]]["operation"] == "continue")
    emitted_acts = sum(e.receipts[item["receipt"]]["operation"] == "apply" for item in emissions)
    phases = {}
    for item in emissions:
        receipt = e.receipts[item["receipt"]]
        if receipt["operation"] == "continue":
            phase = e.point(receipt["parent"])["phase"]
            phases[phase] = phases.get(phase, 0) + receipt["steps"]
    counts = summary["counts"]
    require(sum(path["native_steps"] for path in paths.values()) + 10 == emitted_native == counts["adapter_steps"] == 220
            and act_total == emitted_acts == counts["acts"] == 11 and len(emissions) == counts["receipt_emissions"] == 231
            and counts["candidate_calls"] == 3 and counts["images"] == counts["vae_calls"] == 9
            and counts["prompt_encodings"] == counts["noise_draws"] == 0
            and counts["model_loads"] == counts["physical_batch"] == 1, "physical work census differs")
    comparisons = {name: {base: {"latent": distance(latents[name], latents[base]), "rgb": distance(pixels[name], pixels[base])}
                          for base in ("zero", "own")} for name in rows}
    interactions = {key: factorial_interaction(*(values[name] for name in ("zero", "late.image", "late.text", "late.both")))
                    for key, values in (("Z4_latent", latents), ("final_RGB", pixels))}
    controls = {**candidate_controls,
        "zero_png_equals_prior": rows["zero"]["image"]["sha256"] == plan["expected_pngs"]["zero"],
        "own_png_equals_prior": rows["own"]["image"]["sha256"] == plan["expected_pngs"]["own"]}
    for name, base in (("sham", "zero"), ("late.both", "own"), ("early.coherent", "own")):
        controls[name + "_png_exact_" + base] = rows[name]["image"]["sha256"] == rows[base]["image"]["sha256"]
        controls[name + "_latent_exact_" + base] = comparisons[name][base]["latent"]["exact_base"]
    declared_controls = {name + "_candidate_" + key + "_exact_native": candidate_controls[f"{name}_candidate_{key}_equals_native"]
                         for name in ("zero", "own") for key in ("image", "text")}
    declared_controls.update(kv_candidate_image_exact_own_candidate=candidate_controls["kv_image_equals_own"],
        kv_candidate_image_exact_own_native=exact_bits(candidate_arrays["kv"]["image"], e.slot(prefixes["own"]["joint0_post"], "image")),
        zero_png_equals_prior=controls["zero_png_equals_prior"], own_png_equals_prior=controls["own_png_equals_prior"],
        sham_png_equals_zero=controls["sham_png_exact_zero"], late_both_png_equals_own=controls["late.both_png_exact_own"],
        early_coherent_png_equals_own=controls["early.coherent_png_exact_own"])
    require(observed["control_observations"] == summary["control_observations"] == declared_controls,
            "declared native control observations differ from retained bytes")
    qualified = all(controls.values())
    here = Path(__file__).resolve().parent
    return {"schema": "mdb-flux-rescue-routes-audit-v1", "mechanics_status": "valid",
        "instrument_status": "native-replay-qualified" if qualified else "replay-unresolved", "terminal_status": "unassessed",
        "source": run["source"], "counts": counts, "typed_store_counts": e.store["counts"],
        "physical_native_transition_phases": phases, "direct_candidate_block_calls": 3,
        "replay_qualification": {"controls": controls, "status": "qualified" if qualified else "unresolved",
            "scope": "Named source/native/QKV/local-output/replay controls only; negative controls retain all rows."},
        "control_observations": controls, "candidate_checks": candidate_checks,
        "paired_source_checks": {"only_C_and_text_differ": True, "common_latent_image_context_schedule_ids": True,
            "source_cuts": source["input_cuts"], "prefixes": prefix_checks},
        "native_paths": paths, "comparisons": comparisons, "late_factorial_interactions": interactions,
        "sealed_hypotheses": hypotheses,
        "preseal_counts": seals[0]["counts"], "preseal_seconds": seals[0]["seconds"], "models_executed": False,
        "auditor_source_sha256": h["sha"](Path(__file__).read_bytes()),
        "audit_helpers": {name: h["sha"]((here / name).read_bytes()) for name in
            ("summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")},
        "limits": ["One observed altered Z3 history and paired native C read; dependent branches, no independent prompt or seed replication.",
            "Only selected full StateCuts retained. Intermediate point/footprint coverage uses public receipts and byte-bound unchanged native adapter; missing full states not fabricated or independently read-traced.",
            "Post-normalization/RoPE text-KV substitution; image write includes sequential MLP, not attention-only mediation.",
            "QKV/native ports observed and bound; SDPA/projection weights not independently reevaluated offline.",
            "Early and late interventions occur at different frontiers; later compensation may change their final effect.",
            "All numerical ranks and text-only semantic effects exploratory. No automatic semantic or terminal gate."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "counts")}))
