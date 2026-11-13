"""Audit a native R/A/M factorial from saved ports, cuts and actual RGB futures."""
from __future__ import annotations

import argparse
import itertools
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_path(start, final, edges, *, reset=None, limit=10000):
    """Follow stored native receipts, inserting one explicitly bound reset Act."""
    current, path, visited, reset_count = start, [], set(), 0
    while current != final:
        require(current not in visited and len(path) < limit, "cyclic or over-budget native lineage")
        visited.add(current)
        if reset is not None and current == reset["parent"]:
            require(reset_count == 0, "native lineage repeated conditioning reset")
            current = reset["result"]
            reset_count += 1
        require(current in edges, "broken native continuation lineage")
        receipt = edges[current]
        require(receipt["operation"] == "continue" and receipt["parent"] == current
                and receipt["steps"] == len(receipt["transitions"]), "invalid native lineage receipt")
        require(all("conditioning" not in transition["native_operation"]["writes"] for transition in receipt["transitions"]),
                "native continuation rewrites conditioning outside declared Act")
        path.append(receipt["fingerprint"])
        current = receipt["result"]
    require(reset_count == (1 if reset is not None else 0), "native lineage omitted conditioning reset")
    return path


def replacement_closure(action, fields, *, slot, expected, stored_receipt, unpack, public_receipt):
    """Check an exact typed replacement and every nonwritten payload field."""
    receipt = action["receipt"]
    public_receipt(receipt)
    require(receipt == stored_receipt and receipt["operation"] == "apply"
            and receipt["parent"] == action["parent"] and receipt["result"] == action["post_act"]
            and receipt["act"] == action["act"] and action["act"]["writes"] == [slot]
            and action["act"]["operation"]["id"] == "saturn.builtin.replace", "typed replacement receipt/slot differs")
    before, after = fields(action["parent"]), fields(action["post_act"])
    require(after == {**before, slot: expected}, "typed replacement changed undeclared closure")
    require(unpack(receipt["writes"])[slot] == expected
            and unpack(action["act"]["parameters"])["value"] == expected, "typed replacement payload differs")
    return receipt


def analyze(root):
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
    require, read, sha = (h[key] for key in ("require", "read", "sha"))
    run = h["source_run"](root, "flux-merge-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data, plan, tree = run["data"], run["plan"], run["tree"]
    source, summary = (read(data / name) for name in ("source.json", "summary.json"))
    recipient_mode = plan.get("panel") == "shape-step1-recipient-controls"
    shape_mode = plan.get("panel") in {"shape-step1-writer", "shape-step1-recipient-controls"}
    observed = tree.decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = tree.decode(read(run["transaction"] / "forecasts/merge_hypotheses.json")["predictions"])
    require(hypotheses == plan["hypotheses"] and plan == run["request"]["config"], "sealed plan/hypothesis mismatch")
    require(observed["source_sha256"] == sha((data / "source.json").read_bytes()) and observed["counts"] == summary["counts"], "source/work mismatch")
    store = typed["audit_store"](data / "typed-store")
    require(source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "source checkpoint mismatch")
    if shape_mode:
        require(not plan.get("fresh_context") and not plan.get("attention_sources")
                and plan.get("start_step", plan["selector"]["logical_step"]) == 1 and plan["reset_step"] == 2 and plan["steps"] == 4
                and plan["selector"] == {"logical_step": 1, "phase": "single", "index": 5 if recipient_mode else 4}
                and plan["block"] == (5 if recipient_mode else 4)
                and plan["prefix_selector"] == {"logical_step": 1, "phase": "single", "index": 0}, "shape writer geometry differs")
        require(source["origin_cuts"] == plan["input_cuts"] and source["prefix_selector"] == plan["prefix_selector"], "shape origin/prefix mismatch")
    elif not plan.get("fresh_context"):
        require(source["input_cuts"] == plan["input_cuts"], "source input mismatch")
    else:
        initial = source["initial"]
        require(initial["prompts"] == plan["prompts"] and initial["seed"] == plan["seed"], "fresh inputs differ")
        require(h["sha"]((root / "payload/campaign-plan.json").read_bytes()) == plan["campaign_sha256"], "frozen campaign differs")
    slot = lambda cut, name: routes["slot_array"](data / "typed-store", cut, name, typed)

    def fields(cut):
        return {item["key"]["value"]: item["value"] for item in store["cuts"][cut]["payload"]["items"]}

    shape_prefix_paths, conditioning_history = {}, {}
    if shape_mode:
        for name in ("base", "changed"):
            origin, selected, post = source["origin_cuts"][name], source["input_cuts"][name], source["post_cuts"][name]
            original = store["cuts"][origin]
            point = typed["unpack"](original["execution_point"])
            require(original["model_identity"] == source["model_identity"]
                    and original["execution"] == store["cuts"][source["origin_cuts"]["base"]]["execution"]
                    and original["execution_point"] == store["cuts"][source["origin_cuts"]["base"]]["execution_point"]
                    and all(point[key] == value for key, value in plan["prefix_selector"].items()), "wrong origin source model/execution/boundary")
            require(all(store["cuts"][cut]["model_identity"] == source["model_identity"]
                        and store["cuts"][cut]["execution"] == original["execution"] for cut in (selected, post)), "source prefix changed model/execution")
            conditioning_history[name] = {cut: fields(cut)["conditioning"] for cut in (origin, selected, post)}
            require(all(value == fields(origin)["conditioning"] for value in conditioning_history[name].values()), "source prefix conditioning changed")
            edges = {}
            for receipt in source["advance_receipts"][name]:
                h["public_receipt"](receipt)
                require(read(data / "typed-store/receipts" / (receipt["fingerprint"] + ".json")) == receipt
                        and receipt["model_identity"] == source["model_identity"]
                        and receipt["parent"] not in edges, "source advance receipt storage/model differs")
                edges[receipt["parent"]] = receipt
            path = native_path(origin, selected, edges, limit=plan["max_adapter_steps"])
            require(len(path) == len(edges) == plan["block"], "source prefix lineage has wrong number of single blocks")
            shape_prefix_paths[name] = path
        left_origin, right_origin = (fields(source["origin_cuts"][name]) for name in ("base", "changed"))
        require({key: value for key, value in left_origin.items() if key not in {"text", "image", "conditioning"}} ==
                {key: value for key, value in right_origin.items() if key not in {"text", "image", "conditioning"}}, "unmatched origin nuisance closure")

    if plan.get("fresh_context"):
        initial = source["initial"]
        before_fields, after_fields = fields(initial["base_projected"]), fields(initial["changed_projected"])
        require(all(after_fields[key] == value for key, value in before_fields.items() if key != "text"), "initial text contrast changed undeclared closure")
        require(initial["text_act"]["writes"] == ["text"], "initial contrast not text-only")
        h["public_receipt"](initial["text_receipt"])
    for name, cut in source["input_cuts"].items():
        point = typed["unpack"](store["cuts"][cut]["execution_point"])
        require(all(point[key] == value for key, value in plan["selector"].items()), "wrong selected source boundary")
    left, right = (fields(source["input_cuts"][name]) for name in ("base", "changed"))
    if shape_mode:
        require(store["cuts"][source["input_cuts"]["base"]]["execution_point"] ==
                store["cuts"][source["input_cuts"]["changed"]]["execution_point"], "selected source clocks differ")
    exceptions = {"text", "image", "conditioning"} if shape_mode else {"text", "image"}
    require({key: value for key, value in right.items() if key not in exceptions} ==
            {key: value for key, value in left.items() if key not in exceptions}, "unmatched source closure")

    def arrays(record):
        path = data / record["path"]
        require(sha(path.read_bytes()) == record["sha256"], "port archive digest mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            result = {key: loaded[key] for key in loaded.files}
        require(set(result) == set(record["descriptors"]), "port inventory mismatch")
        for name, value in result.items():
            require(typed["matches"](value, record["descriptors"][name], bf16=True), "native BF16 port descriptor mismatch")
        return result

    ports = {name: arrays(record) for name, record in source["ports"].items()}
    text_rows, attention_width = source["text_rows"], source["attention_width"]
    gates_equal = np.array_equal(ports["base"]["gate"], ports["changed"]["gate"])
    require(gates_equal, "unmatched time gate")
    if recipient_mode:
        own = source["recipient_controls"]
        require(own["gate_source_cut"] == source["input_cuts"]["changed"]
                and typed["matches"](ports["changed"]["gate"], own["gate_descriptor"], bf16=True), "recipient gate provenance differs")
        own_policy = source["continuation_policies"]["recipient-own-C"]
        own_condition = fields(source["origin_cuts"]["changed"])["conditioning"]
        require(own_policy["source_cut"] == source["origin_cuts"]["changed"] and own_policy["reset"] is None
                and all(own_condition[key] == own_policy["value_descriptor"][key] for key in ("shape", "dtype", "sha256")),
                "recipient conditioning policy differs")
    native_merge_checks = {}
    for name in ("base", "changed"):
        before, after = source["input_cuts"][name], source["post_cuts"][name]
        require(np.array_equal(ports[name]["residual"], np.concatenate([slot(before, "text"), slot(before, "image")], axis=1)), "native residual port differs from source cut")
        merged = typed["quantize_bf16"](ports[name]["residual"] + typed["quantize_bf16"](ports[name]["gate"] * ports[name]["projection"]))
        actual = np.concatenate([slot(after, "text"), slot(after, "image")], axis=1)
        native_merge_checks[name] = bool(np.array_equal(merged, actual))
        require(native_merge_checks[name], "native BF16 merge equation does not close")
        h["public_receipt"](source["prefix_receipts"][name])
        if shape_mode:
            receipt = source["prefix_receipts"][name]
            require(read(data / "typed-store/receipts" / (receipt["fingerprint"] + ".json")) == receipt
                    and receipt["operation"] == "continue" and receipt["steps"] == 1
                    and receipt["parent"] == before and receipt["result"] == after, "selected block prefix lineage differs")
    rows = {row["id"]: row for row in observed["rows"]}
    masks = ["000", "001", "010", "011"] if recipient_mode else ["".join(bits) for bits in itertools.product("01", repeat=3)]
    attention_masks = [mask for mask in masks if mask not in {"000", "111"}] if plan.get("attention_sources") else []
    self_gains = {"self.11": [1., 1.], "self.10": [1., 0.], "self.01": [0., 1.],
        "self.00": [0., 0.], "self.1h": [1., .5], "self.h1": [.5, 1.]}
    if recipient_mode:
        require(plan["recipient_gain_rows"] == self_gains, "recipient gain inventory changed")
    self_names = {"self.native", *self_gains} if recipient_mode else set()
    require(set(rows) == {"base", "changed", "copy.sham", *("merge." + mask for mask in masks), *("attention." + mask for mask in attention_masks), *self_names}
            and len(rows) == plan["max_images"] == summary["counts"]["images"], "row inventory mismatch")
    pixels, latents, diagnostics, recipe_checks, reset_checks = {}, {}, {}, {}, {}
    for name, row in rows.items():
        require(read(data / "rows" / (name + ".json")) == row, "row observation mismatch")
        image_path = data / row["image"]["path"]
        require(sha(image_path.read_bytes()) == row["image"]["sha256"], "PNG digest mismatch")
        pixels[name] = np.asarray(Image.open(image_path).convert("RGB"))
        require(pixels[name].shape == (512, 512, 3), "RGB geometry mismatch")
        diagnostics[name] = typed["image_diagnostics"](pixels[name])
        latents[name] = {}
        for saved in row["schedulers"]:
            path = data / saved["path"]
            value = np.load(path, allow_pickle=False)
            require(sha(path.read_bytes()) == saved["sha256"] and typed["matches"](value, saved["descriptor"], bf16=True), "scheduler descriptor mismatch")
            latents[name][saved["step"]] = value
        expected_steps = {2, 3, 4} if shape_mode else {1, 2, 3, 4}
        require(set(latents[name]) == expected_steps and len(row["schedulers"]) == len(expected_steps), "incomplete continuation")
        self_row = recipient_mode and name in self_names
        if recipient_mode:
            require(row["continuation_policy"] == ("recipient-own-C" if self_row else "common-base-reset"), "row continuation policy differs")
            expected_c = fields(row["parent"])["conditioning"]
            require(all(expected_c[key] == row["conditioning_at_start"][key] for key in ("shape", "dtype", "sha256"))
                    and row["conditioning_source_cut"] == source["origin_cuts"]["base" if name == "base" else "changed"],
                    "row initial conditioning declaration differs")
        if self_row:
            require(row["continuation_acts"] == [], "recipient-only branch imported a future-source Act")
            own_condition = fields(source["origin_cuts"]["changed"])["conditioning"]
            require(fields(row["parent"])["conditioning"] == fields(row["final_cut"])["conditioning"] == own_condition,
                    "recipient-only branch did not retain own C")
            require(row["source_cut"] == source["input_cuts"]["changed"]
                    and all(own_condition[key] == row["own_conditioning"][key] for key in ("shape", "dtype", "sha256")),
                    "recipient-only operand/conditioning provenance differs")
            require(np.array_equal(slot(row["final_cut"], "latent"), latents[name][4]), "self final cut scheduler binding differs")
            reset_checks[name] = {"reset_count": 0, "own_conditioning_exact": True, "conditioning_source_cut": source["origin_cuts"]["changed"]}
        elif shape_mode:
            require(len(row["continuation_acts"]) == 1, "missing common conditioning reset")
            action = row["continuation_acts"][0]
            reset_spec = source["conditioning_reset"]
            require(reset_spec["step"] == plan["reset_step"] and reset_spec["phase"] == "project"
                    and reset_spec["source_cut"] == source["origin_cuts"]["base"] and reset_spec["slot"] == "conditioning"
                    and action["source_cut"] == reset_spec["source_cut"] and action["undeclared_slots_exact"] is True,
                    "conditioning reset source/clock declaration differs")
            base_condition = fields(reset_spec["source_cut"])["conditioning"]
            require(all(base_condition[key] == reset_spec["value_descriptor"][key] for key in ("shape", "dtype", "sha256")), "conditioning reset descriptor differs")
            receipt = replacement_closure(action, fields, slot="conditioning", expected=base_condition,
                stored_receipt=read(data / "typed-store/receipts" / (action["receipt_id"] + ".json")),
                unpack=typed["unpack"], public_receipt=h["public_receipt"])
            require(action["receipt_id"] == receipt["fingerprint"], "conditioning reset receipt ID differs")
            point = typed["unpack"](store["cuts"][action["parent"]]["execution_point"])
            require(point == action["point"] == typed["unpack"](store["cuts"][action["post_act"]]["execution_point"])
                    and point["logical_step"] == plan["reset_step"] and point["phase"] == "project", "conditioning reset occurrence differs")
            require(fields(action["parent"])["conditioning"] == fields(row["parent"])["conditioning"]
                    and fields(row["final_cut"])["conditioning"] == base_condition, "branch conditioning history/final common source differs")
            require(np.array_equal(slot(action["parent"], "latent"), latents[name][2])
                    and np.array_equal(slot(row["final_cut"], "latent"), latents[name][4]), "reset/final cut scheduler binding differs")
            reset_checks[name] = {"receipt_id": receipt["fingerprint"], "source_cut": reset_spec["source_cut"],
                                  "point": point, "conditioning_only_exact": True, "final_common_conditioning": True}
        if row["kind"] == "reference":
            require(row["parent"] == source["post_cuts"]["changed" if self_row else name], "reference parent mismatch")
            continue
        require(row["parent"] == source["post_cuts"]["changed"], "intervention parent mismatch")
        before_fields, post_fields = fields(row["parent"]), fields(row["post_act"])
        require(all(post_fields[key] == value for key, value in before_fields.items() if key != "image"), "undeclared slot changed")
        h["public_receipt"](row["receipt"])
        receipt = read(data / "typed-store/receipts" / (row["receipt"]["fingerprint"] + ".json"))
        require(receipt == row["receipt"] and row["act"]["writes"] == ["image"], "Act receipt mismatch")
        if shape_mode:
            replacement_closure(row, fields, slot="image", expected=post_fields["image"], stored_receipt=receipt,
                                unpack=typed["unpack"], public_receipt=h["public_receipt"])
        actual_image = slot(row["post_act"], "image")
        if row["kind"] == "copy":
            require(np.array_equal(actual_image, slot(row["parent"], "image")), "copy sham changed image")
            continue
        residual = ports["changed"]["residual"].copy()
        operands = ports["changed"]["operands"].copy()
        if self_row:
            require(row["kind"] == "self" and row["gains"] == self_gains[name]
                    and row["recipe_mode"] == "recipient-only-scale", "recipient-only recipe declaration differs")
            a_gain, m_gain = self_gains[name]
            operands[:, text_rows:, :attention_width] = typed["quantize_bf16"](operands[:, text_rows:, :attention_width] * a_gain)
            operands[:, text_rows:, attention_width:] = typed["quantize_bf16"](operands[:, text_rows:, attention_width:] * m_gain)
        elif row["kind"] == "attention":
            selected = arrays(row["positioned_qkv"])
            qkv = {name: arrays(source["ports"][name]["positioned_qkv"]) for name in ("base", "changed")}
            mask = row["mask_Q_TKV_IKV"]
            selected_ports = row["positioned_qkv"]["selected_ports"]
            require(typed["matches"](qkv["base" if mask[0] == "1" else "changed"]["q"], selected_ports["q"], bf16=True), "wrong query source")
            for key in ("k", "v"):
                expected = qkv["changed"][key].copy()
                if mask[1] == "1":
                    expected[:, :text_rows] = qkv["base"][key][:, :text_rows]
                if mask[2] == "1":
                    expected[:, text_rows:] = qkv["base"][key][:, text_rows:]
                require(typed["matches"](expected, selected_ports[key], bf16=True), "wrong K/V source selection")
            residual[:, text_rows:] = ports["base"]["residual"][:, text_rows:]
            operands[:, text_rows:, :attention_width] = selected["attention"][:, text_rows:]
        else:
            mask = row["mask_R_A_M"]
            if mask[0] == "1":
                residual[:, text_rows:] = ports["base"]["residual"][:, text_rows:]
            if mask[1] == "1":
                operands[:, text_rows:, :attention_width] = ports["base"]["operands"][:, text_rows:, :attention_width]
            if mask[2] == "1":
                operands[:, text_rows:, attention_width:] = ports["base"]["operands"][:, text_rows:, attention_width:]
        require(typed["matches"](operands, row["operands_descriptor"], bf16=True), "operand recipe mismatch")
        projected = arrays(row["projection"])["value"]
        result = typed["quantize_bf16"](residual + typed["quantize_bf16"](ports["changed"]["gate"] * projected))
        require(np.array_equal(result[:, text_rows:], actual_image), "BF16 merge residual equation differs from installed image")
        recipe_checks[name] = {"operands_and_residual_equation_verified": True,
            "text_rows_exact_changed": bool(np.array_equal(result[:, :text_rows], slot(source["post_cuts"]["changed"], "text")))}
    attention_joins = {}
    if plan.get("attention_sources"):
        attention_joins = read(data / "attention-joins.json")
        require(set(attention_joins) == {"000", "111"}, "incomplete attention self joins")
        for mask, join in attention_joins.items():
            selected = arrays(join["ports"])
            source_name = "changed" if mask == "000" else "base"
            require(np.array_equal(selected["attention"], ports[source_name]["operands"][:, :, :attention_width]), "native attention self reconstruction differs")
            target_row = rows[join["equivalent_row"]]
            require(typed["matches"](slot(target_row["post_act"], "image"), join["image_descriptor"], bf16=True), "attention corner does not join selected merge image")
            require(join["projection_descriptor"]["sha256"] == target_row["projection"]["descriptors"]["value"]["sha256"], "attention corner projection join differs")
    receipts, continue_edges = {}, {}
    for path in (data / "typed-store/receipts").glob("*.json"):
        receipt = read(path)
        h["public_receipt"](receipt)
        require(path.stem == receipt["fingerprint"] and receipt["model_identity"] == source["model_identity"], "receipt identity mismatch")
        receipts[path.stem] = receipt
        if shape_mode and receipt["operation"] == "continue":
            require(receipt["parent"] not in continue_edges or continue_edges[receipt["parent"]] == receipt, "ambiguous native continuation edge")
            continue_edges[receipt["parent"]] = receipt
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    require([row["sequence"] for row in emissions] == list(range(len(emissions))) and len(emissions) == summary["counts"]["receipt_emissions"], "receipt emission sequence mismatch")
    emitted = [receipts[row["receipt"]] for row in emissions]
    require(sum(row.get("steps", 0) for row in emitted if row["operation"] == "continue") == summary["counts"]["adapter_steps"]
            and sum(row["operation"] != "continue" for row in emitted) == summary["counts"]["acts"], "emitted native work census mismatch")
    shape_native_paths = {}
    if shape_mode:
        for name, row in rows.items():
            start = row["parent"] if row["kind"] == "reference" else row["post_act"]
            final = row["final_cut"]
            point = typed["unpack"](store["cuts"][final]["execution_point"])
            require(point["logical_step"] == 4, "final retained cut has not completed the declared native horizon")
            reset = row["continuation_acts"][0]["receipt"] if row["continuation_acts"] else None
            path = native_path(start, final, continue_edges, reset=reset, limit=plan["max_adapter_steps"])
            shape_native_paths[name] = {"start": start, "final_cut": final, "native_receipts": path,
                                        "native_steps": sum(receipts[key]["steps"] for key in path), "reset_count": int(reset is not None)}
    compare = routes["comparison"]
    comparisons = {name: {"rgb": compare(pixels[name], pixels["base"], pixels["changed"]),
        "scheduler_trajectory": {str(step): compare(value, latents["base"][step], latents["changed"][step]) for step, value in latents[name].items()}} for name in rows}
    shape_controls = {}
    if shape_mode:
        shape_controls = {name + "_changed_future_exact": bool(np.array_equal(pixels[name], pixels["changed"]))
            and all(np.array_equal(latents[name][step], latents["changed"][step]) for step in (2, 3, 4))
            for name in ("copy.sham", "merge.000")}
        if recipient_mode:
            shape_controls["self.11_own_native_future_exact"] = bool(np.array_equal(pixels["self.11"], pixels["self.native"])) and all(
                np.array_equal(latents["self.11"][step], latents["self.native"][step]) for step in (2, 3, 4))
            shape_controls["self.11_post_image_equals_recipient_native"] = bool(np.array_equal(
                slot(rows["self.11"]["post_act"], "image"), slot(source["post_cuts"]["changed"], "image")))
        else:
            shape_controls["merge.111_post_image_equals_native_base"] = bool(np.array_equal(
                slot(rows["merge.111"]["post_act"], "image"), slot(source["post_cuts"]["base"], "image")))
    recipient_comparisons = {}
    if recipient_mode:
        for name in sorted(self_names):
            recipient_comparisons[name] = {
                "rgb": compare(pixels[name], pixels["self.native"], pixels["self.native"]),
                "scheduler_trajectory": {str(step): compare(value, latents["self.native"][step], latents["self.native"][step])
                    for step, value in latents[name].items()},
                "scope": "Distances from the matched own-C self.native future; target projection undefined. Global base/changed geometry uses a different future C policy and is not a matched causal effect for these rows."}
    return {"schema": "mdb-flux-merge-audit-v1", "source": run["source"], "counts": summary["counts"],
        "selected_block": plan["block"], "model_identity": source["model_identity"], "mechanics_status": "valid", "terminal_status": "unassessed",
        "native_merge_equation_checks": native_merge_checks, "recipe_checks": recipe_checks, "attention_joins": attention_joins,
        "typed_store": store["counts"], "unique_receipts": len(receipts), "receipt_emissions": len(emissions),
        "comparisons": comparisons, "image_diagnostics": diagnostics, "sealed_hypotheses": hypotheses,
        **({"recipient_comparisons": recipient_comparisons} if recipient_mode else {}),
        **({"panel": plan["panel"], "source_prefix_paths": shape_prefix_paths, "conditioning_history": conditioning_history,
            "conditioning_resets": reset_checks, "native_paths": shape_native_paths,
            "scheduler_steps": [2, 3, 4], "control_observations": shape_controls,
            "equations_or_models_executed": False} if shape_mode else {}),
        "scope": plan["scope"], "auditor_source_sha256": sha(Path(__file__).read_bytes()),
        "audit_helpers": {name: sha((here / name).read_bytes()) for name in ("summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")},
        "instrument_limitations": "Projection and attention outputs are retained shared-kernel calculations, not recomputed offline from external weights or SDPA. Offline audit verifies port hashes, Q/K/V source selection, exact native attention self joins, operand selection, BF16 residual equations, typed closure and full numerical/RGB futures. No independent mathematical forecast of edited images."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "native_merge_equations": result["native_merge_equation_checks"], "counts": result["counts"]}))
