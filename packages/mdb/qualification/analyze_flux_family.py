"""Independent saved-evidence audit of one resident six-panel Klein campaign.

No native model, external weights, SDPA, scheduler or equation module is run.
Projection tensors are retained observations; BF16 residual installation, source
selection, native futures and the pure prefix resolver are audited separately.
Numerical/categorical disagreements remain observations, never scientific gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import zipfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from mdb.forecast_transaction import load_forecast_tree, verify_transaction


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def decode(root, value):
    def containers(item):
        if isinstance(item, Mapping):
            return {key: containers(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [containers(child) for child in item]
        return item
    return containers(load_forecast_tree(root, value))


def close(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(close(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, (tuple, list)) and len(actual) == len(expected) and all(close(a, b) for a, b in zip(actual, expected, strict=True))
    if type(expected) is float:
        return type(actual) in {int, float} and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    return actual == expected


def prefix_signature(a, b, width, text_rows):
    """NumPy recalculation of the declared prefix-only scalar features."""
    def rms(value):
        maximum = float(np.abs(value).max())
        return maximum * math.sqrt(float(np.mean(np.square(value / maximum)))) if maximum else 0.

    def relationship(first, second):
        left, right = rms(first), rms(second)
        cosine = float(np.mean((first / left) * (second / right))) if left and right else 0.
        return {"mean_dot": float(np.mean(first * second)), "cosine": max(-1., min(1., cosine)), "either_zero": float(not left or not right)}

    pairs = {"R": (a["residual"][:, text_rows:], b["residual"][:, text_rows:]),
        "A": (a["operands"][:, text_rows:, :width], b["operands"][:, text_rows:, :width]),
        "M": (a["operands"][:, text_rows:, width:], b["operands"][:, text_rows:, width:])}
    pairs = {name: tuple(value.astype(np.float64) for value in values) for name, values in pairs.items()}
    features = {}
    for name, (base, changed) in pairs.items():
        before, after, delta = rms(base), rms(changed), rms(changed - base)
        denominator = max(before, after)
        relative = (delta / denominator) / (before / denominator + after / denominator) if denominator else 0.
        features.update({name + "_base_rms": before, name + "_changed_rms": after,
                         name + "_delta_rms": delta, name + "_symmetric_relative_rms_delta": relative})
        features.update({name + "_base_changed_" + key: value for key, value in relationship(base, changed).items()})
    features["R_A_widths_match"] = float(pairs["R"][0].shape[-1] == width)
    if features["R_A_widths_match"]:
        for label, first, second in (("base_R_A", pairs["R"][0], pairs["A"][0]), ("changed_R_A", pairs["R"][1], pairs["A"][1]),
                                     ("delta_R_A", pairs["R"][1] - pairs["R"][0], pairs["A"][1] - pairs["A"][0])):
            features.update({label + "_" + key: value for key, value in relationship(first, second).items()})
    return features


def selected_qkv(base, changed, text_rows, recipe):
    """Select saved positioned operands only; never calculate attention."""
    result = {name: changed[name].copy() for name in ("q", "k", "v")}
    sequence, heads = result["q"].shape[1:3]
    selected_heads = recipe.get("heads", list(range(heads)))
    text = recipe.get("text_positions", list(range(text_rows)))
    image = [text_rows + index for index in recipe.get("image_positions", list(range(sequence - text_rows)))]
    for field in ("q", "k_text", "v_text", "k_image", "v_image"):
        if recipe.get(field, "changed") == "base":
            rows = text + image if field == "q" else text if field.endswith("text") else image
            if rows and selected_heads:
                row_index, head_index = np.asarray(rows)[:, None], np.asarray(selected_heads)[None, :]
                result[field[0]][:, row_index, head_index, :] = base[field[0]][:, row_index, head_index, :]
    return result


def feature_diagnostics(recorded, recomputed):
    """Source-bound finite scalars are authoritative; reduction errors are observations."""
    require(set(recorded) == set(recomputed) and all(type(value) in {int, float} and math.isfinite(value)
            for values in (recorded, recomputed) for value in values.values()), "prefix feature inventory/nonfinite scalar differs")
    deltas = {key: recomputed[key] - recorded[key] for key in recorded}
    return {"recorded": recorded, "recomputed": recomputed, "signed_recompute_deltas": deltas,
            "max_abs_recompute_delta": max(map(abs, deltas.values()), default=0.),
            "scope": "NumPy versus recorded Torch reductions; diagnostic only, no calibrated scalar error gate"}


def array_diagnostic(hybrid, natural):
    """Signed hybrid-minus-natural differences; geometry mismatch stays visible."""
    hybrid, natural = np.asarray(hybrid), np.asarray(natural)
    result = {"hybrid_shape": list(hybrid.shape), "natural_shape": list(natural.shape),
              "hybrid_dtype": str(hybrid.dtype), "natural_dtype": str(natural.dtype)}
    if hybrid.shape != natural.shape:
        return {**result, "status": "different-geometry", "exact": False}
    delta = hybrid.astype(np.float64) - natural.astype(np.float64)
    return {**result, "status": "compared", "exact": bool(np.array_equal(hybrid, natural)),
            "max_abs": float(np.abs(delta).max()) if delta.size else 0.,
            "mean_abs": float(np.abs(delta).mean()) if delta.size else 0.,
            "signed_mean": float(delta.mean()) if delta.size else 0., "l2": float(np.linalg.norm(delta.ravel()))}


def resolver_forecasts(pairs, signatures, targets, rules):
    """Heldout target values are deliberately outside both fitted models."""
    samples = [{"pair_id": pair["id"], "signature": signatures[pair["id"]], "targets": targets[pair["id"]]}
               for pair in pairs if pair["split"] == "development"]
    model = rules["fit_resolver"](samples)
    companion = rules["fit_resolver"](samples, features=rules["RECIPIENT_FEATURES"])
    heldout = [pair["id"] for pair in pairs if pair["split"] == "heldout"]
    return {"model": model, "recipient_only_model": companion,
            "predictions": {name: rules["resolve"](model, signatures[name]) for name in heldout},
            "recipient_only_predictions": {name: rules["resolve"](companion, signatures[name]) for name in heldout}}


def audit(run_dir):
    from PIL import Image

    directory, root = Path(__file__).parent, Path(run_dir).resolve()
    helper_names = ("flux_family_custody.py", "flux_family_plan.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")
    provenance_helpers = (*helper_names, "summarize_context_window.py")
    custody, rules, typed, routes = (runpy.run_path(str(directory / name)) for name in helper_names)
    request = read(root / "request.json")
    plan = request["config"]
    verified = custody["verify"](root, plan["limits"])
    envelope = read(root / "report.json")
    require(envelope["archive_root"] == "flux-family-result", "unexpected family archive root")
    data = root / "extracted/flux-family-result"
    source, summary = read(data / "sources.json"), read(data / "summary.json")
    require(read(data / "plan.json") == plan and summary == envelope["report"]
            and summary["model_identity"] == source["model_identity"]
            and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "plan/model/report binding differs")
    require(sha(directory / "flux_family_plan.py") == request["source_hashes"]["flux_family_plan.py"], "pure resolver source differs from declared campaign")
    declared = rules["make_plan"](plan)
    require({key: plan.get(key) for key in declared} == declared, "declared campaign geometry differs")
    require(summary["weight_versions"]["initial"] == summary["weight_versions"]["final"]
            and not summary["RGB_forecast"], "model identity/version or forecast scope differs")
    store = data / "typed-store"
    checked = typed["audit_store"](store)
    cuts = checked["cuts"]
    execution = cuts[source["histories"]["anchor.base"]["before"]]["execution"]
    require(execution["device"]["type"] == "cuda" and execution["dtype"] == "torch.bfloat16" and execution["batch"]["size"] == 1
            and all(cut["execution"] == execution and cut["model_identity"] == source["model_identity"] for cut in cuts.values()), "typed store mixes model/execution profiles")

    def slots(cut):
        require(cut in cuts and cuts[cut]["model_identity"] == source["model_identity"], "unknown/model-mismatched typed cut")
        return {item["key"]["value"]: item["value"] for item in cuts[cut]["payload"]["items"]}

    def slot(cut, name):
        return routes["slot_array"](store, cut, name, typed)

    def path(relative):
        value = Path(relative)
        require(not value.is_absolute() and ".." not in value.parts, "unsafe artifact path")
        target = data / value
        require(target.is_file() and not target.is_symlink(), "missing/nonregular artifact")
        return target

    def npz(record):
        target = path(record["path"])
        require(sha(target) == record["sha256"], "port/projection/attention file differs")
        with zipfile.ZipFile(target) as archive:
            members = archive.infolist()
            require(sum(item.file_size for item in members) <= 512 << 20
                    and all(item.file_size <= 256 << 20 for item in members), "NPZ expanded byte bound exceeded")
        with np.load(target, allow_pickle=False) as saved:
            arrays = dict(saved)
        require(set(arrays) == set(record["descriptors"]) and all(value.dtype == np.float32
                and typed["matches"](value, record["descriptors"][name], bf16=True) for name, value in arrays.items()), "retained native port descriptor differs")
        return arrays

    @lru_cache(maxsize=3)
    def ports(history):
        return npz(source["histories"][history]["ports"])

    def receipt(value):
        require(value["fingerprint"] == identity({key: item for key, item in value.items() if key != "fingerprint"})
                and read(store / "receipts" / (value["fingerprint"] + ".json")) == value
                and value["model_identity"] == source["model_identity"], "public native/Act receipt differs")

    def replace_receipt(row, slot_name):
        value = row["receipt"]
        receipt(value)
        require(value["operation"] == "apply" and value["parent"] == row["parent"] and value["result"] == row["post_act"]
                and value["act"] == row["act"] and row["act"]["writes"] == [slot_name]
                and row["act"]["operation"]["id"] == "saturn.builtin.replace", "Act root/name/slot binding differs")
        before, after = slots(row["parent"]), slots(row["post_act"])
        require(set(before) == set(after) and all(before[key] == after[key] for key in before if key != slot_name), "Act changed undeclared payload closure")
        written = typed["unpack"](value["writes"])[slot_name]
        parameter = typed["unpack"](row["act"]["parameters"])["value"]
        require(written == after[slot_name] and parameter == after[slot_name], "Act parameter/write does not join typed result")

    histories = source["histories"]
    require(set(histories) == {history["id"] for history in plan["histories"]}
            and set(source["pairs"]) == {pair["id"] for pair in plan["pairs"]}, "source history/pair inventory differs")
    native_equations = {}
    for declared_history in plan["histories"]:
        name, history = declared_history["id"], histories[declared_history["id"]]
        require(all(history[key] == value for key, value in declared_history.items()) and history["boundary"] == plan["selector"], "source history plan differs")
        point = typed["unpack"](cuts[history["before"]]["execution_point"])
        require(all(point[key] == value for key, value in plan["selector"].items()), "source cut is not the declared single4 occurrence")
        value = history["prefix_receipt"]
        receipt(value)
        require(value["operation"] == "continue" and value["parent"] == history["before"] and value["result"] == history["after"] and value["steps"] == 1, "source prefix receipt does not join before/after")
        arrays, text_rows = ports(name), history["text_rows"]
        require(set(arrays) == {"residual", "operands", "projection", "gate"} | ({"q", "k", "v"} if name in {"anchor.base", "anchor.hybrid"} else set()), "source port inventory differs")
        before = np.concatenate([slot(history["before"], "text"), slot(history["before"], "image")], axis=1)
        after = np.concatenate([slot(history["after"], "text"), slot(history["after"], "image")], axis=1)
        calculated = typed["quantize_bf16"](arrays["residual"] + typed["quantize_bf16"](arrays["gate"] * arrays["projection"]))
        require(np.array_equal(before, arrays["residual"]) and np.array_equal(after, calculated)
                and slot(history["before"], "text").shape[1] == text_rows, "native source residual equation/typed port join differs")
        native_equations[name] = {"exact": True, "projection_weight": history["projection_weight"], "projection_bias": history["projection_bias"],
            "scope": "retained native projection plus BF16 gated residual equation; external projection matrix not reevaluated offline"}
        require(history["projection_weight"] == histories["anchor.base"]["projection_weight"]
                and history["projection_bias"] == histories["anchor.base"]["projection_bias"], "selected projection parameter identity differs across histories")
        if name == "anchor.hybrid":
            initial = history["initial_act"]
            adapted = {**initial, "post_act": history["initial"]}
            replace_receipt(adapted, "text")
            require(initial["parent"] == histories["anchor.base"]["initial"]
                    and slots(history["initial"])["text"] == slots(histories["anchor.relation"]["initial"])["text"], "hybrid working-text source differs")
        else:
            require(history["initial_act"] is None, "coherent history has an undeclared initial Act")

    pair_features = {}
    for pair in plan["pairs"]:
        actual = source["pairs"][pair["id"]]
        require(all(actual[key] == value for key, value in pair.items()) and actual["closure_exceptions"] == ["conditioning", "image", "text"], "pair source/recipient/closure declaration differs")
        left, right = (histories[pair[key]] for key in ("source", "recipient"))
        before, after = slots(left["before"]), slots(right["before"])
        require(set(before) == set(after) and all(before[key] == after[key] for key in before if key not in actual["closure_exceptions"]), "cross-history non-exempt closure differs")
        a, b = ports(pair["source"]), ports(pair["recipient"])
        require(np.array_equal(a["gate"], b["gate"]), "recipient/source gate differs")
        feature = prefix_signature(a, b, right["attention_width"], right["text_rows"])
        cos_key = "recipient_conditioning_working_cosine"
        condition_cut = right["initial_act"]["parent"] if right["initial_act"] else right["initial"]
        condition = slot(condition_cut, "text").astype(np.float64)
        working = slot(right["before"], "text").astype(np.float64)
        denominator = float(np.linalg.norm(condition.ravel()) * np.linalg.norm(working.ravel()))
        cosine = float(np.sum(condition * working) / denominator) if denominator else 0.
        feature[cos_key] = cosine
        pair_features[pair["id"]] = {**feature_diagnostics(actual["signature"], feature),
                                    "conditioning_projection_cut": condition_cut}
    corner_checks = {}
    require(set(source["corner_checks"]) == {"base", "changed"}, "native attention/projection corner inventory differs")
    for name, history_name in (("base", "anchor.base"), ("changed", "anchor.hybrid")):
        record = source["corner_checks"][name]
        arrays = ports(history_name)
        attention = npz(record["attention"])["value"]
        require(record["native_exact"] is True and np.array_equal(attention, arrays["operands"][..., :histories[history_name]["attention_width"]])
                and typed["matches"](arrays["projection"], record["projection_descriptor"], bf16=True), "retained attention/projection reconstruction corner differs")
        corner_checks[name] = {"attention_equals_native_operand_A": True, "projection_descriptor_equals_native": True,
                               "scope": "retained native reconstruction witnesses; no offline SDPA or full matrix evaluation"}
    supports = source["support_selection"]
    require(read(data / "support_selection.json") == supports, "prefix supports publication differs")
    base, changed = ports("anchor.base"), ports("anchor.hybrid")
    tr, width, heads = (histories["anchor.base"][name] for name in ("text_rows", "attention_width", "heads"))
    delta = (changed["operands"][:, tr:, :width].astype(np.float64) - base["operands"][:, tr:, :width]).reshape(1, -1, heads, width // heads)
    energy = np.square(delta / np.abs(delta).max()) if np.any(delta) else np.zeros_like(delta)
    text_energy = (np.square(base["k"][:, :tr].astype(np.float64) - changed["k"][:, :tr])
                   + np.square(base["v"][:, :tr].astype(np.float64) - changed["v"][:, :tr])).sum(axis=(0, 2, 3))
    values = {"heads": energy.sum(axis=(0, 1, 3)), "image": energy.sum(axis=(0, 2, 3)), "text": text_energy}
    support_rank = {}
    for axis, energy_values in values.items():
        ranked = sorted(range(len(energy_values)), key=lambda index: (-energy_values[index], index))
        quarter = max(1, len(ranked) // 4)
        expected = {"top": ranked[:quarter], "alternative": ranked[-quarter:]}
        require(supports[axis] == expected, "supports do not follow source-prefix-only energy policy: " + axis)
        support_rank[axis] = {"selected": expected, "authority": "NumPy recomputed native prefix contrast energy; no future outcomes"}

    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    require([row["sequence"] for row in timeline] == list(range(len(timeline))), "timeline sequence differs")
    seals, transactions, observations = {}, {}, {}
    for phase, forecast_name in (("development", "hypotheses"), ("heldout", "applicability")):
        events = [row for row in timeline if row["kind"] == phase + "_sealed"]
        require(len(events) == 1, "missing/duplicate phase seal")
        seals[phase] = events[0]
        report = read(data / ("transaction-" + phase + "-report.json"))
        transaction = data / ("transaction-" + phase)
        verification = verify_transaction(transaction, expected_seal_sha256=report["seal_sha256"], expected_event_head=report["event_head"])
        require(verification == summary["verification"][phase] and events[0]["seal"] == summary["seals"][phase]
                and verification["seal_sha256"] == events[0]["seal"]["seal_sha256"], "public transaction/global seal differs")
        declaration = read(transaction / "declarations.json")
        require(declaration["model_identity"] == source["model_identity"] and declaration["input_identity"] == identity(plan)
                and declaration["source_identity"] == identity(request["source_hashes"])
                and declaration["frontier"]["source_sha256"] == sha(data / "sources.json")
                and decode(transaction, declaration["actual_weight_manifest"]) == {}, "external-model transaction identity differs")
        transactions[phase] = {"verification": verification, "work": report["work"]}
        observations[phase] = decode(transaction, read(transaction / "observations/native.json")["value"])
        forecast = decode(transaction, read(transaction / ("forecasts/" + forecast_name + ".json"))["predictions"])
        if phase == "development":
            require(forecast["plan"] == plan and forecast["source_sha256"] == sha(data / "sources.json")
                    and events[0]["counts"]["images"] == 0, "development hypotheses/source were not sealed before outcomes")
        else:
            resolver = read(data / "resolver.json")
            require(forecast == resolver and resolver["heldout_suffixes_before_seal"] == 0, "heldout resolver differs from seal")
    require(seals["development"]["sequence"] < seals["heldout"]["sequence"], "heldout seal preceded development")
    require(seals["heldout"]["counts"]["images"] == sum(row["split"] == "development" for row in plan["rows"]), "heldout seal lacks the complete development panel")
    ready = [row for row in timeline if row["kind"] == "all_prefixes_ready"]
    require(len(ready) == 1 and ready[0]["source_sha256"] == sha(data / "sources.json")
            and ready[0]["counts"]["images"] == 0 and ready[0]["sequence"] < seals["development"]["sequence"]
            and {row["history"] for row in timeline if row["kind"] == "prefix_ready"} == set(histories)
            and all(row["sequence"] < ready[0]["sequence"] for row in timeline if row["kind"] == "prefix_ready"), "source prefix/support record was not fixed before futures")
    planned_rows = {row["id"]: row for row in plan["rows"]}
    rows, pixels, latents, findings = {}, {}, {}, []
    for definition in plan["rows"]:
        name = definition["id"]
        row = read(data / "rows" / (name + ".json"))
        require(all(row.get(key) == value for key, value in definition.items()), "executed row differs from frozen plan: " + name)
        starts = [event for event in timeline if event["kind"] == "branch_start" and event["row"] == name]
        ends = [event for event in timeline if event["kind"] == "branch_complete" and event["row"] == name]
        require(len(starts) == len(ends) == 1 and seals[row["split"]]["sequence"] < starts[0]["sequence"] < ends[0]["sequence"], "row ran before applicable seal or lacks completion")
        require(row["parent"] in cuts, "row root not retained")
        if row["kind"] == "reference":
            require(row["parent"] == histories[row["history"]]["after"] and "act" not in row, "reference uses wrong native history")
        else:
            replace_receipt(row, "image")
            pair = source["pairs"][row["pair"]]
            a, b = ports(pair["source"]), ports(pair["recipient"])
            recipient = histories[pair["recipient"]]
            text_rows, attention_width = recipient["text_rows"], recipient["attention_width"]
            if row["kind"] == "rescue":
                broken, clean = (rows[key] for key in ("break.image", "kv.KV"))
                require(row["parent"] == broken["future_cuts"][row["when"]] and row["donor"] == clean["future_cuts"][row["when"]]
                        and row["selected_rows"] == supports["image"][row["support"]], "rescue source/point/selected rows differ")
                point = typed["unpack"](cuts[row["parent"]]["execution_point"])
                require(all(point[key] == value for key, value in plan["rescue_points"][row["when"]].items()), "rescue is at a different occurrence")
                require(typed["unpack"](cuts[row["donor"]]["execution_point"]) == point, "rescue donor and recipient clocks differ")
                candidate, donor = slot(row["parent"], "image").copy(), slot(row["donor"], "image")
                candidate[:, row["selected_rows"]] = donor[:, row["selected_rows"]]
            else:
                require(row["parent"] == recipient["after"], "edited row does not start from declared recipient")
                if row["kind"] == "copy":
                    candidate = slot(row["parent"], "image")
                else:
                    calculation, operands, residual = row["calculations"], b["operands"].copy(), b["residual"].copy()
                    require(calculation["gate_history"] == pair["recipient"], "projection uses wrong gate history")
                    if row["kind"] == "merge":
                        mask = row["mask"]
                        if mask[1] == "1":
                            operands[:, text_rows:, :attention_width] = a["operands"][:, text_rows:, :attention_width]
                        if mask[2] == "1":
                            operands[:, text_rows:, attention_width:] = a["operands"][:, text_rows:, attention_width:]
                    else:
                        recipe = dict(row.get("recipe", {"k_text": "base", "v_text": "base"}))
                        receive = None
                        if row["kind"] == "support":
                            selected = supports[row["axis"]][row["side"]]
                            if row["axis"] in {"heads", "text"}:
                                recipe["heads" if row["axis"] == "heads" else "text_positions"] = selected
                            else:
                                receive = selected
                        require(calculation["recipe"] == recipe and calculation["receive_rows"] == receive, "attention recipe/support metadata differs")
                        selected = selected_qkv(a, b, text_rows, recipe)
                        require(all(typed["matches"](value, calculation["selected_ports"][key], bf16=True) for key, value in selected.items()), "selected Q/K/V operand bytes differ")
                        attention = npz(calculation["attention"])["value"]
                        indices = list(range(operands.shape[1] - text_rows)) if receive is None else receive
                        operands[:, [text_rows + index for index in indices], :attention_width] = attention[:, [text_rows + index for index in indices]]
                        if row["kind"] == "break":
                            require(calculation["omit_rows"] == supports["image"]["top"], "break omission support differs")
                            indices = [text_rows + index for index in calculation["omit_rows"]]
                            operands[:, indices, :attention_width] = b["operands"][:, indices, :attention_width]
                        mask = "100" if row.get("restore_residual", True) else "000"
                    if mask[0] == "1":
                        residual[:, text_rows:] = a["residual"][:, text_rows:]
                    require(typed["matches"](operands, calculation["operands_descriptor"], bf16=True)
                            and typed["matches"](residual, calculation["residual_descriptor"], bf16=True), "merge residual/operand recipe differs")
                    projection = npz(calculation["projection"])["value"]
                    candidate = typed["quantize_bf16"](residual + typed["quantize_bf16"](b["gate"] * projection))[:, text_rows:]
            require(np.array_equal(candidate, slot(row["post_act"], "image")), "post-Act image differs from selected-row or BF16 residual recipe")
        require([value["step"] for value in row["schedulers"]] == [1, 2, 3, 4], "row lacks all four ordered scheduler observations")
        latents[name] = {}
        for saved in row["schedulers"]:
            target = path(saved["path"])
            value = np.load(target, allow_pickle=False)
            require(sha(target) == saved["sha256"] and value.dtype == np.float32 and typed["matches"](value, saved["descriptor"], bf16=True), "scheduler bytes/descriptor differs")
            if "inherited_from" in saved:
                require(row["kind"] == "rescue" and row["when"] == "late" and saved["step"] == 1
                        and saved["inherited_from"] == "break.image"
                        and {key: item for key, item in saved.items() if key != "inherited_from"} == rows["break.image"]["schedulers"][0], "late rescue inherited step is not the broken own-state prefix")
            latents[name][saved["step"]] = value
        target = path(row["image"]["path"])
        require(sha(target) == row["image"]["sha256"], "native PNG differs")
        with Image.open(target) as image:
            value = np.asarray(image).copy()
        require(value.dtype == np.uint8 and value.shape == (plan["height"], plan["width"], 3), "native RGB geometry differs")
        pixels[name], rows[name] = value, row
    require(set(rows) == set(planned_rows) and len(rows) == summary["counts"]["images"] == 47, "completed row/image inventory differs")
    for phase in ("development", "heldout"):
        observed_rows = list(observations[phase]["rows"])
        require({row["id"]: row for row in observed_rows} == {name: row for name, row in rows.items() if row["split"] == phase}, "durable transaction rows differ from saved row files")
    require(observations["development"]["source_sha256"] == sha(data / "sources.json")
            and dict(observations["heldout"]["counts"]) == summary["counts"], "observation/source/final work binding differs")
    require(not any(event["kind"] == "branch_start" and event["split"] == "heldout"
                    and event["sequence"] < seals["heldout"]["sequence"] for event in timeline), "heldout reference/branch leaked before seal")

    comparisons, effects = {}, {}
    for pair in plan["pairs"]:
        source_ref, recipient_ref = "ref." + pair["source"], "ref." + pair["recipient"]
        members = [source_ref, recipient_ref] + [name for name, row in rows.items() if row.get("pair") == pair["id"]]
        comparisons[pair["id"]] = {name: {"RGB": routes["comparison"](pixels[name], pixels[source_ref], pixels[recipient_ref]),
            "schedulers": {str(step): routes["comparison"](latents[name][step], latents[source_ref][step], latents[recipient_ref][step]) for step in range(1, 5)}} for name in members}
        p = {action: comparisons[pair["id"]][pair["id"] + "." + action]["schedulers"]["4"]["target_projection"] for action in rules["ACTIONS"]}
        defined = all(value is not None for value in p.values())
        effects[pair["id"]] = {"split": pair["split"], "contrast": pair["contrast"], "source_ref": source_ref, "recipient_ref": recipient_ref,
            "p": p, "axis": "0=source reference,1=recipient reference; projection is numerical latent geometry, not semantic correctness",
            "hypothesis_pRA_below_R_and_A": (p["RA"] < min(p["R"], p["A"])) if defined else None,
            "metric_status": "observed" if defined else "not-applicable-zero-native-reference-contrast"}
    rebuilt = resolver_forecasts(plan["pairs"], {name: pair["signature"] for name, pair in source["pairs"].items()},
                                {name: pair["p"] for name, pair in effects.items()}, rules)
    fitted = rebuilt["model"]
    require(close(resolver["model"], fitted) and list(resolver["development_rows"]) == sorted(name for name, row in rows.items() if row["split"] == "development"), "resolver fit consumes different development evidence")
    companion = rebuilt["recipient_only_model"]
    require(close(resolver["recipient_only_model"], companion), "recipient-only companion uses different development evidence")
    prediction_misses = {}
    for pair in plan["pairs"]:
        if pair["split"] != "heldout":
            continue
        name = pair["id"]
        expected = rebuilt["predictions"][name]
        require(close(resolver["predictions"][name], expected)
                and close(resolver["prefix_features"][name], source["pairs"][name]["signature"]), "heldout prediction not resolved from prefix-only features")
        actual_p, prediction = effects[name]["p"], expected["predictions"]
        prediction_misses[name] = {"sealed": expected, "observed_p": actual_p,
            "actual_minus_predicted": None if prediction is None else {action: None if actual_p[action] is None else actual_p[action] - prediction[action] for action in rules["ACTIONS"]},
            "scope": "empirical development-neighbor forecast; abstentions and counterexamples retained, not an independent edited-RGB prediction"}
        expected_companion = rebuilt["recipient_only_predictions"][name]
        require(close(resolver["recipient_only_predictions"][name], expected_companion), "recipient-only prediction differs from its frozen feature family")
        predicted = expected_companion["predictions"]
        prediction_misses[name]["recipient_only_companion"] = {"sealed": expected_companion,
            "actual_minus_predicted": None if predicted is None else {action: None if actual_p[action] is None else actual_p[action] - predicted[action] for action in rules["ACTIONS"]}}
    for name, row in rows.items():
        if row["kind"] == "copy":
            reference = "ref." + source["pairs"][row["pair"]]["recipient"]
            if not np.array_equal(pixels[name], pixels[reference]) or not all(np.array_equal(latents[name][step], latents[reference][step]) for step in range(1, 5)):
                findings.append({"row": name, "finding": "same-parent copy sham native future differs"})
    rescue_effects = {name: {"schedulers": {str(step): routes["comparison"](latents[name][step], latents["kv.KV"][step], latents["break.image"][step]) for step in range(1, 5)},
                            "RGB": routes["comparison"](pixels[name], pixels["kv.KV"], pixels["break.image"])}
                      for name, row in rows.items() if row["kind"] == "rescue"}
    hybrid = comparisons["hybrid"]
    p = lambda name: hybrid[name]["schedulers"]["4"]["target_projection"]
    support_effects = {axis: {side: p("support." + axis + "." + side) for side in ("top", "alternative")} for axis in supports}
    panel_observations = {"KV": {name: p("kv." + name) for name in ("K", "V", "KV", "Q.changedR")},
        "support_final_p": support_effects,
        "break_distance_to_clean_final": routes["comparison"](latents["break.image"][4], latents["kv.KV"][4], latents["break.image"][4])["base_l2"],
        "scope": "predeclared numerical contrasts, not named-object classifications or scientific hard gates"}
    hybrid_history, natural_history = histories["anchor.hybrid"], histories["anchor.relation"]
    prefix_payloads = {}
    for stage in ("initial", "before", "after"):
        hybrid_cut, natural_cut = hybrid_history[stage], natural_history[stage]
        hybrid_slots, natural_slots = slots(hybrid_cut), slots(natural_cut)
        prefix_payloads[stage] = {"hybrid_cut": hybrid_cut, "natural_cut": natural_cut,
            "fields": {name: {"hybrid_present": name in hybrid_slots, "natural_present": name in natural_slots,
                              "descriptor_equal": name in hybrid_slots and name in natural_slots and hybrid_slots[name] == natural_slots[name]}
                       for name in sorted(set(hybrid_slots) | set(natural_slots))}}
    hybrid_ports, natural_ports = ports("anchor.hybrid"), ports("anchor.relation")
    common_ports = set(hybrid_ports) & set(natural_ports)
    hybrid_natural = {"hybrid_history": "anchor.hybrid", "natural_history": "anchor.relation",
        "typed_payloads": prefix_payloads,
        "native_ports": {name: array_diagnostic(hybrid_ports[name], natural_ports[name]) for name in sorted(common_ports)},
        "ports_only_hybrid": sorted(set(hybrid_ports) - common_ports), "ports_only_natural": sorted(set(natural_ports) - common_ports),
        "reference_futures": {"schedulers": {str(step): array_diagnostic(latents["ref.anchor.hybrid"][step], latents["ref.anchor.relation"][step])
                                               for step in range(1, 5)},
                              "RGB": array_diagnostic(pixels["ref.anchor.hybrid"], pixels["ref.anchor.relation"])},
        "scope": "saved hybrid versus natural relation history; signed differences are hybrid minus natural; descriptor equality and divergence are observations, not gates or semantic equivalence claims"}
    hybrid_natural["first_observed_scheduler_divergence"] = next((step for step in range(1, 5)
        if not hybrid_natural["reference_futures"]["schedulers"][str(step)]["exact"]), None)
    hybrid_natural["matched_edits"] = {}
    for action in rules["ACTIONS"]:
        hybrid_name, natural_name = "hybrid." + action, "natural." + action
        hybrid_cut, natural_cut = rows[hybrid_name]["post_act"], rows[natural_name]["post_act"]
        hybrid_slots, natural_slots = slots(hybrid_cut), slots(natural_cut)
        schedulers = {str(step): array_diagnostic(latents[hybrid_name][step], latents[natural_name][step]) for step in range(1, 5)}
        hybrid_natural["matched_edits"][action] = {"hybrid_row": hybrid_name, "natural_row": natural_name,
            "hybrid_post_act_cut": hybrid_cut, "natural_post_act_cut": natural_cut,
            "post_act_fields": {name: {"hybrid_present": name in hybrid_slots, "natural_present": name in natural_slots,
                                       "descriptor_equal": name in hybrid_slots and name in natural_slots and hybrid_slots[name] == natural_slots[name]}
                                for name in sorted(set(hybrid_slots) | set(natural_slots))},
            "schedulers": schedulers, "RGB": array_diagnostic(pixels[hybrid_name], pixels[natural_name]),
            "first_observed_scheduler_divergence": next((step for step in range(1, 5) if not schedulers[str(step)]["exact"]), None)}
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    require([row["sequence"] for row in emissions] == list(range(len(emissions))), "physical receipt emission sequence differs")
    work = {"acts": 0, "adapter_steps": 0}
    for emission in emissions:
        value = read(store / "receipts" / (emission["receipt"] + ".json"))
        receipt(value)
        if value["operation"] == "continue":
            require(value["steps"] == len(value["transitions"]), "native receipt transition count differs")
            work["adapter_steps"] += value["steps"]
        else:
            require(value["operation"] == "apply", "unknown emission operation")
            work["acts"] += 1
    counts = summary["counts"]
    require(counts["receipt_emissions"] == len(emissions) and all(counts[key] == value for key, value in work.items())
            and counts["model_loads"] == counts["physical_batch"] == 1 and counts["prompt_encodings"] == len(plan["prompts"])
            and counts["prefix_projections"] == sum(history["initial_act"] is None for history in histories.values())
            and counts["rewinds"] == len(rows) + sum(history["initial_act"] is not None for history in histories.values())
            and counts["acts"] == sum(row["kind"] != "reference" for row in plan["rows"]) + 1
            and counts["projection_evaluations"] == len(corner_checks) + sum(row["kind"] in {"merge", "attention", "support", "break"} for row in plan["rows"])
            and counts["attention_evaluations"] == len(corner_checks) + sum(row["kind"] in {"attention", "support", "break"} for row in plan["rows"]), "physical work accounting differs")
    require(not summary["phase_native_entries"].get("development_forecast", 0) and not summary["phase_native_entries"].get("heldout_forecast", 0)
            and sum(summary["phase_native_entries"].values()) == counts["native_module_entries"], "native prediction-window guard accounting differs")
    reference_paths = {"request": root / "request.json", "transport_report": root / "report.json", "sources": data / "sources.json",
                       "summary": data / "summary.json", "resolver": data / "resolver.json", "timeline": data / "timeline.jsonl"}
    return {"schema": "mdb-flux-family-audit-v1", "mechanics_status": "valid", "custody": verified,
        "model_identity": source["model_identity"], "checkpoint_sha256": source["checkpoint_sha256"], "execution": execution, "plan": plan,
        "counts": counts, "store_counts": checked["counts"], "transactions": transactions, "native_equations": native_equations,
        "prefix_features": pair_features, "corner_checks": corner_checks, "support_selection": support_rank, "pair_effects": effects, "comparisons": comparisons,
        "rescue_effects": rescue_effects, "panel_observations": panel_observations,
        "hybrid_natural_prefix_comparison": hybrid_natural, "resolver": resolver, "prediction_misses": prediction_misses,
        "instrument_status": "control-findings-retained" if findings else "copy-shams-exact", "control_findings": findings,
        "image_diagnostics": {name: typed["image_diagnostics"](value) for name, value in pixels.items()},
        "images": {name: {**row["image"], "path": str((data / row["image"]["path"]).relative_to(root))} for name, row in rows.items()},
        "references": {name: {"path": str(value.relative_to(root)), "sha256": sha(value)} for name, value in reference_paths.items()},
        "audit_helpers": {name: sha(directory / name) for name in provenance_helpers}, "terminal_status": "unassessed",
        "new_model_calls": 0, "native_weight_evaluations": 0, "SDPA_evaluations": 0,
        "limits": [plan["scope"], "47 dependent futures share9 observed histories and one checkpoint; they are not47 independent inputs.",
            "External projection weights and attention kernels are not evaluated offline; retained tensors and source-selection descriptors are audited.",
            "Latent/RGB projections describe geometry, not object correctness. Zero reference contrast leaves p undefined and remains evidence.",
            "Color regions use uncalibrated red/blue hue thresholds; novel green/yellow objects may be unknown, never scored as scientific failure.",
            "The resolver is a small development-neighbor empirical predictor; every heldout reference and branch remains visible, including abstentions and misses."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    result = audit(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_name(args.output.stem + "-source.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    result["auditor_source_sha256"] = sha(snapshot)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"mechanics_status": result["mechanics_status"], "images": len(result["images"]), "new_model_calls": 0}))


if __name__ == "__main__":
    main()
