"""Fresh conditional Qwen contrast-carrying/neutral signed-write experiment.

One source-only decomposition rule and one global seal precede every native
MLP/readout target. Ordinary eight-step futures are observations, not forecasts.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "mdb-qwen-contrast-collateral-v1"
RESULT = "qwen-contrast-collateral-result"
RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
HELPERS = {
    "qwen_spectral_readout_campaign.py": "8a50826ba0d1f879a93ed5e2a76533a9ec9f81187b26b27d0e1a846e1cdeae74",
    "qwen_spectral_readout_math.py": "11158458215504818147170b9d2b03522cc9a3d18f3440452c57ddd525a73eee",
    "qwen_spectral_parameter_recipes.py": "c518ebcdc9e0c881575b38e6286ad931baa8d7a7e9b3d230eb311830c3d94b8c",
    "flux_image_tail_microscope.py": "b9efb61874005c8b11307c197a2ea5299078b42d5b4ab4b5cc719f7a4a4a32bb",
    "qwen_contrast_components_math.py": "44fad64beae93b6bf1610f824621b32f198aff10e7c4614d6908555da5e3a1ee",
}
PROMPTS = [
    dict(id="card", text="A card has the word blue written in black ink. The word on the card is"),
    dict(id="sequence", text="Complete the sequence: 5, 10, 20, 40,"),
    dict(id="code", text="Python:\ndef double(value):\n    return"),
    dict(id="translation", text="Translate to French: house\nFrench:"),
]
CELLS = [dict(id="baseline", carrying=0., neutral=0.)]
for _axis in ("carrying", "neutral"):
    for _suffix, _dose in (("n1", -1.), ("n025", -.25), ("p025", .25), ("p1", 1.)):
        CELLS.append(dict(id=_axis+"-"+_suffix, carrying=_dose if _axis == "carrying" else 0., neutral=_dose if _axis == "neutral" else 0.))
FUTURES = ["baseline", "carrying-n1", "carrying-p1", "neutral-n1", "neutral-p1"]
COLLATERAL = ["OptionsResolver", "/stdc", "PointerException", "XCTestCase"]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False,
            default=lambda item: dict(item) if isinstance(item, Mapping) else _unsupported(item)); stream.write("\n")


def _unsupported(value):
    raise TypeError("unsupported JSON metadata: " + type(value).__name__)


def dependencies():
    result = {}
    for name, expected in HELPERS.items():
        path = Path(__file__).with_name(name)
        require(sha(path) == expected, "frozen source dependency changed: " + name)
        result[name] = runpy.run_path(str(path))
    return result


def make_config(base):
    config = {key: base[key] for key in ("model_path", "retrieval_script", "output_scratch") if key in base}
    config.setdefault("model_path", "/mnt/big/llm-models/Qwen2.5-0.5B-Instruct")
    config.setdefault("retrieval_script", "retrieve_weights.py"); config.setdefault("output_scratch", "/mnt/big/mdb/research")
    config.update(schema=SCHEMA, output_name=RESULT, prompts=copy.deepcopy(PROMPTS), cells=copy.deepcopy(CELLS), future_cells=list(FUTURES),
        consumer_strings=[" red", " blue"], collateral_strings=list(COLLATERAL), helper_source_sha256=dict(HELPERS), retrieval_script_sha256=RETRIEVER_SHA256,
        device="cpu", dtype="float32", attention="eager", cpu_threads=4, seed=2718, lanczos_steps=128,
        max_prompt_tokens=64, max_prefix_ops=16384, continuation_tokens=8, max_native_entries=256, max_wall_seconds=900,
        transaction_bytes=512 << 20, transport_limits=dict(max_expanded_bytes=1 << 30, max_member_bytes=512 << 20),
        common_norm="sqrt((norm(old_top_FP32)^2+norm(old_bottom_FP32)^2)/2)", task_quality_gate=False)
    return config


def validate_config(config):
    fixed = make_config(config)
    for key in fixed.keys() - {"model_path", "retrieval_script", "output_scratch"}:
        require(config.get(key) == fixed[key], "fixed contrast protocol differs: " + key)
    require(all(type(config.get(key)) is str and bool(config[key]) for key in ("model_path", "retrieval_script", "output_scratch")), "explicit model/retriever/output paths required")
    return dict(prefixes=4, native_MLP_bases=4, component_readout_targets=36, ordinary_prefill_adjudications=20,
        ordinary_clone_shams=4, spectral_antipode_targets=16, random_antipode_targets=0, autonomous_cached_calls_max=160,
        native_physical_entries_max=244, model_hydrations=1, full_vocabulary_forecast_rows=36)


def campaign(model, prompts, consumer_ids, output, settings, *, model_identity, source_identity, collateral_ids=None, decode=lambda values: str(values), setup=None):
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor

    h = dependencies(); B = h["qwen_spectral_readout_campaign.py"]; M = h["qwen_spectral_readout_math.py"]; A = h["qwen_contrast_components_math.py"]
    layer = model.model.layers[-1]; hidden = model.config.hidden_size; eps = float(model.config.rms_norm_eps)
    require(all(value.dtype == torch.float32 and value.device.type == "cpu" for value in model.parameters()), "original CPU FP32 native model required")
    require(model.lm_head.bias is None and all(getattr(layer.mlp, name).bias is None for name in ("gate_proj", "up_proj", "down_proj")), "bias-free original MLP/head required")
    weights = M["prepare"]({name: value.detach().numpy() for name, value in dict(postnorm=layer.post_attention_layernorm.weight,
        gate=layer.mlp.gate_proj.weight, up=layer.mlp.up_proj.weight, down=layer.mlp.down_proj.weight, finalnorm=model.model.norm.weight).items()})
    head = M["readonly"](model.lm_head.weight.detach().numpy(), np.float32); red, blue = consumer_ids
    require(type(red) is type(blue) is int and red != blue and 0 <= min(red, blue) and max(red, blue) < len(head), "fixed distinct reader rows required")
    c = M["readonly"](weights["finalnorm"]*(head[red].astype(np.float64)-head[blue].astype(np.float64)))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if setup is not None:
        setup(output)
    started = time.monotonic(); phase = dict(released=False, prefix=False, lane="source")
    counts = dict(prefixes=0, native_MLP_bases=0, parity_targets=0, component_readouts=0, ordinary_prefills=0,
        ordinary_shams=0, cached_futures=0, physical_entries=0, final_postnorm_entries=0, final_MLP_entries=0, final_norm_entries=0, head_entries=0, denied_preseal_entries=0)
    events = []

    def event(kind, **detail):
        row = dict(number=len(events), kind=kind, **detail); events.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")

    def charge(name):
        counts[name] += 1; counts["physical_entries"] += 1
        require(counts["physical_entries"] <= settings["max_native_entries"] and time.monotonic()-started < settings["max_wall_seconds"], "finite native entry/wall allowance exhausted")

    def root_guard(_module, _args):
        if not phase["released"] or phase["lane"] not in ("ordinary", "sham", "future"):
            counts["denied_preseal_entries"] += 1; raise RuntimeError("undeclared ordinary native LM entry")
        charge({"ordinary": "ordinary_prefills", "sham": "ordinary_shams", "future": "cached_futures"}[phase["lane"]])

    def body_guard(_module, _args):
        require(phase["released"] or phase["prefix"], "native body outside genuine prefix/released targets")

    def tail_guard(name):
        def guard(_module, _args):
            if not phase["released"]:
                counts["denied_preseal_entries"] += 1; raise RuntimeError("native suffix before global seal")
            counts[name] += 1
        return guard

    hooks = [model.register_forward_pre_hook(root_guard), model.model.register_forward_pre_hook(body_guard)]
    hooks += [module.register_forward_pre_hook(tail_guard(name)) for name, module in (("final_postnorm_entries", layer.post_attention_layernorm),
        ("final_MLP_entries", layer.mlp), ("final_norm_entries", model.model.norm), ("head_entries", model.lm_head))]
    before = B["parameter_snapshot"](model); prefixes, parents, parent_signatures = {}, {}, {}
    order = [row["id"] for row in prompts]; require(len(set(order)) == len(order), "unique fresh context IDs required")
    eos = model.generation_config.eos_token_id; eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    try:
        search = M["lanczos"](M["even_action"](weights, c), hidden, steps=settings["lanczos_steps"], seed=settings["seed"])
        for prompt in prompts:
            key, ids = prompt["id"], prompt["ids"]
            require(ids.ndim == 2 and ids.shape[0] == 1 and 1 <= ids.shape[1] <= settings["max_prompt_tokens"], "bounded batch1 raw tokens required")
            positions = torch.arange(ids.shape[1]).unsqueeze(0)
            mask = torch.triu(torch.full((1, 1, ids.shape[1], ids.shape[1]), torch.finfo(torch.float32).min), diagonal=1)
            cache = DynamicCache(config=model.config); phase["prefix"] = True; charge("prefixes")
            try:
                capture = B["prefix_with_cache"](model, ids, mask, positions, cache, identity=model_identity,
                    execution=dict(backend="cpu", dtype="float32", attention="eager", threads=torch.get_num_threads()), maximum=settings["max_prefix_ops"])
            finally:
                phase["prefix"] = False
            require(cache.get_seq_length() == ids.shape[1] and len(B["cache_arrays"](cache)) == len(model.model.layers), "complete genuine prompt KV required")
            save(output / "prefixes" / (key+".json"), capture.trace)
            cut = capture.output["boundary"]["args"][0][:, -1:, :].detach().contiguous()
            parents[key] = cache; parent_signatures[key] = B["cache_snapshot"](cache)
            prefixes[key] = dict(text=prompt["text"], cut=cut.numpy().copy(), cut_descriptor=tensor_descriptor(cut), ids=ids.numpy().copy(), positions=positions.numpy().copy(), mask=mask.numpy().copy(),
                trace_fingerprint=capture.trace["fingerprint"], cache=[dict(key=a.numpy().copy(), value=b.numpy().copy()) for a, b in B["cache_arrays"](cache)], cache_descriptors=parent_signatures[key])
            event("genuine_prefix_complete", context=key, final_MLP_entries=counts["final_MLP_entries"], cache_digest=digest(parent_signatures[key]))
            del capture
        require(all(value == 0 for name, value in counts.items() if name not in ("prefixes", "physical_entries")), "pretarget suffix exclusion differs")
        input_identity = digest({key: value["cut_descriptor"] for key, value in prefixes.items()})
        tx = ForecastTransaction(output / "transaction", model_identity=model_identity, input_identity=input_identity, source_identity=source_identity,
            frontier=dict(kind="architectural", origin="genuine-native-prefix", scope="Own last-token pre-final-MLP state and prompt KV; contrast-dual endpoint writes"),
            execution_scope="Explicit guarded CPU FP32 eager model; source-only NumPy FP64 predictions; ordinary futures measured only; no weight bank",
            budgets=dict(max_bytes=settings["transaction_bytes"], max_wall_ms=settings["max_wall_seconds"]*1000, max_native_calls=settings["max_native_entries"]))
        with tx:
            tx.frontier("prefixes", prefixes)
            tx.frontier("source", dict(order=order, search=search, consumer=c, consumer_ids=list(consumer_ids), collateral_ids=collateral_ids or {}, pretarget_counts=dict(counts),
                factor_identity={name: dict(shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name, value in weights.items()},
                head_identity=dict(shape=list(head.shape), sha256=hashlib.sha256(memoryview(head).cast("B")).hexdigest())))

            def predict(context):
                planned, payloads = {}, {}
                for key in context.frontiers["source"]["order"]:
                    raw = context.frontiers["prefixes"][key]["cut"].reshape(-1); rho = float(np.linalg.norm(raw.astype(np.float64)))
                    old = {name: M["antipodal"](result["direction"], rho, weights, eps, c) for name, result in search["extremes"].items()}
                    axes = A["decompose"]({name: row["write"] for name, row in old.items()}, c)
                    original = M["mlp"](raw, weights, eps)
                    forecast = A["readout"](raw.astype(np.float64)+original["delta"], axes, weights["finalnorm"], head, eps, settings["cells"], consumer_ids, collateral_ids=collateral_ids)
                    planned[key] = dict(base=original, old_parity=old, axes=axes, readout=forecast, future_cells=list(settings["future_cells"]))
                    for axis, value in axes["realized"].items():
                        payloads[key+"."+axis] = value
                    for name, value in old.items():
                        for sign in ("positive", "negative"):
                            payloads[key+".old-"+name+"."+sign] = value[sign+"_input"]
                return dict(predictions=dict(order=order, contexts=planned, directions_shared_once=True, payloads_reinstantiated_by_rho=True,
                    continuation_forecast=False, scope="Full-vocabulary immediate conditional readouts; nine axial cells; native futures measured"), payloads=payloads)

            tx.predict("cohort", predict)
            require(B["parameter_snapshot"](model) == before, "original parameters changed before seal")
            seal = tx.seal(); phase["released"] = True
            event("global_forecasts_sealed", seal_sha256=seal["seal_sha256"], pretarget_counts=dict(counts))

            def execute(context):
                observed = {}; payloads = context.payloads["cohort"]
                for key in context.predictions["cohort"]["order"]:
                    prefix = context.frontiers["prefixes"][key]; planned = context.predictions["cohort"]["contexts"][key]
                    cut = torch.from_numpy(prefix["cut"].copy()); phase["lane"] = "component"; charge("native_MLP_bases")
                    with torch.inference_mode():
                        norm = layer.post_attention_layernorm(cut); update = layer.mlp(norm); base = cut+update
                    native_base = base.numpy().copy(); base_vector = native_base.reshape(-1).astype(np.float64)
                    writes = {name: torch.from_numpy(payloads[key+"."+name].copy()) for name in ("carrying", "neutral")}
                    parity, even = {}, {}
                    for name in ("plus", "minus"):
                        for sign in ("positive", "negative"):
                            charge("parity_targets"); operand = torch.from_numpy(payloads[key+".old-"+name+"."+sign].copy()).reshape(1, 1, hidden)
                            with torch.inference_mode():
                                post = layer.post_attention_layernorm(operand); delta = layer.mlp(post)
                            parity[name+"."+sign] = dict(input=tensor_descriptor(operand), postnorm=post.numpy().copy(), delta=delta.numpy().copy(),
                                error=B["metric"](planned["old_parity"][name][sign]["delta"], delta.numpy().reshape(-1)))
                        value = (parity[name+".positive"]["delta"].astype(np.float64)+parity[name+".negative"]["delta"].astype(np.float64))/2
                        even[name] = dict(value=value, consumer=float(c @ value.reshape(-1)), error=B["metric"](planned["old_parity"][name]["even"], value.reshape(-1)))
                    component = {}
                    for cell in settings["cells"]:
                        name, alpha, beta = cell["id"], cell["carrying"], cell["neutral"]; charge("component_readouts")
                        with torch.inference_mode():
                            carrier = base if alpha == beta == 0 else (base+alpha*writes["carrying"])+beta*writes["neutral"]
                            final = model.model.norm(carrier); logits = model.lm_head(final)[0, 0]
                        actual = logits.numpy().copy(); vector = carrier.numpy().reshape(-1).astype(np.float64)
                        expected_displacement = alpha*planned["axes"]["realized"]["carrying"].astype(np.float64)+beta*planned["axes"]["realized"]["neutral"].astype(np.float64)
                        addition = vector-base_vector-expected_displacement
                        reference = planned["readout"]["rows"][name]; baseline = actual if name == "baseline" else component["baseline"]["logits"]
                        component[name] = dict(logits=actual, carrier=carrier.numpy().copy(), final_norm=final.numpy().copy(),
                            rank_order=np.argsort(-actual, kind="stable").astype(np.uint32), co_maximizers=np.flatnonzero(actual == actual.max()).astype(np.uint32),
                            greedy_token=int(logits.argmax()), winner_text=decode([int(logits.argmax())]), contrast=float(actual[red])-float(actual[blue]),
                            geometry=dict(contrast_numerator=float(c @ vector), numerator_delta=float(c @ (vector-base_vector)), payload_numerator_delta=float(c @ expected_displacement),
                                addition_contrast_leak=float(c @ addition), addition_error_l2=float(np.linalg.norm(addition)), RMS=float(np.sqrt(np.mean(vector*vector)+eps)),
                                scope="FP64 geometry of the actual captured native post-addition state; not a captured FP32 RMS kernel denominator"),
                            error=B["metric"](reference["logits"], actual), response_error=B["metric"](reference["logits"]-planned["readout"]["rows"]["baseline"]["logits"], actual.astype(np.float64)-baseline.astype(np.float64)))
                    ordinary = {}
                    for name in [*planned["future_cells"], "sham"]:
                        cell = next(row for row in settings["cells"] if row["id"] == ("baseline" if name == "sham" else name)); received = {}

                        def observe_cut(_module, args):
                            received["cut"] = args[0][:, -1:, :].detach().clone()

                        def write(_module, args):
                            value = args[0]; received["before"] = value[:, -1:, :].detach().clone()
                            changed = value.clone()
                            if name != "sham" and (cell["carrying"] != 0 or cell["neutral"] != 0):
                                changed[:, -1:, :] = (value[:, -1:, :]+cell["carrying"]*writes["carrying"])+cell["neutral"]*writes["neutral"]
                            received["after"] = changed[:, -1:, :].detach().clone(); received["earlier_rows_exact"] = bool(torch.equal(changed[:, :-1], value[:, :-1]))
                            return (changed,)

                        handles = [layer.post_attention_layernorm.register_forward_pre_hook(observe_cut), model.model.norm.register_forward_pre_hook(write)]
                        phase["lane"] = "sham" if name == "sham" else "ordinary"
                        try:
                            with torch.inference_mode():
                                first = model(torch.from_numpy(prefix["ids"].copy()), attention_mask=torch.from_numpy(prefix["mask"].copy()), position_ids=torch.from_numpy(prefix["positions"].copy()),
                                    past_key_values=DynamicCache(config=model.config), use_cache=True, return_dict=True)
                        finally:
                            for handle in handles:
                                handle.remove()
                        actual = first.logits[0, -1].numpy().copy(); token = int(first.logits[0, -1].argmax())
                        original_cache = B["cache_snapshot"](first.past_key_values); private = copy.deepcopy(first.past_key_values)
                        require(all(a.data_ptr() != b.data_ptr() for left, right in zip(B["cache_arrays"](private), B["cache_arrays"](first.past_key_values), strict=True) for a, b in zip(left, right, strict=True)), "private future aliases its ordinary parent")
                        future_rows = []; previous = token
                        if name != "sham":
                            phase["lane"] = "future"
                            for step in range(settings["continuation_tokens"]):
                                length = private.get_seq_length()
                                with torch.inference_mode():
                                    future = model(torch.tensor([[previous]]), past_key_values=private, use_cache=True, return_dict=True)
                                private = future.past_key_values; following = int(future.logits[0, -1].argmax())
                                future_rows.append(dict(step=step+1, consumed_token=previous, greedy_token=following, emitted_eos=following in eos_ids,
                                    cache_length_before=length, cache_length_after=private.get_seq_length(), logits=tensor_descriptor(future.logits[:, -1]), text=decode([following])))
                                previous = following
                        require(B["cache_snapshot"](first.past_key_values) == original_cache, "private future mutated ordinary parent cache")
                        reference_name = "baseline" if name == "sham" else name
                        ordinary[name] = dict(logits=actual, greedy_token=token, generated_tokens=[token]+[row["greedy_token"] for row in future_rows], generated_text=decode([token]+[row["greedy_token"] for row in future_rows]),
                            continuation=future_rows, continuation_forecast=False, eos_policy="Recorded; fixed future budget continues after EOS", first_token_is_eos=token in eos_ids,
                            original_receiving_state=received["before"].numpy().copy(), written_state=received["after"].numpy().copy(), earlier_rows_exact=received["earlier_rows_exact"],
                            prefix_cut_error=B["metric"](prefix["cut"], received["cut"].numpy()), component_vs_ordinary=B["metric"](component[reference_name]["logits"], actual),
                            predicted_vs_ordinary=B["metric"](planned["readout"]["rows"][reference_name]["logits"], actual), ordinary_parent_cache=original_cache,
                            cache_exact_vs_observed_prefix=original_cache == parent_signatures[key], future_cache_authority="Always private copy of this ordinary parent; history preserved, no fallback")
                    require(B["cache_snapshot"](parents[key]) == parent_signatures[key], "observed prefix KV changed")
                    observed[key] = dict(component=component, ordinary=ordinary, old_parity=parity, native_old_even=even,
                        native_base=dict(postnorm=norm.numpy().copy(), delta=update.numpy().copy(), residual=native_base, error=B["metric"](planned["base"]["delta"], update.numpy().reshape(-1))),
                        sham_vs_baseline=B["metric"](ordinary["sham"]["logits"], ordinary["baseline"]["logits"]))
                    event("context_targets_complete", context=key, future_cells=list(planned["future_cells"]), counts=dict(counts))
                return dict(contexts=observed, order=order, counts=dict(counts), continuation_forecast=False)

            tx.execute("native", execute)
        require(B["parameter_snapshot"](model) == before, "original parameters changed during native work")
        verified = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
        summary = dict(schema=SCHEMA, job_id=settings.get("job_id"), request_sha256=settings.get("request_sha256"), model_identity=model_identity, input_identity=input_identity,
            counts=dict(counts), forecast_seal=seal, transaction=verified, context_order=order, consumer_ids=list(consumer_ids), collateral_ids=collateral_ids or {},
            frontier_identities={key: value["cut_descriptor"]["sha256"] for key, value in prefixes.items()}, seconds=time.monotonic()-started,
            scope="Conditional genuine observed prefixes; immediate full-vocabulary readout predictions; eight ordinary native future steps measured only",
            limitations=["Supplied red-blue contrast is not a semantic quality oracle.", "Axes re-instantiate one shared source rule at each receiving rho; payload bytes need not match across contexts.",
                "Ideal numerator-neutral does not imply unchanged RMS, other-token numerators or winner; FP32 payload/addition leakage is retained.",
                "Nine axial cells do not estimate interactions or enumerate2D regions; no accuracy, quality or divergence gate."])
        save(output / "summary.json", summary)
        return dict(summary=summary, predictions=tx.prediction("cohort"), observations=tx.observation("native"))
    finally:
        for hook in hooks:
            hook.remove()


def publish_accounts(result, output, *, execution, source_identity, run_id):
    """Publish every fixed cell through the installed catalog and atlas."""
    import numpy as np

    from mdb_lanes.computation_atlas import ComputationAtlas
    from mdb_lanes.equation_comparison import EquationCatalog, compare_equations

    output = Path(output); catalog = EquationCatalog(output / "catalog")
    references = [dict(path="summary.json", sha256=sha(output / "summary.json")), dict(path="transaction/seal.json", sha256=sha(output / "transaction/seal.json"))]
    rows, effects, futures, exceptions, geometry = [], {}, {}, {}, {}; summary = result["summary"]
    for key in result["predictions"]["order"]:
        predicted = result["predictions"]["contexts"][key]; observed = result["observations"]["contexts"][key]
        baseline, predicted_baseline = observed["component"]["baseline"], predicted["readout"]["rows"]["baseline"]
        for name, value in predicted["readout"]["rows"].items():
            actual = observed["component"][name]
            context = dict(model_identity=summary["model_identity"], source_identity=source_identity, execution=execution,
                input_identity=summary["input_identity"], frontier_identity=summary["frontier_identities"][key],
                frontier="genuine-native-final-pre-MLP; own-state additive final-norm input",
                horizon="conditional-last-token-final-readout", context=key, cell=name, carrying=value["carrying"], neutral=value["neutral"], axis=value["scope"],
                receiving_port="model.norm:input:last-token", clock="after-one-global-forecast-seal", cache="genuine-observed-prompt; private ordinary future")
            report = compare_equations(dict(logits=value["logits"]), dict(logits=actual["logits"]), decision_rules=dict(logits="argmax"),
                baseline=dict(logits=baseline["logits"]), predicted_baseline=dict(logits=predicted_baseline["logits"]), max_elements=len(value["logits"]))
            record = catalog.record(law_id="rms-reader.contrast-dual-signed", route_id="observed-final-MLP-to-full-head",
                binding={field: context[field] for field in ("model_identity", "source_identity", "execution", "input_identity", "frontier", "frontier_identity", "horizon")},
                report=report, conditions=[dict(field=field, op="eq", value=item) for field, item in context.items()], tested_context=context,
                readers=["full-vocabulary-greedy", "fixed-red-minus-blue", "numerator-neutral-versus-carrying"],
                evidence=dict(provenance=references, context=key, cell=name, conditional=True, raw_prompt_forecast=False, continuation_forecast=False))
            identifier = record["fingerprint"]
            effects[identifier] = [dict(consumer_id="fixed-red-minus-blue", contrast_id="signed-cell-minus-own-baseline", units="native-FP32-logit",
                value=actual["contrast"]-baseline["contrast"], provenance=references)]
            selected = observed["ordinary"].get(name)
            futures[identifier] = dict(status="measured-autonomous-native" if selected else "not-measured", provenance=references if selected else [])
            if selected:
                futures[identifier].update(generated_tokens=selected["generated_tokens"], generated_text=selected["generated_text"], numerical_forecast=False,
                    authority=selected["future_cache_authority"], component_vs_ordinary=dict(selected["component_vs_ordinary"]))
            exceptions[identifier] = [dict(claim_id="unconditional-FP32-byte-identity", status="finite-error-retained", error=dict(actual["error"]),
                predicted_winner_matches_native=actual["greedy_token"] == value["greedy_token"], native_receiving_geometry=dict(actual["geometry"]), provenance=references)]
            rows.append(dict(context=key, cell=name, axis=value["scope"], carrying=value["carrying"], neutral=value["neutral"], record_id=identifier,
                predicted_token=value["greedy_token"], native_token=actual["greedy_token"], predicted_co_maximizers=value["co_maximizers"].tolist(), native_co_maximizers=actual["co_maximizers"].tolist(),
                winner_text=actual["winner_text"], endpoint_error=dict(actual["error"]), response_error=dict(actual["response_error"]),
                predicted_numerator=value["contrast_numerator"], ideal_numerator=value["ideal_contrast_numerator"], native_geometry=dict(actual["geometry"]),
                predicted_denominator=value["denominator"], predicted_denominator_vs_baseline=value["denominator_vs_baseline"],
                predicted_contrast=value["realized_contrast"], native_contrast=actual["contrast"], predicted_red_rank=value["red_rank"], predicted_blue_rank=value["blue_rank"],
                collateral_ranks=dict(value["collateral_ranks"]), response=dict(value["response"]), selected_future=selected is not None))
        axes = predicted["axes"]
        geometry[key] = {field: axes[field].tolist() if isinstance(axes[field], np.ndarray) else dict(axes[field]) if isinstance(axes[field], Mapping) else axes[field]
            for field in ("gram", "contrast_coefficients", "singular_values", "numeric_rank", "rank_threshold", "common_norm", "applicability", "unit_q1_contrast", "carrying_gain", "ideal_neutral_dot", "ideal_orthogonality", "neutral_sign_pivot", "axis_diagnostics", "scope")}
        geometry[key].update(implications=list(predicted["readout"]["implications"]),
            old_parity={name: dict(requested_raw_norm=value["requested_raw_norm"], realized_raw_norm=value["realized_raw_norm"], parity_error=dict(value["parity_error"])) for name, value in predicted["old_parity"].items()},
            native_old_even={name: dict(consumer=value["consumer"], error=dict(value["error"])) for name, value in observed["native_old_even"].items()},
            native_base_error=dict(observed["native_base"]["error"]), sham_vs_baseline=dict(observed["sham_vs_baseline"]),
            ordinary={name: {field: item for field, item in value.items() if field not in ("logits", "original_receiving_state", "written_state")} for name, value in observed["ordinary"].items()})
    receipts = ComputationAtlas(output / "atlas").ingest(output / "catalog", [row["record_id"] for row in rows],
        provenance=dict(run_id=run_id, artifacts=references), signed_effects=effects, continuation=futures, counterexamples=exceptions)
    accounts = dict(schema="mdb-qwen-contrast-collateral-accounts-v1", rows=rows, contexts=geometry, record_ids=[row["record_id"] for row in rows],
        atlas_ids=[row["fingerprint"] for row in receipts], mechanics_status="valid", terminal_status="unassessed",
        coverage=dict(fixed_rows=len(rows), independent_contexts=len(geometry), full_vocabulary=True, futures="fixed baseline and signed-unit axes; measured only"), limitations=list(summary["limitations"]))
    save(output / "accounts.json", accounts)
    return accounts


def authenticate_request(request, config, source_root):
    B = dependencies()["qwen_spectral_readout_campaign.py"]
    require(request.get("config") == config and request.get("request_sha256") == B["request_identity"](request), "source-bound request/config identity differs")
    require(isinstance(request.get("source_hashes"), dict) and Path(__file__).name in request["source_hashes"], "worker must belong to request source closure")
    for name, expected in request["source_hashes"].items():
        require(Path(name).name == name and sha(Path(source_root) / name) == expected, "request source changed: " + name)
    require(all(request["source_hashes"].get(name) == expected for name, expected in HELPERS.items()), "qualified helper closure missing from request")


def worker(config_path):
    """Hydrate once, capture genuine prefixes, seal, execute and publish."""
    print("qwen-contrast bootstrap; four fresh prefixes, source-only axes, one global seal", flush=True)
    config_path = Path(config_path).resolve(); config = json.loads(config_path.read_text()); planned = validate_config(config)
    request_path = Path(os.environ["SATURN_DEBUG_REQUEST"]).resolve(); request = json.loads(request_path.read_text())
    authenticate_request(request, config, config_path.parent)
    require(sha(config["retrieval_script"]) == RETRIEVER_SHA256, "standalone parameter retriever changed")
    job_id = os.environ["MRUN_JOB_ID"]; require(job_id.startswith("job-") and Path(job_id).name == job_id, "actual job identity required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(config["seed"]); torch.use_deterministic_algorithms(True)
    faulthandler.enable(); faulthandler.dump_traceback_later(60, repeat=True)
    try:
        h = dependencies(); B = h["qwen_spectral_readout_campaign.py"]
        checkpoint = Path(config["model_path"]); inventory, signatures = B["checkpoint_inventory"](checkpoint, B["CHECKPOINT"])
        weights_sha256 = digest({name: value["sha256"] for name, value in inventory.items() if name.endswith(".safetensors")})
        require(weights_sha256 == "2b226fe4944dc744596b1142a15d6676160e9d06f245d31c1a36633f6c86baa3", "original weight-file identity differs")
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
        layer = model.model.layers[-1]; eps = float(model.config.rms_norm_eps)
        require(model.config.model_type == "qwen2" and model.config.hidden_act == "silu" and model.config._attn_implementation == "eager" and model.config.hidden_size == 896 and len(model.model.layers) == 24 and model.config.vocab_size == 151936, "bound original Qwen2 CPU eager geometry required")
        require(type(layer.post_attention_layernorm).__name__ == type(model.model.norm).__name__ == "Qwen2RMSNorm" and np.isfinite(eps) and eps > 0
            and layer.post_attention_layernorm.variance_epsilon == model.model.norm.variance_epsilon == eps, "native RMSNorm type/epsilon differs")
        prompts, consumers, encodings = B["tokenize"](tokenizer, config)
        collateral_encodings = {text: tokenizer.encode(text, add_special_tokens=False) for text in config["collateral_strings"]}
        collateral_ids = {text: values[0] if len(values) == 1 else None for text, values in collateral_encodings.items()}
        identity = dict(model_type="qwen2", weights_sha256=weights_sha256, checkpoint_files_sha256=digest(inventory), config_sha256=digest(model.config.to_dict()), source="Qwen2.5-0.5B-Instruct original checkpoint")
        execution = dict(backend="cpu", dtype="float32", attention="eager", threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
            mkldnn=torch.backends.mkldnn.enabled, matmul_precision=torch.get_float32_matmul_precision(), numpy=str(np.__version__), torch=str(torch.__version__), transformers=str(importlib.metadata.version("transformers")),
            component_shape="1x1xH", ordinary_prefill_shape="1xT(prompt)xH", kernel_profile="Source-bound CPU eager; exact selected hardware kernels unprofiled")
        kinds = (type(model), type(model.model), type(layer), type(layer.mlp), type(layer.self_attn), type(model.model.norm), type(model.config), type(tokenizer))
        native_sources = {str(Path(inspect.getfile(kind)).resolve()): sha(inspect.getfile(kind)) for kind in kinds}
        source_identity = dict(request["source_hashes"]) | {"native:"+name: value for name, value in native_sources.items()} | dict(parameter_retriever=RETRIEVER_SHA256)
        output = Path(config["output_scratch"]) / job_id / config["output_name"]

        def setup(path):
            save(path / "request.json", request); save(path / "declared-config.json", config)
            save(path / "source.json", dict(schema=SCHEMA, job_id=job_id, request_sha256=request["request_sha256"], model_identity=identity, execution=execution,
                source_identity=source_identity, checkpoint_files=inventory, model_config=model.config.to_dict(), consumer_encodings=encodings, consumer_ids=consumers,
                prompt_encodings={value["id"]: value["ids"].tolist() for value in prompts}, collateral_encodings=collateral_encodings, collateral_ids=collateral_ids,
                genuine_record_prefix_capture=True, raw_input_prediction=False, continuation_forecast=False, planned_counts=planned, pretrained_parameter_bank_written=False,
                scope="Own observed last-token residual and original factors; source-rederived contrast-dual axes; full-vocabulary immediate forecast; measured ordinary futures"))
            save(path / "runtime.json", dict(versions={name: str(importlib.metadata.version(name)) for name in ("torch", "transformers", "numpy", "mdb-runtime", "saturn-pub")},
                native_sources=native_sources, execution=execution, platform=platform.platform(), machine=platform.machine(), torch_build=torch.__config__.show()))
            h["qwen_spectral_parameter_recipes.py"]["prepare"](model, checkpoint, inventory, path, config["retrieval_script"])

        result = campaign(model, prompts, consumers, output, config | dict(job_id=job_id, request_sha256=request["request_sha256"]), model_identity=identity,
            source_identity=source_identity, collateral_ids=collateral_ids, decode=lambda values: tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False), setup=setup)
        for name, expected in signatures.items():
            value = (checkpoint / name).stat()
            require((value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns) == expected, "original checkpoint mutated during run")
        accounts = publish_accounts(result, output, execution=execution, source_identity=source_identity, run_id=job_id)
        ready = h["flux_image_tail_microscope.py"]["finalize"](output, job_id=job_id, request_sha256=request["request_sha256"], **config["transport_limits"])
        ready["equation_records"] = len(accounts["rows"])
        print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
        return ready
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
