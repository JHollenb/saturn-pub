"""Native arithmetic tasks and explicit local affine capture contracts."""
from __future__ import annotations

import numpy as np


def arithmetic_cases(seed, style, count=8):
    rng = np.random.default_rng(seed)
    result = []
    base_style = style.removesuffix("_space")
    for digit in rng.permutation(10):
        if len(result) == count:
            break
        a = int(rng.integers(10, 80))
        b = int(rng.integers(1, 8))*10 + ((int(digit)-a) % 10)
        if base_style == "last_digit":
            prompt = "Write only the last digit of the sum.\n17 + 25 = 2\n36 + 48 = 4\n"+str(a)+" + "+str(b)+" = "
            answer = (a+b) % 10
        elif base_style == "compact_modulo":
            prompt = "Compute (a+b) mod 10.\n(17+25) mod 10 = 2\n(36+48) mod 10 = 4\n("+str(a)+"+"+str(b)+") mod 10 ="
            answer = (a+b) % 10
        elif base_style == "small_sum":
            answer = int(digit)
            left = int(rng.integers(0, answer+1))
            right = answer-left
            prompt = "2 + 3 = 5\n1 + 6 = 7\n"+str(left)+" + "+str(right)+" ="
        else:
            raise ValueError("unknown arithmetic style")
        if style.endswith("_space"):
            prompt += " "
        result.append({"prompt": prompt, "answer": str(answer)})
    if count > 10:
        raise ValueError("balanced task batch maximum is ten")
    return result


def native_context(net, tokenizer, seed, style, *, examples=None, evaluation=False):
    import torch
    from qwen_profile import Consumer

    examples = arithmetic_cases(seed, style) if examples is None else examples
    for row in examples:
        prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
        suffix = tokenizer.encode(row["answer"], add_special_tokens=False)
        complete = tokenizer.encode(row["prompt"]+row["answer"], add_special_tokens=False)
        if len(suffix) != 1 or complete != prefix+suffix:
            raise ValueError("answer is not exactly one token after the native prompt boundary")
    model = Consumer(net, tokenizer, seed, examples=examples,
        evaluation=evaluation)
    sites = ["net.model.layers."+str(i)+".mlp.down_proj" for i in range(net.config.num_hidden_layers)]
    def validate_native_call(args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        mask = kwargs.get("attention_mask")
        if ids is None or ids.ndim != 2 or ids.shape[1] != model.input_ids.shape[1] or not 1 <= len(ids) <= 16:
            raise ValueError("native physical input geometry differs from the fixed table")
        matches = (ids[:, None, :] == model.input_ids[None, :, :]).all(-1)
        if not bool(matches.any(-1).all()):
            raise ValueError("native input IDs differ from the fixed table")
        selected = matches.long().argmax(-1)
        if mask is None or not torch.equal(mask, model.attention_mask[selected]):
            raise ValueError("native attention mask differs from the fixed table")
        if kwargs.get("use_cache") is not False or kwargs.get("logits_to_keep") != 1 or kwargs.get("past_key_values") is not None:
            raise ValueError("native consumer requires no KV state and one-position logits")
    return {"model": model, "sites": sites, "guard_model": net, "input_guard": model.comparison_input_identity,
        "validate_native_call": validate_native_call,
        "input_identity": {"native_table": model.input_identity},
        "canonical_to_wrapper": {s.removeprefix("net."): s for s in sites},
        "splits": {"discovery": (torch.arange(8), torch.zeros(8, 1)),
                   "preservation": (torch.arange(8, 16), torch.zeros(8, 1))}}


def local_capture(model, rows, sites, directory, identity):
    """Tap native operands, then type local affine replays; pay for both scopes."""
    import torch
    from protocol import digest, save

    from mdb_lanes.forward_ports import NativePortCatalog
    from mdb_lanes.forward_recording import record_forward

    taps, handles = {}, []
    modules = dict(model.named_modules())
    directory.mkdir(parents=True, exist_ok=False)
    for site in sites:
        def tap(module, args, output, site=site):
            taps[site] = {"input": args[0][:, -1, :].detach().clone(),
                          "output": output[:, -1, :].detach().clone()}
        handles.append(modules[site].register_forward_hook(tap))
    try:
        with torch.inference_mode():
            model(rows)
    finally:
        for handle in handles:
            handle.remove()
    readout = {k: v.copy() for k, v in model.last_readout.items()}
    metadata = []
    for index, site in enumerate(sites):
        # This is a separate replay on the recorded final-position operand,
        # not a claim to have recorded the full transformer ATen execution.
        h = taps[site]["input"]
        captured = record_forward(modules[site], (h,), model_identity=identity+":"+site,
            input_identity=digest(h.tolist()), execution={"backend": "CPU FP32 local affine replay"}, capture_ports=True)
        catalog = NativePortCatalog(captured.trace)
        a = catalog.resolve("module:<root>:input", path=["args", 0])
        b = catalog.resolve("module:<root>:return")
        values = {**captured.initial_values, **captured.observed_values}
        delta = captured.output-taps[site]["output"]
        taps[site].update(input_port=a, output_port=b, local_replay_output=values[b["value_id"]].clone(),
            maximum_replay_difference=float(delta.abs().max()))
        save(directory / (str(index)+"-trace.json"), captured.trace)
        metadata.append({"site": site, "trace_fingerprint": catalog.trace_fingerprint,
            "maximum_replay_difference": taps[site]["maximum_replay_difference"],
            "native_operand_identity": digest(h.tolist()),
            "scope": "typed local affine replay of final-position native operand; remainder measured against live native return"})
        del captured, values, catalog
    arrays = {}
    for index, site in enumerate(sites):
        arrays[str(index)+"_input"] = taps[site]["input"].numpy()
        arrays[str(index)+"_output"] = taps[site]["output"].numpy()
        arrays[str(index)+"_replay"] = taps[site]["local_replay_output"].numpy()
    np.savez_compressed(directory / "operands.npz", **arrays)
    save(directory / "metadata.json", metadata)
    model.last_readout = readout
    return taps, {"sites": metadata, "full_model_forwards": 1, "local_projection_replays": len(sites)}


def qualify(config, root):
    import torch
    from protocol import save
    from qwen_profile import Consumer, load, save_readout

    from mdb_lanes.wsp_weights import parameter_inventory

    net, tokenizer = load(config, root)
    calibration = []
    for style in ("compact_modulo_space", "small_sum_space"):
        reader = Consumer(net, tokenizer, config["seed"], examples=arithmetic_cases(config["seed"], style))
        score = reader.anchor_score()
        calibration.append({"style": style, "baseline": score, "examples": reader.examples,
            "prediction_text": [tokenizer.decode([t]) for t in score["winner_ids"]]})
    save(root / "spacing-calibration.json", calibration)
    ctx = native_context(net, tokenizer, config["seed"], config["style"])
    model = ctx["model"]
    identity = parameter_inventory(model)["identity"]
    original = model.anchor_score()
    save_readout(model, root / "anchor-readout.npz")
    taps, receipt = local_capture(model, torch.arange(16), ctx["sites"], root / "local-capture", identity)
    recorded = model.readout_score()
    save_readout(model, root / "captured-readout.npz")
    no_op_equal = np.array_equal(model.last_readout["logp"], model.anchor_logp.cpu().numpy())
    result = {"baseline": original, "captured": recorded, "capture": receipt,
        "no_op_full_vocab_readout_equal": no_op_equal, "parameter_identity_restored": parameter_inventory(model)["identity"] == identity,
        "maximum_local_replay_difference": max(t["maximum_replay_difference"] for t in taps.values()),
        "work": {"spacing_calibration_anchor_forwards": 2, "original_anchor_forwards": 1,
            "capture_full_model_forwards": 1, "local_projection_replays": len(taps)},
        "scope": "native no-op parity and typed local affine capture qualification; no editor ranking or held-out task"}
    save(root / "capture-qualification.json", result)
    return result


def run_phase(config, root):
    if config["phase"] == "qualify":
        return qualify(config, root)
    if config["phase"] in {"prepare", "search", "evaluate", "kernel-replay"}:
        return campaign_phase(config, root)
    raise ValueError("unknown native phase")


def campaign_phase(config, root):
    import hashlib
    import json
    import time
    from pathlib import Path

    import torch
    from protocol import ARMS, digest, save, validate_edits
    from qwen_profile import load, save_readout
    from worker_common import fingerprint, search_one

    net, tokenizer = load(config, root)
    if config["phase"] == "prepare":
        packets = {}
        for specimen in config["specimens"]:
            ctx = native_context(net, tokenizer, specimen["seed"], config["style"])
            model = ctx["model"]
            baseline = model.anchor_score()
            parameters = dict(model.named_parameters())
            packets[specimen["id"]] = {"specimen": specimen["id"], "model_identity": fingerprint(model),
                "architecture": "original Qwen2.5-0.5B-Instruct; CPU FP32 eager; full prefixes, no KV reuse",
                "checkpoint_sha256": config["checkpoint_sha256"], "input_identity": model.input_identity,
                "canonical_to_wrapper": ctx["canonical_to_wrapper"], "sites": ctx["sites"],
                "examples": model.examples, "preservation_prompts": model.preservation_prompts, "baseline": baseline,
                "weight_summaries": {s: {"shape": list(parameters[s+".weight"].shape),
                    "rms": float(parameters[s+".weight"].square().mean().sqrt())} for s in ctx["sites"]}}
            save_readout(model, root / (specimen["id"]+"-anchor.npz"))
        save(root / "public-packets.json", packets)
        return {"phase": "prepare", "anchor_forwards": len(packets), "scope": "native discovery only; no evaluation inputs released"}
    if config["phase"] in {"search", "kernel-replay"}:
        history = json.loads(Path("history.json").read_text())
        trials = json.loads(Path("proposals.json").read_text())
        expected = {(s["id"], a) for s in config["specimens"] for a in ARMS} if config["phase"] == "search" else {tuple(v) for v in config["replay_pairs"]}
        if {(t["specimen"]["id"], t["arm"]) for t in trials} != expected or len(trials) != len(expected):
            raise ValueError("native paired trial inventory differs")
        rows = []
        for trial in trials:
            start = time.perf_counter()
            ctx = native_context(net, tokenizer, trial["specimen"]["seed"], config["style"])
            setup_seconds = time.perf_counter()-start
            directory = root / trial["id"]
            directory.mkdir()
            trial_config = {**config, "round": trial["source_round"]} if config["phase"] == "kernel-replay" else config
            row = search_one(trial_config, trial, directory, history.get(trial["id"], []), context=ctx)
            row["common_setup"] = {"anchor_forwards": 1, "seconds": setup_seconds}
            if config["phase"] == "kernel-replay":
                row["scope"] = "fixed-program kernel compatibility replay; excluded from primary selections and workflow totals"
            save(directory / "row.json", row)
            rows.append(row)
            save(root / ("partial-"+str(len(rows))+".json"), rows)
            print("FOLLOWUP_PROGRESS search "+trial["id"]+" probes="+str(len(row["probes"]))+" status="+row["instrument_status"], flush=True)
            if row["parameter_drift"]:
                raise ValueError("unsafe to reuse resident model after parameter drift")
        save(root / "public-feedback.json", rows)
        return {"phase": config["phase"], "rows": len(rows), "setup_anchor_forwards": len(rows),
            "discovery_network_forwards": sum(r["native_work"]["started_native_calls"] for r in rows),
            "local_projection_replays": sum(r["native_work"].get("local_projection_replays", 0) for r in rows)}
    selections = json.loads(Path("selections.json").read_text())
    expected = {(s["id"], a) for s in config["specimens"] for a in ARMS}
    if {(r["specimen"]["id"], r["arm"]) for r in selections} != expected or len(selections) != len(expected):
        raise ValueError("final paired selection inventory differs")
    if any(r["plan_identity"] != config["identity"] or r["execution_plan_identity"] != config["execution_plan_identity"] for r in selections):
        raise ValueError("final selection plan identity differs")
    rows = []
    # Evaluation inputs are generated only in this sealed evaluation worker.
    # Two fresh batches per specimen, excluding every discovery prompt.
    for specimen in config["specimens"]:
        excluded = {r["prompt"] for r in arithmetic_cases(specimen["seed"], config["style"])}
        batches, seed = [], specimen["seed"]+100003
        for batch in range(2):
            fresh = []
            while len(fresh) < 8:
                for r in arithmetic_cases(seed, config["style"]):
                    if r["prompt"] not in excluded:
                        fresh.append(r)
                        excluded.add(r["prompt"])
                        if len(fresh) == 8:
                            break
                seed += 1
            batches.append(native_context(net, tokenizer, seed, config["style"], examples=fresh, evaluation=True))
        for arm in ARMS:
            selection = next(r for r in selections if r["specimen"]["id"] == specimen["id"] and r["arm"] == arm)
            directory = root / selection["id"]
            directory.mkdir()
            batch_rows = []
            for batch, ctx in enumerate(batches):
                model, splits = ctx["model"], ctx["splits"]
                identity = fingerprint(model)
                edits = validate_edits(selection["selection"], ctx["sites"])
                parameters = dict(model.named_parameters())
                originals = {r["site"]+".weight": parameters[r["site"]+".weight"].detach().clone() for r in edits}
                observed = {}
                for name in ("native", "edited", "restored"):
                    if name == "edited":
                        with torch.no_grad():
                            for r in edits:
                                parameters[r["site"]+".weight"].copy_(originals[r["site"]+".weight"]*r["gain"])
                    try:
                        result = model.comparison_score(splits, ("discovery", "preservation"))
                        observed[name] = {**result, "readout": save_readout(model, directory / (str(batch)+"-"+name+".npz"), compact=True)}
                        # Retain a full-vocabulary audit panel on the first
                        # fresh batch, shared native once and edited per arm.
                        if batch == 0 and (name == "edited" or (name == "native" and arm == ARMS[0])):
                            save_readout(model, directory / (name+"-full.npz"))
                    finally:
                        with torch.no_grad():
                            for n, value in originals.items():
                                parameters[n].copy_(value)
                if fingerprint(model) != identity:
                    raise ValueError("final consumer parameter restoration differs")
                with np.load(directory / observed["native"]["readout"]["path"], allow_pickle=False) as a, np.load(directory / observed["restored"]["readout"]["path"], allow_pickle=False) as b:
                    restoration = bool(np.array_equal(a["logp_sha256"], b["logp_sha256"]))
                batch_rows.append({"batch": batch, "examples": model.examples, "preservation": model.preservation_prompts,
                    "input_identity": model.input_identity, "outcomes": observed, "original_restored_readout_equal": restoration})
            row = {"trial": selection["id"], "arm": arm, "specimen": specimen["id"], "selection": selection["selection"],
                "selection_identity": digest(selection), "batches": batch_rows, "parameter_restoration": True,
                "selection_file_sha256": hashlib.sha256(Path("selections.json").read_bytes()).hexdigest()}
            save(directory / "evaluation.json", row)
            rows.append(row)
            print("FOLLOWUP_PROGRESS evaluate "+selection["id"], flush=True)
    save(root / "evaluation.json", rows)
    return {"phase": "evaluate", "trials": len(rows), "fresh_anchor_forwards": 6,
        "physical_network_forwards": 6+6*len(rows), "scope": "fresh authored task and preservation prefixes; one-token arithmetic readout, no generation or general superiority certification"}


NATIVE_COMMON = """You are the fixed experimental weight-editing policy. Return JSON only, without invoking tools.
The target is original pretrained Qwen2.5-0.5B-Instruct, frozen CPU FP32 eager. Improve
the supplied arithmetic next-token readout while preserving the original next-token distribution
on the supplied preservation prefixes. No injected fault or healthy donor exists. Discovery
examples/labels are public; fresh evaluation inputs are absent. No autograd, backward,
optimizer or training. Explicit local response algebra/sensitivities are permitted.
Write analysis_python for the numerical kernel. Available: np, torch, model, x (8 task row
IDs), y (unused zeros), keep_x (8 preservation row IDs), keep_y (unused zeros), sites (24
ordered down projections), weights (copies of those original matrices), history, probe, print.
model(torch.cat([x, keep_x])) runs the real frozen model on those fixed prefixes. It returns
sqrt(NLL) for task rows and sqrt(clipped-roundoff KL) for preservation rows; the authoritative
probe retains signed native KL. model.net is the actual frozen transformer. Every native
root forward, including direct model.net calls and captures, spends this round's six-call budget.
probe([{'site': name, 'gain': g}, ...]) compiles gains into original weight matrices,
runs one full-prefix batch16 and restores them. At most two distinct sites; gains in [-4,4].
It returns task_mean_logp, preservation_kl (nats), objective=task_mean_logp-preservation_kl,
per-row target_logp, winner_ids, target_ids and correct (out of8). Compatibility names task_mse
and preservation_mse mean NLL and KL here, NOT MSE. Select for the exact objective; greedy
accuracy and preservation costs remain separate observations. Two rounds, six calls each.
Selection is automatic from completed probes: highest discovery objective across rounds,
native on ties. No evaluation-dependent edits. Permanent parameter/input/anchor mutation is
invalid. Do not rebuild the whole network to evade the native budget. Local algebra is allowed
and timed. Kernel permits bounded loops/functions/hooks, but no imports, files, private
attributes, while loops, autograd, network or optimizers. Program deadline is in the packet.
Every completed probe remains evidence after a later ordinary program error. Forecast native
effects before this round executes; distinguish known local equations from unknown task effects.
"""

NATIVE_DOCS = {
    "wsp_python": """Workflow: WSP-style Python weight inspection and outcome feedback. All common
Torch, numerical algebra and hook capabilities remain available. No extra interpretation
procedure is supplied. This is a defined workflow treatment, not a publication-harness replication.
""",
    "pytorch_hooks": """Workflow: competent Python/PyTorch with ordinary hooks. Example capture of
the final-position incoming operand and write at every site (one native model forward):
taps = {}
handles = []
for name, module in model.named_modules():
    if name in sites:
        def hook(module, args, out, name=name):
            taps[name] = (args[0][:,-1,:].detach().clone(), out[:,-1,:].detach().clone())
        handles.append(module.register_forward_hook(hook))
try:
    model(torch.cat([x, keep_x]))
finally:
    for handle in handles: handle.remove()
At fixed incoming operand h, gain g changes the direct affine write by (g-1)*(h@W.T).
Later operands change too. Local write size need not predict the final native task effect.
probe handles actual parameter edits and try/finally restoration. Ordinary raw analysis remains available.
""",
    "mdb": """Workflow: the same raw Python/PyTorch environment with MDB. capture(value=torch.cat([x,keep_x]))
uses ordinary live hooks for final-position down-projection inputs/outputs during one native
forward, then records each selected affine module replay via record_forward(capture_ports=True)
and NativePortCatalog. It returns input/output tensors, typed local input_port/output_port,
local_replay_output and maximum_replay_difference, by site. This is NOT a whole-transformer
ATen trace; the local replay's finite-arithmetic remainder is measured against the live return.
The one native forward and 24 extra local projection replays are counted in separate work scopes.
linear_response(W,Wprime,h) is MDB's FP64 equation returning NumPy baseline/edited/direct arrays;
it describes fixed operands, not downstream task improvement. FP32 native probes remain authority.
probe uses ParameterPatch and a sealed model-bound WeightStudy with forecast, parameter
hashing, outputs and restoration records. All this overhead is timed. All raw hooks/analysis remain available.
"""}
