"""Trace a frozen value response, fit its role condition, then replay frozen weights."""
from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from wsp_binding_protocol import ROLES, contrasts, reductions, save, sha


def run(config, request):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from mdb.wsp import WeightStudy, verify_weight_study
    from mdb.wsp_binding import fit_response_selector
    from mdb.wsp_selective import synthesize_low_rank_patch, virtual_low_rank_edit
    from mdb.wsp_weights import WeightSpaceError, parameter_inventory

    started = time.monotonic()
    root = Path(config["output_scratch"])/("wsp-binding-"+os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True, exist_ok=False)
    phase = config["phase"]
    save(root/"config.json", config)
    save(root/"request.json", request)
    for name in ("upstream.json", "upstream.npz", "parameter-artifacts.json", "retrieve_weights.py"):
        if sha(name) != request["source_hashes"][name]:
            raise WeightSpaceError("upstream evidence changed")
        shutil.copyfile(name, root/name)
    upstream = json.loads(Path("upstream.json").read_text())
    with np.load("upstream.npz", allow_pickle=False) as values:
        source = {k: values[k].copy() for k in values.files}
    direction = source["direction"]
    old = source["left"], source["right"]
    old_dose = upstream["selected"]["dose"]
    site = upstream["site"]
    torch.set_num_threads(config["cpu_threads"])
    checkpoint = Path(config["model_path"])
    if sha(checkpoint/"model.safetensors") != config["checkpoint_sha256"]:
        raise WeightSpaceError("checkpoint differs")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if len(tokenizer.encode(upstream["prefix"], add_special_tokens=False)) != 45:
        raise WeightSpaceError("role addresses inapplicable")
    model = Qwen2ForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").cpu().eval().requires_grad_(False)
    original = parameter_inventory(model)
    module = model.get_submodule(site)
    save(root/"runtime.json", {"torch": str(torch.__version__), "numpy": np.__version__,
        "device": "cpu", "dtype": "float32", "execution": "hf-cpu-fp32-eager",
        "generation": {"max_new_tokens": 3, "do_sample": False, "repetition_penalty": 1., "use_cache": True}})

    def progress(**row):
        row["elapsed_s"] = time.monotonic()-started
        with (root/"timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True)+"\n")
        print("WSP_BINDING_PROGRESS="+json.dumps(row, sort_keys=True), flush=True)

    if phase == "discover":
        rows = contrasts(config["development_seed"], config["development_blocks"], upstream["prefix"], upstream["exclude_a"])
        rows += upstream["diagnostic_controls"]
    else:
        frozen = Path("frozen-selection.json")
        if sha(frozen) != config["selection_sha256"] or sha("selected.npz") != config["factors_sha256"]:
            raise WeightSpaceError("frozen selection differs")
        chosen = json.loads(frozen.read_text())
        shutil.copyfile(frozen, root/frozen.name)
        shutil.copyfile("selected.npz", root/"selected.npz")
        with np.load("selected.npz", allow_pickle=False) as values:
            fitted = values["left"].copy(), values["right"].copy()
        rows = contrasts(config["heldout_seed"], config["heldout_blocks"], upstream["prefix"],
            set(upstream["exclude_a"]) | set(chosen["development_a"]))
    save(root/"inputs.json", {"rows": rows, "roles": ROLES,
        "supplied": "digit and role labels, fixed prefix, historical source account, original response direction",
        "selection_scoring": "last answer digit after supplying two unchanged answer digits",
        "generation": "raw prefix, own three tokens and native KV",
        "scope": "fixed format; matched rewrite changes no carry; old failures are development diagnostics"})
    encoded = tokenizer([r["prompt"]+r["original"][:2] for r in rows], padding=True,
        return_tensors="pt", add_special_tokens=False)
    raw = tokenizer([r["prompt"] for r in rows], padding=True, return_tensors="pt", add_special_tokens=False)
    if encoded.input_ids.shape[1] != 55 or raw.input_ids.shape[1] != 53:
        raise WeightSpaceError("fixed role layout differs")
    for role, pos in ROLES.items():
        operand, column = role.split(".")
        expected = [15+(r[operand]//(10 if column == "tens" else 1) % 10) for r in rows]
        if raw.input_ids[:, pos].tolist() != expected:
            raise WeightSpaceError("role does not resolve to declared digit")
    anchor, capture_hidden, scoring = None, None, False
    active_chunk_start = 0
    trace, gen_trace = {}, []
    current_factors, current_dose = old, old_dose
    # Express both factorizations' scalars in units of the frozen semantic direction.
    direction_norm2 = float(np.dot(direction, direction))

    @contextmanager
    def instrument(capture=False):
        handles = []
        def writer(layer, args, output):
            nonlocal capture_hidden
            left, right = current_factors
            coefficient = (args[0].detach().double() @ torch.tensor(right).T).squeeze(-1)
            scale = float(np.dot(left[:, 0], direction)/direction_norm2)*current_dose
            values = coefficient.numpy().copy()*scale
            if scoring:
                trace.setdefault("selector_coefficients", []).append(values)
                if capture:
                    capture_hidden.append(args[0].detach().numpy().copy())
            else:
                gen_trace.append({"row_start": active_chunk_start,
                    "physical_width": output.shape[1], "coefficients": values})
        handles.append(module.register_forward_hook(writer))
        for address in ("model.layers.22.self_attn", "model.layers.23"):
            def consumer(layer, args, output, address=address):
                if scoring:
                    tensor = output[0] if isinstance(output, tuple) else output
                    trace.setdefault(address, []).append(tensor[:, [46, 47, 50, 51, 52, -1]].detach().numpy().copy())
            handles.append(model.get_submodule(address).register_forward_hook(consumer))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def scoped_old(scope):
        left = torch.tensor(old[0], dtype=torch.float32)
        right = torch.tensor(old[1], dtype=torch.float32)
        def hook(layer, args, output):
            delta = old_dose*((args[0] @ right.T) @ left.T)
            mask = torch.zeros(output.shape[1], dtype=torch.bool)
            if scope == "source-only":
                if len(mask) > 47:
                    mask[47] = True
            else:
                mask[:] = True
                if len(mask) > 47:
                    mask[47] = False
            return output+delta*mask[None, :, None]
        handle = module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def evaluate(*, generation, capture=False, traced=False):
        nonlocal anchor, scoring, trace, gen_trace, capture_hidden, active_chunk_start
        if time.monotonic()-started > config["max_wall_seconds"]:
            raise WeightSpaceError("finite scientific wall budget exceeded")
        trace, gen_trace = {}, []
        if capture:
            capture_hidden = []
        with instrument(capture) if traced else nullcontext():
            scoring = True
            pieces = []
            for start in range(0, len(rows), 32):
                active_chunk_start = start
                batch = {k: v[start:start+32] for k, v in encoded.items()}
                pieces.append(model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float().detach().numpy().copy())
            logits = np.concatenate(pieces)
            scoring = False
            if anchor is None:
                anchor = logits.copy()
            observed = {**reductions(logits, rows, anchor), "logits": logits}
            if generation:
                new = []
                for start in range(0, len(rows), 32):
                    active_chunk_start = start
                    batch = {k: v[start:start+32] for k, v in raw.items()}
                    tokens = model.generate(**batch, max_new_tokens=3, do_sample=False, repetition_penalty=1.,
                        use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    new.extend(tokens[:, raw.input_ids.shape[1]:].tolist())
                expected = [[15+int(ch) for ch in r["expected"]] for r in rows]
                correct = [x == y for x, y in zip(new, expected, strict=True)]
                observed.update(generated_ids=new, generated_text=tokenizer.batch_decode(new),
                    generated_target_correct=sum(c for c, r in zip(correct, rows, strict=True) if r["matched"]),
                    generated_control_correct=sum(c for c, r in zip(correct, rows, strict=True) if not r["matched"]))
        if traced:
            observed["trace"] = {k: np.concatenate(v) for k, v in trace.items()}
            observed["generation_selector_trace"] = gen_trace
        if capture:
            capture_hidden = np.concatenate(capture_hidden)
        return {"observations": observed, "work": {"physical_scoring_forwards": (len(rows)+31)//32,
            "batch_size": min(32, len(rows)), "logical_rows": len(rows), "generation_rows": len(rows) if generation else 0,
            "max_new_tokens": 3 if generation else 0}, "state": {"policy": "restart", "native_KV": True}}

    binding = {"model": {"checkpoint_sha256": config["checkpoint_sha256"]},
        "checkpoint": config["checkpoint_sha256"], "execution": "hf-cpu-fp32-eager",
        "consumer": "unchanged Qwen arithmetic suffix", "horizon": "own three answer tokens"}
    options = {"model": model, "binding": binding, "input_identity": {"rows": rows},
        "source_identity": {"request_sha256": request["request_sha256"], "worker": sha(__file__)},
        "budgets": {"max_wall_ms": config["max_wall_seconds"]*1000, "max_bytes": 1024**3}}
    results, verifications = {}, {}

    def record(study, arm):
        from mdb.wsp_discovery import _plain
        value = study.tx.observation(arm)["result"]
        if "error" in value:
            results[arm] = {"instrument_status": "evidence-incomplete", "error": _plain(value["error"])}
        else:
            results[arm] = _plain({k: v for k, v in value["observations"].items()
                if k not in {"logits", "trace", "generation_selector_trace"}})
        progress(phase=phase, arm=arm, observations=results[arm])

    if phase == "discover":
        with WeightStudy(root/"diagnostic", **options) as study:
            for name in ("native", "broad", "source-only", "other-only"):
                study.add_arm(name, role="native" if name == "native" else "edited", state_policy="restart",
                    predictor=lambda ctx, name=name: {"predictions": {"arm": name,
                        "question": "source-row selector versus writes at other positions; scopes are external diagnostics"}})
            study.seal()
            for name in ("native", "broad", "source-only", "other-only"):
                manager = (virtual_low_rank_edit(model, site, *old, dose=old_dose) if name == "broad"
                    else scoped_old(name) if name in {"source-only", "other-only"} else nullcontext())
                def execute(ctx, manager=manager, name=name):
                    with manager:
                        return evaluate(generation=True, capture=name == "native", traced=True)
                study.run(name, execute)
                record(study, name)
        verifications["diagnostic"] = verify_weight_study(root/"diagnostic")
        if capture_hidden is None or "error" in results["native"]:
            raise WeightSpaceError("native selector capture missing")
        target_mask = np.array([r["matched"] for r in rows])
        h = np.concatenate((source["positive"], capture_hidden[target_mask, 47]))
        negative, key_roles, seen = [], [], set()
        for i, row in enumerate(rows):
            for pos in ROLES.values():
                if row["matched"] and pos == 47:
                    continue
                key = capture_hidden[i, pos]
                if key.tobytes() not in seen:
                    seen.add(key.tobytes())
                    negative.append(key)
                    key_roles.append({"row": i, "position": pos})
        # Other positions include prefix and answer states. They supplement,
        # rather than replace, the explicit operand/column counterexamples.
        other = capture_hidden.reshape(-1, capture_hidden.shape[-1])
        indices = np.linspace(0, len(other)-1, 64, dtype=int)
        forbidden = {capture_hidden[i, 47].tobytes() for i in np.flatnonzero(target_mask)}
        other = np.stack([other[j] for j in indices if other[j].tobytes() not in forbidden])
        k = np.concatenate((source["preservation"], np.stack(negative), other))
        np.savez_compressed(root/"selector-inputs.npz", hidden=capture_hidden, positive=h, preservation=k)
        save(root/"selector-inputs.json", {"structured_roles": key_roles,
            "historical_positive_rows": len(source["positive"]), "new_positive_rows": int(target_mask.sum()),
            "historical_preservation_rows": len(source["preservation"]),
            "structured_preservation_rows": len(negative), "other_preservation_rows": len(other),
            "capture_sha256": sha(root/"selector-inputs.npz")})
        del capture_hidden
        candidates, fits, factors = [], {}, {}
        ridge = .001*float(np.mean(np.sum(h.astype(np.float64)**2, axis=1)))
        (root/"factors").mkdir()
        for i, penalty in enumerate((1., 100.)):
            left, right, report = fit_response_selector(h, direction, k, preserve_weight=penalty, ridge=ridge)
            key = "role"+str(i)
            factors[key] = left, right
            fits[key] = report
            np.savez(root/"factors"/(key+".npz"), left=left, right=right)
            for dose in (.75, 1., 1.25):
                candidates.append({"key": key, "dose": dose})
        save(root/"fits.json", fits)
        with WeightStudy(root/"discovery", **options) as study:
            for i, candidate in enumerate(candidates):
                study.add_arm("candidate"+str(i), role="edited", state_policy="restart", context=candidate,
                    predictor=lambda ctx, candidate=candidate: {"predictions": {"candidate": candidate,
                        "request": "a.units 7->3", "condition": "native selector, no external position mask"}})
            study.seal()
            for i, candidate in enumerate(candidates):
                current_factors, current_dose = factors[candidate["key"]], candidate["dose"]
                def execute(ctx, candidate=candidate):
                    with virtual_low_rank_edit(model, site, *factors[candidate["key"]], dose=candidate["dose"]):
                        return evaluate(generation=False)
                arm = "candidate"+str(i)
                study.run(arm, execute)
                record(study, arm)
        verifications["discovery"] = verify_weight_study(root/"discovery")
        observed = [(i, c) for i, c in enumerate(candidates) if "error" not in results["candidate"+str(i)]]
        winner, selected = max(observed, key=lambda pair: results["candidate"+str(pair[0])]["objective"])
        left, right = factors[selected["key"]]
        np.savez(root/"selected.npz", left=left, right=right)
        patch = synthesize_low_rank_patch(model, site, left, right, dose=selected["dose"], model_identity=binding["model"])
        # Persist only the manifest here. Fresh replay must reproduce its exact
        # edited tensor bytes; original constants remain retrieval-bound.
        save(root/"selected-patch-manifest.json", patch.manifest)
        selection = {"schema": "mdb-role-selector-selection-v1", "selected": selected,
            "arm": "candidate"+str(winner), "development_a": sorted({r["a"] for r in rows}),
            "policy": "maximum development rewritten-target log probability minus control KL; six fixed candidates",
            "direction_sha256": sha(root/"upstream.npz"), "factors_sha256": sha(root/"selected.npz"),
            "manifest_sha256": sha(root/"selected-patch-manifest.json"), "job_id": os.environ["MRUN_JOB_ID"],
            "terminal_status": "not-evaluated"}
        save(root/"selection.json", selection)
    else:
        selected = chosen["selected"]
        patch = synthesize_low_rank_patch(model, site, *fitted, dose=selected["dose"], model_identity=binding["model"])
        old_patch = synthesize_low_rank_patch(model, site, *old, dose=old_dose, model_identity=binding["model"])
        if old_patch.manifest != upstream["old_patch_manifest"]:
            raise WeightSpaceError("original broad patch bytes differ")
        expected = json.loads(Path("selected-patch-manifest.json").read_text())
        if patch.manifest != expected:
            raise WeightSpaceError("frozen compiled parameter bytes differ")
        save(root/"selected-patch-manifest.json", patch.manifest)
        with WeightStudy(root/"evaluation", **options) as study:
            for name in ("native", "broad", "role-compiled", "role-factored", "restored"):
                study.add_arm(name, role="native" if name == "native" else "restoration" if name == "restored" else "edited",
                    patch=patch if name == "role-compiled" else old_patch if name == "broad" else None, state_policy="restart",
                    predictor=lambda ctx, name=name: {"predictions": {"arm": name,
                        "frozen_before_fresh_release": True, "rewrite": "a.units 7->3; other roles unchanged"}})
            study.seal()
            for name in ("native", "broad", "role-compiled", "role-factored", "restored"):
                current_factors, current_dose = (old, old_dose) if name == "broad" else (fitted, selected["dose"])
                manager = (virtual_low_rank_edit(model, site, *fitted, dose=selected["dose"]) if name == "role-factored"
                    else nullcontext())
                def execute(ctx, manager=manager, name=name):
                    with manager:
                        return evaluate(generation=True, traced=True, capture=name == "native")
                study.run(name, execute)
                record(study, name)
        verifications["evaluation"] = verify_weight_study(root/"evaluation")
        np.savez_compressed(root/"evaluation-inputs.npz", hidden=capture_hidden)
    summary = {"schema": "mdb-role-selector-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "phase": phase, "rows": len(rows),
        "target_rows": sum(r["matched"] for r in rows), "results": results,
        "parameters_restored": parameter_inventory(model) == original, "verifications": verifications,
        "elapsed_s": time.monotonic()-started, "terminal_status": "not-evaluated"}
    save(root/"summary.json", summary)
    save(root/"files.json", {p.relative_to(root).as_posix(): sha(p)
        for p in sorted(root.rglob("*")) if p.is_file()})
    from flux_family_custody import emit
    emit(root, summary, compression_level=1)


if __name__ == "__main__":
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()),
        json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text()))
