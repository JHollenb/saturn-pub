"""Lower an evidence-backed operand rewrite into recipient weight bytes."""
from __future__ import annotations

import json
import os
import runpy
import tarfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from wsp_auto_qwen import save, sha


def run(config, request):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from mdb.wsp import WeightStudy, verify_weight_study
    from mdb.wsp_discovery import _plain
    from mdb.wsp_program import DigitRewrite
    from mdb.wsp_selective import (
        fit_low_rank_delta,
        synthesize_low_rank_patch,
        virtual_low_rank_edit,
    )
    from mdb.wsp_weights import WeightSpaceError, parameter_inventory

    started = time.monotonic()
    root = Path(config["output_scratch"]) / ("wsp-program-" + os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "config.json", config)
    save(root / "request.json", request)
    account = json.loads(Path("computation-account.json").read_text())
    for entry in account["evidence"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise WeightSpaceError("interpretation evidence changed")
    save(root / "computation-account.json", account)
    (root / "prior-evidence").mkdir()
    for entry in account["evidence"]:
        (root / "prior-evidence" / Path(entry["path"]).name).write_bytes(Path(entry["path"]).read_bytes())
    program = DigitRewrite(**config["program"])
    recipe = program.lower(account, checkpoint_sha256=config["checkpoint_sha256"], execution=config["execution"])
    save(root / "recipe.json", recipe)
    torch.set_num_threads(config["cpu_threads"])
    checkpoint = Path(config["model_path"])
    if sha(checkpoint / "model.safetensors") != config["checkpoint_sha256"]:
        raise WeightSpaceError("checkpoint identity differs")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2ForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
    runpy.run_path("qwen_family_campaign.py")["prepare_parameters"](checkpoint, model, root)
    original = parameter_inventory(model)
    sites = {row["module"]: model.get_submodule(row["module"]) for row in recipe["writers"]}
    position = account["symbols"][program.symbol]["token_position"]
    prefix = account["prompt_prefix"]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(prefix_ids) != 45 or [tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)] != [[15+i] for i in range(10)]:
        raise WeightSpaceError("historical token-role layout is inapplicable")
    binding = {"model": {"checkpoint_sha256": config["checkpoint_sha256"]},
        "checkpoint": config["checkpoint_sha256"], "execution": config["execution"],
        "consumer": "original Qwen addition suffix and full vocabulary; authored operand rewrite",
        "horizon": "last answer digit with supplied unchanged prefix; separate three own-state answer tokens"}
    kwargs = {"model": model, "binding": binding, "source_identity": {"recipe": recipe["identity"], "worker": sha(__file__)},
        "budgets": {"max_wall_ms": config["max_wall_seconds"]*1000, "max_bytes": 512*1024*1024}}

    def progress(**row):
        row["elapsed_s"] = time.monotonic() - started
        with (root / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
        print("WSP_PROGRAM_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)

    def cases(seed, count, exclude=(), matched=True):
        rng, rows, seen = np.random.default_rng(seed), [], set(exclude)
        while len(rows) < count:
            leading = int(rng.integers(10, 50))
            digit = program.source if matched else int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 9]))
            if not matched and len(rows) % 2 == 0:
                leading = 10*int(rng.integers(1, 5)) + program.source
            a = 10*leading + digit
            # Both original and rewritten sums stay in the same carry class.
            b = 10*int(rng.integers(10, 40)) + int(rng.integers(0, 3))
            if not matched and len(rows) % 2:
                b = b//10*10 + program.source
            if a in seen:
                continue
            seen.add(a)
            aa, bb = program.apply(a, b)
            before, after = str(a+b), str(aa+bb)
            if len(before) != 3 or before[:2] != after[:2]:
                raise WeightSpaceError("declared no-carry-change domain violated")
            rows.append({"a": a, "b": b, "original": before, "expected": after,
                "matched": matched, "prompt": prefix + str(a) + "+" + str(b) + "=",
                "donor_prompt": prefix + str(aa) + "+" + str(bb) + "="})
        return rows

    dev = cases(config["discovery_seed"], 16)
    controls = cases(config["discovery_seed"]+1, 8, matched=False)
    fresh = cases(config["heldout_seed"], 16, (r["a"] for r in dev))
    fresh_controls = cases(config["heldout_seed"]+1, 8, (r["a"] for r in controls), matched=False)
    protected = ["The capital of France is", "The opposite of hot is", "A week has", "Water freezes at",
        "The capital of Japan is", "The opposite of up is", "A triangle has", "The color of snow is"]
    fresh_protected = ["The largest ocean is", "The chemical symbol for gold is", "The author of Hamlet was",
        "A year has", "The opposite of empty is", "The language spoken in Brazil is",
        "The planet closest to the Sun is", "Birds use their wings to"]
    save(root / "inputs.json", {"development": dev, "development_controls": controls,
        "development_preservation": protected, "heldout": fresh, "heldout_controls": fresh_controls,
        "heldout_preservation": fresh_protected, "program": config["program"],
        "domain": "fixed three-digit format; units addends 0..2 so 7->3 never changes a carry",
        "supplied": "symbolic request, role map, arithmetic labels and common two-digit answer prefix",
        "learned": "contextual response direction, input selector, writer choice, preservation penalty and dose"})

    class Evaluator:
        def __init__(self, task, ctrl, general):
            self.task, self.ctrl, self.general, self.anchor = task, ctrl, general, None
            self.rows = task + ctrl
            self.inputs = tokenizer([r["prompt"]+r["original"][:2] for r in self.rows] + general,
                padding=True, return_tensors="pt", add_special_tokens=False)
            if self.inputs.input_ids.shape[1] > 64:
                raise WeightSpaceError("prompt bound exceeded")
            self.labels = torch.tensor([15+int(r["expected"][-1]) for r in self.rows])
            self.mask = torch.tensor([r["matched"] for r in self.rows] + [False]*len(general))

        def __call__(self, *, full=False, generate=False):
            if time.monotonic()-started > config["max_wall_seconds"]:
                raise WeightSpaceError("finite scientific wall budget exceeded")
            logits = model(**self.inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            logp = logits.log_softmax(-1)
            n, nrows = len(self.task), len(self.rows)
            if self.anchor is None:
                self.anchor = logp[n:].detach().clone()
            kl = (self.anchor.exp()*(self.anchor-logp[n:])).sum(-1)
            target = logp[:nrows][torch.arange(nrows), self.labels]
            winners = logits[:nrows].argmax(-1)
            obs = {"objective": float(target[:n].mean()-kl.mean()),
                "rewrite_correct": int((winners[:n]==self.labels[:n]).sum()), "task_rows": n,
                "control_correct": int((winners[n:]==self.labels[n:]).sum()), "control_rows": len(self.ctrl),
                "winner_ids": winners.tolist(), "target_ids": self.labels.tolist(), "target_logp": target.tolist(),
                "mean_rewrite_logp": float(target[:n].mean()), "preservation_kl": float(kl.mean()),
                "preservation_kl_rows": kl.tolist()}
            if full:
                obs["logits"] = logits.detach().numpy().copy()
            if generate:
                inputs = tokenizer([r["prompt"] for r in self.rows], padding=True, return_tensors="pt", add_special_tokens=False)
                tokens = model.generate(**inputs, max_new_tokens=3, do_sample=False, repetition_penalty=1.,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                new = tokens[:, inputs.input_ids.shape[1]:]
                expected = [[15+int(ch) for ch in r["expected"]] for r in self.rows]
                original_ids = [[15+int(ch) for ch in r["original"]] for r in self.rows]
                obs.update(generated_ids=new.tolist(), generated_text=tokenizer.batch_decode(new),
                    expected_ids=expected, original_ids=original_ids,
                    generated_rewrite_correct=sum(x==y for x,y in zip(new.tolist()[:n], expected[:n], strict=True)),
                    generated_control_correct=sum(x==y for x,y in zip(new.tolist()[n:], expected[n:], strict=True)))
            return {"observations": obs, "work": {"physical_forwards": 1, "batch_size": len(logits),
                "padded_tokens": int(self.inputs.input_ids.numel()), "generation_rows": len(self.rows) if generate else 0,
                "max_new_tokens": 3 if generate else 0},
                "state": {"policy": "restart", "scoring": "two supplied answer digits; final-digit full-vocabulary choice",
                    "generation": "separate fresh raw prefix, own tokens and native KV, no supplied answer digits"}}

    evaluator = Evaluator(dev, controls, protected)
    # All historical source locations are resolved before native outcomes.
    capture_prompts = [r["prompt"]+r["original"][:2] for r in dev] + [r["donor_prompt"]+r["expected"][:2] for r in dev]
    capture_prompts += [r["prompt"]+r["original"][:2] for r in controls] + protected
    capture_inputs = tokenizer(capture_prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    samples = {}
    with WeightStudy(root / "capture", input_identity={"prompts": capture_prompts}, **kwargs) as study:
        study.add_arm("contrasts", role="native", state_policy="restart", symbol_refs=(recipe["identity"],),
            predictor=lambda c: {"predictions": {"source": program.symbol, "sites": recipe["writers"],
                "hypothesis": "native counterfactual digit contrast supplies a reusable response direction"}})
        study.seal()
        def capture(context):
            handles = []
            for site, module in sites.items():
                def hook(layer, args, output, site=site):
                    h = args[0].detach().numpy()
                    y = output.detach().numpy()
                    mask = capture_inputs.attention_mask.bool().numpy().copy()
                    mask[:16, position] = False
                    raw = h[mask].copy()
                    # Deduplicate exact causal-prefix rows, then retain bounded
                    # deterministic support. This is a preservation sample.
                    unique, seen = [], set()
                    for row in raw:
                        identity = row.tobytes()
                        if identity not in seen:
                            unique.append(row)
                            seen.add(identity)
                    indices = np.linspace(0, len(unique)-1, min(128, len(unique)), dtype=int)
                    samples[site] = {"hidden": h[:16, position].copy(),
                        "desired": (y[16:32, position]-y[:16, position]).mean(0).copy(),
                        "preservation": np.stack([unique[i] for i in indices]),
                        "native_source": y[:16, position].copy(), "counterfactual_source": y[16:32, position].copy(),
                        "unique_preservation_rows": len(unique), "sampled_preservation_rows": len(indices)}
                handles.append(module.register_forward_hook(hook))
            # A down-projection update writes additively to the residual. Its
            # desired change is the whole counterfactual residual displacement,
            # not merely the difference between two native MLP writes.
            residual_writer = next(row for row in recipe["writers"] if row["target"]=="residual")
            def residual_hook(layer, args, output):
                value = output[0] if isinstance(output, tuple) else output
                y = value.detach().numpy()
                entry = samples[residual_writer["module"]]
                entry["desired"] = (y[16:32,position]-y[:16,position]).mean(0).copy()
                entry["native_source"] = y[:16,position].copy()
                entry["counterfactual_source"] = y[16:32,position].copy()
            handles.append(model.get_submodule(residual_writer["residual_module"]).register_forward_hook(residual_hook))
            try:
                model(**capture_inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            return {"observations": {"samples": samples}, "work": {"physical_forwards": 1, "batch_size": len(capture_prompts)},
                    "state": {"policy": "restart", "use_cache": False}}
        study.run("contrasts", capture)
    progress(phase="capture", sites=list(samples))

    @contextmanager
    def semantic(site, mask, *, sign=1., wrong_location=False):
        direction = torch.tensor(samples[site]["desired"], dtype=torch.float32)
        row = account["wrong_location"] if wrong_location else position
        def hook(layer, args, output):
            if output.shape[1] <= row:
                return output
            # This explicit predicate is the experimental reference only.
            # Compiled weight execution below has no token predicate or donor.
            result = output.clone()
            result[mask.to(result.device), row] += sign*direction.to(result.device)
            return result
        handle = sites[site].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    causal = []
    with WeightStudy(root / "causal", input_identity={"task": dev, "controls": controls, "preservation": protected}, **kwargs) as study:
        declarations = [("native", None, False)]
        declarations += [("source"+str(i), site, False) for i,site in enumerate(sites)]
        declarations += [("wrong-location"+str(i), site, True) for i,site in enumerate(sites)]
        for name, site, wrong in declarations:
            study.add_arm(name, role="native" if site is None else "edited", state_policy="restart",
                context={"site": site, "wrong_location": wrong}, symbol_refs=(recipe["identity"],),
                predictor=lambda c, wrong=wrong: {"predictions": {"intended": recipe["predictions"], "wrong_location": wrong}})
        study.seal()
        for name, site, wrong in declarations:
            def evaluate(context, site=site, wrong=wrong):
                with semantic(site, evaluator.mask, wrong_location=wrong) if site else nullcontext():
                    return evaluator()
            study.run(name, evaluate)
            result = _plain(study.tx.observation(name)["result"])
            causal.append({"arm": name, "site": site, "wrong_location": wrong, **result})
            progress(phase="causal", arm=name, observations=result.get("observations"), error=result.get("error"))
    save(root / "causal-selection.json", causal)
    valid = [r for r in causal if r["site"] and not r["wrong_location"] and "observations" in r]
    selected_site = max(valid, key=lambda r:r["observations"]["objective"])["site"]
    progress(phase="writer-selected", site=selected_site)
    s = samples[selected_site]
    h, k = s["hidden"], s["preservation"]
    desired = np.tile(s["desired"], (len(h), 1))
    ridge = .001*float(np.mean(np.sum(h*h, axis=1)))
    factors, fits, candidates = {}, {}, []
    (root / "factors").mkdir()
    for family in ("semantic", "random"):
        for index, penalty in enumerate((1., 100.)):
            r = desired
            if family == "random":
                v = np.random.default_rng(951+index).normal(size=desired.shape[1])
                v *= float(np.linalg.norm(s["desired"]))/np.linalg.norm(v)
                r = np.tile(v, (len(h), 1))
            key = family+str(index)
            left, right, report = fit_low_rank_delta(h, r, k, rank=1, preserve_weight=penalty, ridge=ridge)
            report["target_direction_norm"] = float(np.linalg.norm(r[0]))
            report["correction_frobenius"] = float(np.linalg.norm(left)*np.linalg.norm(right))
            factors[key], fits[key] = (left, right), report
            np.savez(root / "factors" / (key+".npz"), left=left, right=right, desired=r)
            for dose in (.5, 1., 1.5):
                candidates.append({"key": key, "family": family, "dose": dose, "site": selected_site})
    save(root / "fits.json", {"reports": fits, "site": selected_site,
        "baseline": "six norm-matched random-response candidates share the same site, keys and fit budgets",
        "source": "mean native counterfactual digit response, derived only on development contexts"})
    kwargs["source_identity"] = {**kwargs["source_identity"], "fits_sha256": sha(root / "fits.json"),
        "factors": {p.name: sha(p) for p in sorted((root / "factors").iterdir())}}
    records = []
    with WeightStudy(root / "discovery", input_identity={"task": dev, "controls": controls, "preservation": protected}, **kwargs) as study:
        for i,c in enumerate(candidates):
            study.add_arm("candidate"+str(i), role="edited", state_policy="restart", context=c,
                symbol_refs=(recipe["identity"],), predictor=lambda ctx,c=c: {"predictions": {
                    "candidate": c, "semantic_prediction": recipe["predictions"], "weight_realization": "unmeasured"}})
        study.seal()
        for i,c in enumerate(candidates):
            def evaluate(context,c=c):
                with virtual_low_rank_edit(model, selected_site, *factors[c["key"]], dose=c["dose"]):
                    return evaluator()
            name = "candidate"+str(i)
            study.run(name, evaluate)
            result = _plain(study.tx.observation(name)["result"])
            records.append({"arm": name, "candidate": c, **result})
            progress(phase="discovery", arm=name, observations=result.get("observations"), error=result.get("error"))
    chosen = {family:max((r for r in records if r["candidate"]["family"]==family and "observations" in r),
                        key=lambda r:r["observations"]["objective"])["candidate"] for family in ("semantic", "random")}
    save(root / "selection.json", {"site": selected_site, "selected": chosen, "records": records,
        "policy": "per family maximum development rewrite logp minus preservation KL; equal six candidate budgets",
        "causal_native": causal[0], "terminal_status": "not-evaluated"})
    patches = {family:synthesize_low_rank_patch(model, selected_site, *factors[c["key"]],
        dose=c["dose"], model_identity=binding["model"]) for family,c in chosen.items()}
    for family,patch in patches.items():
        patch.save(root / (family+"-patch"))
    old_weight = sites[selected_site].weight.detach().clone()
    old_bias = sites[selected_site].bias.detach().clone() if sites[selected_site].bias is not None else None
    final = Evaluator(fresh, fresh_controls, fresh_protected)
    arms = ("native", "semantic-program", "compiled", "factored", "wrong-location", "random", "restored")
    outcomes = {}
    with WeightStudy(root / "heldout", input_identity={"task": fresh, "controls": fresh_controls, "preservation": fresh_protected}, **kwargs) as study:
        for name in arms:
            patch = patches["semantic"] if name in {"compiled", "restored"} else patches["random"] if name=="random" else None
            study.add_arm(name, role="native" if name=="native" else "restoration" if name=="restored" else "edited",
                state_policy="restart", patch=patch, symbol_refs=(recipe["identity"],),
                predictor=lambda c,name=name: {"predictions": {"arm": name, "request": config["program"],
                    "expected": recipe["predictions"], "scope": "new first operands; no carry change"}})
        study.seal()
        for name in arms:
            def evaluate(context,name=name):
                handles = []
                if name=="restored":
                    handles.append(sites[selected_site].register_forward_hook(lambda layer,args,output:
                        torch.nn.functional.linear(args[0], old_weight, old_bias)))
                c = chosen["semantic"]
                if name in {"semantic-program", "wrong-location"}:
                    # Scoring and generation have different batch sizes. The
                    # hook uses the first task rows in either explicitly bound batch.
                    mask = torch.tensor([True]*len(fresh)+[False]*(len(fresh_controls)+len(fresh_protected)))
                    direction = torch.tensor(samples[selected_site]["desired"])
                    row = account["wrong_location"] if name=="wrong-location" else position
                    def hook(layer,args,output):
                        if output.shape[1] <= row:
                            return output
                        result = output.clone()
                        result[mask[:len(output)], row] += direction
                        return result
                    handles.append(sites[selected_site].register_forward_hook(hook))
                manager = virtual_low_rank_edit(model, selected_site, *factors[c["key"]], dose=c["dose"]) if name=="factored" else nullcontext()
                try:
                    with manager:
                        return final(full=True, generate=True)
                finally:
                    for handle in handles:
                        handle.remove()
            study.run(name, evaluate)
            result = study.tx.observation(name)["result"]
            outcomes[name] = _plain({k:v for k,v in result.get("observations",{}).items() if k!="logits"})
            if "error" in result:
                outcomes[name]["error"] = _plain(result["error"])
            progress(phase="heldout", arm=name, observations=outcomes[name])
    summary = {"schema": "mdb-computation-program-qwen-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "recipe": recipe, "selected": chosen, "heldout": outcomes,
        "parameters_restored": parameter_inventory(model)==original,
        "heldout_verification": verify_weight_study(root / "heldout"),
        "elapsed_s": time.monotonic()-started, "terminal_status": "not-evaluated"}
    save(root / "summary.json", summary)
    save(root / "files.json", {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*")) if p.is_file()})
    archive = root.parent / (root.name+".tar.gz")
    with tarfile.open(archive,"w:gz") as stream:
        stream.add(root,arcname="evidence")
    print("WSP_PROGRAM_RESULT="+json.dumps({"summary":summary,"artifact": {"path":str(archive), "sha256":sha(archive),
        "bytes":archive.stat().st_size,"files_manifest_sha256":sha(root/"files.json"),"summary_sha256":sha(root/"summary.json")}},sort_keys=True),flush=True)


if __name__ == "__main__":
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()),
        json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text()))
