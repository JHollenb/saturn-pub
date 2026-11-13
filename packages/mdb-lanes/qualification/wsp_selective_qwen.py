"""Bounded preservation-aware refinement of the previously discovered MLP edit."""
from __future__ import annotations

import json
import os
import runpy
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path

from wsp_auto_qwen import save, sha


def run(config, request):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from mdb_lanes.wsp import WeightStudy, verify_weight_study
    from mdb_lanes.wsp_discovery import AffineEdit, _plain, synthesize_affine_patch
    from mdb_lanes.wsp_selective import (
        fit_low_rank_delta,
        synthesize_low_rank_patch,
        virtual_low_rank_edit,
    )
    from mdb_lanes.wsp_weights import WeightSpaceError, parameter_inventory

    started = time.monotonic()
    root = Path(config["output_scratch"]) / ("wsp-selective-" + os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "config.json", config)
    save(root / "request.json", request)
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    checkpoint = Path(config["model_path"])
    if sha(checkpoint / "model.safetensors") != config["checkpoint_sha256"]:
        raise WeightSpaceError("checkpoint identity differs")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2ForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
    runpy.run_path("qwen_family_campaign.py")["prepare_parameters"](checkpoint, model, root)
    original = parameter_inventory(model)
    module = model.get_submodule(config["site"])
    old_weight = module.weight.detach().clone()
    old_bias = module.bias.detach().clone() if module.bias is not None else None
    binding = {"model": {"checkpoint_sha256": config["checkpoint_sha256"]},
        "checkpoint": config["checkpoint_sha256"], "execution": "CPU FP32 eager, left-padded fixed batch per split",
        "consumer": "full-vocabulary last-digit first-token choice; native-distribution preservation",
        "horizon": "one-token score plus four own-state tokens on four evaluation cases"}
    save(root / "runtime.json", {"torch": str(torch.__version__), "numpy": np.__version__,
        "model_class": type(model).__name__, "device": "cpu", "dtype": "float32", "gradients": False,
        "generation": {"do_sample": False, "repetition_penalty": 1., "max_new_tokens": 4,
            "use_cache": True, "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id},
        "checkpoint_generation_config": model.generation_config.to_dict()})

    def cases(seed, count, exclude=()):
        rng, result, seen = np.random.default_rng(seed), [], set(exclude)
        while len(result) < count:
            digit = len(result) % 10
            a = int(rng.integers(10, 90))
            b = int(rng.integers(1, 8)) * 10 + ((digit - a) % 10)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            result.append({"a": a, "b": b, "answer": str((a + b) % 10),
                "prompt": "Write only the last digit of the sum.\n17 + 25 = 2\n36 + 48 = 4\n"
                          + str(a) + " + " + str(b) + " = "})
        return result

    first = cases(20261110, 8)
    second = cases(20261111, 8)
    discovery = first + second  # Both previously opened splits are now development data.
    preservation = ["The capital of France is", "The opposite of hot is", "A week has", "Water freezes at",
                    "The capital of Japan is", "The opposite of up is", "A triangle has", "The color of snow is"]
    # Evaluation inputs are deterministic but no activations/outcomes enter the fit or selection.
    heldout = cases(config["heldout_seed"], 20, ((r["a"], r["b"]) for r in discovery))
    fresh_preservation = ["The largest ocean is", "The chemical symbol for gold is",
        "The author of Hamlet was", "The square root of nine is", "A year has",
        "The opposite of empty is", "The language spoken in Brazil is",
        "The planet closest to the Sun is", "The currency of the United States is",
        "Birds use their wings to", "The first month of the year is", "The main ingredient in bread is"]
    save(root / "inputs.json", {"discovery": discovery, "discovery_preservation": preservation,
        "heldout": heldout, "heldout_preservation": fresh_preservation,
        "oracle": "supplied integer arithmetic", "split_policy": "previous task holdout now development; new task/preservation outcomes quarantined until selection"})

    class Evaluator:
        def __init__(self, task, protected):
            self.task, self.protected, self.anchor = task, protected, None
            self.inputs = tokenizer([r["prompt"] for r in task] + protected,
                padding=True, return_tensors="pt", add_special_tokens=False)
            if self.inputs.input_ids.shape[1] > config["max_prompt_tokens"]:
                raise WeightSpaceError("prompt geometry exceeded")
            targets = [tokenizer.encode(r["answer"], add_special_tokens=False) for r in task]
            if any(len(t) != 1 for t in targets):
                raise WeightSpaceError("non-single-token target")
            self.targets = torch.tensor([t[0] for t in targets])

        def __call__(self, *, capture=False, retain_logits=False, generate=False):
            if time.monotonic() - started > config["max_wall_seconds"]:
                raise WeightSpaceError("scientific wall budget exceeded")
            captured = {}
            def hook(layer, args, output):
                n = len(self.task)
                captured["hidden"] = args[0][:n, -1].detach().numpy().copy()
                captured["desired_change"] = -output[:n, -1].detach().numpy().copy()
                mask = self.inputs.attention_mask[n:].bool()
                captured["preservation"] = args[0][n:][mask].detach().numpy().copy()
            handle = module.register_forward_hook(hook) if capture else None
            try:
                logits = model(**self.inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            finally:
                if handle:
                    handle.remove()
            logp = logits.log_softmax(-1)
            n = len(self.task)
            if self.anchor is None:
                self.anchor = logp[n:].detach().clone()
            kl = (self.anchor.exp() * (self.anchor - logp[n:])).sum(-1)
            correct = logp[:n][torch.arange(n), self.targets]
            winners = logits[:n].argmax(-1)
            observed = {"objective": float(correct.mean() - config["preserve_weight"] * kl.mean()),
                "task_mean_logp": float(correct.mean()), "preservation_kl": float(kl.mean()),
                "correct": int((winners == self.targets).sum()), "rows": n,
                "winner_ids": winners.tolist(), "target_ids": self.targets.tolist(),
                "target_logp": correct.tolist(), "preservation_kl_rows": kl.tolist()}
            if capture:
                observed["fit_samples"] = captured
            if retain_logits:
                observed["logits"] = logits.detach().numpy().copy()
            if generate:
                inputs = tokenizer([r["prompt"] for r in self.task[:4]], padding=True,
                                   return_tensors="pt", add_special_tokens=False)
                tokens = model.generate(**inputs, max_new_tokens=4, do_sample=False, repetition_penalty=1.,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                new = tokens[:, inputs.input_ids.shape[1]:]
                observed.update(generated_ids=new.tolist(), generated_text=tokenizer.batch_decode(new))
            return {"observations": observed,
                "work": {"physical_forwards": 1, "batch_size": len(logits),
                    "padded_tokens": int(self.inputs.input_ids.numel()), "logits_to_keep": 1,
                    "generation_rows": 4 if generate else 0, "max_new_tokens": 4 if generate else 0},
                "state": {"policy": "restart", "use_cache": False,
                    "generation": "fresh prefix; branch owns generated tokens and full native KV"}}

    def progress(row):
        row = {**row, "elapsed_s": time.monotonic() - started}
        with (root / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print("WSP_SELECTIVE_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)

    dev = Evaluator(discovery, preservation)
    source = {"worker_sha256": sha(__file__), "predecessor": config["predecessor"]}
    kwargs = {"model": model, "binding": binding, "source_identity": source,
              "budgets": {"max_wall_ms": config["max_wall_seconds"] * 1000}}
    with WeightStudy(root / "capture", input_identity={"task": discovery, "preservation": preservation}, **kwargs) as study:
        study.add_arm("native", role="native", state_policy="restart", predictor=lambda c:
            {"predictions": {"desired_response": "negative native write at last task token; zero on all valid preservation tokens"}})
        study.seal()
        study.run("native", lambda c: dev(capture=True))
        samples = _plain(study.tx.observation("native")["result"]["observations"]["fit_samples"])
    h, r, k = (np.asarray(samples[key]) for key in ("hidden", "desired_change", "preservation"))
    ridge = config["ridge_relative"] * float(np.mean(np.sum(h * h, axis=1)))
    factors, candidates, fit_reports = {}, [], {}
    (root / "factors").mkdir()
    for penalty in config["fit_preserve_weights"]:
        for rank in config["ranks"]:
            left, right, report = fit_low_rank_delta(h, r, k, rank=rank, preserve_weight=penalty, ridge=ridge)
            key = "rank" + str(rank) + "-preserve" + str(penalty)
            factors[key] = (left, right)
            np.savez(root / "factors" / (key + ".npz"), left=left, right=right)
            report["relative_weight_frobenius"] = float(np.linalg.norm(left @ right) / old_weight.double().norm())
            fit_reports[key] = report
            for dose in config["doses"]:
                candidates.append({"key": key, "rank": rank, "fit_preserve_weight": penalty, "dose": dose})
    save(root / "fits.json", {"site": config["site"], "reports": fit_reports,
        "supplied": "site and suppression direction from previous experiment; task/preservation examples",
        "learned": "low-rank response coefficients and discovery-selected rank/penalty/dose",
        "rank_is_not_sparsity": True})
    kwargs["source_identity"] = {**source, "fits_sha256": sha(root / "fits.json"),
        "factor_sha256": {p.name: sha(p) for p in sorted((root / "factors").iterdir())}}
    # Native wins ties. Whole-write deletion is a reference, outside the narrower grammar.
    records = []
    full = synthesize_affine_patch(model, AffineEdit(config["site"], "write-gain", 0.), model_identity=binding["model"])
    with WeightStudy(root / "discovery", input_identity={"task": discovery, "preservation": preservation}, **kwargs) as study:
        for name, edit in [("native", None), ("whole-write", None)] + [("candidate" + str(i), c) for i, c in enumerate(candidates)]:
            study.add_arm(name, role="native" if name == "native" else "edited", state_policy="restart",
                patch=full if name == "whole-write" else None, context={"candidate": edit},
                predictor=lambda c, edit=edit: {"predictions": {"candidate": edit,
                    "consumer_effect": "unknown; local fit is a proposal"}})
        study.seal()
        for name, edit in [("native", None), ("whole-write", None)] + [("candidate" + str(i), c) for i, c in enumerate(candidates)]:
            def evaluate(context, edit=edit):
                manager = virtual_low_rank_edit(model, config["site"], *factors[edit["key"]], dose=edit["dose"]) if edit else nullcontext()
                with manager:
                    return dev()
            study.run(name, evaluate)
            observation = _plain(study.tx.observation(name)["result"])
            if "error" in observation:
                records.append({"arm": name, "candidate": edit, "error": observation["error"], "objective": None})
            else:
                records.append({"arm": name, "candidate": edit, **observation["observations"]})
            progress({"phase": "discovery", "arm": name, "objective": records[-1]["objective"]})
    eligible = [row for row in records if row["arm"] != "whole-write" and row["objective"] is not None]
    selected = max(eligible, key=lambda row: row["objective"])
    chosen = selected["candidate"]
    save(root / "selection.json", {"selected": chosen, "selected_arm": selected["arm"], "records": records,
        "policy": "maximum discovery task mean logp minus preservation mean KL; native first wins ties; whole-write reference excluded",
        "terminal_status": "not-evaluated"})
    if chosen:
        a, b = factors[chosen["key"]]
        patch = synthesize_low_rank_patch(model, config["site"], a, b,
            dose=chosen["dose"], model_identity=binding["model"])
        patch.save(root / "selected-patch")
    else:
        a, b, patch = np.zeros((module.out_features, 1)), np.zeros((1, module.in_features)), None
    # Only now instantiate the fresh evaluator or consume its model outputs.
    final = Evaluator(heldout, fresh_preservation)
    arms = ("native", "whole-write", "virtual", "compiled", "restored", "wrong-sign")
    outcomes = {}
    with WeightStudy(root / "heldout", input_identity={"task": heldout, "preservation": fresh_preservation}, **kwargs) as study:
        for name in arms:
            study.add_arm(name, role="native" if name == "native" else "restoration" if name == "restored" else "edited",
                state_policy="restart", patch=full if name == "whole-write" else patch if name in {"compiled", "restored"} else None,
                context={"selected": chosen}, predictor=lambda c, name=name: {"predictions": {
                    "arm": name, "task_and_locality": "unknown on fresh inputs",
                    "parity": "factored and compiled same real-arithmetic map; FP32 rounding measured"}})
        study.seal()
        for name in arms:
            def evaluate(context, name=name):
                handle = None
                if name == "restored":
                    handle = module.register_forward_hook(lambda layer, args, output:
                        torch.nn.functional.linear(args[0], old_weight, old_bias))
                manager = virtual_low_rank_edit(model, config["site"], a, b,
                    dose=(chosen["dose"] if chosen else 0.) * (-1 if name == "wrong-sign" else 1)) if name in {"virtual", "wrong-sign"} else nullcontext()
                try:
                    with manager:
                        return final(retain_logits=True, generate=True)
                finally:
                    if handle:
                        handle.remove()
            study.run(name, evaluate)
            result = study.tx.observation(name)["result"]
            outcomes[name] = _plain({key: value for key, value in result.get("observations", {}).items() if key != "logits"})
            if "error" in result:
                outcomes[name]["error"] = _plain(result["error"])
            progress({"phase": "heldout", "arm": name, "correct": outcomes[name].get("correct"),
                      "preservation_kl": outcomes[name].get("preservation_kl")})
    summary = {"schema": "mdb-selective-qwen-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "selected": chosen, "heldout": outcomes, "parameters_restored": parameter_inventory(model) == original,
        "heldout_verification": verify_weight_study(root / "heldout"),
        "elapsed_s": time.monotonic() - started, "terminal_status": "not-evaluated"}
    save(root / "summary.json", summary)
    save(root / "files.json", {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*")) if p.is_file()})
    archive = root.parent / (root.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(root, arcname="evidence")
    print("WSP_SELECTIVE_RESULT=" + json.dumps({"summary": summary, "artifact": {
        "path": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size,
        "files_manifest_sha256": sha(root / "files.json"), "summary_sha256": sha(root / "summary.json")}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()),
        json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text()))
