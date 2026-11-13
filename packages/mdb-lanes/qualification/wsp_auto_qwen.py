"""One finite native-Qwen automatic affine-edit discovery experiment."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import tarfile
import time
from pathlib import Path


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(config, request):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from mdb_lanes.wsp import WeightStudy, verify_weight_study
    from mdb_lanes.wsp_discovery import AffineDiscovery, _plain, virtual_edit
    from mdb_lanes.wsp_weights import WeightSpaceError, parameter_inventory

    started = time.monotonic()
    root = Path(config["output_scratch"]) / ("wsp-auto-" + os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "config.json", config)
    save(root / "request.json", request)
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    checkpoint = Path(config["model_path"])
    if sha(checkpoint / "model.safetensors") != config["checkpoint_sha256"]:
        raise ValueError("original checkpoint bytes differ")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2ForCausalLM.from_pretrained(checkpoint, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
    runpy.run_path("qwen_family_campaign.py")["prepare_parameters"](checkpoint, model, root)
    original = parameter_inventory(model)
    binding = {"model": {"checkpoint_sha256": config["checkpoint_sha256"]},
        "checkpoint": config["checkpoint_sha256"], "execution": "native CPU FP32 eager; fixed left-padded batch",
        "consumer": "last-digit addition log probability and native-distribution preservation",
        "horizon": "one-token selection; separate four-token own-state continuation"}
    # Deterministic task construction is frozen in config. No held-out examples
    # or outcomes are passed to the discovery evaluator or controller.
    def cases(seed, count):
        rng = np.random.default_rng(seed)
        result = []
        for digit in range(count):
            a = int(rng.integers(10, 90))
            b = int(rng.integers(1, 8)) * 10 + ((digit - a) % 10)
            result.append({"a": a, "b": b, "answer": str((a + b) % 10),
                "prompt": "Write only the last digit of the sum.\n17 + 25 = 2\n36 + 48 = 4\n"
                          + str(a) + " + " + str(b) + " = "})
        return result
    discovery = cases(config["discovery_seed"], 8)
    heldout = cases(config["heldout_seed"], 8)
    preservation = ["The capital of France is", "The opposite of hot is", "A week has", "Water freezes at",
                    "The capital of Japan is", "The opposite of up is", "A triangle has", "The color of snow is"]
    save(root / "inputs.json", {"discovery": discovery, "heldout": heldout,
                                "preservation": preservation, "oracle": "supplied integer arithmetic"})
    save(root / "runtime.json", {"torch": str(torch.__version__), "numpy": np.__version__,
        "model_class": type(model).__name__, "device": "cpu", "dtype": "float32", "gradients": False,
        "max_sites": config["max_sites"], "max_wall_seconds": config["max_wall_seconds"]})

    class Evaluator:
        def __init__(self, task, *, anchor=None):
            self.task = task
            self.inputs = tokenizer([row["prompt"] for row in task] + preservation,
                                    padding=True, return_tensors="pt", add_special_tokens=False)
            if self.inputs.input_ids.shape[1] > config["max_prompt_tokens"]:
                raise ValueError("declared prompt bound exceeded")
            target = [tokenizer.encode(row["answer"], add_special_tokens=False) for row in task]
            if any(len(tokens) != 1 for tokens in target):
                raise ValueError("oracle answer must have one native token")
            self.targets = torch.tensor([tokens[0] for tokens in target])
            self.anchor = anchor

        def __call__(self, net):
            if time.monotonic() - started > config["max_wall_seconds"]:
                raise WeightSpaceError("finite scientific wall budget exceeded")
            logits = net(**self.inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            logp = logits.log_softmax(-1)
            task_logp, preserve_logp = logp[:len(self.task)], logp[len(self.task):]
            if self.anchor is None:
                self.anchor = preserve_logp.detach().clone()
            kl = (self.anchor.exp() * (self.anchor - preserve_logp)).sum(-1)
            correct = task_logp[torch.arange(len(self.task)), self.targets]
            winners = task_logp.argmax(-1)
            objective = float(correct.mean() - config["preserve_weight"] * kl.mean())
            return {"observations": {"objective": objective,
                "task_mean_logp": float(correct.mean()), "preservation_kl": float(kl.mean()),
                "correct": int((winners == self.targets).sum()), "rows": len(self.task),
                "winner_ids": winners.tolist(), "target_ids": self.targets.tolist(),
                "target_logp": correct.tolist(), "preservation_kl_rows": kl.tolist()},
                "work": {"physical_forwards": 1, "batch_size": len(logits),
                    "padded_tokens": int(self.inputs.input_ids.numel()), "logits_to_keep": 1},
                "state": {"policy": "restart", "use_cache": False, "padding": "left",
                          "sequence_length": self.inputs.input_ids.shape[1]}}

    evaluator = Evaluator(discovery)
    def progress(row):
        row = {**row, "elapsed_s": time.monotonic() - started}
        with (root / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print("WSP_AUTO_PROGRESS=" + json.dumps(row, sort_keys=True), flush=True)

    controller = AffineDiscovery(model, root / "discovery", binding=binding,
        input_identity={"examples": discovery, "preservation": preservation},
        source_identity={"worker_sha256": sha(__file__)}, max_sites=config["max_sites"],
        shortlist=2, channels_per_site=1, max_wall_ms=config["max_wall_seconds"] * 1000,
        on_progress=progress)
    chosen, patch, selection = controller.discover(evaluator)
    heldout_evaluator = Evaluator(heldout, anchor=evaluator.anchor)
    selected_module = dict(model.named_modules())[chosen.site] if chosen else None
    old_weight = selected_module.weight.detach().clone() if chosen else None
    old_bias = selected_module.bias.detach().clone() if chosen and selected_module.bias is not None else None
    generation_inputs = tokenizer([row["prompt"] for row in heldout[:4]], padding=True,
                                  return_tensors="pt", add_special_tokens=False)
    final_root = root / "heldout"
    with WeightStudy(final_root, model=model, binding=binding,
        input_identity={"heldout_examples": heldout, "preservation": preservation},
        source_identity={"selection_sha256": sha(root / "discovery/selection.json")},
        budgets={"max_wall_ms": config["max_wall_seconds"] * 1000}) as study:
        for name in ("native", "virtual", "compiled", "restored"):
            study.add_arm(name, role="native" if name == "native" else "restoration" if name == "restored" else "edited",
                patch=patch if name in {"compiled", "restored"} else None, state_policy="restart",
                predictor=lambda context, name=name: {"predictions": {
                    "arm": name, "selected_edit": selection["selected"],
                    "forecast": "virtual/compiled affine equivalence; restoration returns original map",
                    "task_effect": "unmeasured on held-out cases"}})
        study.seal()
        for name in ("native", "virtual", "compiled", "restored"):
            def evaluate(context, name=name):
                def consume():
                    result = heldout_evaluator(context.model)
                    tokens = context.model.generate(**generation_inputs, max_new_tokens=4, do_sample=False,
                        use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                    new = tokens[:, generation_inputs.input_ids.shape[1]:]
                    result["observations"]["generated_ids"] = new.tolist()
                    result["observations"]["generated_text"] = tokenizer.batch_decode(new)
                    result["work"]["generation_batch"] = len(new)
                    result["work"]["max_new_tokens"] = 4
                    result["state"]["generation"] = "fresh prefix; each branch owns generated tokens and full native KV"
                    return result
                if name == "virtual" and chosen:
                    with virtual_edit(context.model, chosen):
                        return consume()
                if name == "restored" and chosen:
                    hook = selected_module.register_forward_hook(lambda module, args, output:
                        torch.nn.functional.linear(args[0], old_weight, old_bias))
                    try:
                        return consume()
                    finally:
                        hook.remove()
                return consume()
            study.run(name, evaluate)
        observations = {name: _plain(study.tx.observation(name)["result"])
                        for name in ("native", "virtual", "compiled", "restored")}
    summary = {"schema": "mdb-native-qwen-affine-discovery-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "scope": "bounded gain/channel discovery on supplied arithmetic task; no WSP loop-repair reproduction",
        "selected": selection["selected"], "universe_size": selection["universe_size"],
        "scanned_sites": selection["scanned_sites"], "discovery_candidates": len(selection["records"]),
        "discovery_baseline_objective": selection["baseline_objective"],
        "discovery_selected_objective": selection["selected_objective"], "heldout": observations,
        "parameters_restored": parameter_inventory(model) == original,
        "heldout_verification": verify_weight_study(final_root),
        "elapsed_s": time.monotonic() - started, "terminal_status": "not-evaluated"}
    save(root / "summary.json", summary)
    # Seal every array/file in the finished native study and discovery rounds.
    files = {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file()}
    save(root / "files.json", files)
    archive = root.parent / (root.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(root, arcname="evidence")
    artifact = {"path": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size,
                "files_manifest_sha256": sha(root / "files.json"), "summary_sha256": sha(root / "summary.json")}
    print("WSP_AUTO_RESULT=" + json.dumps({"summary": summary, "artifact": artifact}, sort_keys=True), flush=True)


if __name__ == "__main__":
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()),
        json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text()))
