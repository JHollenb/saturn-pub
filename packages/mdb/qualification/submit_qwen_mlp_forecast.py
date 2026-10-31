"""Admit one finite trained-Qwen conditional forecast and branch experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plan(*, full_readout=False, prompt_panel=None):
    if prompt_panel not in (None, "language-a", "language-b"):
        raise ValueError("unknown fixed prompt panel")
    if prompt_panel is not None and not full_readout:
        raise ValueError("fresh prompt panels require full-vocabulary readout")
    value = {
        "schema": "mdb-qwen-final-mlp-plan-v1",
        "model": "qwen2.5-0.5b-instruct",
        "model_path": "/mnt/big/llm-models/Qwen2.5-0.5B-Instruct",
        "expected_weights_sha256": "2b226fe4944dc744596b1142a15d6676160e9d06f245d31c1a36633f6c86baa3",
        "prompts": [
            {"id": "arithmetic", "text": "Complete the equation: 2 + 3 = "},
            {"id": "capital", "text": "The capital of France is"},
            {"id": "antonym", "text": "The opposite of hot is"},
        ],
        "candidate_texts": ["5", "4", "6", " 5", " 4", " Paris", "Paris", " London", "London",
                            " cold", "cold", " hot", "hot", ".", "\n"],
        "candidate_token_ids": [],
        "candidate_policy": "deduplicate only single-token literal encodings before any prefix; retain excluded strings",
        "arms": {"native": 1., "sham": 1., "zero": 0., "half": .5},
        "cpu_threads": 4,
        "seed": 19,
        "continuation_tokens": 3,
        "max_prompt_tokens": 32,
        "max_prefix_ops": 8192,
        "max_archive_bytes": 128 << 20,
        "frontier": "observed last-token residual before final post-attention RMSNorm",
        "prediction_scope": "last MLP, final residual/norm and fixed selected logits; no full-vocabulary or continuation forecast",
        "future_scope": "one scoped prefill edit; first native greedy token and three unedited cached native successors",
        "predeclared_invariants": [
            "zero full causal-LM calls and zero final-suffix entries before global seal; genuine decoder prefixes allowed",
            "independent selected suffix and full native call receive the same last-token boundary",
            "same native prompt cache across arms because final-MLP edit occurs after KV construction",
            "same first greedy token and same prompt cache imply identical unedited continuation",
        ],
        "limits": [
            "This is a prefix-conditioned trained-language-model assay, not a raw-prompt forecast.",
            "Native continuations are observations, not forecasts; no oracle tokens enter a branch.",
            "Candidates, prompts and doses are fixed before forecasts and native outputs.",
            "No required token divergence or scientific numeric-error threshold; retain all outcomes.",
        ],
    }
    if full_readout:
        value.update({"schema": "mdb-qwen-full-readout-plan-v1", "full_vocabulary_readout": True,
            "prompts": [{"id": "capital", "text": "The capital of France is"},
                        {"id": "planet", "text": "The largest planet in our solar system is"}],
            "arms": {"native": 1., "sham": 1., "zero": 0., "quarter": .25, "half": .5},
            "prediction_scope": "observed-prefix final MLP plus full-vocabulary first-token logits, winners and affine dose envelope; later tokens measured only"})
        value["candidate_texts"] += [" Jupiter", "Jupiter", " Mars", "Mars"]
        value["predeclared_invariants"].append(
            "bias-free final RMSNorm readout ranks tokens by a_v + dose*b_v; its denominator is shared and positive")
        value["limits"][1] = "Full-vocabulary first-token decisions are forecast; cached later tokens are native observations."
        value["limits"].append("Capital is an explicitly repeated anchor; planet is a fixed new prompt within this campaign.")
    if prompt_panel is not None:
        panels = {
            "language-a": [
                {"id": "arithmeticchat", "text": "<|im_start|>user\nWhat is 7 times 8?<|im_end|>\n<|im_start|>assistant\n"},
                {"id": "translation", "text": "English: cat\nFrench:"},
                {"id": "code", "text": "def square(x):\n    return"},
            ],
            "language-b": [
                {"id": "context", "text": "Alice has a red ball. Bob has a blue ball.\nBob's ball is"},
                {"id": "sequence", "text": "Continue the sequence: 2, 4, 6, 8,"},
                {"id": "antonymchat", "text": "<|im_start|>user\nWhat is the opposite of ancient?<|im_end|>\n<|im_start|>assistant\n"},
            ],
        }
        value.update({"prompt_panel": prompt_panel, "prompts": panels[prompt_panel],
            "arms": {"native": 1., "sham": 1., "zero": 0., "half": .5},
            "prompt_input_authority": "fixed raw literals, including explicit ChatML where shown; add_special_tokens=False, no implicit chat template",
            "eos_handling": "fixed budget: three cached successors after the first token, even after EOS; all token IDs/text retained; not EOS-terminated generation"})
        value["limits"][-1] = (
            "All six literal prompts in language-a/b are declared before either panel runs; "
            "fresh assay contexts, no claim of training-data holdout or automatic semantic-law discovery.")
    return value


def stage(run_dir, *, full_readout=False, prompt_panel=None):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=False)
    config = plan(full_readout=full_readout, prompt_panel=prompt_panel)
    save(run_dir / "plan.json", config)
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    preflight = run_dir / "preflight.v1.json"
    extra_roots = ["qwen_readout_law.py"] if full_readout else []
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": ["qwen-mlp-result", config["model_path"]],
        "hashed_roots": ["qwen_mlp_forecast_campaign.py", "qwen_mlp_equations.py", "campaign_support.py",
                         "plan.json", "preflight.v1.json", *extra_roots]})
    worker = here / "qwen_mlp_forecast_campaign.py"
    extras = [(here / "qwen_mlp_equations.py", "qwen_mlp_equations.py"),
              (here / "discovery_upgrades.py", "campaign_support.py"),
              (run_dir / "plan.json", "plan.json"), (preflight, preflight.name)]
    if full_readout:
        extras.append((here / "qwen_readout_law.py", "qwen_readout_law.py"))
    branches = len(config["prompts"]) * len(config["arms"])
    physical = len(config["prompts"]) + branches * (2 + config["continuation_tokens"])
    scheduling = one_lease_scheduling_contract(logical_operations=physical, logical_branches=branches,
        logical_checkpoint_replays=branches, max_local_work_items=64,
        checkpoint_policy="one resident CPU FP32 Qwen; genuine prefixes then one global suffix seal; native own-KV futures",
        extra={"max_native_full_forwards": branches * (1 + config["continuation_tokens"]),
               "max_prefix_calls": len(config["prompts"]), "max_suffix_records": branches})
    request = {"schema": "mdb-qwen-mlp-forecast-request-v1", "task_family": "recorder",
        "model": config["model"], "needs": {"cpu": True}, "backend": "hf", "device": "cpu", "dtype": "float32",
        "batch_size": 1, "seq_len": 32,
        "backend_options": {"physical_batch_size": 1, "full_prefill_logits": True},
        "execution_environment": intent(), "scheduling_contract": scheduling,
        "config": config,
        "config_bytes_sha256": sha(raw),
        "source_hashes": {name: sha(path.read_bytes()) for path, name in [(worker, worker.name), *extras]},
        "resource_basis": ("intent-first HF CPU FP32 recorder planning uses mrun's measured capture multiplier; "
                           "sequential prefix captures released per prompt"
                           + ("; retain one immutable 544538624-byte FP32 full-head copy, plus <=4096-row FP64 chunks; "
                              "prior measured peak 6176.2MB plus this copy remains below the 7825.7MB recorder plan"
                              if full_readout else "")),
        "selection_policy": "fixed plan prompts and arms; vocabulary contract chosen before prefix collection; retain counterexamples"}
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    command = worker_command(worker.name)
    # Public wheel pins are explicit. The configured ext4 cache contains the
    # qualified runtime; allow uv to resolve a missing wheel instead of assuming it.
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, model=config["model"],
        experiment="mdb-qwen-full-readout-forecast" if full_readout else "mdb-qwen-final-mlp-forecast",
        task="recorder", payload=run_dir / "payload",
        prefer_host="beast", gpu=False, needs=request["needs"], timeout_s=300,
        retry_on_kill=False, preflight=True,
        note=(f"CPU FP32 eager B1 Qwen0.5B: {len(config['prompts'])} aborted prefixes, {branches} suffix captures, "
              f"{physical} total physical attempts. One global seal before all suffixes. "
              + ("Full-vocabulary first-token affine-dose forecasts; " if full_readout else "Fixed candidate-logit forecasts; ")
              + "later own-KV tokens measured only. Shared suffix weights retained once; stop at panel/archive; no retries."))
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--collect")
    parser.add_argument("--full-readout", action="store_true")
    parser.add_argument("--prompt-panel", choices=("language-a", "language-b"))
    args = parser.parse_args()
    if args.collect:
        helper = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))
        result = helper["collect"](args.run_dir, args.collect, max_compressed_bytes=128 << 20,
                                   max_expanded_bytes=512 << 20)
    else:
        result = stage(args.run_dir.resolve(), full_readout=args.full_readout, prompt_panel=args.prompt_panel)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
