"""Installed-worker qualification of native Qwen gain/weight-edit consumers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import secrets
import shutil
import time
import zipfile
from pathlib import Path

from protocol import digest, save

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def worker():
    import numpy as np
    import torch
    from qwen_profile import context, gains, load, save_readout

    from mdb_lanes.wsp import WeightStudy, verify_weight_study
    from mdb_lanes.wsp_weights import ParameterPatch, parameter_inventory

    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    root = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "native-result"
    root.mkdir(parents=True, exist_ok=False)
    save(root / "request.json", request)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    print("NATIVE_PROGRESS load", flush=True)
    net, tokenizer = load(config, root)
    ctx = context(net, tokenizer, config["seed"])
    model, splits = ctx["model"], ctx["splits"]
    identity = parameter_inventory(model)["identity"]
    site, gain = "net.model.layers.6.mlp.down_proj", .5
    module = dict(model.named_modules())[site]
    if module.bias is not None:
        raise ValueError("bias-free down projection required")
    original = module.weight.detach().clone()
    patch = ParameterPatch(model, {site+".weight": original*gain}, model_identity=identity)
    # The declared equality is real arithmetic. Native FP32 remainders are observations.
    save(root / "inputs.json", {"examples": model.examples, "preservation": model.preservation_prompts,
        "input_ids": model.input_ids.tolist(), "mask": model.attention_mask.tolist(),
        "target_ids": model.target_ids.tolist(), "input_identity": model.input_identity})
    np.savez_compressed(root / "anchor.npz", logp=model.anchor_logp.cpu().numpy())
    counts = {"target_network_forwards": 0}
    def count(module, args):
        counts["target_network_forwards"] += 1
    counter = net.register_forward_pre_hook(count)
    observed = {}
    with WeightStudy(root / "study", model=model,
            binding={"model": identity, "checkpoint": config["checkpoint_sha256"],
                "execution": "CPU FP32 eager, fixed B16 full prefixes, use_cache=False",
                "consumer": "native correct-token log probability and original-distribution preservation KL",
                "horizon": "one-token readout; no generation or loop-repair claim"},
            input_identity=ctx["input_identity"], source_identity={"request_sha256": request["request_sha256"]},
            budgets={"max_native_calls": 4, "max_bytes": 128 << 20, "max_wall_ms": 600000}) as study:
        for name, role, candidate in (("native", "native", None), ("virtual", "edited", None),
                                      ("compiled", "edited", patch), ("restored", "restoration", patch)):
            study.add_arm(name, role=role, patch=candidate, state_policy="restart",
                predictor=lambda context, name=name: {"predictions": {"arm": name, "site": site, "gain": gain,
                    "expectation": "virtual and compiled agree in real arithmetic; FP32 remainder measured; restoring original return reproduces native"}})
        study.seal()
        for name in ("native", "virtual", "compiled", "restored"):
            def execute(arm, name=name):
                hook = None
                if name == "restored":
                    hook = module.register_forward_hook(lambda layer, args, out: torch.nn.functional.linear(args[0], original))
                try:
                    if name == "virtual":
                        with gains(model, [{"site": site, "gain": gain}]):
                            result = model.comparison_score(splits, ("discovery", "preservation"))
                    else:
                        result = model.comparison_score(splits, ("discovery", "preservation"))
                    readout = save_readout(model, root / (name + "-readout.npz"))
                    observed[name] = {**result, "readout": readout}
                    return {"observations": observed[name], "work": {"physical_network_forwards": 1, "batch_size": 16},
                        "state": {"policy": "restart", "use_cache": False, "anchor": model.input_identity}}
                finally:
                    if hook is not None:
                        hook.remove()
            study.run(name, execute)
            receipt = study.tx.observation(name)
            if receipt["instrument_status"] != "observed":
                observed[name] = {"instrument_status": receipt["instrument_status"], "result": receipt["result"]}
            print("NATIVE_PROGRESS arm " + name, flush=True)
    counter.remove()
    comparisons = {}
    for a, b in (("native", "restored"), ("virtual", "compiled")):
        if any(observed.get(name, {}).get("instrument_status") == "evidence-incomplete" for name in (a,b)):
            comparisons[a+"_vs_"+b] = {"instrument_status": "evidence-incomplete"}
            continue
        with np.load(root / (a+"-readout.npz"), allow_pickle=False) as left, np.load(root / (b+"-readout.npz"), allow_pickle=False) as right:
            delta = left["logp"].astype(np.float64)-right["logp"].astype(np.float64)
            comparisons[a+"_vs_"+b] = {"maximum_logp_difference": float(np.max(np.abs(delta))),
                "different_logp_elements": int(np.count_nonzero(delta)), "compared_elements": int(delta.size),
                "winner_ids_equal": bool(np.array_equal(left["winner_ids"], right["winner_ids"]))}
    summary = {"schema": "mdb-native-workflow-smoke-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "outcomes": observed, "comparisons": comparisons,
        "work": {**counts, "original_anchor_forwards_before_study": 1},
        "parameter_restoration": parameter_inventory(model)["identity"] == identity,
        "instrument_status": "observed" if len(observed) == 4 and all("readout" in r for r in observed.values()) else "evidence-incomplete",
        "elapsed_seconds": time.perf_counter()-started, "verification": verify_weight_study(root / "study"),
        "mechanics_status": "valid", "trend_status": "uninterpreted", "terminal_status": "not-evaluated",
        "scope": "pretrained numerical/consumer mechanics; supplied site and dose, not an editor-search result"}
    save(root / "summary.json", summary)
    save(root / "files.json", {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()})
    from flux_family_custody import emit
    emit(root, summary, limits={"max_archive_bytes": 64 << 20, "max_expanded_bytes": 192 << 20,
                                "max_member_bytes": 64 << 20}, compression_level=1)


def stage(base):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    base.mkdir(parents=True, exist_ok=False)
    payload = base / "payload"
    payload.mkdir()
    config = {"schema": "mdb-native-workflow-smoke-plan-v1", "model_path": "/mnt/big/llm-models/Qwen2.5-0.5B-Instruct",
        "checkpoint_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "device": "cpu", "dtype": "float32", "seed": secrets.randbits(31),
        "output_scratch": "/mnt/hdd8tb/mdb/workflow-native", "scientific_gates": [], "retry": False}
    files = {"worker.py": HERE / "native_smoke.py", "protocol.py": HERE / "protocol.py", "qwen_profile.py": HERE / "qwen_profile.py",
        "qwen_family_campaign.py": HERE.parent / "qwen_family_campaign.py",
        "retrieve_qwen_family_weights.py": HERE.parent / "retrieve_qwen_family_weights.py",
        "flux_family_custody.py": HERE.parent / "flux_family_custody.py"}
    save(payload / "preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["model_path"], config["output_scratch"]], "hashed_roots": [*files, "preflight.v1.json"]})
    files["preflight.v1.json"] = payload / "preflight.v1.json"
    helper = runpy.run_path(str(HERE.parent / "wsp_case_study/launch.py"))
    wheel_map = {p.name:p for p in helper["wheels"]()}
    request = {"schema": "mdb-native-workflow-smoke-request-v1", "config": config,
        "execution_environment": {"python": "3.14.6", "uv_link_mode": "symlink", "uv_cache_dir": "/mnt/hdd8tb/mdb/uv-cache",
            "torch_home": "/mnt/hdd8tb/mdb/torch", "extra_model_roots": "/mnt/big/llm-models",
            "requirements": ["torch==2.13.0", "numpy==2.5.4", "pyyaml==6.0.3", "psutil==7.2.2", "safetensors==0.8.0", "transformers==5.14.1", "accelerate==1.14.0"]},
        "source_hashes": {n:sha(p) for n,p in files.items()}, "runtime_wheels": {n:sha(p) for n,p in wheel_map.items()},
        "model": "qwen2.5-0.5b-instruct", "task_family": "recorder", "backend": "hf", "device": "cpu", "dtype": "float32",
        "needs": {"cpu": True}, "batch_size": 16, "seq_len": 128,
        "backend_options": {"physical_batch_size": 16, "full_prefill_logits": False},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_branches=4, logical_operations=8,
            logical_checkpoint_replays=0, max_local_work_items=64)}
    request["request_sha256"] = digest(request)
    save(base / "request.json", request)
    for name,path in {**files, **wheel_map}.items():
        if path.resolve() != (payload/name).resolve():
            shutil.copyfile(path, payload/name)
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    (payload/"saturn_run_config.json").write_bytes(raw)
    save(payload/"saturn_debug_request.json",request)
    packages = {}
    for name,path in wheel_map.items():
        package = "mdb" if name.startswith("mdb_") else "saturn_pub" if name.startswith("saturn_") else "mrun"
        with zipfile.ZipFile(path) as archive:
            packages[package] = {member[len(package)+1:]:hashlib.sha256(archive.read(member)).hexdigest()
                for member in archive.namelist() if member.startswith(package+"/") and not member.endswith("/")}
    save(payload/"payload-seal.json", {"packages":packages,"wheels":request["runtime_wheels"],"files":request["source_hashes"],
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"worker":"worker.py","worker_sha256":request["source_hashes"]["worker.py"]})
    save(base/"command.json",helper["command"](request))
    return {"run":str(base),"request_sha256":request["request_sha256"]}


def submit(base):
    from mdb_lanes.job_support import guarded_mrun_launch
    if (base/"submission.json").exists():
        raise ValueError("finite smoke already submitted")
    request=json.loads((base/"request.json").read_text())
    for n,h in {**request["source_hashes"],**request["runtime_wheels"]}.items():
        if sha(base/"payload"/n)!=h:
            raise ValueError("staged source differs")
    os.environ.setdefault("MRUN_URL","http://192.168.68.68:9025")
    job=guarded_mrun_launch(json.loads((base/"command.json").read_text()),request=request,
        experiment="mdb-workflow-native-qwen-smoke",task="recorder",model=request["model"],payload=base/"payload",
        prefer_host="beast",gpu=False,needs={"cpu":True},timeout_s=840,retry_on_kill=False,preflight=True,
        note="Native workflow mechanics Qwen0.5B CPU FP32 eager B16: one original anchor plus4 sealed native/virtual-gain/compiled/restored full-prefix reads. No generation/training/editor trials/retries/children. Full original archive retrieval and full-vocab readout custody; stop after four arms.")
    value={"job_id":str(job),"request_sha256":request["request_sha256"]}
    save(base/"submission.json",value)
    return value


def collect(base):
    job=json.loads((base/"submission.json").read_text())["job_id"]
    result=runpy.run_path(str(HERE.parent/"flux_family_custody.py"))["collect"](base,job,
        limits={"max_archive_bytes":64<<20,"max_expanded_bytes":192<<20,"max_member_bytes":64<<20})
    if result.get("collection_status")=="not-terminal":
        return {k:result[k] for k in ("collection_status","job_id","state")}
    return result


if __name__=="__main__":
    if os.environ.get("SATURN_RUN_CONFIG") and os.environ.get("MRUN_JOB_ID"):
        worker()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("action",choices=("stage","submit","collect"))
        parser.add_argument("base",type=Path)
        args=parser.parse_args()
        print(json.dumps({"stage":stage,"submit":submit,"collect":collect}[args.action](args.base.resolve()),sort_keys=True))
