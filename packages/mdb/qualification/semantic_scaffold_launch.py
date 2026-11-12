"""Stage and submit a finite source-aware task expansion; no implicit retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def environment():
    os.environ.setdefault("MDB_WORKER_UV_CACHE_DIR", "/mnt/big/llm-models/.uv-cache")
    os.environ.setdefault("MDB_WORKER_TORCH_HOME", "/mnt/big/llm-models/.torch-cache")
    os.environ.setdefault("MDB_WORKER_MODEL_ROOTS", "/mnt/big/llm-models")
    from mdb.execution_environment import intent
    return intent()


def stage(args):
    from mdb import parameter_artifacts
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    from mdb.semantic_scaffold import calibration_plan, localization_plan, plan
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = plan()
    if args.mode == "frozen-pair":
        from mdb.frozen_value_pair import cases as frozen_cases
        cases = frozen_cases()
    if args.mode == "resolver-execution":
        from resolver_execution_plan import plan as resolver_plan
        cases, evaluator = resolver_plan()
        save(root / "evaluation-only.json", evaluator)
    if args.mode in {"attention", "coalition"}:
        cases = cases[:1]
    if args.mode.startswith("closure"):
        if args.mode in {"closure-semantics", "closure-timing", "closure-support"}:
            cases = [case for case in cases if case["lane"] == "semantics"][:1]
        else:
            cases = [case for case in cases if case["lane"] == "coding"]
        if args.mode == "closure-smoke":
            cases = cases[:1]
        elif args.mode == "closure":
            cases = cases[1:]
    if args.mode == "calibration":
        if args.frozen_routes is None:
            raise ValueError("calibration requires an explicit frozen-route file")
        cases = calibration_plan()
    if args.mode == "depth":
        if args.frozen_routes is None:
            raise ValueError("depth replay requires an explicit frozen-route file")
        cases = localization_plan()
    config = {"model": "qwen2.5-1.5b", "model_path": "/mnt/big/llm-models/Qwen2.5-1.5B",
        "expected_weight_file_sha256": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
        "output_root": "/mnt/big/mdb/qualification/semantic-scaffold-20261111", "seed": 20261111,
        "cases": cases[:1] if args.mode == "smoke" else cases, "mode": args.mode}
    config["depth_bands"] = args.mode == "depth"
    if args.mode == "attention":
        config["output_root"] = "/mnt/big/mdb/qualification/attention-allocation-20261111"
    if args.mode == "coalition":
        config["output_root"] = "/mnt/big/mdb/qualification/value-coalition-20261112"
    if args.mode == "frozen-pair":
        config["output_root"] = "/mnt/big/mdb/qualification/frozen-value-pair-20261112"
    if args.mode == "resolver-execution":
        config["output_root"] = "/mnt/big/mdb/qualification/resolver-execution-20261112"
    if args.mode.startswith("closure"):
        config["output_root"] = "/mnt/big/mdb/qualification/live-symbol-closure-20261111"
        config["timing_only"] = args.mode == "closure-timing"
        config["support_only"] = args.mode == "closure-support"
    if args.frozen_routes:
        config["frozen_routes"] = json.loads(args.frozen_routes.read_text())
        config["frozen_routes_source_sha256"] = sha(args.frozen_routes)
    save(root / "plan.json", config)
    worker = Path(__file__).with_name("live_symbol_closure_worker.py" if args.mode.startswith("closure")
        else "semantic_scaffold_worker.py")
    if args.mode == "attention":
        worker = Path(__file__).with_name("attention_allocation_worker.py")
    if args.mode == "coalition":
        worker = Path(__file__).with_name("value_coalition_worker.py")
    if args.mode == "frozen-pair":
        worker = Path(__file__).with_name("frozen_value_pair_worker.py")
    if args.mode == "resolver-execution":
        worker = Path(__file__).with_name("resolver_execution_worker.py")
    retrieval = Path(parameter_artifacts.__file__)
    helper, scaffold_helper = args.parameter_helper.resolve(), args.scaffold_helper.resolve()
    preflight = {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, "parameter_source.py", "scaffold_io.py", "retrieve_weights.py", "preflight.v1.json"],
        "path_token_allowlist": [config["model_path"], config["output_root"]]}
    save(root / "preflight.v1.json", preflight)
    request = {"schema": "mdb-semantic-scaffold-request-v1", "model": config["model"],
        "task_family": "forward", "backend": "hf", "device": "cuda", "dtype": "bfloat16",
        "batch_size": 1, "seq_len": 128, "backend_options": {"physical_batch_size": 1, "full_prefill_logits": False},
        "needs": {"cuda": True}, "execution_environment": environment(), "config": config,
        "source_hashes": {worker.name: sha(worker), "parameter_source.py": sha(helper),
                          "scaffold_io.py": sha(scaffold_helper), "retrieve_weights.py": sha(retrieval)},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_branches=96,
            logical_operations=240, logical_checkpoint_replays=64, max_local_work_items=256,
            checkpoint_policy="One BF16 resident; native MDB parents and source-value Acts; scalar physical batch; <=8 task pairs, 3-token own suffixes, no nested leases or retries"),
        "workload_geometry": {"model_loads": 1, "physical_batch": 1, "max_prompt_tokens": 128,
            "explicit_native_steps": 256, "source_discovery_transactions": 4,
            "per_discovery_layer_token_budget": 2000, "output_archive_max_bytes": 512 << 20,
            "output_expanded_max_bytes": 1024 << 20},
        "resource_basis": "Intent-first; previous resident native-eager MDB panel measured 3458MB VRAM/4040.3MB RSS; no hand reservation.",
        "timeout_s": 840,
        "queue_note": "Source-aware semantics/copy/recall/code exploration; one Qwen1.5B BF16/eager resident, scalar batch; 8 pairs, <=256 explicit native transitions plus 4 bounded source-restricted discovery transactions; 3-token branch-owned futures; one report or 840s. No retries."}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if args.mode == "attention":
        request["schema"] = "mdb-attention-allocation-request-v1"
        request["worker"] = worker.name
        request["workload_geometry"]["explicit_native_steps"] = 4000
        request["workload_geometry"]["source_discovery_transactions"] = 0
        request["scheduling_contract"] = one_lease_scheduling_contract(model_loads=1,
            logical_branches=12, logical_operations=128, logical_checkpoint_replays=8,
            max_local_work_items=256, checkpoint_policy="One resident; two existing eager lanes, six futures per lane including passive/sham controls; 3 own tokens each, full Q/K/V/A and head returns; <=4000 native substeps, no discovery/retries")
        request["queue_note"] = "Attention allocation in the freezing/boiling temperature circuit: one Qwen1.5B BF16 CUDA resident; Bench and public Session run sequentially, six futures each (passive/sham/native/late/all), 3 own tokens, 28 layers/12 heads. Record Q/K/V/attention/head returns unchanged; stop after report or 840s."
        request["request_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True).encode()).hexdigest()
    if args.mode.startswith("closure"):
        request["schema"] = "mdb-live-symbol-closure-request-v1"
        request["worker"] = worker.name
        request["scheduling_contract"] = one_lease_scheduling_contract(model_loads=1,
            logical_branches=8, logical_operations=240, logical_checkpoint_replays=0,
            max_local_work_items=256, checkpoint_policy="One resident; one coding contrast, six arms; <=240 macro transactions with <=7000 declared native layer/operation substeps; source-writer and prompt ingestion through public typed Sessions, canonical Project symbols and 3-token own futures; no retries")
        request["workload_geometry"]["explicit_native_steps"] = 7000
        request["queue_note"] = "Close live symbol/source-writer gaps: one Qwen1.5B BF16 CUDA resident, one paired source contrast x6 arms, <=240 macro transactions and <=7000 native layer/operation substeps, canonical MDB Project, own 3-token continuation. Stop after report or 840s; no retries."
        request["request_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True).encode()).hexdigest()
    additional = []
    if args.mode == "coalition":
        support = Path(__file__).with_name("attention_allocation_worker.py")
        additional.append((support, "attention_support.py"))
        request["source_hashes"]["attention_support.py"] = sha(support)
        request["schema"] = "mdb-value-coalition-request-v1"
        request["worker"] = worker.name
        request["workload_geometry"].update(explicit_native_steps=6500, source_discovery_transactions=0,
            branches=56, generated_tokens_per_branch=3, first_receiver_attention_captures=56)
        request["scheduling_contract"] = one_lease_scheduling_contract(model_loads=1,
            logical_branches=56, logical_operations=240, logical_checkpoint_replays=56,
            max_local_work_items=256, checkpoint_policy="One resident, scalar public-native Session branches from two sealed prompt parents; <=240 macro token/Act transactions (readout+commit is one token transaction), 56 checkpoint replays, 56 three-token futures; exact cell Acts, first-receiver attention and canonical symbols; <=6500 native substeps; no retries")
        request["queue_note"] = "Decompose temperature source-V coalition: one Qwen1.5B BF16/eager CUDA resident; 56 scalar same-parent branches, 18 singleton and 18 complement cells, 9 paired layers, KV groups, sham/wrong-source/reciprocal/dose controls; 3 own tokens each, <=6500 native substeps. Stop at report or 840s; no retries."
        request["request_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True).encode()).hexdigest()
        preflight["hashed_roots"].append("attention_support.py")
        # The preflight source is staged only after all mode-specific roots exist.
        (root / "preflight.v1.json").write_text(json.dumps(preflight, indent=2, sort_keys=True))
    if args.mode in {"frozen-pair", "resolver-execution"}:
        for source_name, target in (("attention_allocation_worker.py", "attention_support.py"),
                                    ("value_coalition_worker.py", "coalition_support.py")):
            support = Path(__file__).with_name(source_name)
            additional.append((support, target))
            request["source_hashes"][target] = sha(support)
            preflight["hashed_roots"].append(target)
        request["schema"] = "mdb-frozen-value-pair-request-v1"
        request["worker"] = worker.name
        request["workload_geometry"].update(explicit_native_steps=8500, source_discovery_transactions=0,
            branches=50, contexts=5, generated_tokens_per_branch=3, first_receiver_capture_layers=[20, 21, 26])
        request["scheduling_contract"] = one_lease_scheduling_contract(model_loads=1,
            logical_branches=50, logical_operations=240, logical_checkpoint_replays=50,
            max_local_work_items=256, checkpoint_policy="One resident, scalar public native Sessions; 5 prospective contexts x10 arms, 3 own tokens each, <=240 macro token/Act transactions plus 50 checkpoint replays and <=8500 native substeps; fixed L20 pair, matched controls/full-band comparator; no retuning/retries")
        request["queue_note"] = "Frozen L20 value pair: Qwen1.5B BF16/eager CUDA resident; 5 wording/position contexts x10 same-parent arms, singleton/pair/sham/wrong-source/wrong-target/reciprocal/full-band; 3 own tokens and L20/21/26 captures; <=8500 native substeps. Stop at report or 840s, no retries."
        request["request_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True).encode()).hexdigest()
        (root / "preflight.v1.json").write_text(json.dumps(preflight, indent=2, sort_keys=True))
    if args.mode == "resolver-execution":
        request["schema"] = "mdb-live-source-resolver-request-v1"
        request["workload_geometry"].update(branches=45, contexts=5,
            source_discovery_transactions=10, discovery_substeps_per_pass=22)
        request["scheduling_contract"] = one_lease_scheduling_contract(model_loads=1,
            logical_branches=45, logical_operations=240, logical_checkpoint_replays=64,
            max_local_work_items=256, checkpoint_policy="One resident, scalar native Sessions; 10 partial native discovery passes with exact restore; 5 contexts x9 arms, 3 own tokens; <=240 macro token/Act transactions, <=64 replays, <=8500 native substeps; no oracle source rows, retuning or retries")
        request["queue_note"] = "Execute fixed L20/H5 source resolver: Qwen1.5B BF16/eager CUDA, one resident; 5 contexts x9 arms, 10 native partial-pass nominations, exact rewind then attention-addressed pair/full Acts and matched controls; 3 own tokens each, <=8500 substeps. Source annotations withheld. Stop at report or 840s; no retries."
        request["request_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True).encode()).hexdigest()
    save(root / "request.json", request)
    stage_worker(root / "payload", worker=worker, config=config, request=request,
        additional_files=[(helper, "parameter_source.py"), (scaffold_helper, "scaffold_io.py"),
                          (retrieval, "retrieve_weights.py"), (root / "preflight.v1.json", "preflight.v1.json"), *additional])
    print(json.dumps({"root": str(root), "cases": len(config["cases"]), "request_sha256": request["request_sha256"]}))


def submit(root):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    if (root / "submission.json").exists():
        raise ValueError("already submitted; collect the existing job")
    request = json.loads((root / "request.json").read_text())
    environment()
    job = guarded_mrun_launch(worker_command(request.get("worker", "semantic_scaffold_worker.py")), request=request,
        model=request["model"], task="forward", experiment=({"value_coalition_worker.py": "mdb-value-coalition-20261112", "frozen_value_pair_worker.py": "mdb-frozen-value-pair-20261112", "resolver_execution_worker.py": "mdb-live-source-resolver-20261112"}.get(request.get("worker"), "mdb-semantic-scaffold-20261111")),
        payload=root / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        timeout_s=request["timeout_s"], retry_on_kill=False, preflight=True, detach=True,
        note=request["queue_note"])
    save(root / "submission.json", {"job_id": str(job), "request_sha256": request["request_sha256"]})
    print(json.dumps({"job_id": str(job), "root": str(root)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=("smoke", "main", "calibration", "depth", "closure-smoke", "closure", "closure-semantics", "closure-timing", "closure-support", "attention", "coalition", "frozen-pair", "resolver-execution"), default="smoke")
    parser.add_argument("--parameter-helper", type=Path)
    parser.add_argument("--scaffold-helper", type=Path)
    parser.add_argument("--frozen-routes", type=Path)
    args = parser.parse_args()
    if args.action == "stage":
        if args.parameter_helper is None or args.scaffold_helper is None:
            parser.error("stage requires explicit --parameter-helper and --scaffold-helper")
        stage(args)
    else:
        submit(args.root.resolve())


if __name__ == "__main__":
    main()
