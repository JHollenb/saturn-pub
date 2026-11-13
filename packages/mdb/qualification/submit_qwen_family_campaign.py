"""Freeze, stage and admit finite public-MDB Qwen family campaigns."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

WEIGHTS = "2b226fe4944dc744596b1142a15d6676160e9d06f245d31c1a36633f6c86baa3"
CHECKPOINT_FILE = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
WORKER = "qwen_family_campaign.py"
SOURCES = (WORKER, "qwen_family_math.py", "retrieve_qwen_family_weights.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def contexts():
    def pair(cid, text, p, q, answer_p, answer_q):
        return dict(id=cid, seed=0, prompts=dict(P=text.format(query=p), Q=text.format(query=q)),
                    answer_labels=dict(P=answer_p, Q=answer_q))
    return dict(development=[pair("binding-dev", "A: red\nB: blue\nRead {query}.\nAnswer:", "A", "B", " red", " blue")],
        fresh=[pair("binding-swapped", "A: blue\nB: red\nRead {query}.\nAnswer:", "A", "B", " blue", " red"),
               pair("binding-reworded", "first: green\nsecond: yellow\nRead {query}.\nAnswer:", "first", "second", " green", " yellow"),
               pair("binding-animals-reversed", "A: cat\nB: dog\nRead {query}.\nAnswer:", "B", "A", " dog", " cat")])


def specification(family):
    if family == "attention":
        ports = dict(q=dict(shape=[1, 14, 1, 64], dtype="torch.float32", stage="formed positioned final query"),
            k=dict(shape=[1, 2, "T", 64], dtype="torch.float32", stage="complete visible positioned key history"),
            v=dict(shape=[1, 2, "T", 64], dtype="torch.float32", stage="complete visible value history"))
        factors = [dict(id="route", ports=["q", "k"]), dict(id="value", ports=["v"])]
        nodes = [dict(id="result", op="sdpa", inputs=["q", "k", "v"])]
    else:
        ports = {name: dict(shape=[1, 1, 4864], dtype="torch.float32", stage="formed final-token raw MLP " + name) for name in ("G", "U")}
        factors = [dict(id="gate", ports=["G"]), dict(id="value", ports=["U"])]
        nodes = [dict(id="activated", op="silu", inputs=["G"]), dict(id="result", op="multiply", inputs=["activated", "U"])]
    return dict(schema="mdb-computation-family-spec-v2", family_id="qwen2.5-0.5b-instruct.layer12." + family + ".tail.v1",
        model_identity=WEIGHTS, endpoint_adapter="qwen2-eager-original-final-row-v1", ports=ports,
        factors=factors, nodes=nodes, output="result", max_contexts=3,
        source_policy="mathematically predicted natural P/Q sources from raw literal prompt IDs; native source operands measured only after global seal",
        receiver=dict(residual="Q", selected_layer_cache="Q", prefix_rows="Q"),
        clock=dict(token="final-prompt-token", layer=12, edge="formed-attention-head" if family == "attention" else "formed-swiglu-product"),
        continuation=dict(consumer="original Qwen full-vocabulary greedy head and own KV futures", transitions=6, execution_kind="autoregressive", vae_decodes=0,
            state_policy="six emitted greedy choices: prefill then five own-token cached calls; final pending token not consumed; fixed budget through EOS"),
        metric=dict(observable="first-readout logit(answer_Q)-logit(answer_P)", units="logits", direction="minimize",
            reference="predeclared single-token answer labels; unnormalized signed contrast", scope="one layer12 final-row edit followed by full native suffix"))


def wheel_paths(workspace):
    parent = workspace / "artifacts/mdb-parallel-relation-family-20261107/campaign-01/payload"
    return [workspace / "artifacts/mdb-qwen-families-20261107/runtime-v1/mdb_runtime-0.1.0-py3-none-any.whl",
            workspace / "artifacts/mdb-qwen-families-20261107/runtime-v1/saturn_pub-0.4.1-py3-none-any.whl", parent / "mrun_pub-0.1.2-py3-none-any.whl"]


def installed_command(workspace):
    cmd = ["rtk", "proxy", "uv", "run", "--no-project", "--offline"]
    for path in wheel_paths(workspace)[:2]:
        cmd += ["--with", str(path)]
    return cmd + ["python", "-I", "-B"]


def freeze(workspace, base):
    design = dict(schema="mdb-qwen-family-design-v1", inputs=contexts(), families=["attention", "mlp"], layer_zero_based=12,
        model_weights_sha256=WEIGHTS, profile="CPU FP32 eager Qwen2.5-0.5B-Instruct; default RoPE; B1 no padding",
        numerical_forecast="Raw prompt IDs -> original tensor equations -> full logits -> six own greedy choices / own KV; all forecasts seal before native targets in each family lease",
        effect_prediction="Choose the better single factor only from development; freeze both family rules and fresh request bytes before launching either fresh family",
        receiver="Only final prompt token's formed head/product changes; selected-layer original Q cache retained; same-layer MLP recomputed for attention edits; all upper-layer writes recomputed",
        controls="Natural Q, natural P, QQ no-op baseline, two individual factors, PP component hybrid. Component hybrid is not natural P.",
        limits=["Six emitted choices consume prompt plus five generated tokens; sixth token remains pending", "Frozen ranking can fail; preserve signed margins and all arrays", "Generic graph/ports supplied; no automatic semantic ontology or computational shortcut", "No training-data holdout claim; no sampling or default EOS stop"])
    if (base / "design-v1.json").exists():
        if read(base / "design-v1.json") != design:
            raise ValueError("frozen input/scientific design differs")
    else:
        save(base / "design-v1.json", design)
    for family in design["families"]:
        root = base / family
        save(root / "specification-v2.json", specification(family))
        subprocess.run([*installed_command(workspace), "-m", "mdb", "family", "compile", str(root / "specification-v2.json"), "--out", str(root / "program-v2.json")], cwd="/tmp", check=True)
    return dict(status="frozen", design_sha256=sha(base / "design-v1.json"), fresh_contexts=3)


def command(request):
    environment = request["execution_environment"]
    cmd = ["/usr/bin/env", "PYTHONUNBUFFERED=1", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=", "CUDA_VISIBLE_DEVICES=",
        "UV_LINK_MODE=" + environment["uv_link_mode"], "UV_CACHE_DIR=" + environment["uv_cache_dir"],
        "TORCH_HOME=" + environment["torch_home"], "LLM_MODELS_EXTRA_ROOT=" + environment["extra_model_roots"],
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "TOKENIZERS_PARALLELISM=false", "SATURN_RESIDENT_WORKER=1",
        "SATURN_RUN_CONFIG=saturn_run_config.json", "SATURN_DEBUG_REQUEST=saturn_debug_request.json", "TMPDIR=/mnt/hdd8tb/mdb/tmp",
        "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4",
        "uv", "run", "--no-project", "--offline", "--python", environment["python"]]
    for requirement in [*("./" + name for name in request["runtime_wheels"]), *environment["requirements"]]:
        cmd += ["--with", requirement]
    return cmd + ["python", "-B", "-u", "-m", "mdb.worker_bootstrap", WORKER]


def stage(workspace, base, family, phase, run):
    from mdb.job_support import one_lease_scheduling_contract

    run.mkdir(parents=True, exist_ok=False)
    root = base / family
    design = read(base / "design-v1.json")
    config = dict(schema="mdb-qwen-primitive-family-campaign-v1", family=family, phase=phase,
        model_path="/mnt/big/llm-models/Qwen2.5-0.5B-Instruct", weights_sha256=WEIGHTS, checkpoint_file_sha256=CHECKPOINT_FILE,
        compiled_protocol=read(root / "program-v2.json"), contexts=design["inputs"][phase],
        tokens=6, max_prompt_tokens=32, layer=12, cpu_threads=4, dtype="float32", device="cpu",
        output_scratch="/mnt/hdd8tb/mdb/research", max_wall_seconds=840,
        scientific_gates=[], retry=False, nested_mrun=False,
        design_sha256=sha(base / "design-v1.json"), effect_rule=read(root / "effect-rule-v1.json") if phase == "fresh" else None)
    qual = workspace / "mdb/qualification"
    save(run / "plan.json", config)
    save(run / "preflight.v1.json", dict(schema_version="mrun-preflight-config-v1",
        path_token_allowlist=[config["model_path"], config["output_scratch"]],
        hashed_roots=[*SOURCES, "preflight.v1.json", "plan.json"]))
    environment = dict(uv_link_mode="symlink", uv_cache_dir="/mnt/hdd8tb/mdb/uv-cache", torch_home="/mnt/hdd8tb/mdb/torch",
        extra_model_roots="", python="3.14.6", requirements=["torch==2.13.0", "transformers==5.14.1", "accelerate==1.14.0", "numpy==2.5.4", "safetensors==0.8.0", "psutil==7.2.2", "pyyaml==6.0.3"])
    sources = {name: qual / name for name in SOURCES}
    sources.update({name: run / name for name in ("preflight.v1.json", "plan.json")})
    wheels = {path.name: path for path in wheel_paths(workspace)}
    n = len(config["contexts"])
    geometry = config["compiled_protocol"]["execution_geometry"]
    forwards = geometry["per_context_per_lane"]["physical_forwards"] * n
    request = dict(schema="mdb-qwen-family-request-v1", config=config, execution_environment=environment,
        source_hashes={name: sha(path) for name, path in sources.items()}, runtime_wheels={name: sha(path) for name, path in wheels.items()},
        model="qwen2.5-0.5b-instruct", backend="hf", device="cpu", dtype="float32", needs=dict(cpu=True),
        task_family="recorder", batch_size=1, seq_len=32, backend_options=dict(physical_batch_size=1, full_prefill_logits=False),
        launcher_sha256=sha(__file__), scheduling_contract=one_lease_scheduling_contract(model_loads=1,
            logical_branches=6*n, logical_operations=geometry["aggregate_work_items_per_context_both_lanes"]*n, logical_checkpoint_replays=0, max_local_work_items=256,
            checkpoint_policy="One resident original Qwen; functional own-token/KV forecasts, global family seal, native independent own-token/KV targets; no retries or nested jobs",
            extra=dict(physical_forward_census=dict(functional=forwards, native=forwards), eos_policy="fixed six-choice recurrence")))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(run / "request.json", request)
    shutil.copyfile(__file__, run / "frozen-launcher.py")
    payload = run / "payload"
    payload.mkdir()
    for name, path in {**sources, **wheels}.items():
        shutil.copyfile(path, payload / name)
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    (payload / "saturn_run_config.json").write_bytes(raw)
    save(payload / "saturn_debug_request.json", request)
    packages = {}
    for name, path in wheels.items():
        package = "mdb" if name.startswith("mdb_") else "saturn_pub" if name.startswith("saturn_") else "mrun"
        with zipfile.ZipFile(path) as archive:
            packages[package] = {member[len(package)+1:]: hashlib.sha256(archive.read(member)).hexdigest()
                for member in archive.namelist() if member.startswith(package + "/") and not member.endswith("/")}
    save(payload / "payload-seal.json", dict(packages=packages, wheels=request["runtime_wheels"], files=request["source_hashes"],
        config_bytes_sha256=hashlib.sha256(raw).hexdigest(), worker=WORKER, worker_sha256=request["source_hashes"][WORKER]))
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"]))
    return dict(request_sha256=request["request_sha256"], family=family, phase=phase, contexts=n, native_calls=36*n)


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    request = read(run / "request.json")
    if (run / "submission.json").exists():
        raise ValueError("finite request already submitted")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("staged member differs")
    if sha(__file__) != request["launcher_sha256"]:
        raise ValueError("frozen launcher changed")
    review = read(run / "finite-run-review.json")
    if review["request_sha256"] != request["request_sha256"] or review["blocking_issues"]:
        raise ValueError("exact finite review missing")
    args = command(request)
    save(run / "admission-command.json", dict(command=args))
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    job = guarded_mrun_launch(args, request=request, model=request["model"], task="recorder",
        experiment="mdb-qwen-family-" + request["config"]["family"] + "-" + request["config"]["phase"],
        payload=run / "payload", prefer_host="beast", gpu=False, needs=request["needs"],
        timeout_s=900, retry_on_kill=False, preflight=True,
        note="Finite CPU FP32 eager Qwen0.5B family: six roles/context, six emitted own-token choices (prefill+five cached calls), full logits and every layer KV retained. One family-wide numeric seal before native. Original weights external; no retries/children.")
    result = dict(job_id=str(job), request_sha256=request["request_sha256"])
    save(run / "submission.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "stage", "submit"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--family", choices=("attention", "mlp"))
    parser.add_argument("--phase", choices=("development", "fresh"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    workspace, base = args.workspace.resolve(), args.base.resolve()
    value = freeze(workspace, base) if args.action == "freeze" else stage(workspace, base, args.family, args.phase, args.run.resolve()) if args.action == "stage" else submit(args.run.resolve())
    print(json.dumps(value, sort_keys=True))
