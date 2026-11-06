"""Stage one source-bound block0 attention diagnostic, with no local model run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "qwen_block0_attention_microscope.py"
WORKER_SHA256 = "70e52a6624947029b906e0a02f2f36d1a165b321a40c9e42f665224a3710361b"
PARENT_REQUEST = "783e8dc0c0a731c458eed739c5d8b68c932bf622ef5f426d7f484596a8c011dd"
PARENT_JOB = "job-0b6c4340d5e6"
PARENT_MANIFEST = "89dafe939e0143596edf88b42e073f97cc5c945c893b5b8cf9897eff809c7154"
COMPACT_MANIFEST = "14cee5b4030644aff01fdbffdfc45e42eb7f2a0dfd0ae15943449ec7d766afa8"
HELPERS = {
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
}
QUEUE_NOTE = (
    "Qwen block0 drift microscope,B1/512rows:12 original tensors(980MB),one native block hydration,2 functional source blocks+3 fixed-bank SDPA arms sealed before2 native blocks;7 SDPA+2 rotary calls,total. "
    "No full encoder/FLUX/VAE/tokenizer/profiler repeats. CPU4/RAM8GiB/VRAM4GiB,disk2GB,240s estimate,600s worker/900s outer,no retries. Full8GB source archives authenticated once;small raw ports stay near data. Stop atH1."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()


def bind(environment):
    from mdb.execution_environment import intent

    for key, name in {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
                      "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
                      "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}.items():
        os.environ[name] = environment[key]
    if intent() != {key: environment[key] for key in intent()}:
        raise ValueError("Installed execution requirements differ from source cohort")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"


def command(request):
    from mdb.payload import worker_command

    bind(request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH",
                      "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def select_inputs(compact, wrapper):
    inventory = wrapper["files"]
    files, cases = {}, {}

    def admit(name):
        row = inventory[name]
        files[name] = {"sha256": row["sha256"], "bytes": row["bytes"]}

    for case in ("short", "long"):
        cases[case] = {}
        for lane in ("forecast", "native"):
            name = "cases/" + case + "/" + lane + "-qwen.json"
            admit(name)
            if sha(compact / name) != inventory[name]["sha256"]:
                raise ValueError("Authenticated original Qwen sidecar differs: " + name)
            sidecar = read(compact / name)
            cases[case][lane + "_qwen"] = name
            for record in (sidecar["input_ids"], sidecar["attention_mask"], *sidecar["hidden_states"][:2]):
                relative = "cases/" + case + "/" + record["path"]
                admit(relative)
                if files[relative] != {"sha256": record["sha256"], "bytes": record["bytes"]}:
                    raise ValueError("Original array descriptor and wrapper differ")
    return cases, files


def stage(run, parent_run, compact):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, parent_run, compact = map(lambda value: Path(value).resolve(), (run, parent_run, compact))
    parent = read(parent_run / "request.json")
    if identity(parent) != PARENT_REQUEST or parent["request_sha256"] != PARENT_REQUEST:
        raise ValueError("Exact qualified paired source request required")
    if sha(compact / "files.json") != COMPACT_MANIFEST or sha(compact / "source-files.json") != PARENT_MANIFEST:
        raise ValueError("Authenticated compact and original wrappers required")
    wrapper = read(compact / "source-files.json")
    if wrapper["job_id"] != PARENT_JOB or wrapper["request_sha256"] != PARENT_REQUEST:
        raise ValueError("Original wrapper identities differ")
    for name, digest in {WORKER: WORKER_SHA256, **HELPERS}.items():
        if sha(HERE / name) != digest:
            raise ValueError("Qualified source changed: " + name)
    cases, inputs = select_inputs(compact, wrapper)
    manifest = read(parent_run / "qwen-parameter-artifacts.json")
    prefix = runpy.run_path(str(HERE / "flux_functional_qwen_prefix.py"))
    wanted = {"model.embed_tokens.weight", *("model.layers.0." + name for name in prefix["QWEN_LAYER_PARAMETER_SUFFIXES"])}
    selected = [row for row in manifest["artifacts"] if row["source"]["tensor"] in wanted]
    payload_bytes = sum(row["source"]["data_offsets"][1] - row["source"]["data_offsets"][0] for row in selected)
    if len(selected) != 12 or payload_bytes != 979773952:
        raise ValueError("Original twelve-tensor geometry differs")
    run.mkdir(parents=True, exist_ok=False)
    source_pairs = {WORKER: HERE / WORKER, **{name: HERE / name for name in HELPERS}, Path(__file__).name: Path(__file__)}
    for name in ("qwen-config.json", "qwen-parameter-artifacts.json", "retrieve_qwen_weights.py", "qwen-source-binding.json"):
        source = parent_run / "payload" / name
        if sha(source) != parent["source_hashes"][name]:
            raise ValueError("Original recipe/config metadata differs: " + name)
        shutil.copyfile(source, run / name)
        source_pairs[name] = run / name
    shutil.copyfile(compact / "source-files.json", run / "parent-files.json")
    source_pairs["parent-files.json"] = run / "parent-files.json"
    binding = read(run / "qwen-source-binding.json")
    transformer = importlib.metadata.distribution("transformers")
    native_sources = {role: sha(Path(transformer.locate_file(relative))) for role, relative in {
        "qwen3": "transformers/models/qwen3/modeling_qwen3.py",
        "sdpa": "transformers/integrations/sdpa_attention.py",
        "masking": "transformers/masking_utils.py",
    }.items()}
    if native_sources["qwen3"] != parent["config"]["qwen_native_source_sha256"]:
        raise ValueError("Original installed Qwen source differs")
    for source in binding["installed_environment"]["implementation_sources"]:
        if source["relative"] == "transformers/masking_utils.py" and source["sha256"] != native_sources["masking"]:
            raise ValueError("Original mask utility source differs")
    config = {
        "schema": "mdb-qwen-block0-attention-plan-v1", "device": "cuda", "dtype": "bfloat16",
        "input_root": "/mnt/big/mdb/research/" + PARENT_JOB + "/raw-mask-pair-result",
        "input_files": inputs, "cases": cases,
        "source_provenance": {"job_id": PARENT_JOB, "request_sha256": PARENT_REQUEST, "manifest_sha256": PARENT_MANIFEST},
        "qwen_config": "qwen-config.json", "qwen_manifest": "qwen-parameter-artifacts.json",
        "retrieval_script": "retrieve_qwen_weights.py", "parameter_root": parent["config"]["qwen_parameter_root"],
        "native_sources": native_sources, "fixed_bank_case": "long", "shared_prefix_rows": 30,
        "short_valid_rows": 39, "source_reduction_policy": True, "cpu_threads": 4, "profile_sdpa": True,
        "max_wall_seconds": 600, "output_scratch": "/mnt/big/mdb/research",
        "output_name": "qwen-block0-attention-result", "directory_custody": True,
        "transport_limits": {"max_expanded_bytes": 1 << 30, "max_member_bytes": 64 << 20},
    }
    for key in ("qwen_config", "qwen_manifest", "retrieval_script"):
        config[key + "_sha256"] = sha(run / config[key])
    local = dict(config, **{key: str(run / config[key]) for key in ("qwen_config", "qwen_manifest", "retrieval_script")})
    counts = runpy.run_path(str(HERE / WORKER))["validate_config"](local)
    source_archives = {row["source"]["file"]: row["source"]["file_sha256"] for row in manifest["artifacts"]}
    geometry = {"selected_tensors": 12, "selected_payload_bytes": payload_bytes,
                "selected_parameters": {row["source"]["tensor"]: {key: row["source"][key] for key in ("shape", "dtype", "payload_sha256")} for row in selected},
                "complete_checkpoint_archives_authenticated": source_archives,
                "original_input_members": len(inputs), "original_input_member_bytes": sum(row["bytes"] for row in inputs.values()),
                "native_weight_assignment": "Eleven block parameters alias the once-loaded recipe bank; no duplicate native bank.",
                "new_frontier": "Retained original IDs/masks plus original parameters; historical H0/H1 are known replay controls.",
                "estimated_peak_ram_mb": "3000–4000, unmeasured at production geometry", "estimated_peak_vram_mb": "1500, unmeasured at production geometry"}
    for name, value in (("plan.json", config), ("count-contract.json", counts), ("parameter-geometry.json", geometry)):
        save(run / name, value)
        source_pairs[name] = run / name
    preflight = {"schema": "mrun-preflight-config-v1", "hashed_roots": [*source_pairs, "preflight.v1.json"],
                 "path_token_allowlist": [config[key] for key in ("input_root", "parameter_root", "output_scratch", "output_name")],
                 "no_native_blocks_before_functional_seal": True, "no_pretrained_weight_exports": True}
    save(run / "preflight.v1.json", preflight)
    source_pairs["preflight.v1.json"] = run / "preflight.v1.json"
    request = {
        "schema": "mdb-qwen-block0-attention-request-v1", "config": config,
        "parent_request_sha256": PARENT_REQUEST, "execution_environment": parent["execution_environment"],
        "scheduler_url": parent["scheduler_url"], "model": "qwen3-in-flux2-klein-4b", "task_family": "analysis",
        "backend": "torch-qwen3-block0-attention", "device": "cuda", "dtype": "bfloat16", "capture_sites": True,
        "needs": {"cuda": True, "host": "beast"}, "runtime_wheels": parent["runtime_wheels"],
        "source_hashes": {name: sha(path) for name, path in source_pairs.items()}, "launcher_sha256": sha(__file__),
        "workload_geometry": {"model_loads": 1, "physical_native_block_hydrations": 1, "peak_physical_batch": 1,
                              "retained_sequence_rows": 512, "shared_prefix_rows": 30, "selected_parameter_tensors": 12,
                              "selected_parameter_payload_bytes": payload_bytes, **counts},
        "resource_exception": {"ram_mb": 8192, "vram_mb": 4096, "cpu_threads": 4, "disk_gb": 2,
                               "est_wall_s": 240, "timeout_s": 900,
                               "calibration": "Metadata-derived12BF16 tensors979773952B; one aliasing native block, retained source/triangle ports and streamed parameter hashes. Tiny qualification uses actual installed equations/serializers; author estimates3–4GB RSS/~1.5GB VRAM. First original-geometry peak unmeasured. No full encoder/FLUX/VAE; full original source archives hash once.240s estimate,600s cooperative/900s outer,observe actual peaks."},
        "scheduling_contract": one_lease_scheduling_contract(
            model_loads=1, logical_branches=5, logical_operations=7, max_local_work_items=32,
            extra={"pipeline_assemblies": 1, "physical_component_hydrations": 1,
                   "native_block_calls": 2, "native_rotary_calls": 2, "total_sdpa_calls": 7,
                   "logical_operations_scope": "2functional source blocks+3fixed-bank attention/tail arms+2native source blocks; parameter-free native rotary called twice separately.",
                   "logical_branches_scope": "2source cases+3triangle variants; source cases have separately counted functional/native executions."},
            checkpoint_policy="One finite diagnostic: source functional ports and three fixed-bank arms sealed before two native block calls. Historical H0/H1 are known replay controls; no retries, nesting or extra profiler forwards.",
        ),
    }
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(path, name) for name, path in source_pairs.items() if name != WORKER])
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Staged closure differs: " + name)
    save(run / "staged-command.json", {"command": command(request), "request_sha256": request["request_sha256"], "submitted": False})
    return {"request_sha256": request["request_sha256"], "source_files": len(source_pairs),
            "selected_parameter_bytes": payload_bytes, "selected_input_members": len(inputs), "submitted": False}


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    review = read(run / "finite-run-review-v1.json")
    if (run / "submission.json").exists() or identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("Submission/request/launcher changed")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("Exact reviewed finite request required")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Reviewed staged closure changed: " + name)
    if len(QUEUE_NOTE) > 500:
        raise ValueError("Queue note exceeds scheduler ABI")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    resources = request["resource_exception"]
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"],
                                          "review_sha256": sha(run / "finite-run-review-v1.json")})
    job = guarded_mrun_launch(args, request=request, experiment="mdb-qwen-block0-attention", task="analysis", model=None,
                             payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
                             **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
                             retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--compact", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and (args.parent_run is None or args.compact is None):
        parser.error("--parent-run and --compact are required to stage")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.parent_run, args.compact), sort_keys=True))
