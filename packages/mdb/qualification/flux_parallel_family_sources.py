"""Natural paired prefixes stopping before step0/single4, with no target images."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "mdb-parallel-relation-prefix-v1"
RESULT = "flux-parallel-relation-inputs"
MODEL_ID = "a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def contexts():
    rows = []
    suffix = " on a plain white background. Two separate objects, simple studio product photograph."
    templates = (("plain", "A red cube to the {side} of a blue sphere" + suffix),
        ("reworded", "On a plain white background, a blue sphere is to the {opposite} of a red cube. Two separate objects, simple studio product photograph."))
    for name, template in templates:
        for seed in (2119, 2137):
            rows.append(dict(id=f"cube-sphere-{name}-{seed}", seed=seed, panel="primary",
                prompts=dict(P=template.format(side="left", opposite="right"), Q=template.format(side="right", opposite="left"))))
    for name, objects in (("cone-cylinder", "A green cone to the {side} of a yellow cylinder"),
            ("pyramid-sphere", "A purple pyramid to the {side} of an orange sphere")):
        template = objects + suffix
        rows.append(dict(id=name + "-probe-2153", seed=2153, panel="shape-probe",
            prompts=dict(P=template.format(side="left"), Q=template.format(side="right"))))
    return rows


def counts():
    return dict(contexts=6, prompt_encodes=12, noise_draws=3, native_transitions=120,
        native_blocks=108, selected_block_calls=0, decoder_calls=0, images=0, input_cuts=12,
        pipeline_loads=1, encoder_cuda_placements=1, transformer_cuda_placements=1)


def before_source(point):
    require(isinstance(point, dict) and type(point.get("logical_step")) is int
        and point.get("logical_step") == 0 and point.get("edge") == "before", "prefix clock/edge differs")
    phase, index = point.get("phase"), point.get("index")
    require((phase == "project" and index is None) or (type(index) is int
        and ((phase == "joint" and 0 <= index < 5) or (phase == "single" and 0 <= index < 4))),
        "selected block or later operation forbidden in native prefix")
    return point


def run(config):
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    require(config["contexts"] == config["family_rule"]["fresh_contexts"] == contexts(), "frozen context census differs")
    require(sha("family-rule.json") == config["family_rule_file_sha256"] and read("family-rule.json") == config["family_rule"], "frozen rule bytes differ")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {k: v for k, v in config.items() if k != "_request_record"})
    save(output / "request.json", config["_request_record"])
    shutil.copyfile("family-rule.json", output / "family-rule.json")
    expected, counter = counts(), {k: 0 for k in counts()}
    counter["contexts"] = 6
    started, released, current = time.monotonic(), dict(prefix=False), dict(point=None)
    store = LocalStore(output / "state-store")

    def event(kind, **fields):
        require(time.monotonic()-started <= config["max_wall_seconds"], "prefix wall allowance exhausted")
        value = dict(kind=kind, seconds=time.monotonic()-started, counts=dict(counter), **fields)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(value) + b"\n")
        print("MDB_PARALLEL_PREFIX " + canonical(value).decode(), flush=True)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available() and str(torch.__version__).split("+")[0] == "2.13.0", "original CUDA/Torch profile required")
    for name, digest in config["original_source_sha256"].items():
        require(sha(importlib.metadata.distribution("diffusers").locate_file(name)) == digest, "native Diffusers source differs")
    for name, digest in config["checkpoint_files"].items():
        require(sha(Path(config["model_path"]) / name) == digest, "encoder/tokenizer checkpoint differs")
    manifest = read(config["flux_manifest"])
    for entry in read(config["vae_manifest"])["artifacts"]:
        entry = json.loads(json.dumps(entry))
        entry["path"] = "vae-" + entry["path"]
        entry["source"]["file"] = "vae/" + entry["source"]["file"]
        manifest["artifacts"].append(entry)
    manifest["defaults"] = dict(checkpoint_root=config["model_path"], artifact_root="parameter-cache", ssh_host="beast", python="python3")
    save(output / "parameter-artifacts.json", manifest)
    shutil.copyfile(config["flux_retrieval_script"], output / "retrieve_weights.py")
    shutil.copyfile(HERE / "retrieve_flux_encoder.py", output / "retrieve_flux_encoder.py")
    save(output / "encoder-source.json", dict(checkpoint_files=config["checkpoint_files"], supplied_native_encodes=12))
    pipe = Flux2KleinPipeline.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    counter["pipeline_loads"] += 1
    handles = []

    def forbidden(_module, _args):
        counter["decoder_calls"] += 1
        raise ValueError("prefix decoder call forbidden")

    def guard(phase, index):
        def check(_module, _args):
            point = current["point"]
            require(released["prefix"] and point is not None, "unscoped native prefix block")
            if point["logical_step"] != 0 or (phase == "single" and index >= 4):
                counter["selected_block_calls"] += 1
                raise ValueError("selected block/output crossed before prefix frontier")
            before_source(point)
            require((point["phase"], point["index"]) == (phase, index), "native prefix dispatch differs")
            counter["native_blocks"] += 1
        return check

    for phase, blocks in (("joint", pipe.transformer.transformer_blocks), ("single", pipe.transformer.single_transformer_blocks)):
        handles.extend(block.register_forward_pre_hook(guard(phase, i)) for i, block in enumerate(blocks))
    handles.append(pipe.vae.decoder.register_forward_pre_hook(forbidden))

    def retain(session):
        cut = session.capture(retain=False)
        store.save(cut)
        return dict(cut=cut.fingerprint, parent=cut.parent, model_identity=cut.model_identity,
            execution=dict(cut.execution), execution_point=dict(cut.execution_point), store="state-store")

    def execute(_):
        released["prefix"] = True
        embeddings = {}
        pipe.text_encoder.eval().requires_grad_(False).to("cuda")
        counter["encoder_cuda_placements"] += 1
        with torch.inference_mode():
            for context in contexts():
                for role, prompt in context["prompts"].items():
                    value, ids = pipe.encode_prompt(prompt, device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=512)
                    embeddings[context["id"], role] = (value.cpu(), ids.cpu())
                    counter["prompt_encodes"] += 1
                event("prompt_pair_encoded", context=context["id"])
            pipe.text_encoder = None
            pipe.tokenizer = None
            del value, ids
            gc.collect()
            torch.cuda.empty_cache()
            pipe.transformer.eval().requires_grad_(False).to("cuda")
            pipe.vae.eval().requires_grad_(False)
            counter["transformer_cuda_placements"] += 1
            adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", device="cuda")
            require(adapter.model_identity == MODEL_ID, "original complete model identity differs")
            noise = {}
            for seed in sorted({c["seed"] for c in contexts()}):
                latent, ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4, 512, 512,
                    torch.bfloat16, torch.device("cuda"), torch.Generator(device="cuda").manual_seed(seed))
                noise[seed] = (latent.cpu(), ids.cpu())
                counter["noise_draws"] += 1
            roots = {}
            for context in contexts():
                cid = context["id"]
                roots[cid] = dict(inputs={}, context=context, source_policy="natural-paired-prompts/same-noise/step0-single4-input", support="all image rows; Q text retained in hybrids")
                for role in ("Q", "P"):
                    latent, ids = (v.to("cuda") for v in noise[context["seed"]])
                    conditioning, txt_ids = (v.to("cuda") for v in embeddings[cid, role])
                    session = adapter.session(latent=latent.clone(), conditioning=conditioning, img_ids=ids, txt_ids=txt_ids,
                        timesteps=torch.tensor(config["schedule"]["timesteps"], dtype=torch.float32, device="cuda"),
                        sigmas=torch.tensor(config["schedule"]["sigmas"], dtype=torch.float32, device="cuda"))
                    retain(session)
                    for _step in range(10):
                        current["point"] = before_source(dict(session.capture(retain=False).execution_point))
                        receipt = session.continue_()
                        with (output / "receipts.jsonl").open("ab") as stream:
                            stream.write(canonical(dict(context=cid, role=role, receipt=receipt.to_dict())) + b"\n")
                        counter["native_transitions"] += 1
                    current["point"] = None
                    cut = retain(session)
                    require((cut["execution_point"]["logical_step"], cut["execution_point"]["phase"], cut["execution_point"]["index"]) == (0, "single", 4), "prefix stop differs")
                    roots[cid]["inputs"][role] = cut
                    counter["input_cuts"] += 1
                    del session
                event("prefix_pair_captured", context=cid)
            save(output / "source-roots.json", roots)
            return dict(counts=counter, roots=roots)

    tx = ForecastTransaction(output / "source-transaction", model_identity=MODEL_ID,
        input_identity=hashlib.sha256(canonical(contexts())).hexdigest(), source_identity=config["family_rule_file_sha256"],
        frontier=dict(kind="architectural", scope="Native preparation ending before selected block; no images"),
        budgets=dict(max_predictions=1, max_executions=1, max_bytes=4 << 30, max_files=10000, max_wall_ms=config["max_wall_seconds"]*1000))
    tx.frontier("frozen-natural-source-policy", dict(rule=config["family_rule"], counts=expected))
    tx.predict("preparation-only", lambda _: dict(predictions=dict(stop="step0/single4/before", selected_block_calls=0, decoder_calls=0)))
    seal = tx.seal()
    event("source_policy_sealed", seal=seal["seal_sha256"])
    try:
        tx.execute("native-prefix-inputs", execute)
        tx.close()
    finally:
        for handle in handles:
            handle.remove()
    require(counter == expected, "physical prefix census differs")
    result = dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], counts=counter,
        source_policy_seal=seal["seal_sha256"], transaction=verify_transaction(output / "source-transaction"),
        mechanics_status="completed", terminal_status="unassessed", seconds=time.monotonic()-started)
    save(output / "summary.json", result)
    return result


def load_sources(config, output, store, adapter):
    """Authenticate exactly the published input-cut closures; no future reads."""
    import torch
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import verify_transaction

    source = config["fresh_source"]
    root, output = Path(source["output_root"]).resolve(), Path(output).resolve()
    require(not output.is_relative_to(root) and not root.is_relative_to(output), "source/output overlap")
    require(sha(root / "files.json") == source["files_manifest_sha256"], "source manifest differs")
    manifest = read(root / "files.json")
    require(manifest["job_id"] == source["job_id"] and manifest["request_sha256"] == source["request_sha256"], "source identity differs")
    consumed = {}

    def member(name):
        require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe source member")
        path = root / name
        require(path.resolve().is_relative_to(root) and not any(p.is_symlink() for p in (path, *path.parents)), "source member escapes")
        record = manifest["files"][name]
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "source member changed")
        consumed[name] = record
        return path

    request, summary = read(member("request.json")), read(member("summary.json"))
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] == source["request_sha256"], "source request differs")
    require(request["source_hashes"][Path(__file__).name] == sha(__file__), "executed source worker differs")
    require(summary["schema"] == SCHEMA and summary["job_id"] == source["job_id"]
        and summary["request_sha256"] == identity and summary["counts"] == counts(), "source census differs")
    require(sha(member("family-rule.json")) == config["family_rule_file_sha256"]
        and read(root / "family-rule.json") == config["family_rule"] == request["config"]["family_rule"], "frozen family-rule lineage differs")
    for name in manifest["files"]:
        if name.startswith("source-transaction/"):
            member(name)
    verification = verify_transaction(root / "source-transaction", expected_seal_sha256=summary["source_policy_seal"])
    roots = read(member("source-roots.json"))
    require(set(roots) == {c["id"] for c in contexts()}, "prefix contexts incomplete")
    original = LocalStore(root / "state-store")
    for context in contexts():
        cid, cuts = context["id"], {}
        require(roots[cid]["context"] == context, "prefix prompt metadata differs")
        for role in ("P", "Q"):
            record = roots[cid]["inputs"][role]
            metadata = read(member("state-store/cuts/" + record["cut"] + ".json"))
            require(metadata["schema"] == "saturn-pub-local-cut-v2", "public v2 cut required")
            for page in metadata["slots"].values():
                member("state-store/pages/" + page + ".json")
            for blob in metadata["physical_blobs"]:
                member("state-store/blobs/" + blob + ".safetensors")
            cut = original.load(record["cut"], device="cpu")
            adapter.validate(cut.payload)
            require(cut.model_identity == record["model_identity"] == adapter.model_identity == MODEL_ID
                and dict(cut.execution_point) == record["execution_point"] == adapter.point(cut.payload).to_dict()
                and cut.parent == record["parent"] and dict(cut.execution) == record["execution"], "source cut closure differs")
            require((cut.payload["step"], cut.payload["phase"], cut.payload["cursor"]) == (0, "single", 4), "source frontier differs")
            expected_execution = dict(adapter.execution, device=dict(cut.execution)["device"])
            require(dict(cut.execution) == expected_execution and cut.execution["dtype"] == "torch.bfloat16"
                and cut.execution["device"]["type"] == "cuda", "source numerical profile differs")
            store.save(cut)
            cuts[role] = cut
        for key in ("latent", "img_ids", "txt_ids", "timesteps", "sigmas"):
            a, b = (cuts[role].payload[key] for role in ("P", "Q"))
            require(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), "paired source noise/schedule/IDs differ: " + key)
    return roots, dict(source=source, family_rule_sha256=config["family_rule_file_sha256"],
        transaction=verification, consumed_members=consumed, model_calls=0,
        scope="Only observed natural input-cut closures; conditioning/text/image differences supplied, no selected-block output or image")


def worker(config_path):
    config, request = read(config_path), read(os.environ["SATURN_DEBUG_REQUEST"])
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request["request_sha256"] and request["config"] == config, "request/config differs")
    for name, digest in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == digest, "payload source differs")
    job = os.environ["MRUN_JOB_ID"]
    config.update(job_id=job, request_sha256=identity, _request_record=request,
        output=str(Path(config["output_scratch"]) / job / RESULT))
    result = run(config)
    ready = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))["finalize"](config["output"], job_id=job,
        request_sha256=identity, max_expanded_bytes=4 << 30, max_member_bytes=134221824)
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
