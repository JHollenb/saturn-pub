"""Saved-only audit of one globally sealed raw-prompt-to-image program.

The pinned parser may be reapplied offline. No neural model, learned recurrence,
native decoder or CUDA RNG is called. Shared-primitive forecasts and observed
native consumers remain separate from independent arithmetic or semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_SHA256 = "b04db3835522a007a15293ef177e4463cf05e9d01122d47d42531dc72bd9700a"
PINS = {
    "audit_flux_functional_qwen_prefix.py": "2331126c9b3d918b1ac5376a6a13250c14806bdbcf151277f59320440e191c7a",
    "audit_flux_functional_remaining.py": "81f77f5f8e6a24915095875f1373097616c03b62c4f4983a588f0bc1fd957ea2",
    "audit_flux_functional_pass.py": "43e163b7543ec225da007b6ecc87418c7c0654f464c36c8e7f1e5234642aeaa8",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
NUMERICAL_PINS = {
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
}
NATIVE_MODULES = {
    "diffusers/pipelines/flux2/pipeline_flux2_klein.py", "diffusers/pipelines/flux2/image_processor.py",
    "diffusers/models/transformers/transformer_flux2.py", "diffusers/models/autoencoders/autoencoder_kl_flux2.py",
    "diffusers/schedulers/scheduling_flow_match_euler_discrete.py", "diffusers/utils/torch_utils.py",
}
HELPERS = tuple(PINS)
PROMPT = "A cobalt blue cube to the right of a matte orange ring on a gray tabletop. Two separate objects, simple studio product photograph."
WORKING = ("image", "text", "context", "noise_prediction")
CARRIED = ("conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(8 << 20), b""):
            digest.update(value)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def helpers():
    for name, digest in PINS.items():
        require(sha(HERE / name) == digest, "frozen saved-reader source differs: " + name)
    return {name: runpy.run_path(str(HERE / name)) for name in PINS}


def schedule(joints, singles, steps):
    """Independent complete clock geometry, including every commit reset."""
    result = []
    for step in range(steps):
        result += [dict(step=step, phase="project", cursor=0)]
        result += [dict(step=step, phase="joint", cursor=k) for k in range(joints)]
        result += [dict(step=step, phase="single", cursor=k) for k in range(singles)]
        result += [dict(step=step, phase=phase, cursor=singles) for phase in ("readout", "scheduler")]
    return result, [*result[1:], dict(step=steps, phase="project", cursor=0)]


def bytes_equal(a, b):
    import numpy as np
    if hasattr(a, "detach"):
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return False
        a, b = (value.detach().cpu() for value in (a, b))
        return bool(__import__("torch").equal(a.contiguous().view(__import__("torch").uint8), b.contiguous().view(__import__("torch").uint8)))
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def metric(predicted, native):
    """Full-array signed reduction; exact zeros and all numerical misses survive."""
    require(sha(HERE / "audit_flux_functional_qwen_prefix.py") == PINS["audit_flux_functional_qwen_prefix.py"], "qualified reduction source differs")
    return runpy.run_path(str(HERE / "audit_flux_functional_qwen_prefix.py"))["compare"](predicted, native)


def image_account(functional, same_latent_native, native):
    """Coordinate-wise decomposition, not causal shares or VAE error isolation."""
    import numpy as np
    require(functional.shape == same_latent_native.shape == native.shape and functional.size > 0, "three image geometries differ")
    p, s, n = (np.asarray(value, np.float64) for value in (functional, same_latent_native, native))
    require(all(np.isfinite(value).all() for value in (p, s, n)), "nonfinite image account")
    a, b, total = p - s, s - n, p - n
    dot = float(np.dot(a.ravel(), b.ravel()))
    energy = lambda value: float(np.dot(value.ravel(), value.ravel()))
    a2, b2, t2, n2 = (energy(value) for value in (a, b, total, n))
    return dict(terms={
        "functional_minus_same_forecast_latent_native_VAE": metric(functional, same_latent_native),
        "same_forecast_latent_native_VAE_minus_native_latent_native_VAE": metric(same_latent_native, native),
        "whole_functional_minus_native": metric(functional, native)},
        common_native_rms=math.sqrt(n2 / n.size),
        common_native_relative_rms={name: None if n2 == 0 else math.sqrt(value / n2)
            for name, value in (("decoder_component", a2), ("inherited_latent_consumer_effect", b2), ("whole", t2))},
        squared_norms=dict(decoder_component=a2, inherited_latent_consumer_effect=b2, whole=t2),
        cross_dot=dot, signed_cross_term=2 * dot,
        cross_cosine=None if a2 * b2 == 0 else dot / math.sqrt(a2 * b2),
        coordinate_identity_exact=bool(np.array_equal(total, a + b)),
        energy_identity_residual=t2 - (a2 + b2 + 2 * dot),
        scope="Observed same-latent native VAE consumer partitions the complete endpoint discrepancy; no causal percentage shares or independent decoder arithmetic.")


def initial_closure(lanes):
    """Verify packing/IDs/clock from saved operands, not a CUDA RNG emulator."""
    import numpy as np
    require(set(lanes) == {"forecast", "native"}, "two independently formed initial lanes required")
    for lane, row in lanes.items():
        raw, packed = row["raw_noise"], row["latent"]
        require(raw.ndim == 4 and raw.shape[0] == 1 and packed.shape == (1, raw.shape[2] * raw.shape[3], raw.shape[1]), "initial packed noise geometry differs")
        require(bytes_equal(raw.reshape(1, raw.shape[1], -1).transpose(0, 2, 1), packed), "initial noise is not own declared packing")
        require(row["step"] == row["cursor"] == 0 and row["phase"] == "project"
                and all(row.get(key) is None for key in WORKING), "initial working-state/clock differs")
        ids = row["img_ids"]
        expected = np.asarray([[0, y, x, 0] for y in range(raw.shape[2]) for x in range(raw.shape[3])], dtype=np.int64)[None]
        require(ids.dtype == np.int64 and bytes_equal(ids, expected), "native-order image integer IDs differ")
        tids = row["txt_ids"]
        texpected = np.zeros_like(tids)
        texpected[0, :, 3] = np.arange(tids.shape[1])
        require(tids.dtype == np.int64 and tids.shape[0] == 1 and tids.shape[2] == 4
                and bytes_equal(tids, texpected), "native-order text integer IDs differ")
        require(row["timesteps"].ndim == row["sigmas"].ndim == 1
                and len(row["sigmas"]) == len(row["timesteps"]) + 1 and row["sigmas"][-1] == 0,
                "initial schedule geometry/terminal sigma differs")
        require(row["formation"]["noise_draws"] == 1 and row["formation"]["conditioning_lane"] == lane,
                "independent input primitive formation differs")
    return {key: metric(lanes["forecast"][key], lanes["native"][key])
        for key in ("raw_noise", "latent", "img_ids", "txt_ids", "timesteps", "sigmas")}


def own_state_join(previous, current, before, after):
    """Only structural/carry joins are hard checks; numerical outputs are not."""
    require({key: previous[key] for key in ("step", "phase", "cursor")} == before
            and {key: current[key] for key in ("step", "phase", "cursor")} == after, "state clock differs")
    for key in CARRIED:
        require(bytes_equal(previous[key], current[key]), "state refreshed a carried input: " + key)
    if before["phase"] == "scheduler":
        require(all(current[key] is None for key in WORKING), "commit failed to reset working fields")
    else:
        require(bytes_equal(previous["latent"], current["latent"]), "latent refreshed outside own scheduler commit")
    if before["phase"] in ("joint", "single", "readout", "scheduler"):
        require(previous["context"] is not None, "own context missing before learned/commit operation")
    if before["phase"] in ("joint", "single", "readout"):
        require(current["context"] is not None, "own time context missing")
        for key in previous["context"]:
            p, n = previous["context"][key], current["context"][key]
            require(all(bytes_equal(a, b) for a, b in zip(p, n, strict=True)) if isinstance(p, (tuple, list))
                    else bytes_equal(p, n), "context refreshed within a pass")


def catalog_comparison(predicted, reference, *, compare_core=None):
    if compare_core is None:
        from mdb_lanes.equation_comparison import compare_equations
        compare_core = compare_equations
    return compare_core(predicted, reference, max_elements=sum(int(value.size) for value in predicted.values()))


def field_arrays(payload):
    """Only live numerical fields; carried input comparisons have separate scope."""
    import numpy as np
    values = {key: payload[key] for key in ("image", "text", "latent", "noise_prediction") if payload[key] is not None}
    if payload["context"] is not None:
        for key, value in payload["context"].items():
            if key == "rope":
                values.update({"context.rope." + str(index): part for index, part in enumerate(value)})
            else:
                values["context." + key] = value
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            return value.float().numpy() if str(value.dtype) == "torch.bfloat16" else value.numpy()
        return np.asarray(value)
    return {key: array(value) for key, value in values.items()}


def trajectory_accounts(predicted, native, origins, load, *, joints, singles, steps):
    """Stream successive own-state pairs, preserving all full-field accounts.

    ``load(record, lane)`` authenticates a saved cut and returns its payload.
    Only the previous/current pair per lane is retained, never a weight bank.
    """
    before, after = schedule(joints, singles, steps)
    require(len(predicted) == len(native) == len(before), "complete trajectory coverage differs")
    previous = {lane: load(origins[lane], lane) for lane in ("forecast", "native")}
    parents = {lane: origins[lane]["cut"] for lane in previous}
    rows, first_bits, first_values = [], None, None
    for index, (prow, nrow) in enumerate(zip(predicted, native, strict=True)):
        records = dict(forecast=prow, native=nrow)
        current = {}
        for lane, record in records.items():
            require(record["parent"] == parents[lane], "own trajectory parent lineage differs")
            current[lane] = load(record, lane)
            own_state_join(previous[lane], current[lane], before[index], after[index])
        p, n = (field_arrays(current[lane]) for lane in ("forecast", "native"))
        require(set(p) == set(n), "paired native/forecast live-field coverage differs")
        fields = {key: metric(p[key], n[key]) for key in p}
        commit_equation = {}
        if before[index]["phase"] == "scheduler":
            for lane in ("forecast", "native"):
                import numpy as np
                source = previous[lane]
                step = before[index]["step"]
                sigma = field_to_numpy(source["sigmas"])
                dt = np.float32(sigma[step + 1] - sigma[step])
                rne = runpy.run_path(str(HERE / "audit_flux_functional_pass.py"))["rne_bf16"]
                delta = rne(rne(dt) * field_to_numpy(source["noise_prediction"]))
                expected = rne(field_to_numpy(source["latent"]) + delta)
                commit_equation[lane] = dict(dt_fp32=float(dt), comparison=metric(expected, field_to_numpy(current[lane]["latent"])),
                    scope="Source scalar/noise BF16 product then FP32 latent addition and final BF16 cast; independent saved-operand pointwise check, not a learned recurrence.")
        if first_bits is None:
            field = next((key for key, value in fields.items() if not value["bitwise_exact"]), None)
            if field is not None:
                first_bits = dict(number=index, before=before[index], field=field)
        if first_values is None:
            field = next((key for key, value in fields.items() if value["changed_values"]), None)
            if field is not None:
                first_values = dict(number=index, before=before[index], field=field)
        rows.append(dict(number=index, before=before[index], after=after[index],
            forecast_cut=prow["cut"], native_cut=nrow["cut"], fields=fields, commit_source_equation=commit_equation))
        previous = current
        parents = {lane: records[lane]["cut"] for lane in current}
    return dict(boundaries=rows, first_bit_discrepancy=first_bits, first_value_discrepancy=first_values,
        complete_transitions=len(rows), independent_neural_calls=0,
        scope="Every retained own-state boundary in both lanes. Full-field differences are observations; working-state lifetimes and carried identities are structural joins.")


def member(root, name):
    path = root / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts and path.is_file()
            and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "unsafe or missing evidence member")
    return path


def load_array(root, record):
    """Decode exact producer NPY payloads; keep integers and signed zeros lossless."""
    import numpy as np
    path = member(root, record["path"])
    require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "saved array bytes differ")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(list(value.shape) == record["shape"] and value.dtype.kind in "iufb" and value.flags.c_contiguous, "saved array shape/storage differs")
    dtype = record.get("stored_dtype", record.get("dtype"))
    require(str(value.dtype) == dtype, "saved array stored dtype differs")
    if "content_sha256" in record:
        require(hashlib.sha256(value.tobytes()).hexdigest() == record["content_sha256"], "saved raw array content differs")
    if record.get("encoding") == "bf16-bits-v1":
        require(value.dtype == np.dtype("uint16") and record["logical_dtype"] == "torch.bfloat16", "BF16 archive dtype differs")
        value = (value.astype(np.uint32) << 16).view(np.float32)
    elif "logical_dtype" in record:
        require(record["encoding"] == "native-numpy-v1" and record["logical_dtype"] == "torch." + str(value.dtype), "native array logical dtype differs")
    require(np.isfinite(value).all(), "nonfinite saved array")
    value.flags.writeable = False
    return value


def source_metadata(root, config, source):
    expected = ("qwen_config", "qwen_manifest", "qwen_source_binding", "qwen_retrieval_script", "qwen_recipe_helper",
        "flux_config", "flux_scheduler_config", "flux_manifest", "flux_retrieval_script", "flux_recipe_helper",
        "vae_config", "vae_manifest", "vae_retrieval_script", "vae_recipe_helper")
    require(set(source["source_files"]) == set(expected), "complete standalone source metadata differs")
    paths = {}
    for key in expected:
        row = source["source_files"][key]
        path = member(root, row["path"])
        require(path.name == Path(config[key]).name and sha(path) == row["sha256"] == config[key + "_sha256"]
                and path.stat().st_size == row["bytes"], "copied metadata source differs: " + key)
        paths[key] = path
    return paths


def parameter_audit(root, config, source, contract, metadata, helper):
    result = {}
    for component in ("qwen", "flux", "vae"):
        manifest = read(metadata[component + "_manifest"])
        expected = {row["source"]["tensor"]: row["source"]["payload_sha256"] for row in manifest["artifacts"]}
        require(len(expected) == len(manifest["artifacts"]), "duplicate parameter recipe identity")
        before, after = (source[component + "_parameter_identity_" + when] for when in ("before", "after"))
        require(set(before) == set(after), "resident parameter/buffer coverage changed")
        for name, first in before.items():
            last = after[name]
            require(all(first[key] == last[key] for key in ("shape", "stride", "dtype", "sha256"))
                    and first["device"] == last["device"] == "cpu", "resident identity/buffer changed: " + name)
        for row in manifest["artifacts"]:
            name = row["source"]["tensor"]
            require(name in before and before[name]["sha256"] == row["source"]["payload_sha256"]
                    and before[name]["shape"] == row["source"]["shape"]
                    and before[name]["dtype"] == "torch.bfloat16", "actual learned parameter differs from original recipe: " + name)
            require(not (root / row["path"]).exists(), "persistent per-run learned bank forbidden")
        result[component] = helper["restore_parameters"](manifest, config[component + "_parameter_root"], expected,
            expected_module_sha256=config[component + "_retrieval_script_sha256"])
        result[component]["extra_runtime_buffers"] = sorted(set(before) - set(expected))
        require(contract[component + "_identity_before"] == before and contract[component + "_identity_after"] == after,
                "final resident source/contract identity join differs")
    if config.get("profile") != "tiny-cpu":
        require([result[key]["tensors"] for key in ("qwen", "flux", "vae")] == [398, 169, 142], "original709recipe coverage differs")
        rotary = {name: row for name, row in source["qwen_parameter_identity_before"].items() if name.endswith("inv_freq")}
        require(rotary and all(row["dtype"] == "torch.float32" and row["shape"] == [64] for row in rotary.values()), "native CPU-init FP32 Qwen rotary buffers differ")
    require(source["qwen_tied_alias"] is contract["qwen_tied_alias"] is True, "Qwen tied LM/embedding alias differs")
    require(source["placement_records"] == contract["placement_records"], "actual placement source/contract records differ")
    for row in source["placement_records"]:
        component = row["component"]
        cpu = source[component + "_parameter_identity_before"]
        gpu = {name: {**value, "device": "cpu" if component == "vae" and name.startswith("bn.") else "cuda:0"} for name, value in cpu.items()}
        expected_before, expected_after = (cpu, gpu) if row["direction"] == "cuda" else (gpu, cpu)
        identity = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(row["before_sha256"] == identity(expected_before) and row["after_sha256"] == identity(expected_after)
                and row["before_signature"] == expected_before and row["after_signature"] == expected_after
                and row["tensor_count"] == len(cpu) and row["identity_preserved"] is True, "per-placement tensor/buffer signatures differ")
    return result


def runtime_audit(config, runtime):
    for name, row in runtime["implementation_sources"].items():
        if config[name] == "runtime-bound":
            require(name == "tokenizers_native_source_sha256" and row["expected_sha256"] == "runtime-bound"
                    and row["binding"] == "runtime-observed-on-CUDA-host" and len(row["sha256"]) == 64
                    and set(row["sha256"]) <= set("0123456789abcdef"), "actual Linux tokenizer native binding differs")
        else:
            require(row["sha256"] == row["expected_sha256"] == config[name] and row["binding"] == "staged-source-hash", "actual implementation source differs")
    require(runtime["rope_buffer_policy"] == config["qwen_rope_buffer_policy"]
            and runtime["attention_dispatch_source"] == config["attention_dispatch"], "original Qwen runtime profile differs")
    require(set(config["native_sources"]) == set(runtime["native_diffusers_sources"]) == NATIVE_MODULES, "complete native pipeline/transformer/decoder/scheduler source bindings differ")
    for name, row in runtime["native_diffusers_sources"].items():
        declared = config["native_sources"][name]
        expected = declared["sha256"] if isinstance(declared, dict) else declared
        require(row["expected_sha256"] == expected and len(row["sha256"]) == 64 and set(row["sha256"]) <= set("0123456789abcdef")
                and row["binding"] == ("runtime-observed" if expected == "runtime-bound" else "staged-source-hash")
                and (expected == "runtime-bound" or row["sha256"] == expected), "actual native implementation source differs: " + name)
    flags = runtime["numeric_flags_before_forecast"]
    require(flags["deterministic_algorithms"] is True and flags["float32_matmul_precision"] == "highest"
            and flags["cuda_tf32"] is False and flags["cudnn_tf32"] is False and flags["cudnn_benchmark"] is False
            and flags["cudnn_deterministic"] is True and flags["cuda_bf16_reduced_precision_reduction"] is config["source_reduction_policy"],
            "original numerical flags differ")
    return runtime


def parser_audit(root, config, source, forecast, native):
    import numpy as np
    from transformers import AutoTokenizer
    binding = read(member(root, source["source_files"]["qwen_source_binding"]["path"]))
    expected_sources = {name for name in binding["source_files"] if name.startswith("tokenizer/")
        or name in {"text_encoder/config.json", "text_encoder/generation_config.json"}}
    require(set(source["tokenizer_files"]) == expected_sources, "complete pinned tokenizer source set differs")
    for name, row in source["tokenizer_files"].items():
        path = member(root, row["retained"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "pinned tokenizer member differs")
        require(row["sha256"] == binding["source_files"][name]["sha256"] and row["bytes"] == binding["source_files"][name]["bytes"], "tokenizer/source recipe identity differs")
    tokenizer = AutoTokenizer.from_pretrained(root / "source/tokenizer", local_files_only=True, padding_side="right")
    require(tokenizer.is_fast is True and type(tokenizer).__name__ == read(root / "source/tokenizer/tokenizer_config.json")["tokenizer_class"], "actual parser class differs")
    messages = [dict(role="user", content=config["prompt"])]
    options = dict(add_generation_prompt=True, enable_thinking=False)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, padding="max_length", truncation=True,
        max_length=config["max_length"], return_dict=True, return_tensors="np", **options)
    ids = np.asarray(tokens["input_ids"], np.int64).reshape(1, -1)
    mask = np.asarray(tokens["attention_mask"], np.int64).reshape(1, -1)
    require(ids.shape == mask.shape == (1, config["max_length"]) and np.isin(mask, (0, 1)).all()
            and mask.sum() > 0 and np.all(mask[0, 1:] <= mask[0, :-1]), "tokenized mask/row geometry differs")
    require(forecast["prompt"] == native["prompt"] == config["prompt"] and forecast["formatted_chat"] == formatted, "raw/formatted prompt bytes differ")
    for row in (forecast, native):
        require(bytes_equal(load_array(root, row["input_ids"]), ids) and bytes_equal(load_array(root, row["attention_mask"]), mask), "actual token/mask input bytes differ")
    require(bytes_equal(load_array(root, forecast["position_ids"]), np.arange(config["max_length"], dtype=np.int64)[None]), "native prefix position IDs differ")
    return mask, dict(prompt_utf8_sha256=hashlib.sha256(config["prompt"].encode()).hexdigest(),
        formatted_chat_sha256=hashlib.sha256(formatted.encode()).hexdigest(),
        input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(), attention_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        attended_rows=int(mask.sum()), padded_rows=int(mask.size - mask.sum()),
        scope="Same pinned tokenizer/template reapplied offline; zero neural calls, not zero tokenizer calls. All query rows retained and consumed.")


def authenticate(root, *, expected_manifest, expected_request):
    require(sha(root / "files.json") == expected_manifest, "raw manifest bytes differ")
    inventory = read(root / "files.json")
    require(inventory.get("schema") == "mdb-image-tail-files-v1" and inventory.get("request_sha256") == expected_request,
            "qualified finalizer manifest schema/request differs")
    files = inventory["files"]
    require(set(files) == {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} - {"files.json"}, "complete raw member inventory differs")
    for name, row in files.items():
        path = member(root, name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "raw evidence checksum differs: " + name)
    config, source, summary, contract, request = [read(root / name) for name in ("plan.json", "source.json", "summary.json", "native-contract.json", "request.json")]
    require(inventory["job_id"] == source["job_id"] and inventory["expanded_bytes"] == sum(row["bytes"] for row in files.values()), "manifest actual job/expanded byte census differs")
    canonical_request = json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()
    require(hashlib.sha256(canonical_request).hexdigest() == expected_request == request["request_sha256"]
            and all(value["request_sha256"] == expected_request for value in (config, source, summary))
            and source["job_id"] == summary["job_id"] == config["job_id"], "executed request/job identity differs")
    require(all(config[key] == value for key, value in request["config"].items())
            and request["source_hashes"]["flux_raw_to_image.py"] == source["worker_sha256"] == WORKER_SHA256
            and source["helper_source_sha256"] == config["helper_source_sha256"]
            and all(request["source_hashes"][name] == digest for name, digest in source["helper_source_sha256"].items()), "executed worker/helper/effective plan differs")
    require(all(source["helper_source_sha256"].get(name) == digest for name, digest in NUMERICAL_PINS.items()), "qualified numerical graph source differs")
    require(config["prompt"] == summary["prompt"] == PROMPT and config["seed"] == summary["seed"] == 277
            and config["steps"] == 4 and config["batch_size"] == 1, "fixed raw specimen differs")
    require(summary["counts"] == source["physical_counts"] == contract["counts"], "actual physical census sidecars differ")
    counts = summary["counts"]
    expected = dict(pipeline_assemblies=1, component_hydrations=3, cuda_placements=6, cpu_offloads=6,
        noise_draws=2, native_qwen_calls=1, functional_decode_calls=1, native_vae_calls=2, guard_refusals=0)
    require(all(counts[key] == value for key, value in expected.items()) and contract["native_module_forwards_before_seal"] == 0,
            "actual component/guard census differs")
    return config, source, summary, contract, inventory


def qwen_accounts(root, config, forecast, native, mask, helper, *, hidden_width):
    import numpy as np
    prefix = config.get("prefix_layers", 27)
    selected = config.get("selected_hidden_states", [9, 18, 27])
    require(len(forecast["hidden_states"]) == len(native["hidden_states"]) == prefix + 1
            and forecast["native_module_forwards"] == 0 and native["native_module_forwards"] == 1
            and native["executed_full_layers"] == config.get("full_layers", 36) > prefix, "complete interior Qwen prefix/reference capture differs")
    rows, first_bits, first_values = {}, None, None
    selected_values = {lane: [] for lane in ("forecast", "native")}
    for index, (p, n) in enumerate(zip(forecast["hidden_states"], native["hidden_states"], strict=True)):
        require(p["logical_dtype"] == n["logical_dtype"] == "torch.bfloat16", "native/functional hidden dtype differs")
        p, n = load_array(root, p), load_array(root, n)
        require(p.shape == n.shape == (*mask.shape, hidden_width), "complete hidden query-row geometry differs")
        rows["hidden." + str(index)] = helper["row_accounts"](p, n, mask)
        if first_bits is None and not rows["hidden." + str(index)]["full"]["bitwise_exact"]:
            first_bits = "hidden." + str(index)
        if first_values is None and rows["hidden." + str(index)]["full"]["changed_values"]:
            first_values = "hidden." + str(index)
        if index in selected:
            selected_values["forecast"].append(p)
            selected_values["native"].append(n)
    for lane, row in (("forecast", forecast), ("native", native)):
        require(bytes_equal(np.concatenate(selected_values[lane], axis=-1), load_array(root, row["conditioning"])), "E refreshed or not own selected interior states")
    p, n = (load_array(root, row["conditioning"]) for row in (forecast, native))
    rows["conditioning"] = helper["row_accounts"](p, n, mask)
    for name in ("conditioning",):
        if first_bits is None and not rows[name]["full"]["bitwise_exact"]:
            first_bits = name
        if first_values is None and rows[name]["full"]["changed_values"]:
            first_values = name
    return dict(fields=rows, first_bit_discrepancy=first_bits, first_value_discrepancy=first_values,
        interior_states=selected, final_normalization="Reference-only after full encoder, not an interior prefix target.",
        scope="All query rows retained, including padding. Shared original primitives; no independent attention arithmetic or semantic alignment.")


def frontend_audit(root, config, frontends, origins):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    lanes, generators = {}, {}
    for lane, row in frontends.items():
        require(row["lane"] == lane and row["seed"] == 277, "independently formed frontend occurrence differs")
        arrays = {key: load_array(root, row[key]) for key in ("raw_noise", "packed_noise", "img_ids", "txt_ids", "timesteps", "sigmas")}
        source = origins[lane]
        for key, value in arrays.items():
            slot = "latent" if key == "packed_noise" else key
            if key != "raw_noise":
                observed = field_to_numpy(source[slot])
                require(bytes_equal(value, observed), "initial state did not consume own frontend operand: " + slot)
        lanes[lane] = dict(source, **{key: value for key, value in arrays.items() if key != "packed_noise"}, latent=arrays["packed_noise"],
            formation=dict(noise_draws=1, conditioning_lane=lane))
        generators[lane] = {key: load_array(root, row[key]) for key in ("generator_before", "generator_after")}
        require(all(value.dtype == np.uint8 and value.ndim == 1 for value in generators[lane].values()), "CUDA generator raw-state archive differs")
        base = np.linspace(1., 1 / config["steps"], config["steps"])
        mu = compute_empirical_mu(image_seq_len=arrays["packed_noise"].shape[1], num_steps=config["steps"])
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(row["scheduler_config"])
        scheduler.set_timesteps(num_inference_steps=config["steps"], device="cpu", sigmas=base, mu=mu)
        require(base.tolist() == row["base_sigmas"] and row["mu"] == float(mu)
                and bytes_equal(scheduler.timesteps.numpy(), arrays["timesteps"])
                and bytes_equal(scheduler.sigmas.numpy(), arrays["sigmas"]), "source scheduler/sigma/timestep construction differs")
    result = initial_closure(lanes)
    result["generator_state"] = {key: metric(generators["forecast"][key], generators["native"][key]) for key in ("generator_before", "generator_after")}
    result["rng_scope"] = "Two independently created same-seed CUDA generators, one draw each; saved raw states/noise and packing audited. No independent CUDA RNG emulator or seed-only identity claim."
    return result


def field_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.float().numpy() if str(value.dtype) == "torch.bfloat16" else value.numpy()
    return value


def state_loader(root, helper, config):
    from saturn_pub.store import LocalStore

    from mdb_lanes.forward_recording import tensor_descriptor
    identities = {}
    def load(record, lane):
        store = LocalStore(root / (lane + "-state-store"))
        cut = store.load(record["cut"])
        require(cut.fingerprint == record["cut"] and cut.parent == record["parent"] and cut.boundary == record["boundary"]
                and plain(cut.execution_point) == record["execution_point"] == helper["point"](record["clock"]), "typed state cut/clock/sidecar differs")
        require(cut.execution["composition"] == "mdb-flux-raw-to-image-v1" and cut.execution["lane"] == lane
                and cut.execution["transition"] == record["number"]
                and cut.execution["receipt_kind"] == "component-derived StateCut, not Session/Act receipt", "component-derived state lineage differs")
        identities.setdefault("model", cut.model_identity)
        require(cut.model_identity == identities["model"], "adapter transformer identity changed across lanes")
        p = cut.payload
        require({key: p[key] for key in ("step", "phase", "cursor")} == record["clock"], "payload actual clock differs")
        for key, declared in record["state"].items():
            value = p[key]
            require(tensor_descriptor(value)["sha256"] == declared["sha256"]
                    and list(value.shape) == declared["shape"] and list(value.stride()) == declared["stride"]
                    and str(value.dtype) == declared["dtype"], "retained actual slot descriptor differs: " + key)
        reset = all(p[key] is None for key in WORKING)
        require(record["reset"] == (reset if p["phase"] == "project" else None), "actual field lifetime/reset sidecar differs")
        if config.get("profile") != "tiny-cpu":
            require(tuple(p["latent"].shape) == (1, 1024, 128) and tuple(p["conditioning"].shape) == (1, 512, 7680)
                    and str(p["latent"].dtype) == str(p["conditioning"].dtype) == "torch.bfloat16", "original latent/conditioning geometry differs")
            for key, shape in (("image", (1, 1024, 3072)), ("text", (1, 512, 3072)), ("noise_prediction", (1, 1024, 128))):
                require(p[key] is None or tuple(p[key].shape) == shape and str(p[key].dtype) == "torch.bfloat16", "original live-field geometry differs: " + key)
        return p
    return load, identities


def decoder_accounts(root, config, records, finals, helper, remaining, metadata):
    import numpy as np
    from PIL import Image
    bn = remaining["small_checkpoint_tensors"](read(metadata["vae_manifest"]), config["vae_parameter_root"], ("bn.running_mean", "bn.running_var"))
    spec = read(metadata["vae_config"])
    values, operands = {}, {}
    for lane, record in records.items():
        source = finals["native" if lane == "native-reference" else "forecast"]
        expected = remaining["decoder_preinput"](source, bn, spec, config["height"], config["width"])
        operand = load_array(root, record["decoder_input"])
        require(bytes_equal(expected, operand), "VAE before-post-quant operand is not own committed latent/IDs/BN/unpatch")
        require(record["decoder_input"]["device_at_capture"] == "cpu", "decoder CPU-upstream profile differs")
        decoded, pixels = (load_array(root, record[key]) for key in ("decoded", "pixels"))
        image = np.clip(helper["rne_bf16"](helper["rne_bf16"](decoded * np.float32(.5)) + np.float32(.5)), 0, 1).transpose(0, 2, 3, 1)
        require(bytes_equal(np.round(image[0] * 255).astype(np.uint8), pixels), "decoded raw output to pixel rounding differs")
        png = member(root, record["png"]["path"])
        require(png.stat().st_size == record["png"]["bytes"] and sha(png) == record["png"]["sha256"]
                and bytes_equal(np.asarray(Image.open(png).convert("RGB")), pixels), "actual pixels/original PNG coupling differs")
        values[lane] = dict(decoded=decoded, pixels=pixels)
        operands[lane] = dict(before_post_quant=record["decoder_input"])
        if lane != "forecast":
            require(record["operand_boundary"] == "after_post_quant_conv_before_decoder", "actual decoder entry boundary differs")
            require(bytes_equal(operand, load_array(root, record["actual_before_post_quant"])), "actual native post-quant module did not consume declared own operand")
            operands[lane]["actual_before_post_quant"] = record["actual_before_post_quant"]
            operands[lane]["actual_after_post_quant"] = record["actual_after_post_quant"]
            load_array(root, record["actual_after_post_quant"])
    require(bytes_equal(load_array(root, records["forecast"]["decoder_input"]), load_array(root, records["native-own"]["decoder_input"])), "same-latent decoder branch did not consume same pre-quant bytes")
    post = records["forecast"]["macros"]["post_quant_conv"]
    after = records["native-own"]["actual_after_post_quant"]
    same_post_quant = metric(load_array(root, post), load_array(root, after))
    comparisons = {field: image_account(values["forecast"][field], values["native-own"][field], values["native-reference"][field]) for field in ("decoded", "pixels")}
    sites = {"same_latent_post_quant": same_post_quant, **{name: row["terms"]["functional_minus_same_forecast_latent_native_VAE"] for name, row in comparisons.items()}}
    first_retained = next((name for name, value in sites.items() if not value["bitwise_exact"]), None)
    macro_descriptors = {name: dict(record=record, complete_array=True, raw_layout_retained="stride" in record) for name, record in records["forecast"]["macros"].items()}
    for record in records["forecast"]["macros"].values():
        load_array(root, record)
    return dict(fields=comparisons, actual_operands=operands, same_latent_post_quant_comparison=same_post_quant,
        forecast_macros=macro_descriptors, first_retained_same_latent_discrepancy=first_retained,
        coverage="Complete functional macro arrays. Native consumers retain argument, actual after-post-quant decoder entry, decoded and pixel fields; no unrecorded native intermediate macro norms claimed.",
        consumer_scope="Native VAE on forecast terminal latent and separate native-reference terminal latent. Explicit functional decoder forecast uses shared original primitives, not independently reconstructed kernel arithmetic."), values


def chronology_audit(events, summary):
    positions = lambda kind: [(index, row) for index, row in enumerate(events) if row["kind"] == kind]
    seals = positions("forecast_globally_sealed")
    require(len(seals) == 1, "unique global full-program seal missing")
    index, seal = seals[0]
    require(seal["native_targets"] == 0 and seal["seal_sha256"] == summary["forecast_seal"]["seal_sha256"], "global target-free seal differs")
    forecasts = [positions(kind) for kind in ("qwen_forecast_complete", "flux_forecast_complete", "decoder_forecast_complete")]
    native = [positions(kind) for kind in ("qwen_native_complete", "flux_native_complete", "native_consumers_complete")]
    require(all(len(rows) == 1 for rows in forecasts + native)
            and forecasts[0][0][0] < forecasts[1][0][0] < forecasts[2][0][0] < index < native[0][0][0] < native[1][0][0] < native[2][0][0], "whole-program forecast/reference chronology differs")
    require(forecasts[0][0][1]["native_forwards"] == 0 and forecasts[1][0][1]["transitions"] == native[1][0][1]["transitions"] == summary["counts"]["native_transitions"], "forecaster/reference transition census differs")
    placements = positions("component_placement")
    require(len(placements) == 12, "six placements and six offloads required")
    hydrations = [positions(kind) for kind in ("qwen_hydrated_cpu", "flux_hydrated_cpu", "vae_hydrated_cpu")]
    require(all(len(rows) == 1 for rows in hydrations) and all(rows[0][0] < placements[0][0] for rows in hydrations), "three one-time CPU hydration events differ")
    expected = [(component, direction) for component in ("qwen", "flux", "vae", "qwen", "flux", "vae") for direction in ("cuda", "cpu")]
    require([(row["component"], row["direction"]) for _, row in placements] == expected
            and all(row["identity_preserved"] is True for _, row in placements)
            and placements[5][0] < index < placements[6][0], "one-GPU phase placement order differs")
    return dict(seal_event=seal, forecast_events=[row[0][1] for row in forecasts], native_events=[row[0][1] for row in native],
        placements=[row for _, row in placements], hydrations=[row[0][1] for row in hydrations],
        placement_identity_scope="Source-bound per-placement checks and recorded outcome; final CPU tensor identities independently joined to all recipes. Per-device tensor arrays are not reloaded by this saved-only audit.")


def audit(root, output, *, expected_manifest, expected_request):
    from mdb_lanes.equation_comparison import EquationCatalog
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction
    root, output = Path(root), Path(output)
    h = helpers()
    a, q, remaining = (h[name] for name in ("audit_flux_functional_pass.py", "audit_flux_functional_qwen_prefix.py", "audit_flux_functional_remaining.py"))
    config, source, summary, contract, inventory = authenticate(root, expected_manifest=expected_manifest, expected_request=expected_request)
    metadata = source_metadata(root, config, source)
    restoration = parameter_audit(root, config, source, contract, metadata, a)
    runtime = runtime_audit(config, read(root / "runtime-contract.json"))
    require(sha(root / "runtime-contract.json") == source["runtime_contract_sha256"], "sealed runtime profile differs")
    tx = root / "forecast-transaction"
    verification = verify_transaction(tx, expected_seal_sha256=summary["forecast_seal"]["seal_sha256"],
        expected_event_head=summary["transaction"]["event_head"], max_bytes=config["forecast_transaction_bytes"])
    require(verification["phase"] == "closed", "global transaction is not closed")
    decode = lambda value: plain(load_forecast_tree(tx, value))
    record = read(tx / "forecasts/raw-composition-forecast.json")
    prediction = decode(record["predictions"])
    f, n = (read(root / (lane + "-qwen.json")) for lane in ("forecast", "native"))
    front = {lane: read(root / ("frontend-" + lane + ".json")) for lane in ("forecast", "native")}
    decoders = {lane: read(root / ("decoder-" + lane + ".json")) for lane in ("forecast", "native-own", "native-reference")}
    require(prediction["prompt"] == PROMPT and prediction["seed"] == 277 and prediction["native_targets_at_seal"] == 0
            and prediction["qwen"] == f and prediction["frontend"] == front["forecast"] and prediction["decoder"] == decoders["forecast"], "sealed full-program sidecars differ")
    external = prediction["external_files"]
    for name, item in external.items():
        path = member(root, name)
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "sealed external forecast evidence differs")
        require(not name.startswith(("native-state-store/", "qwen/native/", "decoder/native-")), "future native evidence in forecast frontier")
    for name in ("forecast-qwen.json", "frontend-forecast.json", "forecast-state-rows.json", "decoder-forecast.json", "runtime-contract.json", "plan.json", "request.json"):
        require(name in external, "complete raw-input/forecast dependency missing from seal: " + name)
    observation = decode(read(tx / "observations/native_qwen_flux_and_vae_targets.json")["value"])
    require(observation["native_qwen"] == n and observation["native_decoder_own"] == decoders["native-own"]
            and observation["native_decoder_reference"] == decoders["native-reference"], "closed native observation/sidecar differs")
    roles = q["parameter_roles"](read(metadata["qwen_manifest"]), read(metadata["qwen_config"]), config.get("prefix_layers", 27))
    if config.get("profile") != "tiny-cpu":
        require(config.get("prefix_layers", 27) == 27 and config.get("selected_hidden_states", [9, 18, 27]) == [9, 18, 27]
                and roles["counts"] == dict(all=398, forecast=298, reference_only=100)
                and roles["hidden_width"] == 2560 and roles["query_projection_width"] == 4096
                and config["height"] == config["width"] == config["max_length"] == 512, "original complete production geometry differs")
    mask, parser = parser_audit(root, config, source, f, n)
    qwen = qwen_accounts(root, config, f, n, mask, q, hidden_width=roles["hidden_width"])
    states = {lane: read(root / (lane + "-state-rows.json")) for lane in ("forecast", "native")}
    joints, singles = config.get("joint_layers", 5), config.get("single_layers", 20)
    expected_count = 4 * (joints + singles + 3)
    if config.get("profile") != "tiny-cpu":
        require((joints, singles, expected_count) == (5, 20, 112), "original denoiser program differs")
    require(summary["counts"]["forecast_transitions"] == summary["counts"]["native_transitions"] == expected_count
            and len(states["forecast"]) == len(states["native"]) == expected_count + 1
            and prediction["denoiser"] == dict(rows="forecast-state-rows.json", transitions=expected_count), "full112 own-state coverage differs")
    load, adapter_identity = state_loader(root, a, config)
    origins = {lane: load(rows[0], lane) for lane, rows in states.items()}
    for lane, row in (("forecast", f), ("native", n)):
        require(bytes_equal(field_to_numpy(origins[lane]["conditioning"]), load_array(root, row["conditioning"])), "initial denoiser refreshed own Qwen E")
    frontend = frontend_audit(root, config, front, origins)
    trajectory = trajectory_accounts(states["forecast"][1:], states["native"][1:], {lane: rows[0] for lane, rows in states.items()}, load,
        joints=joints, singles=singles, steps=4)
    finals = {lane: load(rows[-1], lane) for lane, rows in states.items()}
    decoder, endpoints = decoder_accounts(root, config, decoders, finals, a, remaining, metadata)
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    chronology = chronology_audit(events, summary)
    result = dict(schema="mdb-flux-raw-to-image-audit-v1", job_id=source["job_id"], native_request_sha256=expected_request,
        native_manifest_sha256=expected_manifest, worker_sha256=WORKER_SHA256, auditor_sha256=sha(__file__), source=source,
        source_model_identity="Original checkpoint recipes and implementation source cohort, not interchangeable with adapter structural model ID.",
        transformer_adapter_identity=adapter_identity, parameter_restoration=restoration, qwen_parameter_roles=roles, runtime_contract=runtime, parser=parser,
        transaction_verification=verification, counts=summary["counts"], native_contract=contract, chronology=chronology,
        qwen=qwen, frontend=frontend, denoiser=trajectory, decoder=decoder,
        comparison_census=dict(qwen_primary_fields=len(qwen["fields"]),
            denoiser_primary_fields=sum(len(row["fields"]) for row in trajectory["boundaries"]),
            whole_decoded_and_pixel_fields=2,
            primary_scalar_positions=sum(row["full"]["sample_count"] for row in qwen["fields"].values())
                + sum(value["sample_count"] for row in trajectory["boundaries"] for value in row["fields"].values())
                + sum(row["terms"]["whole_functional_minus_native"]["sample_count"] for row in decoder["fields"].values()),
            dependent_same_latent_and_inherited_consumer_endpoint_fields=4,
            same_latent_post_quant_fields=1, frontend_fields=6, generator_state_fields=2,
            source_euler_equation_diagnostics=2 * config["steps"],
            scope="Counts overlap/repeat values across layers and carried fields; conditional consumer comparisons and row diagnostics are dependent, never additional independent histories."),
        coverage=dict(qwen_full_hidden_states=len(f["hidden_states"]), full_denoiser_transitions=expected_count,
            same_latent_native_consumers=1, native_reference_consumers=1, raw_arrays_remain_remote=True,
            stride_scope="Original captured strides retained in array/cut records; CPU loading does not independently recover selected device kernels.",
            primitive_scope="Only retained source/full fields and macro arrays; no unrecorded micro/full native submodule census inferred.",
            actual_consumer_scope="Qwen IDs/masks and complete denoiser input/state closures joined; actual native VAE decoder prehooks capture after-post-quant operands. No unrecorded x_embedder/context_embedder prehook claim."),
        mechanics_status="authenticated-saved-raw-to-image-audit", trend_status="observed", terminal_status="unassessed",
        new_neural_forward_calls=0, new_learned_recurrence_calls=0,
        scope=f"One fixed raw prompt/seed, globally sealed Qwen-prefix/{expected_count}denoiser transitions/explicit decoder composition using shared original Torch primitives; no universal, semantic, independent CUDA arithmetic or conventional all-CUDA pipeline claim.")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    save(output / "array-index.json", dict(qwen=dict(forecast=f, native=n), frontend=front, denoiser=states, decoder=decoders))
    catalog = EquationCatalog(output / "catalog/equations")
    contexts = []
    for name, p, observed, scope in (
        ("whole", endpoints["forecast"], endpoints["native-reference"], "Whole globally sealed raw program versus independent native reference."),
        ("same-latent-decoder", endpoints["forecast"], endpoints["native-own"], "Conditional same-forecast-latent native VAE consumer; dependent diagnostic.")):
        context = dict(occurrence=name, prompt_utf8_sha256=parser["prompt_utf8_sha256"], seed=277,
            mask_identity=parser["attention_mask_sha256"], clock=dict(step=0, phase="project", cursor=0),
            parameter_manifests={key: config[key + "_manifest_sha256"] for key in ("qwen", "flux", "vae")},
            actual_noise_identity=front["forecast"]["packed_noise"]["content_sha256"],
            request_sha256=expected_request, frontier="raw prompt plus declared seed before any native learned target", horizon=dict(qwen_prefix=config.get("prefix_layers", 27), denoiser_transitions=expected_count, functional_decoder=1),
            forecast_seal=summary["forecast_seal"]["seal_sha256"], numeric_profile_sha256=source["runtime_contract_sha256"])
        catalog_record = catalog.record(law_id="law.flux.raw-prompt-own-state-composition.v1", route_id="route.flux.shared-torch-primitives.raw-to-image.v1", alias="cmp.flux.raw-to-image." + name,
            binding=dict(model_identity="original-qwen-flux-vae-checkpoint-recipes", execution=dict(id="mdb-flux-raw-to-image-v1", dtype="torch.bfloat16", cpu_decoder_frontend=True),
                input_identity=parser["input_ids_sha256"], source_identity=expected_request, frontier=context["frontier"], horizon=context["horizon"]),
            report=catalog_comparison(p, observed), conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], tested_context=context,
            readers=["native independent full reference", "same-latent native VAE"], evidence=dict(audit_sha256=sha(output / "audit.json"), scope=scope, dependent_paths=1, independent_kernel_arithmetic=False))
        exact = len(catalog.find(applicable_to=context))
        wrong = len(catalog.find(applicable_to={**context, "seed": 278}))
        wrong_clock = len(catalog.find(applicable_to={**context, "clock": dict(step=1, phase="project", cursor=0)}))
        missing = len(catalog.find(applicable_to={key: value for key, value in context.items() if key != "prompt_utf8_sha256"}))
        require(exact == 1 and wrong == wrong_clock == missing == 0, "exact-context catalog applicability differs")
        contexts.append(dict(occurrence=name, fingerprint=catalog_record["fingerprint"], exact=exact, changed_seed=wrong, changed_clock=wrong_clock, missing_input=missing))
    save(output / "catalog/provenance.json", dict(records=2, queries=contexts, dependent_paths=1, family_induction=False))
    return result, inventory


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged audit source differs")
    root, started = Path(config["native_root"]), time.monotonic()
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json", "files.json", "runtime-contract.json", "timeline.jsonl",
                 "forecast-qwen.json", "native-qwen.json", "frontend-forecast.json", "frontend-native.json", "forecast-state-rows.json", "native-state-rows.json",
                 "decoder-forecast.json", "decoder-native-own.json", "decoder-native-reference.json", "comparisons.json"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for key, row in result["source"]["source_files"].items():
        target = output / "source-metadata" / (key + "-" + Path(row["path"]).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row["path"], target)
    for lane, name in (("forecast", "decoder-forecast.json"), ("native-own", "decoder-native-own.json"), ("native-reference", "decoder-native-reference.json")):
        shutil.copyfile(root / read(root / name)["png"]["path"], output / (lane + ".png"))
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots=dict(native=str(root)), manifests=dict(native={name: row["sha256"] for name, row in inventory["files"].items()}),
        scope="All raw full states/arrays remain near data, individually retrievable by original checksums; compact contains scalar accounts, metadata and original three PNGs."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-flux-raw-to-image-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], source_job=result["job_id"],
        records=2, model_calls=0, seconds=time.monotonic() - started, mechanics_status="verified-saved-evidence"))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output),
        files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        launch = runpy.run_path(str(HERE / "stage_saved_flux_audit.py"))
        print(json.dumps(launch["submit"](args.run, worker=Path(__file__)) if args.submit else launch["stage"](args.run, args.study, worker=Path(__file__), helpers=HELPERS,
            experiment="mdb-flux-raw-to-image-audit", operations=1, scope="One globally sealed raw-prompt/seed to Qwen/112denoiser/decoder path; full saved-array audit, exact recipes, no neural calls.")))
