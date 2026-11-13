"""Original BF16/F32 safetensor recipes for a CPU-FP32 Qwen suffix.

No parameter archives are written. Explicit shared-Parameter aliases resolve
tied heads; original payload and native FP32 conversion hashes remain distinct.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
from pathlib import Path

RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def header(path):
    with Path(path).open("rb") as stream:
        lead = stream.read(8); length = int.from_bytes(lead, "little")
        require(len(lead) == 8 and 0 < length <= 16 << 20, "bounded original safetensor header required")
        raw = lead + stream.read(length)
    require(len(raw) == length+8, "original safetensor header truncated")
    return json.loads(raw[8:]), dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def prepare(model, checkpoint, inventory, output, retriever):
    import numpy as np

    from mdb_lanes.parameter_artifacts import validate_manifest

    checkpoint, output, retriever = Path(checkpoint), Path(output), Path(retriever)
    require(hashlib.sha256(retriever.read_bytes()).hexdigest() == RETRIEVER_SHA256, "qualified standalone retriever changed")
    last = len(model.model.layers)-1
    names = dict(postnorm=f"model.layers.{last}.post_attention_layernorm.weight", gate=f"model.layers.{last}.mlp.gate_proj.weight",
        up=f"model.layers.{last}.mlp.up_proj.weight", down=f"model.layers.{last}.mlp.down_proj.weight", finalnorm="model.norm.weight", head="lm_head.weight")
    headers = {name: header(checkpoint / name) for name in inventory if name.endswith(".safetensors")}
    aliases = list(model.named_parameters(remove_duplicate=False))
    rows, bindings = [], {}
    for logical, requested in names.items():
        parameter = model.get_parameter(requested)
        native = parameter.detach().cpu().numpy()
        require(native.dtype == np.float32 and native.flags.c_contiguous, "actual CPU-FP32 contiguous native parameter required")
        native_sha = hashlib.sha256(memoryview(native).cast("B")).hexdigest()
        equivalent = sorted(name for name, value in aliases if value is parameter)
        matches = [(file, name, values[name]) for file, (values, _) in headers.items() for name in equivalent if name in values]
        require(bool(matches), "no original named checkpoint alias for " + requested)
        file, tensor, source = sorted(matches, key=lambda value: (value[1], value[0]))[0]
        require(source["dtype"] in ("BF16", "F32") and source["shape"] == list(native.shape), "original/native dtype or shape differs")
        dtype = "<u2" if source["dtype"] == "BF16" else "<f4"
        stream = io.BytesIO(); np.lib.format.write_array_header_1_0(stream, dict(descr=dtype, fortran_order=False, shape=tuple(source["shape"])))
        prefix = stream.getvalue(); full, raw_hash, converted = hashlib.sha256(prefix), hashlib.sha256(), hashlib.sha256()
        start, stop = source["data_offsets"]; remaining = stop-start
        require(remaining == native.size * (2 if source["dtype"] == "BF16" else 4), "original payload geometry differs")
        with (checkpoint / file).open("rb") as original:
            original.seek(headers[file][1]["bytes"]+start)
            while remaining:
                block = original.read(min(8 << 20, remaining)); require(bool(block), "original payload truncated")
                remaining -= len(block); full.update(block); raw_hash.update(block)
                if source["dtype"] == "BF16":
                    widened = np.frombuffer(block, dtype="<u2").astype("<u4") << 16
                    converted.update(memoryview(widened).cast("B"))
                else:
                    converted.update(block)
        require(converted.hexdigest() == native_sha, "original raw payload does not reproduce actual native FP32 parameter")
        row = dict(path="parameters/"+logical+".npy", bytes=len(prefix)+stop-start, sha256=full.hexdigest(), npy_prefix_b64=base64.b64encode(prefix).decode(),
            source=dict(file=file, file_sha256=inventory[file]["sha256"], tensor=tensor, dtype=source["dtype"], shape=source["shape"],
                data_offsets=source["data_offsets"], payload_sha256=raw_hash.hexdigest(), header_bytes=headers[file][1]["bytes"], header_sha256=headers[file][1]["sha256"]))
        rows.append(row)
        bindings[logical] = dict(native_name=requested, actual_shared_parameter_aliases=equivalent, original_tensor=tensor, original_file=file,
            original_dtype=source["dtype"], original_payload_sha256=raw_hash.hexdigest(), native_dtype="float32", native_shape=list(native.shape), native_sha256=native_sha,
            conversion="BF16 words shifted to FP32 bits" if source["dtype"] == "BF16" else "Original F32 bytes unchanged", parameter_archive_written=False)
    manifest = validate_manifest(dict(schema="mdb-parameter-artifacts-v1", artifacts=rows,
        defaults=dict(checkpoint_root=str(checkpoint)), provenance=dict(scope="Exact virtual original archives; no parameter bank written", retrieval_module_sha256=RETRIEVER_SHA256)))
    save(output / "parameter-artifacts.json", manifest)
    save(output / "checkpoint-headers.json", {name: dict(descriptor=description, selected={row["source"]["tensor"]: values[row["source"]["tensor"]] for row in rows if row["source"]["file"] == name}) for name, (values, description) in headers.items()})
    save(output / "weight-binding.json", bindings)
    shutil.copyfile(retriever, output / "retrieve_weights.py")
    return bindings
