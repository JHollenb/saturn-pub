"""Validate block-streamed residency for decoder LM adapters on real checkpoints.

For each checkpoint this measures that a *streamed* adapter -- frozen weights parked in host
memory, one native block copied to the device at a time -- reproduces the native decode, and
reports its cost. Two comparison modes:

``compare == "resident"`` (the model fits the device resident)
    Build a resident adapter and a streamed adapter from the same checkpoint and assert, over an
    N-token greedy decode, that the streamed run is *bitwise identical* to the resident run
    (same generated tokens and bit-equal final logits), and that the uninstrumented
    ``native_logits`` comparator matches bitwise. Records peak VRAM and tokens/s for each mode.

``compare == "reference"`` (the model does NOT fit the device resident)
    Build only a streamed adapter and reproduce an N-token greedy decode. The reference is HF
    Transformers ``AutoModelForCausalLM.generate`` with accelerate ``device_map`` CPU-offload,
    eager attention, the same checkpoint and dtype, greedy (``do_sample=False, num_beams=1``);
    the check is exact equality of the N generated token IDs. Records peak VRAM and tokens/s and
    the resident weight footprint that exceeds the card.

Both modes capture a mid-layer StateCut under streaming, save it to a ``LocalStore``, reload it
in a brand-new interpreter, restore, continue, and check the continuation token-for-token
(fresh-process replay). This file imports only ``saturn_pub`` and public frameworks; no
scheduler code lives here. Reproduce with plain Python:

    python experiments/lm_block_residency/run.py --checkpoints checkpoints.json \
        --device cuda --new-tokens 16 --output lm-residency.json
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from saturn_pub.adapters import load
from saturn_pub.store import LocalStore

PROMPT_TEXT = "The tower stands by the river and the city lights"
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

_FRESH = """
import json, sys, torch
from saturn_pub import Session
from saturn_pub.adapters import load
from saturn_pub.store import LocalStore
path, root, identifier, device, dtype, steps, expected = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]),
    json.loads(sys.argv[7]),
)
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
# Rebuild the SAME streamed adapter in a fresh interpreter: weights in host memory, one block
# copied to the device at a time. The cut carries the streamed execution contract.
adapter = load(path, residency="streamed", device=device, dtype=torch_dtype, local_files_only=True)
cut = LocalStore(root).load(identifier, device=device)
session = Session(adapter, cut.payload)
session.restore(cut)
adapter.generate(session, steps)
got = session.read("tokens")[0].tolist()[-steps:]
assert got == expected, (got, expected)
print("FRESH_OK")
"""


def _encode_prompt(path: str, max_len: int = 12) -> list[int]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        ids = tokenizer(PROMPT_TEXT)["input_ids"]
        if ids:
            return [int(token) for token in ids[:max_len]]
    except Exception:  # noqa: BLE001 - tokenizer is optional for a mechanics probe
        pass
    return list(range(10, 10 + max_len))


def _peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def _greedy_decode(adapter, prompt: list[int], new_tokens: int) -> tuple[list[int], torch.Tensor, float]:
    session = adapter.session(prompt)
    start = time.perf_counter()
    adapter.generate(session, new_tokens)
    decode_s = time.perf_counter() - start
    tokens = session.read("tokens")[0].tolist()[-new_tokens:]
    return tokens, session.read("logits"), decode_s


def _replay_prepare(workdir, label, adapter, prompt, steps):
    """Capture a mid-layer streamed cut, save it, and record the in-process continuation."""
    mid = adapter.session(prompt)
    mid.continue_(2)  # embed + first layer: a mid-stack carrier boundary
    cut = mid.capture()
    store = LocalStore(workdir / label)
    identifier = store.save(cut)
    branch = mid.fork(cut)
    adapter.generate(branch, steps)
    return identifier, branch.read("tokens")[0].tolist()[-steps:]


def _replay_verify(path, workdir, label, identifier, expected, device, dtype, steps) -> dict:
    """Rebuild the streamed adapter in a fresh interpreter, restore the cut, and compare."""
    proc = subprocess.run(
        [
            sys.executable, "-c", _FRESH, path, str(workdir / label), identifier,
            device, dtype, str(steps), json.dumps(expected),
        ],
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0 and proc.stdout.strip().endswith("FRESH_OK")
    out = {"fresh_process_replay_exact": ok}
    if not ok:
        out["fresh_process_error"] = proc.stderr.strip()[-400:]
    return out


def _accelerate_reference(path, prompt, torch_dtype, device, new_tokens, model_type) -> list[int]:
    """Full-model greedy decode on a device that cannot hold it, via accelerate CPU-offload."""
    kwargs = {"dtype": torch_dtype, "local_files_only": True}
    if model_type != "mamba":
        kwargs["attn_implementation"] = "eager"
    if device == "cuda":
        # Force an offload split so the reference genuinely streams layers to the card.
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "10GiB", "cpu": "44GiB"}
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    else:
        # No accelerator: a plain full-model forward on the target device is the reference.
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()
    tokens = torch.tensor([prompt], device=device)
    with torch.inference_mode():
        out = model.generate(
            tokens,
            attention_mask=torch.ones_like(tokens),
            max_new_tokens=new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    result = out[0].tolist()[-new_tokens:]
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def validate_one(entry, *, device, new_tokens, workdir) -> dict:
    path = entry["path"]
    dtype = entry.get("dtype", "float32")
    compare = entry.get("compare", "resident")
    torch_dtype = _DTYPES[dtype]
    model_type = AutoConfig.from_pretrained(path, local_files_only=True).model_type
    record = {
        "family": entry["family"],
        "label": entry["label"],
        "path": path,
        "dtype": dtype,
        "compare": compare,
        "model_type": model_type,
    }
    prompt = _encode_prompt(path)
    record["prompt_len"] = len(prompt)
    record["new_tokens"] = new_tokens
    start = time.perf_counter()

    if compare == "resident":
        # Resident run (the historical zero-overhead path).
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        resident = load(path, residency="resident", device=device, dtype=torch_dtype,
                        local_files_only=True)
        record["adapter"] = type(resident).__name__
        record["parity_contract"] = resident.execution["parity"]
        res_tokens, res_logits, res_decode_s = _greedy_decode(resident, prompt, new_tokens)
        res_native = resident.native_logits(prompt)
        record["resident_peak_vram_mb"] = _peak_vram_mb()
        record["resident_tokens_per_s"] = round(new_tokens / res_decode_s, 3)
        del resident
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Streamed run (frozen weights in host memory, one block to the device at a time).
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        streamed = load(path, residency="streamed", device=device, dtype=torch_dtype,
                        local_files_only=True)
        record["residency"] = streamed.execution["residency"]
        str_tokens, str_logits, str_decode_s = _greedy_decode(streamed, prompt, new_tokens)
        str_native = streamed.native_logits(prompt)
        record["streamed_peak_vram_mb"] = _peak_vram_mb()
        record["streamed_tokens_per_s"] = round(new_tokens / str_decode_s, 3)

        record["greedy_tokens_equal"] = str_tokens == res_tokens
        record["final_logits_bitwise_equal"] = bool(torch.equal(str_logits, res_logits))
        record["native_logits_bitwise_equal"] = bool(torch.equal(str_native, res_native))
        record["streamed_equals_resident_exact"] = (
            record["greedy_tokens_equal"]
            and record["final_logits_bitwise_equal"]
            and record["native_logits_bitwise_equal"]
        )
        record["max_abs_logit_delta_vs_resident"] = float((str_logits - res_logits).abs().max())
        if record["resident_peak_vram_mb"] and record["streamed_peak_vram_mb"]:
            record["vram_reduction_x"] = round(
                record["resident_peak_vram_mb"] / record["streamed_peak_vram_mb"], 2
            )
        identifier, expected = _replay_prepare(workdir, record["label"], streamed, prompt, 4)
        del streamed
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        record.update(
            _replay_verify(path, workdir, record["label"], identifier, expected, device, dtype, 4)
        )

    elif compare == "reference":
        # The model does not fit the card resident; reference is accelerate CPU-offload.
        ref_tokens = _accelerate_reference(path, prompt, torch_dtype, device, new_tokens, model_type)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        streamed = load(path, residency="streamed", device=device, dtype=torch_dtype,
                        local_files_only=True)
        record["adapter"] = type(streamed).__name__
        record["parity_contract"] = streamed.execution["parity"]
        record["residency"] = streamed.execution["residency"]
        weight_bytes = sum(p.numel() * p.element_size() for p in streamed.model.parameters())
        record["resident_weight_bytes"] = int(weight_bytes)
        record["resident_weight_gb"] = round(weight_bytes / 2**30, 2)
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            record["device_total_vram_gb"] = round(total / 2**30, 2)
            record["fits_resident"] = weight_bytes < total
        str_tokens, _, str_decode_s = _greedy_decode(streamed, prompt, new_tokens)
        record["streamed_peak_vram_mb"] = _peak_vram_mb()
        record["streamed_tokens_per_s"] = round(new_tokens / str_decode_s, 3)
        record["reference"] = (
            "HF AutoModelForCausalLM.generate, accelerate device_map CPU-offload "
            "(max_memory 10GiB GPU), eager attention, same checkpoint/dtype, greedy"
        )
        record["greedy_tokens_equal_reference"] = str_tokens == ref_tokens
        record["streamed_tokens"] = str_tokens
        record["reference_tokens"] = ref_tokens
        identifier, expected = _replay_prepare(workdir, record["label"], streamed, prompt, 2)
        del streamed
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        record.update(
            _replay_verify(path, workdir, record["label"], identifier, expected, device, dtype, 2)
        )
    else:
        raise ValueError(f"unknown compare mode: {compare!r}")

    record["wall_s"] = round(time.perf_counter() - start, 3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True,
                        help="JSON list of {family,label,path,dtype,compare}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--output", default="lm-residency.json")
    parser.add_argument("--workdir", default="lm-residency-state")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoints = json.loads(Path(args.checkpoints).read_text())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in checkpoints:
        try:
            record = validate_one(entry, device=args.device, new_tokens=args.new_tokens,
                                  workdir=workdir)
            results.append(record)
            print(f"OK {entry['label']}: {record}", flush=True)
            # Per-row marker so a wall-clock kill still yields the rows already finished.
            row_marker = base64.b64encode(json.dumps(record).encode()).decode()
            print(f"SATURN_PUB_LM_RESIDENCY_ROW={row_marker}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
            import traceback

            results.append({
                "family": entry.get("family"),
                "label": entry.get("label"),
                "path": entry.get("path"),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"FAIL {entry.get('label')}: {exc}", flush=True)
            traceback.print_exc()
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "schema": "saturn-pub-lm-block-residency-v1",
        "device": args.device,
        "new_tokens": args.new_tokens,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    marker = base64.b64encode(json.dumps(summary).encode()).decode()
    print(f"SATURN_PUB_LM_RESIDENCY={marker}", flush=True)


if __name__ == "__main__":
    main()
