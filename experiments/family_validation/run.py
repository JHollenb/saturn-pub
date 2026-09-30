"""Measure native-adapter fidelity for AR families on real checkpoints.

For each checkpoint this loads the native model, wraps it with ``saturn_pub.adapters.load``,
and records four measurements plus cost:

* ``max_abs_logit_delta`` -- max |stepped logits - native full-forward logits| on a short prompt.
* ``greedy_agreement`` -- fraction of a 16-token greedy continuation that matches the native
  ``model.generate`` greedy decode.
* ``fresh_process_replay_exact`` -- a mid-layer StateCut saved to a LocalStore, reloaded in a
  brand new interpreter, restored, continued, and checked token-for-token against the in-process
  continuation.
* ``wall_s`` and ``peak_vram_mb`` -- cost of one checkpoint's measurement.

It imports only ``saturn_pub`` and public frameworks. No scheduler code lives here.
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
torch.set_num_threads(max(1, torch.get_num_threads()))
from transformers import AutoConfig, AutoModelForCausalLM
from saturn_pub import Session
from saturn_pub.adapters import load
from saturn_pub.store import LocalStore
path, root, identifier, device, dtype, steps, expected = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]),
    json.loads(sys.argv[7]),
)
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
model_type = AutoConfig.from_pretrained(path, local_files_only=True).model_type
kwargs = dict(dtype=torch_dtype, local_files_only=True)
if model_type != "mamba":
    kwargs["attn_implementation"] = "eager"
model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()
adapter = load(model)
cut = LocalStore(root).load(identifier, device=device)
session = Session(adapter, cut.payload)
session.restore(cut)
adapter.generate(session, steps)
got = session.read("tokens")[0].tolist()[-steps:]
assert got == expected, (got, expected)
print("FRESH_OK")
"""


def _encode_prompt(path: str, model_type: str, max_len: int = 12) -> list[int]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        ids = tokenizer(PROMPT_TEXT)["input_ids"]
        if ids:
            return [int(token) for token in ids[:max_len]]
    except Exception:  # noqa: BLE001 - tokenizer is optional for a mechanics probe
        pass
    return list(range(10, 10 + max_len))


def _native_greedy(model, prompt: list[int], device: str, new_tokens: int) -> list[int]:
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
    return out[0].tolist()


def validate_one(entry: dict, *, device: str, dtype: str, new_tokens: int, workdir: Path) -> dict:
    path = entry["path"]
    record: dict = {"family": entry["family"], "label": entry["label"], "path": path}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    torch_dtype = _DTYPES[dtype]
    model_type = AutoConfig.from_pretrained(path, local_files_only=True).model_type
    record["model_type"] = model_type
    kwargs = {"dtype": torch_dtype, "local_files_only": True}
    if model_type != "mamba":
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()
    adapter = load(model)
    record["adapter"] = type(adapter).__name__
    record["parity_contract"] = adapter.execution["parity"]

    prompt = _encode_prompt(path, model_type)
    record["prompt_len"] = len(prompt)

    # 1. stepped vs native logits
    session = adapter.session(prompt)
    session.continue_(adapter.layers + 2)
    stepped = session.read("logits")
    native = adapter.native_logits(prompt)
    record["max_abs_logit_delta"] = float((stepped - native).abs().max().item())

    # 2. greedy agreement over new_tokens
    reference = _native_greedy(model, prompt, device, new_tokens)[-new_tokens:]
    greedy = adapter.session(prompt)
    adapter.generate(greedy, new_tokens)
    mine = greedy.read("tokens")[0].tolist()[-new_tokens:]
    record["greedy_agreement"] = sum(a == b for a, b in zip(mine, reference)) / new_tokens
    record["greedy_exact"] = mine == reference

    # 3. fresh-process replay from a mid-layer cut
    mid = adapter.session(prompt)
    mid.continue_(2)  # embed + first layer: a mid-stack carrier boundary
    cut = mid.capture()
    store = LocalStore(workdir / record["label"])
    identifier = store.save(cut)
    branch = mid.fork(cut)
    adapter.generate(branch, 8)
    expected = branch.read("tokens")[0].tolist()[-8:]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _FRESH,
            path,
            str(workdir / record["label"]),
            identifier,
            device,
            dtype,
            "8",
            json.dumps(expected),
        ],
        capture_output=True,
        text=True,
    )
    record["fresh_process_replay_exact"] = proc.returncode == 0 and proc.stdout.strip().endswith(
        "FRESH_OK"
    )
    if not record["fresh_process_replay_exact"]:
        record["fresh_process_error"] = proc.stderr.strip()[-400:]

    record["wall_s"] = round(time.perf_counter() - start, 3)
    record["peak_vram_mb"] = (
        round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None
    )

    del model, adapter
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="JSON list of {family,label,path}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=sorted(_DTYPES))
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--output", default="family-validation.json")
    parser.add_argument("--workdir", default="family-validation-state")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoints = json.loads(Path(args.checkpoints).read_text())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in checkpoints:
        try:
            results.append(
                validate_one(
                    entry,
                    device=args.device,
                    dtype=args.dtype,
                    new_tokens=args.new_tokens,
                    workdir=workdir,
                )
            )
            print(f"OK {entry['label']}: {results[-1]}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
            import traceback

            results.append(
                {
                    "family": entry.get("family"),
                    "label": entry.get("label"),
                    "path": entry.get("path"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"FAIL {entry.get('label')}: {exc}", flush=True)
            traceback.print_exc()
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "schema": "saturn-pub-family-validation-v1",
        "device": args.device,
        "dtype": args.dtype,
        "new_tokens": args.new_tokens,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    marker = base64.b64encode(json.dumps(summary).encode()).decode()
    print(f"SATURN_PUB_FAMILY_VALIDATION={marker}", flush=True)


if __name__ == "__main__":
    main()
