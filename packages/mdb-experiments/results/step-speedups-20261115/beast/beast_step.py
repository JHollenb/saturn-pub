"""Real-model stepping bench, one source tree per process (main or branch on PYTHONPATH).

Qwen2.5-1.5B bf16 on CUDA. Prints one RESULT line per case with wall time, peak VRAM, peak RSS
and the digests that must match between trees: trail fingerprints, tokens, logits bytes,
port-read bytes.
  rows   : stepped rows at prefill=T-ROWS (the default procedure's generated rows)
  p0     : prefill=0 (every prompt row stepped) until the last row's logits, then held_bytes()
  ports  : one row walked port by port, lookahead none / layer (/ auto when the tree has it)
usage: python beast_step.py TREE_LABEL [smoke]
"""
import hashlib
import json
import os
import resource
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mdb.engine.run import LOOKAHEADS, Run
from saturn_pub.adapters.decoder import DecoderAdapter

LABEL = sys.argv[1]
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
MODEL = os.environ.get("MDB_MODEL", "/mnt/big/llm-models/Qwen2.5-1.5B")
torch.manual_seed(0)
t0 = time.monotonic()
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager",
                                             local_files_only=True).to("cuda").eval()
ad = DecoderAdapter(model, granularity="operation")
print(f"[{LABEL}] loaded {time.monotonic() - t0:.1f}s layers {ad.layers} device {ad.device}", flush=True)
TEXT = ("The history of the city begins with a small settlement on the river, where traders "
        "met to exchange salt, grain and cloth. Over the centuries the market grew into a town, "
        "the town into a city, and the city into the capital of a kingdom. ")
IDS = tok(TEXT * 40)["input_ids"]


def sha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy()).hexdigest()[:16]


def trail(run):
    return hashlib.sha256("".join(m.fingerprint for m in run._trail).encode()).hexdigest()[:16]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # linux: KB


def case(name, fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    out = fn()
    torch.cuda.synchronize()
    out.update({"case": name, "tree": LABEL, "s": round(time.monotonic() - start, 3),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_rss_mb": round(rss_mb(), 1)})
    print("RESULT " + json.dumps(out, sort_keys=True), flush=True)


def rows_case(T, rows):
    def fn():
        tokens = IDS[:T]
        run = Run(ad, tokens, prefill=T - rows)
        times = []
        for _ in range(rows):
            torch.cuda.synchronize()
            s = time.monotonic()
            run.step("token")
            torch.cuda.synchronize()
            times.append(time.monotonic() - s)
        return {"T": T, "rows": rows, "ms_per_row": [round(x * 1e3, 1) for x in times],
                "trail": trail(run), "tokens": run.session.read("tokens")[0, -rows - 1:].tolist(),
                "logits": [sha(run.read(f"t{T - rows + i}.logits")) for i in range(rows)]}
    return fn


def p0_case(T):
    def fn():
        run = Run(ad, IDS[:T], prefill=0)
        s = time.monotonic()
        run.until(f"t{T - 1}.logits")
        torch.cuda.synchronize()
        until = time.monotonic() - s
        s = time.monotonic()
        held = run.held_bytes()
        return {"T": T, "until_s": round(until, 2), "held_bytes_s": round(time.monotonic() - s, 3),
                "held_total": held["total"], "marks": held["marks"], "trail": trail(run),
                "logits": [sha(run.read(f"t{r}.logits")) for r in range(T)]}
    return fn


def ports_case(T, mode):
    def fn():
        run = Run(ad, IDS[:T], lookahead=mode)
        reads = []
        s = time.monotonic()
        while run.where != f"t{T}.start":
            run.step("port")
            if run._port is not None:
                reads.append(sha(run.read()))
        torch.cuda.synchronize()
        return {"T": T, "lookahead": mode, "row_s": round(time.monotonic() - s, 2),
                "calls": dict(run.calls), "trail": trail(run),
                "reads": hashlib.sha256("".join(reads).encode()).hexdigest()[:16],
                "n_reads": len(reads)}
    return fn


if SMOKE:
    case("rows", rows_case(64, 1))
    case("p0", p0_case(6))
    case("ports", ports_case(32, "none"))
else:
    case("rows", rows_case(1024, 4))
    case("rows", rows_case(256, 4))
    case("p0", p0_case(32))
    for mode in LOOKAHEADS:
        case("ports", ports_case(256, mode))
print(f"[{LABEL}] done {time.monotonic() - t0:.1f}s", flush=True)
