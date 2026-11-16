"""#23a: what share of an mdb Qwen bench decode is durable publication, on ext4 NVMe today?

usage: python bench23a.py <out.json> [tokens]
Ships saturn_pub (./pub) and mdb + mdb_lanes (./mdbsrc) in the payload (mdb_lanes since the 9ec645b split); model Qwen2.5-0.5B-Instruct BF16 CUDA,
virtual lane, root under /mnt/big/agent-work/miss-path/bench23/<job>/ (NVMe; never the HDD).
Arms, same process, same prompt, each on a fresh root:
  warmup   4 tokens (Triton/JIT warm, discarded)
  durable  N tokens, default journal publication, os.fsync wrapped to count and time it
  nofsync  N tokens with os.fsync a no-op: an upper bound on what async publication could save
  profile  N tokens durable under cProfile (top functions by own time)
  command  N tokens with command_publication=True (only with --command)
  stock    N-token HF greedy decode with the same model (consumer floor; different lane)
"""
from __future__ import annotations

import cProfile
import hashlib
import io
import json
import os
import pstats
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "pub"), os.path.join(HERE, "mdbsrc")]

import torch  # noqa: E402

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
JOB = os.environ.get("MRUN_JOB_ID", "local")
BASE = f"/mnt/big/agent-work/miss-path/bench23/{JOB}"
PROMPT = "The capital of France is"
R: dict = {"job": JOB, "tokens": N, "torch": torch.__version__}

import transformers  # noqa: E402

R["transformers"] = transformers.__version__
from mdb_lanes.bench import BenchSession  # noqa: E402
from mdb_lanes.bench_loading import load_bench_model  # noqa: E402

R["mdb_file"] = sys.modules["mdb_lanes.bench"].__file__
torch.backends.cuda.matmul.allow_tf32 = False
t = time.perf_counter()
spec, model, tok = load_bench_model("qwen2.5-0.5b-instruct")
R["load_s"] = time.perf_counter() - t

_real_fsync = os.fsync
FS = {"count": 0, "seconds": 0.0}


def counting_fsync(fd):
    t0 = time.perf_counter()
    _real_fsync(fd)
    FS["count"] += 1
    FS["seconds"] += time.perf_counter() - t0


def dump():
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1, sort_keys=True, default=str)


def run_arm(name, tokens, *, fsync=None, profile=False, command=False):
    os.fsync = fsync or counting_fsync
    FS.update(count=0, seconds=0.0)
    root = f"{BASE}/{name}"
    try:
        extra = {"command_publication": True} if command else {}
        b = BenchSession.from_resident(model, tok, root=root, session_id=f"{JOB}-{name}",
                                       continuation_capacity=64, **extra)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b.prefill(PROMPT)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        FS_pre = dict(FS)
        prof = cProfile.Profile() if profile else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if prof:
            prof.enable()
        rows = b.generate(tokens)
        if prof:
            prof.disable()
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        perf = b.inspect("performance")
        out = {"prefill_s": prefill_s, "generate_s": gen_s, "per_token_ms": gen_s / tokens * 1e3,
               "fsync_count_generate": FS["count"] - FS_pre["count"],
               "fsync_seconds_generate": FS["seconds"] - FS_pre["seconds"],
               "token_ids": [r["token_id"] for r in rows],
               "logits_digest": hashlib.sha256("".join(r["logits_sha256"] for r in rows).encode()).hexdigest(),
               "publication": perf.get("publication"), "metrics": {k: v for k, v in perf.items()
                                                                   if k not in ("publication",)}}
        if prof:
            s = io.StringIO()
            st = pstats.Stats(prof, stream=s)
            st.sort_stats("tottime").print_stats(30)
            out["profile_tottime"] = s.getvalue()[-9000:]
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(45)
            out["profile_cumulative"] = s.getvalue()[-12000:]
        b.close()
        files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        out["root_files"] = len(files)
        out["root_bytes"] = sum(os.path.getsize(f) for f in files)
        return out
    finally:
        os.fsync = _real_fsync


def stock(tokens):
    ids = torch.tensor([tok.encode(PROMPT)], device="cuda")
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model(ids, use_cache=True)
        out = []
        for _ in range(tokens):
            nxt = int(res.logits[:, -1].argmax(-1))
            out.append(nxt)
            res = model(torch.tensor([[nxt]], device="cuda"), past_key_values=res.past_key_values,
                        use_cache=True)
        torch.cuda.synchronize()
    return {"seconds_incl_prefill": time.perf_counter() - t0, "token_ids": out}


for name, kw in (("warmup", {"tokens": 4}), ("durable", {"tokens": N}),
                 ("nofsync", {"tokens": N, "fsync": lambda fd: None}),
                 ("profile", {"tokens": N, "profile": True}),
                 *((("command", {"tokens": N, "command": True}),) if "--command" in sys.argv else ())):
    try:
        R[name] = run_arm(name, **kw)
    except Exception as exc:  # keep partial results
        import traceback
        traceback.print_exc()
        R[f"{name}_error"] = repr(exc)
    dump()
R["stock"] = stock(N)
R["stock_repeat"] = stock(N)
R["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
dump()
for k in ("durable", "nofsync", "profile", "command"):
    if k in R:
        r = R[k]
        print(k, round(r["generate_s"], 3), "s", round(r["per_token_ms"], 1), "ms/tok fsync",
              r["fsync_count_generate"], round(r["fsync_seconds_generate"], 3), r["logits_digest"][:12])
print("stock", R["stock"]["seconds_incl_prefill"], R["stock_repeat"]["seconds_incl_prefill"])
