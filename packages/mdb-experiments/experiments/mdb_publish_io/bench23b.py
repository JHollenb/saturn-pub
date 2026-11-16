"""#23b: BlockResidency streamed decode rate for a model that does not fit the card.

usage: python bench23b.py <out.json> <arms> [tokens]
  arms: comma list of  pageable | pinned | pinned+prefetch | pinned+prefetch+hostemb | pinned+hostemb
        (pinning is one-way: pageable first)
Ships saturn_pub as ./pub. Qwen3-8B BF16 (15.3 GB), eager attention, greedy, CUDA.
Each arm builds a new streamed DecoderAdapter over the SAME host model, opens a session on a
12-token prompt (streamed prefill), then decodes N tokens. Output tokens and final-logit bytes
are hashed so arms can be compared bit for bit.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pub"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402

OUT = sys.argv[1]
ARMS = sys.argv[2].split(",")
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
PATH = "/mnt/big/llm-models/Qwen3-8B"
R: dict = {"job": os.environ.get("MRUN_JOB_ID"), "tokens": N, "torch": torch.__version__,
           "realpath": os.path.realpath(PATH), "arms": {}}


def rss_mb():
    with open("/proc/self/status") as fh:
        return {l.split(":")[0]: int(l.split()[1]) / 1024 for l in fh if l.startswith(("VmRSS", "VmHWM"))}


def dump():
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1, sort_keys=True)


torch.backends.cuda.matmul.allow_tf32 = False
t = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, attn_implementation="eager",
                                             local_files_only=True).eval()
R["load_s"] = time.perf_counter() - t
R["weight_gb"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
R["rss_after_load"] = rss_mb()
tok = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
prompt = tok("The tower stands by the river and the city lights")["input_ids"][:12]
R["prompt"] = prompt

for arm in ARMS:
    kw = {"residency": "streamed", "device": "cuda", "pin_host": arm.startswith("pinned")}
    if "prefetch" in arm:
        kw["prefetch"] = True
    if "hostemb" in arm:
        kw["host_embedding"] = True
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    adapter = DecoderAdapter(model, **kw)
    place_s = time.perf_counter() - t
    torch.cuda.synchronize()
    t = time.perf_counter()
    session = adapter.session(prompt)
    torch.cuda.synchronize()
    session_s = time.perf_counter() - t
    adapter.generate(session, 1)  # first decode token: allocator/kernel warm
    torch.cuda.synchronize()
    t = time.perf_counter()
    adapter.generate(session, N)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t
    toks = session.read("tokens")[0].tolist()[len(prompt):]
    logits = session.read("logits")
    R["arms"][arm] = {
        "place_s": place_s, "session_prefill_s": session_s, "decode_s": decode_s,
        "decode_tok_s": N / decode_s, "token_ids": toks,
        "logits_sha256": hashlib.sha256(logits.float().cpu().numpy().tobytes()).hexdigest(),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20, "rss": rss_mb(),
        "residency": adapter.execution.get("residency"),
        "residency_stats": dict(getattr(adapter._residency, "stats", {}) or {}),  # noqa: SLF001
        "streamed_gb_per_token_est": R["weight_gb"],
        "effective_gb_s": R["weight_gb"] * N / decode_s,
    }
    del session, adapter
    dump()
    print(arm, json.dumps({k: v for k, v in R["arms"][arm].items() if k != "token_ids"}), flush=True)
dump()
