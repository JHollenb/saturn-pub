"""CPU microbench of mdb stepping, the reviewer's probe config (Qwen2.5-1.5B K/V geometry:
2 KV heads x 128 head_dim; hidden 256 so the native layer is cheap), bf16, 4 threads.

usage: python bench.py row L T ROWS        stepped rows at prefill=T-ROWS (p3_fix base)
       python bench.py prefill0 L T        prefill=0, until t{T-1}.logits, then held_bytes()
       python bench.py ports L T           one row port by port, lookahead none and layer
Prints one JSON line per measurement.
"""
import hashlib
import json
import sys
import time

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.engine.run import LOOKAHEADS, Run
from saturn_pub.adapters.decoder import DecoderAdapter

torch.manual_seed(0)
torch.set_num_threads(4)
mode, L, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = Qwen2Config(vocab_size=4096, hidden_size=256, intermediate_size=512, num_hidden_layers=L,
                  num_attention_heads=2, num_key_value_heads=2, head_dim=128,
                  max_position_embeddings=8192, attn_implementation="eager")
model = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
ad = DecoderAdapter(model, granularity="operation")
tokens = torch.randint(0, 4096, (T,)).tolist()


def digest(run):
    return hashlib.sha256("".join(m.fingerprint for m in run._trail).encode()).hexdigest()[:16]


def logits_sha(run, row):
    t = run.read(f"t{row}.logits").contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy()).hexdigest()[:16]


if mode == "row":
    rows = int(sys.argv[4])
    warm = Run(ad, tokens, prefill=T - 1)
    warm.step("token")
    run = Run(ad, tokens, prefill=T - rows)
    t0 = time.perf_counter()
    for _ in range(rows):
        run.step("token")
    dt = time.perf_counter() - t0
    print(json.dumps({"mode": mode, "L": L, "T": T, "rows": rows,
                      "ms_per_row": round(dt / rows * 1e3, 1), "trail": digest(run),
                      "tokens_tail": run.session.read("tokens")[0, -3:].tolist(),
                      "logits": logits_sha(run, T - 1)}))
elif mode == "prefill0":
    run = Run(ad, tokens, prefill=0)
    t0 = time.perf_counter()
    run.until(f"t{T - 1}.logits")
    dt = time.perf_counter() - t0
    t0 = time.perf_counter()
    hb = run.held_bytes()
    hbt = time.perf_counter() - t0
    print(json.dumps({"mode": mode, "L": L, "T": T, "until_s": round(dt, 2),
                      "held_bytes_s": round(hbt, 3), "held": hb, "trail": digest(run),
                      "logits": logits_sha(run, T - 1)}))
elif mode == "ports":
    for la in LOOKAHEADS:
        run = Run(ad, tokens, lookahead=la)
        t0 = time.perf_counter()
        seen = {}
        while run.where != f"t{T}.start":
            run.step("port")
            if run._port is not None:
                v = run.read().contiguous()
                seen[run.where] = hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy()).hexdigest()[:12]
        dt = time.perf_counter() - t0
        print(json.dumps({"mode": mode, "lookahead": la, "L": L, "T": T,
                          "ms_row": round(dt * 1e3, 1), "calls": run.calls,
                          "reads": hashlib.sha256(json.dumps(seen, sort_keys=True).encode()).hexdigest()[:16],
                          "trail": digest(run)}))
