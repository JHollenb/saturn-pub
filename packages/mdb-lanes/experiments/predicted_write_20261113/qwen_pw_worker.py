"""Qwen2.5-1.5B side check: exact first-hop write of each value cell at the first receiver.

The value-coalition case of ``docs/QWEN-PANELS.md`` (recipient ``freezing``, donor ``boiling``,
source row 18, first receiver row 21). Value edits are placed in the cache after rows 19 and 20 ran,
so row 21 is the first reader and its attention map at the edited layer is unchanged by the edit.
For cell (layer l, KV group g) the change of row 21's attention write at layer l is therefore,
exactly in real arithmetic (the head mix of ``mdb.analysis.reads``: ``p[h, s] W_o,h v``)::

    dW(l, g) = sum_{h in group g} p[l, h, 21, 18] * W_o,h (v_donor[l, g, 18] - v_recipient[l, g, 18])

computed here in float64 on the captured FP32 operands. Also: the same write read through the final
norm (scale at the native final state) onto the hot - cold unembedding difference (the readout
convention of ``explain_margin``), and the existing mass predictors. FP32, eager, one load.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

WORKER = "qwen_pw_worker.py"
MODULES = ("analysis_metrics", "qwen_panels", "predicted_write")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    payload = Path.cwd()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config.get("schema") != "mdb-predicted-write-qwen-v1":
        raise ValueError("not a predicted-write Qwen config")
    job = os.environ.get("MRUN_JOB_ID", "local")
    out = Path(config["output_scratch"]) / job / "predicted-write-qwen"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    timeline = []

    def event(kind, **fields):
        row = dict(kind=kind, t=round(time.monotonic() - started, 3), **fields)
        timeline.append(row)
        print("PWQWEN " + json.dumps(row, sort_keys=True, default=str), flush=True)

    import importlib
    seal = json.loads((payload / "payload-seal.json").read_text())
    modules = {}
    for name in MODULES:
        m = importlib.import_module("mdb_lanes." + name)
        d = sha(m.__file__)
        modules[name] = {"file": m.__file__, "sha256": d}
        if d != seal["packages"]["mdb_lanes"].get(name + ".py") or d != request["source_hashes"].get(f"mdb_lanes/{name}.py"):
            raise RuntimeError(f"mdb_lanes.{name} differs from the staged seal")
    import numpy as np
    import torch
    from transformers import DynamicCache
    from mdb_lanes import qwen_panels as Q
    from mdb_lanes.ar.model import load_ar_model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    weight = Path(config["model_path"]) / "model.safetensors"
    wsha = sha(weight)
    if wsha != config["expected_weight_file_sha256"]:
        raise RuntimeError("Qwen weight file differs from the case's checkpoint identity")
    event("weights_ok", sha256=wsha)
    torch.cuda.reset_peak_memory_stats()
    spec, model, tok = load_ar_model(config["model"], device="cuda", torch_dtype=torch.float32,
                                     attn_implementation="eager")
    model.eval().requires_grad_(False)
    if spec.family != "qwen2" or model.config._attn_implementation != "eager":
        raise RuntimeError("eager Qwen2 required")
    ids = {side: tuple(tok(text, add_special_tokens=False)["input_ids"])
           for side, text in (("recipient", Q.RECIPIENT_TEXT), ("donor", Q.DONOR_TEXT))}
    if ids["recipient"] != Q.RECIPIENT_IDS or ids["donor"] != Q.DONOR_IDS:
        raise RuntimeError("tokenization differs from the saved case")
    event("loaded")

    def prefill(tokens):
        cache = DynamicCache(config=model.config)
        t = torch.tensor([tokens], device=dev)
        with torch.inference_mode():
            model(input_ids=t[:, :Q.SOURCE_ROW + 1], past_key_values=cache, use_cache=True)
            for j in range(Q.SOURCE_ROW + 1, len(tokens) - 1):
                model(input_ids=t[:, j:j + 1], past_key_values=cache, use_cache=True)
        if cache.get_seq_length() != len(tokens) - 1:
            raise RuntimeError("prefill cache length differs")
        return cache

    parents = {side: prefill(ids[side]) for side in ids}
    values = {side: [layer.values.detach().double().cpu() for layer in parents[side].layers] for side in parents}
    captured = {}

    def norm_hook(module, args):
        captured["final_in"] = args[0][0, -1].detach().double().cpu()
    handle = model.model.norm.register_forward_pre_hook(norm_hook)
    cache = copy.deepcopy(parents["recipient"])
    with torch.inference_mode():
        res = model(input_ids=torch.tensor([[ids["recipient"][-1]]], device=dev), past_key_values=cache,
                    use_cache=True, output_attentions=True)
    handle.remove()
    logits = res.logits[0, -1].double().cpu()
    hot, cold = Q.TRACKED["hot"], Q.TRACKED["cold"]
    margin = float(logits[hot] - logits[cold])
    att = torch.stack([a[0, :, -1, :].double().cpu() for a in res.attentions])     # [L, H, 22]
    L, H = att.shape[0], att.shape[1]
    kv = model.config.num_key_value_heads
    group = H // kv
    D = model.config.hidden_size // H
    x = captured["final_in"]
    rho = float(1.0 / torch.sqrt((x * x).mean() + model.config.rms_norm_eps))
    gamma = model.model.norm.weight.detach().double().cpu()
    u = (model.lm_head.weight[hot] - model.lm_head.weight[cold]).detach().double().cpu()
    cells = []
    for layer, g in Q.LATE_CELLS:
        O = model.model.layers[layer].self_attn.o_proj.weight.detach().double().cpu()     # [hidden, H*D]
        dv = values["donor"][layer][0, g, Q.SOURCE_ROW] - values["recipient"][layer][0, g, Q.SOURCE_ROW]
        vr = values["recipient"][layer][0, g, Q.SOURCE_ROW]
        write = torch.zeros(O.shape[0], dtype=torch.float64)
        heads = []
        for h in range(g * group, (g + 1) * group):
            Oh = O[:, h * D:(h + 1) * D]
            p = float(att[layer, h, Q.SOURCE_ROW])
            c = p * (Oh @ dv)
            write += c
            heads.append({"head": h, "p": p, "pw_norm": float(c.norm()),
                          "native_contribution_norm": float((p * (Oh @ vr)).norm())})
        cells.append({"cell": f"L{layer}/KV{g}", "layer": layer, "kv": g,
                      "pw_norm": float(write.norm()),
                      "pw_dla_margin": float(rho * (u * gamma * write).sum()),
                      "pw_sum_head_norms": sum(r["pw_norm"] for r in heads),
                      "dv_norm": float(dv.norm()),
                      "max_source_attention": max(r["p"] for r in heads),
                      "sum_source_attention": sum(r["p"] for r in heads),
                      "native_contribution_norm": float(sum((r["p"] * (O[:, r["head"] * D:(r["head"] + 1) * D] @ vr))
                                                            for r in heads).norm()),
                      "heads": heads})
    result = {"margin_native": margin, "rho": rho, "attention_row21": att.tolist(), "cells": cells,
              "first_tokens_logit_argmax": int(logits.argmax()), "modules": modules, "weight_sha256": wsha,
              "peak_vram_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
              "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
              "worker_seconds": time.monotonic() - started, "worker_sha256": sha(payload / WORKER),
              "job_id": job, "request_sha256": request.get("request_sha256")}
    import resource
    result["ru_maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    save(out / "qwen-pw.json", result)
    save(out / "timeline.json", timeline)
    event("done", margin=margin, peak_mb=result["peak_vram_allocated_mb"])


if __name__ == "__main__":
    main()
