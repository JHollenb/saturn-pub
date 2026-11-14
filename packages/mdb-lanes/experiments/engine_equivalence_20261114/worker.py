"""Lane path vs engine path on the Qwen read panel, FP32, one resident Qwen2.5-1.5B (see PREREG.md).

Lane path: the steps of ``qualification/qwen_panel_worker.py`` (prefill rows 0..18 in one forward,
rows 19 and 20 one forward each; row 21 and two greedy tokens; ``attention_port`` cuts from the
sealed plan through ``attention_panels.port_cuts``). Engine path: ``mdb_lanes.qwen_engine``
(``mdb.attach`` stepped run with ``prefill=19``, ``mdb.trace``, ``whatif_many``). Writes
``result.json`` under the output scratch and prints it on one ``QWENENG_RESULT`` line.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    payload = Path.cwd()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    job = os.environ.get("MRUN_JOB_ID", os.environ.get("MRUN_RUN_ID", "local"))
    root = Path(config["output_scratch"]) / job / "qwen-engine"
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def event(kind, **fields):
        print("QWENENG " + json.dumps(dict(kind=kind, t=round(time.monotonic() - started, 3), **fields),
                                      sort_keys=True, default=str), flush=True)

    import torch
    import mdb
    import mdb_lanes.attention_panels as P
    import mdb_lanes.attention_port as ap
    import mdb_lanes.qwen_engine as QE
    import mdb_lanes.qwen_panels as Q
    import saturn_pub
    import saturn_pub.numerics as numerics
    import mdb.engine.trace as engine_trace
    from mdb.analysis.forecast import Margin
    modules = {m.__name__: {"file": m.__file__, "sha256": sha(m.__file__)}
               for m in (mdb, engine_trace, saturn_pub, numerics, P, ap, QE, Q)}
    event("modules", modules=modules)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("admitted CUDA required")
    torch.cuda.reset_peak_memory_stats()

    plan_raw = (payload / config["plan"]["file"]).read_bytes()
    panel = P.Panel.from_json(json.loads(plan_raw))
    if panel.sha256() != config["plan"]["plan_sha256"]:
        raise RuntimeError("plan sha256 differs from the config")
    P.verify_seal(panel, json.loads((payload / config["plan"]["seal"]).read_text()))
    if panel.port.get("lane") != "float32":
        raise RuntimeError("plan is not the FP32 lane")

    weight_file = Path(config["model_path"]) / "model.safetensors"
    weight_sha = sha(weight_file)
    if weight_sha != config["expected_weight_file_sha256"]:
        raise RuntimeError("Qwen model.safetensors differs from the case's checkpoint identity")
    event("weights", sha256=weight_sha)

    from transformers import DynamicCache
    from mdb_lanes.ar.model import load_ar_model
    spec, model, tokenizer = load_ar_model(config["model"], device="cuda", torch_dtype=torch.float32,
                                           attn_implementation="eager")
    model.eval().requires_grad_(False)
    if spec.family != "qwen2" or model.config._attn_implementation != "eager":
        raise RuntimeError("eager Qwen2 required")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError("FP32 parameters required")
    layers = len(model.model.layers)
    event("loaded", layers=layers, vram_mb=torch.cuda.memory_allocated() / 2 ** 20,
          buffers={n: str(b.dtype) for n, b in model.named_buffers()})
    ids = tuple(tokenizer(Q.RECIPIENT_TEXT, add_special_tokens=False)["input_ids"])
    if ids != Q.RECIPIENT_IDS:
        raise RuntimeError("tokenization differs from the saved case")
    hot, cold = Q.TRACKED["hot"], Q.TRACKED["cold"]
    identity = {"model": config["model"], "weight_file_sha256": weight_sha, "lane": "float32"}

    # ------------------------------------------------------------------ lane path
    t = torch.tensor([ids], device=dev)
    parent = DynamicCache(config=model.config)
    with torch.inference_mode():
        model(input_ids=t[:, :Q.SOURCE_ROW + 1], past_key_values=parent, use_cache=True)
        for j in range(Q.SOURCE_ROW + 1, len(ids) - 1):
            model(input_ids=t[:, j:j + 1], past_key_values=parent, use_cache=True)

    def generate():
        cache, token, first = copy.deepcopy(parent), ids[-1], None
        with torch.inference_mode():
            for step in range(Q.GENERATED):
                logits = model(input_ids=torch.tensor([[token]], device=dev), past_key_values=cache,
                               use_cache=True).logits[0, -1]
                if step == 0:
                    first = logits.float().cpu()
                token = int(logits.argmax())
        return first

    lane = {"native": generate()}
    port = ap.AttentionPort(ap.QwenBackend(model, identity=identity))
    obs = ap.observe(lambda read: read.probs()[0, :, -1, :].double().cpu().numpy(), steps={0})
    with port.apply(observers=[obs], forwards=Q.GENERATED):
        observed_logits = generate()
    att_lane = torch.as_tensor(obs.stack()[0])
    att_trace, trace_info, traced_logits = QE.trace_attention(model, generate, layers=layers, call=0)
    att_trace = torch.tensor(att_trace, dtype=torch.float64)
    event("lane_native", margin=QE.margin_of(lane["native"], hot, cold))

    rows_table = Q.row_table(sorted({a.rows_item for a in panel.arms if a.rows_item}))
    chosen = []
    for arm in panel.arms:
        cuts, _ = QE.expressible(arm, rows_table, panel)
        if cuts:
            chosen.append(arm.arm)
            chosen.append(arm.floor_of)            # the delta arm it is the recompute twin of
    chosen = list(dict.fromkeys(chosen))
    lport = ap.AttentionPort(ap.QwenBackend(model, identity=identity), **P.port_kwargs(panel, rows_table))
    P.check_port(panel, lport, forwards=Q.GENERATED)
    ids_by = panel.by_id()
    for name in chosen:
        arm = ids_by[name]
        with lport.apply(*P.port_cuts(arm, panel=panel, port=lport), forwards=Q.GENERATED):
            lane[name] = generate()
    event("lane_arms", arms=len(chosen))

    # ------------------------------------------------------------------ engine path
    t_engine = time.monotonic()
    dbg, run = QE.attach_run(model, tokenizer, ids, prefill=Q.SOURCE_ROW + 1, row=Q.RECEIVER_ROW)
    engine_native = QE.run_logits(run, Q.RECEIVER_ROW).float().cpu()
    att_run = torch.tensor(QE.run_attention(run, Q.RECEIVER_ROW, layers), dtype=torch.float64)
    event("engine_run", seconds=round(time.monotonic() - t_engine, 2),
          margin=QE.margin_of(engine_native, hot, cold), vram_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
    t_cuts = time.monotonic()
    rows, lane_only = QE.engine_rows(panel, run, rows_table, hot=hot, cold=cold, row=Q.RECEIVER_ROW)
    event("engine_cuts", rows=len(rows), seconds=round(time.monotonic() - t_cuts, 2),
          vram_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
    t_contrib = time.monotonic()
    contrib = QE.contributions(run, Q.RECEIVER_ROW, Q.SOURCE_ROW, Margin(Q.RECEIVER_ROW, hot, cold))
    event("contributions", seconds=round(time.monotonic() - t_contrib, 2))

    # ------------------------------------------------------------------ compare
    def cmp(a, b):
        a, b = a.double(), b.double()
        return {"equal": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max())}

    engine_margin = {r["arm"]: r["margin"] for r in rows}
    arms = {}
    for r in rows:
        name = r["arm"]
        if name == "native":
            continue
        arm = ids_by[name]
        twin = arm.floor_of
        arms[name] = {
            "engine_margin": r["margin"], "lane_margin": QE.margin_of(lane[name], hot, cold),
            "lane_delta_margin": QE.margin_of(lane[twin], hot, cold), "delta_arm": twin,
            "margin_equal": r["margin"] == QE.margin_of(lane[name], hot, cold),
            "margin_equal_delta": r["margin"] == QE.margin_of(lane[twin], hot, cold),
            "engine": r["engine"]}
    result = {
        "schema": "mdb-qwen-engine-equivalence-v1", "job": job, "plan_sha256": panel.sha256(),
        "modules": modules, "identity": identity, "trace_info": trace_info,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "base": {"lane_vs_engine_logits": cmp(lane["native"], engine_native),
                 "lane_vs_observed_logits": cmp(lane["native"], observed_logits),
                 "lane_vs_traced_logits": cmp(lane["native"], traced_logits.float().cpu()),
                 "lane_margin": QE.margin_of(lane["native"], hot, cold),
                 "engine_margin": engine_margin.get("native"),
                 "lane_logits_sha256": tsha(lane["native"]), "engine_logits_sha256": tsha(engine_native)},
        "attention": {"lane_recompute_vs_trace": cmp(att_lane, att_trace), "trace_vs_run": cmp(att_trace, att_run),
                      "lane_recompute_vs_run": cmp(att_lane, att_run), "shape": list(att_run.shape)},
        "arms": arms,
        "summary": {"engine_rows": len(rows), "lane_only": len(lane_only),
                    "recompute_equal": sum(a["margin_equal"] for a in arms.values()),
                    "delta_equal": sum(a["margin_equal_delta"] for a in arms.values()),
                    "max_abs_recompute": max(abs(a["engine_margin"] - a["lane_margin"]) for a in arms.values()),
                    "max_abs_delta": max(abs(a["engine_margin"] - a["lane_delta_margin"]) for a in arms.values()),
                    "verdicts": sorted({a["engine"]["verdict"] for a in arms.values()})},
        "lane_margins": {name: QE.margin_of(v, hot, cold) for name, v in lane.items()},
        "contributions_named": {k: contrib[tuple(v)] for k, v in Q.NAMED_HEADS.items()},
        "contributions_top": sorted(([list(k), v] for k, v in contrib.items() if v["to_consumer"] is not None),
                                    key=lambda kv: -abs(kv[1]["to_consumer"]))[:8],
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "seconds": round(time.monotonic() - started, 2)}
    (root / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print("QWENENG_RESULT " + json.dumps(result, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
