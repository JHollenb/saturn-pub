"""FLUX.2 Klein 4B smoke for mdb.attention_port (one model load, ~20 short branches).

Arms: native; passthrough port; observe; shams (new and legacy, both cut types);
the read-graph arm z:own60:ball (job-a0bca63168b4) under legacy and new precision,
its renormalising twin m:own60:ball, and each cut's matched sham.

Model load, prompt encoding, noise, hybrid "own" conditioning and the fast
transition lane are copied from saturn/experiments/2026-11-11-flux-read-graph/
read_graph_worker.py (sha 3f67f1d3...) so the legacy arms can reproduce its images.
Images, latents and reductions are evidence; no model weights are written.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

JOINT, SINGLE = 5, 20
BLOCKS = JOINT + SINGLE
CLASSES = ("template", "content", "pad", "image")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def delete_word(prompt, word):
    out = re.sub(r"\s*\b" + re.escape(word) + r"\b", "", prompt, count=1)
    out = re.sub(r"\s+", " ", out).replace(" ,", ",").strip()
    return out


def worker():
    payload = Path.cwd()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    job = os.environ.get("MRUN_JOB_ID", "local")
    scratch = Path(config["output_scratch"]) / job
    scratch.mkdir(parents=True, exist_ok=False)
    out = scratch / "attention-port-smoke"
    out.mkdir()
    save(out / "plan.json", config)
    started = time.monotonic()

    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    import mdb_lanes.attention_port as ap

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError("admitted CUDA required")
    dev = torch.device("cuda")
    H_PX, W_PX, STEPS = int(config["height"]), int(config["width"]), int(config["steps"])
    timeline = []

    def event(kind, **fields):
        row = dict(kind=kind, t=round(time.monotonic() - started, 3), **fields)
        timeline.append(row)
        print("PORTSMOKE " + json.dumps(row, sort_keys=True), flush=True)

    event("load_start")
    pipe = Flux2KleinPipeline.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled:
        raise ValueError("distilled Klein required")
    tr = pipe.transformer
    if len(tr.transformer_blocks) != JOINT or len(tr.single_transformer_blocks) != SINGLE:
        raise ValueError("architecture differs")
    for component in (pipe.text_encoder, tr, pipe.vae):
        component.eval().requires_grad_(False)
    event("load_done")

    def encode(prompt):  # verbatim from read_graph_worker.encode (record trimmed to what is used)
        tok = pipe.tokenizer
        formatted = tok.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        enc = tok(formatted, return_tensors="pt", padding="max_length", truncation=True, max_length=512,
                  return_offsets_mapping=True)
        mask = enc["attention_mask"][0].tolist()
        offsets = enc["offset_mapping"][0].tolist()
        ids = enc["input_ids"][0].tolist()
        with torch.inference_mode():
            c, txt_ids = pipe.encode_prompt(prompt, device=dev, max_sequence_length=512)
        start = formatted.find(prompt) if prompt else -1
        stop = start + len(prompt)
        cls = []
        for j, (a, b) in enumerate(offsets):
            if not mask[j]:
                cls.append("pad")
            elif prompt and start >= 0 and b > a and a < stop and b > start:
                cls.append("content")
            else:
                cls.append("template")
        words = []
        if prompt and start >= 0:
            for m in re.finditer(r"[A-Za-z0-9']+", prompt):
                s, e = start + m.start(), start + m.end()
                rows = [j for j, (a, b) in enumerate(offsets) if mask[j] and b > a and a < e and b > s]
                words.append(dict(word=m.group(0).lower(), rows=rows))
        real = [j for j in range(512) if mask[j]]
        record = dict(prompt=prompt, input_ids=ids[: max(real) + 2] if real else [], n_real=len(real),
                      real_rows=real, classes=cls, words=words)
        return c.detach().to("cpu"), txt_ids.detach().to("cpu"), record

    prompt, word = config["prompt"], config["word"]
    pipe.text_encoder.to(dev)
    enc_o = encode(prompt)
    enc_d = encode(delete_word(prompt, word))
    event("encoded")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    pipe.tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    tr.to(dev)
    pipe.vae.to(dev)
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    HEADS = tr.transformer_blocks[0].attn.heads
    T = 512
    N_IMG = (H_PX // 16) * (W_PX // 16)
    GRID = H_PX // 16
    S = T + N_IMG

    def noise(seed):
        with torch.inference_mode():
            latent, img_ids = pipe.prepare_latents(1, tr.config.in_channels // 4, H_PX, W_PX, torch.bfloat16, dev,
                                                   torch.Generator(device="cuda").manual_seed(int(seed)))
            sched = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1.0, 1 / STEPS, STEPS)
            timesteps, n = retrieve_timesteps(pipe.scheduler, STEPS, dev, sigmas=sched,
                                              mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=STEPS))
        if n != STEPS or latent.shape[1] != N_IMG:
            raise ValueError("schedule or geometry differs")
        return latent, img_ids, timesteps.clone(), pipe.scheduler.sigmas.clone()

    def hybrid():  # verbatim logic of read_graph_worker.hybrid(pid, word, "own")
        c_o, ids_o, rec_o = enc_o
        c_d, _, rec_d = enc_d
        own = sorted(r for w in rec_o["words"] if w["word"] == word for r in w["rows"])
        n_o, n_d = rec_o["n_real"], rec_d["n_real"]
        shift = n_o - n_d
        if rec_o["real_rows"] != list(range(n_o)) or rec_d["real_rows"] != list(range(n_d)) or shift < 1:
            raise ValueError("alignment")
        c_h = c_o.clone()
        later = list(range(max(own) + 1, n_o))
        id_mismatch = sum(rec_o["input_ids"][j] != rec_d["input_ids"][j - shift] for j in later)
        before = list(range(0, min(own)))
        prefix_equal = bool(torch.equal(c_o[0, before], c_d[0, before]))
        c_h[0, later] = c_d[0, [j - shift for j in later]]
        c_h[0, n_o:] = c_d[0, n_o:]
        return (c_h, ids_o), dict(shift=shift, own_rows=own, later_rows=len(later),
                                  later_id_mismatch=int(id_mismatch), prefix_equal=prefix_equal)

    def run_fast(cond, init, port=None, cuts=(), observers=()):  # read_graph_worker.run_fast without schedules
        c, txt_ids = cond
        latent, img_ids, timesteps, sigmas = init
        sess = adapter.session(latent=latent.clone(), conditioning=c.to(dev), img_ids=img_ids.clone(),
                               txt_ids=txt_ids.to(dev), timesteps=timesteps.clone(), sigmas=sigmas.clone())
        state = sess._state
        calls = None
        if port is None:
            while state["step"] < STEPS:
                state = adapter.advance(state)
        else:
            with port.apply(*cuts, observers=observers, log=True):
                while state["step"] < STEPS:
                    state = adapter.advance(state)
                calls = port.calls
            if calls != STEPS * BLOCKS:
                raise ValueError(f"attention call count differs: {calls}")
        sess._state = dict(state)
        latent = sess.read("latent").detach().float().cpu().numpy()[0]
        image = adapter.decode(sess, height=H_PX, width=W_PX)
        return image, latent

    def pixels(image):
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    def compare(a_img, a_lat, b_img, b_lat):
        d = np.abs(pixels(a_img) - pixels(b_img)).mean(-1)
        return dict(px_mad=float(d.mean()), px_max=float(d.max()),
                    latent_rel=float(np.linalg.norm(a_lat - b_lat) / np.linalg.norm(b_lat)),
                    latent_max_abs=float(np.abs(a_lat - b_lat).max()), latent_equal=bool(np.array_equal(a_lat, b_lat)))

    # --------------------------------------------------------------- rows
    rec = enc_o[2]
    cls = rec["classes"]
    rows = {name: [j for j in range(T) if cls[j] == name] for name in CLASSES[:3]}
    rows["image"] = list(range(T, S))
    rows["text"] = list(range(T))
    rows["real"] = rows["template"] + rows["content"]
    own = sorted(r for w in rec["words"] if w["word"] == word for r in w["rows"])
    others = [j for j in range(T) if j not in set(own)]
    heads = [tuple(x) for x in config["heads"]]
    backend = ap.FluxBackend(tr, text_rows=T, identity=adapter.model_identity)
    ports = {"delta": ap.AttentionPort(backend, rows=rows, zero_method="delta"),
             "recompute": ap.AttentionPort(backend, rows=rows, zero_method="recompute"),
             "legacy": ap.AttentionPort(backend, rows=rows, precision="legacy")}
    z_cut = ap.cut(["image", others], [own], heads=heads, renormalize=False)
    m_cut = ap.cut(["image", others], [own], heads=heads, renormalize=True)

    init = noise(int(config["seed"]))
    cond_native = (enc_o[0], enc_o[1])
    cond_hyb, hyb_info = hybrid()
    save(out / "hybrid_info.json", hyb_info)
    event("ready", own=own, heads=len(heads), blocks_touched=len({b for b, _ in heads}))

    src_onehot = np.zeros((S, 4), np.float32)
    for j, name in enumerate(cls):
        src_onehot[j, CLASSES.index(name)] = 1.0
    src_onehot[T:, 3] = 1.0

    def tok(read):  # make_capture tok_p / tok_w
        p = read.probs()[0]
        pw = torch.cat([p[:, sl] * n[:, None, :] for sl, n in read.write_norms()], dim=1)
        return dict(tok_p=p[:, T:, :T].mean(1).cpu().numpy(), tok_w=pw[:, T:, :T].mean(1).cpu().numpy())

    arms = [
        ("native", cond_native, None, (), "native"),
        ("passthrough", cond_native, "delta", (), "native"),
        ("observe", cond_native, "delta", (), "native"),
        ("new:sham-zero-delta:all", cond_native, "delta", (ap.sham(renormalize=False),), "native"),
        ("new:sham-zero-recompute:all", cond_native, "recompute", (ap.sham(renormalize=False),), "native"),
        ("new:sham-renorm:all", cond_native, "delta", (ap.sham(renormalize=True),), "native"),
        ("legacy:sham-zero:all", cond_native, "legacy", (ap.sham(renormalize=False),), "native"),
        ("legacy:sham-renorm:all", cond_native, "legacy", (ap.sham(renormalize=True),), "native"),
        ("iso", cond_hyb, None, (), "iso"),
        ("legacy:z:own60", cond_hyb, "legacy", (z_cut,), "iso"),
        ("legacy:sham(z:own60)", cond_hyb, "legacy", (ap.sham(z_cut),), "iso"),
        ("new-delta:z:own60", cond_hyb, "delta", (z_cut,), "iso"),
        ("new-delta:sham(z:own60)", cond_hyb, "delta", (ap.sham(z_cut),), "iso"),
        ("new-recompute:z:own60", cond_hyb, "recompute", (z_cut,), "iso"),
        ("new-recompute:sham(z:own60)", cond_hyb, "recompute", (ap.sham(z_cut),), "iso"),
        ("legacy:m:own60", cond_hyb, "legacy", (m_cut,), "iso"),
        ("legacy:sham(m:own60)", cond_hyb, "legacy", (ap.sham(m_cut),), "iso"),
        ("new:m:own60", cond_hyb, "delta", (m_cut,), "iso"),
        ("new:sham(m:own60)", cond_hyb, "delta", (ap.sham(m_cut),), "iso"),
    ]
    results, refs, observed = [], {}, {}
    for name, cond, port_name, cuts, ref in arms:
        observers = ()
        if name == "observe":
            observers = (ap.observe(ap.class_flow(src_onehot), name="flow_p"),
                         ap.observe(ap.class_flow(src_onehot, weighted=True), name="flow_w"),
                         ap.observe(tok, name="tok"))
        port = None if port_name is None else ports[port_name]
        torch.cuda.synchronize()
        t0 = time.monotonic()
        image, latent = run_fast(cond, init, port, cuts, observers)
        torch.cuda.synchronize()
        seconds = time.monotonic() - t0
        if name in ("native", "iso"):
            refs[name] = (image, latent)
        row = dict(arm=name, port=port_name, reference=ref, seconds=seconds,
                   cuts=[c.to_dict() for c in cuts], **compare(image, latent, *refs[ref]))
        if port is not None and cuts:
            log = port.log
            row["recomputed"] = dict(calls=len(log), head_rows=sum(len(g["heads"]) * g["receiver_rows"] for e in log for g in e["groups"]),
                                     removed_reads=sum(g["removed_reads"] * len(g["heads"]) for e in log for g in e["groups"]))
        results.append(row)
        safe = re.sub("[^A-Za-z0-9]+", "_", name).strip("_")
        image.save(out / f"arm.{safe}.png")
        np.save(out / f"latent.{safe}.npy", latent.astype(np.float32))
        for o in observers:
            observed[o.name] = o
        event("arm", **{k: v for k, v in row.items() if k != "cuts"})
    flow = {"flow_p": observed["flow_p"].stack(), "flow_w": observed["flow_w"].stack(),
            "tok_p": observed["tok"].stack("tok_p"), "tok_w": observed["tok"].stack("tok_w")}
    np.savez_compressed(out / "observe.npz", **{k: v.astype(np.float32) for k, v in flow.items()})
    save(out / "metrics.json", results)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    summary = dict(schema="mdb-attention-port-smoke-v1", job_id=job, request_sha256=request.get("request_sha256"),
                   worker_sha256=sha(payload / "attention_port_smoke_worker.py"),
                   attention_port_sha256=sha(ap.__file__), port_metadata=ports["delta"].apply(z_cut).metadata(),
                   peak_vram_allocated_mb=peak, worker_seconds=time.monotonic() - started, output_root=str(out),
                   heads=HEADS, text_rows=T, image_rows=N_IMG, model_identity=adapter.model_identity,
                   parameter_pointers={b: backend.parameter_names(b) for b in (0, JOINT)})
    save(out / "summary.json", summary)
    save(out / "timeline.json", timeline)
    save(out / "files.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print("PORTSMOKE_SUMMARY " + json.dumps(summary, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    worker()
