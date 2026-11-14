"""Value-swap controls, FLUX.2 Klein 4B (PREREG-ADDENDUM-2.md of predicted_write_20261113).

Same swap as the capture worker's ``vswap-all`` arm (``pw_worker.make_swap_backend``, shipped
beside this file and imported unchanged): in all 20 single blocks and all 4 steps, image
receivers read replaced text-row values through the live attention map (delta form
``native out + P[:, text] (V_src - V)``); joint blocks and text receivers untouched; native
conditioning of the scene prompt. Only the source of the replacement values differs per arm:

* ``vswap-all``      deleted-scene-word run, rows aligned by ``deletion_alignment`` (reproduces
                     the capture job's arm; its PNG is compared to the saved one);
* ``vswap-unrel``    a run of an UNRELATED prompt with the same number of real tokens; rows
                     aligned one to one (all 512 text rows, the same rows as ``vswap-all``);
* ``vswap-nonscene`` the deleted-word run of a word that is NOT the scene word (rows aligned by
                     ``deletion_alignment`` for that word);
* ``vswap-sham``     recompute sham (FP32 P V, nothing replaced).

Roots (no port): native, deleted (scene word), iso (hybrid own). Every image and its latent are
saved. Also logged: the RMS of the outer attention gates per (step, block) in the native root run
(post-hoc check of the stratum gate scale used by gated mass). Writes JSON / PNG / npy only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import flux_read_panel_worker as base  # noqa: E402
import pw_worker as PWW  # noqa: E402  (unchanged capture worker: make_swap_backend, Gates)

JOINT, SINGLE, BLOCKS, T = 5, 20, 25, 512
WORKER = "vswap2_worker.py"
MODULES = ("attention_port", "attention_panels", "flux_read_panels", "predicted_write")
SCHEMA = "mdb-predicted-write-vswap2-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_proof(payload):
    import importlib
    seal = json.loads((payload / "payload-seal.json").read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    rows = {}
    for name in MODULES:
        module = importlib.import_module("mdb_lanes." + name)
        digest = sha(module.__file__)
        sealed = seal["packages"]["mdb_lanes"].get(name + ".py")
        declared = request.get("source_hashes", {}).get("mdb_lanes/" + name + ".py")
        rows[name] = {"file": module.__file__, "sha256": digest, "sealed": sealed, "declared": declared}
        if digest != sealed or digest != declared:
            raise RuntimeError(f"mdb_lanes.{name} bytes differ from the staged seal / declared hash")
    for f in ("flux_read_panel_worker.py", "pw_worker.py"):
        rows[f] = {"sha256": sha(payload / f), "declared": request["source_hashes"].get(f)}
        if rows[f]["sha256"] != rows[f]["declared"]:
            raise RuntimeError(f + " differs from its declared hash")
    return rows


class GateLog(PWW.Gates):
    """Gates plus a per-call log of gate RMS (image, text) when ``on``."""

    def __init__(self, tr):
        super().__init__(tr)
        self.on, self.log = False, []

    def _hook(self, index):
        inner = super()._hook(index)

        def hook(module, args, kwargs):
            inner(module, args, kwargs)
            if self.on:
                gi, gt = self.now[index]
                self.log.append([index, float(gi.pow(2).mean().sqrt()), float(gt.pow(2).mean().sqrt()),
                                 float(gi.abs().mean()), float(gt.abs().mean())])
        return hook


def main():
    payload = Path.cwd()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("not a vswap2 config")
    job = os.environ.get("MRUN_JOB_ID", "local")
    out = Path(config["output_scratch"]) / job / "predicted-write-vswap2"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    timeline = []

    def event(kind, **fields):
        row = dict(kind=kind, t=round(time.monotonic() - started, 3), **fields)
        timeline.append(row)
        print("VSWAP2 " + json.dumps(row, sort_keys=True, default=str), flush=True)

    modules = module_proof(payload)
    base.save(out / "modules.json", modules)
    base.save(out / "run-config.json", config)
    budget = float(config["max_wall_seconds"])

    from mdb_lanes import attention_port as ap
    from mdb_lanes import flux_read_panels as frp
    from mdb_lanes import predicted_write as PW
    lane = base.Lane(config, event)
    import numpy as np
    import torch
    prompts = config["prompts"]
    scenes = config["scenes"]           # [{word, prompt_id, unrelated, nonscene: {word} or None}]
    texts = set()
    for sc in scenes:
        p = prompts[sc["prompt_id"]]
        texts |= {p, frp.delete_word(p, sc["word"]), sc["unrelated"]}
        if sc.get("nonscene"):
            texts.add(frp.delete_word(p, sc["nonscene"]))
    lane.encode_all(sorted(texts))
    base.save(out / "tokens.json", {k: v[2] for k, v in lane.encodings.items()})
    lane.place()
    dev, S = lane.dev, lane.S
    gates = GateLog(lane.tr)
    backend = ap.FluxBackend(lane.tr, text_rows=T, identity=lane.adapter.model_identity)
    Swap = PWW.make_swap_backend(ap)
    sb = Swap(lane.tr, text_rows=T, identity=lane.adapter.model_identity)
    seed = int(config["seed"])
    init = lane.noise(seed)
    init_sha = lane.init_sha(init)
    STEPS = lane.STEPS

    def run(cond, port=None, observers=()):
        c, txt_ids = cond
        latent, img_ids, timesteps, sigmas = init
        sess = lane.adapter.session(latent=latent.clone(), conditioning=c.to(dev), img_ids=img_ids.clone(),
                                    txt_ids=txt_ids.to(dev), timesteps=timesteps.clone(), sigmas=sigmas.clone())
        state = sess._state
        if port is None:
            while state["step"] < STEPS:
                state = lane.adapter.advance(state)
        else:
            with port.apply(observers=list(observers), forwards=STEPS):
                while state["step"] < STEPS:
                    state = lane.adapter.advance(state)
            if port.calls != STEPS * BLOCKS or port.forward_count != STEPS:
                raise ValueError(f"call/forward count differs: {port.calls}/{port.forward_count}")
        sess._state = dict(state)
        lat = sess.read("latent").detach().float().cpu().numpy()[0]
        image = lane.adapter.decode(sess, height=lane.H, width=lane.W)
        return image, lat

    def keep(word, name, image, lat):
        safe = frp.safe_name(name)
        png = out / "images" / word / f"{safe}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        image.save(png)
        lp = out / "latents" / word / f"{safe}.npy"
        lp.parent.mkdir(parents=True, exist_ok=True)
        np.save(lp, lat.astype(np.float32))
        rgb = base.pixels(np, image)
        return rgb, lat, {"png": str(png.relative_to(out)), "png_sha256": sha(png),
                          "latent": str(lp.relative_to(out)),
                          "latent_sha256": hashlib.sha256(lat.astype(np.float32).tobytes()).hexdigest(),
                          "image_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}

    def store_values(cond, perm_text):
        """Single-block text-row values of one run (rows reordered by ``perm_text``), kept on the GPU."""
        vstore = {}

        def reduce(read):
            if read.block >= JOINT:
                vstore[(read.step, read.block)] = read.v[0][:, perm_text].detach().clone()
            return 0
        obs = ap.observe(reduce, name="vstore")
        port = ap.AttentionPort(backend)
        image, lat = run(cond, port, observers=[obs])
        return vstore, image, lat

    results = {"init_latent_sha256": init_sha, "scenes": {}}
    for sc in scenes:
        if time.monotonic() - started > budget:
            event("budget_stop", word=sc["word"])
            break
        w, pid = sc["word"], sc["prompt_id"]
        prompt = prompts[pid]
        rec_o = lane.encodings[prompt][2]
        rows = {"word": w, "prompt": prompt}
        conds = {k: lane.conditioning({"kind": kind, "prompt": prompt, "word": w})
                 for k, kind in (("native", "native"), ("del", "deletion"), ("iso", "hybrid_own"))}
        roots = {}
        gates.on, gates.log = True, []
        for name in ("native", "del", "iso"):
            image, lat = run(conds[name][0])
            roots[name] = keep(w, name, image, lat)
            gates.on = False
        rows["gate_rms_native_root"] = gates.log            # call order: step-major, 25 blocks per step
        rows["roots"] = {k: v[2] for k, v in roots.items()}
        rows["conditioning_sha256"] = {k: v[2] for k, v in conds.items()}
        rows["mad_native_del"] = base.compare(np, *roots["native"][:2], *roots["del"][:2])["px_mad"]

        sources = {}
        # deleted scene word, aligned
        rec_d = lane.encodings[frp.delete_word(prompt, w)][2]
        al = PW.deletion_alignment(rec_o, rec_d, w, T)
        sources["vswap-all"] = ("del", torch.tensor(al["pi"], device=dev), conds["del"][0],
                                {"source": "deleted scene word", "deleted_word": w, "word_rows": al["word"]})
        # unrelated prompt, same number of real tokens, rows one to one
        rec_u = lane.encodings[sc["unrelated"]][2]
        cond_u, _, csha_u = lane.conditioning({"kind": "native", "prompt": sc["unrelated"]})
        sources["vswap-unrel"] = ("unrel", torch.arange(T, device=dev), cond_u,
                                  {"source": "unrelated prompt", "prompt": sc["unrelated"], "n_real": rec_u["n_real"],
                                   "scene_n_real": rec_o["n_real"], "same_length": rec_u["n_real"] == rec_o["n_real"],
                                   "conditioning_sha256": csha_u})
        if sc.get("nonscene"):
            nw = sc["nonscene"]
            rec_n = lane.encodings[frp.delete_word(prompt, nw)][2]
            al_n = PW.deletion_alignment(rec_o, rec_n, nw, T)
            if al_n is None:
                raise ValueError(f"deletion alignment fails for {nw}")
            cond_n, _, csha_n = lane.conditioning({"kind": "deletion", "prompt": prompt, "word": nw})
            sources["vswap-nonscene"] = ("nonscene", torch.tensor(al_n["pi"], device=dev), cond_n,
                                         {"source": "deleted non-scene word", "deleted_word": nw,
                                          "word_rows": al_n["word"], "conditioning_sha256": csha_n})
        arms = []
        for name, (tag, perm, cond_src, info) in sources.items():
            vstore, image_s, lat_s = store_values(cond_src, perm)
            src_rgb, src_lat, src_hash = keep(w, f"source-{tag}", image_s, lat_s)
            sb.configure(vstore, torch.arange(T, device=dev), "delta")
            port = ap.AttentionPort(sb)
            t0 = time.monotonic()
            image, lat = run(conds["native"][0], port)
            rgb, lat, hashes = keep(w, name, image, lat)
            row = {"arm": name, **info, "touched_calls": sb.touched, "seconds": time.monotonic() - t0,
                   "source_run": src_hash, **hashes}
            for r in ("native", "del", "iso"):
                row["vs_" + r] = base.compare(np, rgb, lat, *roots[r][:2])
            row["d"] = frp.d_score(row["vs_del"]["px_mad"], rows["mad_native_del"])
            arms.append(row)
            event("vswap", word=w, arm=name, d=round(row["d"], 4), touched=sb.touched)
            vstore.clear()
            torch.cuda.empty_cache()
        sb.configure({}, torch.arange(T, device=dev), "sham")
        port = ap.AttentionPort(sb)
        image, lat = run(conds["native"][0], port)
        rgb, lat, hashes = keep(w, "vswap-sham", image, lat)
        row = {"arm": "vswap-sham", "source": "none (recompute sham)", "touched_calls": sb.touched, **hashes}
        for r in ("native", "del", "iso"):
            row["vs_" + r] = base.compare(np, rgb, lat, *roots[r][:2])
        row["d"] = frp.d_score(row["vs_del"]["px_mad"], rows["mad_native_del"])
        arms.append(row)
        rows["arms"] = arms
        rows["peak_vram_mb"] = lane.job_peak()
        base.save(out / f"vswap2.{w}.json", rows)
        results["scenes"][w] = {"arms": [a["arm"] for a in arms]}
        event("scene_done", word=w)

    gates.remove()
    summary = dict(schema=SCHEMA + "-summary", job_id=job, request_sha256=request.get("request_sha256"),
                   worker_sha256=sha(payload / WORKER), modules=modules, init_latent_sha256=init_sha,
                   worker_seconds=time.monotonic() - started, peak_vram_allocated_mb=lane.job_peak(),
                   model_identity=lane.adapter.model_identity, scenes=results["scenes"])
    base.save(out / "summary.json", summary)
    base.save(out / "timeline.json", timeline)
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    base.save(out / "files.json", files)
    print("VSWAP2_SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "modules"}, default=str), flush=True)


if __name__ == "__main__":
    main()
