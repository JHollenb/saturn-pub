"""FLUX.2 Klein read-panel worker: run sealed attention-panel leases through mdb.attention_port.

Panel mode (default): for each pass in the run config (a sealed ``mdb.attention_panels`` plan, its
sha256, one lease, a precision), refuse a plan or lease whose hash differs, encode every prompt
the lease needs, release the text encoder, then for every (context = seed) run the reference
roots (native, word deleted, hybrid "own") without the port and every lease branch through
``AttentionPort.apply(*port_cuts(arm), forwards=STEPS)``. Per branch it writes the decoded PNG, a
sealed regeneration recipe (no tensors), hashes, distances to the arm's reference, to the
native, deletion and hybrid roots, the direct twin gap for recompute-twin floors, timing and
peak VRAM. Latents are written only when ``retain`` contains ``"latents"``.

Regenerate mode: ``--regenerate CONTEXT:ARM_ID [...] --recipes DIR`` rebuilds those arms from
their recipes (prompt text, conditioning construction, seed, geometry, port cut specs and
precision) and reports whether the conditioning, initial noise, final latent, decoded RGB and
PNG hashes equal the recipe's. Same host and software lane only.

Model load, prompt encoding, noise, the hybrid "own" conditioning and the fast transition lane
follow saturn/experiments/2026-11-11-flux-read-graph/read_graph_worker.py (sha 3f67f1d3...) so
``precision="legacy"`` arms reproduce its images. One resident model, no retries.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

JOINT, SINGLE = 5, 20
BLOCKS = JOINT + SINGLE
T = 512
WORKER = "flux_read_panel_worker.py"
MODULES = ("attention_port", "attention_panels", "flux_read_panels")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_bytes(*parts):
    h = hashlib.sha256()
    for name, raw in parts:
        h.update(name.encode() + b"\0" + len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")


def module_proof(payload):
    """Where mdb's shared modules were imported from, their sha256, and the payload seal's entry."""
    import importlib
    import mdb
    seal = json.loads((payload / "payload-seal.json").read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    rows = {"mdb": {"file": mdb.__file__}}
    for name in MODULES:
        module = importlib.import_module("mdb." + name)
        digest = sha(module.__file__)
        sealed = seal["packages"]["mdb"].get(name + ".py")
        declared = request.get("source_hashes", {}).get("mdb/" + name + ".py")
        rows[name] = {"file": module.__file__, "sha256": digest, "payload_seal_sha256": sealed,
                      "request_declared_sha256": declared,
                      "inside_checkout": "/Users/" in module.__file__ or "/.worktrees/" in module.__file__}
        if digest != sealed or (declared is not None and digest != declared):
            raise RuntimeError(f"mdb.{name} bytes differ from the staged payload seal")
    return rows


def checkpoint_manifest(model_path, full_hash):
    """Checkpoint file records: bytes, safetensors header sha256, and (optionally) full file sha256."""
    t0 = time.monotonic()
    root = Path(model_path)
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts):
        rel = path.relative_to(root).as_posix()
        rec = {"path": rel, "bytes": path.stat().st_size}
        if path.suffix == ".safetensors":
            with path.open("rb") as stream:
                n = int.from_bytes(stream.read(8), "little")
                header = stream.read(n)
            rec["header_sha256"] = hashlib.sha256(len(header).to_bytes(8, "little") + header).hexdigest()
            rec["tensors"] = len([k for k in json.loads(header) if k != "__metadata__"])
            rec["file_sha256"] = sha(path) if full_hash else None
        elif path.suffix in (".json", ".txt", ".jinja", ".md"):
            rec["file_sha256"] = sha(path)
        files.append(rec)
    return {"schema": "mdb-flux-checkpoint-files-v1", "root": str(root), "full_hash": bool(full_hash),
            "files": files, "seconds": round(time.monotonic() - t0, 3)}


def parameter_pointers(model_path, manifest):
    """Attention parameter pointers (``parameter_artifacts`` source fields, no payload hash)."""
    root = Path(model_path)
    rel = "transformer/diffusion_pytorch_model.safetensors"
    with (root / rel).open("rb") as stream:
        n = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(n))
    file_sha = next((f.get("file_sha256") for f in manifest["files"] if f["path"] == rel), None)
    rows = [{"file": rel, "file_sha256": file_sha, "tensor": name, "dtype": meta["dtype"], "shape": meta["shape"],
             "data_offsets": meta["data_offsets"], "payload_sha256": None}
            for name, meta in sorted(header.items()) if name != "__metadata__" and ".attn." in name]
    return {"schema": "mdb-flux-parameter-pointers-v1",
            "note": "parameter_artifacts source fields; payload_sha256 not computed (pointers, not restorable "
                    "artifacts); file_sha256 is null unless the run hashed the checkpoint",
            "parameters": rows}


# ===================================================================================== the lane

class Lane:
    """Klein 4B BF16 resident lane (read_graph_worker.py load / encode / noise / run_fast)."""

    def __init__(self, config, event):
        import numpy as np
        import torch
        from diffusers import Flux2KleinPipeline
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        self.np, self.torch, self.event = np, torch, event
        self._mu, self._timesteps = compute_empirical_mu, retrieve_timesteps
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise ValueError("admitted CUDA required")
        self.dev = torch.device("cuda")
        self.H, self.W, self.STEPS = int(config["height"]), int(config["width"]), int(config["steps"])
        self.model_path = config["model_path"]
        event("load_start")
        pipe = Flux2KleinPipeline.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.set_progress_bar_config(disable=True)
        if not pipe.config.is_distilled:
            raise ValueError("distilled Klein required")
        tr = pipe.transformer
        if len(tr.transformer_blocks) != JOINT or len(tr.single_transformer_blocks) != SINGLE:
            raise ValueError("architecture differs")
        for component in (pipe.text_encoder, tr, pipe.vae):
            component.eval().requires_grad_(False)
        self.pipe, self.tr = pipe, tr
        self.encodings = {}
        self.peak_mb = 0.0
        event("load_done")

    def reset_peak(self):
        """Per-arm peak tracking that keeps the job-wide peak (reset_peak_memory_stats loses it)."""
        torch = self.torch
        self.peak_mb = max(self.peak_mb, torch.cuda.max_memory_allocated() / 2 ** 20)
        torch.cuda.reset_peak_memory_stats()

    def job_peak(self):
        return max(self.peak_mb, self.torch.cuda.max_memory_allocated() / 2 ** 20)

    # ------------------------------------------------------------------- encode (verbatim logic)
    def encode(self, prompt):
        torch, pipe = self.torch, self.pipe
        tok = pipe.tokenizer
        formatted = tok.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        enc = tok(formatted, return_tensors="pt", padding="max_length", truncation=True, max_length=512,
                  return_offsets_mapping=True)
        mask = enc["attention_mask"][0].tolist()
        offsets = enc["offset_mapping"][0].tolist()
        ids = enc["input_ids"][0].tolist()
        with torch.inference_mode():
            c, txt_ids = pipe.encode_prompt(prompt, device=self.dev, max_sequence_length=512)
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

    def encode_all(self, prompts):
        self.pipe.text_encoder.to(self.dev)
        for p in prompts:
            if p not in self.encodings:
                self.encodings[p] = self.encode(p)
        self.event("encoded", count=len(self.encodings))

    def place(self):
        """Release the text encoder, then place transformer + VAE and wrap the native adapter."""
        torch, pipe = self.torch, self.pipe
        from saturn_pub.adapters.flux2 import Flux2KleinAdapter
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
        pipe.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        self.tr.to(self.dev)
        pipe.vae.to(self.dev)
        pipe.vae.bn.to("cpu")
        self.adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
        self.heads = self.tr.transformer_blocks[0].attn.heads
        self.n_img = (self.H // 16) * (self.W // 16)
        self.S = T + self.n_img
        self.event("placed", heads=self.heads, image_rows=self.n_img)

    # --------------------------------------------------------------------------------- inputs
    def noise(self, seed):
        torch, np, pipe, STEPS = self.torch, self.np, self.pipe, self.STEPS
        with torch.inference_mode():
            latent, img_ids = pipe.prepare_latents(1, self.tr.config.in_channels // 4, self.H, self.W, torch.bfloat16,
                                                   self.dev, torch.Generator(device="cuda").manual_seed(int(seed)))
            sched = None if getattr(pipe.scheduler.config, "use_flow_sigmas", False) else np.linspace(1.0, 1 / STEPS, STEPS)
            timesteps, n = self._timesteps(pipe.scheduler, STEPS, self.dev, sigmas=sched,
                                           mu=self._mu(image_seq_len=latent.shape[1], num_steps=STEPS))
        if n != STEPS or latent.shape[1] != self.n_img:
            raise ValueError("schedule or geometry differs")
        return latent, img_ids, timesteps.clone(), pipe.scheduler.sigmas.clone()

    def tensor_bytes(self, x):
        return x.detach().float().cpu().contiguous().numpy().tobytes()

    def init_sha(self, init):
        latent, img_ids, timesteps, sigmas = init
        return sha_bytes(("latent", self.tensor_bytes(latent)), ("img_ids", self.tensor_bytes(img_ids)),
                         ("timesteps", self.tensor_bytes(timesteps)), ("sigmas", self.tensor_bytes(sigmas)))

    def conditioning(self, spec):
        """Build (c, txt_ids) for a conditioning spec; returns (cond, info, sha256)."""
        from mdb.flux_read_panels import delete_word, hybrid_own_plan
        prompt, kind = spec["prompt"], spec["kind"]
        c_o, ids_o, rec_o = self.encodings[prompt]
        info = {}
        if kind == "native":
            cond = (c_o, ids_o)
        else:
            deleted = delete_word(prompt, spec["word"])
            if spec.get("deleted_prompt") not in (None, deleted):
                raise ValueError("recipe's deleted prompt differs from delete_word()")
            c_d, ids_d, rec_d = self.encodings[deleted]
            info["deleted_prompt"] = deleted
            if kind == "deletion":
                cond = (c_d, ids_d)
            elif kind == "hybrid_own":
                plan = hybrid_own_plan(rec_o, rec_d, spec["word"])
                if plan is None:
                    raise ValueError(f"hybrid alignment fails for {spec['word']!r}")
                c_h = c_o.clone()
                c_h[0, plan["later_rows"]] = c_d[0, plan["later_source_rows"]]
                c_h[0, plan["pad_from"]:] = c_d[0, plan["pad_from"]:]
                info.update({k: plan[k] for k in ("own_rows", "shift", "later_id_mismatch")},
                            later_rows=len(plan["later_rows"]),
                            prefix_equal=bool(self.torch.equal(c_o[0, plan["before_rows"]], c_d[0, plan["before_rows"]])))
                cond = (c_h, ids_o)
            else:
                raise ValueError("unknown conditioning kind " + kind)
        digest = sha_bytes(("conditioning", self.tensor_bytes(cond[0])), ("txt_ids", self.tensor_bytes(cond[1])))
        return cond, info, digest

    # ----------------------------------------------------------------------------------- run
    def run(self, cond, init, port=None, cuts=(), log=False):
        """read_graph_worker.run_fast without schedules, through ``port.apply(..., forwards=STEPS)``."""
        torch, adapter, STEPS = self.torch, self.adapter, self.STEPS
        c, txt_ids = cond
        latent, img_ids, timesteps, sigmas = init
        sess = adapter.session(latent=latent.clone(), conditioning=c.to(self.dev), img_ids=img_ids.clone(),
                               txt_ids=txt_ids.to(self.dev), timesteps=timesteps.clone(), sigmas=sigmas.clone())
        state = sess._state
        stats = {}
        if port is None:
            while state["step"] < STEPS:
                state = adapter.advance(state)
        else:
            with port.apply(*cuts, forwards=STEPS, log=log):
                while state["step"] < STEPS:
                    state = adapter.advance(state)
                metadata = port.metadata()
            stats = {"port_calls": port.calls, "forward_count": port.forward_count, "metadata": metadata}
            if port.calls != STEPS * BLOCKS or port.forward_count != STEPS:
                raise ValueError(f"attention call/forward count differs: {port.calls}/{port.forward_count}")
            if log and port.log:
                stats["recomputed"] = {
                    "calls": len(port.log),
                    "head_rows": sum(len(g["heads"]) * g["receiver_rows"] for e in port.log for g in e["groups"]),
                    "removed_reads": sum(g["removed_reads"] * len(g["heads"]) for e in port.log for g in e["groups"])}
        sess._state = dict(state)
        lat = sess.read("latent").detach().float().cpu().numpy()[0]
        image = adapter.decode(sess, height=self.H, width=self.W)
        return image, lat, stats


def pixels(np, image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compare(np, a_rgb, a_lat, b_rgb, b_lat):
    d = np.abs(a_rgb.astype(np.float32) / 255.0 - b_rgb.astype(np.float32) / 255.0).mean(-1)
    return dict(px_mad=float(d.mean()), px_max=float(d.max()),
                latent_rel=float(np.linalg.norm(a_lat - b_lat) / np.linalg.norm(b_lat)),
                rgb_equal=bool(np.array_equal(a_rgb, b_rgb)), latent_equal=bool(np.array_equal(a_lat, b_lat)))


def software_lane(lane, modules, payload):
    import diffusers
    import numpy
    import PIL
    import torch
    import transformers
    from importlib.metadata import version
    from mdb.flux_read_panels import fingerprint
    return {"python": platform.python_version(), "torch": torch.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__,
            "saturn_pub": version("saturn-pub"), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0), "host": platform.node(),
            "adapter_execution_sha256": fingerprint(lane.adapter.execution),
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "worker_sha256": sha(payload / WORKER),
            **{f"mdb.{name}_sha256": modules[name]["sha256"] for name in MODULES}}


# ================================================================================== panel mode

def panel_mode(config, out, payload, event, started, modules):
    from mdb import attention_port as ap
    from mdb.attention_panels import port_cuts
    from mdb import flux_read_panels as frp
    passes = []
    for p in config["passes"]:
        plan_json = json.loads((payload / p["plan_file"]).read_text())
        panel = frp.load_plan(plan_json, p["plan_sha256"])        # refuses a plan whose hash differs
        lease = frp.check_lease(panel, p["lease"])
        if p["precision"] not in ("native", "legacy"):
            raise ValueError("precision must be native or legacy")
        passes.append(dict(p, panel=panel, lease=lease))
        event("plan_ok", tag=p["tag"], plan_sha256=p["plan_sha256"], lease=lease["lease"], of=lease["of"],
              branches=len(lease["branches"]), precision=p["precision"])
    items, prompts = config["items"], config["prompts"]
    retain = set(config.get("retain", []))
    if retain - set(frp.RETAIN_OPTIONS):
        raise ValueError(f"retain options are {frp.RETAIN_OPTIONS}")
    budget = float(config["max_wall_seconds"])

    lane = Lane(config, event)
    import numpy as np
    import torch
    texts = set()
    for p in passes:
        for _, arm_id in p["lease"]["branches"]:
            spec = items[p["panel"].by_id()[arm_id].item]
            texts.add(prompts[spec["prompt"]])
            if spec.get("word"):
                texts.add(frp.delete_word(prompts[spec["prompt"]], spec["word"]))
    lane.encode_all(sorted(texts))
    save(out / "tokens.json", {k: v[2] for k, v in lane.encodings.items()})
    lane.place()
    if lane.heads != 24:
        raise ValueError("unexpected head count")
    manifest = checkpoint_manifest(lane.model_path, bool(config.get("hash_checkpoint")))
    save(out / "checkpoint.json", manifest)
    pointers = parameter_pointers(lane.model_path, manifest)
    save(out / "parameters.json", pointers)
    manifest_sha, pointers_sha = sha(out / "checkpoint.json"), sha(out / "parameters.json")
    event("checkpoint", seconds=manifest["seconds"], full_hash=manifest["full_hash"])
    software = software_lane(lane, modules, payload)
    backend = ap.FluxBackend(lane.tr, text_rows=T, identity=lane.adapter.model_identity)
    for p in passes:
        universe_ok = all(0 <= b < BLOCKS and 0 <= h < lane.heads for a in p["panel"].arms for b, h in (a.units or ()))
        if not universe_ok:
            raise ValueError("plan units outside the model's (block, head) grid")

    # row tables per prompt: image, text, own/others for every item word of that prompt
    def table_of(pid):
        rec = lane.encodings[prompts[pid]][2]
        word_rows = {}
        for item, spec in items.items():
            if spec["prompt"] == pid and spec.get("word"):
                rows = sorted(r for w in rec["words"] if w["word"] == spec["word"] for r in w["rows"])
                if not rows:
                    raise ValueError(f"word {spec['word']!r} has no rows in {pid}")
                word_rows[spec["word"]] = rows
        return frp.row_table(word_rows, T, lane.S)
    tables = {pid: table_of(pid) for pid in prompts if prompts[pid] in lane.encodings}

    inits, cond_cache = {}, {}

    def init_of(context):
        if context not in inits:
            seed = int(config["contexts"][context]["seed"])
            init = lane.noise(seed)
            inits[context] = (init, lane.init_sha(init), seed)
        return inits[context]

    def cond_of(kind, pid, word):
        key = (kind, pid, word)
        if key not in cond_cache:
            spec = {"kind": kind, "prompt": prompts[pid], "word": word}
            cond, info, digest = lane.conditioning(spec)
            spec.update(prompt_id=pid, deleted_prompt=info.get("deleted_prompt"), info=info)
            cond_cache[key] = (cond, spec, digest)
        return cond_cache[key]

    def condition_key(arm, spec):
        if arm.condition is None:
            return ("native", spec["prompt"], None)
        kind, _, word = arm.condition.partition(":")
        return ({"iso": "hybrid_own", "delete": "deletion"}[kind], spec["prompt"], word)

    def recipe(arm_info, cond_spec, cond_sha, context, port_section, expected, blocks):
        init, init_sha, seed = init_of(context)
        body = dict(schema=frp.RECIPE_SCHEMA, arm=arm_info,
                    model=dict(model_identity=lane.adapter.model_identity, model_path=lane.model_path,
                               pipeline="Flux2KleinPipeline", dtype="bfloat16", distilled=True,
                               checkpoint_files_sha256=manifest_sha, parameter_pointers_sha256=pointers_sha,
                               parameter_blocks=blocks,
                               text_encoder="pipe.encode_prompt(prompt, max_sequence_length=512) on cuda, then released"),
                    software=software,
                    input=dict(prompt_id=cond_spec["prompt_id"], prompt=cond_spec["prompt"],
                               conditioning={k: cond_spec[k] for k in ("kind", "word", "deleted_prompt", "info")},
                               tokenizer=dict(chat_template=True, add_generation_prompt=True, enable_thinking=False,
                                              padding="max_length", max_length=512)),
                    noise=dict(seed=seed, generator="torch.Generator(device='cuda').manual_seed(seed)",
                               height=lane.H, width=lane.W, steps=lane.STEPS,
                               schedule="retrieve_timesteps(sigmas=linspace(1, 1/steps), mu=compute_empirical_mu)"),
                    port=port_section,
                    expected=dict(expected, conditioning_sha256=cond_sha, init_latent_sha256=init_sha))
        return frp.seal_recipe(body)

    def finish(tag, context, name, image, lat, cond_sha, extra):
        safe = frp.safe_name(name)
        png = out / "images" / tag / context / f"{safe}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        image.save(png)
        rgb = pixels(np, image)
        if "latents" in retain:
            lp = out / "latents" / tag / context / f"{safe}.npy"
            lp.parent.mkdir(parents=True, exist_ok=True)
            np.save(lp, lat.astype(np.float32))
        hashes = dict(latent_sha256=hashlib.sha256(lat.astype(np.float32).tobytes()).hexdigest(),
                      latent_shape=list(lat.shape), image_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                      png_sha256=sha(png))
        return rgb, hashes, png

    def legacy_compare(ref_dir, pid, seed, arm_id, rgb):
        if not ref_dir:
            return None
        name = frp.legacy_file_name(pid, seed, arm_id)
        path = Path(ref_dir) / name if name else None
        if path is None or not path.exists():
            return {"reference": name, "found": False}
        from PIL import Image
        ref = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        d = np.abs(rgb.astype(np.float32) - ref.astype(np.float32)).mean(-1) / 255.0
        return {"reference": name, "found": True, "identical": bool(np.array_equal(rgb, ref)),
                "px_mad": float(d.mean()), "px_max": float(d.max())}

    # ------------------------------------------------------------------------------- roots
    roots, root_rows = {}, []
    ref_dirs = [p.get("reference_dir") for p in passes if p.get("reference_dir")]
    for p in passes:
        for context, kind, pid, word in frp.lease_roots(p["panel"], p["lease"], items):
            key = (context, kind, pid, word)
            if key in roots:
                continue
            ckind = {"native": "native", "del": "deletion", "iso": "hybrid_own"}[kind]
            cond, cspec, csha = cond_of(ckind, pid, word)
            init, init_sha, seed = init_of(context)
            torch.cuda.synchronize()
            t0 = time.monotonic()
            image, lat, _ = lane.run(cond, init)
            torch.cuda.synchronize()
            seconds = time.monotonic() - t0
            name = f"{kind}:{word}" if word else f"native:{pid}"
            rgb, hashes, png = finish("roots", context, name, image, lat, csha, {})
            roots[key] = (rgb, lat)
            rec = recipe(dict(pass_tag="roots", plan_sha256=None, lease_sha256=None, context=context, arm=name,
                              kind=kind, role="root", condition=None, reference=None, item=word or pid),
                         cspec, csha, context, None, hashes, [])
            save(out / "recipes" / "roots" / context / f"{frp.safe_name(name)}.json", rec)
            row = dict(context=context, root=name, kind=kind, prompt=pid, word=word, seconds=seconds, **hashes,
                       conditioning_sha256=csha, init_latent_sha256=init_sha, recipe=rec["fingerprint"])
            if ref_dirs:
                old = {"native": f"native:{pid}", "del": f"del:{word}", "iso": f"iso:{word}"}[kind]
                row["legacy_reference"] = legacy_compare(ref_dirs[0], pid, seed, old, rgb)
            root_rows.append(row)
            event("root", **{k: v for k, v in row.items() if k in ("context", "root", "seconds", "latent_sha256")})
    save(out / "roots.json", root_rows)

    def root_of(context, kind, pid, word):
        return roots[(context, kind, pid, word)]

    # ------------------------------------------------------------------------------ branches
    summary_passes = []
    stop = False
    for p in passes:
        panel, lease, tag, precision = p["panel"], p["lease"], p["tag"], p["precision"]
        zero_method = p.get("zero_method", "delta")
        ids = panel.by_id()
        ports = {pid: ap.AttentionPort(backend, rows=tables[pid], zero_method=zero_method, precision=precision)
                 for pid in tables}
        rows, done = [], {}
        names = set()
        for context, arm_id in lease["branches"]:
            if time.monotonic() - started > budget:
                event("budget_stop", tag=tag, context=context, arm=arm_id)
                stop = True
                break
            arm = ids[arm_id]
            spec = items[arm.item]
            pid, word = spec["prompt"], spec.get("word")
            ckey = condition_key(arm, spec)
            cond, cspec, csha = cond_of(*ckey)
            init, init_sha, seed = init_of(context)
            cuts = port_cuts(arm)
            port = ports[pid]
            torch.cuda.synchronize()
            lane.reset_peak()
            t0 = time.monotonic()
            image, lat, stats = lane.run(cond, init, port, cuts, log=precision != "legacy")
            torch.cuda.synchronize()
            seconds = time.monotonic() - t0
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            safe = frp.safe_name(arm_id)
            if (context, safe) in names:
                raise ValueError("file-name collision: " + arm_id)
            names.add((context, safe))
            rgb, hashes, png = finish(tag, context, arm_id, image, lat, csha, {})
            done[arm_id] = (rgb, lat)
            # references: the arm's own reference arm, and the roots
            ref = arm.reference
            if ref in (None, "native") or ref == f"native:{pid}":
                ref_img = root_of(context, "native", pid, None)
            elif ref.startswith("iso:"):
                ref_img = root_of(context, "iso", pid, ref[4:])
            elif ref.startswith("del:"):
                ref_img = root_of(context, "del", pid, ref[4:])
            else:
                raise ValueError(f"unresolvable reference {ref!r} for {arm_id}")
            row = dict(context=context, seed=seed, arm=arm_id, kind=arm.kind, role=arm.role, item=arm.item,
                       cut_type=arm.cut_type, unit_set=arm.unit_set, condition=arm.condition, reference=ref,
                       seconds=seconds, peak_vram_mb=peak, port_calls=stats["port_calls"],
                       forward_count=stats["forward_count"], recomputed=stats.get("recomputed"), **hashes,
                       vs_reference=compare(np, rgb, lat, *ref_img),
                       vs_native=compare(np, rgb, lat, *root_of(context, "native", pid, None)))
            if word:
                row["vs_del"] = compare(np, rgb, lat, *root_of(context, "del", pid, word))
                row["vs_iso"] = compare(np, rgb, lat, *root_of(context, "iso", pid, word))
            if arm.kind in ("condition", "deletion", "native"):
                root_kind = {"condition": "iso", "deletion": "del", "native": "native"}[arm.kind]
                r_rgb, r_lat = root_of(context, root_kind, pid, word if root_kind != "native" else None)
                row["equals_root"] = bool(np.array_equal(rgb, r_rgb) and np.array_equal(lat, r_lat))
            if arm.kind == "floor_cut_recompute" and arm.floor_of in done:
                row["vs_floor_of"] = compare(np, rgb, lat, *done[arm.floor_of])
            if p.get("reference_dir"):
                row["legacy_reference"] = legacy_compare(p["reference_dir"], pid, seed, arm_id, rgb)
            cut_dicts = [c.to_dict() for c in cuts]
            names_used = sorted({x for c in cut_dicts for x in (*c["receivers"], *c["sources"]) if isinstance(x, str)})
            if not cuts:
                blocks = []
            elif all(c.heads is not None and all(b is not None for b, _ in c.heads) for c in cuts):
                blocks = sorted({b for c in cuts for b, _ in c.heads})
            else:
                blocks = list(range(BLOCKS))
            port_section = dict(backend=backend.name, precision=precision, zero_method=zero_method, empty="raise",
                                forwards=lane.STEPS, text_rows=T, cuts=cut_dicts,
                                rows={n: frp.runs(tables[pid][n]) for n in names_used if n != "all"},
                                metadata_cuts_sha256=frp.fingerprint(stats["metadata"]["cuts"]))
            rec = recipe(dict(pass_tag=tag, plan_sha256=panel.sha256(), lease_sha256=lease["lease_sha256"],
                              context=context, arm=arm_id, kind=arm.kind, role=arm.role, condition=arm.condition,
                              reference=ref, item=arm.item),
                         cspec, csha, context, port_section, hashes, [int(b) for b in blocks])
            save(out / "recipes" / tag / context / f"{safe}.json", rec)
            row["recipe"] = rec["fingerprint"]
            rows.append(row)
            event("arm", tag=tag, context=context, arm=arm_id, seconds=round(seconds, 3), peak_mb=round(peak),
                  px_ref=round(row["vs_reference"]["px_mad"], 5),
                  legacy=(row.get("legacy_reference") or {}).get("identical"))
        save(out / f"metrics.{tag}.json", rows)
        results = {m: frp.results_from_rows(panel, rows, metric=m) for m in ("px_mad", "latent_rel")}
        for m, rec in results.items():
            save(out / f"results.{tag}.{m}.json", rec)
        summary_passes.append(dict(tag=tag, plan_sha256=panel.sha256(), lease=lease["lease"], of=lease["of"],
                                   lease_sha256=lease["lease_sha256"], precision=precision, zero_method=zero_method,
                                   planned=len(lease["branches"]), executed=len(rows),
                                   seconds_mean=float(np.mean([r["seconds"] for r in rows])) if rows else None,
                                   seconds_max=float(np.max([r["seconds"] for r in rows])) if rows else None))
        if stop:
            break
    return dict(passes=summary_passes, roots=len(root_rows), budget_stopped=stop, software=software,
                model_identity=lane.adapter.model_identity, checkpoint_seconds=manifest["seconds"],
                peak_vram_allocated_mb=lane.job_peak())


# ============================================================================= regenerate mode

def regenerate_mode(config, out, payload, event, started, modules, wanted, recipes_dir):
    from mdb import attention_port as ap
    from mdb import flux_read_panels as frp
    import numpy as np
    import torch
    recipes = {}
    for path in sorted(Path(recipes_dir).glob("*.json")):
        r = frp.validate_recipe(json.loads(path.read_text()))
        recipes[(r["arm"]["context"], r["arm"]["arm"])] = r
    chosen = []
    for item in wanted:
        context, _, arm = item.partition(":")
        if (context, arm) not in recipes:
            raise ValueError(f"no recipe for {item}")
        chosen.append(recipes[(context, arm)])
    lane = Lane(config, event)
    texts = set()
    for r in chosen:
        texts.add(r["input"]["prompt"])
        if r["input"]["conditioning"]["kind"] != "native":
            texts.add(frp.delete_word(r["input"]["prompt"], r["input"]["conditioning"]["word"]))
    lane.encode_all(sorted(texts))
    lane.place()
    manifest = checkpoint_manifest(lane.model_path, bool(config.get("hash_checkpoint")))
    save(out / "checkpoint.json", manifest)
    software = software_lane(lane, modules, payload)
    backend = ap.FluxBackend(lane.tr, text_rows=T, identity=lane.adapter.model_identity)
    rows = []
    for r in chosen:
        if r["model"]["model_identity"] != lane.adapter.model_identity:
            raise ValueError("recipe model identity differs from the loaded model; refusing")
        if (r["noise"]["height"], r["noise"]["width"], r["noise"]["steps"]) != (lane.H, lane.W, lane.STEPS):
            raise ValueError("recipe geometry differs from the run config")
        cspec = dict(r["input"]["conditioning"], prompt=r["input"]["prompt"])
        cond, info, csha = lane.conditioning(cspec)
        init = lane.noise(r["noise"]["seed"])
        init_sha = lane.init_sha(init)
        port, cuts = None, ()
        if r["port"] is not None:
            rows_table = {n: frp.unruns(v) for n, v in r["port"]["rows"].items()}
            port = ap.AttentionPort(backend, rows=rows_table, zero_method=r["port"]["zero_method"],
                                    precision=r["port"]["precision"], empty=r["port"]["empty"])
            cuts = frp.cut_from_dict_all(r["port"]["cuts"])
        torch.cuda.synchronize()
        t0 = time.monotonic()
        image, lat, stats = lane.run(cond, init, port, cuts)
        torch.cuda.synchronize()
        seconds = time.monotonic() - t0
        safe = frp.safe_name(r["arm"]["arm"])
        png = out / "images" / r["arm"]["context"] / f"{safe}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        image.save(png)
        rgb = pixels(np, image)
        observed = dict(conditioning_sha256=csha, init_latent_sha256=init_sha,
                        latent_sha256=hashlib.sha256(lat.astype(np.float32).tobytes()).hexdigest(),
                        image_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), png_sha256=sha(png))
        row = frp.compare_regeneration(r, observed, lane=software)
        row.update(seconds=seconds, port_calls=stats.get("port_calls"), forward_count=stats.get("forward_count"),
                   pass_tag=r["arm"]["pass_tag"], precision=None if r["port"] is None else r["port"]["precision"])
        if stats.get("metadata") is not None:
            row["port_cuts_equal"] = frp.fingerprint(stats["metadata"]["cuts"]) == r["port"]["metadata_cuts_sha256"]
        rows.append(row)
        event("regenerated", context=row["context"], arm=row["arm"], match=row["match"], seconds=round(seconds, 3))
    save(out / "regenerate.json", rows)
    return dict(regenerated=len(rows), all_match=all(r["match"] for r in rows), software=software,
                model_identity=lane.adapter.model_identity, checkpoint_seconds=manifest["seconds"],
                peak_vram_allocated_mb=lane.job_peak())


# ===================================================================================== main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", nargs="+", metavar="CONTEXT:ARM_ID",
                        help="rebuild these arms from their recipes and compare hashes")
    parser.add_argument("--recipes", default="recipes", help="directory of sealed recipe JSON files")
    args = parser.parse_args(argv)
    payload = Path.cwd()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    job = os.environ.get("MRUN_JOB_ID", "local")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    from mdb.flux_read_panels import RUN_SCHEMA
    if config.get("schema") != RUN_SCHEMA:
        raise ValueError("not a FLUX read-panel run config")
    mode = "regenerate" if args.regenerate else "panel"
    if mode != config.get("mode", "panel"):
        raise ValueError(f"run config is for mode {config.get('mode')!r}, the command asks for {mode!r}")
    scratch = Path(config["output_scratch"]) / job
    scratch.mkdir(parents=True, exist_ok=False)
    out = scratch / ("flux-read-panel" if mode == "panel" else "flux-read-panel-regenerate")
    out.mkdir()
    started = time.monotonic()
    timeline = []

    def event(kind, **fields):
        row = dict(kind=kind, t=round(time.monotonic() - started, 3), **fields)
        timeline.append(row)
        print("FLUXPANEL " + json.dumps(row, sort_keys=True, default=str), flush=True)

    modules = module_proof(payload)
    print("FLUXPANEL_MODULES " + json.dumps(modules, sort_keys=True), flush=True)
    save(out / "modules.json", modules)
    save(out / "run-config.json", config)
    if mode == "panel":
        result = panel_mode(config, out, payload, event, started, modules)
    else:
        result = regenerate_mode(config, out, payload, event, started, modules, args.regenerate, args.recipes)
    summary = dict(schema="mdb-flux-read-panel-summary-v1", mode=mode, job_id=job,
                   request_sha256=request.get("request_sha256"), worker_sha256=sha(payload / WORKER),
                   modules=modules,
                   worker_seconds=time.monotonic() - started, output_root=str(out), **result)
    save(out / "summary.json", summary)
    save(out / "timeline.json", timeline)
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    save(out / "files.json", files)
    print("FLUXPANEL_SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in ("software", "modules")},
                                            sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
