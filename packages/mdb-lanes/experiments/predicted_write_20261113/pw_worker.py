"""Predicted-write capture worker, FLUX.2 Klein 4B (experiment predicted_write_20261113).

One resident Klein lane (the read-panel lane's load / encode / noise / conditioning code, imported
from ``flux_read_panel_worker.py`` shipped beside this file). Per word, seed 4242, 256 px, 4 steps:

* roots without the port: native, deleted, iso (hybrid "own") images;
* score run (iso conditioning; then native conditioning) with an observer that computes, per
  (step, block, head), the predicted write change of cutting the word's rows for the receiver
  groups the lane's arms cut (``image`` and ``others:<w>``), for both cut types, from the native
  FP32 softmax of that call (no cut runs), plus the read-graph write mass ``tok_w``;
* closure cells: single-head, single-(step, block) cuts through ``mdb_lanes.attention_port`` in both
  cut types (word rows, and all text rows as a large-change check) with the renormalising sham, and
  the measured change at the attention output map's input and output against the identity;
* routing / payload split: native (word present) and deleted runs at matched geometry (keys
  aligned by ``predicted_write.deletion_alignment``), exact split of the image rows' attention write;
* value swap (G3): single blocks, image receivers read the deleted run's text-row values through
  the live attention map (delta form), plus a recompute sham; native and iso conditioning.

Writes JSON / npz summaries and PNGs only. No retries.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import flux_read_panel_worker as base  # noqa: E402  (payload file beside this worker)

JOINT, SINGLE, BLOCKS, T = 5, 20, 25, 512
WORKER = "pw_worker.py"
MODULES = ("attention_port", "attention_panels", "flux_read_panels", "predicted_write")
SCHEMA = "mdb-predicted-write-capture-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    base.save(path, value)


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
    rows["flux_read_panel_worker"] = {"sha256": sha(payload / "flux_read_panel_worker.py"),
                                      "declared": request["source_hashes"].get("flux_read_panel_worker.py")}
    return rows


class Gates:
    """Forward pre-hooks on every block: the outer attention gate g (image, text) of the current call."""

    def __init__(self, tr):
        from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
        self.split, self.now, self.handles = Flux2Modulation.split, {}, []
        blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
        for i, b in enumerate(blocks):
            self.handles.append(b.register_forward_pre_hook(self._hook(i), with_kwargs=True))

    def _hook(self, index):
        def hook(module, args, kwargs):
            if index < JOINT:
                gi = self.split(kwargs["temb_mod_img"], 2)[0][2]
                gt = self.split(kwargs["temb_mod_txt"], 2)[0][2]
            else:
                gi = gt = self.split(kwargs["temb_mod"], 1)[0][2]
            self.now[index] = (gi.reshape(-1).float(), gt.reshape(-1).float())
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class Maps:
    """Per-head output maps [H, D, C] (FP32) of the image and text attention writes, per block (cached)."""

    def __init__(self, backend, heads):
        self.backend, self.heads, self.cache = backend, heads, {}

    def __call__(self, block):
        from mdb_lanes.predicted_write import head_maps
        if block not in self.cache:
            wi, wt = self.backend._out_weights(block)
            mi = head_maps(wi.float(), self.heads).contiguous()
            mt = mi if wt is wi else head_maps(wt.float(), self.heads).contiguous()
            self.cache = {block: (mi, mt)}          # keep one block resident (memory)
        return self.cache[block]


class OutTap:
    """Forward hooks on the attention output maps: input (head outputs) and output at chosen (step, block)."""

    def __init__(self, tr, backend):
        self.backend, self.want, self.got, self.handles = backend, set(), {}, []
        inner = tr.transformer_blocks[0].attn.inner_dim
        for b in range(JOINT):
            attn = tr.transformer_blocks[b].attn
            self.handles.append(attn.to_out[0].register_forward_hook(self._hook(b, "img", inner)))
            self.handles.append(attn.to_add_out.register_forward_hook(self._hook(b, "txt", inner)))
        for i, blk in enumerate(tr.single_transformer_blocks):
            self.handles.append(blk.attn.to_out.register_forward_hook(self._hook(JOINT + i, "all", inner)))

    def _hook(self, block, part, inner):
        def hook(module, args, output):
            if not self.want:
                return
            key = (getattr(self.backend, "step", None), block)
            if key not in self.want:
                return
            x, y = args[0][0, :, :inner].detach().float().cpu(), output[0].detach().float().cpu()
            rec = self.got.setdefault(key, {})
            if part == "all":
                rec.update(txt_in=x[:T], img_in=x[T:], txt_out=y[:T], img_out=y[T:])
            else:
                rec[part + "_in"], rec[part + "_out"] = x, y
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def half_ulp_bf16(torch, x):
    """Half a bfloat16 unit in the last place of |x| (8 significant bits)."""
    ax = x.abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(ax)) - 8)


def make_swap_backend(ap):
    class SwapBackend(ap.FluxBackend):
        """FluxBackend placement; in single blocks image receivers read replaced text-row values."""

        def configure(self, store, rows, mode):
            self.store, self.sel, self.mode, self.touched = store, rows, mode, 0

        def dispatch(self, q, k, v, *args, **kwargs):
            import torch
            port = self.port
            self._validate(q, k, v)
            port.calls += 1
            step, block = self.step, self._block
            port.plan(step, block)
            out = self.original(q, k, v, *args, **kwargs)
            if block < JOINT:
                return out
            if args or kwargs.get("attn_mask") is not None:
                raise ValueError("value swap: native call carries a mask or positional extras")
            scale = kwargs.get("scale")
            scale = scale if scale is not None else q.shape[-1] ** -0.5
            qh = q[0, T:].float().permute(1, 0, 2)
            kh, vh = k[0].float().permute(1, 0, 2), v[0].float().permute(1, 0, 2)
            p = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * scale, dim=-1)   # [H, I, S]
            if self.mode == "sham":
                new = torch.matmul(p, vh)
            else:
                sel = self.sel
                vs = self.store[(step, block)].float()[:, sel]                         # [H, |sel|, D]
                new = out[0, T:].float().permute(1, 0, 2) + torch.matmul(p[:, :, sel], vs - vh[:, sel])
            out = out.clone()
            out[0, T:] = new.permute(1, 0, 2).to(out.dtype)
            self.touched += 1
            return out
    return SwapBackend


def main():
    payload = Path.cwd()
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("not a predicted-write capture config")
    job = os.environ.get("MRUN_JOB_ID", "local")
    out = Path(config["output_scratch"]) / job / "predicted-write"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    timeline = []

    def event(kind, **fields):
        row = dict(kind=kind, t=round(time.monotonic() - started, 3), **fields)
        timeline.append(row)
        print("PWCAP " + json.dumps(row, sort_keys=True, default=str), flush=True)

    modules = module_proof(payload)
    save(out / "modules.json", modules)
    save(out / "run-config.json", config)
    budget = float(config["max_wall_seconds"])

    from mdb_lanes import attention_port as ap
    from mdb_lanes import flux_read_panels as frp
    from mdb_lanes import predicted_write as PW
    lane = base.Lane(config, event)
    import numpy as np
    import torch
    words = config["words"]                                     # {word: prompt id}
    prompts = config["prompts"]
    texts = set()
    for w, pid in words.items():
        texts.add(prompts[pid])
        texts.add(frp.delete_word(prompts[pid], w))
    lane.encode_all(sorted(texts))
    save(out / "tokens.json", {k: v[2] for k, v in lane.encodings.items()})
    lane.place()
    dev, H, S = lane.dev, lane.heads, lane.S
    backend = ap.FluxBackend(lane.tr, text_rows=T, identity=lane.adapter.model_identity)
    gates = Gates(lane.tr)
    maps = Maps(backend, H)
    tap = OutTap(lane.tr, backend)
    seed = int(config["seed"])
    init = lane.noise(seed)
    init_sha = lane.init_sha(init)
    STEPS = lane.STEPS

    def run(cond, port=None, cuts=(), observers=(), log=False):
        c, txt_ids = cond
        latent, img_ids, timesteps, sigmas = init
        sess = lane.adapter.session(latent=latent.clone(), conditioning=c.to(dev), img_ids=img_ids.clone(),
                                    txt_ids=txt_ids.to(dev), timesteps=timesteps.clone(), sigmas=sigmas.clone())
        state = sess._state
        if port is None:
            while state["step"] < STEPS:
                state = lane.adapter.advance(state)
        else:
            with port.apply(*cuts, observers=list(observers), forwards=STEPS, log=log):
                while state["step"] < STEPS:
                    state = lane.adapter.advance(state)
            if port.calls != STEPS * BLOCKS or port.forward_count != STEPS:
                raise ValueError(f"call/forward count differs: {port.calls}/{port.forward_count}")
        sess._state = dict(state)
        lat = sess.read("latent").detach().float().cpu().numpy()[0]
        image = lane.adapter.decode(sess, height=lane.H, width=lane.W)
        return image, lat

    def keep_image(word, name, image, lat):
        png = out / "images" / word / f"{frp.safe_name(name)}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        image.save(png)
        rgb = base.pixels(np, image)
        return rgb, lat, {"png": str(png.relative_to(out)), "png_sha256": sha(png),
                          "latent_sha256": hashlib.sha256(lat.astype(np.float32).tobytes()).hexdigest(),
                          "image_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}

    summary_words = {}
    for w, pid in words.items():
        if time.monotonic() - started > budget:
            event("budget_stop", word=w)
            break
        t_word = time.monotonic()
        rec_o = lane.encodings[prompts[pid]][2]
        deleted = frp.delete_word(prompts[pid], w)
        rec_d = lane.encodings[deleted][2]
        own = sorted(r for x in rec_o["words"] if x["word"] == w for r in x["rows"])
        table = frp.row_table({w: own}, T, S)
        al = PW.deletion_alignment(rec_o, rec_d, w, T)
        if al is None:
            raise ValueError(f"deletion alignment fails for {w}")
        conds = {}
        for kind in ("native", "deletion", "hybrid_own"):
            cond, info, digest = lane.conditioning({"kind": kind, "prompt": prompts[pid], "word": w})
            conds[kind] = (cond, digest)
        m_own = torch.zeros(S, device=dev)
        m_own[own] = 1.0
        others = torch.tensor(table[f"others:{w}"], device=dev)
        rows = {"word": w, "prompt": pid, "own_rows": own, "alignment": {k: al[k] for k in ("word", "shift",
                                                                                         "word_partner_rows")},
                "n_real": rec_o["n_real"], "conditioning_sha256": {k: v[1] for k, v in conds.items()}}
        # ------------------------------------------------------------------------------ roots
        roots = {}
        for name, kind in (("native", "native"), ("del", "deletion"), ("iso", "hybrid_own")):
            image, lat = run(conds[kind][0])
            roots[name] = keep_image(w, name, image, lat)
        rows["roots"] = {k: v[2] for k, v in roots.items()}
        rows["mad_native_del"] = base.compare(np, *roots["native"][:2], *roots["del"][:2])["px_mad"]

        # ------------------------------------------------------------------------- score observer
        closure_cells = [tuple(c) for c in config.get("closure", {}).get("cells", [])] \
            if w in config.get("closure", {}).get("words", []) else []
        saved_cells = {}
        capture_cells = [True]

        def score_reduce(read):
            s, b = read.step, read.block
            q, k, v = read.q[0].float(), read.k[0].float(), read.v[0].float()
            p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * read.scale, dim=-1)
            a = torch.matmul(p, v)
            gi, gt = gates.now[b]
            mi, mt = maps(b)
            res = {}
            dz = PW.zero_cut_delta(p, v, m_own)
            dm = PW.mask_cut_delta(p, v, m_own, a)
            for kind, d in (("zero", dz), ("mask", dm)):
                res[kind + "_img"] = PW.write_norms(d[:, T:], mi, gi).sum(-1).cpu().numpy()
                res[kind + "_oth"] = PW.write_norms(d[:, others], mt, gt).sum(-1).cpu().numpy()
            wn_img = torch.linalg.vector_norm(torch.matmul(v, mi), dim=-1)          # [H, S]  |O_h v_j| (no gate)
            res["tok_w"] = (p[:, T:, :T] * wn_img[:, None, :T]).mean(1).cpu().numpy()
            res["tok_p"] = p[:, T:, :T].mean(1).cpu().numpy()
            ps = (p * m_own).sum(-1)
            res["max_pS"] = ps.max(-1).values.cpu().numpy()
            if (s, b) in closure_cells and capture_cells[0]:      # iso run only: the cut runs' operands
                saved_cells[(s, b)] = {"q": read.q[0].detach().clone(), "k": read.k[0].detach().clone(),
                                       "v": read.v[0].detach().clone(), "scale": read.scale,
                                       "gate": (gi.clone(), gt.clone())}
            return res

        def stack(obs, key):
            return np.stack([np.stack([obs.records[(s, b)][key] for b in range(BLOCKS)]) for s in range(STEPS)])

        score = {}
        tap.want = set(closure_cells)
        tap.got = {}
        gate_check = {}
        handles = []
        for (s, b) in closure_cells:
            if b >= JOINT:
                blk = lane.tr.single_transformer_blocks[b - JOINT]

                def bhook(module, args, kwargs, output, s=s, b=b):
                    if (s, b) not in tap.want or getattr(backend, "step", None) != s:
                        return
                    x = kwargs["hidden_states"][0].float()
                    gi, _ = gates.now[b]
                    got = tap.got.get((s, b), {})
                    if "img_out" in got:
                        upd = output[0].float() - x
                        pred = torch.cat([got["txt_out"], got["img_out"]]).to(dev) * gi
                        # the block computes bf16(x + bf16(g * o)): two roundings
                        bound = half_ulp_bf16(torch, output[0].float()) + half_ulp_bf16(torch, pred)
                        err = (upd - pred).abs()
                        gate_check[f"{s}:{b}"] = {
                            "rel": float(torch.linalg.vector_norm(upd - pred) / torch.linalg.vector_norm(upd)),
                            "within_bf16_bound_frac": float((err <= bound * (1 + 1e-6)).float().mean()),
                            "max_err_over_bound": float((err / bound).max())}
                handles.append(blk.register_forward_hook(bhook, with_kwargs=True))
        for cname, ckind in (("iso", "hybrid_own"), ("native", "native")):
            obs = ap.observe(score_reduce, name="score")
            port = ap.AttentionPort(backend, rows=table)
            image, lat = run(conds[ckind][0], port, observers=[obs])
            rgb = base.pixels(np, image)
            same = bool(np.array_equal(rgb, roots[cname][0]) and np.array_equal(lat, roots[cname][1]))
            arrays = {k: stack(obs, k) for k in ("zero_img", "zero_oth", "mask_img", "mask_oth", "tok_w", "tok_p",
                                                 "max_pS")}
            np.savez_compressed(out / f"score.{w}.{cname}.npz", **arrays)
            score[cname] = {"observer_run_equals_root": same,
                            "max_pS_max": float(arrays["max_pS"].max())}
            if cname == "iso":
                native_cells = {k: dict(v) for k, v in tap.got.items()}
                tap.want = set()
                capture_cells[0] = False
            event("score", word=w, cond=cname, equals_root=same)
        for h in handles:
            h.remove()
        rows["score"] = score
        rows["single_block_gate_check_rel"] = gate_check

        # ------------------------------------------------------------------------------- closure
        if closure_cells:
            zi = np.load(out / f"score.{w}.iso.npz")["zero_img"]
            cells_out = []
            for (s, b) in closure_cells:
                h = int(np.argmax(zi[s, b]))
                cell = saved_cells[(s, b)]
                q, k, v = cell["q"].float(), cell["k"].float(), cell["v"].float()
                p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * cell["scale"], dim=-1)
                a = torch.matmul(p, v)
                gi, gt = cell["gate"]
                mi, mt = maps(b)
                nat = native_cells[(s, b)]
                D = q.shape[-1]
                hs = slice(h * D, (h + 1) * D)
                runs = {}
                for src_name, src_rows in (("own", own), ("text", list(range(T)))):
                    m = torch.zeros(S, device=dev)
                    m[src_rows] = 1.0
                    srcs = (f"own:{w}",) if src_name == "own" else ("text",)
                    for ctype in ("zero", "renorm"):
                        c = ap.cut(("image", f"others:{w}"), srcs, heads=[(b, h)], blocks=[b], steps=[s],
                                   renormalize=ctype == "renorm")
                        runs[(src_name, ctype)] = (c, m)
                c_sham = ap.sham(runs[("own", "renorm")][0])
                results = {}
                for key, cut_spec in [(k_, v_[0]) for k_, v_ in runs.items()] + [(("sham", "renorm"), c_sham)]:
                    tap.want, tap.got = {(s, b)}, {}
                    port = ap.AttentionPort(backend, rows=table, zero_method="delta")
                    run(conds["hybrid_own"][0], port, cuts=[cut_spec])
                    results[key] = tap.got[(s, b)]
                tap.want = set()
                sham = results[("sham", "renorm")]
                for (src_name, ctype), (c, m) in runs.items():
                    got = results[(src_name, ctype)]
                    if ctype == "zero":
                        pred = PW.zero_cut_delta(p, v, m)
                    else:
                        pred = PW.mask_cut_delta(p, v, m, a)
                    row = {"step": s, "block": b, "head": h, "sources": src_name, "cut": ctype}
                    for part, recv, gate, mp in (("img", slice(T, S), gi, mi), ("txt", others, gt, mt)):
                        if part == "img":
                            meas_in = (got["img_in"] - nat["img_in"]).to(dev)
                            meas_out = (got["img_out"] - nat["img_out"]).to(dev)
                            sham_out = (sham["img_out"] - nat["img_out"]).to(dev)
                            new_in = got["img_in"].to(dev)
                        else:
                            idx = others
                            meas_in = (got["txt_in"] - nat["txt_in"]).to(dev)[idx]
                            meas_out = (got["txt_out"] - nat["txt_out"]).to(dev)[idx]
                            sham_out = (sham["txt_out"] - nat["txt_out"]).to(dev)[idx]
                            new_in = got["txt_in"].to(dev)[idx]
                        dp = pred[h, recv]                                   # [R, D]
                        dm_h = meas_in[:, hs]
                        other_heads = torch.cat([meas_in[:, : h * D], meas_in[:, (h + 1) * D:]], dim=1)
                        hu = half_ulp_bf16(torch, new_in[:, hs])
                        if ctype == "zero":
                            bound = hu
                        else:
                            nat_in = (nat["img_in"] if part == "img" else nat["txt_in"][others.cpu()]).to(dev)[:, hs]
                            bound = (a[h, recv] - nat_in).abs() + hu
                        err = (dm_h - dp).abs()
                        wp = (dp @ mp[h]) * gate
                        wm_proj = (dm_h @ mp[h]) * gate
                        wm = meas_out * gate
                        ws = sham_out * gate
                        nwp = float(torch.linalg.vector_norm(wp))
                        row[part] = {
                            "pred_head_norm": float(torch.linalg.vector_norm(dp)),
                            "meas_head_norm": float(torch.linalg.vector_norm(dm_h)),
                            "head_rel_err": float(torch.linalg.vector_norm(dm_h - dp) /
                                                  torch.linalg.vector_norm(dp).clamp_min(1e-30)),
                            "head_within_bound_frac": float((err <= bound * (1 + 1e-6)).float().mean()),
                            "head_max_err_over_bound": float((err / bound.clamp_min(1e-30)).max()),
                            "other_heads_max_abs_change": float(other_heads.abs().max()) if other_heads.numel() else 0.0,
                            "pred_write_norm": nwp,
                            "write_rel_err_model_bf16": float(torch.linalg.vector_norm(wm - wp)) / max(nwp, 1e-30),
                            "write_rel_err_fp32_projection_of_measured": float(torch.linalg.vector_norm(wm_proj - wp)) / max(nwp, 1e-30),
                            "sham_write_norm_over_pred": float(torch.linalg.vector_norm(ws)) / max(nwp, 1e-30)}
                    cells_out.append(row)
                    event("closure", word=w, step=s, block=b, head=h, sources=src_name, cut=ctype,
                          head_rel=round(row["img"]["head_rel_err"], 6), write_rel=round(row["img"]["write_rel_err_model_bf16"], 5),
                          within=row["img"]["head_within_bound_frac"])
            save(out / f"closure.{w}.json", cells_out)
        saved_cells.clear()

        # ------------------------------------------------------------------- routing / payload split
        do_split = w in config.get("split", [])
        do_swap = w in config.get("vswap", {}).get("words", [])
        if do_split or do_swap:
            store = {}

            def store_reduce(read):
                q = read.q[0][:, T:].detach().to("cpu", copy=True)
                return {"q": q, "k": read.k[0].detach().to("cpu", copy=True),
                        "v": read.v[0].detach().to("cpu", copy=True),
                        "out": read.out[0][:, T:].detach().to("cpu", copy=True), "scale": read.scale}

            obs = ap.observe(store_reduce, name="store")
            port = ap.AttentionPort(backend, rows=table)
            image, lat = run(conds["native"][0], port, observers=[obs])
            store = obs.records
            perm = torch.tensor(al["pi"] + list(range(T, S)), device=dev)
            perm_text = torch.tensor(al["pi"], device=dev)
            groups = {}
            for gname, idx in (("word", al["word"]), ("real", al["real"]), ("pad", al["pad"]),
                               ("img", list(range(T, S)))):
                g = torch.zeros(S, device=dev)
                g[idx] = 1.0
                groups[gname] = g
            text_mask = groups["word"] + groups["real"] + groups["pad"]
            vstore = {}
            split_cells = {}

            def split_reduce(read):
                s, b = read.step, read.block
                if do_swap and b >= JOINT:
                    vstore[(s, b)] = read.v[0][:, perm_text].detach().clone()
                if not do_split:
                    return 0
                st = store[(s, b)]
                qn, kn, vn = st["q"].to(dev).float(), st["k"].to(dev).float(), st["v"].to(dev).float()
                pn = torch.softmax(torch.matmul(qn, kn.transpose(-1, -2)) * st["scale"], dim=-1)
                qd = read.q[0][:, T:].float()
                kd, vd = read.k[0][:, perm].float(), read.v[0][:, perm].float()
                pd = torch.softmax(torch.matmul(qd, kd.transpose(-1, -2)) * read.scale, dim=-1)
                parts = PW.split3(pn, vn, pd, vd)
                pay = PW.payload_parts(pn, vd - vn, groups)
                common, specific = PW.common_payload(pn, vd - vn, text_mask)
                native_diff = read.out[0][:, T:].float() - st["out"].to(dev).float()
                gi, _ = gates.now[b]
                mi, _ = maps(b)
                W = {k_: PW.block_write(x, mi, gi) for k_, x in (
                    ("total", parts["total"]), ("routing", parts["routing"]), ("payload", parts["payload"]),
                    ("interaction", parts["interaction"]), ("native_diff", native_diff),
                    *((f"payload_{g}", pay[g]) for g in pay), ("text_common", common), ("text_specific", specific))}
                W["remainder_alg"] = W["total"] - W["routing"] - W["payload"] - W["interaction"]
                W["remainder_native"] = W["native_diff"] - W["routing"] - W["payload"] - W["interaction"]
                W["text_payload"] = W["payload_word"] + W["payload_real"] + W["payload_pad"]
                tot = W["total"]
                rec = {k_: {"nn": PW.inner(x, x), "ip": PW.inner(x, tot)} for k_, x in W.items()}
                for g in ("word", "real", "pad", "img"):
                    rec[f"payload_{g}"]["ip_payload"] = PW.inner(W[f"payload_{g}"], W["payload"])
                for k_ in ("text_common", "text_specific"):
                    rec[k_]["ip_text_payload"] = PW.inner(W[k_], W["text_payload"])
                rec["payload"]["ip_payload"] = rec["payload"]["nn"]
                rec["text_payload"]["ip_text_payload"] = rec["text_payload"]["nn"]
                vbar = vn.mean(1)
                rc = PW.routing_centered(pn, pd, vn, vbar)
                rec["invariant_max_abs_diff"] = float((rc - parts["routing"]).abs().max())
                rec["routing_max_abs"] = float(parts["routing"].abs().max())
                rec["dP_rowsum_max_abs"] = float((pd - pn).sum(-1).abs().max())
                split_cells[f"{s}:{b}"] = rec
                return 0

            obs = ap.observe(split_reduce, name="split")
            port = ap.AttentionPort(backend, rows=table)
            image_d, lat_d = run(conds["deletion"][0], port, observers=[obs])
            same_d = bool(np.array_equal(base.pixels(np, image_d), roots["del"][0]))
            store = None
            if do_split:
                save(out / f"split.{w}.json", {"word": w, "alignment": {k: al[k] for k in ("word", "shift",
                                                                                         "word_partner_rows")},
                                              "groups": {k: int(v.sum()) for k, v in groups.items()},
                                              "deleted_observer_run_equals_root": same_d, "cells": split_cells})
            event("split", word=w, cells=len(split_cells), equals_root=same_d)

            # --------------------------------------------------------------------------- value swap
            if do_swap:
                Swap = make_swap_backend(ap)
                sb = Swap(lane.tr, text_rows=T, identity=lane.adapter.model_identity)
                swap_rows = []
                sel_of = {"all": list(range(T)), "word": al["word"],
                          "nonword": [j for j in range(T) if j not in set(al["word"])]}
                for arm in config["vswap"]["arms"]:
                    cond_kind = {"native": "native", "iso": "hybrid_own"}[arm["cond"]]
                    sb.configure(vstore, torch.tensor(sel_of[arm["rows"]], device=dev), arm["mode"])
                    port = ap.AttentionPort(sb)
                    t0 = time.monotonic()
                    image, lat = run(conds[cond_kind][0], port)
                    secs = time.monotonic() - t0
                    name = arm["name"]
                    rgb, lat, hashes = keep_image(w, name, image, lat)
                    row = {"arm": name, **arm, "touched_calls": sb.touched, "seconds": secs, **hashes}
                    for r in ("native", "del", "iso"):
                        row["vs_" + r] = base.compare(np, rgb, lat, *roots[r][:2])
                    row["d"] = frp.d_score(row["vs_del"]["px_mad"], rows["mad_native_del"])
                    swap_rows.append(row)
                    event("vswap", word=w, arm=name, d=round(row["d"], 4), touched=sb.touched)
                save(out / f"vswap.{w}.json", swap_rows)
            vstore.clear()
        torch.cuda.empty_cache()
        rows["seconds"] = time.monotonic() - t_word
        rows["peak_vram_mb"] = lane.job_peak()
        save(out / f"word.{w}.json", rows)
        summary_words[w] = {"seconds": rows["seconds"]}
        event("word_done", word=w, seconds=round(rows["seconds"], 1))

    gates.remove()
    tap.remove()
    summary = dict(schema=SCHEMA + "-summary", job_id=job, request_sha256=request.get("request_sha256"),
                   worker_sha256=sha(payload / WORKER), modules=modules, init_latent_sha256=init_sha,
                   worker_seconds=time.monotonic() - started, words=summary_words,
                   peak_vram_allocated_mb=lane.job_peak(), model_identity=lane.adapter.model_identity)
    save(out / "summary.json", summary)
    save(out / "timeline.json", timeline)
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    save(out / "files.json", files)
    print("PWCAP_SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "modules"}, default=str), flush=True)


if __name__ == "__main__":
    main()
