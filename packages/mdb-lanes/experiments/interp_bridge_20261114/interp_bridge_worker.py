"""Interp-bridge GPU worker: nnsight vs attention_port on FLUX.2 Klein 4B, and first-order attribution.

Modes (``config["mode"]``):

``smoke``  Q1 (a-e) and Q2 (a, b, linearity spot checks) in one resident lease:
    1. lane roots without the port (native cat, del/iso for each word) -> hashes vs the lane's.
    2. attention_port: observe 3 cells (q, k, v, out, FP32 probs) at steps 0 and 2; zero and mask
       cut of one (block, head, source rows) cell at every step; timings and peak VRAM.
    3. Q2: differentiable replay of each word's iso trajectory, backward of the latent outcome
       (two directions) with probes, per-head first-order terms for own-word and other-word
       source rows (bridge.head_terms), single-head port cuts versus their prediction.
    4. nnsight: wrap the transformer, native run without a trace, a no-op trace, the same
       capture and the same two cuts through nnsight (port arithmetic, nnsight access), timings,
       extra access points, an nnsight gradient for one cell versus the probe gradient.
``cuts``   Q2 c: the preregistered head groups (PREREG.md section 4) as port cuts under iso
    conditioning, zero and mask, both words; per branch the latent outcome (both directions),
    pixel distances to the roots, hashes.

Model load, prompt encoding, noise and decode come from the read-panel lane's ``Lane`` class
(``flux_read_panel_worker.py``, shipped in the payload unchanged) so arithmetic matches it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

WORKER = "interp_bridge_worker.py"
SCHEMA = "mdb-interp-bridge-run-v1"


def main():
    payload = Path.cwd()
    sys.path.insert(0, str(payload))
    import flux_read_panel_worker as frw
    import bridge
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("not an interp-bridge run config")
    job = os.environ.get("MRUN_JOB_ID", "local")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    out = Path(config["output_scratch"]) / job / "interp-bridge"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    timeline = []

    def event(kind, **fields):
        row = dict(kind=kind, t=round(time.monotonic() - started, 3), **fields)
        timeline.append(row)
        print("BRIDGE " + json.dumps(row, sort_keys=True, default=str), flush=True)

    modules = frw.module_proof(payload)
    frw.save(out / "modules.json", modules)
    frw.save(out / "run-config.json", config)
    try:
        if config["mode"] == "smoke":
            result = smoke(config, out, payload, event, frw, bridge, modules)
        elif config["mode"] == "cuts":
            result = cuts_mode(config, out, payload, event, frw, bridge, modules)
        else:
            raise ValueError("mode must be smoke or cuts")
    finally:
        frw.save(out / "timeline.json", timeline)
    summary = dict(schema="mdb-interp-bridge-summary-v1", mode=config["mode"], job_id=job,
                   worker_sha256=frw.sha(payload / WORKER), bridge_sha256=frw.sha(payload / "bridge.py"),
                   lane_worker_sha256=frw.sha(payload / "flux_read_panel_worker.py"),
                   worker_seconds=time.monotonic() - started, output_root=str(out), **result)
    frw.save(out / "summary.json", summary)
    files = {p.relative_to(out).as_posix(): frw.sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    frw.save(out / "files.json", files)
    print("BRIDGE_SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in ("software",)},
                                          sort_keys=True, default=str), flush=True)


# ===================================================================================== shared

class Ctx:
    """Loaded lane, conditionings, noise, row tables and run helpers."""

    def __init__(self, config, out, payload, event, frw, bridge, modules, *, words):
        import numpy as np
        import torch
        from mdb_lanes import attention_port as ap
        from mdb_lanes import flux_read_panels as frp
        self.np, self.torch, self.ap, self.frp, self.frw, self.bridge = np, torch, ap, frp, frw, bridge
        self.config, self.out, self.event = config, out, event
        self.prompt = config["prompt"]
        self.words, self.other = list(words), dict(config["other"])
        lane = frw.Lane(config, event)
        texts = {self.prompt} | {frp.delete_word(self.prompt, w) for w in self.words}
        lane.encode_all(sorted(texts))
        frw.save(out / "tokens.json", {k: v[2] for k, v in lane.encodings.items()})
        lane.place()
        if lane.heads != 24:
            raise ValueError("unexpected head count")
        self.lane = lane
        self.software = frw.software_lane(lane, modules, payload)
        self.software["worker_sha256"] = frw.sha(payload / WORKER)
        self.software["bridge_sha256"] = frw.sha(payload / "bridge.py")
        try:
            import nnsight
            self.software["nnsight"] = nnsight.__version__
        except Exception as e:  # recorded; smoke needs it
            self.software["nnsight"] = f"import failed: {type(e).__name__}: {e}"
        frw.save(out / "software.json", self.software)
        self.T, self.S = frw.T, lane.S
        self.joint = frw.JOINT
        self.blocks = frw.BLOCKS
        rec = lane.encodings[self.prompt][2]
        word_rows = {}
        for w in set(self.words) | {self.other[w] for w in self.words}:
            rows = sorted(r for x in rec["words"] if x["word"] == w for r in x["rows"])
            if not rows:
                raise ValueError(f"word {w!r} has no rows")
            word_rows[w] = rows
        self.table = frp.row_table(word_rows, self.T, self.S)
        self.word_rows = word_rows
        self.seed = int(config["seed"])
        self.init = lane.noise(self.seed)
        self.init_sha = lane.init_sha(self.init)
        self.backend = ap.FluxBackend(lane.tr, text_rows=self.T, identity=lane.adapter.model_identity)
        self.conds = {}
        self.conds["native"] = lane.conditioning({"kind": "native", "prompt": self.prompt})
        for w in self.words:
            self.conds[f"del:{w}"] = lane.conditioning({"kind": "deletion", "prompt": self.prompt, "word": w})
            self.conds[f"iso:{w}"] = lane.conditioning({"kind": "hybrid_own", "prompt": self.prompt, "word": w})
        self.peak_job = 0.0

    # ---------------------------------------------------------------------------- runs
    def session(self, cond):
        lane = self.lane
        c, txt_ids = cond[0]
        latent, img_ids, timesteps, sigmas = self.init
        return lane.adapter.session(latent=latent.clone(), conditioning=c.to(lane.dev), img_ids=img_ids.clone(),
                                    txt_ids=txt_ids.to(lane.dev), timesteps=timesteps.clone(),
                                    sigmas=sigmas.clone())

    def loop_fn(self, cond, holder):
        lane = self.lane

        def loop():
            sess = self.session(cond)
            state = sess._state
            while state["step"] < lane.STEPS:
                state = lane.adapter.advance(state)
            sess._state = dict(state)
            holder.append(sess)
            return state["latent"]
        return loop

    def finish(self, sess):
        lat = sess.read("latent").detach().float().cpu().numpy()[0]
        image = self.lane.adapter.decode(sess, height=self.lane.H, width=self.lane.W)
        return image, lat

    def timed(self, fn):
        """Run fn() -> (image, lat, extra); returns dict with seconds, peak, hashes, rgb, lat."""
        torch, np = self.torch, self.np
        torch.cuda.synchronize()
        self.peak_job = max(self.peak_job, torch.cuda.max_memory_allocated() / 2 ** 20)
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 2 ** 20
        t0 = time.monotonic()
        image, lat, extra = fn()
        torch.cuda.synchronize()
        seconds = time.monotonic() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        self.peak_job = max(self.peak_job, peak)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return dict(seconds=seconds, peak_mb=peak, allocated_before_mb=before,
                    latent_sha256=hashlib.sha256(lat.astype(np.float32).tobytes()).hexdigest(),
                    image_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), rgb=rgb, lat=lat, extra=extra,
                    image=image)

    def run_plain(self, cond_name, port=None, cuts=(), observers=()):
        cond = self.conds[cond_name]

        def fn():
            holder = []
            loop = self.loop_fn(cond, holder)
            if port is None:
                loop()
            else:
                with port.apply(*cuts, observers=observers, forwards=self.lane.STEPS):
                    loop()
                if port.calls != self.lane.STEPS * self.blocks:
                    raise ValueError("port call count differs")
            image, lat = self.finish(holder[0])
            return image, lat, None
        return self.timed(fn)

    def port(self):
        return self.ap.AttentionPort(self.backend, rows=self.table, zero_method="delta")

    def save_png(self, name, image):
        path = self.out / "images" / f"{self.frp.safe_name(name)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return self.frw.sha(path)

    def outcome_np(self, lat, base, direction):
        np = self.np
        d = direction.astype(np.float64)
        return float(((lat.astype(np.float64) - base.astype(np.float64)) * d).sum() / (d * d).sum())

    def row(self, name, r, refs=None):
        """JSON row without arrays; refs = {name: (rgb, lat)} distances."""
        out = {k: v for k, v in r.items() if k not in ("rgb", "lat", "extra", "image")}
        out["arm"] = name
        out["png_sha256"] = self.save_png(name, r["image"])
        for ref, (rgb, lat) in (refs or {}).items():
            out[f"vs_{ref}"] = self.frw.compare(self.np, r["rgb"], r["lat"], rgb, lat)
        return out


def references(config):
    return config.get("reference_hashes", {})


# ====================================================================================== smoke

def smoke(config, out, payload, event, frw, bridge, modules):
    ctx = Ctx(config, out, payload, event, frw, bridge, modules, words=config["words"])
    np, torch, ap = ctx.np, ctx.torch, ctx.ap
    STEPS, reps = ctx.lane.STEPS, int(config.get("timing_reps", 3))
    refs = references(config)
    res = {"software": ctx.software, "init_latent_sha256": ctx.init_sha,
           "rows": {k: ctx.frp.runs(v) for k, v in ctx.table.items()}}
    rows = {}

    # ---------------------------------------------------------------- 1. roots (no port)
    roots = {}
    for name in ["native"] + [f"{k}:{w}" for w in ctx.words for k in ("del", "iso")]:
        r = ctx.run_plain(name)
        roots[name] = r
        key = "native:cat" if name == "native" else name
        rows[f"root:{name}"] = ctx.row(f"root:{name}", r)
        rows[f"root:{name}"]["reference_latent_sha256"] = refs.get(key)
        rows[f"root:{name}"]["equals_lane_reference"] = refs.get(key) == r["latent_sha256"]
        event("root", arm=name, seconds=round(r["seconds"], 3), equal=rows[f"root:{name}"]["equals_lane_reference"])
    native = roots["native"]

    # ---------------------------------------------------------------- 2. port: capture + cuts
    cells = [tuple(c) for c in config["cells"]]
    cap_steps = list(config["capture_steps"])
    timing = {"native_no_port": [], "port_capture": [], "port_zero": [], "port_mask": [],
              "nnsight_wrapped_no_trace": [], "nnsight_noop_trace": [], "nnsight_capture": [],
              "nnsight_zero": [], "nnsight_mask": []}
    for _ in range(reps):
        r = ctx.run_plain("native")
        timing["native_no_port"].append(dict(seconds=r["seconds"], peak_mb=r["peak_mb"],
                                             equal=r["latent_sha256"] == native["latent_sha256"]))
    port = ctx.port()
    port_records = None
    for i in range(reps):
        obs = [ap.observe(bridge.port_observer_reduce(h), steps=cap_steps, blocks=[b], name=f"b{b}h{h}")
               for b, h in cells]
        r = ctx.run_plain("native", port=port, observers=obs)
        timing["port_capture"].append(dict(seconds=r["seconds"], peak_mb=r["peak_mb"],
                                           equal=r["latent_sha256"] == native["latent_sha256"]))
        if i == 0:
            port_records = {(s, b): o.records[(s, b)] for (b, h), o in zip(cells, obs) for s in cap_steps}
    cb, ch = config["cut_cell"]
    src_word = config["cut_cell_word"]
    q1_cuts = {"zero": ap.cut(["image"], [f"own:{src_word}"], heads=[(cb, ch)], renormalize=False),
               "mask": ap.cut(["image"], [f"own:{src_word}"], heads=[(cb, ch)], renormalize=True)}
    port_cut = {}
    for kind, c in q1_cuts.items():
        for i in range(reps):
            r = ctx.run_plain("native", port=port, cuts=[c])
            timing[f"port_{kind}"].append(dict(seconds=r["seconds"], peak_mb=r["peak_mb"]))
            if i == 0:
                port_cut[kind] = r
                rows[f"port:{kind}"] = ctx.row(f"port:{kind}", r, {"native": (native["rgb"], native["lat"])})
            elif r["latent_sha256"] != port_cut[kind]["latent_sha256"]:
                rows[f"port:{kind}"]["repeat_differs"] = True
    event("port_done")

    # ---------------------------------------------------------------- 3. Q2 attribution
    q2 = q2_attribution(ctx, config, roots, event)
    res["q2"] = q2

    # ---------------------------------------------------------------- 4. nnsight
    res["q1"] = q1_nnsight(ctx, config, native, port_records, port_cut, q1_cuts, timing, cells, cap_steps, q2, rows,
                           event)
    res["timing"] = timing
    frw.save(out / "rows.json", rows)
    res["peak_vram_allocated_mb"] = max(ctx.peak_job, torch.cuda.max_memory_allocated() / 2 ** 20)
    return res


def q2_attribution(ctx, config, roots, event):
    """Per-head first-order terms for each word under its iso trajectory; single-head checks."""
    np, torch, ap, bridge, frw = ctx.np, ctx.torch, ctx.ap, ctx.bridge, ctx.frw
    lane, STEPS = ctx.lane, ctx.lane.STEPS
    out_dir = ctx.out / "attribution"
    out_dir.mkdir()
    report = {}
    probe = bridge.Probe()
    H = lane.heads
    for w in ctx.words:
        cond_name = f"iso:{w}"
        sess = ctx.session(ctx.conds[cond_name])
        st = sess._state
        inputs = dict(conditioning=st["conditioning"], img_ids=st["img_ids"], txt_ids=st["txt_ids"],
                      timesteps=st["timesteps"], sigmas=st["sigmas"], shape=(1, ctx.S, H, lane.tr.config.attention_head_dim))
        xs = [st["latent"]]
        t0 = time.monotonic()
        with torch.no_grad():
            for s in range(STEPS):
                xs.append(bridge.step_forward(lane.adapter, xs[-1], inputs["conditioning"], inputs["img_ids"],
                                              inputs["txt_ids"], inputs["timesteps"], inputs["sigmas"], s))
        replay_sha = bridge.latent_sha(xs[-1])
        rep = dict(replay_latent_sha256=replay_sha, iso_root_sha256=roots[cond_name]["latent_sha256"],
                   replay_equals_adapter=replay_sha == roots[cond_name]["latent_sha256"],
                   replay_forward_seconds=time.monotonic() - t0)
        dev = lane.dev
        L = {k: torch.from_numpy(roots[k]["lat"]).to(dev) for k in ("native", f"del:{w}", cond_name)}
        directions = {"nd": (L[f"del:{w}"], L["native"] - L[f"del:{w}"]),
                      "id": (L[f"del:{w}"], L[cond_name] - L[f"del:{w}"])}
        sources = {"own": w, "other": ctx.other[w]}
        arrays = {}
        for dname, (base, direction) in directions.items():
            terms = {}

            def on_step(s, pr):
                for b in range(ctx.blocks):
                    q, k, v, o, scale = pr.seen[b]
                    g = pr.zeros[b].grad
                    w_img, w_txt = bridge.out_weights(lane.tr, b)
                    for sname, sw in sources.items():
                        recv = torch.tensor(ctx.table["image"] + ctx.table[f"others:{sw}"], device=dev)
                        src = torch.tensor(ctx.table[f"own:{sw}"], device=dev)
                        terms[(sname, s, b)] = bridge.head_terms(q, k, v, o, g, recv, src,
                                                                 scale or q.shape[-1] ** -0.5, w_img, w_txt, ctx.T)
                    if dname == "nd" and s == STEPS - 1 and b == int(config["grad_check_block"]):
                        report.setdefault("probe_grad", {})[w] = g.detach().cpu().clone()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            brep = bridge.backward_trajectory(lane.adapter, xs, inputs, base, direction, probe,
                                              checkpoint=bool(config.get("checkpoint", True)), on_step=on_step)
            torch.cuda.synchronize()
            brep["seconds"] = time.monotonic() - t0
            brep["peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            ctx.peak_job = max(ctx.peak_job, brep["peak_mb"])
            brep["y_root"] = {k: ctx.outcome_np(roots[k]["lat"], base.cpu().numpy(), direction.cpu().numpy())
                              for k in ("native", f"del:{w}", cond_name)}
            rep[f"backward_{dname}"] = brep
            keys = sorted(next(iter(terms.values())).keys())
            for sname in sources:
                for key in keys:
                    arr = np.zeros((STEPS, ctx.blocks, H), dtype=np.float64)
                    for s in range(STEPS):
                        for b in range(ctx.blocks):
                            arr[s, b] = terms[(sname, s, b)][key].double().numpy()
                    if key.startswith("att_") or dname == "nd":
                        arrays[f"{dname}.{sname}.{key}"] = arr
            event("backward", word=w, direction=dname, seconds=round(brep["seconds"], 2), peak_mb=round(brep["peak_mb"]))
        np.savez(out_dir / f"terms.{w}.npz", **arrays)
        rep["terms_sha256"] = frw.sha(out_dir / f"terms.{w}.npz")
        rep["source_rows"] = {k: ctx.table[f"own:{v}"] for k, v in sources.items()}
        # single-head linearity checks under iso conditioning (port cuts, every step)
        removal = -(arrays["nd.own.att_zero_img"] + arrays["nd.own.att_zero_txt"]).sum(0).reshape(-1)
        mass = (arrays["nd.own.mass_img"] + arrays["nd.own.mass_txt"]).sum(0).reshape(-1)
        order = np.argsort(-removal, kind="stable")
        picks = {"top_removal": int(order[0]), "median_removal": int(order[len(order) // 2]),
                 "top_mass": int(np.argsort(-mass, kind="stable")[0])}
        y_iso = {d: rep[f"backward_{d}"]["y_root"][cond_name] for d in directions}
        port = ctx.port()
        checks = []
        for label, flat in picks.items():
            b, h = divmod(flat, H)
            for kind, renorm in (("zero", False), ("mask", True)):
                c = ap.cut(["image", f"others:{w}"], [f"own:{w}"], heads=[(b, h)], renormalize=renorm)
                r = ctx.run_plain(cond_name, port=port, cuts=[c])
                row = dict(label=label, block=b, head=h, kind=kind, seconds=r["seconds"])
                for d, (base, direction) in directions.items():
                    y = ctx.outcome_np(r["lat"], base.cpu().numpy(), direction.cpu().numpy())
                    pred = float((arrays[f"{d}.own.att_{kind}_img"] + arrays[f"{d}.own.att_{kind}_txt"])[:, b, h].sum())
                    row[d] = dict(measured_dy=y - y_iso[d], predicted_dy=pred)
                checks.append(row)
        rep["linearity"] = checks
        report[w] = rep
        event("q2_word_done", word=w)
    grads = report.pop("probe_grad", {})
    torch.save(grads, out_dir / "probe_grad_check.pt")
    return report


def q1_nnsight(ctx, config, native, port_records, port_cut, q1_cuts, timing, cells, cap_steps, q2, rows, event):
    np, torch, bridge, frw = ctx.np, ctx.torch, ctx.bridge, ctx.frw
    lane, STEPS, reps = ctx.lane, ctx.lane.STEPS, int(config.get("timing_reps", 3))
    rep = {}
    try:
        from nnsight import NNsight
    except Exception as e:
        rep["import_error"] = f"{type(e).__name__}: {e}"
        return rep
    t0 = time.monotonic()
    nn_tr = NNsight(lane.tr)
    rep["wrap_seconds"] = time.monotonic() - t0
    cond = ctx.conds["native"]

    def traced(body):
        """Run a trace body(run) and return the timed record; errors are recorded, not raised."""
        def fn():
            holder = []
            body(ctx.loop_fn(cond, holder))
            image, lat = ctx.finish(holder[0])
            return image, lat, None
        return ctx.timed(fn)

    def attempt(name, f):
        try:
            return f()
        except Exception as e:
            rep.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-3000:]
            event("nnsight_error", step=name, error=f"{type(e).__name__}: {e}")
            return None

    # (a) wrapped, no trace
    for i in range(reps):
        r = ctx.run_plain("native")
        timing["nnsight_wrapped_no_trace"].append(dict(seconds=r["seconds"], peak_mb=r["peak_mb"],
                                                       equal=r["latent_sha256"] == native["latent_sha256"]))
    # no-op trace
    for i in range(reps):
        r = attempt("noop", lambda: traced(lambda run: bridge.nnsight_noop(nn_tr, run)))
        if r is None:
            break
        timing["nnsight_noop_trace"].append(dict(seconds=r["seconds"], peak_mb=r["peak_mb"],
                                                 equal=r["latent_sha256"] == native["latent_sha256"]))
    # (b) capture
    store = {}
    for i in range(reps):
        holder = {}

        def body(run):
            holder["store"], _ = bridge.nnsight_capture(nn_tr, run, cells, cap_steps, joint=ctx.joint)
        r = attempt("capture", lambda: traced(body))
        if r is None:
            break
        timing["nnsight_capture"].append(dict(seconds=r["seconds"], peak_mb=r["peak_mb"],
                                              equal=r["latent_sha256"] == native["latent_sha256"]))
        if i == 0:
            store = holder["store"]
            rows["nnsight:capture"] = ctx.row("nnsight:capture", r, {"native": (native["rgb"], native["lat"])})
    cmp = {}
    for (b, h) in cells:
        for s in cap_steps:
            pr, nr = port_records.get((s, b)), store.get((s, b))
            if pr is None or nr is None:
                cmp[f"s{s}.b{b}.h{h}"] = "missing"
                continue
            p_nn = bridge.probs_of(nr["q"].to(lane.dev), nr["k"].to(lane.dev), nr["scale"]).cpu()
            row = {f: bridge.compare_tensors(pr[f], nr[f]) for f in ("q", "k", "v", "out")}
            row["p"] = bridge.compare_tensors(pr["p"], p_nn)
            # the native kernel's output against the FP32 recompute p @ v (kernel rounding, not a path difference)
            row["kernel_vs_fp32_pv"] = bridge.compare_tensors(nr["out"], (p_nn @ nr["v"].float()))
            row["scale"] = [pr["scale"], nr["scale"]]
            row["dispatch_kwargs"] = nr["kwargs"]
            row["dispatch_positional_args"] = nr["n_args"]
            cmp[f"s{s}.b{b}.h{h}"] = row
    rep["capture_compare"] = cmp
    if store:
        torch.save({f"s{s}.b{b}": {k: v for k, v in store[(s, b)].items() if k in ("q", "k", "v", "out")}
                    for (s, b) in store}, ctx.out / "nnsight_capture.pt")
        torch.save({f"s{s}.b{b}": {k: v for k, v in port_records[(s, b)].items() if k in ("q", "k", "v", "out", "p")}
                    for (s, b) in port_records}, ctx.out / "port_capture.pt")
    event("nnsight_capture_done", ok=bool(store))
    # (c) cuts
    cb = config["cut_cell"][0]
    for kind, c in q1_cuts.items():
        for i in range(reps):
            r = attempt(f"cut_{kind}", lambda: traced(lambda run: bridge.nnsight_cut(
                nn_tr, run, [c], ctx.table, range(STEPS), [cb], joint=ctx.joint, zero_method="delta")))
            if r is None:
                break
            timing[f"nnsight_{kind}"].append(dict(seconds=r["seconds"], peak_mb=r["peak_mb"]))
            if i == 0:
                pc = port_cut[kind]
                rows[f"nnsight:{kind}"] = ctx.row(f"nnsight:{kind}", r, {"port": (pc["rgb"], pc["lat"]),
                                                                       "native": (native["rgb"], native["lat"])})
                rep[f"cut_{kind}"] = dict(latent_equal_port=r["latent_sha256"] == pc["latent_sha256"],
                                          rgb_equal_port=r["image_rgb_sha256"] == pc["image_rgb_sha256"],
                                          vs_port=rows[f"nnsight:{kind}"]["vs_port"],
                                          port_vs_native=rows[f"port:{kind}"]["vs_native"])
    event("nnsight_cuts_done")
    # port still exact after nnsight wrapping and tracing
    r = ctx.run_plain("native", port=ctx.port(), cuts=[q1_cuts["zero"]])
    rep["port_zero_after_wrap_equal"] = r["latent_sha256"] == port_cut["zero"]["latent_sha256"]
    r = ctx.run_plain("native")
    rep["native_after_traces_equal"] = r["latent_sha256"] == native["latent_sha256"]
    # (e) extra access points (one generation each; shapes and identities)
    rep["extras"] = attempt("extras", lambda: extras(ctx, nn_tr, cond, config))
    rep["grad_check"] = attempt("grad", lambda: grad_check(ctx, nn_tr, config))
    return rep


def extras(ctx, nn_tr, cond, config):
    """Read access points the port does not expose, step 0, one joint and one single block."""
    torch, bridge = ctx.torch, ctx.bridge
    jb, sb = int(config["extras_joint_block"]), int(config["extras_single_block"])
    jenv = nn_tr.transformer_blocks[jb]
    senv = nn_tr.single_transformer_blocks[sb - ctx.joint]
    names_j = [op.name for op in jenv.source]
    names_s = [op.name for op in senv.source]
    split_j = "Flux2Modulation_split_0" if "Flux2Modulation_split_0" in names_j else next(
        n for n in names_j if n.startswith("Flux2Modulation_split"))
    gate_j = [n for n in names_j if n.startswith("attn_output_")]
    found = {}
    holder = []
    run = ctx.loop_fn(cond, holder)
    with nn_tr.trace(fn=run) as tracer:
        with tracer.invoke():
            mod = getattr(jenv.source, split_j).output
            found["joint_gate_msa"] = mod[0][2].detach().cpu().clone()
            op = bridge.dispatch_op(nn_tr, jb, ctx.joint)
            d_out = op.output
            found["joint_dispatch_out"] = d_out.detach().cpu().clone()
            found["joint_to_add_out_in"] = jenv.attn.to_add_out.input.detach().cpu().clone()
            found["joint_to_out_in"] = jenv.attn.to_out[0].input.detach().cpu().clone()
            found["joint_ff_linear_in_out"] = jenv.ff.linear_in.output.detach().cpu().clone()
            sop = bridge.dispatch_op(nn_tr, sb, ctx.joint)
            found["single_dispatch_out"] = sop.output.detach().cpu().clone()
            found["single_mlp_act_in"] = senv.attn.mlp_act_fn.input.detach().cpu().clone()
            found["single_to_out_in"] = senv.attn.to_out.input.detach().cpu().clone()
            done = tracer.result.save()
    T = ctx.T
    inner = found["single_dispatch_out"].shape[2] * found["single_dispatch_out"].shape[3]
    checks = {
        "joint_to_out_in == dispatch_out[image rows].flatten": bool(torch.equal(
            found["joint_to_out_in"], found["joint_dispatch_out"][:, T:].flatten(2, 3))),
        "joint_to_add_out_in == dispatch_out[text rows].flatten": bool(torch.equal(
            found["joint_to_add_out_in"], found["joint_dispatch_out"][:, :T].flatten(2, 3))),
        "single_to_out_in[:, :, :inner] == dispatch_out.flatten": bool(torch.equal(
            found["single_to_out_in"][..., :inner], found["single_dispatch_out"].flatten(2, 3))),
    }
    return dict(joint_block=jb, single_block=sb, joint_source_ops=names_j, single_source_ops=names_s,
                split_op=split_j, assignment_ops_attn_output=gate_j,
                shapes={k: list(v.shape) for k, v in found.items()}, identities=checks,
                latent_sha256=bridge.latent_sha(done))


def grad_check(ctx, nn_tr, config):
    """nnsight gradient of the latent outcome w.r.t. one block's attention output at the last step.

    Runs bridge.step_forward (graph-enabled replay) as the trace fn from the iso trajectory's last
    step input, outcome direction "nd", and compares with the probe gradient saved by q2.
    """
    torch, bridge, lane = ctx.torch, ctx.bridge, ctx.lane
    w = ctx.words[0]
    b = int(config["grad_check_block"])
    sess = ctx.session(ctx.conds[f"iso:{w}"])
    st = sess._state
    x = st["latent"]
    with torch.no_grad():
        for s in range(lane.STEPS - 1):
            x = bridge.step_forward(lane.adapter, x, st["conditioning"], st["img_ids"], st["txt_ids"],
                                    st["timesteps"], st["sigmas"], s)
    roots = {}
    for k in ("native", f"del:{w}"):
        holder = []
        ctx.loop_fn(ctx.conds[k], holder)()
        roots[k] = holder[0].read("latent").detach().float()[0]
    base, direction = roots[f"del:{w}"], roots["native"] - roots[f"del:{w}"]
    x3 = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)

    def one_step():
        return bridge.step_forward(lane.adapter, x3, st["conditioning"], st["img_ids"], st["txt_ids"],
                                   st["timesteps"], st["sigmas"], lane.STEPS - 1)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    with torch.enable_grad():
        with nn_tr.trace(fn=one_step) as tracer:
            with tracer.invoke():
                out = bridge.dispatch_op(nn_tr, b, ctx.joint).output
                y = bridge.outcome(tracer.result, base, direction)
                with y.backward():
                    g = out.grad.save()
    seconds = time.monotonic() - t0
    probe = torch.load(ctx.out / "attribution" / "probe_grad_check.pt")[w]
    return dict(word=w, block=b, step=lane.STEPS - 1, seconds=seconds,
                peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                vs_probe=bridge.compare_tensors(g.detach().cpu(), probe),
                note="probe gradient came from the checkpointed step-level backward (bridge.backward_trajectory)")


# ======================================================================================= cuts

def cuts_mode(config, out, payload, event, frw, bridge, modules):
    ctx = Ctx(config, out, payload, event, frw, bridge, modules, words=config["words"])
    np, torch, ap = ctx.np, ctx.torch, ctx.ap
    refs = references(config)
    roots = {}
    rows = []
    for name in ["native"] + [f"{k}:{w}" for w in ctx.words for k in ("del", "iso")]:
        r = ctx.run_plain(name)
        roots[name] = r
        key = "native:cat" if name == "native" else name
        row = ctx.row(f"root:{name}", r)
        row.update(kind="root", equals_lane_reference=refs.get(key) == r["latent_sha256"])
        rows.append(row)
    dirs = {w: {"nd": (roots[f"del:{w}"]["lat"], roots["native"]["lat"] - roots[f"del:{w}"]["lat"]),
                "id": (roots[f"del:{w}"]["lat"], roots[f"iso:{w}"]["lat"] - roots[f"del:{w}"]["lat"])}
            for w in ctx.words}
    port = ctx.port()
    budget = float(config["max_wall_seconds"])
    t_start = time.monotonic()
    stopped = False
    for br in config["branches"]:
        if time.monotonic() - t_start > budget:
            stopped = True
            event("budget_stop", branch=br["id"])
            break
        w = br["word"]
        heads = None if br["units"] is None else [tuple(u) for u in br["units"]]
        c = ap.cut(["image", f"others:{w}"], [f"own:{w}"], heads=heads, renormalize=br["cut"] == "mask")
        r = ctx.run_plain(f"iso:{w}", port=port, cuts=[c])
        row = ctx.row(br["id"], r, {"del": (roots[f"del:{w}"]["rgb"], roots[f"del:{w}"]["lat"]),
                                    "iso": (roots[f"iso:{w}"]["rgb"], roots[f"iso:{w}"]["lat"]),
                                    "native": (roots["native"]["rgb"], roots["native"]["lat"])})
        row.update(kind="branch", word=w, cut=br["cut"], partition=br.get("partition"), group=br.get("group"),
                   n_units=None if heads is None else len(heads))
        for d, (base, direction) in dirs[w].items():
            row[f"y_{d}"] = ctx.outcome_np(r["lat"], base, direction)
            row[f"y_{d}_iso"] = ctx.outcome_np(roots[f"iso:{w}"]["lat"], base, direction)
        rows.append(row)
        event("branch", id=br["id"], seconds=round(r["seconds"], 3), dy=round(row["y_nd"] - row["y_nd_iso"], 5))
    frw.save(out / "rows.json", rows)
    return dict(software=ctx.software, init_latent_sha256=ctx.init_sha, branches=len(rows) - len(roots),
                planned=len(config["branches"]), budget_stopped=stopped,
                peak_vram_allocated_mb=max(ctx.peak_job, torch.cuda.max_memory_allocated() / 2 ** 20))


if __name__ == "__main__":
    main()
