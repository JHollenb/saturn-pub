"""BFL demo-control worker (FLUX.2 Klein 4B) for two experiments, run inside one mrun lease.

Experiment 1 (``exp1``): is a route edit more than a prompt swap? Reproduces the
counterfactual-diffusion-futures route arm (text-state substitution at joint.2/3/4 and the
text rows of single.0) and compares it to a mid-trajectory conditioning (prompt) swap.

Experiment 2 (``exp2``): text-only camera-contact replication. Captures native joint.3 text
states from matched left/right prompts, builds M/D and the predicate/complement split, and
writes M and M±D_rest / M±D_pred arms into a neutral-conditioned trajectory. Images are
written under random ids with a separate key for blinded judging; this worker never judges.

Both modes run a frozen mechanics smoke gate first (abort => non-zero exit => failed job),
then the full panel in the same process. ``--tiny`` runs the identical control flow on the
adapter's tiny CPU fixtures (no weights, no GPU) as a local unit test.

Driving uses the authoritative ``Flux2KleinAdapter.advance`` block transition directly for
speed; StateCut capture/restore is exercised in the smoke gate for the resume-parity check.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from saturn_pub import Session
from saturn_pub.values import clone

STEP_DTYPE = torch.bfloat16
EPS = 1e-6

EXP1_SITES = ("after:joint.2", "after:joint.3", "after:joint.4", "after:single.0")
EXP2_SITE = "after:joint.3"

EXP1_PROMPTS = {
    "source": "a photorealistic red fox sitting in fresh snow at dawn, soft red light",
    "scene": "a photorealistic red fox standing in a sunlit desert at noon, warm red light",
    "subject": "a photorealistic red cat sitting in fresh snow at dawn, soft red light",
    "hostile": "a photorealistic blue fox sitting in fresh snow at dawn, soft blue light",
}
EXP1_SPECIMENS = (
    {"id": "scene-seed9001", "axis": "scene", "donor": "scene", "seed": 9001},
    {"id": "scene-seed1337", "axis": "scene", "donor": "scene", "seed": 1337},
    {"id": "subject-seed4242", "axis": "subject", "donor": "subject", "seed": 4242},
    {"id": "subject-seed9001", "axis": "subject", "donor": "subject", "seed": 9001},
)
# Demo's originally reported progress, for the reproduction check.
EXP1_ORIGINAL = {
    "scene-seed9001": {"cut0": 0.9164435267365753, "cut2": 0.1651337439486651},
    "scene-seed1337": {"cut0": 0.907, "cut2": 0.095},
    "subject-seed4242": {"cut0": 0.971, "cut2": 0.099},
    "subject-seed9001": {"cut0": 0.904, "cut2": 0.358},
}

EXP2_PROMPTS = {
    "neutral": "a person sits at a wooden table between two cameras on tripods",
    "left": "a person sits at a wooden table between two cameras on tripods "
    "and adjusts the focus of the left camera",
    "right": "a person sits at a wooden table between two cameras on tripods "
    "and adjusts the focus of the right camera",
}
EXP2_PREDICATE = "adjusts the focus"
EXP2_SEEDS = (
    7001, 7013, 7027, 7039, 7043, 7057, 7069, 7079,
    7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177,
)

# --- Exp2 viability retry (gate-first): stronger, side-explicit, tripod-contact prompts ---
# Predicate span = the action phrase; left/right differ only at the side token.
EXP2_RETRY_VARIANTS = {
    "A": {
        "predicate": "adjusting its focus ring",
        "left": "a photo of a person seated at a wooden table between two large cameras on "
        "tripods, reaching out with one hand and touching the camera on the left side of the "
        "image, adjusting its focus ring",
        "right": "a photo of a person seated at a wooden table between two large cameras on "
        "tripods, reaching out with one hand and touching the camera on the right side of the "
        "image, adjusting its focus ring",
    },
    "B": {
        "predicate": "firmly gripping the lens",
        "left": "a photo of a person seated at a wooden table between two large cameras on "
        "separate tripods, reaching out with one hand and firmly gripping the lens of the "
        "camera on the left side of the image",
        "right": "a photo of a person seated at a wooden table between two large cameras on "
        "separate tripods, reaching out with one hand and firmly gripping the lens of the "
        "camera on the right side of the image",
    },
}
# 16 fresh seeds (disjoint from EXP2_SEEDS and the historical 26091741 set).
EXP2_VIAB_SEEDS = (
    8101, 8111, 8117, 8123, 8147, 8161, 8167, 8171,
    8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237,
)
# Variant B passed the coordinator's viability gate (left ~14/16, right ~15/16); A not used.
# Neutral = the Variant B sentence without the action/side clause (same structure).
EXP2B_NEUTRAL = (
    "a photo of a person seated at a wooden table between two large cameras on separate tripods"
)


# --------------------------------------------------------------------------------------
# memory / digest helpers
# --------------------------------------------------------------------------------------
def _cuda() -> bool:
    return torch.cuda.is_available()


def _reset_peak() -> None:
    if _cuda():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_vram_mb():
    return None if not _cuda() else torch.cuda.max_memory_allocated() / 2**20


def _rss_mb():
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    except Exception:
        return None


def _free() -> None:
    gc.collect()
    if _cuda():
        torch.cuda.empty_cache()


def _digest(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def progress(image: np.ndarray, src: np.ndarray, donor: np.ndarray) -> float:
    return 1.0 - mad(image, donor) / (mad(src, donor) + EPS)


# --------------------------------------------------------------------------------------
# pipeline + sampling inputs (mirrors experiments/diffusion_family_validation/run.py)
# --------------------------------------------------------------------------------------
def load_encoder_pipeline(model_path, device):
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path, transformer=None, torch_dtype=STEP_DTYPE, local_files_only=True
    )
    pipe.text_encoder.to(device)
    return pipe


def encode_prompt(pipe, prompt, device):
    embeds = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)[0]
    txt_ids = pipe._prepare_text_ids(embeds).to(device)
    return embeds.detach().cpu(), txt_ids.detach().cpu()


def resolve_predicate_rows(tokenizer, prompt, predicate, max_len=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    start = text.find(predicate)
    if start < 0:
        raise ValueError(f"predicate {predicate!r} not found in templated prompt")
    end = start + len(predicate)
    rows = [i for i, (a, b) in enumerate(offsets) if b > a and not (b <= start or a >= end)]
    decoded = [tokenizer.decode(enc["input_ids"][0][i]) for i in rows]
    if not rows:
        raise ValueError("predicate resolved to zero rows")
    return rows, decoded


def load_denoise_pipeline(model_path, device):
    from diffusers import Flux2KleinPipeline

    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path, text_encoder=None, tokenizer=None, torch_dtype=STEP_DTYPE, local_files_only=True
    )
    if type(pipe).__name__ == "Flux2KleinKVPipeline":
        raise SystemExit("refused: Flux2KleinKVPipeline uses a distinct reference-KV ABI")
    pipe.vae.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, residency="resident", device=device)
    return pipe, adapter


def prepare_sampling(pipe, seed, size, steps, device, in_channels):
    import numpy as _np
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )

    generator = torch.Generator("cpu").manual_seed(seed)
    latent, img_ids = pipe.prepare_latents(
        1, in_channels // 4, size, size, STEP_DTYPE, device, generator
    )
    sigmas = (
        None
        if pipe.scheduler.config.get("use_flow_sigmas", False)
        else _np.linspace(1.0, 0.25, steps)
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        steps,
        device,
        sigmas=sigmas,
        mu=compute_empirical_mu(image_seq_len=latent.shape[1], num_steps=steps),
    )
    return (
        latent.detach().cpu(),
        img_ids.detach().cpu(),
        timesteps.detach().cpu(),
        pipe.scheduler.sigmas.detach().cpu(),
    )


# --------------------------------------------------------------------------------------
# driving the adapter directly (fast); Session only for the StateCut resume check + decode
# --------------------------------------------------------------------------------------
def init_state(adapter, *, latent, conditioning, img_ids, txt_ids, timesteps, sigmas):
    session = adapter.session(
        latent=latent,
        conditioning=conditioning,
        img_ids=img_ids,
        txt_ids=txt_ids,
        timesteps=timesteps,
        sigmas=sigmas,
    )
    return clone(session._state)


def _site_of(adapter, state):
    bd = adapter.boundary(state)
    return bd.split("/", 1)[1] if "/" in bd else bd


def drive(adapter, state, *, patch=None, capture_sites=(), swap=None):
    """Advance a copy of ``state`` to completion.

    patch(step, site, text)->tensor|None : replacement for the text slot at a site.
    capture_sites : site strings whose text is captured as {(step, site): cpu tensor}.
    swap(step, site, state)->tensor|None : replacement for the conditioning slot (applied at
        the ``project`` boundary of a step, before the project transition consumes it).
    Returns (final_state, captured, per_step_latent_digests).
    """
    state = clone(state)
    total = len(state["timesteps"])
    captured, per_step = {}, [None] * total
    # cut-0 swaps happen before the first project transition (step 0 has no prior commit).
    if swap is not None:
        newc = swap(0, "project", state)
        if newc is not None:
            state["conditioning"] = newc.to(state["conditioning"].device, state["conditioning"].dtype)
    while state["step"] < total:
        state = adapter.advance(state)
        step, site = state["step"], _site_of(adapter, state)
        if site == "project" and step >= 1:
            per_step[step - 1] = _digest(state["latent"])
            if swap is not None and step < total:
                newc = swap(step, site, state)
                if newc is not None:
                    state["conditioning"] = newc.to(
                        state["conditioning"].device, state["conditioning"].dtype
                    )
        if state["text"] is not None and site in capture_sites:
            captured[(step, site)] = state["text"].detach().to("cpu")
        if patch is not None and state["text"] is not None:
            new = patch(step, site, state["text"])
            if new is not None:
                if new.shape != state["text"].shape or new.dtype != state["text"].dtype:
                    raise ValueError("patch shape/dtype mismatch")
                state["text"] = new.to(state["text"].device)
    return state, captured, per_step


def native_trajectory(adapter, state):
    """Ordinary diffusers full-forward trajectory for the parity oracle."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    model = adapter.model
    latent = state["latent"].clone()
    sched = FlowMatchEulerDiscreteScheduler.from_config(adapter.scheduler_config)
    sched.timesteps, sched.sigmas = state["timesteps"], state["sigmas"]
    sched.set_begin_index(0)
    per_step = []
    with torch.inference_mode():
        for timestep in state["timesteps"]:
            noise = model(
                hidden_states=latent,
                encoder_hidden_states=state["conditioning"],
                timestep=timestep.expand(1).to(latent.dtype) / 1000,
                img_ids=state["img_ids"],
                txt_ids=state["txt_ids"],
                return_dict=False,
            )[0]
            latent = sched.step(noise, timestep, latent, return_dict=False)[0]
            per_step.append(_digest(latent))
    return latent, per_step


def decode_image(adapter, final_state, size):
    session = Session(adapter, final_state)
    image = adapter.decode(session, height=size, width=size)
    return np.asarray(image.convert("RGB"))


def statecut_resume_final(adapter, state, cut_step):
    """Advance to the project boundary of ``cut_step`` with the Session, capture a StateCut,
    restore it into a fresh Session, finish via raw drive, return final-latent digest."""
    session = adapter.session(
        latent=state["latent"], conditioning=state["conditioning"], img_ids=state["img_ids"],
        txt_ids=state["txt_ids"], timesteps=state["timesteps"], sigmas=state["sigmas"],
    )
    # drive the session cheaply to the cut boundary using raw advance on its state
    s = clone(session._state)
    while s["step"] < cut_step:
        s = adapter.advance(s)
    cut_session = Session(adapter, s)
    cut = cut_session.capture()
    resumed = Session.from_cut(adapter, cut)
    final, _, _ = drive(adapter, resumed._state)
    return _digest(final["latent"]), cut.boundary, cut.fingerprint


# --------------------------------------------------------------------------------------
# contact sheets
# --------------------------------------------------------------------------------------
def save_png(arr: np.ndarray, path: Path, max_px=None):
    from PIL import Image

    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    if max_px is not None:
        img = img.copy()
        img.thumbnail((max_px, max_px))
    img.save(path, optimize=True)


def contact_sheet(cells, path: Path, cols, cell_px, labels=None, pad=6, label_h=16):
    from PIL import Image, ImageDraw

    n = len(cells)
    rows = (n + cols - 1) // cols
    W = cols * (cell_px + pad) + pad
    H = rows * (cell_px + pad + label_h) + pad
    sheet = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, arr in enumerate(cells):
        r, c = divmod(i, cols)
        x = pad + c * (cell_px + pad)
        y = pad + r * (cell_px + pad + label_h)
        cell = Image.fromarray(arr.astype(np.uint8), "RGB").resize((cell_px, cell_px))
        sheet.paste(cell, (x, y))
        if labels:
            draw.text((x + 1, y + cell_px + 2), str(labels[i])[:28], fill=(10, 10, 10))
    sheet.save(path, optimize=True)


# --------------------------------------------------------------------------------------
# tiny fixtures (local CPU unit test) — same control flow, no weights
# --------------------------------------------------------------------------------------
def build_tiny():
    from diffusers import FlowMatchEulerDiscreteScheduler

    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    adapter = Flux2KleinAdapter.tiny(residency="resident", device="cpu")
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(3)
    g = torch.Generator().manual_seed(11)

    def inputs(seed):
        gg = torch.Generator().manual_seed(seed)
        return dict(
            latent=torch.randn(1, 8, 5, generator=gg).transpose(1, 2),
            conditioning=torch.randn(1, 3, 24, generator=gg),
            img_ids=torch.zeros(1, 5, 4),
            txt_ids=torch.zeros(1, 3, 4),
            timesteps=scheduler.timesteps,
            sigmas=scheduler.sigmas,
        )

    return adapter, inputs


def tiny_image(final_state):
    lat = final_state["latent"].detach().float().cpu().reshape(-1).numpy()
    arr = (np.clip((lat - lat.min()) / (np.ptp(lat) + EPS), 0, 1) * 255).astype(np.uint8)
    side = int(np.ceil(np.sqrt(arr.size)))
    pad = np.zeros(side * side, dtype=np.uint8)
    pad[: arr.size] = arr
    return np.stack([pad.reshape(side, side)] * 3, axis=-1)


def run_tiny_selftest():
    """Exercise driving, capture, route patch (dose 0 == source, dose 1 == donor-text),
    prompt swap, and M/D/P arithmetic on CPU tiny fixtures. Returns a checks dict."""
    adapter, inputs = build_tiny()
    sites = ("after:joint.0", "after:single.0")
    # source and donor share the initial latent (same seed); donor differs only in conditioning
    base = inputs(1)
    donor_cond = torch.randn(1, 3, 24, generator=torch.Generator().manual_seed(2))
    src = init_state(adapter, **base)
    donor = init_state(adapter, **{**base, "conditioning": donor_cond})

    src_final, h_src, src_steps = drive(adapter, src, capture_sites=sites)
    donor_final, h_donor, _ = drive(adapter, donor, capture_sites=sites)
    assert set(h_src) == set(h_donor) and len(h_src) == len(sites) * len(inputs(1)["timesteps"])

    def route(dose, cut):
        def patch(step, site, text):
            if step < cut or site not in sites:
                return None
            hs, hd = h_src[(step, site)], h_donor[(step, site)]
            return hs + dose * (hd - hs)

        final, _, _ = drive(adapter, src, patch=patch)
        return final

    dose0 = route(0.0, 0)
    assert _digest(dose0["latent"]) == _digest(src_final["latent"]), "dose0 must equal source"
    dose1 = route(1.0, 0)
    assert _digest(dose1["latent"]) != _digest(src_final["latent"]), "dose1 must move"

    # prompt swap at cut 0 == pure donor (shared latent, donor conditioning)
    def swap(step, site, state):
        return donor_cond

    swap0, _, _ = drive(adapter, src, swap=swap)
    assert _digest(swap0["latent"]) == _digest(donor_final["latent"]), "cut0 swap == donor"

    # M/D/P arithmetic through the real exp2 functions (joint.0 as the exp2-style site)
    site = "after:joint.0"
    hL_step = {s: h_src[(s, si)] for (s, si) in h_src if si == site}
    hR_step = {s: h_donor[(s, si)] for (s, si) in h_donor if si == site}
    seq_len = hL_step[0].shape[1]
    pred_mask = _projector(seq_len, [1])
    arms = exp2_arms(hL_step, hR_step, pred_mask, "cpu")
    HR = hR_step[0]
    HL = hL_step[0]
    # (M+D_rest) + D_pred == H_R and (M-D_rest) + (-D_pred) == H_L, with D_pred=(M+D_pred)-M
    M0 = (HL + HR) / 2
    assert torch.allclose(arms["M"][1][0], M0)
    assert torch.allclose(arms["M+D_rest"][1][0] + (arms["M+D_pred"][1][0] - M0), HR)
    assert torch.allclose(arms["M-D_rest"][1][0] + (arms["M-D_pred"][1][0] - M0), HL)
    assert arms["M+D_rest"][0] == "right" and arms["M-D_rest"][0] == "left"
    # scoring plumbing
    s_img, d_img, r_img = tiny_image(src_final), tiny_image(donor_final), tiny_image(dose1)
    p = progress(r_img, s_img, d_img)
    dd = mad(r_img, s_img) / (mad(s_img, d_img) + EPS)
    return {
        "dose0_equals_source": True,
        "cut0_swap_equals_donor": True,
        "M_plus_D_equals_HR": True,
        "pred_rest_split": True,
        "progress_finite": bool(np.isfinite(p)),
        "d_finite": bool(np.isfinite(dd)),
        "stepped_vs_native_shapes": len(src_steps) == len(inputs(1)["timesteps"]),
    }


# --------------------------------------------------------------------------------------
# Experiment 1
# --------------------------------------------------------------------------------------
def exp1_smoke(adapter, pipe, size, steps, device, in_channels, src_cond, src_txt):
    spec = EXP1_SPECIMENS[0]
    lat, img_ids, timesteps, sigmas = prepare_sampling(
        pipe, spec["seed"], size, steps, device, in_channels
    )
    state = init_state(
        adapter, latent=lat, conditioning=src_cond, img_ids=img_ids,
        txt_ids=src_txt, timesteps=timesteps, sigmas=sigmas,
    )
    checks = {}
    # 1. unpatched stepped-vs-native parity
    _, _, adapter_steps = drive(adapter, state)
    _, native_steps = native_trajectory(adapter, state)
    checks["stepped_vs_native_exact"] = adapter_steps == native_steps
    # 2. StateCut resume parity (cut at step 2)
    resumed_digest, cut_boundary, cut_fp = statecut_resume_final(adapter, state, 2)
    checks["statecut_resume_exact"] = resumed_digest == native_steps[-1]
    checks["cut_boundary"] = cut_boundary
    # 3. zero-dose identity (route dose 0 == unpatched)
    unp_final, _, _ = drive(adapter, state)
    def zero_patch(step, site, text):
        return text * 1.0 if site in EXP1_SITES else None
    z_final, _, _ = drive(adapter, state, patch=zero_patch)
    checks["zero_dose_bytes_exact"] = _digest(z_final["latent"]) == _digest(unp_final["latent"])
    checks["passed"] = bool(
        checks["stepped_vs_native_exact"]
        and checks["statecut_resume_exact"]
        and checks["zero_dose_bytes_exact"]
    )
    return checks


def exp1_run(args, report, out_dir, device):
    pipe_enc = load_encoder_pipeline(args.model, device)
    tok = pipe_enc.tokenizer
    conds = {}
    for name, prompt in EXP1_PROMPTS.items():
        conds[name] = encode_prompt(pipe_enc, prompt, device)
    del pipe_enc
    _free()
    report["encode_rss_mb"] = _rss_mb()

    _reset_peak()
    pipe, adapter = load_denoise_pipeline(args.model, device)
    in_channels = adapter.model.config.in_channels
    report["adapter_identity"] = adapter.model_identity
    report["execution"] = dict(adapter.execution)

    report["smoke"] = exp1_smoke(
        adapter, pipe, args.size, args.steps, device, in_channels,
        conds["source"][0], conds["source"][1],
    )
    if not report["smoke"]["passed"]:
        return False

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cut_steps = [0, 2]
    doses = {"route": 1.0}
    specimens_out = []
    sheet_cells, sheet_labels = [], []

    for spec in EXP1_SPECIMENS:
        axis_cond = conds[spec["donor"]]
        src_cond, src_txt = conds["source"]
        lat, img_ids, timesteps, sigmas = prepare_sampling(
            pipe, spec["seed"], args.size, args.steps, device, in_channels
        )
        base = dict(latent=lat, img_ids=img_ids, timesteps=timesteps, sigmas=sigmas)
        src_state = init_state(adapter, conditioning=src_cond, txt_ids=src_txt, **base)

        # pure trajectories: source, axis donor, hostile donor (same latent/img_ids)
        src_final, h_src, _ = drive(adapter, src_state, capture_sites=EXP1_SITES)
        I_src = decode_image(adapter, src_final, args.size)

        donor_state = init_state(adapter, conditioning=axis_cond[0], txt_ids=axis_cond[1], **base)
        donor_final, h_donor, _ = drive(adapter, donor_state, capture_sites=EXP1_SITES)
        I_donor = decode_image(adapter, donor_final, args.size)

        host_cond = conds["hostile"]
        host_state = init_state(adapter, conditioning=host_cond[0], txt_ids=host_cond[1], **base)
        host_final, h_host, _ = drive(adapter, host_state, capture_sites=EXP1_SITES)
        I_host = decode_image(adapter, host_final, args.size)

        denom = mad(I_src, I_donor) + EPS
        denom_host = mad(I_src, I_host) + EPS

        def route_patch(hA, hB, dose, cut):
            def patch(step, site, text):
                if step < cut or site not in EXP1_SITES:
                    return None
                a, b = hA[(step, site)].to(device), hB[(step, site)].to(device)
                return a + dose * (b - a)
            return patch

        def swap_fn(donor_c):
            def swap(step, site, state):
                return donor_c if step >= swap.cut else None
            return swap

        rows = {}
        images = {"src": I_src, "donor": I_donor, "hostile_donor": I_host}
        for cut in cut_steps:
            # (a) ROUTE dose 1 toward axis donor
            r_final, _, _ = drive(
                adapter, src_state, patch=route_patch(h_src, h_donor, 1.0, cut)
            )
            I_route = decode_image(adapter, r_final, args.size)
            # (b) PROMPT-SWAP toward axis donor
            sw = swap_fn(axis_cond[0]); sw.cut = cut
            if cut == 0:
                I_swap = I_donor  # swap at cut 0 is the pure donor generation
            else:
                s_final, _, _ = drive(adapter, src_state, swap=sw)
                I_swap = decode_image(adapter, s_final, args.size)
            P_route = progress(I_route, I_src, I_donor)
            P_swap = progress(I_swap, I_src, I_donor)
            d = mad(I_route, I_swap) / denom
            rows[f"cut{cut}"] = {
                "P_route": P_route,
                "P_swap": P_swap,
                "d": d,
                "original_P": EXP1_ORIGINAL[spec["id"]].get(f"cut{cut}"),
            }
            images[f"route_cut{cut}"] = I_route
            if cut != 0:
                images[f"swap_cut{cut}"] = I_swap

        # (c) hostile arms at cut 0
        rh_final, _, _ = drive(adapter, src_state, patch=route_patch(h_src, h_host, 1.0, 0))
        I_route_host = decode_image(adapter, rh_final, args.size)
        I_swap_host = I_host  # hostile swap at cut 0 == hostile donor
        hostile = {
            "route_own_progress": progress(I_route_host, I_src, I_host),
            "route_target_progress": progress(I_route_host, I_src, I_donor),
            "swap_own_progress": progress(I_swap_host, I_src, I_host),
            "d_hostile": mad(I_route_host, I_swap_host) / denom_host,
        }
        images["route_hostile_cut0"] = I_route_host

        # rollback evidence: re-run source unpatched, require identical bytes
        src_final2, _, _ = drive(adapter, src_state)
        rollback_exact = _digest(src_final2["latent"]) == _digest(src_final["latent"])

        for name, arr in images.items():
            save_png(arr, images_dir / f"{spec['id']}-{name}.png", max_px=args.size)
        for name in ["src", "donor", "route_cut0", "route_cut2", "route_hostile_cut0"]:
            sheet_cells.append(images[name])
            sheet_labels.append(f"{spec['id']} {name}")

        specimens_out.append(
            {
                "id": spec["id"],
                "axis": spec["axis"],
                "seed": spec["seed"],
                "prompts": {"source": EXP1_PROMPTS["source"], "donor": EXP1_PROMPTS[spec["donor"]],
                            "hostile": EXP1_PROMPTS["hostile"]},
                "cuts": rows,
                "hostile_cut0": hostile,
                "rollback_latent_bytes_exact": bool(rollback_exact),
                "peak_vram_mb": _peak_vram_mb(),
            }
        )
        report.setdefault("_vram", []).append(_peak_vram_mb())

    # decisions per cut
    decisions = {}
    for cut in cut_steps:
        key = f"cut{cut}"
        rows = [s["cuts"][key] for s in specimens_out]
        approx = all(abs(r["P_route"] - r["P_swap"]) <= 0.05 and r["d"] <= 0.10 for r in rows)
        differs = sum(r["d"] > 0.25 for r in rows) >= 3
        decisions[key] = "route≈prompt-swap" if approx else ("route-differs" if differs else "partial")
    report["specimens"] = specimens_out
    report["decisions"] = decisions
    report["route_sites"] = list(EXP1_SITES)
    report["cut_steps"] = cut_steps

    contact_sheet(
        sheet_cells, out_dir / "contact-sheet-exp1.png", cols=5, cell_px=200, labels=sheet_labels
    )
    report["contact_sheet"] = "contact-sheet-exp1.png"
    report["peak_vram_mb"] = max([v for v in report.pop("_vram", []) if v] or [None] or [0])
    return True


# --------------------------------------------------------------------------------------
# Experiment 2
# --------------------------------------------------------------------------------------
def _projector(seq_len, rows):
    """Boolean mask over the sequence (token-row) dimension; True at predicate rows."""
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[rows] = True
    return mask


def exp2_arms(h_L, h_R, pred_mask, device):
    """Return {arm_name: (requested_side, {step: joint3_text_vector})} for M and M±D arms.

    h_L / h_R are {step: text-state tensor}; pred_mask is a bool mask over token rows.
    Sign convention follows D=(H_R-H_L)/2 so M+D = H_R (right); M-D = H_L (left).
    """
    steps = sorted(h_L)
    M, Dpred, Drest = {}, {}, {}
    for s in steps:
        HL = h_L[s].to(device)
        HR = h_R[s].to(device)
        M[s] = (HL + HR) / 2
        D = (HR - HL) / 2
        dp = D.clone()
        dp[:, ~pred_mask, :] = 0  # keep only predicate rows
        Dpred[s] = dp
        Drest[s] = D - dp
    arms = {
        "M": ("left", {s: M[s] for s in steps}),  # M-alone: expected left-bias (secondary)
        "M+D_rest": ("right", {s: M[s] + Drest[s] for s in steps}),
        "M-D_rest": ("left", {s: M[s] - Drest[s] for s in steps}),
        "M+D_pred": ("right", {s: M[s] + Dpred[s] for s in steps}),
        "M-D_pred": ("left", {s: M[s] - Dpred[s] for s in steps}),
    }
    return arms


def exp2_run(args, report, out_dir, device, prompts=EXP2_PROMPTS, predicate=EXP2_PREDICATE,
             seeds=EXP2_SEEDS):
    pipe_enc = load_encoder_pipeline(args.model, device)
    tok = pipe_enc.tokenizer
    conds = {name: encode_prompt(pipe_enc, p, device) for name, p in prompts.items()}
    rows_L, dec_L = resolve_predicate_rows(tok, prompts["left"], predicate)
    rows_R, dec_R = resolve_predicate_rows(tok, prompts["right"], predicate)
    if rows_L != rows_R:
        raise SystemExit(f"predicate rows differ between left/right: {rows_L} vs {rows_R}")
    del pipe_enc
    _free()
    report["encode_rss_mb"] = _rss_mb()
    report["prompts"] = dict(prompts)
    report["predicate"] = predicate
    report["predicate_rows"] = rows_L
    report["predicate_tokens"] = dec_L

    _reset_peak()
    pipe, adapter = load_denoise_pipeline(args.model, device)
    in_channels = adapter.model.config.in_channels
    width = adapter.model.config.num_attention_heads * adapter.model.config.attention_head_dim
    seq_len = conds["left"][0].shape[1]
    pred_mask = _projector(seq_len, rows_L)  # mask over the 512 token rows
    report["adapter_identity"] = adapter.model_identity
    report["execution"] = dict(adapter.execution)
    report["text_width"] = width
    report["seq_len"] = int(seq_len)

    # ---- smoke gate on first seed ----
    seed0 = seeds[0]
    lat, img_ids, timesteps, sigmas = prepare_sampling(
        pipe, seed0, args.size, args.steps, device, in_channels
    )
    base0 = dict(latent=lat, img_ids=img_ids, timesteps=timesteps, sigmas=sigmas)
    left0 = init_state(adapter, conditioning=conds["left"][0], txt_ids=conds["left"][1], **base0)
    checks = {}
    _, _, a_steps = drive(adapter, left0)
    _, n_steps = native_trajectory(adapter, left0)
    checks["stepped_vs_native_exact"] = a_steps == n_steps
    rd, cut_bd, _ = statecut_resume_final(adapter, left0, 2)
    checks["statecut_resume_exact"] = rd == n_steps[-1]
    # zero-dose identity: writing captured H_L back at joint.3 must reproduce native-left bytes
    lf_un, hL0, _ = drive(adapter, left0, capture_sites=(EXP2_SITE,))
    def writeback(step, site, text):
        return hL0[(step, site)].to(device) if site == EXP2_SITE else None
    lf_wb, _, _ = drive(adapter, left0, patch=writeback)
    checks["writeback_bytes_exact"] = _digest(lf_wb["latent"]) == _digest(lf_un["latent"])
    checks["passed"] = bool(
        checks["stepped_vs_native_exact"]
        and checks["statecut_resume_exact"]
        and checks["writeback_bytes_exact"]
    )
    report["smoke"] = checks
    if not checks["passed"]:
        return False

    # ---- full panel: 16 seeds x 8 arms, blinded ----
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    key = {}
    per_arm_files = {}
    vram = []
    rng = np.random.default_rng(0xBF1C0)

    for seed in seeds:
        lat, img_ids, timesteps, sigmas = prepare_sampling(
            pipe, seed, args.size, args.steps, device, in_channels
        )
        base = dict(latent=lat, img_ids=img_ids, timesteps=timesteps, sigmas=sigmas)
        neu = init_state(adapter, conditioning=conds["neutral"][0], txt_ids=conds["neutral"][1], **base)
        left = init_state(adapter, conditioning=conds["left"][0], txt_ids=conds["left"][1], **base)
        right = init_state(adapter, conditioning=conds["right"][0], txt_ids=conds["right"][1], **base)

        lf, hL, _ = drive(adapter, left, capture_sites=(EXP2_SITE,))
        rf, hR, _ = drive(adapter, right, capture_sites=(EXP2_SITE,))
        nf, _, _ = drive(adapter, neu)
        images = {
            "neutral": decode_image(adapter, nf, args.size),
            "native_left": decode_image(adapter, lf, args.size),
            "native_right": decode_image(adapter, rf, args.size),
        }
        sides = {"neutral": "none", "native_left": "left", "native_right": "right"}

        hL_step = {s: hL[(s, EXP2_SITE)] for (s, _) in hL}
        hR_step = {s: hR[(s, EXP2_SITE)] for (s, _) in hR}
        arms = exp2_arms(hL_step, hR_step, pred_mask, device)
        for arm, (side, byd) in arms.items():
            def patch(step, site, text, byd=byd):
                return byd[step].to(device) if site == EXP2_SITE and step in byd else None
            af, _, _ = drive(adapter, neu, patch=patch)
            images[arm] = decode_image(adapter, af, args.size)
            sides[arm] = side

        for arm, arr in images.items():
            jid = uuid.uuid4().hex[:12]
            save_png(arr, judge_dir / f"{jid}.png", max_px=args.judge_px)
            key[jid] = {"arm": arm, "seed": seed, "requested_side": sides[arm]}
            per_arm_files.setdefault(arm, []).append(jid)
        vram.append(_peak_vram_mb())

    # blinded contact sheets over shuffled ids
    ids = list(key)
    rng.shuffle(ids)
    id_to_arr = {}
    from PIL import Image
    for jid in ids:
        id_to_arr[jid] = np.asarray(Image.open(judge_dir / f"{jid}.png").convert("RGB"))
    per_sheet = 16
    sheets = []
    for i in range(0, len(ids), per_sheet):
        chunk = ids[i : i + per_sheet]
        sp = out_dir / f"contact-sheet-{args.experiment}-{i // per_sheet:02d}.png"
        contact_sheet(
            [id_to_arr[j] for j in chunk], sp, cols=4, cell_px=200, labels=chunk
        )
        sheets.append(sp.name)

    # non-blinded native-left/right viability sheet for the coordinator's scene gate
    nat_ids = [(j, key[j]) for j in key if key[j]["arm"] in ("native_left", "native_right")]
    nat_ids.sort(key=lambda kv: (kv[1]["seed"], kv[1]["arm"]))
    contact_sheet(
        [id_to_arr[j] for j, _ in nat_ids],
        out_dir / "native-left-right-viability.png",
        cols=4,
        cell_px=188,
        labels=[f"s{v['seed']} {v['arm'].split('_')[1]}" for _, v in nat_ids],
    )

    (out_dir / "judge-key.json").write_text(json.dumps(key, indent=2))
    report["judge_dir"] = "judge"
    report["judge_key"] = "judge-key.json"
    report["contact_sheets"] = sheets
    report["native_viability_sheet"] = "native-left-right-viability.png"
    # record arm -> requested_side map and counts
    side_map = {}
    for jid, meta in key.items():
        side_map[meta["arm"]] = meta["requested_side"]
    report["arms"] = {a: {"requested_side": side_map[a], "n": len(per_arm_files[a])} for a in per_arm_files}
    report["seeds"] = list(seeds)
    report["n_images"] = len(key)
    report["peak_vram_mb"] = max([v for v in vram if v] or [0])
    report["judging"] = "blinded; not judged by worker; primary=M±D_rest requested-side contact"
    return True


def exp2b_run(args, report, out_dir, device):
    """Full 8-arm blinded run on the coordinator-approved Variant B prompts + 16 fresh seeds."""
    v = EXP2_RETRY_VARIANTS["B"]
    prompts = {"neutral": EXP2B_NEUTRAL, "left": v["left"], "right": v["right"]}
    report["variant"] = "B"
    report["gate_decision"] = (
        "Variant B passed (coordinator non-blinded read: left ~14/16, right ~15/16); "
        "Variant A not used (crowded/ambiguous)."
    )
    return exp2_run(
        args, report, out_dir, device,
        prompts=prompts, predicate=v["predicate"], seeds=EXP2_VIAB_SEEDS,
    )


# --------------------------------------------------------------------------------------
# Experiment 2 — viability retry (gate-first: native left/right only, 2 prompt variants)
# --------------------------------------------------------------------------------------
def exp2viab_run(args, report, out_dir, device):
    pipe_enc = load_encoder_pipeline(args.model, device)
    tok = pipe_enc.tokenizer
    conds, variants_meta = {}, {}
    for vname, v in EXP2_RETRY_VARIANTS.items():
        conds[(vname, "left")] = encode_prompt(pipe_enc, v["left"], device)
        conds[(vname, "right")] = encode_prompt(pipe_enc, v["right"], device)
        rL, dL = resolve_predicate_rows(tok, v["left"], v["predicate"])
        rR, dR = resolve_predicate_rows(tok, v["right"], v["predicate"])
        if rL != rR:
            raise SystemExit(f"variant {vname}: predicate rows differ L={rL} R={rR}")
        il = tok(
            tok.apply_chat_template(
                [{"role": "user", "content": v["left"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ),
            padding="max_length", truncation=True, max_length=512,
        )["input_ids"]
        ir = tok(
            tok.apply_chat_template(
                [{"role": "user", "content": v["right"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ),
            padding="max_length", truncation=True, max_length=512,
        )["input_ids"]
        diff = [i for i, (a, b) in enumerate(zip(il, ir)) if a != b]
        if len(diff) != 1:
            raise SystemExit(f"variant {vname}: left/right differ at {diff}, expected 1 side token")
        variants_meta[vname] = {
            "predicate": v["predicate"],
            "predicate_rows": rL,
            "predicate_tokens": dL,
            "side_token_position": diff[0],
            "left": v["left"],
            "right": v["right"],
        }
    del pipe_enc
    _free()
    report["encode_rss_mb"] = _rss_mb()
    report["variants"] = variants_meta

    _reset_peak()
    pipe, adapter = load_denoise_pipeline(args.model, device)
    in_channels = adapter.model.config.in_channels
    report["adapter_identity"] = adapter.model_identity
    report["execution"] = dict(adapter.execution)

    # smoke gate on variant A / left / first viability seed
    seed0 = EXP2_VIAB_SEEDS[0]
    lat, img_ids, timesteps, sigmas = prepare_sampling(
        pipe, seed0, args.size, args.steps, device, in_channels
    )
    base0 = dict(latent=lat, img_ids=img_ids, timesteps=timesteps, sigmas=sigmas)
    s0 = init_state(adapter, conditioning=conds[("A", "left")][0], txt_ids=conds[("A", "left")][1], **base0)
    checks = {}
    _, _, a_steps = drive(adapter, s0)
    _, n_steps = native_trajectory(adapter, s0)
    checks["stepped_vs_native_exact"] = a_steps == n_steps
    rd, _, _ = statecut_resume_final(adapter, s0, 2)
    checks["statecut_resume_exact"] = rd == n_steps[-1]
    lf_un, hc, _ = drive(adapter, s0, capture_sites=(EXP2_SITE,))
    def writeback(step, site, text):
        return hc[(step, site)].to(device) if site == EXP2_SITE else None
    lf_wb, _, _ = drive(adapter, s0, patch=writeback)
    checks["writeback_bytes_exact"] = _digest(lf_wb["latent"]) == _digest(lf_un["latent"])
    checks["passed"] = bool(
        checks["stepped_vs_native_exact"]
        and checks["statecut_resume_exact"]
        and checks["writeback_bytes_exact"]
    )
    report["smoke"] = checks
    if not checks["passed"]:
        return False

    # render native left/right per variant across the 16 fresh seeds; one sheet per variant
    sheets, vram = {}, []
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for vname in EXP2_RETRY_VARIANTS:
        cells, labels = [], []
        for seed in EXP2_VIAB_SEEDS:
            lat, img_ids, timesteps, sigmas = prepare_sampling(
                pipe, seed, args.size, args.steps, device, in_channels
            )
            base = dict(latent=lat, img_ids=img_ids, timesteps=timesteps, sigmas=sigmas)
            for side in ("left", "right"):
                st = init_state(
                    adapter, conditioning=conds[(vname, side)][0],
                    txt_ids=conds[(vname, side)][1], **base,
                )
                fin, _, _ = drive(adapter, st)
                img = decode_image(adapter, fin, args.size)
                cells.append(img)
                labels.append(f"s{seed} {side}")
                save_png(img, images_dir / f"variant{vname}-s{seed}-{side}.png", max_px=args.judge_px)
            vram.append(_peak_vram_mb())
        sp = out_dir / f"viability-variant-{vname}.png"
        contact_sheet(cells, sp, cols=4, cell_px=200, labels=labels)
        sheets[vname] = sp.name
    report["viability_sheets"] = sheets
    report["seeds"] = list(EXP2_VIAB_SEEDS)
    report["peak_vram_mb"] = max([v for v in vram if v] or [0])
    report["note"] = (
        "native left/right only, 2 prompt variants, 16 fresh seeds; gate-first retry after the "
        "coordinator's viability gate failed the first scene (L~0/16, R~4/16). No M/D arms, no "
        "blinded judging. Stop for coordinator to apply the >=10/16-each gate."
    )
    return True


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=["exp1", "exp2", "exp2viab", "exp2b"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--judge-px", type=int, default=320)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--source-sha256", default="")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "schema": "bfl-demo-controls-v1",
        "experiment": args.experiment,
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "device": "cuda" if _cuda() else "cpu",
        "dtype": "bfloat16",
        "backend": "diffusers-native-flux2-klein",
        "versions": {
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "source_sha256": args.source_sha256,
        "tiny": args.tiny,
    }
    try:
        import diffusers

        report["versions"]["diffusers"] = diffusers.__version__
    except Exception:
        pass

    if args.tiny:
        report["tiny_selftest"] = run_tiny_selftest()
        report["elapsed_s"] = time.perf_counter() - started
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not all(v for v in report["tiny_selftest"].values()):
            raise SystemExit("tiny self-test failed")
        return

    if args.size is None:
        args.size = 256 if args.experiment == "exp1" else 512
    report["size"] = args.size
    report["steps"] = args.steps
    device = "cuda" if _cuda() else "cpu"
    runner = {
        "exp1": exp1_run,
        "exp2": exp2_run,
        "exp2viab": exp2viab_run,
        "exp2b": exp2b_run,
    }[args.experiment]
    ok = runner(args, report, out_dir, device)
    report["elapsed_s"] = time.perf_counter() - started
    report["peak_rss_mb"] = _rss_mb()
    report["ok"] = bool(ok)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    # compact stdout summary (keeps logs small; images collected by rsync)
    summary = {k: report[k] for k in report if k not in {"specimens", "execution"}}
    print("BFL_CONTROLS_SUMMARY=" + json.dumps(summary, default=str))
    if not ok:
        raise SystemExit("smoke gate failed; aborted before full panel")


if __name__ == "__main__":
    main()
