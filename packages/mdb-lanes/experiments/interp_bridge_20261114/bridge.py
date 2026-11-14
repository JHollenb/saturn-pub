"""Interp bridge 2026-11-14: nnsight access to FLUX.2 Klein attention, and first-order attribution.

Two independent pieces, both over the same saturn_pub ``Flux2KleinAdapter`` loop the read-panel
lane runs (``qualification/flux_read_panel_worker.py``):

1. nnsight (``nnsight.NNsight`` around the diffusers ``Flux2Transformer2DModel``). The adapter calls
   the transformer's blocks one by one under ``torch.inference_mode`` and never calls the
   transformer's own ``forward``, so a trace runs the adapter loop as its ``fn``. Every block's
   attention call is reached as
   ``<block>.attn.source.self_processor_0.source.dispatch_attention_fn_0``: nnsight rewrites
   ``Flux2Attention.forward`` (or ``Flux2ParallelSelfAttention.forward``), drills into the
   processor object's ``__call__`` and exposes the ``dispatch_attention_fn`` call's inputs
   (q, k, v in [B, S, H, D]) and output. Cuts use attention_port's own arithmetic
   (:func:`port_edit`), so a comparison with the port tests the access layer only.

2. Attribution with plain torch autograd. ``Flux2KleinAdapter.advance`` is decorated with
   ``torch.inference_mode`` (no graph), so :func:`step_forward` replays one denoising step with
   the same modules, arguments and order outside inference mode. :class:`Probe` overrides
   ``transformer_flux2.dispatch_attention_fn`` (as attention_port's FluxBackend does), records
   q/k/v/out per block and adds a zero tensor whose gradient is d outcome / d attention output.
   Backward runs one step at a time (step-level recompute from saved step inputs), each block
   optionally under non-reentrant activation checkpointing.

Nothing here loads a model; the GPU worker is ``worker.py``.
"""
from __future__ import annotations

import hashlib
import math

JOINT_DEFAULT = 5


# ============================================================================== nnsight access

def attn_envoy(nn_tr, block, joint=JOINT_DEFAULT):
    """The attention module envoy of global block index ``block`` (joint blocks first)."""
    if block < joint:
        return nn_tr.transformer_blocks[block].attn
    return nn_tr.single_transformer_blocks[block - joint].attn


def dispatch_op(nn_tr, block, joint=JOINT_DEFAULT):
    """nnsight operation envoy of the block's ``dispatch_attention_fn`` call (inside a trace only)."""
    return attn_envoy(nn_tr, block, joint).source.self_processor_0.source.dispatch_attention_fn_0


DISPATCH_PATH = "<attn>.source.self_processor_0.source.dispatch_attention_fn_0"


def head_record(args, kwargs, out, head):
    """CPU copies of one head's q, k, v, native output (bf16 as served) plus the call's scale."""
    q, k, v = args[0], args[1], args[2]
    scale = kwargs.get("scale")
    return dict(q=q[0, :, head].detach().cpu().clone(), k=k[0, :, head].detach().cpu().clone(),
                v=v[0, :, head].detach().cpu().clone(), out=out[0, :, head].detach().cpu().clone(),
                scale=float(scale) if scale is not None else float(q.shape[-1]) ** -0.5,
                kwargs=sorted(kwargs), n_args=len(args))


def nnsight_capture(nn_tr, run, cells, steps, joint=JOINT_DEFAULT):
    """Trace ``run`` (the adapter loop) and return {(step, block): head_record} for ``cells``.

    ``cells``: [(block, head)] with distinct blocks; read in execution order (block ascending)
    at each step in ``steps`` (ascending). Returns (records, run's return value).
    """
    cells = sorted(cells)
    steps = sorted(int(s) for s in steps)
    store = {}
    with nn_tr.trace(fn=run) as tracer:
        with tracer.invoke():          # no model input: one empty invoke sees the whole run
            for step in tracer.iter[steps]:
                for block, head in cells:
                    op = dispatch_op(nn_tr, block, joint)
                    args, kwargs = op.inputs
                    out = op.output
                    store[(int(step), int(block))] = head_record(args, kwargs, out, head)
            result = tracer.result.save()
    return store, result


def nnsight_cut(nn_tr, run, cuts, rows, steps, blocks, joint=JOINT_DEFAULT, zero_method="delta"):
    """Trace ``run`` and replace each listed block's attention output by :func:`port_edit` of it."""
    steps = sorted(int(s) for s in steps)
    blocks = sorted(int(b) for b in blocks)
    with nn_tr.trace(fn=run) as tracer:
        with tracer.invoke():
            for step in tracer.iter[steps]:
                for block in blocks:
                    op = dispatch_op(nn_tr, block, joint)
                    args, kwargs = op.inputs
                    out = op.output
                    op.output = port_edit(args[0], args[1], args[2], out, int(step), block, cuts, rows,
                                          zero_method=zero_method, scale=kwargs.get("scale"))
            result = tracer.result.save()
    return result


def nnsight_noop(nn_tr, run):
    """Trace ``run`` with a body that reads nothing (interleaving overhead only)."""
    with nn_tr.trace(fn=run) as tracer:
        with tracer.invoke():
            result = tracer.result.save()
    return result


def port_edit(q, k, v, out, step, block, cuts, rows, *, zero_method="delta", scale=None):
    """attention_port's cut arithmetic applied to one dispatch call seen through any access layer.

    q, k, v, out: [1, S, H, D] (diffusers layout). Same construction as FluxBackend.dispatch:
    ``Read`` over head-major views, ``AttentionPort.edit``, permute back, contiguous.
    """
    from mdb_lanes import attention_port as ap
    port = ap.AttentionPort(None, rows=rows, zero_method=zero_method)
    read = ap.Read(None, step, block, q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                   scale if scale is not None else q.shape[-1] ** -0.5, None, out.permute(0, 2, 1, 3))
    active = [c for c in cuts if c.active(step, block)]
    if not active:
        return out
    new, _ = port.edit(read, read.out, active)
    return new.permute(0, 2, 1, 3).contiguous()


def port_observer_reduce(head):
    """attention_port reduction storing the same fields as :func:`head_record` plus the port's FP32 probs."""
    def reduce(read):
        return dict(q=read.q[0, head].detach().cpu().clone(), k=read.k[0, head].detach().cpu().clone(),
                    v=read.v[0, head].detach().cpu().clone(), out=read.out[0, head].detach().cpu().clone(),
                    p=read.probs()[0, head].detach().cpu().clone(), scale=float(read.scale))
    reduce.__name__ = f"head{head}_qkv_out_p"
    return reduce


def probs_of(q, k, scale):
    """FP32 softmax(q k^T * scale) for one head, [S, S] (the port's Read.probs formula)."""
    import torch
    return torch.softmax(torch.matmul(q.float(), k.float().t()) * scale, dim=-1)


def compare_tensors(a, b):
    import torch
    a, b = a.float(), b.float()
    return dict(bitwise_equal=bool(torch.equal(a, b)), max_abs=float((a - b).abs().max()),
                shape=list(a.shape))


# =================================================================== differentiable replay step

class Probe:
    """Scoped override of ``transformer_flux2.dispatch_attention_fn`` for :func:`step_forward`.

    The caller sets ``probe.block`` before each block (also inside a checkpoint recompute).
    ``record=True`` keeps detached q, k, v, out per block (first call per block only, so a
    checkpoint recompute does not overwrite). ``grads=True`` adds ``self.zeros[block]`` (a
    zero leaf that requires grad, created by :meth:`arm`) to the output; its ``.grad`` is the
    gradient of the backward's outcome with respect to that block's attention output.
    """

    def __init__(self, module=None):
        if module is None:
            from diffusers.models.transformers import transformer_flux2 as module
        self.module = module
        self.block = None
        self.record, self.grads = False, False
        self.seen, self.zeros = {}, {}
        self.calls = 0

    def arm(self, *, record, grads, blocks=None, like=None):
        self.record, self.grads = bool(record), bool(grads)
        self.seen, self.zeros, self.calls = {}, {}, 0
        if grads:
            import torch
            if like is None:
                raise ValueError("grads=True needs like=(shape, dtype, device)")
            shape, dtype, device = like
            self.zeros = {b: torch.zeros(shape, dtype=dtype, device=device, requires_grad=True) for b in blocks}
        return self

    def __enter__(self):
        if getattr(self.module.dispatch_attention_fn, "_interp_bridge_probe", False):
            raise ValueError("nested probes")
        self.original = self.module.dispatch_attention_fn

        def dispatch(q, k, v, *args, **kwargs):
            return self.dispatch(q, k, v, *args, **kwargs)
        dispatch._interp_bridge_probe = True
        self.module.dispatch_attention_fn = dispatch
        return self

    def __exit__(self, *exc):
        self.module.dispatch_attention_fn = self.original
        return False

    def dispatch(self, q, k, v, *args, **kwargs):
        if args or kwargs.get("attn_mask") is not None:
            raise ValueError("Probe: native call carries a mask or positional extras; unsupported")
        self.calls += 1
        out = self.original(q, k, v, **kwargs)
        b = self.block
        if self.record and b not in self.seen:
            self.seen[b] = (q.detach(), k.detach(), v.detach(), out.detach(), kwargs.get("scale"))
        if self.grads:
            out = out + self.zeros[b]
        return out


def step_forward(adapter, x, conditioning, img_ids, txt_ids, timesteps, sigmas, step, *, probe=None,
                 checkpoint=False):
    """One denoising step of ``Flux2KleinAdapter.advance`` (operation granularity), graph-enabled.

    Same modules, keyword arguments and order as advance(): project (time embedding with the
    /1000*1000 round trip, rope, modulations, x/context embedders), joint blocks, single blocks
    on cat([text, image]), norm_out/proj_out readout, a fresh FlowMatchEulerDiscreteScheduler
    with the session's timesteps/sigmas and ``set_begin_index(step)``. Returns the next latent.
    """
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    model, dtype = adapter.model, adapter.dtype
    joint = len(model.transformer_blocks)
    timestep = timesteps[step].expand(1).to(dtype)
    temb = model.time_guidance_embed((timestep / 1000).to(dtype) * 1000, None)
    img_rope, txt_rope = model.pos_embed(img_ids[0]), model.pos_embed(txt_ids[0])
    img_mod = model.double_stream_modulation_img(temb)
    txt_mod = model.double_stream_modulation_txt(temb)
    single_mod = model.single_stream_modulation(temb)
    rope = tuple(torch.cat([t, i], dim=0) for t, i in zip(txt_rope, img_rope))
    image = model.x_embedder(x)
    text = model.context_embedder(conditioning)

    def call(index, fn, *tensors):
        def run(*ts):
            if probe is not None:
                probe.block = index
            return fn(*ts)
        if checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as ckpt
            return ckpt(run, *tensors, use_reentrant=False)
        return run(*tensors)

    for b in range(joint):
        blk = model.transformer_blocks[b]
        text, image = call(b, lambda img, txt, blk=blk: blk(
            hidden_states=img, encoder_hidden_states=txt, temb_mod_img=img_mod, temb_mod_txt=txt_mod,
            image_rotary_emb=rope, joint_attention_kwargs=None), image, text)
    T = text.shape[1]
    for i, blk in enumerate(model.single_transformer_blocks):
        combined = call(joint + i, lambda txt, img, blk=blk: blk(
            hidden_states=torch.cat([txt, img], dim=1), encoder_hidden_states=None, temb_mod=single_mod,
            image_rotary_emb=rope, joint_attention_kwargs=None), text, image)
        text, image = combined[:, :T], combined[:, T:]
    noise = model.proj_out(model.norm_out(image, temb))
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(adapter.scheduler_config)
    scheduler.timesteps, scheduler.sigmas = timesteps, sigmas
    scheduler.set_begin_index(step)
    return scheduler.step(noise, timesteps[step], x, return_dict=False)[0]


def latent_sha(x):
    """The lane's latent hash: sha256 of float32 bytes of latent[0] (worker ``finish``)."""
    import numpy as np
    lat = x.detach().float().cpu().numpy()[0]
    return hashlib.sha256(lat.astype(np.float32).tobytes()).hexdigest()


# ============================================================================ outcome + grads

def outcome(x, base, direction):
    """y = <x[0].float() - base, direction> / |direction|^2 (1 at base + direction, 0 at base)."""
    d = direction.float()
    return ((x[0].float() - base.float()) * d).sum() / (d * d).sum()


def backward_trajectory(adapter, xs, inputs, base, direction, probe, *, checkpoint=True, on_step=None):
    """Gradients of :func:`outcome` at the final latent with respect to every block's attention output.

    ``xs``: the trajectory's step inputs [x_0, ..., x_N] (x_N final, all detached). Runs
    step N-1 ... 0, each from its saved input with the graph enabled (step-level recompute),
    backward with the downstream latent gradient, and calls ``on_step(step, probe, recomputed)``
    while the probe still holds that step's q/k/v/out and per-block gradients.
    Returns dict(final_y, recompute_equal=[...], grad_x0 norm).
    """
    import torch
    steps = len(xs) - 1
    g_next = None
    report = {"recompute_equal": {}, "y_final": None}
    for s in range(steps - 1, -1, -1):
        x = xs[s].detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
        with torch.enable_grad():
            probe.arm(record=True, grads=True, blocks=range(len(adapter.model.transformer_blocks)
                                                            + len(adapter.model.single_transformer_blocks)),
                      like=(inputs["shape"], adapter.dtype, x.device))
            with probe:     # patched through backward too: checkpoint recompute re-adds the probes
                nxt = step_forward(adapter, x, inputs["conditioning"], inputs["img_ids"], inputs["txt_ids"],
                                   inputs["timesteps"], inputs["sigmas"], s, probe=probe, checkpoint=checkpoint)
                report["recompute_equal"][s] = bool(torch.equal(nxt.detach(), xs[s + 1]))
                if g_next is None:
                    y = outcome(nxt, base, direction)
                    report["y_final"] = float(y.detach())
                    y.backward()
                else:
                    nxt.backward(g_next)
        g_next = x.grad.detach()
        if on_step is not None:
            on_step(s, probe)
        probe.arm(record=False, grads=False)
    report["grad_x0_norm"] = float(g_next.float().norm())
    return report


# ============================================================================ attribution math

def head_terms(q, k, v, out_native, g, receivers, sources, scale, w_img, w_txt, text_rows):
    """Per-head first-order terms for one (step, block), every head at once.

    q, k, v, out_native, g: [1, S, H, D] (g = d outcome / d attention output, zeros where no
    gradient). ``receivers``: long index of query rows (image and/or text); ``sources``: long
    index of key rows cut. Returns tensors [H] keyed by name, split by receiver class
    ("img" = rows >= text_rows, "txt" = rows < text_rows):

    - ``att_zero_*``: sum_i <g_i, -sum_{j in S} p_ij v_j>  (first-order change of the outcome
      under the no-renormalisation cut);
    - ``att_mask_*``: sum_i <g_i, (m_i o_i - sum_{j in S} p_ij v_j) / (1 - m_i)>  with
      o_i = sum_j p_ij v_j and m_i = sum_{j in S} p_ij (renormalising cut);
    - ``mass_*``: sum_i m_i (read mass); ``wmass_*``: sum_i sum_{j in S} p_ij |W_O,h v_j|
      (write mass through the receiver's output projection);
    - ``dwrite_zero_*`` / ``dwrite_mask_*``: sum_i |W_O,h Delta o_i| (predicted write size,
      no consumer).
    All arithmetic FP32 from the bf16 tensors; p is recomputed (the native kernel never
    materialises it).
    """
    import torch
    H = q.shape[2]
    qh = q[0].permute(1, 0, 2).float()
    kh = k[0].permute(1, 0, 2).float()
    vh = v[0].permute(1, 0, 2).float()
    gh = g[0].permute(1, 0, 2).float()
    out = {}
    for cls, mask in (("img", receivers >= text_rows), ("txt", receivers < text_rows)):
        rows = receivers[mask]
        if rows.numel() == 0:
            for name in ("att_zero", "att_mask", "mass", "wmass", "dwrite_zero", "dwrite_mask"):
                out[f"{name}_{cls}"] = torch.zeros(H)
            continue
        p = torch.softmax(torch.matmul(qh[:, rows], kh.transpose(1, 2)) * scale, dim=-1)   # [H, R, S]
        ps = p[:, :, sources]                                                               # [H, R, |S|]
        m = ps.sum(-1)                                                                      # [H, R]
        pv = torch.matmul(ps, vh[:, sources])                                               # [H, R, D]
        o = torch.matmul(p, vh)                                                             # [H, R, D]
        dz = -pv
        dm = (m[..., None] * o - pv) / (1.0 - m)[..., None].clamp_min(1e-12)
        gr = gh[:, rows]
        out[f"att_zero_{cls}"] = (gr * dz).sum((1, 2)).cpu()
        out[f"att_mask_{cls}"] = (gr * dm).sum((1, 2)).cpu()
        out[f"mass_{cls}"] = m.sum(-1).cpu()
        w = w_img if cls == "img" else w_txt
        D = vh.shape[-1]
        wh = w.float().view(w.shape[0], H, D).permute(1, 0, 2)                               # [H, out, D]
        vnorm = torch.matmul(vh[:, sources], wh.transpose(1, 2)).norm(dim=-1)              # [H, |S|]
        out[f"wmass_{cls}"] = (ps * vnorm[:, None, :]).sum((1, 2)).cpu()
        out[f"dwrite_zero_{cls}"] = torch.matmul(dz, wh.transpose(1, 2)).norm(dim=-1).sum(-1).cpu()
        out[f"dwrite_mask_{cls}"] = torch.matmul(dm, wh.transpose(1, 2)).norm(dim=-1).sum(-1).cpu()
        del p, ps, pv, o, dz, dm
    return out


def out_weights(transformer, block):
    """(image-receiver W_O, text-receiver W_O) of a block, as attention_port.FluxBackend._out_weights."""
    joint = len(transformer.transformer_blocks)
    if block < joint:
        attn = transformer.transformer_blocks[block].attn
        return attn.to_out[0].weight, attn.to_add_out.weight
    attn = transformer.single_transformer_blocks[block - joint].attn
    w = attn.to_out.weight[:, : attn.inner_dim]
    return w, w


# ============================================================================ group selection

def rank_desc(values):
    """Rank 0 = largest value; ties broken by index (stable)."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def partitions(removal, mass, n_groups=20):
    """The preregistered head partitions (PREREG.md section 4).

    ``removal[i]``: attribution-predicted removal of head i (minus its first-order outcome
    change); ``mass[i]``: its read mass. Heads are ranked by each (rank 0 = largest); the
    anti-diagonal partition sorts heads by rank_removal - rank_mass and cuts the order into
    ``n_groups`` consecutive equal groups; the diagonal partition sorts by
    rank_removal + rank_mass. Ties by head index. Returns {"anti": [[i...]...], "diag": [...]}.
    """
    n = len(removal)
    if n % n_groups:
        raise ValueError("heads must split evenly into groups")
    ra, rm = rank_desc(removal), rank_desc(mass)
    size = n // n_groups
    out = {}
    for name, key in (("anti", lambda i: (ra[i] - rm[i], i)), ("diag", lambda i: (ra[i] + rm[i], i))):
        order = sorted(range(n), key=key)
        out[name] = [order[g * size:(g + 1) * size] for g in range(n_groups)]
    return out


# ================================================================================== statistics

def spearman(x, y):
    """Spearman rank correlation with average ranks for ties (pure Python)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(list(x)), ranks(list(y))
    n = len(rx)
    if n < 3:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def loo_spearman(x, y):
    """Leave-one-out Spearman values (one per left-out item)."""
    x, y = list(x), list(y)
    return [spearman(x[:i] + x[i + 1:], y[:i] + y[i + 1:]) for i in range(len(x))]
