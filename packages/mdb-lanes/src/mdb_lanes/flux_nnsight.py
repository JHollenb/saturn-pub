"""nnsight as an optional access layer to FLUX.2 Klein attention, with attention_port's arithmetic on top.

Optional: needs the ``nnsight`` extra of ``mdb-lanes`` (``pip install 'mdb-lanes[nnsight]'``, pinned
``nnsight==0.8.2``). Importing this module never imports nnsight; :func:`wrap` does, and refuses
another version unless asked (the parity below was measured on 0.8.2 only).

Promoted verbatim from ``experiments/interp_bridge_20261114/bridge.py`` (sha256 dd9de615..., the
file the jobs ran; it stays there unchanged as evidence): the functions from :func:`attn_envoy`
to :func:`compare_tensors`. The attribution half of that file (``Probe``, ``step_forward``,
``backward_trajectory``, ``head_terms``, group selection) is not promoted: it is the experiment's
question, and its first-order prediction failed for large cuts (FINDINGS F2.4, F2.5).

How it is used. The saturn_pub ``Flux2KleinAdapter`` loop calls the transformer's blocks one by
one and never the transformer's ``forward``, so a trace runs that loop as its ``fn``; every
block's attention call is reached at ``DISPATCH_PATH``. Cuts go through :func:`port_edit`, which
is ``attention_port.AttentionPort.edit`` on a ``Read`` built from the call's q, k, v and output,
so the cut arithmetic, the port's checks (rows, heads, empty receivers) and every verdict computed
downstream by ``attention_panels`` are the lane's, unchanged. nnsight only delivers the operands
and writes the edited output back.

What the parity covers (MEASURED, interp-bridge smoke job-8fd7bb3b1604, FLUX.2 Klein 4B, BF16,
CUDA, nnsight 0.8.2, 256 px, 4 steps, seed 4242; ``q1_table.json``):

* capture: q, k, v, native head output and the FP32 softmax from nnsight's q, k are bitwise equal
  to the port observer's, at one cell (block 10, head 13, step 0);
* cuts: zero and mask cuts (block 10, head 13, ``own:ball`` sources, every step) give final
  latents and RGB bitwise equal to the port's, and differ from native;
* passivity: wrapping, a no-op trace and the capture trace leave the native latent unchanged; the
  5 root latents equal the lane reference hashes;
* cost: capture 0.337-0.356 s per generation vs the port's 0.341-0.361 s; peak 7,760 MB vs 7,773 MB.

Not covered (not yet known): other cells, blocks and steps than the one cell; set cuts of many
heads (the designed-group job job-1d4b6df1ba24 cut through the port, not nnsight); the ``legacy``
precision of the port; other nnsight versions, resolutions, step counts and models; Qwen (the
engine's ``mdb.trace`` covers Qwen). nnsight's gradient differed from the plain-torch probe by up
to 2.3e-6 (not bitwise).
"""
from __future__ import annotations

NNSIGHT_VERSION = "0.8.2"
EXTRA = "mdb-lanes[nnsight]"


def available():
    """Whether nnsight can be imported (without importing anything else)."""
    import importlib.util
    return importlib.util.find_spec("nnsight") is not None


def wrap(transformer, *, allow_other_version=False):
    """``nnsight.NNsight(transformer)``: the envoy the functions below take as ``nn_tr``.

    Raises ``ImportError`` naming the extra when nnsight is missing, and ``RuntimeError`` for a
    version other than ``NNSIGHT_VERSION`` unless ``allow_other_version``."""
    try:
        import nnsight
    except ImportError as error:
        raise ImportError(f"flux_nnsight needs nnsight=={NNSIGHT_VERSION}: install the optional "
                          f"extra {EXTRA!r}") from error
    version = getattr(nnsight, "__version__", None)
    if version != NNSIGHT_VERSION and not allow_other_version:
        raise RuntimeError(f"nnsight {version} is installed; the port parity was measured on "
                           f"{NNSIGHT_VERSION} only (pass allow_other_version=True to use it anyway)")
    return nnsight.NNsight(transformer)


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


