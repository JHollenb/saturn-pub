"""Fixed-operand attention microscope; no model or attention-module wrappers."""
from contextlib import nullcontext

LANES = ('auto-text', 'auto-full-zero', 'auto-full-repeat', 'math-text',
         'flash-text', 'staged-fp32', 'staged-probability-bf16')


def queries(q, total, mode):
    """Only synthetic unused query rows change; every text query is preserved."""
    import torch
    if q.ndim != 4 or total < q.shape[1]:
        raise ValueError('invalid query geometry')
    if mode == 'text':
        return q
    extra = total - q.shape[1]
    if mode == 'zero':
        filler = torch.zeros((q.shape[0], extra, *q.shape[2:]), device=q.device, dtype=q.dtype)
    elif mode == 'repeat':
        filler = q.repeat(1, (extra + q.shape[1] - 1) // q.shape[1], 1, 1)[:, :extra]
    else:
        raise ValueError('unknown padding control')
    return torch.cat((q, filler), dim=1)


def attention(q, k, v, lane, counts):
    """Return B/text/width and the actual selected ATen operations."""
    import torch
    from torch.nn import functional as f
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if lane not in LANES or q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError('attention lane/geometry differs')
    nt = q.shape[1]
    mode = lane.removeprefix('auto-full-') if lane.startswith('auto-full-') else 'text'
    Q, K, V = (t.permute(0, 2, 1, 3) for t in (queries(q, k.shape[1], mode), k, v))
    profile = dict(q_shape=list(Q.shape), k_shape=list(K.shape), v_shape=list(V.shape),
                   q_stride=list(Q.stride()), k_stride=list(K.stride()), v_stride=list(V.stride()),
                   unused_query_rows=Q.shape[2]-nt, padding=mode)
    policy = {'math-text': SDPBackend.MATH, 'flash-text': SDPBackend.FLASH_ATTENTION}.get(lane)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        if lane.startswith('staged-'):
            scores = (Q.float() @ K.float().transpose(-1, -2)) * (Q.shape[-1] ** -.5)
            probability = torch.softmax(scores, dim=-1)
            if lane == 'staged-probability-bf16':
                result = probability.to(V.dtype) @ V
            else:
                result = (probability @ V.float()).to(V.dtype)
            counts['staged_matmuls'] += 2
            counts['staged_softmax'] += 1
        else:
            with sdpa_kernel(policy) if policy is not None else nullcontext():
                result = f.scaled_dot_product_attention(Q, K, V, attn_mask=None,
                    dropout_p=0., is_causal=False, scale=None, enable_gqa=False)
            counts['sdpa'] += 1
        if result.is_cuda:
            torch.cuda.synchronize()
    profile['selected_aten_operations'] = [dict(name=x.key, calls=x.count) for x in trace.key_averages()
        if any(s in x.key for s in ('scaled_dot', 'softmax', 'bmm'))]
    return result[:, :, :nt].permute(0, 2, 1, 3).reshape(q.shape[0], nt, -1), profile
