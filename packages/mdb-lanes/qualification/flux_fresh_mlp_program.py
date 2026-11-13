"""Frozen two-row intervention and two own-state numerical programs."""
import time
import numpy as np


def make_text(base, historical_delta, doses, bf16):
    if base.ndim != 3 or base.shape[0] != 1 or base.shape[1] <= 11 or historical_delta.shape != (1, 2, base.shape[-1]):
        raise ValueError('declared two-row geometry required')
    if len(doses) != 2 or any(not np.isfinite(d) or not 0 <= d <= 1 for d in doses):
        raise ValueError('finite predeclared doses required')
    result = base.copy()
    result[:, [4, 11]] = bf16(base[:, [4, 11]].astype(np.float64)
        + np.asarray(doses, np.float64)[None, :, None]*historical_delta.astype(np.float64))
    return result


def predict_pair(cache, text, law, reference, math):
    """Both worlds consume the same calculated upstream operands, never native targets."""
    started = time.monotonic()
    observed = {}
    old = law['predict'](cache, text, observer=lambda key, value: observed.__setitem__(key, value))
    output = dict(old={}, new={})
    width = cache['image'].shape[-1]
    for role, prefix, key in [('image', 'ff', 'img_mod'), ('text', 'ff_context', 'txt_mod')]:
        fields = old['fields'][role]
        U = observed[role+'.after_attention']
        gate = np.split(cache['context'][key].reshape(1, 1, 6*width), 6, axis=-1)[5]
        old_write = math['writeback'](U, gate, fields['mlp'], reference['bf16'])
        if not math['same_bits'](old_write['exit'], fields['exit']):
            raise ValueError('old complete program writeback differs')
        stages = {name: cache['stages'][role+'.'+name] for name in ('mlp_input', 'mlp_paired', 'mlp_activation', 'mlp')}
        weights = dict(in_=cache['weights'][prefix+'.linear_in.weight'], out=cache['weights'][prefix+'.linear_out.weight'])
        M = math['mlp'](fields['mlp_input'], stages, {'in': weights['in_'], 'out': weights['out']}, 'direct', reference, law)
        new_write = math['writeback'](U, gate, M, reference['bf16'])
        output['old'][role] = {**fields, **old_write}
        output['new'][role] = {**fields, **new_write}
    return output, dict(old_program_seconds=old['seconds'], old_affine_work=old['affine_work'], old_attention_work=old['attention_work'],
        complete_pair_seconds=time.monotonic()-started, additional_mlp_calculations=2,
        scope='Complete pair timing includes old upstream/MLPs and two direct-first replacement MLPs; archive writes and cache preparation are separate.')
