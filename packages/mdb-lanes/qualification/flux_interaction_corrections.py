"""Finite source-derived correction algebra, before native consumers run."""
from __future__ import annotations


def correction_fields(fields):
    """Return BF16-rounded current-write candidates and unrounded diagnostics.

    Alinear = Acolor + Ashape - Abase. Consumer C maps Alinear to Aadd;
    representation R maps Aadd to Ajoint. Source roles are observed-entry
    equation predictions, not independently inferred semantic variables.
    """
    import numpy as np
    if set(fields)!={"base","color","shape","additive","joint"}:
        raise ValueError("all five declared attention roles are required")
    shape=fields["base"].shape
    if any(not isinstance(v,np.ndarray) or v.dtype!=np.float32 or v.shape!=shape or v.ndim!=3
           or not np.isfinite(v).all() or np.any(v.view(np.uint32)&65535) for v in fields.values()):
        raise ValueError("finite matching widened BF16 fields required")
    f={k:v.astype(np.float64) for k,v in fields.items()}
    linear=f["color"]+f["shape"]-f["base"]
    consumer=f["additive"]-linear;representation=f["joint"]-f["additive"]
    def bf16(value):
        bits=np.asarray(value,np.float32).view(np.uint32)
        return ((bits+np.uint32(32767)+((bits>>16)&np.uint32(1)))&np.uint32(0xffff0000)).view(np.float32)
    doses=[(0,0),(0,1),(1,0),(1,1),(-1,0),(0,-1)]
    candidates={"r%dc%d"%(r,c):bf16(linear+r*representation+c*consumer) for r,c in doses}
    if not np.array_equal(candidates["r0c1"],fields["additive"]) or not np.array_equal(candidates["r1c1"],fields["joint"]):
        raise ValueError("declared correction corners fail arithmetic closure")
    return candidates,{"linear":linear,"consumer":consumer,"representation":representation}


__all__=["correction_fields"]
