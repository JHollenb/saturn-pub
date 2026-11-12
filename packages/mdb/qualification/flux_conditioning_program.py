"""Observed conditioning compositions and entry-only native-projector reference."""


def bf16(value):
    import numpy as np
    bits=np.asarray(value,np.float32).view(np.uint32)
    return ((bits+np.uint32(32767)+((bits>>16)&np.uint32(1)))&np.uint32(0xffff0000)).view(np.float32)


def conditioning_cases(values):
    import numpy as np
    if set(values)!={"base","color","shape","joint"}:raise ValueError("four conditioning sources required")
    shape=values["base"].shape
    if any(v.dtype!=np.float32 or v.shape!=shape or v.ndim!=3 or not np.isfinite(v).all() for v in values.values()):raise ValueError("conditioning geometry differs")
    result={k:v.copy() for k,v in values.items()}
    result["additive"]=bf16(values["color"].astype(np.float64)+values["shape"]-values["base"])
    result["interaction_only"]=bf16(values["base"].astype(np.float64)+values["joint"]-result["additive"])
    return result


def predict_projection(conditioning,weight,bias=None,*,chunk=256):
    import numpy as np
    if (conditioning.ndim!=3 or conditioning.shape[0]!=1 or weight.ndim!=2 or conditioning.shape[-1]!=weight.shape[-1]
            or type(chunk)!=int or chunk<1 or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in (conditioning,weight))):raise ValueError("projection operands differ")
    if bias is not None and (bias.shape!=(weight.shape[0],) or bias.dtype!=np.float32 or not np.isfinite(bias).all()):raise ValueError("bias differs")
    x=conditioning.astype(np.float64);out=np.empty((*conditioning.shape[:2],weight.shape[0]),np.float32)
    for start in range(0,weight.shape[0],chunk):
        stop=min(start+chunk,weight.shape[0]);value=x@weight[start:stop].astype(np.float64).T
        if bias is not None:value+=bias[start:stop].astype(np.float64)
        out[:,:,start:stop]=bf16(value)
    return out
