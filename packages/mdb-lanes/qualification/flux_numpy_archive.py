"""Framework-free CAS writer compatible with MDB's public BF16 archive schema.

Only exact finite FP32 widenings of BF16 are accepted. Original live tensor
strides are not invented: records describe the new contiguous prediction.
"""
import hashlib
import json
import sys


def record_bf16(root,value,records):
    import numpy as np
    value=np.ascontiguousarray(value,np.float32)
    if not value.size or not np.isfinite(value).all() or np.any(value.view(np.uint32)&65535):raise ValueError('expected nonempty finite BF16 widening')
    bits=(value.view(np.uint32)>>16).astype('<u2');native=bits.astype('<u2' if sys.byteorder=='little' else '>u2',copy=False).tobytes()
    descriptor=dict(shape=list(value.shape),dtype='torch.bfloat16',sha256=hashlib.sha256(native).hexdigest(),preview=value.reshape(-1)[:8].tolist());key=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest()
    if key not in records:
        p=root/'arrays'/(key+'.npy');p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:np.save(f,bits,allow_pickle=False)
        records[key]=dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),descriptor=descriptor,stored_dtype=str(bits.dtype),stored_shape=list(value.shape),native_stride=[s//value.itemsize for s in value.strides],encoding='bf16-bits-v1',native_byteorder=sys.byteorder)
    return records[key]
