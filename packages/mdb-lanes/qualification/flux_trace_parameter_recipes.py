"""Describe required native parameters by original checkpoint recipes, without exporting them."""
import base64
import hashlib
import io
import json
import shutil
import struct
from pathlib import Path


def recipes(model,root,inventory,out,payload):
    import numpy as np
    import torch
    from mdb_lanes.parameter_artifacts import validate_manifest
    filename='transformer/diffusion_pytorch_model.safetensors';path=Path(root)/filename
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
    prefixes=tuple(['transformer_blocks.'+str(i)+'.' for i in (2,3,4)]+['single_transformer_blocks.'+str(i)+'.' for i in range(20)]+['norm_out.','proj_out.'])
    artifacts=[]
    for name,value in model.named_parameters():
        if not name.startswith(prefixes):continue
        spec=header[name]
        if spec['dtype']!='BF16' or list(value.shape)!=spec['shape'] or value.dtype!=torch.bfloat16:raise ValueError('native parameter header differs')
        bits=value.detach().view(torch.uint16).cpu().numpy().astype('<u2',copy=False);raw=bits.tobytes()
        with path.open('rb') as f:
            first,last=spec['data_offsets'];f.seek(8+size+first);checkpoint=f.read(last-first)
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=hashlib.sha256(checkpoint).hexdigest():raise ValueError('native parameter differs from original checkpoint')
        prefix=io.BytesIO();np.lib.format.write_array_header_1_0(prefix,dict(descr='<u2',fortran_order=False,shape=bits.shape));prefix=prefix.getvalue()
        artifacts.append(dict(path='parameters/'+name+'.npy',bytes=len(prefix)+len(raw),sha256=hashlib.sha256(prefix+raw).hexdigest(),npy_prefix_b64=base64.b64encode(prefix).decode(),
            parameter_ports=[name],source=dict(file=filename,file_sha256=inventory[filename],tensor=name,**spec,payload_sha256=digest)))
    manifest=validate_manifest(dict(schema='mdb-parameter-artifacts-v1',artifacts=artifacts,defaults=dict(checkpoint_root=str(root),artifact_root='.',ssh_host='beast',python='python3'),
        provenance=dict(scope='Original parameters actually used by joint2..4,single0..19,norm_out/proj_out. Native bytes match named original checkpoint entries. No parameter files persist in this run.')))
    (out/'parameter-artifacts.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    return dict(tensors=len(artifacts),logical_parameter_bytes=sum(x['bytes'] for x in artifacts),persisted_parameter_bytes=0)
