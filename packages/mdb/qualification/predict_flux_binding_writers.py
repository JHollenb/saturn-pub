"""Freeze causal write vectors from saved native ports, before new suffixes."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


def prepare(root,output):
    from mdb.tensor_archive import write_tensor_archive
    # Only the NumPy reader/rounder is used; no model or Torch is imported.
    del write_tensor_archive
    here=Path(__file__).resolve().parent
    h=runpy.run_path(str(here/"analyze_flux_binding_minimal.py"));slot=h["read_slot"]
    B=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    data=root/"streams-02/extracted/binding-minimal-result";read=lambda p:json.loads(p.read_text())
    audit=read(root/"streams-analysis-v1.json");assert audit["mechanics_status"]=="verified"
    rows={r["id"]:r for r in read(data/"rows.json")};writes={w["row"]:w for w in read(data/"writes.json")}
    def arr(name,key):
        r=writes[name]["fields"][key];p=data/r["path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==r["sha256"]
        return np.load(p,allow_pickle=False)
    entry=rows["IC"]["marks"]["joint0_entry"];baseentry=rows["I"]["marks"]["joint0_entry"]
    R=slot(data/"store",entry,"image");assert np.array_equal(R,slot(data/"store",baseentry,"image"))
    for key in ("gate_attention","gate_mlp"):assert np.array_equal(arr("IC",key),arr("I",key))
    ga,gm=arr("IC","gate_attention"),arr("IC","gate_mlp")
    A0,A1,M0,M1=arr("I","attention"),arr("IC","attention"),arr("I","mlp"),arr("IC","mlp")
    native0=slot(data/"store",rows["I"]["marks"]["joint0_exit"],"image")
    native1=slot(data/"store",rows["IC"]["marks"]["joint0_exit"],"image")
    values={name:B(B(R+B(ga*a))+B(gm*m))for name,a,m in [("A.fixed",A0,M1),("M.fixed",A1,M0),("AM.predicted",A0,M0)]}
    assert np.array_equal(values["AM.predicted"],native0)
    output.mkdir(parents=True,exist_ok=False);files={};descriptors={};local_effects={}
    delta=(native1-native0).astype(np.float64);den=np.sum(delta*delta)
    for name,v in values.items():
        # Lossless BF16 bits keep all three predicted full image fields <19MB.
        assert np.all((np.ascontiguousarray(v).view(np.uint32)&65535)==0)
        bits=(np.ascontiguousarray(v).view(np.uint32)>>16).astype("<u2")
        p=output/(name+".npy");np.save(p,bits,allow_pickle=False)
        digest=hashlib.sha256(p.read_bytes()).hexdigest();files[p.name]=dict(bytes=p.stat().st_size,sha256=digest)
        descriptors[name]=dict(path=p.name,sha256=digest,encoding="bf16-bits-v1",stored_dtype="uint16",stored_shape=list(v.shape),native_byteorder="little",native_stride=list(np.asarray(v).strides[i]//4 for i in range(v.ndim)),
            descriptor=dict(dtype="torch.bfloat16",shape=list(v.shape),sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
        local_effects[name]=dict(projection_on_full_change=float(np.sum((v-native0).astype(np.float64)*delta)/den),correction_rms=float(np.sqrt(np.mean((v-native1).astype(np.float64)**2))))
    port_records={}
    for label,name,key in [("base_attention","I","attention"),("base_mlp","I","mlp"),("edited_mlp","IC","mlp")]:
        r=writes[name]["fields"][key];port_records[label]=dict(r,bytes=(data/r["path"]).stat().st_size)
    manifest=dict(schema="mdb-binding-writer-forecasts-v1",source_job=read(data/"summary.json")["job_id"],
        writer_root=read(data/"summary.json")["output_root"]+"/binding-minimal-result",source_audit_sha256=hashlib.sha256((root/"streams-analysis-v1.json").read_bytes()).hexdigest(),
        positive_control_png_sha256=rows["IC"]["png_sha256"],vectors=descriptors,files=files,ports=port_records,local_effects=local_effects,
        hypotheses=dict(categorical={n:"blue sphere left of red cube" for n in ["A.fixed","M.fixed","AM.predicted","A.recomputed.native"]},
            signed_consumer="Tentative linear-continuation rule: each restoration lowers RGB projection toward the binding donor; M.fixed lowers it more than A.fixed. Test and retain reversals.",
            causal="Predict single J00 restoration is repaired by remaining donor-C computation, retaining categorical binding; this may fail.",
            mathematical="Pure BF16 sequential image equations predict three full installed fields. Native A-only recomputation must equal the AM.predicted field because R/gates and the resulting MLP input match I. Text exit remains native IC. Compare both suffixes, not only equations.",
            scope="Observed native module operands from first panel, independent NumPy elementary BF16 write calculation; edited suffixes have not run. No independent raw-prompt/whole-image prediction."),
        new_model_calls=0,AM_equals_observed_I_exit=True,
        scope="Local write vectors use saved full R, native gates and attention/MLP returns. Corrected futures remain prospective native consumers.")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(vectors=len(values),bytes=sum(v["bytes"]for v in files.values()),local_effects=local_effects,new_model_calls=0)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("output",type=Path);a=p.parse_args();prepare(a.root.resolve(),a.output.resolve())
