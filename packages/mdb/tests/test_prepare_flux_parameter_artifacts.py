"""Synthetic source/alias/role/custody checks for parameter recipe preparation."""
from __future__ import annotations

import base64
import hashlib
import json
import runpy
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from mdb import parameter_artifacts

P = runpy.run_path(str(Path(__file__).parents[1]/"qualification/prepare_flux_parameter_artifacts.py"))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))


def npy(payload,shape=(2,2)):
    header = repr({"descr":"<u2","fortran_order":False,"shape":shape})
    raw = (header+" "*((64-(10+len(header)+1)%64)%64)+"\n").encode()
    return b"\x93NUMPY\x01\x00"+struct.pack("<H",len(raw))+raw+payload


def fixture(tmp_path,*,shared=False,generic_index=False,unrelated_parameters=None):
    run = tmp_path/"run"
    data = run/"extracted/synthetic-result"
    data.mkdir(parents=True)
    names = ("x_embedder.weight","single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
        "transformer_blocks.0.attn.to_q.weight","norm_out.linear.weight","proj_out.weight")
    tensors,records,header,body = {},{},{},b""
    for index,name in enumerate(names):
        payload = struct.pack("<4H",*(range(16256+index*4,16260+index*4)))
        start = len(body)
        body += payload
        tensors[name] = payload
        header[name] = {"dtype":"BF16","shape":[2,2],"data_offsets":[start,len(body)]}
        raw = npy(payload)
        path = "arrays/"+str(index)+".npy"
        (data/path).parent.mkdir(exist_ok=True)
        (data/path).write_bytes(raw)
        records[name] = {"path":path,"sha256":hashlib.sha256(raw).hexdigest(),
            "descriptor":{"dtype":"torch.bfloat16","shape":[2,2],"sha256":hashlib.sha256(payload).hexdigest()},
            "encoding":"bf16-bits-v1","stored_dtype":"uint16","stored_shape":[2,2],"native_byteorder":"little"}
    raw_header = json.dumps(header,separators=(",",":")).encode()
    raw_header += b" "*((8-len(raw_header)%8)%8)
    filename = "transformer/model.safetensors"
    checkpoint = tmp_path/"checkpoint"
    sourcefile = checkpoint/filename
    sourcefile.parent.mkdir(parents=True)
    sourcefile.write_bytes(struct.pack("<Q",len(raw_header))+raw_header+body)
    header_path = tmp_path/"checkpoint-header.json"
    save(header_path,{"checkpoint_root":str(checkpoint),"file":filename,"file_bytes":sourcefile.stat().st_size,
        "data_start":8+len(raw_header),"header":header})
    bindings = {"schema":"mdb-synthetic-binding-v1",
        "project":{"parameters":{names[0]:records[names[0]]},"native_parameters":{names[0]:records[names[0]]["descriptor"]}},
        "final_blocks":{"0":{"parameters":{"qkv_mlp_weight":records[names[1]]},
            "native_parameters":{"attn.to_qkv_mlp_proj.weight":records[names[1]]["descriptor"]},
            "native_parameter_prefix":"single_transformer_blocks.0."}},
        "joint0":{"parameters":{"attn.to_q.weight":records[names[2]]},
            "native_parameters":{"attn.to_q.weight":records[names[2]]["descriptor"]}},
        "readout":{"parameters":{"modulation_weight":records[names[3]],"projection_weight":records[names[4]]},
            "native_parameters":{"linear.weight":records[names[3]]["descriptor"]},
            "projection_native_parameters":{"weight":records[names[4]]["descriptor"]}}}
    job = {"job_id":"synthetic-completed","state":"succeeded","result":{"returncode":0}}
    request = {"source_hashes":{"worker.py":"1"*64}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    checkpoint_files = {filename:hashlib.sha256(sourcefile.read_bytes()).hexdigest()}
    source = {"schema":"mdb-synthetic-source-v1","job_id":job["job_id"],"request_sha256":request["request_sha256"],
        "source_hashes":request["source_hashes"],"checkpoint_files":checkpoint_files,"checkpoint_sha256":P["identity"](checkpoint_files)}
    summary = {"schema":"mdb-synthetic-result-v1","job_id":job["job_id"],"request_sha256":request["request_sha256"]}
    for name,value in (("source.json",source),("summary.json",summary),("binding.json",bindings)):
        save(data/name,value)
    def encode(value):
        if isinstance(value,dict):
            return {"kind":"mapping","items":{key:encode(child) for key,child in value.items()}}
        if isinstance(value,list):
            return {"kind":"list","items":[encode(child) for child in value]}
        return {"kind":"scalar","value":value}
    # Only an exact frozen bank copy preserves its established port exemptions.
    save(data/"transaction/frontiers/source_binding.json",{"value":encode(bindings)})
    if unrelated_parameters is not None:
        value = records[names[0]]["path"] if unrelated_parameters == "path" else records[names[0]]
        metadata = {"parameters":{"foo":value}}
        save(data/"unrelated.json",encode(metadata) if unrelated_parameters == "encoded-record" else metadata)
    if shared:
        save(data/"activation.json",{"output":records[names[0]]})
    if generic_index:
        save(data/"index.json",{record["path"]:{"sha256":record["sha256"]} for record in records.values()})
        save(data/"files.json",{"files":[{"path":record["path"]} for record in records.values()]})
    files = {p.relative_to(data).as_posix():{"sha256":P["sha"](p),"bytes":p.stat().st_size,"filetype":"regular"}
        for p in data.rglob("*") if p.is_file()}
    save(data/"remote-manifest.json",{"files":files,"job_id":job["job_id"],"request_sha256":request["request_sha256"]})
    files["remote-manifest.json"] = {"sha256":P["sha"](data/"remote-manifest.json"),
        "bytes":(data/"remote-manifest.json").stat().st_size,"filetype":"regular"}
    for name in ("collection-local-inventory.json","collection-remote-before.json","collection-remote-after.json"):
        save(run/name,{"files":files})
    collection = {"job_id":job["job_id"],"request_sha256":request["request_sha256"],"bindings":{},
        "summary_sha256":P["sha"](data/"summary.json"),"manifest_sha256":P["sha"](data/"remote-manifest.json")}
    for name,value in (("request.json",request),("job.json",job),("collection.json",collection)):
        save(run/name,value)
    return run,data,checkpoint,header_path,records


def prepare(run,checkpoint,header):
    return P["prepare"](run,checkpoint_header=header,checkpoint_root=str(checkpoint),ssh_host="beast")


def test_aliases_native_prefixes_exact_framing_and_standalone_roundtrip(tmp_path):
    run,data,checkpoint,header,records = fixture(tmp_path,generic_index=True)
    before = {p.relative_to(data).as_posix():P["sha"](p) for p in data.rglob("*") if p.is_file()}
    manifest = prepare(run,checkpoint,header)
    assert manifest["counts"] == {"artifacts":5,"archive_bytes":sum((data/r["path"]).stat().st_size for r in records.values()),"retained_shared_archives":0}
    assert {entry["source"]["tensor"] for entry in manifest["artifacts"]} == set(records)
    assert (run/"retrieve_weights.py").read_bytes() == Path(parameter_artifacts.__file__).read_bytes()
    assert before == {p.relative_to(data).as_posix():P["sha"](p) for p in data.rglob("*") if p.is_file()}
    for entry in manifest["artifacts"]:
        raw = (data/entry["path"]).read_bytes()
        prefix = base64.b64decode(entry["npy_prefix_b64"])
        assert raw.startswith(prefix) and hashlib.sha256(raw[len(prefix):]).hexdigest() == entry["source"]["payload_sha256"]
    stage = tmp_path/"restore"
    cmd = [sys.executable,"-I",str(run/"retrieve_weights.py"),"--checkpoint",str(checkpoint),"--root",str(stage)]
    result = subprocess.run(cmd,capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    for entry in manifest["artifacts"]:
        assert (stage/entry["path"]).read_bytes() == (data/entry["path"]).read_bytes()
    with pytest.raises(ValueError,match="overwrite"):
        prepare(run,checkpoint,header)


def test_nonparameter_metadata_use_retains_shared_archive(tmp_path):
    run,data,checkpoint,header,records = fixture(tmp_path,shared=True)
    manifest = prepare(run,checkpoint,header)
    assert len(manifest["artifacts"]) == 4
    assert records["x_embedder.weight"]["path"] not in {entry["path"] for entry in manifest["artifacts"]}
    retained = manifest["provenance"]["retained_shared_archives"]
    assert retained[0]["reason"] == "nonparameter-metadata-use" and retained[0]["uses"][0]["file"] == "activation.json"
    assert (data/retained[0]["path"]).is_file()


@pytest.mark.parametrize("form",["path","record","encoded-record"])
def test_arbitrary_parameters_ancestor_never_exempts_evidence(tmp_path,form):
    run,data,checkpoint,header,records = fixture(tmp_path,unrelated_parameters=form)
    manifest = prepare(run,checkpoint,header)
    retained = manifest["provenance"]["retained_shared_archives"]
    assert len(manifest["artifacts"]) == 4 and len(retained) == 1
    assert retained[0]["path"] == records["x_embedder.weight"]["path"]
    assert retained[0]["uses"][0]["file"] == "unrelated.json"
    assert (data/retained[0]["path"]).is_file()


@pytest.mark.parametrize("corruption",["archive","descriptor","metadata","header","job","source"])
def test_corrupt_sources_refused_before_writes(tmp_path,corruption):
    run,data,checkpoint,header,records = fixture(tmp_path)
    if corruption == "archive":
        path = data/records["x_embedder.weight"]["path"]
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    elif corruption in ("descriptor","metadata"):
        path = data/"binding.json"
        binding = json.loads(path.read_text())
        if corruption == "descriptor":
            binding["project"]["parameters"]["x_embedder.weight"]["descriptor"]["sha256"] = "0"*64
        else:
            binding["undeclared"]="changed"
        save(path,binding)
    elif corruption == "header":
        value = json.loads(header.read_text())
        value["header"]["x_embedder.weight"]["shape"] = [1,4]
        save(header,value)
    elif corruption == "job":
        job = json.loads((run/"job.json").read_text())
        job["state"]="failed"
        save(run/"job.json",job)
    else:
        source = json.loads((data/"source.json").read_text())
        source["checkpoint_sha256"]="0"*64
        save(data/"source.json",source)
    with pytest.raises(ValueError):
        prepare(run,checkpoint,header)
    assert not (run/"parameter-artifacts.json").exists() and not (run/"retrieve_weights.py").exists()


def test_native_descriptor_is_payload_authority(tmp_path):
    run,data,checkpoint,header,records = fixture(tmp_path)
    record = records["x_embedder.weight"]
    with pytest.raises(ValueError,match="native descriptor"):
        P["verify_archive"](data/record["path"],{**record,"descriptor":{**record["descriptor"],"sha256":"0"*64}},
            {"shape":[2,2],"data_offsets":[0,8]},P["sha"](checkpoint/"transformer/model.safetensors"),"transformer/model.safetensors")


@pytest.mark.parametrize("value",["../escape","/absolute","a/../b","a\\b"])
def test_paths_cannot_escape(value):
    with pytest.raises(ValueError,match="path"):
        P["relative"](value)
