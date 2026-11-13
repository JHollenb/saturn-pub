"""Prepare source-bound exact parameter retrieval recipes; never remove evidence.

Run after completed authenticated collection. The original result tree and its
historical inventories remain immutable. Reconstruction must be proved separately
before an operator chooses to thin duplicate parameter files.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath

SCHEMA = "mdb-parameter-artifacts-v1"
MAX_JSON_BYTES = 32 << 20
CHUNK = 8 << 20
SINGLE_ALIASES = {"qkv_mlp_weight":"attn.to_qkv_mlp_proj.weight","qkv_mlp_bias":"attn.to_qkv_mlp_proj.bias",
    "to_out_weight":"attn.to_out.weight","to_out_bias":"attn.to_out.bias",
    "norm_q_weight":"attn.norm_q.weight","norm_k_weight":"attn.norm_k.weight"}


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def pairs(items):
    result = {}
    for key,value in items:
        require(key not in result,"duplicate JSON key")
        result[key] = value
    return result


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_JSON_BYTES,"invalid bounded JSON metadata file")
    return json.loads(path.read_text(),object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite metadata")))


def relative(value):
    require(isinstance(value,str) and value and "\\" not in value and "\0" not in value,"invalid relative artifact path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value and value != ".","unsafe relative artifact path")
    return path


def member(root,name):
    path = Path(root)/str(relative(name))
    require(not any(parent.is_symlink() for parent in (path,*path.parents)),"symlink evidence path")
    return path


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def descriptor_key(value):
    require(isinstance(value,dict) and value.get("dtype") == "torch.bfloat16"
        and isinstance(value.get("shape"),list) and isinstance(value.get("sha256"),str),"invalid BF16 native descriptor")
    return value["dtype"],tuple(value["shape"]),value["sha256"]


def infer_data(run,collection,job,request):
    candidates = []
    for path in sorted((run/"extracted").iterdir()):
        if path.is_dir() and not path.is_symlink() and all((path/name).is_file() for name in ("source.json","summary.json","binding.json","remote-manifest.json")):
            source,summary = read(path/"source.json"),read(path/"summary.json")
            if source.get("job_id") == summary.get("job_id") == job["job_id"]:
                candidates.append(path)
    require(len(candidates) == 1,"expected one source-bound collected result root")
    data = candidates[0]
    source,summary,binding = (read(data/name) for name in ("source.json","summary.json","binding.json"))
    require(isinstance(source.get("schema"),str) and source["schema"].startswith("mdb-")
        and isinstance(summary.get("schema"),str) and summary["schema"].startswith("mdb-")
        and source["request_sha256"] == summary["request_sha256"] == collection["request_sha256"] == request["request_sha256"]
        and source["source_hashes"] == request["source_hashes"]
        and sha(data/"summary.json") == collection["summary_sha256"],"collected producer source/summary/request differs")
    return data,source,summary,binding


def qualified_native_names(spec,path,port,record):
    native = spec["native_parameters"]
    selected = port if port in native else SINGLE_ALIASES.get(port)
    if path and path[-1] == "readout":
        if port.startswith("projection_"):
            native = spec.get("projection_native_parameters",{})
            selected,prefix = port.removeprefix("projection_"),"proj_out."
        else:
            selected,prefix = "linear."+port.removeprefix("modulation_"),"norm_out."
    else:
        prefix = spec.get("native_parameter_prefix")
        if prefix is None:
            prefix = ""
            for index,part in enumerate(path):
                if part.startswith("joint") and part[5:].isdigit():
                    prefix = "transformer_blocks."+part[5:]+"."
                elif part in ("joints","blocks","final_blocks","extra_blocks") and index+1 < len(path) and path[index+1].isdigit():
                    prefix = ("transformer_blocks." if part == "joints" else "single_transformer_blocks.")+path[index+1]+"."
    matches = [name for name,value in native.items() if descriptor_key(value) == descriptor_key(record["descriptor"])]
    if selected is not None:
        require(selected in matches,"parameter alias/native descriptor differs")
        matches = [selected]
    require(len(matches) == 1,"ambiguous parameter/native descriptor join")
    return prefix+matches[0]


def parameter_candidates(binding,header):
    """Match explicit parameter ports to qualified checkpoint names, never values alone."""
    result = {}
    def visit(node,path):
        if isinstance(node,dict):
            if isinstance(node.get("parameters"),dict) and isinstance(node.get("native_parameters"),dict):
                for port,record in node["parameters"].items():
                    if record is None:
                        continue
                    if not (isinstance(record,dict) and record.get("encoding") == "bf16-bits-v1"):
                        continue
                    name = qualified_native_names(node,path,port,record)
                    require(name in header,"parameter name absent from checkpoint header: "+name)
                    tensor = header[name]
                    require(tensor["dtype"] == "BF16" and tensor["shape"] == record["descriptor"]["shape"],
                        "parameter/checkpoint tensor dtype or shape differs")
                    archive = str(relative(record["path"]))
                    old = result.get(archive)
                    if old is not None:
                        require(old["record"] == record,"same parameter archive has conflicting metadata")
                        old["tensors"].add(name)
                        old["ports"].append("/".join((*path,"parameters",port)))
                        old["port_locations"].append([*path,"parameters",port])
                    else:
                        result[archive] = {"record":record,"tensors":{name},"ports":["/".join((*path,"parameters",port))],
                            "port_locations":[[*path,"parameters",port]]}
            for key,value in node.items():
                visit(value,(*path,str(key)))
        elif isinstance(node,list):
            for index,value in enumerate(node):
                visit(value,(*path,str(index)))
    visit(binding,())
    return result


def metadata_uses(data,inventory,candidates):
    """Exempt exact established port records, including exact encoded bank copies.

    A key named parameters alone grants no exemption. Recognized bank metadata
    must equal its bound counterpart, and only the independently established
    complete parameter record subtrees within that bank are exempted.
    """
    uses = {name:[] for name in candidates}
    scanned = []
    binding = read(Path(data)/"binding.json")
    banks = {}
    for item in candidates.values():
        for location in item["port_locations"]:
            bank = binding
            for key in location[:-2]:
                bank = bank[int(key)] if isinstance(bank,list) else bank[key]
            port = location[-1]
            require(bank["parameters"][port] == item["record"],"established parameter port differs")
            banks.setdefault(identity(bank),set()).add(port)
    def decode(node):
        require(isinstance(node,dict),"invalid encoded metadata")
        if set(node) == {"kind","value"} and node["kind"] == "scalar":
            return node["value"]
        if set(node) == {"kind","items"} and node["kind"] == "mapping" and isinstance(node["items"],dict):
            return {key:decode(value) for key,value in node["items"].items()}
        if set(node) == {"kind","items"} and node["kind"] in ("list","tuple") and isinstance(node["items"],list):
            return [decode(value) for value in node["items"]]
        raise ValueError("invalid encoded binding copy")
    def established_records(node):
        if "parameters" in node and "native_parameters" in node:
            ports = banks.get(identity(node),())
            return {id(node["parameters"][port]) for port in ports}
        if (node.get("kind") == "mapping" and isinstance(node.get("items"),dict)
                and {"parameters","native_parameters"} <= set(node["items"])):
            try:
                ports = banks.get(identity(decode(node)),())
            except (ValueError,TypeError,KeyError):
                return set()
            return {id(node["items"]["parameters"]["items"][port]) for port in ports}
        return set()
    def walk(node,location,name,exempt=frozenset()):
        if id(node) in exempt:
            return
        if isinstance(node,str) and node in uses:
            uses[node].append({"file":name,"location":"/".join(location)})
        elif isinstance(node,dict):
            exempt = exempt | established_records(node)
            for key,value in node.items():
                walk(value,(*location,str(key)),name,exempt)
        elif isinstance(node,list):
            for index,value in enumerate(node):
                walk(value,(*location,str(index)),name,exempt)
    for name in sorted(inventory["files"]):
        if not name.endswith((".json",".jsonl")) or name in ("files.json","remote-manifest.json"):
            continue
        path = member(data,name)
        require(path.is_file() and sha(path) == inventory["files"][name]["sha256"],"metadata source inventory differs: "+name)
        scanned.append(name)
        if name.endswith(".json"):
            walk(read(path),(),name)
        else:
            require(path.stat().st_size <= MAX_JSON_BYTES,"JSONL metadata exceeds bound")
            for index,line in enumerate(path.read_text().splitlines()):
                walk(json.loads(line,object_pairs_hook=pairs),(str(index),),name)
    return uses,scanned


def verify_archive(path,record,tensor,file_sha,checkpoint_file):
    """One streaming pass proves complete archive and exact raw payload identity."""
    full,payload = hashlib.sha256(),hashlib.sha256()
    with path.open("rb") as stream:
        start = stream.read(8)
        require(start[:6] == b"\x93NUMPY" and tuple(start[6:8]) in ((1,0),(2,0),(3,0)),"unsupported parameter NPY framing")
        width = 2 if start[6] == 1 else 4
        size = stream.read(width)
        require(len(size) == width,"truncated NPY prefix")
        length = int.from_bytes(size,"little")
        require(0 < length <= 65536,"parameter NPY header exceeds bound")
        raw = stream.read(length)
        require(len(raw) == length,"truncated parameter NPY header")
        prefix = start+size+raw
        parsed = ast.literal_eval(raw.decode("utf-8" if start[6] == 3 else "latin1"))
        require(parsed == {"descr":"<u2","fortran_order":False,"shape":tuple(tensor["shape"])},
            "parameter NPY must preserve exact C-order little-endian BF16 words")
        full.update(prefix)
        bytes_payload = 0
        while block := stream.read(CHUNK):
            full.update(block)
            payload.update(block)
            bytes_payload += len(block)
    require(full.hexdigest() == record["sha256"],"parameter full archive SHA256 differs")
    require(payload.hexdigest() == record["descriptor"]["sha256"],"parameter native descriptor/raw payload SHA256 differs")
    require(tensor["data_offsets"][1]-tensor["data_offsets"][0] == bytes_payload
        and record["stored_dtype"] == "uint16" and record["stored_shape"] == tensor["shape"]
        and record.get("native_byteorder") == "little","parameter checkpoint/archive byte geometry differs")
    return {"path":record["path"],"bytes":len(prefix)+bytes_payload,"sha256":full.hexdigest(),
        "npy_prefix_b64":base64.b64encode(prefix).decode(),"source":{"file":checkpoint_file,"file_sha256":file_sha,
            "dtype":"BF16","shape":tensor["shape"],"data_offsets":tensor["data_offsets"],"payload_sha256":payload.hexdigest()}}


def prepare(run,*,checkpoint_header,checkpoint_root="/mnt/big/llm-models/FLUX.2-klein-4B",ssh_host="beast",python="python3"):
    from mdb_lanes import parameter_artifacts

    run,checkpoint_header = Path(run).resolve(),Path(checkpoint_header).resolve()
    require(not any((run/name).exists() for name in ("parameter-artifacts.json","retrieve_weights.py")),"refuse to overwrite retrieval artifacts")
    request,job,collection = (read(run/name) for name in ("request.json","job.json","collection.json"))
    unsigned = {key:value for key,value in request.items() if key != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest() == request["request_sha256"],
        "request identity differs")
    require(job["state"] == "succeeded" and job["result"]["returncode"] == 0
        and collection["job_id"] == job["job_id"],"completed authenticated job required")
    for name,digest in collection["bindings"].items():
        require(sha(member(run,name)) == digest,"collection custody metadata differs")
    inventory = read(run/"collection-local-inventory.json")
    require(inventory == read(run/"collection-remote-before.json") == read(run/"collection-remote-after.json"),
        "authenticated before/after/local inventories differ")
    data,source,summary,binding = infer_data(run,collection,job,request)
    manifest = read(data/"remote-manifest.json")
    require(sha(data/"remote-manifest.json") == collection["manifest_sha256"]
        and manifest["job_id"] == job["job_id"] and manifest["request_sha256"] == request["request_sha256"]
        and manifest["files"] == {key:value for key,value in inventory["files"].items() if key != "remote-manifest.json"},
        "producer inventory binding differs")
    require(identity(source["checkpoint_files"]) == source["checkpoint_sha256"],"checkpoint source inventory identity differs")
    header = read(checkpoint_header)
    filename = str(relative(header["file"]))
    require(filename in source["checkpoint_files"] and header["data_start"] > 8
        and type(header["file_bytes"]) is int and header["file_bytes"] > header["data_start"],
        "checkpoint header/source file differs")
    tensors = {name:value for name,value in header["header"].items() if name != "__metadata__"}
    require(tensors and header["data_start"]+max(value["data_offsets"][1] for value in tensors.values()) == header["file_bytes"],
        "checkpoint header full file geometry differs")
    candidates = parameter_candidates(binding,tensors)
    uses,scanned = metadata_uses(data,inventory,candidates)
    artifacts,retained = [],[]
    for name,item in sorted(candidates.items()):
        if uses[name]:
            retained.append({"path":name,"reason":"nonparameter-metadata-use","uses":uses[name],"parameter_ports":item["ports"]})
            continue
        path = member(data,name)
        require(path.is_file() and inventory["files"][name]["sha256"] == item["record"]["sha256"]
            and path.stat().st_size == inventory["files"][name]["bytes"],"parameter archive inventory differs")
        tensor_name = sorted(item["tensors"])[0]
        entry = verify_archive(path,item["record"],tensors[tensor_name],source["checkpoint_files"][filename],filename)
        entry["source"]["tensor"] = tensor_name
        entry["parameter_ports"],entry["checkpoint_tensor_aliases"] = item["ports"],sorted(item["tensors"])
        artifacts.append(entry)
    require(artifacts,"no exclusive lossless BF16 parameter archives found")
    module = Path(inspect.getfile(parameter_artifacts)).resolve()
    result = {"schema":SCHEMA,"artifacts":artifacts,"defaults":{"artifact_root":data.relative_to(run).as_posix(),
        "checkpoint_root":str(Path(checkpoint_root)),"ssh_host":ssh_host,"python":python},
        "provenance":{"job_id":job["job_id"],"request_sha256":request["request_sha256"],"result_schema":summary["schema"],
            "job_sha256":sha(run/"job.json"),"collection_sha256":sha(run/"collection.json"),
            "old_inventory_sha256":sha(run/"collection-local-inventory.json"),
            "producer_manifest_sha256":sha(data/"remote-manifest.json"),"binding_sha256":sha(data/"binding.json"),
            "source_sha256":sha(data/"source.json"),"checkpoint_inventory_sha256":source["checkpoint_sha256"],
            "checkpoint_header_sha256":sha(checkpoint_header),"checkpoint_header":os.path.relpath(checkpoint_header,run),
            "retrieval_module_sha256":sha(module),"preparer_source_sha256":sha(Path(__file__)),
            "metadata_files_scanned":scanned,"retained_shared_archives":retained,
            "scope":"Preparation only: original archives fully hashed, raw payload bound to native descriptors and pinned checkpoint tensor geometry. No deletion and no checkpoint byte roundtrip performed. Prove exact rehydration before thinning; restore before unchanged historical full-inventory audits."},
        "counts":{"artifacts":len(artifacts),"archive_bytes":sum(row["bytes"] for row in artifacts),
            "retained_shared_archives":len(retained)}}
    result = parameter_artifacts.validate_manifest(result)
    with (run/"retrieve_weights.py").open("xb") as stream:
        stream.write(module.read_bytes())
    try:
        with (run/"parameter-artifacts.json").open("x") as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write("\n")
    except BaseException:
        # Preserve a created standalone script for inspection; never alter result evidence.
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--checkpoint-header",type=Path,required=True)
    parser.add_argument("--checkpoint-root",default="/mnt/big/llm-models/FLUX.2-klein-4B")
    parser.add_argument("--ssh-host",default="beast")
    parser.add_argument("--python",default="python3")
    args = parser.parse_args()
    result = prepare(args.run,checkpoint_header=args.checkpoint_header,checkpoint_root=args.checkpoint_root,
        ssh_host=args.ssh_host,python=args.python)
    print(json.dumps({"schema":result["schema"],"counts":result["counts"],
        "manifest":str(args.run/"parameter-artifacts.json"),"model_calls":0,"files_removed":0}))
