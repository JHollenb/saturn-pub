"""Hydrate native computation from input/Act metadata; export hashes only."""
import base64
from contextlib import nullcontext
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
import torch
from mdb_lanes.attention_observer import AttentionObserver
from mdb_lanes.native_replay_recipe import NativeReplay
from mdb_lanes.prompt_qwen import PromptQwenAdapter
from saturn_pub import StateAddress


def save(p,v):
    p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")


def native_npy_hash(tensor):
    import numpy as np
    stream=io.BytesIO()
    np.save(stream,tensor.detach().contiguous().cpu().view(torch.uint8).numpy(),allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def main():
    import parameter_source
    config=json.loads(Path("saturn_run_config.json").read_text())
    request=json.loads(Path("saturn_debug_request.json").read_text())
    recipe=json.loads(Path("recipe.json").read_text())
    checks=json.loads(Path("expectations.json").read_text())
    root=Path(config["output_root"])/os.environ.get("MRUN_JOB_ID",os.environ.get("MRUN_RUN_ID","unknown"))
    root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(4)
    manifest=parameter_source._write_parameter_manifest(Path(config["model_path"]),root/"parameter-artifacts.json")
    _,model,_,weights,_,_=parameter_source._load_model(config,manifest["provenance"])
    adapter=PromptQwenAdapter(model)
    observations=checks["observations"]
    counts={"attention_arrays":0,"logits_arrays":0,"decisions":0,"commits":0}
    def observe(event):
        return AttentionObserver(model) if observations.get(event["id"],{}).get("traces") else nullcontext()
    def inspect(event,session,observed):
        expected=observations.get(event["id"],{})
        if "logits" in expected:
            value=session.read("logits")
            if native_npy_hash(value)!=expected["logits"]["sha256"]: raise ValueError("native logits bytes differ")
            if int(value[0].argmax())!=expected["next_token_id"]: raise ValueError("native decision differs")
            counts["logits_arrays"]+=1;counts["decisions"]+=1
        if "committed_token_id" in expected:
            if session.read("tokens").tolist()[0]!=expected["input_ids"]+[expected["committed_token_id"]]:
                raise ValueError("native own-token commit differs")
            counts["commits"]+=1
        if expected.get("traces"):
            captured={r["layer"]:r for r in observed.rows}
            for trace in expected["traces"]:
                for key in ("query","key","value","attention","head_return"):
                    if native_npy_hash(captured[trace["layer"]][key])!=trace[key]["sha256"]:
                        raise ValueError("native recapture differs: "+event["id"]+"/"+key)
                    counts["attention_arrays"]+=1
    replay=NativeReplay(recipe,adapter).run(observe=observe,inspect=inspect)
    target=next(e["id"] for e in recipe["events"] if e["id"].endswith("native-recipient-token0-readout"))
    hydrated=replay.materialize(target,StateAddress("logits"))
    if native_npy_hash(hydrated)!=observations[target]["logits"]["sha256"]: raise ValueError("selected locator hydration differs")
    save(root/"regenerated-events.json",replay.rows)
    summary={"schema":"mdb-native-metadata-regeneration-v1","job_id":root.name,"request_sha256":request["request_sha256"],
        "model_identity":weights,"recipe_fingerprint":recipe["fingerprint"],"worker_wall_s":time.monotonic()-started,
        "counts":{**counts,"regenerated_cuts":len(replay.rows),"roots":len(recipe["roots"]),"selected_locator_reads":1},
        "mechanics_status":"valid","instrument_status":"from token roots and native Acts; no retained activation/cut payload inputs",
        "terminal_status":"bounded replay mechanics only; not all model families or arbitrary mutable closure",
        "native_steps":sum(e.get("steps",0) for e in recipe["events"]),"historical_futures":checks["historical_futures"],
        "activation_payload_bytes_exported":0,"copied_pretrained_parameter_payloads":0}
    save(root/"report.json",summary)
    entries=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob("*")) if p.is_file()]
    save(root/"output-manifest.json",{"files":entries,"expanded_bytes":sum(e["bytes"] for e in entries)})
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w:gz") as archive:
        for p in sorted(root.rglob("*")):
            if p.is_file():archive.add(p,arcname=p.relative_to(root).as_posix())
    raw=stream.getvalue()
    summary["archive"]={"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
    print("SCAFFOLD_REPORT_V1 "+base64.b64encode(gzip.compress(json.dumps(summary).encode())).decode(),flush=True)
    print("SCAFFOLD_ARCHIVE_V1 0/1 "+base64.b64encode(raw).decode(),flush=True)


if __name__=="__main__":main()
