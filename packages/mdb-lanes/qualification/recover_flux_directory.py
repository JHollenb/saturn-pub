"""Custody-preserving direct re-emission of a completed remote output directory.

Preserves failed scheduler collection; does not execute the model. Uses the
original request-sealed stdlib custody encoder and independently verifies all
archive bytes, files, request bindings and retained prefix transport chunks.
"""
import argparse
import gzip
import hashlib
import json
import runpy
import shutil
import subprocess
import time
from pathlib import Path


def recover(source, destination, host="beast", limits=None):
    started=time.monotonic()
    m=runpy.run_path(str(source/"payload/flux_family_custody.py"))
    bounds=m["_limits"](limits)
    read=lambda p:json.loads(p.read_text())
    request=m["_request"](source)
    if not(source/"job.json").exists():
        from mdb_lanes.job_support import scheduler_api
        job_id=read(source/"submission.json")["job_id"]
        status,raw,_=scheduler_api().request("GET","/api/jobs/"+job_id,timeout_s=30)
        if status>=400:raise ValueError("job read failed")
        job=json.loads(raw)
        if job["state"]!="succeeded":raise ValueError("directory is not completed")
        m["_save"](source/"job.json",job)
    job=read(source/"job.json")
    if job["state"]!="succeeded" or host!="beast":raise ValueError("only this completed Beast lane is qualified")
    code=(source/"payload/flux_family_custody.py").read_text()
    if hashlib.sha256(code.encode()).hexdigest()!=request["source_hashes"]["flux_family_custody.py"]:raise ValueError("encoder source differs")
    remote=Path(request["config"]["output_scratch"])/job["job_id"]/"binding-minimal-result"
    destination.mkdir(parents=True,exist_ok=False)
    for name in("request.json","plan.json","job.json"):
        if(source/name).exists():shutil.copyfile(source/name,destination/name)
    shutil.copytree(source/"payload",destination/"payload")
    m["_save"](destination/"collection-limits.json",dict(overrides=limits or {},effective=bounds))
    script="import json\nfrom pathlib import Path\n"+"namespace={}\nexec("+repr(code)+",namespace)\n"+"output=Path("+repr(str(remote))+")\nreport=json.loads((output/'summary.json').read_text())\n"+"assert report['job_id']=="+repr(job["job_id"])+" and report['request_sha256']=="+repr(request["request_sha256"])+"\nnamespace['emit'](output,report,limits="+repr(limits)+",compression_level=1)\n"
    rawhash=hashlib.sha256();count=0
    with(destination/"recovery-stderr.txt").open("wb")as err,(destination/"archive.tar.gz").open("xb")as archive,(destination/"worker.log.gz").open("xb")as log:
        process=subprocess.Popen(["ssh",host,"python3","-I","-"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
        process.stdin.write(script.encode());process.stdin.close();parser=m["_Parts"](archive,bounds)
        with gzip.GzipFile(filename="",fileobj=log,mode="wb",mtime=0,compresslevel=1)as compressed:
            while block:=process.stdout.read(1<<20):
                count+=len(block)
                if count>bounds["max_log_bytes"]:process.kill();raise ValueError("bounded recovery exceeded")
                rawhash.update(block);compressed.write(block);parser.feed(block)
        if process.wait()!=0:raise RuntimeError("SSH custody encoding failed; retained stderr")
        envelope=parser.finish()
    transferred=time.monotonic()
    m["_binding"](envelope,job,request,job["job_id"])
    inventory=m["_scan_archive"](destination/"archive.tar.gz",bounds,envelope["archive_root"])
    extracted=destination/"extracted";extracted.mkdir()
    m["_scan_archive"](destination/"archive.tar.gz",bounds,envelope["archive_root"],extracted)
    m["_check_manifest"](extracted/envelope["archive_root"],inventory,bounds)
    if request["config"].get("directory_custody"):
        from mdb_lanes.job_support import scheduler_api
        api=scheduler_api()
        with m["_open_logs"](api,"/api/jobs/"+job["job_id"]+"/logs?offset=0&wait_s=1")as response:
            log=response.read(1<<20)
            if response.read(1):raise ValueError("unexpectedly large metadata-only scheduler log")
        ready=[json.loads(line[len(b"MDB_DIRECTORY_READY "):])for line in log.splitlines()if line.startswith(b"MDB_DIRECTORY_READY ")]
        if len(ready)!=1:raise ValueError("missing native directory publication")
        v=ready[0]
        if v["job_id"]!=job["job_id"] or v["request_sha256"]!=request["request_sha256"] or v["output_root"]!=str(remote) or v["files_manifest_sha256"]!=envelope["files_manifest_sha256"] or v["summary_sha256"]!=m["_sha"](extracted/envelope["archive_root"]/"summary.json")[0]:raise ValueError("directory publication differs")
        (destination/"scheduler-metadata.log").write_bytes(log)
    previous=read(source/"archive-custody.json") if(source/"archive-custody.json").exists() else None
    if previous and previous["parts"]!=parser.parts[:len(previous["parts"])]:raise ValueError("recovered archive differs from original complete scheduler chunks")
    save=m["_save"];sha=m["_sha"];save(destination/"report.json",envelope)
    save(destination/"log-custody.json",dict(schema="mdb-flux-family-log-custody-v1",storage="gzip",job_id=job["job_id"],complete=True,
        job_json_sha256=sha(destination/"job.json")[0],uncompressed_sha256=rawhash.hexdigest(),uncompressed_bytes=count,
        compressed_sha256=sha(destination/"worker.log.gz")[0],compressed_bytes=sha(destination/"worker.log.gz")[1],
        scope="Exact direct SSH re-emission bytes from original request-sealed custody encoder; NOT the original scheduler log",transport="ssh-directory-reemission",error=None))
    save(destination/"archive-custody.json",dict(schema="mdb-flux-family-archive-custody-v1",job_id=job["job_id"],complete=True,
        archive_sha256=sha(destination/"archive.tar.gz")[0],archive_bytes=sha(destination/"archive.tar.gz")[1],parts=parser.parts,error=None))
    save(destination/"recovery.json",dict(schema="mdb-direct-directory-custody-v1",host=host,remote_root=str(remote),source_run=str(source),
        original_job_record_sha256=sha(source/"job.json")[0],original_partial_collection_sha256=None if previous is None else sha(source/"archive-custody.json")[0],
        original_complete_chunks_matched=0 if previous is None else len(previous["parts"]),encoder_sha256=hashlib.sha256(code.encode()).hexdigest(),model_reexecution=False,
        native_directory_publication_sha256=None if not(destination/"scheduler-metadata.log").exists() else sha(destination/"scheduler-metadata.log")[0],
        scope="Original completed directory re-encoded; any original failed scheduler collection remains unchanged"))
    result=m["verify"](destination,limits);result["transport"]="ssh-directory-reemission"
    save(destination/"collection-timing.json",dict(schema="mdb-directory-collection-timing-v1",stream_encoding_and_transfer_seconds=transferred-started,
        validation_and_extraction_seconds=time.monotonic()-transferred,total_seconds=time.monotonic()-started,
        compressed_archive_bytes=envelope["archive_bytes"],raw_transport_bytes=count,local_log_compression_level=1,new_model_calls=0))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("destination",type=Path)
    p.add_argument("--max-member-bytes",type=int,help="Explicit bounded override for a measured original artifact; does not change source outputs")
    a=p.parse_args()
    limits=None if a.max_member_bytes is None else {"max_member_bytes":a.max_member_bytes}
    print(json.dumps(recover(a.source.resolve(),a.destination.resolve(),limits=limits),sort_keys=True))
