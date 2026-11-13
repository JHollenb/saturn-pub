"""Verify native-published remote evidence and retrieve selected exact members.

This is data custody, not an execution or replay path. Large states remain in
the original remote directory; no duplicate archive or pretrained bank is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


REMOTE_VERIFY = r'''
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); publication=json.loads(sys.argv[2]); started=time.monotonic()
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
if not root.is_absolute() or root.is_symlink() or any(p.is_symlink() for p in root.parents):raise ValueError('unsafe evidence root')
if publication['output_root']!=str(root) or digest(root/'files.json')!=publication['files_manifest_sha256']:raise ValueError('published inventory differs')
manifest=json.loads((root/'files.json').read_text())
for key in ('job_id','request_sha256'):
    if manifest[key]!=publication[key]:raise ValueError('inventory identity differs')
files=manifest['files']; total=0
if len(files)!=publication['file_count'] or len(files)>200000:raise ValueError('inventory count differs')
for name,item in files.items():
    path=root/name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError('unsafe evidence member')
    before=path.stat()
    if before.st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('member differs: '+name)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('member changed during verification')
    total+=after.st_size
if total!=manifest['expanded_bytes'] or total!=publication['expanded_bytes']:raise ValueError('inventory byte census differs')
if digest(root/'summary.json')!=publication['summary_sha256']:raise ValueError('published summary differs')
print(json.dumps(dict(verification='all-published-members-sha256',files=len(files),bytes=total,seconds=time.monotonic()-started,manifest=manifest),sort_keys=True))
'''


def ssh(code, *args):
    return ["ssh", "beast", shlex.join(["python3", "-I", "-c", code, *map(str, args)])]


def collect(run):
    from mdb_lanes.job_support import terminal_job

    run = Path(run).resolve(); request = read(run / "request.json")
    expected = hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if expected != request["request_sha256"]:
        raise ValueError("request changed")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    jid = read(run / "submission.json")["job_id"]
    actual, job, log = terminal_job(jid, wait=False)
    if actual != jid or job["state"] != "succeeded" or job.get("assigned_host") != "beast":
        raise ValueError("direct completed Beast job required")
    marker = "MDB_DIRECTORY_READY "
    rows = [json.loads(line[len(marker):]) for line in log.splitlines() if line.startswith(marker)]
    if len(rows) != 1:
        raise ValueError("one native directory publication required")
    publication = rows[0]
    if publication["job_id"] != jid or publication["request_sha256"] != expected:
        raise ValueError("publication identity differs")
    config = request["config"]
    root = Path(config["output_scratch"]) / jid / config["output_name"]
    if publication["output_root"] != str(root):
        raise ValueError("undeclared evidence directory")
    result = subprocess.run(ssh(REMOTE_VERIFY, root, json.dumps(publication)), check=True, capture_output=True, text=True)
    verification = json.loads(result.stdout)
    pointer = dict(schema="mdb-astra-native-directory-custody-v1", host="beast", job_id=jid, request_sha256=expected,
        output_root=str(root), publication=publication, verification=verification,
        collector_sha256=sha(__file__), local_scientific_bytes=0,
        scope="All native-published members verified at original remote location; no archive duplication, no model execution, no complete local collection")
    save(run / "job.json", job); save(run / "publication.json", publication)
    save(run / "remote-custody.json", pointer)
    with (run / "scheduler-metadata.log").open("x") as stream:
        stream.write(log)
    return dict(pointer=str(run / "remote-custody.json"), job_id=jid, files=verification["files"], remote_bytes=verification["bytes"])


READ_REMOTE = r'''
import sys
from pathlib import Path
path=Path(sys.argv[1]); limit=int(sys.argv[2])
if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError('unsafe member')
with path.open('rb') as stream:
    while limit>=0:
        block=stream.read(min(1<<20,limit+1))
        if not block:break
        sys.stdout.buffer.write(block);limit-=len(block)
'''


def retrieve(run, member):
    run = Path(run).resolve(); pointer = read(run / "remote-custody.json")
    parts = Path(member)
    if parts.is_absolute() or any(value in ("", ".", "..") for value in member.split("/")) or "\\" in member:
        raise ValueError("safe relative member required")
    item = pointer["verification"]["manifest"]["files"].get(member)
    if not item or item["bytes"] > 32 << 20:
        raise ValueError("only inventoried compact members up to32MiB may be retrieved here")
    request = read(run / "request.json")
    if pointer["request_sha256"] != request["request_sha256"] or pointer["job_id"] != read(run / "submission.json")["job_id"]:
        raise ValueError("custody/request identity differs")
    root = Path(request["config"]["output_scratch"]) / pointer["job_id"] / request["config"]["output_name"]
    if pointer["host"] != "beast" or pointer["output_root"] != str(root):
        raise ValueError("undeclared custody root")
    target = run / "compact" / member; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == item["bytes"] and sha(target) == item["sha256"]:
            return dict(path=str(target), bytes=item["bytes"], sha256=item["sha256"], already_present=True)
        raise ValueError("local destination differs")
    partial = target.with_name(target.name + ".retrieving")
    process = subprocess.Popen(ssh(READ_REMOTE, root / member, item["bytes"]), stdout=subprocess.PIPE)
    hasher = hashlib.sha256(); size = 0
    try:
        with partial.open("xb") as stream:
            while block := process.stdout.read(1 << 20):
                size += len(block)
                if size > item["bytes"]:
                    raise ValueError("member exceeded published size")
                hasher.update(block); stream.write(block)
        if process.wait() or size != item["bytes"] or hasher.hexdigest() != item["sha256"]:
            raise ValueError("retrieved member differs; partial preserved")
    finally:
        if process.poll() is None:
            process.terminate()
        process.stdout.close()
    partial.replace(target)
    return dict(path=str(target), bytes=size, sha256=item["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path); parser.add_argument("--retrieve", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps([retrieve(args.run, member) for member in args.retrieve] if args.retrieve else collect(args.run), sort_keys=True))
