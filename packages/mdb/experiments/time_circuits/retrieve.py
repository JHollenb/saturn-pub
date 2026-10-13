"""Download closed native-research state and sealed source from its admitted host."""
import argparse
import json
import re
import shlex
import subprocess
import tarfile
from hashlib import sha256
from pathlib import Path

REMOTE = r'''
import hashlib, json, pathlib, sys, tarfile
request, job, experiment = sys.argv[1:]
base = pathlib.Path("/mnt/hdd8tb/mdb/time-circuits")
state = base / request[:16]
source = pathlib.Path("/mnt/hdd8tb/mrun-work") / job
names = ["worker_" + experiment + ".py", experiment + "_bridge.py", "vendor",
         "mdb_runtime-0.1.0-py3-none-any.whl", "mrun-0.1.0-py3-none-any.whl",
         "payload-seal.json", "saturn_run_config.json", "saturn_debug_request.json",
         "execution-environment.json"]
if experiment == "e6":
    names.append("e6_bootstrap.py")
    names.remove("execution-environment.json")
    # This legacy toolchain emits its environment into the closed result root.
result = {}
for kind in ("state", "source"):
    target = base / (experiment + "-" + job + "-" + kind + ".tar.gz")
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        with tarfile.open(temporary, "w:gz", dereference=False) as archive:
            if kind == "state":
                archive.add(state, arcname=state.name)
            else:
                for name in names:
                    archive.add(source / name, arcname=name)
        temporary.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    result[kind] = {"path": str(target), "sha256": digest.hexdigest(), "bytes": target.stat().st_size}
print(json.dumps(result))
'''


def file_sha(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if sum(member.size for member in members) > 2 << 30:
            raise ValueError("archive exceeds the finite assay extraction budget")
        for member in members:
            target = destination / member.name
            if not target.resolve().is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError("unsafe research archive member: " + member.name)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            raw = source.extractfile(member).read()
            if target.exists() and target.read_bytes() != raw:
                raise ValueError("existing extracted evidence differs: " + member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--host", default="beast", help="SSH host for this host-local custody store")
    args = parser.parse_args()
    root = args.job_dir.resolve()
    request = json.loads((root / "request.json").read_text())
    report = json.loads((root / "report.json").read_text())
    job = json.loads((root / "job.json").read_text())
    identifier, experiment = job["job_id"], request["experiment"].removeprefix("mdb-paper-")
    digest = request["request_sha256"]
    if not re.fullmatch(r"job-[0-9a-f]{12}", identifier) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid job/request identity")
    if job["state"] not in {"succeeded", "failed"} or experiment not in {"e6", "e20"}:
        raise ValueError("closed paper assay required")
    expected = "/mnt/hdd8tb/mdb/time-circuits/" + digest[:16]
    if report["mdb"]["root"] != expected or report["mdb"]["request_sha256"] != digest:
        raise ValueError("worker custody root does not match the sealed request")
    command = shlex.join(["python3", "-c", REMOTE, digest, identifier, experiment])
    records = json.loads(subprocess.run(["ssh", args.host, command], check=True, capture_output=True, text=True).stdout)
    for kind, record in records.items():
        remote_path = Path(record["path"])
        if record["path"] != f"/mnt/hdd8tb/mdb/time-circuits/{experiment}-{identifier}-{kind}.tar.gz":
            raise ValueError("remote archive path changed")
        target = root / remote_path.name
        if not target.exists():
            temporary = root / (target.name + ".partial")
            subprocess.run(["scp", args.host + ":" + record["path"], str(temporary)], check=True)
            if file_sha(temporary) != record["sha256"]:
                raise ValueError("downloaded archive does not match the closed host copy")
            temporary.replace(target)
        if file_sha(target) != record["sha256"]:
            raise ValueError("existing archive differs from host copy")
        extract(target, root / kind)
    (root / "archive-digests.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"job": identifier, "archives_verified": list(records), "state_root": expected}))


if __name__ == "__main__":
    main()
