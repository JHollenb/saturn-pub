"""Recover a completed Beast evidence tree after transport failure; no model work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

MAX_FILES = 65536
MAX_EXPANDED = 512 << 20
MAX_COMPRESSED = 128 << 20


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"job-[0-9a-f]{12}", args.job_id):
        parser.error("expected an exact mrun job ID")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    remote_root = "/mnt/big/mdb/qualification/circuit-discovery-20261106/" + args.job_id
    # Stream a new archive without modifying any remote evidence. Each member
    # is checked against a manifest computed before streaming begins.
    remote = r'''
import pathlib,json,hashlib,tarfile,sys,io
root=pathlib.Path(REMOTE_ROOT)
files=sorted(p for p in root.rglob('*') if p.is_file())
if any(p.is_symlink() for p in root.rglob('*')): raise RuntimeError('symlink in evidence')
if len(files)>MAX_FILES: raise RuntimeError('file count exceeds recovery bound')
if sum(p.stat().st_size for p in files)>MAX_EXPANDED: raise RuntimeError('bytes exceed recovery bound')
rows=[]
for p in files:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
raw=json.dumps({'schema':'mdb-recovered-evidence-manifest-v1','remote_root':str(root),'files':rows},sort_keys=True).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=6) as archive:
    info=tarfile.TarInfo('recovery-manifest.json');info.size=len(raw);info.mtime=0
    archive.addfile(info,io.BytesIO(raw))
    for p in files: archive.add(p,arcname='results/'+p.relative_to(root).as_posix(),recursive=False)
'''
    remote = ("REMOTE_ROOT=" + repr(remote_root) + "\nMAX_FILES=" + str(MAX_FILES)
              + "\nMAX_EXPANDED=" + str(MAX_EXPANDED) + "\n" + remote)
    archive_path = destination / "recovered.tar.gz"
    with subprocess.Popen(["ssh", "beast", "python3", "-u", "-"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as process:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(remote.encode())
        process.stdin.close()
        total = 0
        with archive_path.open("xb") as output:
            for block in iter(lambda: process.stdout.read(1 << 20), b""):
                total += len(block)
                if total > MAX_COMPRESSED:
                    process.kill()
                    raise RuntimeError("compressed recovery exceeds bound; raw host evidence retained")
                output.write(block)
        if process.wait() != 0:
            raise RuntimeError("remote recovery failed; raw host evidence retained")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_FILES + 1 or sum(m.size for m in members) > MAX_EXPANDED + (64 << 20):
            raise RuntimeError("expanded recovery exceeds bound")
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or member.name in names):
                raise RuntimeError("unsafe or duplicate evidence member")
            names.add(member.name)
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with source, target.open("xb") as output:
                for block in iter(lambda: source.read(1 << 20), b""):
                    output.write(block)
    manifest_path = destination / "recovery-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {"recovery-manifest.json"}
    for row in manifest["files"]:
        name = "results/" + row["path"]
        path = destination / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise RuntimeError("recovered evidence differs from the remote frozen manifest")
        expected.add(name)
    if names != expected:
        raise RuntimeError("recovery member set differs from manifest")
    custody = {"schema": "mdb-evidence-recovery-v1", "job_id": args.job_id,
               "remote_root": remote_root, "files": len(manifest["files"]),
               "expanded_bytes": sum(row["bytes"] for row in manifest["files"]),
               "archive_bytes": archive_path.stat().st_size, "archive_sha256": digest(archive_path),
               "manifest_sha256": digest(manifest_path), "all_file_hashes_verified": True,
               "model_rerun": False, "original_job_status_rewritten": False}
    (destination / "recovery-custody.json").write_text(json.dumps(custody, indent=2) + "\n")
    print(json.dumps(custody))


if __name__ == "__main__":
    main()
