"""Bind original safetensors archive bytes, or retrieve verified named tensor bytes.

No model framework is required. Output .raw files contain the original archive
payload, with dtype/shape/header retained in the manifest. Checkpoints stay external.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root):
    root = Path(root).resolve()
    archives, tensors = {}, {}
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            raw_header = f.read(size)
            header = json.loads(raw_header)
            archives[path.name] = {"path": str(path), "sha256": sha_file(path),
                "bytes": path.stat().st_size, "header_length": size,
                "header_hex": raw_header.hex()}
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                start, end = entry["data_offsets"]
                f.seek(8 + size + start)
                h = hashlib.sha256()
                remaining = end - start
                while remaining:
                    chunk = f.read(min(remaining, 8 << 20))
                    if not chunk:
                        raise ValueError("truncated tensor payload")
                    h.update(chunk)
                    remaining -= len(chunk)
                tensors[name] = {"archive": path.name, "header": entry,
                                 "sha256": h.hexdigest()}
    if not tensors:
        raise ValueError("no original safetensors archives found")
    return {"schema": "mdb-parameter-artifacts-v1", "archives": archives, "tensors": tensors}


def retrieve(spec, name, root, dest):
    item = spec["tensors"][name]
    archive = spec["archives"][item["archive"]]
    path = Path(root) / item["archive"]
    if sha_file(path) != archive["sha256"]:
        raise ValueError("original archive hash mismatch")
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = f.read(header_size)
        if header.hex() != archive["header_hex"]:
            raise ValueError("original archive header mismatch")
        start, end = item["header"]["data_offsets"]
        f.seek(8 + header_size + start)
        raw = f.read(end - start)
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("original tensor payload hash mismatch")
    Path(dest).write_bytes(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--make", metavar="MANIFEST")
    parser.add_argument("--manifest")
    parser.add_argument("--tensor")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.make:
        Path(args.make).write_text(json.dumps(manifest(args.root), sort_keys=True, indent=2))
    else:
        retrieve(json.loads(Path(args.manifest).read_text()), args.tensor, args.root, args.out)
