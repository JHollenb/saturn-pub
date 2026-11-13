"""Verify or reconstruct original safetensors parameters without model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def retrieve(manifest, checkpoint_root, output=None):
    root = Path(checkpoint_root)
    arrays = {}
    for filename, record in manifest["checkpoint_files"].items():
        path = root / filename
        if Path(filename).name != filename or file_sha(path) != record["sha256"]:
            raise ValueError("original checkpoint file differs: " + filename)
    for filename, header in manifest["archive_headers"].items():
        path = root / filename
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            raw_header = stream.read(length)
            if hashlib.sha256(raw_header).hexdigest() != header["sha256"]:
                raise ValueError("original archive header differs")
            actual = json.loads(raw_header)
            for name, record in manifest["tensors"].items():
                if record["file"] != filename:
                    continue
                if actual[name] != record["archive_entry"]:
                    raise ValueError("named tensor header differs: " + name)
                start, end = record["archive_entry"]["data_offsets"]
                stream.seek(8 + length + start)
                raw = stream.read(end - start)
                if hashlib.sha256(raw).hexdigest() != record["archive_bytes_sha256"]:
                    raise ValueError("original tensor bytes differ: " + name)
                if output:
                    arrays[name] = raw
    if output:
        target = Path(output)
        header, offset = {}, 0
        for name, raw in sorted(arrays.items()):
            entry = manifest["tensors"][name]["archive_entry"]
            header[name] = dict(dtype=entry["dtype"], shape=entry["shape"],
                                data_offsets=[offset, offset + len(raw)])
            offset += len(raw)
        raw_header = json.dumps(header, separators=(",", ":")).encode()
        raw_header += b" " * (-len(raw_header) % 8)
        with target.open("xb") as stream:
            stream.write(struct.pack("<Q", len(raw_header)))
            stream.write(raw_header)
            for _, raw in sorted(arrays.items()):
                stream.write(raw)
    return dict(status="original-archive-bytes-verified", tensors=len(manifest["tensors"]),
                parameter_exports=bool(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root")
    parser.add_argument("--manifest", default="parameter-artifacts.json")
    parser.add_argument("--output", help="Optional new safetensors archive; omitted verifies only")
    args = parser.parse_args()
    print(json.dumps(retrieve(json.loads(Path(args.manifest).read_text()), args.checkpoint_root, args.output)))
