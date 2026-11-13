"""Verify named original archive bytes or reconstruct selected tensors; no model load."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve(manifest, checkpoint_root, output=None):
    root = Path(checkpoint_root)
    arrays = {}
    for filename, record in manifest["checkpoint_files"].items():
        if Path(filename).name != filename or sha(root / filename) != record["sha256"]:
            raise ValueError("original checkpoint file differs")
    for filename, record in manifest["archive_headers"].items():
        with (root / filename).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            raw = stream.read(length)
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("original archive header differs")
            header = json.loads(raw)
            for name, tensor in manifest["tensors"].items():
                if tensor["file"] != filename:
                    continue
                entry = tensor["archive_entry"]
                if header[name] != entry:
                    raise ValueError("named original tensor header differs")
                start, end = entry["data_offsets"]
                stream.seek(8 + length + start)
                raw = stream.read(end - start)
                if hashlib.sha256(raw).hexdigest() != tensor["sha256"]:
                    raise ValueError("named original tensor bytes differ")
                arrays[name] = raw
    if output:
        header, offset = {}, 0
        for name, raw in sorted(arrays.items()):
            entry = manifest["tensors"][name]["archive_entry"]
            header[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                            "data_offsets": [offset, offset + len(raw)]}
            offset += len(raw)
        raw_header = json.dumps(header, separators=(",", ":")).encode()
        raw_header += b" " * (-len(raw_header) % 8)
        with Path(output).open("xb") as stream:
            stream.write(struct.pack("<Q", len(raw_header)))
            stream.write(raw_header)
            for _, raw in sorted(arrays.items()):
                stream.write(raw)
    return {"status": "original-archive-bytes-verified", "tensors": len(arrays),
            "parameter_exports": bool(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root")
    parser.add_argument("--manifest", default="parameter-artifacts.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(retrieve(json.loads(Path(args.manifest).read_text()),
                              args.checkpoint_root, args.output)))
