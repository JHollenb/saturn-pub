"""Materialize only the recorded runtime's archive payloads on ext4.

uv's installer does not support the additional linked archive-root layout.
Preserve the initial overlay inventory as historical evidence, then replace
selected links with independent copies. Never mutate or prune the source cache.
"""
from email.parser import Parser
from hashlib import sha256
import json
from pathlib import Path
import shutil


def normalize(value):
    return value.lower().replace("_", "-").replace(".", "-")


def main():
    source = Path("/mnt/big/llm-models/.uv-cache/archive-v0")
    target = Path("/mnt/hdd8tb/mdb/uv-cache/archive-v0")
    environment_path = target.parent.parent/"cache-runtime.json"
    raw = environment_path.read_bytes()
    wanted = {normalize(k): v for k, v in json.loads(raw)["packages"].items()
              if normalize(k) not in {"mdb-runtime", "mrun", "pip"}}
    found, rows = set(), []
    for entry in sorted(source.iterdir()):
        if not entry.is_dir(): continue
        for info in entry.glob("*.dist-info"):
            if not (info/"METADATA").is_file() or not (info/"WHEEL").is_file():
                continue
            metadata = Parser().parsestr((info/"METADATA").read_text())
            name, version = normalize(metadata["Name"]), metadata["Version"]
            if wanted.get(name) != version: continue
            wheel = Parser().parsestr((info/"WHEEL").read_text())
            compatible = any((tag.split("-")[0] in {"cp314", "py3", "py2.py3"}
                or tag.split("-")[1] == "abi3" and tag.split("-")[0].startswith("cp3")
                    and int(tag.split("-")[0][2:]) <= 314)
                and (tag.endswith("-any") or "linux" in tag and "x86_64" in tag)
                for tag in wheel.get_all("Tag", []))
            if not compatible: continue
            destination = target/entry.name
            if destination.is_symlink():
                temporary = target/(entry.name+"-materializing")
                shutil.copytree(entry, temporary, symlinks=True)
                destination.unlink(); temporary.rename(destination)
            elif not destination.is_dir() or sha256((destination/info.name/"METADATA").read_bytes()).digest() != sha256((info/"METADATA").read_bytes()).digest():
                raise RuntimeError("existing materialization metadata differs: "+str(destination))
            found.add(name)
            rows.append({"name": name, "version": version, "archive": entry.name,
                         "metadata_sha256": sha256((info/"METADATA").read_bytes()).hexdigest()})
            print(json.dumps(rows[-1]), flush=True)
    missing = sorted(set(wanted)-found)
    report = {"schema": "mdb-cache-materialization-v2", "environment_sha256": sha256(raw).hexdigest(),
              "rows": rows, "missing": missing, "original_cache_unchanged": True,
              "pip_source": "uv interpreter/bootstrap tooling; not a model execution dependency",
              "boundary": "selected archive roots independently copied to ext4; initial linked overlay was not installable"}
    (target.parent.parent/"cache-materialization-v2.json").write_text(json.dumps(report, indent=2)+"\n")
    if missing: raise RuntimeError("recorded runtime archives missing: "+str(missing))


if __name__ == "__main__": main()
