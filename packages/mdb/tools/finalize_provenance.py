"""Record adapted runtime bytes without erasing the original extraction hashes."""
from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "PROVENANCE.json"
manifest = json.loads(path.read_text())
for item in manifest["files"]:
    target = root / item["extracted_path"]
    item["final_sha256"] = sha256(target.read_bytes()).hexdigest()
    item["adapted_after_namespace_extraction"] = item["final_sha256"] != item["extracted_sha256"]
tracked = {item["extracted_path"] for item in manifest["files"]}
manifest["mdb_owned_files"] = [{"path": target.relative_to(root).as_posix(),
    "sha256": sha256(target.read_bytes()).hexdigest()} for target in sorted((root / "src/mdb").rglob("*.py"))
    if target.relative_to(root).as_posix() not in tracked]
manifest["runtime_boundary"] = "installed mdb package; optional installed mrun; no Saturn or common checkout lookup"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
