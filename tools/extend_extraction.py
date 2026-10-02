"""Add missing runtime closure without overwriting adapted modules."""
import json
from pathlib import Path
from extract_runtime import digest, graph, rewrite

root = Path(__file__).resolve().parents[1]
source = root.parent / "saturn"
paths, edges = graph(source)
manifest = json.loads((root / "PROVENANCE.json").read_text())
pending = ["saturn.local_data_plane", "saturn.ar.source_address_policy"]
seen = set()
while pending:
    module = pending.pop()
    if module in seen:
        continue
    seen.add(module)
    pending.extend(edges[module] - seen)
for module in sorted(seen):
    original = paths[module]
    target = root / "src/mdb" / original.relative_to(source / "src/saturn")
    if target.exists():
        continue
    raw = original.read_bytes()
    transformed = rewrite(raw.decode()).encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(transformed)
    manifest["files"].append({"module": module, "original_path": original.relative_to(source).as_posix(),
        "original_sha256": digest(raw), "extracted_path": target.relative_to(root).as_posix(),
        "extracted_sha256": digest(transformed)})
(root / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
