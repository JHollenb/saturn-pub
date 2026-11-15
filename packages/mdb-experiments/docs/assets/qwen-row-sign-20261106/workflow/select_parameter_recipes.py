"""Select exact original block18 recipes without reading any weight payload."""
import hashlib
import json
from pathlib import Path

from mdb.parameter_artifacts import validate_manifest

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "mdb-qwen-sign-shortcut-20261106/run-02/payload/qwen-parameter-artifacts.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert sha(PARENT) == "d5060dd8137bf34568d3d9b0fda805c5979133b0c4a4538e23579196cb5d3374"
    parent = validate_manifest(json.loads(PARENT.read_text()))
    entries = [e for e in parent["artifacts"] if e["source"]["tensor"].startswith("model.layers.18.")]
    assert len(entries) == 11
    payload_bytes = sum(e["source"]["data_offsets"][1] - e["source"]["data_offsets"][0] for e in entries)
    assert payload_bytes == 201861632 and len({e["source"]["file"] for e in entries}) == 1
    provenance = dict(parent_manifest_sha256=sha(PARENT), parent_recipe_count=398,
                      selected_parameter_tensors=[e["source"]["tensor"] for e in entries],
                      selected_recipe_count=11, selected_parameter_payload_bytes=payload_bytes,
                      selection="Exact model.layers.18. prefix; original entry JSON values unchanged.",
                      exported_weight_payloads=0, source_mutations=0)
    manifest = validate_manifest(dict(schema=parent["schema"], defaults=parent["defaults"], artifacts=entries, provenance=provenance))
    path = ROOT / "selection-v1/qwen-block18-parameter-artifacts.json"
    save(path, manifest)
    save(ROOT / "selection-v1/parameter-selection.json", dict(provenance, subset_manifest_sha256=sha(path),
         protocol_sha256=sha(ROOT / "selection-v1/protocol.json"), weight_payload_bytes_read=0, model_calls=0))
    print(json.dumps({"selected_recipes": 11, "selected_payload_bytes": payload_bytes, "subset_manifest_sha256": sha(path), "weight_payload_bytes_read": 0}))


if __name__ == "__main__":
    main()
