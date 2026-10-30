"""Expose actual retained readouts to the external evidence plane without rewriting them."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    output = args.job / "derived/observations.jsonl"
    output.parent.mkdir(exist_ok=True)
    rows = []
    for path in sorted((args.job / "artifacts/rows").glob("*-fast.json")):
        history = json.loads(path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for observed in history["observations"]:
            rows.append({"schema": "mdb-cot-register-evidence-row-v1", "case": history["id"],
                "source_row": str(path.relative_to(args.job)), "source_row_sha256": digest,
                "specimen": observed["model_identity"], "source_cut": observed["source_cut"],
                "observation_ref": observed["observation_ref"], "lens_ref": observed["lens_ref"],
                "mechanics_status": "valid", "semantic_cot_qualified": False,
                "causal_dependency_tested": False,
                "metrics": {"decoded_top_score": observed["candidates"][0]["score"]},
                "observation": observed})
    output.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    print(json.dumps({"path": str(output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
