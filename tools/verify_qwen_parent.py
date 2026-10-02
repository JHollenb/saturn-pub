"""Audit retained Qwen token cuts without loading a model or rerunning a suffix."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from mdb.bench_verify import verify
from mdb.circuit_runtime import DirectoryArtifactStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.directory / "resume.json").read_bytes()
    specification = json.loads(raw)
    rows = []
    for item in specification["resumes"]:
        root = args.directory / Path(item["root"]).name
        closure = verify(root)
        artifacts = DirectoryArtifactStore(root / "artifacts")
        manifest = artifacts.get(json.loads((root / "selected.json").read_text())["manifest"])
        expected = item["expected"]
        checks = []
        if item["case"] != "memory":
            for ref in manifest["cuts"]:
                cut = artifacts.get(ref)
                count = len(cut["token_ids"]) - cut["prompt_length"]
                if cut["execution"] is not None or cut.get("intervention_lineage") or not 0 < count <= len(expected):
                    continue
                exact = (cut["token_ids"][cut["prompt_length"]:] == [r[0] for r in expected[:count]]
                         and cut["logits"]["sha256"] == expected[count - 1][1])
                checks.append({"cut_id": ref, "tokens": count, "exact": exact})
            if not checks or not all(c["exact"] for c in checks):
                raise RuntimeError("retained committed token cut disagrees with reference")
        rows.append({"case": item["case"], "lane": item["lane"], "verification": closure,
                     "checked_cuts": len(checks), "complete_32_token_cuts": sum(c["tokens"] == 32 for c in checks),
                     "checks": checks})
    report = {"schema": "mdb-qwen-retained-cut-audit-v1", "parent_resume_sha256": sha256(raw).hexdigest(),
              "rows": rows, "mechanics_status": "valid",
              "scope": "export byte closure and retained committed token histories; native parity and fresh continuation are separate assays"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mechanics_status": "valid", "rows": len(rows),
                      "checked_cuts": sum(r["checked_cuts"] for r in rows)}))


if __name__ == "__main__": main()
