"""Complete the portable continuation panel of an interrupted parent assay."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from qwen_helpers import load, signature, stock
from mdb import BenchSession
from mdb.bench_verify import verify
from mdb.circuit_runtime import DirectoryArtifactStore
from mdb.bench_qwen import tensor_digest


def main():
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    raw = Path("parent-resume.json").read_bytes()
    if sha256(raw).hexdigest() != config["parent_resume_sha256"]:
        raise RuntimeError("parent continuation specification changed")
    specification = json.loads(raw)
    specification["config"]["expected_weights_sha256"] = specification["weights_sha"]
    spec, model, tokenizer, weights, _ = load(specification["config"])
    root = Path("/mnt/big/mdb/qualification") / os.environ["MRUN_JOB_ID"]
    root.mkdir(exist_ok=False)
    rows = []
    for item in specification["resumes"]:
        source = Path(item["root"])
        destination = root / source.name
        shutil.copytree(source, destination)
        closure = verify(destination)
        artifacts = DirectoryArtifactStore(destination / "artifacts")
        manifest = artifacts.get(json.loads((destination / "selected.json").read_text())["manifest"])
        expected = [tuple(row) for row in item["expected"]]
        expected_ids = [row[0] for row in expected]
        checked = []
        if item["case"] != "memory":
            for ref in manifest["cuts"]:
                cut = artifacts.get(ref)
                count = len(cut["token_ids"]) - cut["prompt_length"]
                if cut["execution"] is not None or cut.get("intervention_lineage") or not 0 < count <= len(expected):
                    continue
                exact = (cut["token_ids"][cut["prompt_length"]:] == expected_ids[:count]
                         and cut["logits"]["sha256"] == expected[count - 1][1])
                checked.append({"cut_id": ref, "tokens": count, "exact": exact})
            if not checked or not all(row["exact"] for row in checked):
                raise RuntimeError("parent committed-cut history disagrees with reference")
        with BenchSession.resume(model, tokenizer, root=destination, model_sha256=weights) as session:
            selected = session.branch.cut
            actual = signature(session.generate(len(expected)))
            identity = session.identity
        native_exact = None
        if item["lane"] == "native" and item["case"] != "memory":
            prompt = specification["config"]["prompts"][item["case"]]
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            tokens, logits = stock(model, ids, len(expected))
            native_exact = list(zip(tokens, (tensor_digest(value) for value in logits))) == expected
        rows.append({"case": item["case"], "lane": item["lane"], "cut_id": selected,
                     "exact": actual == expected, "stock_native_exact": native_exact,
                     "verification": closure, "identity": identity,
                     "parent_cut_checks": checked, "copied_export": str(destination)})
    report = {"schema": "mdb-qwen-continuation-v1", "parent_job": config["parent_job"],
              "parent_resume_sha256": config["parent_resume_sha256"], "rows": rows,
              "mechanics_status": "valid" if all(r["exact"] and r["stock_native_exact"] is not False for r in rows) else "invalid",
              "limitation": "parent timed out; this follow-up certifies retained committed-cut bytes, native comparisons and portable continuation; original timing/aggregate report unavailable"}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print("MDB_QWEN_CONTINUATION_REPORT=" + json.dumps(report), flush=True)
    if report["mechanics_status"] != "valid": raise RuntimeError("portable continuation mismatch")


if __name__ == "__main__": main()
