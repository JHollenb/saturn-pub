"""Check source copies and documents; actually run portable MDB examples."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
PACKAGE = BASE / "saturn-pub/packages/mdb"
ASSET = PACKAGE / "docs/assets/qwen-error-consumer-20261106"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    copies = json.loads((ASSET / "integration-copy.json").read_text())
    for name, row in copies["files"].items():
        path = ASSET / name
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Copied source/report differs: " + name)
    documents = [PACKAGE / "docs/QWEN-ERROR-CONSUMER.md", PACKAGE / "docs/COMPUTATION-APPLICABILITY.md",
                 BASE / "obsidian/blog/2026-11-06-093800-predicting-how-an-error-moves-through-the-next-block.md"]
    links = []
    for path in documents:
        text = path.read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#")):
                continue
            relative = target.split("#", 1)[0].strip("<>")
            if not (path.parent / relative).exists():
                raise ValueError("Document link absent: " + relative)
            links.append(dict(document=str(path.relative_to(BASE)), target=target))
        for target in re.findall(r"^  - (\.\./.*)$", text, re.M):
            if not (path.parent / target).is_file():
                raise ValueError("Frontmatter source absent: " + target)
            links.append(dict(document=str(path.relative_to(BASE)), target=target, kind="frontmatter"))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    commands = [
        [sys.executable, "-I", "-B", "-m", "mdb", "equations", "find", "--catalog", str(ASSET / "compact/catalog/equations"),
         "--alias", "cmp.qwen.error-consumer.full-input"],
        [sys.executable, "-I", "-B", "-m", "mdb.computation_atlas", "--atlas", str(ASSET / "compact/atlas"), "find",
         "--law-id", "law.qwen.complete-component-primitive-graph.bf16.v1"],
    ]
    workflow = ASSET / "workflow"
    workflow.mkdir()
    calls = []
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=PACKAGE, env=environment, capture_output=True, text=True, check=True, timeout=60)
        records = json.loads(result.stdout)
        path = ROOT / f"documented-browse-{index+1}.json"
        with path.open("x") as stream:
            stream.write(result.stdout)
        shutil.copyfile(path, workflow / path.name)
        calls.append(dict(command=command, returncode=result.returncode, matches=len(records), output_sha256=sha(path)))
    command = [sys.executable, "-I", "-B", str(ASSET / "label-repair-sources/label_consumer_cross_terms.py"),
        "--pack", str(ASSET / "compact"), "--out", str(ROOT / "portable-labels-v1"),
        "--expected-manifest", "f3b67a7fe77cf1003775b6e2fd551b79ba4580c4e1f9330b439c4cdf8fe652ab",
        "--expected-audit", "0c42b3e5f5ccf31229e0d4cb05456d1e6c6e0b0e6b4bdc5cab447b285bb63d45",
        "--reader-source", str(ASSET / "label-repair-sources/audit_qwen_error_consumer.py")]
    result = subprocess.run(command, cwd=PACKAGE, env=environment, capture_output=True, text=True, check=True, timeout=60)
    repair = json.loads(result.stdout)
    if sha(ROOT / "portable-labels-v1/cross-term-labels.json") != "9f4b81e3cf4aabfde47b3db487769a195cff87ca0973632bdb343d1caf09ef70":
        raise ValueError("Portable label sidecar differs")
    report = dict(schema="mdb-error-consumer-document-verification-v1", copied_members=copies["members"], copied_bytes=copies["bytes"],
        document_targets_checked=len(links), links=links, documents={str(path.relative_to(BASE)): sha(path) for path in documents},
        actual_documented_queries=calls, portable_label_repair=dict(command=command, returncode=result.returncode,
            affected_records=repair["affected_records"], exact_sidecar_reproduced=True, arrays_read=repair["arrays_read"],
            model_calls=repair["model_calls"], numerical_reductions=repair["numerical_reductions"]),
        actual_figure_inspection=dict(method="Root viewed both rendered publication PNGs through view_image",
            findings=[], figures={path.name: sha(path) for path in (ROOT / "publication-v1").glob("*.png")}),
        original_model_calls=0, arrays_or_weights_copied=False)
    (ROOT / "docs-verification-v1.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name in ("verify_docs.py", "docs-verification-v1.json"):
        shutil.copyfile(ROOT / name, workflow / name)
    print(json.dumps(dict(copied_members=copies["members"], document_targets=len(links), query_matches=[row["matches"] for row in calls],
        portable_label_sidecar_exact=True, receipt_sha256=sha(ROOT / "docs-verification-v1.json")), sort_keys=True))


if __name__ == "__main__":
    main()
