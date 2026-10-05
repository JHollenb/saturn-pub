"""Continue a copied legacy export without writing its original source store."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import time

from family_helpers import load, finish
from mdb import BenchSession
from mdb.bench_verify import verify


def main():
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    source = Path(config["legacy_export"])
    if (source.parent.name != "exports" or source.name != "case-0"
            or source.parent.parent.parent != Path("/mnt/big/saturn-open-bench-families")
            or not re.fullmatch(r"job-[0-9a-f]{12}", source.parent.parent.name)
            or any(p.is_symlink() for p in (source, *source.parents))):
        raise RuntimeError("legacy source is outside the trusted export store")
    if (config["durable_root"] != "/mnt/big/mdb/qualification"
            or not re.fullmatch(r"job-[0-9a-f]{12}", os.environ["MRUN_JOB_ID"])):
        raise RuntimeError("legacy assay job root is invalid")
    selected = source / "selected.json"
    before = sha256(selected.read_bytes()).hexdigest()
    if before != config["selected_sha256"]:
        raise RuntimeError("legacy selected manifest differs from the declared source")
    root = Path("/mnt/big/mdb/qualification") / os.environ["MRUN_JOB_ID"]
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    stages = []
    def stage(name):
        row = {"stage": name, "elapsed_seconds": time.monotonic() - started}
        stages.append(row)
        with (root / "stages.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print("MDB_LEGACY_STAGE=" + json.dumps(row), flush=True)
    stage("copy-start")
    copied = root / "export"
    handoff = None
    if config.get("prepared_store"):
        prepared = Path(config["prepared_store"])
        if (prepared.parent.parent != Path(config["durable_root"]) or prepared.name != "export"
                or not re.fullmatch(r"job-[0-9a-f]{12}", prepared.parent.name)
                or any(p.is_symlink() for p in (prepared, *prepared.parents))):
            raise RuntimeError("prepared copy is outside the private assay store")
        from mdb.bench_publication import recover_locked
        writer, _ = recover_locked(prepared)
        try:
            prepared_digest = sha256((prepared / "selected.json").read_bytes()).hexdigest()
            if prepared_digest != config["prepared_selected_sha256"]:
                raise RuntimeError("prepared selected manifest differs from sealed handoff")
            if prepared_digest != before:
                raise RuntimeError("prepared handoff differs from the declared original selection")
            # Copy bytes, never hard-link mutable metadata or borrow another job's writer.
            if any(path.is_symlink() for path in prepared.rglob("*")):
                raise RuntimeError("linked prepared checkpoint member")
            shutil.copytree(prepared, copied)
            if sha256((copied / "selected.json").read_bytes()).hexdigest() != prepared_digest:
                raise RuntimeError("copied handoff selected manifest differs")
            handoff = {"source_job": prepared.parent.name, "selected_sha256": prepared_digest,
                       "ownership": "exclusive source lock during byte copy; new job owns copied store"}
        finally:
            writer.close()
    else:
        if any(path.is_symlink() for path in source.rglob("*")):
            raise RuntimeError("linked legacy checkpoint member")
        shutil.copytree(source, copied)
    stage("copy-complete")
    closure = verify(copied)
    stage("initial-closure-valid")
    from mdb.circuit_runtime import DirectoryArtifactStore
    declared_cut = DirectoryArtifactStore(copied / "artifacts").get(config["original_cut_id"])
    if declared_cut["cursor"] != config["expected_cut_cursor"] or declared_cut["intervention_lineage"]:
        raise RuntimeError("legacy reference cut coordinates or intervention lineage differ")
    spec, model, tokenizer, weights, files = load(config)
    stage("model-loaded")
    with BenchSession.resume(model, tokenizer, root=copied, model_sha256=weights,
                             legacy_numerical_environment=config.get("legacy_numerical_environment")) as session:
        identity, cut = session.identity, session.branch.cut
        if identity["runtime_sha256"] != config["expected_runtime_sha256"]:
            raise RuntimeError("legacy native execution ABI differs from its qualified identity")
        session.restore(config["original_cut_id"])
        stage("cut-restored")
        if session.cursor() != config["expected_cut_cursor"]:
            raise RuntimeError("legacy restored cursor differs from the declared suffix parent")
        cut = session.branch.cut
        actual = finish(session, 1 if spec.family == "mamba" else None)
        stage("native-suffix-complete")
        # Verify under the resident owner's lock before relinquishing this job's writer.
        continued_closure = verify(copied)
        stage("continued-closure-valid")
    stage("session-closed")
    closed_closure = verify(copied)
    stage("closed-closure-valid")
    expected = config["expected"]
    if spec.family == "mamba": expected = [tuple(row) for row in expected]
    unchanged = sha256(selected.read_bytes()).hexdigest() == before
    report = {"schema": "mdb-legacy-continuation-v1", "family": spec.family,
              "original_selected_sha256": before, "original_selected_unchanged": unchanged,
              "verification": closure, "identity": identity, "cut_id": cut,
              "original_cut_id": config["original_cut_id"], "original_cursor": declared_cut["cursor"],
              "continued_verification": continued_closure, "closed_verification": closed_closure,
              "handoff": handoff,
              "actual": actual, "expected": expected, "exact": actual == expected,
              "weight_files": files, "copied_export": str(copied),
              "stages": stages,
              "execution_environment": json.loads(Path("execution-environment.json").read_text())}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print("MDB_LEGACY_REPORT=" + json.dumps(report), flush=True)
    if not report["exact"] or not unchanged:
        raise RuntimeError("legacy continuation did not match or source manifest changed")


if __name__ == "__main__": main()
