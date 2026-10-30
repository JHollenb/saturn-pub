"""Custody and independent model-free audit of one completed fast-path trial."""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import shutil
import sys
from pathlib import Path


class NoUpstream(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"saturn", "saturn_pub", "common", "common_harness", "mrun"}:
            raise ImportError("offline audit attempted upstream import: " + fullname)
        return None


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--audit-copy", type=Path, required=True,
                        help="new independent scratch copy; native recovery can refresh derived caches")
    args = parser.parse_args()
    sys.path.insert(0, str(args.installed.resolve()))
    sys.meta_path.insert(0, NoUpstream())
    import mdb
    from mdb.bench_pointer import BenchPointer
    from mdb.bench_verify import verify
    if not Path(mdb.__file__).resolve().is_relative_to(args.installed.resolve()):
        raise RuntimeError("audit must import the independently installed wheel")
    job = args.job.resolve()
    root = job / "artifacts"
    inventory = json.loads((root / "artifact-manifest.json").read_text())
    sizes = {}
    for name, digest in inventory.items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("copied output identity differs: " + name)
        sizes[name] = path.stat().st_size
    save(job / "local-custody.json", {"schema": "mdb-fast-trial-custody-v1",
        "origin_job": job.name, "manifest_sha256": hashlib.sha256((root / "artifact-manifest.json").read_bytes()).hexdigest(),
        "verified_files": len(sizes), "verified_bytes": sum(sizes.values()), "file_sha256": inventory})
    report = json.loads((root / "report.json").read_text())
    histories = [json.loads(path.read_text()) for path in sorted((root / "rows").glob("*.json"))]
    totals, hashes, commits, pointers = {}, {}, {}, []
    for mode in ("native", "strict", "fast"):
        rows = [row for row in histories if row["mode"] == mode]
        totals[mode] = sum(row["elapsed_s"] for row in rows)
        hashes[mode] = {key: sum(row["head_hashes"][key] for row in rows)
                        for key in ("vocabulary_tensor_hash_calls", "vocabulary_tensor_hash_bytes")}
        commits[mode] = sum(row["publication"]["commits"] for row in rows)
        if mode == "fast":
            for row in rows:
                probe = row["pointer_probe"]
                pointer = BenchPointer.open(root / "pointers" / row["id"] / "pointer.json")
                browse = pointer.browse()
                plan = pointer.native_closure()
                source = pointer.residual()
                pointers.append({"case": row["id"], "pointer_bytes": probe["pointer"]["pointer_bytes"],
                    "catalog_bytes": probe["pointer"]["catalog_bytes"], "native_closure_bytes": plan["payload_bytes"],
                    "native_closure_payloads": len(plan["payloads"]), "browse_payload_reads": browse["hydration_audit"]["read_attempts"],
                    "selected_residual_bytes": len(source.payload), "residual_audit": pointer.payloads.audit_summary()})
    cot_records = []
    for path in (root / "bench/artifacts").glob("*/*.json"):
        if json.loads(path.read_text()).get("schema") == "mdb-cot-register-observation-v1":
            cot_records.append(path.stat().st_size)
    analysis = {"schema": "mdb-fast-trial-analysis-v1", "origin_job": job.name,
        "mechanics_status": "valid", "instrument_status": report["instrument_status"],
        "trend_status": "faster under the recorded SSD geometry; storage-sensitive historical counter-trends retained",
        "terminal_status": "bounded native numerical/candidate parity and selected restore; no semantic CoT claim",
        "rows": report["rows"], "step_seconds_total": totals,
        "aggregate_strict_over_fast": totals["strict"] / totals["fast"],
        "aggregate_fast_over_native": totals["fast"] / totals["native"],
        "head_hashes": hashes, "native_journal_commits": commits, "pointers": pointers,
        "cot_records": {"count": len(cot_records), "bytes": sum(cot_records),
                        "mean_bytes": sum(cot_records) / len(cot_records)},
        "sealed_output": {"files": len(sizes), "bytes": sum(sizes.values()),
                          "scope": "three arms, all histories, derived pointer journals, views, reports and native backing; excludes output inventory and pretrained weights"},
        "unique_native_payload_bytes": report["audit"]["payload_audit"]["unique_verified_bytes"],
        "lens_file_bytes": (job / "payload/lens/tensors.safetensors").stat().st_size,
        "restore": report["restore"], "semantic_cot_qualified": False,
        "limitations": report["limitations"] + ["one timing sample per arm/case; shared compute/storage host; no latency distribution or isolated DMA profile",
            "hash byte counts describe invoked full-tensor hashing, not a measured PCIe transfer trace",
            "single-boundary real trial has no staged-tensor reuse opportunity; multiple-boundary reuse is qualified by the isolated fixture"]}
    save(job / "performance-analysis.json", analysis)
    mirror = args.audit_copy.resolve()
    if mirror.exists() or mirror.is_relative_to(root) or root.is_relative_to(mirror):
        raise ValueError("audit-copy needs a new independent destination")
    shutil.copytree(root / "bench", mirror)
    audit = verify(mirror)
    save(job / "local-offline-audit.json", {**audit, "audit_copy": str(mirror),
         "scope": "native integrity checked on a copy; recovery may refresh derived pane caches; immutable source files rechecked separately"})
    # Independent auditing may add new receipts, but original worker bytes stay immutable.
    changed = [name for name, digest in inventory.items()
               if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError("audit changed original evidence bytes: " + str(changed))
    print(json.dumps({"origin_job": job.name, "output_files": len(sizes), "output_bytes": sum(sizes.values()),
        "aggregate_strict_over_fast": analysis["aggregate_strict_over_fast"], "aggregate_fast_over_native": analysis["aggregate_fast_over_native"],
        "pointers": [{k: value for k, value in row.items() if k != "residual_audit"} for row in pointers],
        "head_hashes": hashes, "cot_records": analysis["cot_records"], "audit_counts": audit["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
