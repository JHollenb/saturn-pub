"""Authenticated saved-descriptor controls; no predictor, native or weight execution.

Descriptor disagreements are retained diagnostics. Custody/schema failures refuse
execution. Original forecast numerical leaves are validated by MDB but never
materialized: this helper decodes their structural metadata only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

PUBLISHER_SHA256 = "a751f2580aebdd8284973c0a59783378bd382d04b093ff29f7cbe1b2e78d065d"
CELLS = ("c1-o0", "c0-o1")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain_metadata(value):
    """Decode the public transaction tree without loading numerical array leaves."""
    kind = value["kind"]
    if kind == "mapping":
        return {key: plain_metadata(item) for key, item in value["items"].items()}
    if kind in ("tuple", "list"):
        return [plain_metadata(item) for item in value["items"]]
    if kind == "scalar":
        return value["value"]
    require(kind == "array", "unknown public forecast leaf")
    return {"unmaterialized_forecast_array": value}


def descriptors(outputs):
    return {key: value["descriptor"] for key, value in outputs.items()}


def clock(row):
    local = row["local_index"]
    kind, index = (("project", None) if local == 0 else ("joint", local - 1) if local <= 5 else
        ("single", local - 6) if local <= 25 else ("readout", None) if local == 26 else ("commit", None))
    return [row["logical_step"], local, row.get("kind", kind), row.get("index", row.get("block_index", index))]


def compare(left, right, *, name, expected_rows):
    require(len(left) == len(right) == expected_rows, "control transition coverage differs: " + name)
    rows = []
    for position, (a, b) in enumerate(zip(left, right)):
        fields = []
        av, bv = descriptors(a["outputs"]), descriptors(b["outputs"])
        for field in sorted(set(av) | set(bv)):
            fields.append({"field": field, "left": av.get(field), "right": bv.get(field),
                "equal": field in av and field in bv and av[field] == bv[field]})
        rows.append({"position": position, "left_clock": clock(a), "right_clock": clock(b),
            "clock_equal": clock(a) == clock(b), "fields": fields})
    checks = [field for row in rows for field in row["fields"]]
    return {"name": name, "transitions_compared": len(rows), "fields_compared": len(checks),
        "field_misses": sum(not row["equal"] for row in checks),
        "clock_misses": sum(not row["clock_equal"] for row in rows), "rows": rows,
        "interpretation": "All agreements and disagreements are instrument/profile diagnostics; no scientific quality gate."}


def qualified_program(parent, cell):
    from mdb_lanes.forecast_transaction import verify_transaction

    tx = parent["data"] / "transactions" / cell
    verification = verify_transaction(tx, expected_seal_sha256=parent["summary"]["seals"][cell]["seal_sha256"],
        max_bytes=parent["request"]["config"]["transaction_max_bytes"])
    path = tx / "forecasts/program.json"
    program = plain_metadata(read(path)["predictions"])
    require(len(program["transitions"]) == 112, "original qualified program horizon differs")
    return program, {"path": str(path), "sha256": sha(path), "seal_sha256": sha(tx / "seal.json"),
        "public_transaction_verification": verification,
        "array_loading": "Public verifier validates numerical leaves; only structural metadata decoded by this control."}


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def verify(*, cpu_run, parent, output, publisher_source, matched_run=None, native_run=None, prior_native_run=None):
    publisher_source = Path(publisher_source).resolve()
    require(sha(publisher_source) == PUBLISHER_SHA256, "frozen saved-only publisher source differs")
    publisher = runpy.run_path(str(publisher_source))
    p, h, q = publisher["helpers"]()
    cpu = publisher["verify_math"](h, q, Path(cpu_run).resolve())
    source = p["producer"](h, Path(parent).resolve(), "flux-prompt-factorial-result",
        "mdb-prompt-factorial-ssh-collection-v1", "collection-job.json", expected_runtime_wheels=cpu["request"]["runtime_wheels"])
    require(cpu["source"]["parent_job_id"] == source["job"]["job_id"]
        and cpu["source"]["parent_request_sha256"] == source["request"]["request_sha256"], "CPU belongs to another original parent")
    cohort = read(cpu["data"] / "cohort-seal.json")
    require(cohort["fingerprint"] == hashlib.sha256(json.dumps({k: v for k, v in cohort.items() if k != "fingerprint"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(), "CPU cohort self-seal differs")
    output = Path(output).resolve()
    require(not output.exists(), "output already exists")
    original, original_binding = qualified_program(source, "c0-o1")
    known_id = "c0-o1.d-c0-o1"
    known_ref = cpu["summary"]["programs"][known_id]
    known = read(h["file_path"](cpu["data"], known_ref["path"]))
    require(known["branch_id"] == known_id and known["chain_length"] == 84, "known math branch differs")
    result = {"schema": "mdb-latent-direction-known-math-control-v1", "model_calls": 0,
        "mathematical_recurrence_calls": 0, "scope": "Previously observed same-input control, saved descriptors only; no new numerical forecast qualification.",
        "known_math": compare(known["transitions"], original["transitions"][28:112], name=known_id + " vs original P suffix", expected_rows=84),
        "known_input": {"left": cpu["inputs"]["branches"][known_id]["input_descriptor"],
            "right": original["transitions"][27]["outputs"]["latent"]["descriptor"],
            "equal": cpu["inputs"]["branches"][known_id]["input_descriptor"] == original["transitions"][27]["outputs"]["latent"]["descriptor"]},
        "terminal": {"left": known["terminal_latent"]["descriptor"], "right": original["terminal_latent"]["descriptor"],
            "equal": known["terminal_latent"]["descriptor"] == original["terminal_latent"]["descriptor"]},
        "original_program": original_binding,
        "bindings": {"CPU_job_id": cpu["job"]["job_id"], "parent_job_id": source["job"]["job_id"],
            "CPU_program": known_ref, "CPU_cohort_seal_sha256": cpu["summary"]["cohort_seal_sha256"],
            "CPU_inputs_sha256": cpu["summary"]["inputs_sha256"]}}
    if matched_run is not None:
        md, _, _, seal, _, _, prior = q["verify_run"](Path(matched_run), h)
        require(prior["parent_job_id"] == source["job"]["job_id"], "matched Q parent differs")
        result["matched_sham_controls"] = {}
        for cell in CELLS:
            ref = seal["programs"][cell]
            path = h["file_path"](md, ref["path"])
            require(sha(path) == ref["sha256"], "prior Q program seal differs")
            baseline = read(h["file_path"](cpu["data"], cpu["summary"]["programs"][cell + ".sham"]["path"]))
            result["matched_sham_controls"][cell] = compare(baseline["transitions"][:28], read(path)["transitions"],
                name=cell + " CPU sham first pass vs completed Q", expected_rows=28)
    require((native_run is None) == (prior_native_run is None), "provide both new and prior native runs")
    if native_run is not None:
        new = publisher["native_producer"](h, Path(native_run).resolve())
        prior = p["producer"](h, Path(prior_native_run).resolve(), "flux-native-corner-result",
            "mdb-native-corner-collection-v1", "job.json")
        require(new["request"]["config"]["math_job_id"] == cpu["job"]["job_id"], "new native CPU source differs")
        require(prior["request"]["config"]["parent_job_id"] == source["job"]["job_id"], "prior native parent differs")
        result["native_known_control"] = compare(lines(new["data"] / "native" / (known_id + ".jsonl")),
            lines(prior["data"] / "c0-o1.P1.jsonl"), name="known native suffix vs previous P1 suffix", expected_rows=84)
        refs = [new["summary"]["rows"][known_id]["image"], prior["summary"]["rows"]["c0-o1.P1"]["image"]]
        actual = [sha(h["file_path"](record["data"], ref["path"])) for record, ref in zip((new, prior), refs)]
        require(actual == [ref["sha256"] for ref in refs], "actual known PNG custody differs")
        result["native_known_control"]["PNG"] = {"new": refs[0], "prior": refs[1], "byte_exact": actual[0] == actual[1]}
        result["native_provenance"] = {**publisher["native_provenance"](new),
            "reader_custody_sources": new["reader_custody_sources"]}
    # The custody functions hash all original trees before comparisons. Nothing
    # here mutates them; preserve explicit metadata snapshots as inputs too.
    runs = [Path(cpu_run), Path(parent)] + ([Path(matched_run)] if matched_run else [])
    if native_run:
        runs += [Path(native_run), Path(prior_native_run)]
    result["input_metadata"] = {str((run / name).resolve()): sha(run / name) for run in runs for name in
        ("request.json", "collection.json", "collection-local-inventory.json")}
    output.mkdir(parents=True)
    body = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (output / "report.json").write_text(body)
    receipt = {"schema": "mdb-known-control-execution-receipt-v1", "source_sha256": sha(__file__),
        "publisher_source_sha256": sha(publisher_source), "report_sha256": sha(output / "report.json"),
        "inputs": result["input_metadata"], "model_calls": 0, "mathematical_recurrence_calls": 0,
        **result.get("native_provenance", {}),
        "numeric_disagreements_are_diagnostics": True}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-run", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--publisher-source", type=Path, default=Path(__file__).resolve().parent / "publish_flux_latent_direction_cross.py")
    parser.add_argument("--matched-run", type=Path)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--prior-native-run", type=Path)
    args = parser.parse_args()
    result = verify(cpu_run=args.cpu_run, parent=args.parent, output=args.out, publisher_source=args.publisher_source,
        matched_run=args.matched_run, native_run=args.native_run, prior_native_run=args.prior_native_run)
    print(json.dumps({"report": str(args.out / "report.json"), "fields_compared": result["known_math"]["fields_compared"],
        "field_misses": result["known_math"]["field_misses"]}, sort_keys=True))


if __name__ == "__main__":
    main()
