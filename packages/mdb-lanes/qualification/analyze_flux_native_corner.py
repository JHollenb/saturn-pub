"""Saved-array native finite responses and complete continuation error accounts."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import runpy
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELLS = ("c1-o0", "c0-o1")
TERMS = ("implementation_P_minus_R", "native_input_R_minus_M", "historical_replay_M_minus_N")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_value(value):
    """Normalize immutable forecast containers without changing numerical values."""
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value


def stats(value, denominator=None):
    v = np.asarray(value, dtype=np.float64)
    require(bool(v.size) and np.isfinite(v).all(), "nonempty finite vector required")
    energy = float(np.square(v).sum())
    rms = float(np.sqrt(energy / v.size))
    d = None if denominator is None else float(np.sqrt(np.mean(np.square(np.asarray(denominator, dtype=np.float64)))))
    return {"elements": v.size, "rms": rms, "squared_l2": energy, "signed_mean": float(v.mean()),
        "mae": float(np.abs(v).mean()), "max_abs": float(np.abs(v).max()),
        "p95_abs": float(np.percentile(np.abs(v), 95)), "p99_abs": float(np.percentile(np.abs(v), 99)),
        "changed_values": int(np.count_nonzero(v)), "denominator_rms": d,
        "relative_rms": rms / d if d else None}


def response_account(p, q, r, m, input_rms):
    p, q, r, m = [np.asarray(v, dtype=np.float64) for v in (p, q, r, m)]
    require(p.shape == q.shape == r.shape == m.shape, "response vector shape differs")
    dg, df = p - q, r - m
    dg_stats, df_stats = stats(dg), stats(df)
    dot = float(np.sum(dg * df))
    length = float(np.sqrt(dg_stats["squared_l2"] * df_stats["squared_l2"]))
    return {"D_G_P_minus_Q": dg_stats, "D_F_R_minus_M": df_stats, "error_DG_minus_DF": stats(dg - df, df),
        "dot_DG_DF": dot, "cosine_DG_DF": dot / length if length else None,
        "native_field_rms": stats(m)["rms"], "input_latent_discrepancy_rms": input_rms,
        "native_finite_response_ratio": df_stats["rms"] / input_rms if input_rms else None,
        "mathematical_finite_response_ratio": dg_stats["rms"] / input_rms if input_rms else None,
        "scope": "Finite input response at this field/clock; no Jacobian, global sensitivity or isolated-kernel claim."}


def continuation_account(p, r, m, n):
    p, r, m, n = [np.asarray(v, dtype=np.float64) for v in (p, r, m, n)]
    require(p.shape == r.shape == m.shape == n.shape, "continuation vector shape differs")
    vectors = dict(zip(TERMS, (p - r, r - m, m - n)))
    pairs = {a + "__" + b: {"twice_dot": 2 * float(np.sum(vectors[a] * vectors[b]))}
        for a, b in itertools.combinations(TERMS, 2)}
    total, current = p - n, p - m
    values = {name: stats(value, n) for name, value in vectors.items()}
    expected_energy = sum(v["squared_l2"] for v in values.values()) + sum(v["twice_dot"] for v in pairs.values())
    actual_energy = float(np.square(total).sum())
    return {"terms": values, "pairwise_cross_terms": pairs, "P_minus_N_old": stats(total, n),
        "P_minus_M_current": stats(current, m), "vector_identity_max_abs": float(np.max(np.abs(total - sum(vectors.values())))),
        "squared_norm_identity": {"total_squared_l2": actual_energy, "three_terms_plus_three_cross_terms": expected_energy,
            "residual": actual_energy - expected_energy},
        "scope": "Signed accounting of saved operands; terms are not independent causal shares. RGB terms are decoded output contrasts."}


def verify_inputs(run, parent, matched):
    launcher = runpy.run_path(str(HERE / "submit_flux_native_corner.py"))
    c = launcher["C"]
    request, collection, job = [read(run / name) for name in ("request.json", "collection.json", "job.json")]
    launcher["validate_request"](request)
    c["validate_job"](job, collection["job_id"], request)
    require(all(sha(run / name) == digest for name, digest in collection["bindings"].items()), "native collection bytes changed")
    require(job["result"]["diagnostics"]["command"] == read(run / "admission-command.json")["command"], "native admitted command differs")
    require(all(sha(run / "payload" / name) == digest for name, digest in
        {**request["source_hashes"], **request["runtime_wheels"]}.items()), "sealed source/runtime changed")
    expected_log = job["result"]["artifacts"]["job_log"]
    require(sha(run / "worker.log") == expected_log["sha256"] and (run / "worker.log").stat().st_size == expected_log["bytes"], "actual native log differs")
    data = run / "extracted" / launcher["OUTPUT"]
    inventory = c["inventory"](data)
    require(inventory == read(run / "collection-local-inventory.json") == read(run / "collection-remote-before.json")
        == read(run / "collection-remote-after.json"), "native original inventories differ")
    launcher["verify_manifest"](data, inventory, job["job_id"], request)
    require(sha(data / "remote-manifest.json") == collection["manifest_sha256"]
        and sha(data / "summary.json") == collection["summary_sha256"], "native collection manifest/summary differs")
    parent_request = read(parent / "request.json")
    require(parent_request["request_sha256"] == request["config"]["parent_request_sha256"], "wrong original parent")
    parent_data = parent / "extracted/flux-prompt-factorial-result"
    require(c["inventory"](parent_data) == read(parent / "collection-local-inventory.json"), "original producer changed")
    launcher["validate_matched"](parent, matched)
    require(sha(matched / "result.json") == request["config"]["matched_result_sha256"], "different Q replay")
    return launcher, request, job, data, parent_data, inventory


def load_field(root, record):
    from mdb_lanes.tensor_archive import load_tensor_archive

    archive = record["archive"]
    require(archive is not None, "complete field archive required; witnesses cannot replace it")
    path = Path(root) / archive["path"]
    require(sha(path) == archive["sha256"], "saved field archive hash differs")
    value = (load_tensor_archive(root, archive) if archive.get("encoding") else np.load(path, allow_pickle=False))
    value = np.ascontiguousarray(value, dtype=np.float32)
    descriptor = record["descriptor"]
    require(list(value.shape) == descriptor["shape"]
        and hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest() == descriptor["fp32_le_sha256"], "complete saved field identity differs")
    require(np.isfinite(value).all(), "saved field contains nonfinite values")
    return value


def program(parent_data, cell):
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    tx = parent_data / "transactions" / cell
    seal = read(parent_data / "summary.json")["seals"][cell]["seal_sha256"]
    verify_transaction(tx, expected_seal_sha256=seal, max_bytes=64 << 20)
    return json_value(load_forecast_tree(tx, read(tx / "forecasts/program.json")["predictions"]))


def analyze(run, *, parent, matched, output):
    from PIL import Image

    from mdb_lanes.forecast_transaction import verify_transaction

    run, parent, matched, output = map(lambda p: Path(p).resolve(), (run, parent, matched, output))
    require(not output.exists(), "fresh analysis output required")
    launcher, request, job, data, parent_data, inventory = verify_inputs(run, parent, matched)
    summary = read(data / "summary.json")
    require(summary["counts"] == request["config"]["counts"], "actual declared physical census differs")
    require(sha(data / "expectations.json") == request["config"]["expectations_sha256"], "expectation bytes changed")
    require(sha(data / "response-contract.json") == request["config"]["response_contract_sha256"], "paired response contract changed")
    expectations = read(data / "expectations.json")
    require(datetime.fromisoformat(expectations["created_at"]).timestamp() < job["started_ts"]
        and datetime.fromisoformat(read(data / "response-contract.json")["created_at"]).timestamp() < job["started_ts"],
        "native-response expectations must precede actual execution")
    q_data = matched / "extracted/matched-pass-output"
    q_report = read(q_data / "report.json")
    old_summary = read(parent_data / "summary.json")
    records, continuation, replay, input_changes, transaction_verification = {}, {}, {}, {}, {}
    for cell in CELLS:
        transaction_verification[cell] = verify_transaction(data / "transactions" / cell,
            expected_seal_sha256=summary["seals"][cell]["seal_sha256"], max_bytes=64 << 20)
        p = program(parent_data, cell)
        n = [json.loads(line) for line in (parent_data / "native" / (cell + ".jsonl")).read_text().splitlines()]
        branches = {role: [json.loads(line) for line in (data / (cell + "." + role + ".jsonl")).read_text().splitlines()]
            for role in ("N1.sham", "P1")}
        m, r = branches["N1.sham"], branches["P1"]
        require(len(m) == len(r) == 84, "complete native suffix histories required")
        input_change = load_field(parent_data, p["transitions"][27]["outputs"]["latent"]).astype(np.float64) - load_field(parent_data, n[27]["outputs"]["latent"]).astype(np.float64)
        input_changes[cell] = stats(input_change)
        rows = []
        for row in q_report["comparisons"][cell]["rows"]:
            local = row["local_index"]
            require((m[local]["logical_step"], m[local]["local_index"], r[local]["logical_step"], r[local]["local_index"]) == (1, local, 1, local), "native response clocks differ")
            fields = {}
            for field, sources in row["fields"].items():
                require(sources["P"] == p["transitions"][28 + local]["outputs"][field], "Q account's P reference differs")
                values = [load_field(root, ref) for root, ref in ((parent_data, sources["P"]), (q_data, sources["Q"]),
                    (data, r[local]["outputs"][field]), (data, m[local]["outputs"][field]))]
                fields[field] = {"account": response_account(*values, input_changes[cell]["rms"]),
                    "sources": {"P": sources["P"], "Q": sources["Q"], "R": r[local]["outputs"][field], "M": m[local]["outputs"][field]}}
            rows.append({"logical_step": 1, "local_index": local, "fields": fields})
        records[cell] = rows
        latent_rows = []
        for logical in (1, 2, 3):
            current, original = (logical - 1) * 28 + 27, logical * 28 + 27
            refs = (p["transitions"][original]["outputs"]["latent"], r[current]["outputs"]["latent"],
                m[current]["outputs"]["latent"], n[original]["outputs"]["latent"])
            arrays = [load_field(root, ref) for root, ref in zip((parent_data, data, data, parent_data), refs)]
            latent_rows.append({"logical_step": logical, "field": "latent", "account": continuation_account(*arrays)})
        png_refs = (old_summary["rows"][cell + ".predicted"]["image"], summary["rows"][cell + ".P1"]["image"],
            summary["rows"][cell + ".N1.sham"]["image"], old_summary["rows"][cell + ".native"]["image"])
        pngs = []
        for root, ref in zip((parent_data, data, data, parent_data), png_refs):
            require(sha(root / ref["path"]) == ref["sha256"], "original consumer PNG changed")
            with Image.open(root / ref["path"]) as img:
                require(img.mode == "RGB" and img.size == (512, 512), "fixed-coordinate RGB geometry differs")
                pngs.append(np.array(img, dtype=np.float64))
        continuation[cell] = {"latent": latent_rows, "RGB": continuation_account(*pngs), "PNG_sources": list(png_refs)}
        comparisons = [{key: value["descriptor"] == n[28 + i]["outputs"][key]["descriptor"]
            for key, value in row["outputs"].items()} for i, row in enumerate(m)]
        require(all(calculated == recorded["native_replay_diagnostic"] for calculated, recorded in zip(comparisons, m)),
            "stored historical replay diagnostics differ from independent descriptor comparison")
        replay[cell] = {"all_field_hashes_match": all(all(row.values()) for row in comparisons),
            "fields_compared": sum(len(row) for row in comparisons),
            "mismatches": [{"position": i, "fields": [k for k, v in row.items() if not v]}
                for i, row in enumerate(comparisons) if not all(row.values())],
            "terminal_png_exact": png_refs[2]["sha256"] == png_refs[3]["sha256"]}
    image = {cell: next(row for row in records[cell] if row["local_index"] == 25)["fields"]["image"]["account"] for cell in CELLS}
    results = [{"id": "native-high-case-larger-response", "holds": image[CELLS[0]]["D_F_R_minus_M"]["rms"] > image[CELLS[1]]["D_F_R_minus_M"]["rms"]},
        *[{"id": "mathematical-native-response-positive-alignment", "context": cell, "holds": image[cell]["dot_DG_DF"] > 0} for cell in CELLS]]
    require(launcher["C"]["inventory"](data) == inventory, "original native evidence changed during reduction")
    output.mkdir(parents=True)
    report = {"schema": "mdb-native-corner-analysis-v1", "job_id": job["job_id"], "request_sha256": request["request_sha256"],
        "mechanics_status": "authenticated-saved-evidence-reduced", "terminal_status": "unassessed", "scientific_quality_gate": False,
        "scope": "Known selected sources, newly executed native branches. No fresh seed, predictor refit or model call during analysis.",
        "counts": summary["counts"], "input_changes": input_changes, "responses": records, "continuation": continuation,
        "native_replay": replay, "expectations": results, "original_expectations": expectations,
        "native_transactions": transaction_verification, "producer_source": read(data / "source.json"),
        "bindings": {"native_collection_sha256": sha(run / "collection.json"), "native_summary_sha256": sha(data / "summary.json"),
            "parent_collection_sha256": sha(parent / "collection.json"), "matched_collection_sha256": sha(matched / "collection.json"),
            "analyzer_source_sha256": sha(__file__)}, "model_calls": 0, "independent_predictor_recomputation": False}
    with (output / "analysis.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    inputs = {str(path): sha(path) for path in (run / "request.json", run / "job.json", run / "collection.json",
        data / "summary.json", data / "source.json", data / "expectations.json", data / "response-contract.json",
        parent / "collection.json", matched / "collection.json", q_data / "report.json", q_data / "Q-seal.json")}
    receipt = {"schema": "mdb-native-corner-analysis-receipt-v1", "report_sha256": sha(output / "analysis.json"),
        "analyzer_source_sha256": sha(__file__), "inputs": inputs, "model_calls": 0,
        "independent_predictor_recomputation": False, "original_native_inventory_unchanged": True,
        "scope": "Authenticated saved-field reductions; no native/model execution, learned-weight forecast or calibration."}
    with (output / "analysis-receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {"output": str(output), "report_sha256": sha(output / "analysis.json"), "expectations": results,
        "native_replay": replay, "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, parent=args.parent, matched=args.matched, output=args.out), sort_keys=True))
