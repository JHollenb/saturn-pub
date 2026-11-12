"""Descriptive RGB carrier contrasts against two saved, explicitly bound axes.

This reads only existing PNGs and metadata. It performs no model, fit, raw latent
or decoded-array computation and adds no semantic interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = "mdb-saved-carrier-rgb-axes-v1"
RESULT = "saved-carrier-axes-result"
CONTEXTS = ("anchor", "wording")
ARMS = ("QQ", "PQ", "QP", "PP")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    require(Path(path).stat().st_size <= 16 << 20, "metadata exceeds bound")
    value = json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    json.dumps(value, allow_nan=False)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def authenticate(config):
    """Authenticate only declared metadata/PNG members of immutable collections."""
    from mdb.forecast_transaction import verify_transaction

    consumed, collections = {}, {}
    for role, row in config["sources"].items():
        root = Path(row["root"]).resolve()
        collection_path = Path(row["collection"])
        require(sha(collection_path) == row["collection_sha256"], "collection bytes differ: " + role)
        collection = read(collection_path)
        expected_schema = "mdb-flux-carrier-crossover-collection-v1" if role == "carrier" else "mdb-family-authenticated-compact-v1"
        require(collection["schema"] == expected_schema, "source collection schema differs")
        if role != "carrier":
            require(collection["kind"] == role, "source collection role differs")
        custody = collection["custody"] if role == "carrier" else collection["payload_custody"]
        require(custody["required"] is True and len({(custody[k]["sha256"], custody[k]["size_bytes"])
            for k in ("declared", "sealed", "executed")}) == 1, "source payload custody differs")
        collections[role] = collection
        consumed[role] = {}

        def member(name):
            require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe consumed member")
            path = root / name
            require(path.suffix in (".json", ".png") and path.resolve().is_relative_to(root)
                and not any(p.is_symlink() for p in (path, *path.parents)), "non-metadata/image or escaped input")
            files = collection["inventory"]["compactFiles" if role == "carrier" else "files"]
            record = files[name]
            require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "consumed source hash differs: " + role + "/" + name)
            consumed[role][name] = dict(record)
            return path

        for name in row["members"]:
            member(name)
    carrier, native, audit_source = (config["sources"][role] for role in ("carrier", "native", "audit"))
    cr, nr, ar = (Path(source["root"]).resolve() for source in (carrier, native, audit_source))
    report, summary, request, raw, states, audit = (read(path) for path in (
        cr / "report.json", nr / "summary.json", nr / "request.json", nr / "files.json", nr / "state-index.json", ar / "audit.json"))
    cc, nc, ac = (collections[role] for role in ("carrier", "native", "audit"))
    carrier_request = read(cr / "request.json")
    # The immutable carrier producer uses its older camelCase/compact convention.
    carrier_identity = hashlib.sha256(json.dumps({k: v for k, v in carrier_request.items() if k != "requestSHA256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(carrier_identity == carrier_request["requestSHA256"] == report["requestSHA256"], "carrier executed request differs")
    require(report["schema"] == "mdb-flux-carrier-crossover-v1" and report["requestSHA256"] == cc["requestSHA256"], "carrier request/report differs")
    require(identity(request) == request["request_sha256"] == nc["request_sha256"] == summary["request_sha256"]
        == audit["native_request_sha256"], "natural reference request differs")
    require(summary["job_id"] == nc["job_id"] == audit["native_job_id"] == ac["publication"]["source_job"], "natural source job differs")
    require(sha(nr / "files.json") == nc["publication"]["files_manifest_sha256"] == audit["native_manifest_sha256"]
        and raw["request_sha256"] == request["request_sha256"] and raw["job_id"] == summary["job_id"], "natural raw inventory differs")
    require(sha(ar / "audit.json") == ac["publication"]["audit_sha256"]
        and sha(nr / "summary.json") == nc["publication"]["summary_sha256"], "published audit/summary differs")
    require(audit["schema"] == "mdb-flux-binding-family-saved-audit-v1"
        and audit["source_worker_sha256"] == request["source_hashes"]["flux_binding_family_campaign.py"], "saved reference reader/source differs")
    tx = {}
    for role, root, seal, prior in (("carrier", cr, report["sealSHA256"], report["transaction"]),
                                   ("native", nr, summary["forecast_seal"], audit["transaction"])):
        for name in (collections[role]["inventory"]["compactFiles" if role == "carrier" else "files"]):
            if name.startswith("forecast-transaction/"):
                require(name in consumed[role], "transaction metadata was not authenticated")
        tx[role] = verify_transaction(root / "forecast-transaction", expected_seal_sha256=seal, expected_event_head=prior["event_head"])
        require({k: v for k, v in tx[role].items() if k != "identity_scope"}
                == {k: v for k, v in prior.items() if k != "identity_scope"}, "source transaction differs")
    images, joins = {}, {}
    for cid in CONTEXTS:
        images[cid] = {}
        for arm in ARMS:
            key = "binding-" + cid + "/" + arm + "/native"
            descriptor = report["images"][key]
            declared = Path(descriptor["path"])
            member = str(declared.relative_to(cc["inventory"]["sourceRoot"])) if declared.is_absolute() else str(declared)
            expected = {k: descriptor[k] for k in ("sha256", "bytes")}
            require(consumed["carrier"].get(member) == expected == cc["inventory"]["rawFiles"].get(member), "carrier PNG provenance differs")
            images[cid][arm] = cr / member
        joins[cid] = {}
        for arm, name in (("QQQ", "native_Q"), ("PPP", "native_RW")):
            key = cid + "/" + arm
            member = states[key]["root"] + "/decoder/native-reference/image.png"
            record = consumed["native"].get(member)
            known = audit["known_endpoint_checks"][key]
            require(record is not None and record == raw["files"][member]
                and record["sha256"] == known["png_sha256"] and states[key]["native_final"]["cut"] == known["final_cut"]
                and states[key]["native_final"]["model_identity"] == report["modelIdentity"] and known["exact"] is True,
                "natural RW/Q endpoint custody join differs")
            images[cid][name] = nr / member
            joins[cid][name] = dict(case=key, final_cut=known["final_cut"], png=dict(path=member, **record))
    return images, dict(collections={role: dict(path=row["collection"], sha256=row["collection_sha256"],
        source_origin=row.get("origin"), payload_custody=collections[role].get("custody", collections[role].get("payload_custody")))
        for role, row in config["sources"].items()}, consumed=consumed, transactions=tx, natural_endpoint_joins=joins,
        source_model_identity=report["modelIdentity"], carrier_request_sha256=report["requestSHA256"],
        native_request_sha256=request["request_sha256"], audit_job_id=ac["job_id"])


def vector_stats(value):
    import numpy as np

    return dict(elements=int(value.size), rms=float(np.sqrt(np.mean(value * value))), l2=float(np.linalg.norm(value.reshape(-1))),
        mean=float(value.mean()), minimum=float(value.min()), maximum=float(value.max()), max_abs=float(np.max(np.abs(value))),
        nonzero=int(np.count_nonzero(value)))


def projection(value, axis):
    import numpy as np

    dot = float(np.sum(value * axis, dtype=np.float64))
    a2, x2 = float(np.sum(axis * axis, dtype=np.float64)), float(np.sum(value * value, dtype=np.float64))
    coefficient = dot / a2 if a2 else None
    return dict(dot=dot, axis_squared_norm=a2, response_squared_norm=x2, projection=coefficient,
        cosine=dot / math_sqrt(x2 * a2) if x2 and a2 else None,
        signed_projection_rms=coefficient * math_sqrt(a2 / axis.size) if coefficient is not None else None,
        distance_to_axis_rms=float(np.sqrt(np.mean((value - axis) ** 2))),
        direction_status="zero-axis" if not a2 else "zero-response" if not x2 else "defined")


def math_sqrt(value):
    import math

    return math.sqrt(value)


def account(arrays):
    import numpy as np

    require(set(arrays) == {*ARMS, "native_Q", "native_RW"}, "six exact source images required")
    values = {key: np.asarray(value) for key, value in arrays.items()}
    shape = values["QQ"].shape
    require(len(shape) == 3 and shape[-1] == 3 and all(value.dtype == np.uint8 and value.shape == shape for value in values.values()), "matched uint8 RGB geometry required")
    y = {key: value.astype(np.float64) for key, value in values.items()}
    terms = dict(DZ=y["PQ"] - y["QQ"], DV=y["QP"] - y["QQ"], D=y["PP"] - y["QQ"],
        J=y["PP"] - y["PQ"] - y["QP"] + y["QQ"])
    axes = dict(hold_PP_minus_QQ=terms["D"], native_RW_minus_Q=y["native_RW"] - y["native_Q"])
    result = dict(shape=list(shape), units="PNG uint8 RGB levels 0–255; complete decoded RGB pixels, no image rescaling",
        terms={name: dict(summary=vector_stats(value), axes={axis_name: projection(value, axis) for axis_name, axis in axes.items()})
            for name, value in terms.items()}, axes={name: vector_stats(axis) for name, axis in axes.items()},
        hold_axis_vs_native_axis=projection(axes["hold_PP_minus_QQ"], axes["native_RW_minus_Q"]),
        accounting_closure=vector_stats(terms["D"] - terms["DZ"] - terms["DV"] - terms["J"]))
    return result


def run(config):
    import numpy as np
    from PIL import Image

    require(config["contexts"] == list(CONTEXTS) and config["model_forwards"] == 0 and config["device"] == "cpu", "saved-only finite scope differs")
    images, provenance = authenticate(config)
    rows = {}
    for cid, files in images.items():
        arrays = {}
        for key, path in files.items():
            with Image.open(path) as image:
                require(image.format == "PNG" and image.mode == "RGB" and image.size[0] * image.size[1] <= 512 * 512, "native PNG RGB profile differs")
                arrays[key] = np.asarray(image).copy()
        rows[cid] = account(arrays)
    require(not {"torch", "diffusers", "transformers"} & set(sys.modules), "model framework entered saved RGB analysis")
    return dict(schema=SCHEMA, contexts=rows, provenance=provenance,
        formulas=dict(DZ="Y_PQ-Y_QQ", DV="Y_QP-Y_QQ", D="Y_PP-Y_QQ", J="Y_PP-Y_PQ-Y_QP+Y_QQ",
            hold_axis="Y_PP-Y_QQ", native_axis="native_PPP-native_QQQ", projection="sum(response*axis)/sum(axis*axis)",
            cosine="sum(response*axis)/sqrt(sum(response^2)*sum(axis^2))"),
        limitations=["Descriptive accounting on existing saved PNGs, after their outcomes were observed; no new predictions or targets.",
            "Carrier crossings are supplied-state interventions; projections are signed vector coordinates, not causal or semantic shares.",
            "The two axes are separately retained; hold PP−QQ is not silently substituted for the original native RW−Q contrast.",
            "Only RGB is recomputed here. Existing raw-decoded and latent accounts remain in their original evidence.",
            "Two known dependent contexts; zero axes/responses have undefined cosine or projection as declared."],
        physical_work=dict(contexts=2, existing_pngs_decoded=12, model_forwards=0, fits=0, new_targets=0, tensor_array_files_read=0,
                           cuda_calls=0, parameter_payloads_read=0), mechanics_status="authenticated saved RGB account", terminal_status="unassessed")


def worker(config_path):
    config = read(config_path)
    request = read(Path(config_path).with_name("request.json"))
    require(identity(request) == request["request_sha256"] and request["config"] == config, "analysis request/config differs")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        require(sha(Path(config_path).parent / name) == digest, "analysis payload differs: " + name)
    job = os.environ["MRUN_JOB_ID"]
    require(job.startswith("job-") and "/" not in job, "invalid analysis job identity")
    output = Path(config["output_scratch"]) / job / RESULT
    require(not output.exists(), "analysis output already exists")
    result = run(config)
    # Rehash all consumed evidence after decoding; immutable sources stay intact.
    require(authenticate(config)[1] == result["provenance"], "analysis inputs changed during account")
    output.mkdir(parents=True)
    save(output / "account.json", result)
    save(output / "request.json", request)
    save(output / "summary.json", dict(schema=SCHEMA, job_id=job, request_sha256=request["request_sha256"],
        contexts=list(result["contexts"]), physical_work=result["physical_work"], account_sha256=sha(output / "account.json")))
    files = {path.name: dict(sha256=sha(path), bytes=path.stat().st_size) for path in output.iterdir()}
    save(output / "files.json", dict(schema="mdb-saved-rgb-output-files-v1", job_id=job, request_sha256=request["request_sha256"], files=files))
    ready = dict(output_root=str(output), job_id=job, request_sha256=request["request_sha256"],
        files_manifest_sha256=sha(output / "files.json"), file_count=len(files) + 1, model_forwards=0)
    print("MDB_SAVED_CARRIER_AXES_READY " + json.dumps(ready, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)
